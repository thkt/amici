//! Eval harness binary (ADR 0002).
//!
//! Subprocess-isolated evaluator: capture / verify baselines, run known-answer
//! fixtures, and shuffle-mutate ranking for wiring sanity. Gated behind the
//! `eval-harness` feature so default `amici` builds skip the binary.
//!
//! Argv shape (key=value, no `clap` dependency):
//!
//! ```text
//! eval_harness evaluate [kind=full|identity|reverse|single_doc|shuffled]
//! eval_harness capture-baseline output=<path>
//! eval_harness capture-reverse-baseline output=<path>
//! eval_harness capture-oracle output=<path>
//! eval_harness oracle-gap baseline=<path> oracle=<path>
//! eval_harness verify-baseline baseline=<path>
//! ```
//!
//! `evaluate` defaults to `kind=full`. `kind=shuffled` re-uses the full
//! evaluation and shuffles the ranking with a fixed RNG seed before metric
//! computation (FR-014). Required keys (`output=`, `baseline=`) cause a
//! usage-error exit when absent.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use amici::eval::annotation::{ANNOTATION_SCHEMA_VERSION, Entry, Provenance, Session};
use amici::eval::baseline::{
    AggregationKind, BASELINE_SCHEMA_VERSION, BaselineKind, BaselineSnapshot, atomic_write,
    build_metric_result, write_json,
};
use amici::eval::fixture::{
    EvalDocument, EvalQuery, load_documents, load_known_answers, load_queries,
};
use amici::eval::metrics::{
    MetricResult, bootstrap_ci, hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k,
};
use amici::eval::oracle_gap::{compute_gap, format_markdown};
use amici::eval::oracle_pipeline::{OracleError, evaluate_oracle};
use amici::eval::pipeline::{
    PipelineConfig, PipelineError, QueryResult, evaluate as run_pipeline,
    evaluate_first_search_replay,
};
use rurico::embed::Embed;
use rurico::reranker::{LazyReranker, Rerank};
use rurico::retrieval::{
    CandidateSource, DedupeAggregator, HybridSearchConfig, IdentityAggregator, MaxChunkAggregator,
    MergedHit, TopKAverageAggregator,
};
use rurico::sandbox::exit_if_seatbelt;
use rurico::storage::QueryNormalizationConfig;
use rurico::{embed, reranker};

/// Mock-friendly bundle of every external seam the four mode handlers touch.
///
/// Tests swap in `MockEmbedder` / `MockReranker`, or wrap `MockReranker`
/// in `LazyReranker` when verifying lazy semantics (see
/// `replay_first_search_skips_reranker_init`). The `timestamp` closure
/// isolates `SystemTime::now()` from the snapshot-write code path so tests
/// can fix a deterministic capture-time label. Production wiring lives in
/// [`production_context`].
struct EvalContext<E: Embed, R: Rerank> {
    /// Directory holding `documents.jsonl`, `queries.jsonl`, and
    /// `known_answers.jsonl`. Production uses `tests/fixtures/eval/` under
    /// `CARGO_MANIFEST_DIR`; tests redirect to a tempdir.
    fixture_dir: PathBuf,
    embedder: E,
    reranker: R,
    /// Returns the `epoch:N` capture-time label written into
    /// `BaselineSnapshot.timestamp`. Production reads `SystemTime::now()`;
    /// tests inject a fixed string for deterministic snapshot diffs.
    timestamp: Box<dyn Fn() -> String>,
}

/// Build the production [`EvalContext`]. The embedder loads up front; the
/// reranker is wrapped in [`LazyReranker`] so modes that never rerank
/// (`replay-first-search`, `verify-baseline kind=first_search_replay`)
/// skip the cache/download/load cost entirely (ADR-0005 / Issue #68).
fn production_context()
-> Result<EvalContext<embed::Embedder, LazyReranker<reranker::Reranker>>, String> {
    let embedder = init_embedder()?;
    let reranker = LazyReranker::new(init_reranker);
    Ok(EvalContext {
        fixture_dir: Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval"),
        embedder,
        reranker,
        timestamp: Box::new(capture_timestamp_label),
    })
}

const PIPELINE_K: usize = 10;
const SHUFFLE_SEED: u64 = 42;
const BOOTSTRAP_RESAMPLES: usize = 1000;
const BOOTSTRAP_SEED: u64 = 42;

/// Closed set of fixture kinds accepted by `evaluate kind=...`. Anchors the
/// argv validation in `run_evaluate` and the dispatch in `load_fixture_for_kind`.
///
/// Kinds ending in `_oracle` route through [`evaluate_oracle`] (Issue #52)
/// against the underlying fixture (`full_oracle` → `documents.jsonl` +
/// `queries.jsonl`; `identity_oracle` → `known_answers.jsonl::identity`;
/// `single_doc_oracle` → `known_answers.jsonl::single_doc`). The `reverse`
/// and `shuffled` paths have no oracle counterpart — both are designed to
/// degrade ranking quality, so a forced top-rank inject would defeat the
/// fixture's purpose.
const VALID_EVALUATE_KINDS: &[&str] = &[
    "full",
    "shuffled",
    "identity",
    "reverse",
    "single_doc",
    "full_oracle",
    "identity_oracle",
    "single_doc_oracle",
];

/// Suffix on [`VALID_EVALUATE_KINDS`] entries that route through the Oracle
/// retrieval pipeline. Kept as a single constant so the `_oracle` literal
/// only appears in one place across argv validation and dispatch.
const ORACLE_KIND_SUFFIX: &str = "_oracle";

/// Closed set of Stage 3 aggregation kinds accepted by `aggregation=...`.
/// Anchors argv validation; the wire format on [`BaselineSnapshot`] is owned
/// by [`AggregationKind`] in `amici::eval::baseline`.
const VALID_AGGREGATION_KINDS: &[&str] = &["identity", "max-chunk", "dedupe", "topk-average"];

/// Default `k` for the `topk-average` strategy when no `topk_k=` override is
/// supplied. `k = 3` matches the issue body's "top-k average over the top
/// chunks per document" framing without needing a flag for first capture.
const DEFAULT_TOPK_AVERAGE_K: usize = 3;

// The 1 / 2 / 3 codes below intentionally diverge from the `codes::*`
// baseline ([`amici::cli::exit_code::codes`], sysexits 64 / 70 / 75 per
// ADR-0066). That baseline targets downstream Group 2 *production* CLIs
// (sae / yomu / recall); `eval_harness` is internal evaluation tooling
// invoked through `just eval-*` recipes, not a downstream consumer.
//
// `EXIT_REGRESSION = 1` is documented as the public gate signal for
// `oracle-gap` in ADR-0002 § Decision Outcome, and `tests/eval_annotation.rs`
// adopts the same scheme for the `annotate` subcommand. Promoting these to
// `codes::*` would force an ADR-0002 revision plus a breaking-change
// announcement to every `justfile` recipe user that observes the exit
// status, so the divergence is preserved deliberately.

/// Exit code for a metric regression detected by `verify-baseline`. Reserved
/// for *expected* failure modes — the gate fired because numbers moved.
const EXIT_REGRESSION: u8 = 1;
/// Exit code for argv / validation failure (missing required key, malformed
/// path). Distinguishes operator typos from substantive failures.
const EXIT_USAGE: u8 = 2;
/// Exit code for an infrastructure failure (model load, pipeline crash,
/// fixture I/O, JSON parse). Lets CI scripts distinguish "model regressed"
/// from "MLX cache missing" without parsing stderr.
const EXIT_INFRA: u8 = 3;
const MLX_RS_VERSION: &str = "0.25";
/// Pinned ruri-v3-310m revision label. The actual HF commit hash lives in
/// `rurico::embed::ModelId::revision` (private); this label is what the
/// baseline.json carries for provenance.
const RURI_V3_310M_REVISION: &str = "pinned-via-rurico-embed-cache";

/// IR metric function signature shared by [`build_global_metrics`] and
/// [`build_one_metric`]; aliased to silence `clippy::type_complexity`.
///
/// Accepts borrowed `&[&str]` of ranked doc ids — callers project from
/// `MergedHit.doc_id: String` without cloning each id per metric per query.
type MetricFn = fn(&[&str], &HashMap<String, u8>, usize) -> f64;

/// Parse hybrid scoring overrides from argv kvs.
///
/// Recognised keys (all optional):
/// - `rrf_k=<f64>` (default `60.0`)
/// - `fts_weight=<f64>` (default `1.0`)
/// - `vector_weight=<f64>` (default `1.0`)
///
/// Missing keys keep [`HybridSearchConfig::default`] semantics so capture
/// runs without overrides reproduce older baselines bit-equal.
fn parse_merge_config_from_kvs(
    kvs: &HashMap<String, String>,
) -> Result<HybridSearchConfig, String> {
    let mut config = HybridSearchConfig::default();
    if let Some(v) = kvs.get("rrf_k") {
        config.rrf_k = v.parse().map_err(|e| format!("rrf_k= parse error: {e}"))?;
    }
    if let Some(v) = kvs.get("fts_weight") {
        let w: f64 = v
            .parse()
            .map_err(|e| format!("fts_weight= parse error: {e}"))?;
        config.source_weights.insert(CandidateSource::Fts, w);
    }
    if let Some(v) = kvs.get("vector_weight") {
        let w: f64 = v
            .parse()
            .map_err(|e| format!("vector_weight= parse error: {e}"))?;
        config.source_weights.insert(CandidateSource::Vector, w);
    }
    Ok(config)
}

/// Runtime spec for Stage 3 aggregation. Carries the `k` parameter for
/// `TopKAverage`, which the storage-side [`AggregationKind`] preserves on
/// the wire as `"topk-average:k"`.
///
/// Binary-local; the JSON wire form is owned by [`AggregationKind`] in
/// `amici::eval::baseline`. Convert at the boundary via [`From`] /
/// [`TryFrom`] to keep argv parsing and storage concerns separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregationSpec {
    Identity,
    MaxChunk,
    Dedupe,
    TopKAverage(usize),
}

impl AggregationSpec {
    /// Parse `aggregation=<kind>` (and optional `topk_k=<n>`) from argv `kvs`.
    /// Returns `Self::Identity` when the key is absent so callers that don't
    /// pass the flag preserve the pre-Phase-3 behaviour.
    fn from_kvs(kvs: &HashMap<String, String>) -> Result<Self, String> {
        let raw = kvs
            .get("aggregation")
            .map(String::as_str)
            .unwrap_or("identity");
        match raw {
            "identity" => Ok(Self::Identity),
            "max-chunk" => Ok(Self::MaxChunk),
            "dedupe" => Ok(Self::Dedupe),
            "topk-average" => {
                let k = match kvs.get("topk_k") {
                    Some(v) => v
                        .parse::<usize>()
                        .map_err(|e| format!("topk_k= parse error: {e}"))?,
                    None => DEFAULT_TOPK_AVERAGE_K,
                };
                Ok(Self::TopKAverage(k))
            }
            other => Err(format!(
                "unknown aggregation: {other:?}; expected one of {VALID_AGGREGATION_KINDS:?}"
            )),
        }
    }
}

impl From<AggregationSpec> for AggregationKind {
    fn from(spec: AggregationSpec) -> Self {
        match spec {
            AggregationSpec::Identity => Self::Identity,
            AggregationSpec::MaxChunk => Self::MaxChunk,
            AggregationSpec::Dedupe => Self::Dedupe,
            AggregationSpec::TopKAverage(k) => Self::TopKAverage(k),
        }
    }
}

impl TryFrom<AggregationKind> for AggregationSpec {
    type Error = String;

    /// Lower a storage-side [`AggregationKind`] back to a runtime spec for
    /// `verify-baseline` dispatch. [`AggregationKind::NotApplicable`] is the
    /// FirstSearchReplay marker and has no Stage 3 dispatch counterpart, so
    /// it is rejected explicitly rather than silently collapsing to a
    /// runtime default.
    fn try_from(kind: AggregationKind) -> Result<Self, Self::Error> {
        match kind {
            AggregationKind::Identity => Ok(Self::Identity),
            AggregationKind::MaxChunk => Ok(Self::MaxChunk),
            AggregationKind::Dedupe => Ok(Self::Dedupe),
            AggregationKind::TopKAverage(k) => Ok(Self::TopKAverage(k)),
            AggregationKind::NotApplicable => Err(
                "AggregationKind::NotApplicable cannot dispatch as a runtime AggregationSpec \
                 — it is the FirstSearchReplay-only marker; use replay-first-search instead"
                    .to_owned(),
            ),
        }
    }
}

/// Run [`run_pipeline`] with the concrete aggregator selected by `aggregation`.
///
/// Centralises the trait-object-vs-generic dispatch so the four mode handlers
/// (`evaluate`, `capture-baseline`, `capture-reverse-baseline`,
/// `verify-baseline`) share the same fan-out.
#[allow(clippy::too_many_arguments)]
fn dispatch_pipeline<E, R>(
    corpus: &[EvalDocument],
    queries: &[EvalQuery],
    embedder: &E,
    reranker: Option<&R>,
    aggregation: AggregationSpec,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
    config: &PipelineConfig,
) -> Result<Vec<QueryResult>, PipelineError>
where
    E: Embed,
    R: Rerank,
{
    match aggregation {
        AggregationSpec::Identity => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &IdentityAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationSpec::MaxChunk => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &MaxChunkAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationSpec::Dedupe => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &DedupeAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationSpec::TopKAverage(k) => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &TopKAverageAggregator::new(k),
            merge_config,
            normalization,
            config,
        ),
    }
}

/// Run [`evaluate_oracle`] with the concrete aggregator selected by
/// `aggregation`. Mirrors [`dispatch_pipeline`] for the Oracle path; the
/// two helpers stay separate because their return types differ
/// ([`PipelineError`] vs [`OracleError`]).
#[allow(clippy::too_many_arguments)]
fn dispatch_oracle_pipeline<E, R>(
    corpus: &[EvalDocument],
    queries: &[EvalQuery],
    embedder: &E,
    reranker: Option<&R>,
    aggregation: AggregationSpec,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
    config: &PipelineConfig,
) -> Result<Vec<QueryResult>, OracleError>
where
    E: Embed,
    R: Rerank,
{
    match aggregation {
        AggregationSpec::Identity => evaluate_oracle(
            corpus,
            queries,
            embedder,
            reranker,
            &IdentityAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationSpec::MaxChunk => evaluate_oracle(
            corpus,
            queries,
            embedder,
            reranker,
            &MaxChunkAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationSpec::Dedupe => evaluate_oracle(
            corpus,
            queries,
            embedder,
            reranker,
            &DedupeAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationSpec::TopKAverage(k) => evaluate_oracle(
            corpus,
            queries,
            embedder,
            reranker,
            &TopKAverageAggregator::new(k),
            merge_config,
            normalization,
            config,
        ),
    }
}

/// Parse query-normalization overrides from argv kvs.
///
/// Recognised keys (all optional, all boolean):
/// - `normalize_nfkc=<true|false>` (runtime default `true`)
/// - `normalize_lowercase=<true|false>` (runtime default `true`)
/// - `normalize_collapse_whitespace=<true|false>` (runtime default `true`)
///
/// Missing keys keep the [`QueryNormalizationConfig::default`] runtime
/// posture (all on). Operators verifying a pre-Phase-5 baseline pass
/// `normalize_nfkc=false normalize_lowercase=false
/// normalize_collapse_whitespace=false`, or rely on
/// [`BaselineSnapshot::normalization`] which serde-defaults to all-off for
/// historical files.
fn parse_normalization_from_kvs(
    kvs: &HashMap<String, String>,
) -> Result<QueryNormalizationConfig, String> {
    let mut config = QueryNormalizationConfig::default();
    if let Some(v) = kvs.get("normalize_nfkc") {
        config.nfkc = v
            .parse()
            .map_err(|e| format!("normalize_nfkc= parse error: {e}"))?;
    }
    if let Some(v) = kvs.get("normalize_lowercase") {
        config.ascii_lowercase = v
            .parse()
            .map_err(|e| format!("normalize_lowercase= parse error: {e}"))?;
    }
    if let Some(v) = kvs.get("normalize_collapse_whitespace") {
        config.collapse_whitespace = v
            .parse()
            .map_err(|e| format!("normalize_collapse_whitespace= parse error: {e}"))?;
    }
    Ok(config)
}

/// Closed set of metrics the harness emits in `BaselineSnapshot.global` and
/// verifies via `verify-baseline`.
///
/// Anchors the contract that misspelled metric names cannot silently slip
/// past the tolerance gate: `build_global_metrics` iterates `MetricSpec::ALL`,
/// the JSON `name` field is derived from [`MetricSpec::name`], and
/// `verify-baseline` iterates the same `ALL` slice — comparing each spec's
/// committed point estimate against the current run — so a stale baseline
/// missing newer metrics is rejected as a regeneration prompt rather than
/// silently skipping the gate. Per-metric tolerance bounds (FR-017) come
/// from [`MetricSpec::tolerance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricSpec {
    HitAt1,
    HitAt3,
    RecallAt5,
    RecallAt10,
    MrrAt10,
    NdcgAt10,
}

impl MetricSpec {
    /// All specs in canonical emission order (ADR-0003 importance order).
    /// `BaselineSnapshot.global` is produced by mapping over this slice.
    const ALL: &'static [Self] = &[
        Self::HitAt1,
        Self::HitAt3,
        Self::RecallAt5,
        Self::RecallAt10,
        Self::MrrAt10,
        Self::NdcgAt10,
    ];

    /// JSON-serialised metric label as it appears in `MetricResult.name`.
    const fn name(self) -> &'static str {
        match self {
            Self::HitAt1 => "hit@1",
            Self::HitAt3 => "hit@3",
            Self::RecallAt5 => "recall@5",
            Self::RecallAt10 => "recall@10",
            Self::MrrAt10 => "mrr@10",
            Self::NdcgAt10 => "ndcg@10",
        }
    }

    const fn k(self) -> usize {
        match self {
            Self::HitAt1 => 1,
            Self::HitAt3 => 3,
            Self::RecallAt5 => 5,
            Self::RecallAt10 | Self::MrrAt10 | Self::NdcgAt10 => 10,
        }
    }

    fn metric_fn(self) -> MetricFn {
        match self {
            Self::HitAt1 | Self::HitAt3 => hit_at_k,
            Self::RecallAt5 | Self::RecallAt10 => recall_at_k,
            Self::MrrAt10 => mrr_at_k,
            Self::NdcgAt10 => ndcg_at_k,
        }
    }

    /// Per-metric drift tolerance for `verify-baseline` (FR-017).
    ///
    /// Bounds absorb cross-process MLX reranker f32 non-determinism on
    /// Apple Silicon Metal. Set to ≥ 2× empirically observed max drift
    /// (N=10 + historical session max), with floor `1e-3`. `HitAt1` /
    /// `HitAt3` floor applies because the current fixture's `mrr@10 = 1.0`
    /// ceiling pins both metrics at 1.0; recalibrate once the ceiling
    /// lifts. See ADR-0003 § Reproducibility.
    const fn tolerance(self) -> f64 {
        match self {
            Self::HitAt1 | Self::HitAt3 => 1e-3,
            Self::RecallAt5 => 1e-2,
            Self::RecallAt10 | Self::MrrAt10 | Self::NdcgAt10 => 1e-3,
        }
    }

    /// Inverse of [`Self::name`]; returns `None` for unknown labels.
    /// Test-only — `verify-baseline` iterates `MetricSpec::ALL` directly
    /// to defeat stale-baseline blind spots, so production code does not
    /// need `from_name`.
    #[cfg(test)]
    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|s| s.name() == name)
    }
}

/// Every subcommand label accepted by `main()` after argv parsing.
///
/// Source of truth for the seatbelt-sandbox classification: the compile-time
/// `const _: () = assert!(...)` below fails the build unless every entry is
/// assigned to either [`MLX_DEPENDENT_MODES`] or [`NON_MLX_MODES`], so a new
/// subcommand cannot silently bypass the gate.
const ALL_SUBCOMMANDS: &[&str] = &[
    "evaluate",
    "capture-baseline",
    "capture-reverse-baseline",
    "capture-oracle",
    "replay-first-search",
    "oracle-gap",
    "verify-baseline",
    "compare-baselines",
    "annotate",
];

/// Modes that load MLX models and would crash under the Codex seatbelt
/// sandbox. `exit_if_seatbelt` fires for these only; the complement
/// [`NON_MLX_MODES`] runs inside the sandbox.
const MLX_DEPENDENT_MODES: &[&str] = &[
    "evaluate",
    "capture-baseline",
    "capture-reverse-baseline",
    "capture-oracle",
    "replay-first-search",
    "verify-baseline",
];

/// Modes that run safely inside the seatbelt sandbox (no MLX load).
const NON_MLX_MODES: &[&str] = &["oracle-gap", "compare-baselines", "annotate"];

const _: () = assert!(MLX_DEPENDENT_MODES.len() + NON_MLX_MODES.len() == ALL_SUBCOMMANDS.len());

fn main() -> ExitCode {
    rurico::handle_probe_if_needed();

    let args: Vec<String> = env::args().skip(1).collect();
    let Some(mode) = args.first() else {
        eprintln!(
            "usage: eval_harness <{}> [key=value...]",
            ALL_SUBCOMMANDS.join("|")
        );
        return ExitCode::from(EXIT_USAGE);
    };
    if !ALL_SUBCOMMANDS.contains(&mode.as_str()) {
        eprintln!("unknown mode: {mode}");
        return ExitCode::from(EXIT_USAGE);
    }
    if MLX_DEPENDENT_MODES.contains(&mode.as_str()) {
        exit_if_seatbelt(env!("CARGO_BIN_NAME"));
    }
    let kvs: HashMap<String, String> = args[1..]
        .iter()
        .filter_map(|s| s.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
        .collect();

    match mode.as_str() {
        "evaluate" => run_evaluate(&kvs),
        "capture-baseline" => run_capture_baseline(&kvs),
        "capture-reverse-baseline" => run_capture_reverse_baseline(&kvs),
        "capture-oracle" => run_capture_oracle(&kvs),
        "replay-first-search" => run_replay_first_search(&kvs),
        "oracle-gap" => run_oracle_gap(&kvs),
        "verify-baseline" => run_verify_baseline(&kvs),
        "compare-baselines" => run_compare_baselines(&kvs),
        "annotate" => run_annotate(&kvs),
        // ALL_SUBCOMMANDS gate above rejects unknown modes; any label that
        // reaches here is in ALL_SUBCOMMANDS but missing a dispatch arm — a
        // build-time editing mistake that the gate cannot catch.
        other => unreachable!("ALL_SUBCOMMANDS entry {other:?} has no dispatch arm"),
    }
}

/// `evaluate kind=... aggregation=...` — run the reference pipeline against
/// the chosen fixture slice with the chosen aggregation strategy and print
/// metric JSON to stdout.
fn run_evaluate(kvs: &HashMap<String, String>) -> ExitCode {
    let kind = kvs.get("kind").map_or("full", String::as_str);
    if !VALID_EVALUATE_KINDS.contains(&kind) {
        eprintln!("evaluate: unknown kind {kind:?}; expected one of {VALID_EVALUATE_KINDS:?}");
        return ExitCode::from(EXIT_USAGE);
    }
    let aggregation = match AggregationSpec::from_kvs(kvs) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("evaluate: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let merge_config = match parse_merge_config_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("evaluate: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let normalization = match parse_normalization_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("evaluate: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let ctx = match production_context() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("evaluate({kind}): {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    log_run_context(&ctx, "evaluate", Some(kind), None);
    run_evaluate_with(&ctx, kind, aggregation, &merge_config, &normalization)
}

fn run_evaluate_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    kind: &str,
    aggregation: AggregationSpec,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
) -> ExitCode {
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, kind) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("evaluate({kind}): {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let pipeline_result = if kind.ends_with(ORACLE_KIND_SUFFIX) {
        dispatch_oracle_pipeline(
            &corpus,
            &queries,
            &ctx.embedder,
            Some(&ctx.reranker),
            aggregation,
            merge_config,
            normalization,
            &config,
        )
        .map_err(|e| format!("{e}"))
    } else {
        dispatch_pipeline(
            &corpus,
            &queries,
            &ctx.embedder,
            Some(&ctx.reranker),
            aggregation,
            merge_config,
            normalization,
            &config,
        )
        .map_err(|e| format!("{e}"))
    };
    let mut results = match pipeline_result {
        Ok(r) => r,
        Err(msg) => {
            eprintln!("evaluate({kind}): pipeline failed: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    match kind {
        "shuffled" => shuffle_each_ranking(&mut results),
        "reverse" => reverse_each_ranking(&mut results),
        _ => {}
    }
    let summary = serde_json::json!({
        "kind": kind,
        "recall_at_1": global_metric(&results, &queries, recall_at_k, 1),
        "mrr": global_metric(&results, &queries, mrr_at_k, PIPELINE_K),
        "ndcg_at_10": global_metric(&results, &queries, ndcg_at_k, 10),
    });
    match serde_json::to_string_pretty(&summary) {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("evaluate({kind}): serialise failed: {e}");
            ExitCode::from(EXIT_INFRA)
        }
    }
}

/// `capture-baseline output=<path> [aggregation=<kind>] [rrf_k=N] [fts_weight=W]
/// [vector_weight=W]` — run full evaluation + bootstrap CI and write
/// `BaselineSnapshot` to `output=`. Hybrid weight overrides are recorded
/// in `merge_config` so verify-baseline replays the same scoring.
fn run_capture_baseline(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path_raw) = kvs.get("output") else {
        eprintln!("capture-baseline: output= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    let output_path = match validate_output_path(output_path_raw) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let aggregation = match AggregationSpec::from_kvs(kvs) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let merge_config = match parse_merge_config_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let normalization = match parse_normalization_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let ctx = match production_context() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    log_run_context(&ctx, "capture-baseline", None, Some(&output_path));
    run_capture_baseline_with(
        &ctx,
        &output_path,
        aggregation,
        &merge_config,
        &normalization,
    )
}

fn run_capture_baseline_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    output_path: &Path,
    aggregation: AggregationSpec,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
) -> ExitCode {
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, "full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let fixture_hash = match hash_fixture_dir(&ctx.fixture_dir) {
        Ok(h) => h,
        Err(msg) => {
            eprintln!("capture-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let results = match dispatch_pipeline(
        &corpus,
        &queries,
        &ctx.embedder,
        Some(&ctx.reranker),
        aggregation,
        merge_config,
        normalization,
        &config,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("capture-baseline: pipeline failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };

    let snapshot = build_baseline_snapshot(
        BaselineKind::Forward,
        "eval_harness capture-baseline",
        aggregation.into(),
        &results,
        &queries,
        fixture_hash,
        (ctx.timestamp)(),
        merge_config,
        normalization,
    );

    if let Err(e) = write_json(&snapshot, output_path) {
        eprintln!("capture-baseline: write failed: {e}");
        return ExitCode::from(EXIT_INFRA);
    }
    eprintln!("capture-baseline: wrote {}", output_path.display());
    ExitCode::SUCCESS
}

/// `capture-reverse-baseline output=<path>` — measure the reverse-ranker
/// `nDCG@10` lower bound and persist to `output=` so T-014 can pin it.
fn run_capture_reverse_baseline(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path_raw) = kvs.get("output") else {
        eprintln!("capture-reverse-baseline: output= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    let output_path = match validate_output_path(output_path_raw) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("capture-reverse-baseline: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let ctx = match production_context() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-reverse-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    log_run_context(&ctx, "capture-reverse-baseline", None, Some(&output_path));
    run_capture_reverse_baseline_with(&ctx, &output_path)
}

fn run_capture_reverse_baseline_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    output_path: &Path,
) -> ExitCode {
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, "reverse") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("capture-reverse-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let merge_config = HybridSearchConfig::default();
    // Reverse baseline measures the nDCG lower bound under a flipped ranking.
    // Aggregation and query normalization are both irrelevant once
    // `reverse_each_ranking` runs, so pin to `Identity` + `disabled()` to
    // keep the lower-bound contract independent of #67 / #69.
    let normalization = QueryNormalizationConfig::disabled();
    let mut results = match dispatch_pipeline(
        &corpus,
        &queries,
        &ctx.embedder,
        Some(&ctx.reranker),
        AggregationSpec::Identity,
        &merge_config,
        &normalization,
        &config,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("capture-reverse-baseline: pipeline failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    reverse_each_ranking(&mut results);
    let observed_lower_bound = global_metric(&results, &queries, ndcg_at_k, 10);

    let payload = serde_json::json!({
        "schema_version": BASELINE_SCHEMA_VERSION,
        "kind": "reverse",
        "observed_lower_bound": observed_lower_bound,
        "k": 10,
        "captured_with": "eval_harness capture-reverse-baseline",
    });
    let json = match serde_json::to_string_pretty(&payload) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("capture-reverse-baseline: serialise failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    if let Err(e) = atomic_write(output_path, format!("{json}\n").as_bytes()) {
        eprintln!("capture-reverse-baseline: write failed: {e}");
        return ExitCode::from(EXIT_INFRA);
    }
    eprintln!(
        "capture-reverse-baseline: wrote {} (observed_lower_bound={observed_lower_bound:.4})",
        output_path.display()
    );
    ExitCode::SUCCESS
}

/// `capture-oracle output=<path> [aggregation=<kind>] [rrf_k=N] [fts_weight=W]
/// [vector_weight=W]` — run the Oracle retrieval pipeline (Issue #52) and
/// write a `BaselineSnapshot` with `kind=Oracle` to `output=`. Mirrors
/// `capture-baseline` argv shape so operators can swap subcommands without
/// re-learning options.
///
/// Aggregation / merge / normalization knobs flow through the same parsers
/// because the oracle baseline measures the **post-retrieval ceiling under
/// the production rerank/aggregation stack**, not a fully idealised
/// pipeline — Issue #52 design choice anchored to CoSearch's "freeze the
/// retrieval, vary the reasoning" framing.
fn run_capture_oracle(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path_raw) = kvs.get("output") else {
        eprintln!("capture-oracle: output= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    let output_path = match validate_output_path(output_path_raw) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let aggregation = match AggregationSpec::from_kvs(kvs) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let merge_config = match parse_merge_config_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let normalization = match parse_normalization_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let ctx = match production_context() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    log_run_context(&ctx, "capture-oracle", None, Some(&output_path));
    run_capture_oracle_with(
        &ctx,
        &output_path,
        aggregation,
        &merge_config,
        &normalization,
    )
}

fn run_capture_oracle_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    output_path: &Path,
    aggregation: AggregationSpec,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
) -> ExitCode {
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, "full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let fixture_hash = match hash_fixture_dir(&ctx.fixture_dir) {
        Ok(h) => h,
        Err(msg) => {
            eprintln!("capture-oracle: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let results = match dispatch_oracle_pipeline(
        &corpus,
        &queries,
        &ctx.embedder,
        Some(&ctx.reranker),
        aggregation,
        merge_config,
        normalization,
        &config,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("capture-oracle: pipeline failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };

    let snapshot = build_baseline_snapshot(
        BaselineKind::Oracle,
        "eval_harness capture-oracle",
        aggregation.into(),
        &results,
        &queries,
        fixture_hash,
        (ctx.timestamp)(),
        merge_config,
        normalization,
    );

    if let Err(e) = write_json(&snapshot, output_path) {
        eprintln!("capture-oracle: write failed: {e}");
        return ExitCode::from(EXIT_INFRA);
    }
    eprintln!("capture-oracle: wrote {}", output_path.display());
    ExitCode::SUCCESS
}

/// `replay-first-search output=<path> [rrf_k=N] [fts_weight=W] [vector_weight=W]
/// [normalization=...]` — Run Stage 1+2 + parent rollup
/// (`MaxChunkAggregator`) only, skipping Stage 4 (rerank) and Stage 5
/// (final). Writes a `BaselineSnapshot` with `kind=FirstSearchReplay`
/// (Issue #62 / ADR-0003 第 2 deliverable). Records the agentic-search
/// "first-result quality" indicator so downstream consumers can isolate
/// ranking-strategy sensitivity from aggregator / reranker effects.
fn run_replay_first_search(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path_raw) = kvs.get("output") else {
        eprintln!("replay-first-search: output= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    let output_path = match validate_output_path(output_path_raw) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("replay-first-search: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let merge_config = match parse_merge_config_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("replay-first-search: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let normalization = match parse_normalization_from_kvs(kvs) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("replay-first-search: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let ctx = match production_context() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("replay-first-search: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    log_run_context(&ctx, "replay-first-search", None, Some(&output_path));
    run_replay_first_search_with(&ctx, &output_path, &merge_config, &normalization)
}

fn run_replay_first_search_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    output_path: &Path,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
) -> ExitCode {
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, "full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("replay-first-search: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let fixture_hash = match hash_fixture_dir(&ctx.fixture_dir) {
        Ok(h) => h,
        Err(msg) => {
            eprintln!("replay-first-search: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    let results = match evaluate_first_search_replay(
        &corpus,
        &queries,
        &ctx.embedder,
        merge_config,
        normalization,
        &config,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("replay-first-search: pipeline failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };

    let snapshot = build_baseline_snapshot(
        BaselineKind::FirstSearchReplay,
        "eval_harness replay-first-search",
        AggregationKind::NotApplicable,
        &results,
        &queries,
        fixture_hash,
        (ctx.timestamp)(),
        merge_config,
        normalization,
    );

    if let Err(e) = write_json(&snapshot, output_path) {
        eprintln!("replay-first-search: write failed: {e}");
        return ExitCode::from(EXIT_INFRA);
    }
    eprintln!("replay-first-search: wrote {}", output_path.display());
    ExitCode::SUCCESS
}

/// Build a `BaselineSnapshot` envelope. `kind` / `captured_with` /
/// `aggregation` distinguish each subcommand (forward / oracle /
/// first-search-replay); the remaining provenance (schema version,
/// model id and revision, mlx_rs version) is fixed here so adding an
/// envelope field only requires editing this single function (Issue #69
/// / DRY 3+).
#[allow(clippy::too_many_arguments)]
fn build_baseline_snapshot(
    kind: BaselineKind,
    captured_with: &str,
    aggregation: AggregationKind,
    results: &[QueryResult],
    queries: &[EvalQuery],
    fixture_hash: String,
    timestamp: String,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
) -> BaselineSnapshot {
    let global = build_global_metrics(results, queries);
    let per_category = build_per_category_metrics(results, queries);
    let (latency_p50_ms, latency_p95_ms) = compute_latency_percentiles(results);
    BaselineSnapshot {
        schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
        kind,
        captured_with: captured_with.to_owned(),
        timestamp,
        model_id: embed::ModelId::default().repo_id().to_owned(),
        model_revision: RURI_V3_310M_REVISION.to_owned(),
        mlx_rs_version: MLX_RS_VERSION.to_owned(),
        fixture_hash,
        aggregation,
        merge_config: merge_config.clone(),
        normalization: *normalization,
        global,
        per_category,
        latency_p50_ms,
        latency_p95_ms,
    }
}

/// Parse a `kind=<...>` CLI argument into [`BaselineKind`].
///
/// Routes through serde so the wire-format mapping has a single source of
/// truth (`#[serde(rename_all = "snake_case")]` on `BaselineKind`); a new
/// variant gains the CLI mapping automatically without a coordinated edit.
/// Used by `verify-baseline` to compare a user-provided expected kind
/// against the committed baseline file's `kind` field (FR-024 / AC-7).
fn parse_baseline_kind(s: &str) -> Result<BaselineKind, String> {
    serde_json::from_str::<BaselineKind>(&format!("\"{s}\"")).map_err(|e| {
        format!(
            "unknown kind {s:?}: {e}; expected one of: forward, reverse, oracle, first_search_replay"
        )
    })
}

/// Validate the committed baseline envelope (schema_version + kind) before
/// any pipeline dispatch. Centralises FR-023 (schema mismatch →
/// `EXIT_REGRESSION`), FR-024 (kind mismatch via `kind=` argument →
/// `EXIT_REGRESSION`), and Reverse rejection (`EXIT_INFRA`) so the checks
/// are unit-testable independently of the MLX-loading `production_context`.
fn validate_committed_baseline_envelope(
    committed: &BaselineSnapshot,
    kvs: &HashMap<String, String>,
) -> Result<(), u8> {
    if committed.schema_version != BASELINE_SCHEMA_VERSION {
        eprintln!(
            "verify-baseline: failed — committed schema_version {:?} does not match harness {:?}; \
             regenerate via {{capture-baseline | capture-oracle | replay-first-search}} \
             before verifying",
            committed.schema_version, BASELINE_SCHEMA_VERSION
        );
        return Err(EXIT_REGRESSION);
    }
    if let Some(expected_kind_raw) = kvs.get("kind") {
        let expected_kind = match parse_baseline_kind(expected_kind_raw) {
            Ok(k) => k,
            Err(msg) => {
                eprintln!("verify-baseline: {msg}");
                return Err(EXIT_USAGE);
            }
        };
        if expected_kind != committed.kind {
            eprintln!(
                "verify-baseline: failed — committed kind {:?} does not match \
                 expected kind {:?} (kind= argument); did you pass the wrong baseline file?",
                committed.kind, expected_kind
            );
            return Err(EXIT_REGRESSION);
        }
    }
    if committed.kind == BaselineKind::Reverse {
        eprintln!(
            "verify-baseline: failed — committed kind Reverse is not supported; \
             reverse_baseline.json has a different body shape (use compare-baselines instead)"
        );
        return Err(EXIT_INFRA);
    }
    Ok(())
}

/// `verify-baseline baseline=<path>` — re-run evaluation, compare against the
/// committed baseline.json under ADR 0002 tolerance, exit 0 + stderr banner
/// `verify-baseline: passed` on success (FR-017 / AC-5.3).
fn run_verify_baseline(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(baseline_path_raw) = kvs.get("baseline") else {
        eprintln!("verify-baseline: baseline= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    let baseline_path = match validate_baseline_path(baseline_path_raw) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    // Read and parse the committed baseline before paying the multi-second
    // model-load cost — a malformed file fails fast.
    let committed: BaselineSnapshot = match read_snapshot(&baseline_path) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    if let Err(code) = validate_committed_baseline_envelope(&committed, kvs) {
        return ExitCode::from(code);
    }
    let ctx = match production_context() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    log_run_context(&ctx, "verify-baseline", None, Some(&baseline_path));
    eprintln!(
        "verify-baseline: comparing against committed snapshot (timestamp={}, model_id={}, fixture_hash={})",
        committed.timestamp, committed.model_id, committed.fixture_hash
    );
    run_verify_baseline_with(&ctx, &committed)
}

fn run_verify_baseline_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    committed: &BaselineSnapshot,
) -> ExitCode {
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, "full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let config = PipelineConfig { k: PIPELINE_K };
    // FR-021/FR-022: dispatch by committed.kind. Each kind owns its own
    // aggregation parse + pipeline call so a reader can map the variant to
    // its dispatch in one hop without re-splitting a merged arm.
    let pipeline_outcome: Result<Vec<QueryResult>, String> = match committed.kind {
        BaselineKind::Forward => {
            let aggregation = match AggregationSpec::try_from(committed.aggregation) {
                Ok(a) => a,
                Err(msg) => {
                    eprintln!("verify-baseline: {msg}");
                    return ExitCode::from(EXIT_INFRA);
                }
            };
            dispatch_pipeline(
                &corpus,
                &queries,
                &ctx.embedder,
                Some(&ctx.reranker),
                aggregation,
                &committed.merge_config,
                &committed.normalization,
                &config,
            )
            .map_err(|e| format!("{e}"))
        }
        BaselineKind::Oracle => {
            let aggregation = match AggregationSpec::try_from(committed.aggregation) {
                Ok(a) => a,
                Err(msg) => {
                    eprintln!("verify-baseline: {msg}");
                    return ExitCode::from(EXIT_INFRA);
                }
            };
            dispatch_oracle_pipeline(
                &corpus,
                &queries,
                &ctx.embedder,
                Some(&ctx.reranker),
                aggregation,
                &committed.merge_config,
                &committed.normalization,
                &config,
            )
            .map_err(|e| format!("{e}"))
        }
        BaselineKind::FirstSearchReplay => evaluate_first_search_replay(
            &corpus,
            &queries,
            &ctx.embedder,
            &committed.merge_config,
            &committed.normalization,
            &config,
        )
        .map_err(|e| format!("{e}")),
        BaselineKind::Reverse => unreachable!(
            "Reverse kind rejected by validate_committed_baseline_envelope at {}:{}",
            file!(),
            line!()
        ),
    };
    let results = match pipeline_outcome {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify-baseline: pipeline failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let current_global = build_global_metrics(&results, &queries);
    let by_name_current: HashMap<&str, &MetricResult> = current_global
        .iter()
        .map(|m| (m.name.as_str(), m))
        .collect();
    let by_name_committed: HashMap<&str, &MetricResult> = committed
        .global
        .iter()
        .map(|m| (m.name.as_str(), m))
        .collect();

    // Iterate `MetricSpec::ALL` (the current harness contract) so a stale
    // baseline that pre-dates a metric addition cannot silently skip the
    // newly added gate. Every known spec must be present in both committed
    // and current runs; missing committed entry signals a stale snapshot
    // and is rejected as a regeneration prompt.
    for spec in MetricSpec::ALL {
        let name = spec.name();
        let Some(current_m) = by_name_current.get(name) else {
            eprintln!(
                "verify-baseline: failed — current metric {name} missing from harness output"
            );
            return ExitCode::from(EXIT_INFRA);
        };
        let Some(committed_m) = by_name_committed.get(name) else {
            eprintln!(
                "verify-baseline: failed — committed baseline lacks {name}; \
                 regenerate via capture-baseline (likely produced by an older harness version)"
            );
            return ExitCode::from(EXIT_REGRESSION);
        };
        let diff = (committed_m.point_estimate - current_m.point_estimate).abs();
        let tol = spec.tolerance();
        if diff > tol {
            eprintln!(
                "verify-baseline: failed — {name} drifted by {diff:.6} > {tol:.6} \
                 (committed {:.6} vs current {:.6})",
                committed_m.point_estimate, current_m.point_estimate
            );
            return ExitCode::from(EXIT_REGRESSION);
        }
    }
    eprintln!("verify-baseline: passed");
    ExitCode::SUCCESS
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Stderr startup banner — surfaces fixture path, model id, seeds, and the
/// destination/source path before the pipeline takes over. `kind` is the
/// fixture mode for `evaluate`; `path` is the output/baseline file for the
/// capture and verify modes.
fn log_run_context<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    mode: &str,
    kind: Option<&str>,
    path: Option<&Path>,
) {
    let kind_part = kind.map_or_else(String::new, |k| format!(" kind={k}"));
    let path_part = path.map_or_else(String::new, |p| format!(" path={}", p.display()));
    eprintln!(
        "{mode}: fixture={}{kind_part}{path_part} model={} seed_shuffle={SHUFFLE_SEED} seed_bootstrap={BOOTSTRAP_SEED}",
        ctx.fixture_dir.display(),
        embed::ModelId::default().repo_id(),
    );
}

/// Resolve `output=<path>` argument to a canonicalised absolute path.
///
/// Verifies the parent directory exists and resolves `..` so the harness
/// never silently writes to an unexpected location. The destination file
/// itself need not exist yet (capture modes create it).
fn validate_output_path(raw: &str) -> Result<PathBuf, String> {
    let path = Path::new(raw);
    let file_name = path
        .file_name()
        .ok_or_else(|| format!("output= must end in a file name: {raw}"))?;
    let parent_raw = path.parent().unwrap_or_else(|| Path::new("."));
    let parent_for_canon = if parent_raw.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent_raw
    };
    let canonical_parent = parent_for_canon.canonicalize().map_err(|e| {
        format!(
            "output= parent does not exist: {} ({e})",
            parent_for_canon.display()
        )
    })?;
    Ok(canonical_parent.join(file_name))
}

/// Resolve `baseline=<path>` argument to a canonicalised absolute path that
/// must already exist and be readable.
fn validate_baseline_path(raw: &str) -> Result<PathBuf, String> {
    Path::new(raw)
        .canonicalize()
        .map_err(|e| format!("baseline= file not found or unreadable: {raw} ({e})"))
}

/// Load corpus + queries for `kind` from `fixture_dir`. `full` / `shuffled`
/// use `documents.jsonl` + `queries.jsonl`; the known-answer kinds pull a
/// sub-fixture from `known_answers.jsonl`. `_oracle`-suffixed kinds map to
/// the underlying fixture (e.g. `identity_oracle` → identity sub-fixture);
/// the suffix only flips the dispatch in [`run_evaluate_with`].
fn load_fixture_for_kind(
    fixture_dir: &Path,
    kind: &str,
) -> Result<(Vec<EvalDocument>, Vec<EvalQuery>), String> {
    let underlying = kind.strip_suffix(ORACLE_KIND_SUFFIX).unwrap_or(kind);
    match underlying {
        "full" | "shuffled" => {
            let docs = load_documents(&fixture_dir.join("documents.jsonl"))
                .map_err(|e| format!("load_documents: {e}"))?;
            let queries = load_queries(&fixture_dir.join("queries.jsonl"))
                .map_err(|e| format!("load_queries: {e}"))?;
            Ok((docs, queries))
        }
        "identity" => {
            let known = load_known_answers(&fixture_dir.join("known_answers.jsonl"))
                .map_err(|e| format!("load_known_answers: {e}"))?;
            Ok((known.identity.corpus, known.identity.queries))
        }
        "reverse" => {
            let known = load_known_answers(&fixture_dir.join("known_answers.jsonl"))
                .map_err(|e| format!("load_known_answers: {e}"))?;
            Ok((known.reverse.corpus, known.reverse.queries))
        }
        "single_doc" => {
            let known = load_known_answers(&fixture_dir.join("known_answers.jsonl"))
                .map_err(|e| format!("load_known_answers: {e}"))?;
            Ok((known.single_doc.corpus, known.single_doc.queries))
        }
        other => Err(format!("unknown kind: {other}")),
    }
}

fn init_embedder() -> Result<embed::Embedder, String> {
    let model_id = embed::ModelId::default();
    let artifacts =
        match embed::cached_artifacts(model_id).map_err(|e| format!("embed cache lookup: {e}"))? {
            Some(a) => a,
            None => {
                eprintln!(
                    "embed model not cached, downloading {}...",
                    model_id.repo_id()
                );
                embed::download_model(model_id).map_err(|e| format!("embed download: {e}"))?
            }
        };
    embed::Embedder::new(&artifacts).map_err(|e| format!("embedder load: {e}"))
}

fn init_reranker() -> Result<reranker::Reranker, String> {
    let model_id = reranker::RerankerModelId::default();
    let artifacts = match reranker::cached_artifacts(model_id)
        .map_err(|e| format!("reranker cache lookup: {e}"))?
    {
        Some(a) => a,
        None => {
            eprintln!(
                "reranker model not cached, downloading {}...",
                model_id.repo_id()
            );
            reranker::download_model(model_id).map_err(|e| format!("reranker download: {e}"))?
        }
    };
    reranker::Reranker::new(&artifacts).map_err(|e| format!("reranker load: {e}"))
}

/// Shuffle every per-query ranking with a fixed seed so T-016 is deterministic.
fn shuffle_each_ranking(results: &mut [QueryResult]) {
    let mut rng = ChaCha8Rng::seed_from_u64(SHUFFLE_SEED);
    for r in results.iter_mut() {
        r.ranked_hits.shuffle(&mut rng);
    }
}

/// Reverse every per-query ranking. Mirrors the operation
/// `capture-reverse-baseline` performs to derive `observed_lower_bound`
/// (FR-012); shared so `evaluate kind=reverse` and the baseline capture stay
/// in lockstep.
fn reverse_each_ranking(results: &mut [QueryResult]) {
    for r in results.iter_mut() {
        r.ranked_hits.reverse();
    }
}

/// Project `ranked_hits` to a parent-doc list, retaining only the first
/// occurrence of each `doc_id`.
///
/// IR metrics (`recall@k`, `mrr@k`, `ndcg@k`) are defined over unique
/// document identities — duplicate parent ids would let a single relevant
/// doc count multiple times and push `recall@k` above `1.0`. Chunk-level
/// retrieval under the Identity aggregator surfaces sibling chunks of the
/// same parent, so the projection has to dedupe before the metric runs.
fn parent_dedup_ranked(hits: &[MergedHit]) -> Vec<&str> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut out = Vec::with_capacity(hits.len());
    for hit in hits {
        let id = hit.doc_id.as_str();
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

/// Mean of `metric_fn` across every (result, query) pair.
fn global_metric<F>(results: &[QueryResult], queries: &[EvalQuery], metric_fn: F, k: usize) -> f64
where
    F: Fn(&[&str], &HashMap<String, u8>, usize) -> f64,
{
    let scores: Vec<f64> = results
        .iter()
        .zip(queries.iter())
        .map(|(r, q)| {
            let ranked = parent_dedup_ranked(&r.ranked_hits);
            metric_fn(&ranked, &q.relevance_map, k)
        })
        .collect();
    if scores.is_empty() {
        0.0
    } else {
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// `compare-baselines paths=p1,p2,...` — read each forward baseline JSON
/// and emit a markdown comparison table to stdout.
///
/// Each row reports the captured aggregation, hybrid-config knobs
/// (`rrf_k`, FTS / Vector weights), and the four global metrics
/// (`recall@5`, `recall@10`, `mrr@10`, `ndcg@10`). Output is markdown so
/// it pastes directly into PR descriptions or comparison.md fixtures.
fn run_compare_baselines(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(paths_raw) = kvs.get("paths") else {
        eprintln!("compare-baselines: paths= argument required (comma-separated)");
        return ExitCode::from(EXIT_USAGE);
    };
    let paths: Vec<&str> = paths_raw.split(',').filter(|s| !s.is_empty()).collect();
    if paths.is_empty() {
        eprintln!("compare-baselines: paths= must contain at least one path");
        return ExitCode::from(EXIT_USAGE);
    }
    let mut snapshots: Vec<(String, BaselineSnapshot)> = Vec::with_capacity(paths.len());
    for path in &paths {
        let snapshot: BaselineSnapshot = match read_snapshot(Path::new(path)) {
            Ok(s) => s,
            Err(msg) => {
                eprintln!("compare-baselines: {msg}");
                return ExitCode::from(EXIT_INFRA);
            }
        };
        if snapshot.kind != BaselineKind::Forward {
            eprintln!(
                "compare-baselines: {path} is not a forward baseline (kind={:?})",
                snapshot.kind
            );
            return ExitCode::from(EXIT_USAGE);
        }
        snapshots.push(((*path).to_owned(), snapshot));
    }
    print_comparison_table(&snapshots);
    ExitCode::SUCCESS
}

/// `oracle-gap baseline=<path> oracle=<path>` — read a Forward baseline
/// and an Oracle baseline (Issue #52), emit a markdown gap report on
/// stdout, and exit `EXIT_REGRESSION` if AC 4 is violated (any category
/// where `oracle.recall@k < baseline.recall@k`).
fn run_oracle_gap(kvs: &HashMap<String, String>) -> ExitCode {
    match run_oracle_gap_inner(kvs) {
        Ok(()) => ExitCode::SUCCESS,
        Err((code, msg)) => {
            eprintln!("oracle-gap: {msg}");
            ExitCode::from(code)
        }
    }
}

/// Happy-path body for [`run_oracle_gap`]. Each error carries its exit
/// code so the outer wrapper never has to map back from the message.
fn run_oracle_gap_inner(kvs: &HashMap<String, String>) -> Result<(), (u8, String)> {
    let baseline_raw = kvs
        .get("baseline")
        .ok_or((EXIT_USAGE, "baseline= argument required".to_owned()))?;
    let oracle_raw = kvs
        .get("oracle")
        .ok_or((EXIT_USAGE, "oracle= argument required".to_owned()))?;
    let baseline_path =
        validate_baseline_path(baseline_raw).map_err(|m| (EXIT_USAGE, format!("baseline= {m}")))?;
    let oracle_path =
        validate_baseline_path(oracle_raw).map_err(|m| (EXIT_USAGE, format!("oracle= {m}")))?;
    let baseline_snapshot =
        read_snapshot(&baseline_path).map_err(|m| (EXIT_INFRA, format!("baseline= {m}")))?;
    let oracle_snapshot =
        read_snapshot(&oracle_path).map_err(|m| (EXIT_INFRA, format!("oracle= {m}")))?;
    enforce_current_schema(&baseline_snapshot, "baseline")?;
    enforce_current_schema(&oracle_snapshot, "oracle")?;
    let gap = compute_gap(&baseline_snapshot, &oracle_snapshot)
        .map_err(|e| (EXIT_INFRA, format!("{e}")))?;
    println!("{}", format_markdown(&gap));
    if gap.ac4_violations.is_empty() {
        Ok(())
    } else {
        Err((
            EXIT_REGRESSION,
            format!(
                "AC 4 violated by {} per-category recall regression(s)",
                gap.ac4_violations.len()
            ),
        ))
    }
}

/// Reject snapshots whose `schema_version` does not match the current
/// harness, so a stale committed file cannot drift past a schema bump.
fn enforce_current_schema(snapshot: &BaselineSnapshot, label: &str) -> Result<(), (u8, String)> {
    if snapshot.schema_version == BASELINE_SCHEMA_VERSION {
        return Ok(());
    }
    Err((
        EXIT_INFRA,
        format!(
            "{label} schema_version {:?} does not match harness {:?}; \
             regenerate the snapshot before comparing",
            snapshot.schema_version, BASELINE_SCHEMA_VERSION
        ),
    ))
}

/// Read a `BaselineSnapshot` from `path` as JSON. Centralised so
/// `oracle-gap` reads both files through the same parse/error path.
fn read_snapshot(path: &Path) -> Result<BaselineSnapshot, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Format `snapshots` as a markdown comparison table on stdout.
fn print_comparison_table(snapshots: &[(String, BaselineSnapshot)]) {
    println!(
        "| Path | aggregation | rrf_k | fts | vector | recall@5 | recall@10 | mrr@10 | ndcg@10 |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for (path, snap) in snapshots {
        let fts = snap
            .merge_config
            .source_weights
            .get(&CandidateSource::Fts)
            .copied()
            .unwrap_or(0.0);
        let vector = snap
            .merge_config
            .source_weights
            .get(&CandidateSource::Vector)
            .copied()
            .unwrap_or(0.0);
        let metric = |name: &str| {
            snap.global
                .iter()
                .find(|m| m.name == name)
                .map_or_else(|| "—".to_owned(), |m| format!("{:.4}", m.point_estimate))
        };
        println!(
            "| {} | {} | {:.1} | {:.2} | {:.2} | {} | {} | {} | {} |",
            path,
            snap.aggregation,
            snap.merge_config.rrf_k,
            fts,
            vector,
            metric("recall@5"),
            metric("recall@10"),
            metric("mrr@10"),
            metric("ndcg@10"),
        );
    }
}

/// `[recall@5, recall@10, mrr@10, ndcg@10]` with bootstrap CI applied per metric.
fn build_global_metrics(results: &[QueryResult], queries: &[EvalQuery]) -> Vec<MetricResult> {
    MetricSpec::ALL
        .iter()
        .map(|spec| build_one_metric(results, queries, *spec))
        .collect()
}

fn build_one_metric(
    results: &[QueryResult],
    queries: &[EvalQuery],
    spec: MetricSpec,
) -> MetricResult {
    let metric = spec.metric_fn();
    let k = spec.k();
    let scores: Vec<f64> = results
        .iter()
        .zip(queries.iter())
        .map(|(r, q)| {
            let ranked = parent_dedup_ranked(&r.ranked_hits);
            metric(&ranked, &q.relevance_map, k)
        })
        .collect();
    let mean = |xs: &[f64]| {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<f64>() / xs.len() as f64
        }
    };
    let (point, ci_lower, ci_upper) =
        bootstrap_ci(&scores, mean, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED);
    build_metric_result(spec.name().to_owned(), k, point, ci_lower, ci_upper)
}

/// Group queries by category and compute the same metric set per group.
fn build_per_category_metrics(
    results: &[QueryResult],
    queries: &[EvalQuery],
) -> BTreeMap<String, Vec<MetricResult>> {
    let mut buckets: BTreeMap<String, (Vec<QueryResult>, Vec<EvalQuery>)> = BTreeMap::new();
    for (r, q) in results.iter().zip(queries.iter()) {
        let entry = buckets.entry(q.category.clone()).or_default();
        entry.0.push(r.clone());
        entry.1.push(q.clone());
    }
    buckets
        .into_iter()
        .map(|(cat, (rs, qs))| (cat, build_global_metrics(&rs, &qs)))
        .collect()
}

#[allow(clippy::cast_precision_loss)]
fn compute_latency_percentiles(results: &[QueryResult]) -> (f64, f64) {
    if results.is_empty() {
        return (0.0, 0.0);
    }
    let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
    latencies.sort_unstable();
    let p50_idx = latencies.len() / 2;
    let p95_idx = ((latencies.len() * 95) / 100).min(latencies.len() - 1);
    (latencies[p50_idx] as f64, latencies[p95_idx] as f64)
}

/// Opaque capture-time label in `epoch:N` form (Unix seconds since UNIX_EPOCH).
///
/// Avoids pulling `chrono` into the dependency tree just for a strict
/// ISO-8601 timestamp; producer-doc and consumer schema both reflect the
/// actual format.
fn capture_timestamp_label() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

/// FNV-1a 64-bit hash over the three fixture JSONL files. Used as the
/// `fixture_hash` field on [`BaselineSnapshot`]; sha2 is intentionally avoided
/// to keep the dependency graph small (collision risk is acceptable for a
/// fixture-changed signal). Returns a typed error rather than swallowing the
/// `fs::read` failure so a missing fixture surfaces at capture time instead
/// of silently producing a misleading hash.
fn hash_fixture_dir(fixture_dir: &Path) -> Result<String, String> {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for name in ["documents.jsonl", "queries.jsonl", "known_answers.jsonl"] {
        let path = fixture_dir.join(name);
        let content = fs::read(&path)
            .map_err(|e| format!("hash_fixture_dir: read {} failed: {e}", path.display()))?;
        for byte in &content {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok(format!("fnv1a64:{hash:016x}"))
}

/// MLX-free runtime context for the `annotate` subcommand. Carries the
/// fixture directory whose `queries.jsonl` content hash is stamped into
/// `Provenance.queries_jsonl_hash`, plus an injectable timestamp closure
/// so tests can pin a deterministic `epoch:N` label without faking
/// `SystemTime`.
///
/// Distinct from [`EvalContext`] because Phase 1 block-mode authoring
/// touches no MLX runtime — embedder/reranker fields would be dead
/// weight here.
struct AnnotationContext {
    fixture_dir: PathBuf,
    timestamp: Box<dyn Fn() -> String>,
}

/// Build the runtime [`AnnotationContext`] for `annotate`.
///
/// Honours `fixture_dir=<path>` argv (used by the integration tests to
/// redirect to a sandbox tempdir). Falls back to the production
/// `tests/fixtures/eval/` under `CARGO_MANIFEST_DIR`. The timestamp
/// closure resolves to the live [`capture_timestamp_label`].
fn annotation_context(kvs: &HashMap<String, String>) -> AnnotationContext {
    let fixture_dir = kvs.get("fixture_dir").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval"),
        PathBuf::from,
    );
    AnnotationContext {
        fixture_dir,
        timestamp: Box::new(capture_timestamp_label),
    }
}

/// Category allowlist for `Entry.category`. Tracks the eight categories
/// present in `tests/fixtures/eval/queries.jsonl` (AS-001). A new category
/// in the fixture must be propagated here (BR-004) before annotations
/// containing it will be accepted.
const CATEGORY_ALLOWLIST: &[&str] = &[
    "comparative",
    "conceptual",
    "definitional",
    "factoid",
    "howto",
    "listing",
    "troubleshooting",
    "variant_notation",
];

/// Maximum graded relevance value accepted in `Entry.relevance_map`. The
/// `u8` parse already enforces ≥ 0 (FR-V002).
const MAX_RELEVANCE_GRADE: u8 = 3;

/// FR-011: literal stderr line emitted when `annotate` is invoked with a
/// TTY stdin. Phase 1 has no interactive UX (deferred to sub-PR-D), so
/// the only safe ingress is `cat session.jsonl | eval_harness annotate ...`.
const TTY_REJECT_MESSAGE: &str =
    "annotate: pipe jsonl into stdin (interactive UX deferred to sub-PR-D)";

/// FR-011 dispatch helper. Returns `Some(ExitCode)` for a TTY stdin so
/// callers can short-circuit before stdin parsing; `None` lets the caller
/// continue. Pure boolean argument so unit tests can verify the branch
/// table without faking [`std::io::IsTerminal`] (CI lanes inherit stdin as
/// pipe / null and cannot reproduce the TTY case via subprocess spawn).
fn check_stdin_tty_guard(is_tty: bool) -> Option<ExitCode> {
    if is_tty {
        eprintln!("{TTY_REJECT_MESSAGE}");
        Some(ExitCode::from(EXIT_USAGE))
    } else {
        None
    }
}

/// `annotate output=<path> annotator_id=<id> session_id=<id>` — Phase 1
/// block-mode session capture (Issue #53 sub-PR-B).
///
/// Reads jsonl [`Entry`] lines from stdin, builds a [`Session`] with
/// canonical [`Provenance`], runs `validate_schema_version`, and atomic-writes
/// pretty JSON to `<path>`. MLX-free per NFR-001 — excluded from
/// [`MLX_DEPENDENT_MODES`].
fn run_annotate(kvs: &HashMap<String, String>) -> ExitCode {
    let Some(output_path_raw) = kvs.get("output") else {
        eprintln!("annotate: output= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    if let Some(exit) = check_stdin_tty_guard(io::stdin().is_terminal()) {
        return exit;
    }
    let output_path = match validate_output_path(output_path_raw) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("annotate: {msg}");
            return ExitCode::from(EXIT_USAGE);
        }
    };
    let Some(annotator_id) = kvs.get("annotator_id") else {
        eprintln!("annotate: annotator_id= argument required");
        return ExitCode::from(EXIT_USAGE);
    };
    let Some(session_id) = kvs.get("session_id") else {
        eprintln!("annotate: session_id= argument required");
        return ExitCode::from(EXIT_USAGE);
    };

    let ctx = annotation_context(kvs);
    let queries_jsonl_hash = match hash_fixture_dir(&ctx.fixture_dir) {
        Ok(h) => h,
        Err(msg) => {
            eprintln!("annotate: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };

    let stdin = io::stdin();
    let mut entries: Vec<Entry> = Vec::new();
    let mut seen_ids: HashMap<String, usize> = HashMap::new();
    for (line_idx, line_result) in stdin.lock().lines().enumerate() {
        let line_no = line_idx + 1;
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("annotate: line {line_no} read error: {e}");
                return ExitCode::from(EXIT_USAGE);
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let entry: Entry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("annotate: line {line_no} parse error: {e}");
                return ExitCode::from(EXIT_USAGE);
            }
        };
        if !CATEGORY_ALLOWLIST.contains(&entry.category.as_str()) {
            eprintln!(
                "annotate: line {line_no} invalid category {got:?}; expected one of {CATEGORY_ALLOWLIST:?}",
                got = entry.category,
            );
            return ExitCode::from(EXIT_USAGE);
        }
        if let Some((doc_id, grade)) = entry
            .relevance_map
            .iter()
            .find(|(_, g)| **g > MAX_RELEVANCE_GRADE)
        {
            eprintln!("annotate: line {line_no} invalid grade {doc_id}={grade}; expected 0..=3");
            return ExitCode::from(EXIT_USAGE);
        }
        if let Some(prev_line) = seen_ids.get(&entry.id) {
            eprintln!(
                "annotate: line {line_no} duplicate id {id} (first seen on line {prev_line})",
                id = entry.id,
            );
            return ExitCode::from(EXIT_USAGE);
        }
        seen_ids.insert(entry.id.clone(), line_no);
        entries.push(entry);
    }

    if entries.is_empty() {
        eprintln!("annotate: empty session, no entries written");
        return ExitCode::from(EXIT_USAGE);
    }

    let session = Session {
        provenance: Provenance {
            schema_version: ANNOTATION_SCHEMA_VERSION.to_owned(),
            captured_with: "annotate".to_owned(),
            timestamp: (ctx.timestamp)(),
            annotator_id: annotator_id.clone(),
            session_id: session_id.clone(),
            queries_jsonl_hash,
        },
        entries,
    };

    if let Err(e) = session.validate_schema_version() {
        eprintln!("annotate: schema version mismatch: {e}");
        return ExitCode::from(EXIT_INFRA);
    }

    if let Err(e) = session.write_json(&output_path) {
        eprintln!("annotate: write failed: {e}");
        return ExitCode::from(EXIT_INFRA);
    }

    let n_entries = session.entries.len();
    eprintln!(
        "annotate: wrote {} ({n_entries} entries)",
        output_path.display()
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-081-003: subcommand_sets_partition_all_subcommands
    //
    // Issue #81 Code fix #9: ALL_SUBCOMMANDS must be the disjoint union of
    // MLX_DEPENDENT_MODES and NON_MLX_MODES. The compile-time `const_assert_eq!`
    // on lengths is necessary but not sufficient — it permits a swap (one entry
    // moved from MLX to non-MLX with another entry going the other way). This
    // runtime check pins set equality and disjointness so a misclassified
    // subcommand cannot silently bypass the seatbelt-sandbox gate.
    #[test]
    fn subcommand_sets_partition_all_subcommands() {
        let mlx: HashSet<&str> = MLX_DEPENDENT_MODES.iter().copied().collect();
        let non_mlx: HashSet<&str> = NON_MLX_MODES.iter().copied().collect();
        let all: HashSet<&str> = ALL_SUBCOMMANDS.iter().copied().collect();
        assert!(
            mlx.is_disjoint(&non_mlx),
            "MLX-dependent and non-MLX sets must be disjoint; \
             overlap means a subcommand is both gated and ungated. mlx={mlx:?} non_mlx={non_mlx:?}"
        );
        let union: HashSet<&str> = mlx.union(&non_mlx).copied().collect();
        assert_eq!(
            union, all,
            "MLX ∪ non-MLX must equal ALL_SUBCOMMANDS; \
             missing entries can silently bypass the seatbelt classification"
        );
    }

    // T-004: tty_reject_message_contains_fr_011_substring
    // FR-011: TTY_REJECT_MESSAGE must literally contain
    //         "pipe jsonl into stdin" so an interactive run prints the
    //         exact substring required by Spec FR-011.
    //
    // Spec Evolution 2026-05-10: integration variant infeasible —
    // `cargo test` lanes inherit stdin as pipe / null, so
    // `is_terminal() == true` is unreproducible across CI hosts. The
    // boolean dispatch is verified separately in
    // `tty_guard_returns_usage_for_terminal` so the constant + dispatch
    // pair fully cover FR-011.
    #[test]
    fn tty_reject_message_contains_fr_011_substring() {
        assert!(
            TTY_REJECT_MESSAGE.contains("pipe jsonl into stdin"),
            "FR-011: TTY_REJECT_MESSAGE must contain 'pipe jsonl into stdin'; \
             got: {TTY_REJECT_MESSAGE}"
        );
    }

    // T-004: tty_guard_returns_usage_for_terminal
    // FR-011: when stdin is a TTY, `check_stdin_tty_guard` returns
    //         `Some(ExitCode)`; otherwise it returns `None` so
    //         `run_annotate` proceeds with stdin parsing.
    #[test]
    fn tty_guard_returns_usage_for_terminal() {
        assert!(
            check_stdin_tty_guard(true).is_some(),
            "FR-011: TTY stdin must short-circuit run_annotate"
        );
        assert!(
            check_stdin_tty_guard(false).is_none(),
            "FR-011: piped stdin must not short-circuit run_annotate"
        );
    }

    // T-067-007: aggregation_kind_topk_average_roundtrips_through_storage_enum
    //
    // A non-default `topk_k=` passed at capture time used to be silently
    // downgraded to DEFAULT_TOPK_AVERAGE_K by `from_name` at verify time.
    // `AggregationKind::TopKAverage(k)` now encodes `k` in the JSON wire form
    // as `"topk-average:k"`, so a round-trip through serde keeps the strategy
    // bit-identical across capture/verify.
    #[test]
    fn aggregation_kind_topk_average_roundtrips_through_storage_enum() {
        for k in [1, 3, 5, 100] {
            let original_spec = AggregationSpec::TopKAverage(k);
            let kind: AggregationKind = original_spec.into();
            let json = serde_json::to_string(&kind).expect("serialise AggregationKind");
            assert_eq!(
                json,
                format!("\"topk-average:{k}\""),
                "TopKAverage({k}) must serialise with k encoded in the wire form"
            );
            let parsed_kind: AggregationKind =
                serde_json::from_str(&json).expect("deserialise AggregationKind");
            let parsed_spec =
                AggregationSpec::try_from(parsed_kind).expect("storage→runtime bridge");
            assert_eq!(
                parsed_spec, original_spec,
                "round-trip lost k for TopKAverage({k}): {parsed_spec:?}"
            );
        }
    }

    // T-067-008: aggregation_kind_legacy_bare_topk_average_is_rejected
    //
    // Pre-fix baselines that wrote bare "topk-average" without `:k` are
    // explicitly rejected at deserialise time. Silently falling back to
    // DEFAULT_TOPK_AVERAGE_K would re-introduce the bug that the encoded
    // form fixes — verify-baseline would dispatch at the default k instead
    // of the captured k. No committed fixture uses the bare form, so the
    // rejection has zero footprint.
    #[test]
    fn aggregation_kind_legacy_bare_topk_average_is_rejected() {
        let result = serde_json::from_str::<AggregationKind>("\"topk-average\"");
        assert!(
            result.is_err(),
            "bare 'topk-average' must be rejected to prevent silent default-k fallback; \
             got {result:?}"
        );
    }

    // T-067-009: aggregation_kind_simple_variants_roundtrip
    //
    // Identity / MaxChunk / Dedupe / NotApplicable carry no `k` payload, but
    // they share the round-trip contract through the storage enum.
    #[test]
    fn aggregation_kind_simple_variants_roundtrip() {
        for original in [
            AggregationKind::Identity,
            AggregationKind::MaxChunk,
            AggregationKind::Dedupe,
            AggregationKind::NotApplicable,
        ] {
            let json = serde_json::to_string(&original).expect("serialise");
            let parsed: AggregationKind = serde_json::from_str(&json).expect("deserialise");
            assert_eq!(parsed, original, "round-trip mismatch for {original:?}");
        }
    }

    // T-067-010: aggregation_kind_not_applicable_rejects_runtime_dispatch
    //
    // NotApplicable is the FirstSearchReplay-only marker. The
    // `TryFrom<AggregationKind> for AggregationSpec` bridge must reject it
    // so verify-baseline cannot accidentally dispatch a Stage 3 strategy
    // on a replay baseline.
    #[test]
    fn aggregation_kind_not_applicable_rejects_runtime_dispatch() {
        let result = AggregationSpec::try_from(AggregationKind::NotApplicable);
        assert!(
            result.is_err(),
            "NotApplicable must not lower to a runtime AggregationSpec; got {result:?}"
        );
    }

    // T-068-012: parse_merge_config_empty_kvs_returns_default
    #[test]
    fn parse_merge_config_empty_kvs_returns_default() {
        let kvs: HashMap<String, String> = HashMap::new();
        let parsed = parse_merge_config_from_kvs(&kvs).expect("empty kvs must parse");
        assert_eq!(parsed, HybridSearchConfig::default());
    }

    // T-068-013: parse_merge_config_all_overrides_applied
    #[test]
    fn parse_merge_config_all_overrides_applied() {
        let mut kvs: HashMap<String, String> = HashMap::new();
        kvs.insert("rrf_k".to_owned(), "30".to_owned());
        kvs.insert("fts_weight".to_owned(), "2.5".to_owned());
        kvs.insert("vector_weight".to_owned(), "0.5".to_owned());
        let parsed = parse_merge_config_from_kvs(&kvs).expect("override kvs must parse");
        assert!((parsed.rrf_k - 30.0).abs() < f64::EPSILON);
        assert!(
            (parsed.source_weights[&CandidateSource::Fts] - 2.5).abs() < f64::EPSILON,
            "fts_weight override must be applied"
        );
        assert!(
            (parsed.source_weights[&CandidateSource::Vector] - 0.5).abs() < f64::EPSILON,
            "vector_weight override must be applied"
        );
    }

    // T-068-014: parse_merge_config_invalid_value_returns_err
    #[test]
    fn parse_merge_config_invalid_value_returns_err() {
        let mut kvs: HashMap<String, String> = HashMap::new();
        kvs.insert("rrf_k".to_owned(), "not-a-number".to_owned());
        let result = parse_merge_config_from_kvs(&kvs);
        assert!(
            result.is_err(),
            "non-numeric rrf_k must surface a parse error"
        );
    }

    // T-068-015: parse_merge_config_partial_keeps_other_defaults
    #[test]
    fn parse_merge_config_partial_keeps_other_defaults() {
        let mut kvs: HashMap<String, String> = HashMap::new();
        kvs.insert("fts_weight".to_owned(), "3.0".to_owned());
        let parsed = parse_merge_config_from_kvs(&kvs).expect("partial kvs must parse");
        // fts overridden
        assert!((parsed.source_weights[&CandidateSource::Fts] - 3.0).abs() < f64::EPSILON);
        // others stay at default
        assert!((parsed.rrf_k - 60.0).abs() < f64::EPSILON);
        assert!((parsed.source_weights[&CandidateSource::Vector] - 1.0).abs() < f64::EPSILON);
    }

    // T-061-010: `MetricSpec::ALL` leads with HitAt1 / HitAt3.
    #[test]
    fn metric_spec_all_starts_with_hit_variants_in_order() {
        assert_eq!(
            MetricSpec::ALL[0],
            MetricSpec::HitAt1,
            "MetricSpec::ALL[0] must be HitAt1 so baseline.json global[0] is hit@1"
        );
        assert_eq!(
            MetricSpec::ALL[1],
            MetricSpec::HitAt3,
            "MetricSpec::ALL[1] must be HitAt3 so baseline.json global[1] is hit@3"
        );
    }

    // T-061-011: HitAt{1,3}.k() returns the cutoff encoded in the variant name.
    #[test]
    fn metric_spec_hit_variants_report_correct_k() {
        assert_eq!(
            MetricSpec::HitAt1.k(),
            1,
            "HitAt1.k() must be 1 to dispatch hit_at_k against the top-1 window"
        );
        assert_eq!(
            MetricSpec::HitAt3.k(),
            3,
            "HitAt3.k() must be 3 to dispatch hit_at_k against the top-3 window"
        );
    }

    // T-061-012: from_name("hit@{1,3}") resolves to HitAt{1,3}.
    #[test]
    fn metric_spec_from_name_resolves_hit_labels() {
        assert_eq!(
            MetricSpec::from_name("hit@1"),
            Some(MetricSpec::HitAt1),
            "from_name(\"hit@1\") must resolve to HitAt1 so verify-baseline can look up tolerance"
        );
        assert_eq!(
            MetricSpec::from_name("hit@3"),
            Some(MetricSpec::HitAt3),
            "from_name(\"hit@3\") must resolve to HitAt3 so verify-baseline can look up tolerance"
        );
    }

    // T-061-013: HitAt1.metric_fn() dispatches to hit_at_k.
    //
    // Grade-0 input is the discriminator: hit_at_k returns 0.0, while a wrong
    // dispatch to recall_at_k would also return 0.0 only by coincidence (empty
    // relevance set), so this case anchors the dispatch contract specifically.
    #[test]
    fn metric_spec_hit1_metric_fn_dispatches_to_hit_at_k() {
        let ranked: &[&str] = &["d1"];
        let relevance: HashMap<String, u8> = HashMap::from([("d1".to_owned(), 0u8)]);
        let dispatch = MetricSpec::HitAt1.metric_fn();
        let result = dispatch(ranked, &relevance, 1);
        assert!(
            result.abs() < f64::EPSILON,
            "HitAt1.metric_fn() on grade-0 input must yield 0.0; got {result}"
        );
    }

    // ── Issue #62 / Phase 2.3 — replay-first-search subcommand + verify-baseline 拡張 ──

    // T-068-001: ADR-0005 — LazyReranker init must not fire when only the replay path runs.
    #[test]
    fn replay_first_search_skips_reranker_init() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use rurico::embed::MockEmbedder;
        use rurico::reranker::MockReranker;
        use tempfile::tempdir;

        let dir = tempdir().expect("tempdir for replay output");
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval");
        let output_path = dir.path().join("replay.json");

        let init_calls = Arc::new(AtomicUsize::new(0));
        let reranker = LazyReranker::new({
            let init_calls = Arc::clone(&init_calls);
            move || {
                init_calls.fetch_add(1, Ordering::SeqCst);
                Ok(MockReranker::default())
            }
        });

        let ctx = EvalContext {
            fixture_dir,
            embedder: MockEmbedder::default(),
            reranker,
            timestamp: Box::new(|| "epoch:0".to_owned()),
        };

        let merge_config = HybridSearchConfig::default();
        let normalization = QueryNormalizationConfig::default();
        let _exit = run_replay_first_search_with(&ctx, &output_path, &merge_config, &normalization);

        let snapshot = read_snapshot(&output_path).unwrap_or_else(|e| panic!("T-068-001: {e}"));
        assert_eq!(
            snapshot.kind,
            BaselineKind::FirstSearchReplay,
            "T-068-001: snapshot kind must be FirstSearchReplay (proves pipeline ran)"
        );
        assert_eq!(
            init_calls.load(Ordering::SeqCst),
            0,
            "T-068-001: replay-first-search must not invoke the reranker init closure"
        );
    }

    /// Build a BaselineSnapshot with the requested schema/kind so envelope
    /// validation tests don't drag in MockEmbedder + fixture setup.
    fn snapshot_for_envelope_test(schema: &str, kind: BaselineKind) -> BaselineSnapshot {
        BaselineSnapshot {
            schema_version: schema.to_owned(),
            kind,
            captured_with: "test".to_owned(),
            timestamp: "epoch:0".to_owned(),
            model_id: "test/model".to_owned(),
            model_revision: "rev".to_owned(),
            mlx_rs_version: "0.0.0".to_owned(),
            fixture_hash: "fnv1a64:0".to_owned(),
            aggregation: AggregationKind::Identity,
            merge_config: HybridSearchConfig::default(),
            normalization: QueryNormalizationConfig::default(),
            global: vec![],
            per_category: BTreeMap::new(),
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
        }
    }

    /// Minimal (results, queries) pair for snapshot-builder tests.
    /// Empty `ranked_hits` is fine — MetricSpec::ALL still iterates and
    /// emits 0.0 for every metric, so ordering / presence assertions hold.
    fn replay_snapshot_inputs() -> (Vec<QueryResult>, Vec<EvalQuery>) {
        let queries = vec![EvalQuery {
            id: "q1".to_owned(),
            text: "alpha".to_owned(),
            category: "test".to_owned(),
            relevance_map: HashMap::from([("d1".to_owned(), 1u8)]),
            annotation: "test".to_owned(),
        }];
        let results = vec![QueryResult {
            query_id: "q1".to_owned(),
            ranked_hits: vec![],
            latency_ms: 5,
        }];
        (results, queries)
    }

    fn build_test_replay_snapshot() -> BaselineSnapshot {
        let (results, queries) = replay_snapshot_inputs();
        build_baseline_snapshot(
            BaselineKind::FirstSearchReplay,
            "eval_harness replay-first-search",
            AggregationKind::NotApplicable,
            &results,
            &queries,
            "fnv1a64:0".to_owned(),
            "epoch:42".to_owned(),
            &HybridSearchConfig::default(),
            &QueryNormalizationConfig::default(),
        )
    }

    // T-062-016: replay_first_search_subcommand_dispatches_to_evaluate_first_search_replay
    //
    // FR-016 / FR-017: the subcommand kind label maps through
    // `parse_baseline_kind` to `BaselineKind::FirstSearchReplay`. Pinning
    // the kind string round-trip guards against an accidental rename
    // breaking the dispatch wiring (main()'s match arm + verify-baseline
    // kind=).
    #[test]
    fn replay_first_search_subcommand_dispatches_to_evaluate_first_search_replay() {
        assert_eq!(
            parse_baseline_kind("first_search_replay")
                .expect("first_search_replay kind must parse"),
            BaselineKind::FirstSearchReplay,
            "FR-016/FR-017: subcommand kind label must map to FirstSearchReplay"
        );
    }

    // T-062-017: replay_first_search_writes_baseline_with_kind_first_search_replay
    //
    // FR-019: snapshot envelope kind == FirstSearchReplay so consumers can
    // tell the file apart from forward / oracle baselines.
    #[test]
    fn replay_first_search_writes_baseline_with_kind_first_search_replay() {
        let snap = build_test_replay_snapshot();
        assert_eq!(snap.kind, BaselineKind::FirstSearchReplay);
    }

    // T-062-018: replay_first_search_writes_captured_with_replay_first_search
    //
    // FR-019: provenance string lets log readers tell which subcommand
    // produced the file. Also doubles as the regenerate hint that
    // `verify-baseline` emits on schema_version mismatch.
    #[test]
    fn replay_first_search_writes_captured_with_replay_first_search() {
        let snap = build_test_replay_snapshot();
        assert_eq!(snap.captured_with, "eval_harness replay-first-search");
    }

    // T-062-019: replay_first_search_writes_aggregation_none
    //
    // BR-003: replay path runs Stage 3 = MaxChunkAggregator only — no
    // ranking-aware aggregator is invoked, so the field marker is "none".
    #[test]
    fn replay_first_search_writes_aggregation_none() {
        let snap = build_test_replay_snapshot();
        assert_eq!(
            snap.aggregation,
            AggregationKind::NotApplicable,
            "BR-003: replay path's Stage 3 strategy is MaxChunkAggregator only \
             (no ranking-aware aggregator) → aggregation marker is NotApplicable"
        );
    }

    // T-062-020: verify_baseline_dispatches_on_first_search_replay_kind
    //
    // FR-021 / FR-022: when committed.kind == FirstSearchReplay and the
    // optional kind=first_search_replay argument matches, the envelope
    // check passes — caller (`run_verify_baseline_with`) then dispatches
    // to `evaluate_first_search_replay`.
    #[test]
    fn verify_baseline_dispatches_on_first_search_replay_kind() {
        let snap =
            snapshot_for_envelope_test(BASELINE_SCHEMA_VERSION, BaselineKind::FirstSearchReplay);
        let mut kvs = HashMap::new();
        kvs.insert("kind".to_owned(), "first_search_replay".to_owned());
        let result = validate_committed_baseline_envelope(&snap, &kvs);
        assert!(
            result.is_ok(),
            "FR-021/FR-022: replay kind + matching kvs must pass envelope check; got {result:?}"
        );
    }

    // T-062-021: verify_baseline_iterates_metric_spec_all_for_replay_kind
    //
    // FR-022 / NFR-005 / BR-001: replay path emits every MetricSpec::ALL
    // entry to global, in the same order as forward / oracle. Ensures
    // verify-baseline's `MetricSpec::ALL` iterator (Issue #61 soundness
    // gap) covers FirstSearchReplay too.
    #[test]
    fn verify_baseline_iterates_metric_spec_all_for_replay_kind() {
        let snap = build_test_replay_snapshot();
        let names: Vec<&str> = snap.global.iter().map(|m| m.name.as_str()).collect();
        let expected: Vec<&str> = MetricSpec::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(
            names, expected,
            "FR-022/NFR-005/BR-001: replay global must mirror MetricSpec::ALL order"
        );
    }

    // T-062-022: verify_baseline_rejects_committed_baseline_with_schema_version_1_2
    //
    // FR-023 / NFR-005: stale 1.2 baseline must `EXIT_REGRESSION` (not
    // silent skip). Ensures Issue #62's SCHEMA bump gate matches Issue
    // #61's verify-baseline soundness contract.
    #[test]
    fn verify_baseline_rejects_committed_baseline_with_schema_version_1_2() {
        let snap = snapshot_for_envelope_test("1.2", BaselineKind::Forward);
        let kvs = HashMap::new();
        let result = validate_committed_baseline_envelope(&snap, &kvs);
        assert_eq!(
            result,
            Err(EXIT_REGRESSION),
            "FR-023/NFR-005: stale 1.2 baseline must EXIT_REGRESSION"
        );
    }

    // T-062-023: replay_first_search_records_latency_p50_ms
    //
    // FR-019 / NFR-004: the replay snapshot carries latency_p50_ms so the
    // ADR-0003 第 4 reassessment trigger ("forward 比 2-5x 高速化に届かない")
    // can be checked against committed JSON.
    #[test]
    fn replay_first_search_records_latency_p50_ms() {
        let snap = build_test_replay_snapshot();
        assert!(
            snap.latency_p50_ms.is_finite(),
            "FR-019/NFR-004: latency_p50_ms must be finite, got {}",
            snap.latency_p50_ms
        );
    }

    // T-062-024: replay_first_search_records_latency_p95_ms
    //
    // FR-019 / NFR-004: tail latency p95 lives next to p50 in the snapshot
    // envelope so capacity drift can be tracked alongside median speedup.
    #[test]
    fn replay_first_search_records_latency_p95_ms() {
        let snap = build_test_replay_snapshot();
        assert!(
            snap.latency_p95_ms.is_finite(),
            "FR-019/NFR-004: latency_p95_ms must be finite, got {}",
            snap.latency_p95_ms
        );
    }

    // T-062-025: replay_first_search_emits_hit_at_k_in_global_metrics
    //
    // FR-018: replay reuses `build_global_metrics`, which iterates
    // `MetricSpec::ALL` (HitAt1 / HitAt3 first per Issue #61). Pins the
    // first two slots so the ADR-0003 "first-result quality" indicator
    // is visible at the top of the snapshot JSON.
    #[test]
    fn replay_first_search_emits_hit_at_k_in_global_metrics() {
        let snap = build_test_replay_snapshot();
        assert_eq!(
            snap.global.first().map(|m| m.name.as_str()),
            Some("hit@1"),
            "FR-018: replay global[0] must be hit@1 (Issue #61 ALL ordering)"
        );
        assert_eq!(
            snap.global.get(1).map(|m| m.name.as_str()),
            Some("hit@3"),
            "FR-018: replay global[1] must be hit@3"
        );
    }

    // T-062-026: verify_baseline_rejects_kind_mismatch
    //
    // FR-024 / AC-7: `verify-baseline baseline=forward.json kind=replay`
    // must `EXIT_REGRESSION`. Guards against running the replay metric
    // dispatch against a forward snapshot (or vice versa).
    #[test]
    fn verify_baseline_rejects_kind_mismatch() {
        let snap = snapshot_for_envelope_test(BASELINE_SCHEMA_VERSION, BaselineKind::Forward);
        let mut kvs = HashMap::new();
        kvs.insert("kind".to_owned(), "first_search_replay".to_owned());
        let result = validate_committed_baseline_envelope(&snap, &kvs);
        assert_eq!(
            result,
            Err(EXIT_REGRESSION),
            "FR-024/AC-7: kind mismatch (forward baseline + kind=replay) must EXIT_REGRESSION"
        );
    }
}
