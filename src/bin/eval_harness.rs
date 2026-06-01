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

/// Load-bearing constant for the baseline.json schema contract (ADR-0006).
/// The `*_at_k` metric names (`recall@10`, `mrr@10`, `ndcg@10`) and the
/// committed `MetricResult.k` values for those metrics all derive from this
/// value, so any bump silently invalidates every `tests/fixtures/eval/*.json`
/// baseline.
///
/// Change protocol (ADR-0006 § Decision Outcome 2):
/// - (a) bump `BASELINE_SCHEMA_VERSION`
/// - (b) recapture every `tests/fixtures/eval/*.json` baseline
/// - (c) append "PIPELINE_K change" row to ADR-0002 § Reassessment Triggers
/// - (d) supersede ADR-0006 with a new ADR documenting the new value
///
/// `validate_committed_baseline_envelope` rejects a committed baseline whose
/// recorded `MetricResult.k` disagrees with what `MetricSpec` declares for
/// that metric name, so a stale baseline fails fast before model load
/// (ADR-0006 § Implementation Guidelines 第 2 項).
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
/// Pinned ruri-v3-310m HuggingFace repo id. The canonical value lives in
/// `rurico::embed::ModelId::repo_id` (only reachable via the crate-private
/// `ModelArtifact` trait); this label is what baseline.json carries and what
/// download / run logs print for provenance.
const RURI_V3_310M_MODEL_ID: &str = "cl-nagoya/ruri-v3-310m";

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
        model_id: RURI_V3_310M_MODEL_ID.to_owned(),
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
    for metric in &committed.global {
        if let Some(spec) = MetricSpec::ALL.iter().find(|s| s.name() == metric.name)
            && metric.k != spec.k()
        {
            eprintln!(
                "verify-baseline: failed — committed metric {:?} has k={} but \
                 MetricSpec defines k={} (ADR-0006 schema contract); K mismatch \
                 invalidates baseline semantics — regenerate via \
                 {{capture-baseline | capture-oracle | replay-first-search}} before verifying",
                metric.name,
                metric.k,
                spec.k()
            );
            return Err(EXIT_REGRESSION);
        }
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
        "{mode}: fixture={}{kind_part}{path_part} model={RURI_V3_310M_MODEL_ID} seed_shuffle={SHUFFLE_SEED} seed_bootstrap={BOOTSTRAP_SEED}",
        ctx.fixture_dir.display(),
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
    let model_id = embed::ModelId::DEFAULT;
    let artifacts =
        match embed::cached_artifacts(model_id).map_err(|e| format!("embed cache lookup: {e}"))? {
            Some(a) => a,
            None => {
                eprintln!("embed model not cached, downloading {RURI_V3_310M_MODEL_ID}...");
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

// bin クレート root は submodule を同ディレクトリ (src/bin/) から解決するため、
// lib の `foo/tests.rs` 規則と異なり #[path] で明示する。
#[cfg(test)]
#[path = "eval_harness/tests.rs"]
mod tests;
