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
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use amici::eval::baseline::{
    BASELINE_SCHEMA_VERSION, BaselineKind, BaselineSnapshot, atomic_write, build_metric_result,
    write_json,
};
use amici::eval::fixture::{
    EvalDocument, EvalQuery, load_documents, load_known_answers, load_queries,
};
use amici::eval::metrics::{MetricResult, bootstrap_ci, mrr_at_k, ndcg_at_k, recall_at_k};
use amici::eval::pipeline::{PipelineConfig, PipelineError, QueryResult, evaluate as run_pipeline};
use rurico::embed::Embed;
use rurico::reranker::Rerank;
use rurico::retrieval::{
    CandidateSource, DedupeAggregator, HybridSearchConfig, IdentityAggregator, MaxChunkAggregator,
    MergedHit, TopKAverageAggregator,
};
use rurico::sandbox::exit_if_seatbelt;
use rurico::storage::QueryNormalizationConfig;
use rurico::{embed, model_probe, reranker};

/// Mock-friendly bundle of every external seam the four mode handlers touch.
///
/// Generic over the embedder and reranker types so production wiring uses
/// concrete `embed::Embedder` / `reranker::Reranker` while tests can swap
/// in `MockEmbedder` / `MockReranker`. The `timestamp` closure isolates
/// `SystemTime::now()` from the snapshot-write code path so tests can fix a
/// deterministic capture-time label.
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

/// Build the production [`EvalContext`] — loads the cached MLX models and
/// resolves the fixture directory under the crate manifest.
fn production_context() -> Result<EvalContext<embed::Embedder, reranker::Reranker>, String> {
    let embedder = init_embedder()?;
    let reranker = init_reranker()?;
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
const VALID_EVALUATE_KINDS: &[&str] = &["full", "shuffled", "identity", "reverse", "single_doc"];

/// Closed set of Stage 3 aggregation kinds accepted by `aggregation=...`.
/// Anchors argv validation and round-tripping with the `aggregation` field
/// on [`BaselineSnapshot`].
const VALID_AGGREGATION_KINDS: &[&str] = &["identity", "max-chunk", "dedupe", "topk-average"];

/// Default `k` for the `topk-average` strategy when no `topk_k=` override is
/// supplied. `k = 3` matches the issue body's "top-k average over the top
/// chunks per document" framing without needing a flag for first capture.
const DEFAULT_TOPK_AVERAGE_K: usize = 3;

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

/// Closed set of Stage 3 aggregation kinds. Anchors `aggregation=` argv
/// validation, the JSON `aggregation` field on [`BaselineSnapshot`], and
/// [`run_verify_baseline`]'s reverse lookup of which strategy to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregationKind {
    Identity,
    MaxChunk,
    Dedupe,
    TopKAverage(usize),
}

impl AggregationKind {
    /// Stable label written to `BaselineSnapshot.aggregation`. `TopKAverage`
    /// encodes `k` as `"topk-average:K"` so a baseline captured with a
    /// non-default `k` round-trips through [`Self::from_name`] without
    /// silently falling back to [`DEFAULT_TOPK_AVERAGE_K`] at verify time.
    fn name(self) -> String {
        match self {
            Self::Identity => "identity".to_owned(),
            Self::MaxChunk => "max-chunk".to_owned(),
            Self::Dedupe => "dedupe".to_owned(),
            Self::TopKAverage(k) => format!("topk-average:{k}"),
        }
    }

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

    /// Resolve `BaselineSnapshot.aggregation` back to a dispatchable kind for
    /// `verify-baseline`. Recognises both the encoded `"topk-average:K"` form
    /// (post-fix) and the legacy bare `"topk-average"` form (pre-fix
    /// baselines fall back to [`DEFAULT_TOPK_AVERAGE_K`]).
    fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "identity" => Ok(Self::Identity),
            "max-chunk" => Ok(Self::MaxChunk),
            "dedupe" => Ok(Self::Dedupe),
            "topk-average" => Ok(Self::TopKAverage(DEFAULT_TOPK_AVERAGE_K)),
            other => match other.strip_prefix("topk-average:") {
                Some(k_str) => k_str
                    .parse::<usize>()
                    .map(Self::TopKAverage)
                    .map_err(|e| format!("topk-average:K parse error in baseline: {e}")),
                None => Err(format!(
                    "unknown aggregation in baseline: {other:?}; expected one of {VALID_AGGREGATION_KINDS:?}"
                )),
            },
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
    aggregation: AggregationKind,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
    config: &PipelineConfig,
) -> Result<Vec<QueryResult>, PipelineError>
where
    E: Embed,
    R: Rerank,
{
    match aggregation {
        AggregationKind::Identity => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &IdentityAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationKind::MaxChunk => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &MaxChunkAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationKind::Dedupe => run_pipeline(
            corpus,
            queries,
            embedder,
            reranker,
            &DedupeAggregator,
            merge_config,
            normalization,
            config,
        ),
        AggregationKind::TopKAverage(k) => run_pipeline(
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
/// `verify-baseline` resolves the committed name back through
/// [`MetricSpec::from_name`] to look up its tolerance. Per-metric tolerance
/// bounds (FR-017) come from [`MetricSpec::tolerance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricSpec {
    RecallAt5,
    RecallAt10,
    MrrAt10,
    NdcgAt10,
}

impl MetricSpec {
    /// All specs in canonical emission order. `BaselineSnapshot.global` is
    /// produced by mapping over this slice.
    const ALL: &'static [Self] = &[
        Self::RecallAt5,
        Self::RecallAt10,
        Self::MrrAt10,
        Self::NdcgAt10,
    ];

    /// JSON-serialised metric label as it appears in `MetricResult.name`.
    const fn name(self) -> &'static str {
        match self {
            Self::RecallAt5 => "recall@5",
            Self::RecallAt10 => "recall@10",
            Self::MrrAt10 => "mrr@10",
            Self::NdcgAt10 => "ndcg@10",
        }
    }

    const fn k(self) -> usize {
        match self {
            Self::RecallAt5 => 5,
            Self::RecallAt10 | Self::MrrAt10 | Self::NdcgAt10 => 10,
        }
    }

    fn metric_fn(self) -> MetricFn {
        match self {
            Self::RecallAt5 | Self::RecallAt10 => recall_at_k,
            Self::MrrAt10 => mrr_at_k,
            Self::NdcgAt10 => ndcg_at_k,
        }
    }

    /// Per-metric drift tolerance for `verify-baseline` (FR-017).
    ///
    /// Cross-process MLX reranker forward exhibits f32 non-determinism that
    /// propagates to score-sensitive metrics. Embedder forward is bit-identical
    /// (proven by `mlx_smoke verify-fixture`); the bound below absorbs the
    /// reranker-side noise. See ADR 0003 § Reproducibility.
    ///
    /// Bounds set to ≥ 2× empirically observed max drift (N=10 + historical
    /// session max), keeping >1% regression detectable while accepting
    /// non-determinism inherent to Apple Silicon Metal f32 ops.
    const fn tolerance(self) -> f64 {
        match self {
            Self::RecallAt5 => 1e-2,
            Self::RecallAt10 | Self::MrrAt10 | Self::NdcgAt10 => 1e-3,
        }
    }

    /// Inverse of [`name()`]; returns `None` for unknown labels (e.g. a
    /// committed baseline produced by an older harness version).
    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|s| s.name() == name)
    }
}

fn main() -> ExitCode {
    model_probe::handle_probe_if_needed();
    exit_if_seatbelt(env!("CARGO_BIN_NAME"));

    let args: Vec<String> = env::args().skip(1).collect();
    let Some(mode) = args.first() else {
        eprintln!(
            "usage: eval_harness <evaluate|capture-baseline|capture-reverse-baseline|\
             verify-baseline|compare-baselines> [key=value...]"
        );
        return ExitCode::from(EXIT_USAGE);
    };
    let kvs: HashMap<String, String> = args[1..]
        .iter()
        .filter_map(|s| s.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
        .collect();

    match mode.as_str() {
        "evaluate" => run_evaluate(&kvs),
        "capture-baseline" => run_capture_baseline(&kvs),
        "capture-reverse-baseline" => run_capture_reverse_baseline(&kvs),
        "verify-baseline" => run_verify_baseline(&kvs),
        "compare-baselines" => run_compare_baselines(&kvs),
        other => {
            eprintln!("unknown mode: {other}");
            ExitCode::from(EXIT_USAGE)
        }
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
    let aggregation = match AggregationKind::from_kvs(kvs) {
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
    aggregation: AggregationKind,
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
    let mut results = match dispatch_pipeline(
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
            eprintln!("evaluate({kind}): pipeline failed: {e}");
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
    let aggregation = match AggregationKind::from_kvs(kvs) {
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

#[allow(clippy::too_many_arguments)]
fn run_capture_baseline_with<E: Embed, R: Rerank>(
    ctx: &EvalContext<E, R>,
    output_path: &Path,
    aggregation: AggregationKind,
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

    let global = build_global_metrics(&results, &queries);
    let per_category = build_per_category_metrics(&results, &queries);
    let (latency_p50_ms, latency_p95_ms) = compute_latency_percentiles(&results);
    let snapshot = BaselineSnapshot {
        schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
        kind: BaselineKind::Forward,
        captured_with: "eval_harness capture-baseline".to_owned(),
        timestamp: (ctx.timestamp)(),
        model_id: embed::ModelId::default().repo_id().to_owned(),
        model_revision: RURI_V3_310M_REVISION.to_owned(),
        mlx_rs_version: MLX_RS_VERSION.to_owned(),
        fixture_hash,
        aggregation: aggregation.name(),
        merge_config: merge_config.clone(),
        normalization: *normalization,
        global,
        per_category,
        latency_p50_ms,
        latency_p95_ms,
    };

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
        AggregationKind::Identity,
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
    let json = match fs::read_to_string(&baseline_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("verify-baseline: read failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let committed: BaselineSnapshot = match serde_json::from_str(&json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("verify-baseline: parse failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    if committed.schema_version != BASELINE_SCHEMA_VERSION {
        eprintln!(
            "verify-baseline: failed — committed schema_version {:?} does not match harness {:?}; \
             regenerate the baseline before verifying",
            committed.schema_version, BASELINE_SCHEMA_VERSION
        );
        return ExitCode::from(EXIT_INFRA);
    }
    if committed.kind != BaselineKind::Forward {
        eprintln!(
            "verify-baseline: failed — committed kind {:?} is not Forward; \
             did you pass reverse_baseline.json by mistake?",
            committed.kind
        );
        return ExitCode::from(EXIT_INFRA);
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
    let aggregation = match AggregationKind::from_name(&committed.aggregation) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let (corpus, queries) = match load_fixture_for_kind(&ctx.fixture_dir, "full") {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("verify-baseline: {msg}");
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
        &committed.merge_config,
        &committed.normalization,
        &config,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify-baseline: pipeline failed: {e}");
            return ExitCode::from(EXIT_INFRA);
        }
    };
    let current_global = build_global_metrics(&results, &queries);
    let by_name: HashMap<&str, &MetricResult> = current_global
        .iter()
        .map(|m| (m.name.as_str(), m))
        .collect();

    for committed_m in &committed.global {
        let Some(current_m) = by_name.get(committed_m.name.as_str()) else {
            eprintln!(
                "verify-baseline: failed — committed metric {} missing in current run",
                committed_m.name
            );
            return ExitCode::from(EXIT_REGRESSION);
        };
        let Some(spec) = MetricSpec::from_name(&committed_m.name) else {
            eprintln!(
                "verify-baseline: failed — committed metric name {:?} is not a known MetricSpec; \
                 baseline.json may have been produced by a newer harness version",
                committed_m.name
            );
            return ExitCode::from(EXIT_INFRA);
        };
        let diff = (committed_m.point_estimate - current_m.point_estimate).abs();
        let tol = spec.tolerance();
        if diff > tol {
            eprintln!(
                "verify-baseline: failed — {} drifted by {diff:.6} > {tol:.6} \
                 (committed {:.6} vs current {:.6})",
                committed_m.name, committed_m.point_estimate, current_m.point_estimate
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
/// sub-fixture from `known_answers.jsonl`.
fn load_fixture_for_kind(
    fixture_dir: &Path,
    kind: &str,
) -> Result<(Vec<EvalDocument>, Vec<EvalQuery>), String> {
    match kind {
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
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("compare-baselines: failed to read {path}: {e}");
                return ExitCode::from(EXIT_INFRA);
            }
        };
        let snapshot: BaselineSnapshot = match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("compare-baselines: failed to parse {path}: {e}");
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

#[cfg(test)]
mod tests {
    use super::*;

    // T-067-007: aggregation_kind_topk_average_roundtrips_through_name
    //
    // A non-default `topk_k=` passed at
    // capture time used to be silently downgraded to DEFAULT_TOPK_AVERAGE_K
    // by `from_name` at verify time. Encode `k` in the serialised name and
    // round-trip to keep the strategy bit-identical across capture/verify.
    #[test]
    fn aggregation_kind_topk_average_roundtrips_through_name() {
        for k in [1, 3, 5, 100] {
            let original = AggregationKind::TopKAverage(k);
            let name = original.name();
            assert_eq!(
                name,
                format!("topk-average:{k}"),
                "TopKAverage({k}) must encode k in name"
            );
            let parsed =
                AggregationKind::from_name(&name).expect("encoded topk-average must parse back");
            assert_eq!(
                parsed, original,
                "round-trip lost k for TopKAverage({k}): {parsed:?}"
            );
        }
    }

    // T-067-008: aggregation_kind_legacy_topk_average_falls_back_to_default
    //
    // Pre-fix baselines wrote bare "topk-average" without `k`. Verify-baseline
    // must continue to parse those, falling back to DEFAULT_TOPK_AVERAGE_K so
    // operators with committed snapshots aren't forced to recapture.
    #[test]
    fn aggregation_kind_legacy_topk_average_falls_back_to_default() {
        let parsed = AggregationKind::from_name("topk-average")
            .expect("legacy bare topk-average must still parse");
        assert_eq!(parsed, AggregationKind::TopKAverage(DEFAULT_TOPK_AVERAGE_K));
    }

    // T-067-009: aggregation_kind_simple_variants_roundtrip
    //
    // Identity / MaxChunk / Dedupe carry no `k` payload, but they share the
    // round-trip contract — `name()` → `from_name()` must yield the original
    // variant.
    #[test]
    fn aggregation_kind_simple_variants_roundtrip() {
        for original in [
            AggregationKind::Identity,
            AggregationKind::MaxChunk,
            AggregationKind::Dedupe,
        ] {
            let name = original.name();
            let parsed =
                AggregationKind::from_name(&name).expect("simple variant name must round-trip");
            assert_eq!(parsed, original, "round-trip mismatch for {original:?}");
        }
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
}
