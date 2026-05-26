//! Baseline snapshot serialisation (FR-015..FR-016).
//!
//! Persists per-baseline metric results, CI bounds, latency, and provenance
//! (model id / revision / mlx-rs version / fixture hash) so a future run can
//! verify against the committed JSON via `verify-baseline` mode.
//!
//! [`MetricResult`] reuses the [`serde::Serialize`] / [`serde::Deserialize`]
//! derive added on the same type in `metrics.rs`; this module's
//! [`BaselineSnapshot`] composes it directly so its own derive is sufficient.
//!
//! # serde default discipline (ADR-0007)
//!
//! 新フィールドを [`BaselineSnapshot`] に追加するとき、`#[serde(default = "fn_name")]`
//! を必ず付与し、その default 関数は **そのフィールドが存在しなかった時点での実質挙動
//! (pre-existing behavior)** を返す。runtime default (e.g. `Foo::default()`) を割り
//! 当てる場合は、それが historical baseline の実質挙動と一致する根拠を PR 説明または
//! コード上で示すか、[`BASELINE_SCHEMA_VERSION`] を bump する。
//!
//! 既存 3 フィールドは本規律に従う:
//!
//! - `aggregation`: [`AggregationKind::Identity`] を返す ([`default_aggregation_kind`])
//! - `merge_config`: `HybridSearchConfig::default()` 同等 (pre-merge-config 時代と一致、`#[serde(default)]`)
//! - `normalization`: [`pre_phase_5_disabled`] (all OFF、runtime `QueryNormalizationConfig::default` の all ON とは **異なる** historical 側採用)
//!
//! 規律は historical baseline (3 フィールド欠落) を `tests/fixtures/eval/historical/`
//! 配下に pin した unit test (`historical_baseline_resolves_serde_defaults_to_pre_existing_behavior`)
//! で、default 経由の deserialize が pre-existing behavior と一致することを CI で gate する。
//!
//! 規律の根拠と新フィールド追加手順は ADR-0007
//! (`docs/decisions/0007-pin-baselinesnapshot-serde-defaults-to-pre-existing-behavior.md`)
//! 参照。

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use rurico::retrieval::HybridSearchConfig;
use rurico::storage::{QueryNormalizationConfig, pre_phase_5_disabled};
use serde::{Deserialize, Serialize, de};

use crate::eval::metrics::MetricResult;

/// Half-width threshold above which a per-category metric is flagged
/// `uninformative` (FR-016 / BR-002).
pub const UNINFORMATIVE_HALF_WIDTH: f64 = 0.10;

/// Current schema version stamped into every emitted baseline file.
///
/// Bump on a breaking change (renamed/removed fields, semantic shift) so
/// downstream consumers can refuse silently-incompatible files.
///
/// Version log:
/// - `1.0`: initial baseline schema.
/// - `1.1`: chunk-level retrieval. The Stage 2 fusion key widened to
///   `(doc_id, chunk_id)`, splitting FTS hits (`chunk_id=None`) from
///   Vector hits (`chunk_id=Some(_)`) for the same parent doc. Pre-1.1
///   baseline.json files would surface as confusing per-metric drift
///   under the new fusion; the version bump turns that into a clean
///   "regenerate the baseline before verifying" exit instead.
/// - `1.2`: Oracle baseline kind (Issue #52). [`BaselineKind`] gained an
///   `Oracle` variant; older harnesses cannot deserialise an
///   `oracle_baseline.json` (variant unknown to their enum), but
///   Forward and Reverse files round-trip unchanged. The bump lets
///   `oracle-gap` refuse to compare snapshots produced under different
///   semantic versions.
/// - `1.3`: First-search replay baseline kind (Issue #62).
///   [`BaselineKind`] gained a `FirstSearchReplay` variant; older
///   harnesses cannot deserialise a
///   `first_search_replay_baseline.json` (variant unknown to their
///   enum), but Forward, Reverse, and Oracle files round-trip unchanged.
///   The bump lets `verify-baseline` refuse to compare a Forward
///   baseline against a `replay-first-search` capture by mistake.
pub const BASELINE_SCHEMA_VERSION: &str = "1.3";

/// Discriminator distinguishing forward (`capture-baseline`), reverse
/// (`capture-reverse-baseline`), oracle (`capture-oracle`), and
/// first-search replay (`replay-first-search`) baseline files.
///
/// All four files share the [`BASELINE_SCHEMA_VERSION`] envelope; consumers
/// read `kind` first to pick the right body shape rather than inferring
/// from the presence of fields. `Forward`, `Oracle`, and `FirstSearchReplay`
/// share the [`BaselineSnapshot`] body shape — the discriminator
/// distinguishes production-ranker, retrieval-ceiling, and Stage 1+2-only
/// captures so `oracle-gap` (Issue #52) and `verify-baseline` (Issue #62)
/// can refuse to compare snapshots produced under different semantic
/// modes by mistake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineKind {
    /// Forward baseline produced by `capture-baseline`. Body matches
    /// [`BaselineSnapshot`].
    Forward,
    /// Reverse-ranker lower-bound baseline produced by
    /// `capture-reverse-baseline`. Body shape lives in the binary
    /// (`observed_lower_bound`, `k`, `captured_with`).
    Reverse,
    /// Oracle retrieval-ceiling baseline produced by `capture-oracle`
    /// (Issue #52). Body matches [`BaselineSnapshot`]; the difference
    /// from `Forward` is that Stage 2 forced every relevance_map doc to
    /// rank 0 so Stage 3 + Stage 4 ran against an idealised retrieval.
    Oracle,
    /// First-search replay baseline produced by `replay-first-search`
    /// (Issue #62). Body matches [`BaselineSnapshot`]; Stage 1+2 produce
    /// the merged hits, [`MaxChunkAggregator`] performs parent-level
    /// rollup, and Stage 4 (rerank) is skipped. Records the agentic-search
    /// "first-result quality" indicator (ADR-0003) so downstream consumers
    /// can isolate ranking-strategy sensitivity from aggregator / reranker
    /// effects.
    ///
    /// [`MaxChunkAggregator`]: rurico::retrieval::MaxChunkAggregator
    FirstSearchReplay,
}

/// Stage 3 aggregation strategy recorded on a [`BaselineSnapshot`].
///
/// Closed wire-format contract — every reader must handle every variant
/// (no `#[non_exhaustive]`) so a baseline produced by a newer harness
/// fails to deserialise rather than silently dropping an unknown strategy.
///
/// The mapping to JSON strings is intentionally **not** `serde(rename_all)`-driven:
/// `TopKAverage(k)` carries a parameter that has to round-trip through the
/// wire form as `"topk-average:k"`. See the [`Serialize`] / [`Deserialize`]
/// impls below for the exact mapping.
///
/// | Variant         | JSON wire form     | Notes                                  |
/// | --------------- | ------------------ | -------------------------------------- |
/// | `Identity`      | `"identity"`       | Pre-aggregation default                |
/// | `MaxChunk`      | `"max-chunk"`      | Parent rollup, max child score         |
/// | `Dedupe`        | `"dedupe"`         | Drop sibling chunks of the same parent |
/// | `TopKAverage(k)`| `"topk-average:k"` | Average top-`k` chunks per parent      |
/// | `NotApplicable` | `"none"`           | BR-003 marker for FirstSearchReplay    |
///
/// Bare `"topk-average"` (no `:k` suffix) is **rejected** during deserialise;
/// pre-fix baselines that encoded the strategy without `k` would otherwise
/// silently verify against a different `k` than they were captured at. No
/// committed fixture uses the bare form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationKind {
    Identity,
    MaxChunk,
    Dedupe,
    TopKAverage(usize),
    /// BR-003 marker for [`BaselineKind::FirstSearchReplay`] — Stage 3 fixes
    /// `MaxChunkAggregator` outside the parameterised set, so no aggregator
    /// kind applies.
    NotApplicable,
}

impl fmt::Display for AggregationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => f.write_str("identity"),
            Self::MaxChunk => f.write_str("max-chunk"),
            Self::Dedupe => f.write_str("dedupe"),
            Self::TopKAverage(k) => write!(f, "topk-average:{k}"),
            Self::NotApplicable => f.write_str("none"),
        }
    }
}

impl Serialize for AggregationKind {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for AggregationKind {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        match s.as_str() {
            "identity" => Ok(Self::Identity),
            "max-chunk" => Ok(Self::MaxChunk),
            "dedupe" => Ok(Self::Dedupe),
            "none" => Ok(Self::NotApplicable),
            "topk-average" => Err(de::Error::custom(
                "legacy bare 'topk-average' aggregation marker without ':k' suffix \
                 — recapture the baseline so the k parameter is preserved",
            )),
            other => match other.strip_prefix("topk-average:") {
                Some(k_str) => k_str.parse::<usize>().map(Self::TopKAverage).map_err(|e| {
                    de::Error::custom(format!(
                        "topk-average:k parse error in baseline aggregation: {e}"
                    ))
                }),
                None => Err(de::Error::custom(format!(
                    "unknown aggregation marker {other:?}"
                ))),
            },
        }
    }
}

/// Frozen baseline produced by `eval_harness capture-baseline` and verified
/// later by `eval_harness verify-baseline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    /// Schema version envelope. See [`BASELINE_SCHEMA_VERSION`].
    pub schema_version: String,
    /// File-type discriminator. Forward baselines fix this to
    /// [`BaselineKind::Forward`]; reverse baselines live in a separate file
    /// keyed `Reverse`.
    pub kind: BaselineKind,
    /// Subcommand that produced this file (e.g.
    /// `"eval_harness capture-baseline"`). Mirrors the `captured_with` field
    /// already carried by `reverse_baseline.json` so both artifacts share a
    /// symmetric provenance schema.
    pub captured_with: String,
    /// Capture-time label in `epoch:N` form (Unix seconds since UNIX_EPOCH).
    /// Avoids pulling `chrono` in just for an ISO-8601 timestamp.
    pub timestamp: String,
    /// Hugging Face repo id of the embed model used.
    pub model_id: String,
    /// Pinned revision (commit hash) of the embed model.
    pub model_revision: String,
    /// `mlx-rs` semver string at capture time.
    pub mlx_rs_version: String,
    /// Content hash over `documents.jsonl + queries.jsonl + known_answers.jsonl`.
    pub fixture_hash: String,
    /// Stage 3 aggregation strategy used for capture.
    ///
    /// Pre-aggregation baselines lack this field; `serde(default)` resolves
    /// it to [`AggregationKind::Identity`] so existing committed baselines
    /// round-trip. `verify-baseline` reads it back to dispatch the same
    /// aggregator.
    #[serde(default = "default_aggregation_kind")]
    pub aggregation: AggregationKind,
    /// Stage 2 hybrid scoring config used for capture.
    ///
    /// Pre-merge-config baselines lack this field; `serde(default)`
    /// resolves it to [`HybridSearchConfig::default`] (`rrf_k=60`,
    /// `fts/vector weights=1.0`) so existing committed baselines
    /// round-trip bit-equal.
    #[serde(default)]
    pub merge_config: HybridSearchConfig,
    /// Stage 1 query normalization config used for capture.
    ///
    /// Pre-normalization baselines lack this field; the serde-default
    /// points at [`pre_phase_5_disabled`] (all OFF), **not** at runtime
    /// [`QueryNormalizationConfig::default`] (all ON), so historical files
    /// replay under the behaviour they were captured with.
    #[serde(default = "pre_phase_5_disabled")]
    pub normalization: QueryNormalizationConfig,
    /// Global metric results (regression gate per BR-001).
    pub global: Vec<MetricResult>,
    /// Per-category metric breakdown for exploratory inspection.
    pub per_category: BTreeMap<String, Vec<MetricResult>>,
    /// Median per-query latency across the fixture in milliseconds.
    pub latency_p50_ms: f64,
    /// 95th percentile per-query latency in milliseconds.
    pub latency_p95_ms: f64,
}

fn default_aggregation_kind() -> AggregationKind {
    AggregationKind::Identity
}

/// Errors surfaced when writing a [`BaselineSnapshot`].
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum BaselineError {
    /// Filesystem failure while creating the output file.
    #[error("baseline io error: {0}")]
    Io(#[from] io::Error),
    /// JSON serialisation failure.
    #[error("baseline serialise error: {0}")]
    Serialise(#[from] serde_json::Error),
}

/// Build a [`MetricResult`] and set its `uninformative` flag from the CI
/// half-width (FR-016 / BR-002).
///
/// `uninformative` is `true` when `(ci_upper - ci_lower) / 2 > 0.10`, i.e.
/// strictly above [`UNINFORMATIVE_HALF_WIDTH`]. Equality with the threshold
/// stays informative.
#[must_use]
pub fn build_metric_result(
    name: String,
    k: usize,
    point: f64,
    ci_lower: f64,
    ci_upper: f64,
) -> MetricResult {
    let half_width = (ci_upper - ci_lower) / 2.0;
    let uninformative = half_width > UNINFORMATIVE_HALF_WIDTH;
    MetricResult {
        name,
        k,
        point_estimate: point,
        ci_lower,
        ci_upper,
        uninformative,
    }
}

/// Atomically write `bytes` to `path` via temp-file + `fs::rename`.
///
/// Writes to a sibling `.{file_name}.tmp` first, fsyncs, then renames over the
/// destination. A SIGTERM, panic, or disk-full mid-write cannot leave the
/// destination in a partial state — either the prior contents survive or the
/// new contents replace them in one atomic POSIX rename.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] when the temp file cannot be created,
/// written, fsynced, or renamed into place. Surfaces [`io::ErrorKind::InvalidInput`]
/// when `path` has no file name component (caller bug).
pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("atomic_write: path has no file name: {}", path.display()),
        )
    })?;
    let tmp_path = parent.join(format!(".{}.tmp", file_name.to_string_lossy()));
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&tmp_path, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp_path);
    })
}

/// Serialise `snapshot` as pretty JSON to `path` (atomic write).
///
/// Delegates to [`super::io::write_json`] so the `BaselineSnapshot`
/// and `Session` writers share a single atomic-write code path.
///
/// # Errors
///
/// Returns [`BaselineError::Io`] when the destination cannot be
/// created or when JSON encoding fails — `serde_json` failures are
/// wrapped as [`io::ErrorKind::Other`] inside [`super::io::write_json`]
/// before they reach this layer. [`BaselineError::Serialise`] is
/// retained for forward-compatible callers that wrap their own
/// `serde_json::Error` directly.
pub fn write_json(snapshot: &BaselineSnapshot, path: &Path) -> Result<(), BaselineError> {
    super::io::write_json(snapshot, path)?;
    Ok(())
}

/// Render `snapshot` as a human-readable markdown report at `path` (atomic write).
///
/// # Errors
///
/// Returns [`BaselineError::Io`] when the destination cannot be created.
pub fn write_markdown(snapshot: &BaselineSnapshot, path: &Path) -> Result<(), BaselineError> {
    let mut output = String::new();
    output.push_str("# Baseline Snapshot\n\n");
    output.push_str(&format!("- Captured: {}\n", snapshot.timestamp));
    output.push_str(&format!(
        "- Model: {} @ {}\n",
        snapshot.model_id, snapshot.model_revision
    ));
    output.push_str(&format!("- mlx-rs: {}\n", snapshot.mlx_rs_version));
    output.push_str(&format!("- Fixture hash: {}\n", snapshot.fixture_hash));
    output.push_str(&format!(
        "- Latency p50/p95: {:.2}/{:.2} ms\n",
        snapshot.latency_p50_ms, snapshot.latency_p95_ms
    ));
    output.push_str("\n## Global metrics\n\n");
    write_metric_lines(&mut output, &snapshot.global);
    output.push_str("\n## Per-category metrics\n");
    for (category, metrics) in &snapshot.per_category {
        output.push_str(&format!("\n### {category}\n\n"));
        write_metric_lines(&mut output, metrics);
    }
    atomic_write(path, output.as_bytes())?;
    Ok(())
}

fn write_metric_lines(buf: &mut String, metrics: &[MetricResult]) {
    for metric in metrics {
        let flag = if metric.uninformative {
            ", uninformative"
        } else {
            ""
        };
        buf.push_str(&format!(
            "- {} @{}: {:.4} (CI: {:.4}..{:.4}{flag})\n",
            metric.name, metric.k, metric.point_estimate, metric.ci_lower, metric.ci_upper
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-018: build_metric_result_wide_ci_flags_uninformative
    // FR-016 / BR-002: half-width = (0.65 - 0.35) / 2 = 0.15 > 0.10 →
    //                  MetricResult.uninformative == true.
    #[test]
    fn build_metric_result_wide_ci_flags_uninformative() {
        let result = build_metric_result("recall@5".to_owned(), 5, 0.5, 0.35, 0.65);

        assert!(
            result.uninformative,
            "FR-016: half-width 0.15 > threshold 0.10 → uninformative must be true, \
             got result = {result:?}"
        );
    }

    // T-018b: build_metric_result_narrow_ci_stays_informative
    // FR-016 / BR-002: half-width = (0.55 - 0.45) / 2 = 0.05 < 0.10 →
    //                  MetricResult.uninformative == false (negative case).
    #[test]
    fn build_metric_result_narrow_ci_stays_informative() {
        let result = build_metric_result("recall@5".to_owned(), 5, 0.5, 0.45, 0.55);

        assert!(
            !result.uninformative,
            "FR-016: half-width 0.05 < threshold 0.10 → uninformative must be false, \
             got result = {result:?}"
        );
    }

    // T-020: baseline_snapshot_round_trips_with_schema_version_and_kind
    // Pins the schema_version + kind envelope so a future migration that
    // drops or renames either field fails this round-trip explicitly.
    #[test]
    fn baseline_snapshot_round_trips_with_schema_version_and_kind() {
        let snap = BaselineSnapshot {
            schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
            kind: BaselineKind::Forward,
            captured_with: "test".to_owned(),
            timestamp: "epoch:42".to_owned(),
            model_id: "test/model".to_owned(),
            model_revision: "rev".to_owned(),
            mlx_rs_version: "0.0.0".to_owned(),
            fixture_hash: "fnv1a64:0".to_owned(),
            aggregation: default_aggregation_kind(),
            merge_config: HybridSearchConfig::default(),
            normalization: pre_phase_5_disabled(),
            global: vec![],
            per_category: BTreeMap::new(),
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
        };
        let json = serde_json::to_string(&snap).expect("serialise");
        let parsed: BaselineSnapshot = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed.schema_version, BASELINE_SCHEMA_VERSION);
        assert_eq!(parsed.kind, BaselineKind::Forward);
        assert_eq!(parsed.timestamp, "epoch:42");
    }

    // T-052-301: baseline_kind_oracle_serialises_as_snake_case
    //
    // Issue #52: BaselineKind::Oracle round-trips through serde_json as
    // the snake_case label "oracle". Pinning the wire format guards
    // against a future kind variant accidentally changing the discriminator
    // name (which would invalidate every committed oracle_baseline.json).
    #[test]
    fn baseline_kind_oracle_serialises_as_snake_case() {
        let json = serde_json::to_string(&BaselineKind::Oracle).expect("serialise");
        assert_eq!(
            json, "\"oracle\"",
            "Oracle variant must serialise to \"oracle\""
        );
        let parsed: BaselineKind = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed, BaselineKind::Oracle);
    }

    // T-052-302: baseline_snapshot_round_trips_with_oracle_kind
    //
    // The full BaselineSnapshot envelope round-trips with kind=Oracle,
    // proving Forward and Oracle share a body shape and only differ in
    // the discriminator. Required for `capture-oracle` to reuse
    // [`write_json`] without a separate serialisation path.
    #[test]
    fn baseline_snapshot_round_trips_with_oracle_kind() {
        let snap = BaselineSnapshot {
            schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
            kind: BaselineKind::Oracle,
            captured_with: "eval_harness capture-oracle".to_owned(),
            timestamp: "epoch:42".to_owned(),
            model_id: "test/model".to_owned(),
            model_revision: "rev".to_owned(),
            mlx_rs_version: "0.0.0".to_owned(),
            fixture_hash: "fnv1a64:0".to_owned(),
            aggregation: default_aggregation_kind(),
            merge_config: HybridSearchConfig::default(),
            normalization: pre_phase_5_disabled(),
            global: vec![],
            per_category: BTreeMap::new(),
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
        };
        let json = serde_json::to_string(&snap).expect("serialise");
        let parsed: BaselineSnapshot = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed.kind, BaselineKind::Oracle);
        assert_eq!(parsed.captured_with, "eval_harness capture-oracle");
    }

    // T-021: committed_baseline_json_deserialises_under_new_schema
    // Default-lane guard: MLX-gated T-019 won't catch a schema migration
    // that drops the committed fixture's parseability. This test runs in
    // the default `cargo test` lane (no feature flag, no #[ignore]) so a
    // stale fixture fails CI immediately.
    #[test]
    fn committed_baseline_json_deserialises_under_new_schema() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval/baseline.json");
        let text =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let parsed: BaselineSnapshot = serde_json::from_str(&text).unwrap_or_else(|e| {
            panic!(
                "committed baseline.json must deserialise under the current schema ({e}); \
                 update the fixture when bumping {BASELINE_SCHEMA_VERSION:?}. content head: {}",
                &text.chars().take(200).collect::<String>()
            )
        });
        assert_eq!(
            parsed.schema_version, BASELINE_SCHEMA_VERSION,
            "schema_version mismatch — fixture must match {BASELINE_SCHEMA_VERSION:?}"
        );
        assert_eq!(
            parsed.kind,
            BaselineKind::Forward,
            "committed baseline.json must declare kind=forward"
        );
    }

    // T-062-011 / T-062-015: baseline_kind_first_search_replay_serialises_as_snake_case
    //
    // FR-012: BaselineKind::FirstSearchReplay round-trips through serde_json
    // as the snake_case label "first_search_replay". Pinning the wire format
    // guards against a future variant rename silently breaking committed
    // first_search_replay_baseline.json files. A serde-attribute regression
    // (e.g. dropping `rename_all = "snake_case"`) would break both serialize
    // and deserialize together, so wire-format pin and enum-level round-trip
    // collapse into one test (mirrors T-052-301 for Oracle).
    #[test]
    fn baseline_kind_first_search_replay_serialises_as_snake_case() {
        let json = serde_json::to_string(&BaselineKind::FirstSearchReplay).expect("serialise");
        assert_eq!(
            json, "\"first_search_replay\"",
            "FR-012: FirstSearchReplay variant must serialise to \"first_search_replay\""
        );
        let parsed: BaselineKind = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed, BaselineKind::FirstSearchReplay);
    }

    // T-062-012: baseline_snapshot_round_trips_with_first_search_replay_kind
    //
    // FR-013: BaselineSnapshot envelope round-trips with
    // `kind=FirstSearchReplay`, proving the new variant shares Forward /
    // Oracle's body shape and `replay-first-search` can reuse `write_json`
    // without forking the serialisation path. `aggregation="none"` is the
    // BR-003 marker for "Stage 3 ran MaxChunkAggregator only".
    #[test]
    fn baseline_snapshot_round_trips_with_first_search_replay_kind() {
        let snap = BaselineSnapshot {
            schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
            kind: BaselineKind::FirstSearchReplay,
            captured_with: "eval_harness replay-first-search".to_owned(),
            timestamp: "epoch:42".to_owned(),
            model_id: "test/model".to_owned(),
            model_revision: "rev".to_owned(),
            mlx_rs_version: "0.0.0".to_owned(),
            fixture_hash: "fnv1a64:0".to_owned(),
            aggregation: AggregationKind::NotApplicable,
            merge_config: HybridSearchConfig::default(),
            normalization: pre_phase_5_disabled(),
            global: vec![],
            per_category: BTreeMap::new(),
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
        };
        let json = serde_json::to_string(&snap).expect("serialise");
        let parsed: BaselineSnapshot = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(parsed.kind, BaselineKind::FirstSearchReplay);
        assert_eq!(parsed.captured_with, "eval_harness replay-first-search");
        assert_eq!(
            parsed.aggregation,
            AggregationKind::NotApplicable,
            "BR-003: aggregation marker is NotApplicable for FirstSearchReplay"
        );
    }

    // T-062-014: committed_baselines_deserialize_under_schema_1_3
    //
    // FR-014 / AC-6: every committed baseline file (forward / oracle /
    // first-search replay) deserialises cleanly under schema 1.3. Pinning
    // the three-file invariant in a single test surfaces a stale fixture
    // (e.g. one regenerated under a different schema) in CI before
    // verify-baseline hits it at runtime.
    #[test]
    fn committed_baselines_deserialize_under_schema_1_3() {
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval");
        for (name, expected_kind) in [
            ("baseline.json", BaselineKind::Forward),
            ("oracle_baseline.json", BaselineKind::Oracle),
            (
                "first_search_replay_baseline.json",
                BaselineKind::FirstSearchReplay,
            ),
        ] {
            let path = fixture_dir.join(name);
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let parsed: BaselineSnapshot = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{name} must deserialise under schema 1.3 ({e})"));
            assert_eq!(parsed.schema_version, "1.3", "{name} schema must be 1.3");
            assert_eq!(
                parsed.kind, expected_kind,
                "{name} kind must be {expected_kind:?}"
            );
        }
    }

    // T-081-002: aggregation_kind_roundtrips_committed_fixtures
    //
    // Issue #81 Code fix #10: lock in that the typed [`AggregationKind`] enum
    // deserialises every committed baseline fixture to the expected variant
    // **and** re-serialises to a byte-equivalent JSON literal. Catches the
    // most likely regression on this PR — a serde wire-form change that
    // silently rewrites the meaning of historical baselines.
    #[test]
    fn aggregation_kind_roundtrips_committed_fixtures() {
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval");
        for (name, expected_kind) in [
            ("baseline.json", AggregationKind::Identity),
            ("oracle_baseline.json", AggregationKind::Identity),
            (
                "first_search_replay_baseline.json",
                AggregationKind::NotApplicable,
            ),
        ] {
            let path = fixture_dir.join(name);
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let snap: BaselineSnapshot = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{name} must deserialise: {e}"));
            assert_eq!(
                snap.aggregation, expected_kind,
                "{name}: aggregation must deserialise to {expected_kind:?}"
            );
            let reserialized = serde_json::to_value(snap.aggregation)
                .unwrap_or_else(|e| panic!("{name}: serialise aggregation: {e}"));
            let original: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{name}: parse as Value: {e}"));
            assert_eq!(
                reserialized, original["aggregation"],
                "{name}: aggregation field must round-trip byte-equivalent JSON"
            );
        }
    }

    // T-022: atomic_write_replaces_destination_on_each_call
    // Verifies the temp-file + rename path overwrites cleanly across
    // consecutive writes (no leftover .tmp files) and that the final
    // destination matches the most recent write.
    #[test]
    fn atomic_write_replaces_destination_on_each_call() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("artifact.json");
        atomic_write(&path, b"first\n").expect("first write");
        atomic_write(&path, b"second\n").expect("second write");
        let content = fs::read_to_string(&path).expect("read");
        assert_eq!(content, "second\n", "atomic_write must replace contents");
        assert!(
            !dir.path().join(".artifact.json.tmp").exists(),
            "temp file must be renamed away"
        );
    }

    // ADR-0007 #9: historical_baseline_resolves_serde_defaults_to_pre_existing_behavior
    //
    // §Decision Outcome 4 + §Implementation Guidelines step 4: field-less
    // historical baseline (`aggregation` / `merge_config` / `normalization` 3
    // フィールドすべて欠落) を deserialize し、各 serde default が
    // pre-existing behavior と等価な値に解決されることを pin。本 test が fail =
    // 既存 default 関数のいずれかが pre-existing behavior と乖離した = 規律違反、
    // ADR-0007 の bump policy 前提が崩れる。
    #[test]
    fn historical_baseline_resolves_serde_defaults_to_pre_existing_behavior() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/eval/historical/pre_serde_defaults.json");
        let text =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let parsed: BaselineSnapshot = serde_json::from_str(&text).unwrap_or_else(|e| {
            panic!(
                "historical fixture must deserialise via serde defaults (3 missing \
                 fields each resolved by their `#[serde(default = ...)]`): {e}"
            )
        });

        assert_eq!(
            parsed.aggregation,
            AggregationKind::Identity,
            "ADR-0007: pre-aggregation historical baseline must resolve `aggregation` \
             via `default_aggregation_kind` to `AggregationKind::Identity` \
             (== pre-existing behavior, NOT a runtime default that drifted)"
        );
        assert_eq!(
            parsed.merge_config,
            HybridSearchConfig::default(),
            "ADR-0007: pre-merge-config historical baseline must resolve `merge_config` \
             via `#[serde(default)]` to `HybridSearchConfig::default()` \
             (rrf_k=60, fts/vector weights=1.0 — pre-merge-config 時代の実質挙動)"
        );
        assert_eq!(
            parsed.normalization,
            pre_phase_5_disabled(),
            "ADR-0007: pre-normalization historical baseline must resolve `normalization` \
             via `pre_phase_5_disabled` to all-OFF \
             (NOT runtime `QueryNormalizationConfig::default` which is all-ON)"
        );
    }
}
