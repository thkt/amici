//! Baseline snapshot serialisation (FR-015..FR-016).
//!
//! Persists per-baseline metric results, CI bounds, latency, and provenance
//! (model id / revision / mlx-rs version / fixture hash) so a future run can
//! verify against the committed JSON via `verify-baseline` mode.
//!
//! [`MetricResult`] reuses the [`serde::Serialize`] / [`serde::Deserialize`]
//! derive added on the same type in `metrics.rs`; this module's
//! [`BaselineSnapshot`] composes it directly so its own derive is sufficient.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use rurico::retrieval::HybridSearchConfig;
use rurico::storage::{QueryNormalizationConfig, pre_phase_5_disabled};
use serde::{Deserialize, Serialize};

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
pub const BASELINE_SCHEMA_VERSION: &str = "1.1";

/// Discriminator distinguishing forward (`capture-baseline`) from reverse
/// (`capture-reverse-baseline`) baseline files.
///
/// Both files share the [`BASELINE_SCHEMA_VERSION`] envelope; consumers read
/// `kind` first to pick the right body shape rather than inferring from the
/// presence of fields.
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
    /// it to `"identity"` so existing committed baselines round-trip.
    /// `verify-baseline` reads it back to dispatch the same aggregator.
    #[serde(default = "default_aggregation_kind")]
    pub aggregation: String,
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

fn default_aggregation_kind() -> String {
    "identity".to_owned()
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
/// Uses [`atomic_write`] so a partial write never corrupts a committed
/// baseline file in the git tree.
///
/// # Errors
///
/// Returns [`BaselineError::Io`] when the destination cannot be created and
/// [`BaselineError::Serialise`] when JSON encoding fails.
pub fn write_json(snapshot: &BaselineSnapshot, path: &Path) -> Result<(), BaselineError> {
    let mut json = serde_json::to_string_pretty(snapshot)?;
    json.push('\n');
    atomic_write(path, json.as_bytes())?;
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
}
