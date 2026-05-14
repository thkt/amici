//! Atomic JSON write helper shared by [`crate::eval::baseline`] and
//! [`crate::eval::annotation`].
//!
//! Generic over `T: Serialize` so [`BaselineSnapshot`] and [`Session`]
//! reach a single atomic-write code path. `serde_json` failures are
//! funneled into [`io::ErrorKind::Other`] so callers' typed error
//! enums (`BaselineError`, `AnnotationError`) only need a single
//! `Io(#[from] io::Error)` variant for both serialisation and
//! filesystem failures.
//!
//! [`BaselineSnapshot`]: crate::eval::baseline::BaselineSnapshot
//! [`Session`]: crate::eval::annotation::Session

use std::io;
use std::path::Path;

use serde::Serialize;

use crate::eval::baseline::atomic_write;

/// Serialise `value` as pretty JSON terminated by a newline and
/// atomically replace the file at `path`.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] when the temp file cannot be
/// created, written, fsynced, or renamed into place. `serde_json`
/// encoding failures are wrapped via [`io::Error::other`] so callers
/// match a single variant for both serialisation and filesystem
/// faults.
pub fn write_json<T: Serialize>(value: &T, path: &Path) -> io::Result<()> {
    let mut json = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    json.push('\n');
    atomic_write(path, json.as_bytes())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use rurico::retrieval::HybridSearchConfig;
    use rurico::storage::pre_phase_5_disabled;
    use tempfile::tempdir;

    use super::*;
    use crate::eval::baseline::{
        AggregationKind, BASELINE_SCHEMA_VERSION, BaselineKind, BaselineSnapshot,
    };

    /// Build a minimal [`BaselineSnapshot`] with deterministic literal
    /// stub values. Only the round-trip shape is exercised, so each
    /// field carries a placeholder rather than a fixture-derived value.
    fn stub_snapshot() -> BaselineSnapshot {
        BaselineSnapshot {
            schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
            kind: BaselineKind::Forward,
            captured_with: "io::tests".to_owned(),
            timestamp: "epoch:0".to_owned(),
            model_id: "test-model".to_owned(),
            model_revision: "test-rev".to_owned(),
            mlx_rs_version: "0.0".to_owned(),
            fixture_hash: "fnv1a64:0".to_owned(),
            aggregation: AggregationKind::Identity,
            merge_config: HybridSearchConfig::default(),
            normalization: pre_phase_5_disabled(),
            global: Vec::new(),
            per_category: BTreeMap::new(),
            latency_p50_ms: 0.0,
            latency_p95_ms: 0.0,
        }
    }

    // T-001: write_json_round_trips_baseline_snapshot
    // FR-001: write_json<T: Serialize> writes pretty JSON via the
    //         shared atomic-write path. A BaselineSnapshot routed
    //         through write_json must deserialise back to a JSON-value-
    //         equal Value. Comparison goes through serde_json::Value so
    //         the assertion does not depend on HashMap iteration order
    //         inside HybridSearchConfig.source_weights (BaselineSnapshot
    //         does not derive PartialEq, and Phase 1 keeps the derive
    //         set unchanged).
    #[test]
    fn write_json_round_trips_baseline_snapshot() {
        let dir = tempdir().expect("tempdir for write_json round-trip");
        let path = dir.path().join("snapshot.json");
        let snapshot = stub_snapshot();

        write_json(&snapshot, &path).expect("write_json must succeed");

        let body = fs::read_to_string(&path).expect("read snapshot.json");
        let parsed: BaselineSnapshot =
            serde_json::from_str(&body).expect("deserialise snapshot.json");
        let original_value = serde_json::to_value(&snapshot).expect("encode original as Value");
        let parsed_value = serde_json::to_value(&parsed).expect("encode parsed as Value");
        assert_eq!(
            parsed_value, original_value,
            "round-trip via write_json must preserve JSON value equality"
        );
    }
}
