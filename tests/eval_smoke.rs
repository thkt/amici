//! Subprocess smoke tests for the eval harness binary (ADR 0002).
//!
//! All tests spawn `eval_harness` as a child process so MLX FFI failures
//! in the binary do not kill the test runner.
//!
//! Tests are double-gated:
//! - `#[cfg(feature = "eval-harness")]` keeps them out of the default
//!   `cargo test` build.
//! - `#[ignore]` forces explicit opt-in via `-- --ignored` so MLX-required
//!   tests stay out of `cargo test --features eval-harness` by default.

#![cfg(feature = "eval-harness")]

use std::fs;
use std::process::{Command, Output};

use amici::eval::baseline::BaselineSnapshot;
use rurico::sandbox::SEATBELT_SKIP_EXIT;

/// Path to the committed reverse-ranker baseline. T-014 reads
/// `observed_lower_bound` from this file.
const REVERSE_BASELINE_PATH: &str = "tests/fixtures/eval/reverse_baseline.json";

/// Path to the committed full baseline. T-019 verifies against this file.
const BASELINE_PATH: &str = "tests/fixtures/eval/baseline.json";

// T-013: t013_identity_fixture_perfect_metrics
// FR-011: identity_ranker fixture must yield nDCG@10 == 1.0 ∧ Recall@1 == 1.0.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn t013_identity_fixture_perfect_metrics() {
    let output = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args(["evaluate", "kind=identity"])
        .output()
        .expect("spawn eval_harness evaluate kind=identity");
    assert_smoke_success(&output);

    let recall_at_1 = parse_metric_field("T-013", &output, "recall_at_1");
    let ndcg_at_10 = parse_metric_field("T-013", &output, "ndcg_at_10");

    assert!(
        (recall_at_1 - 1.0).abs() < f64::EPSILON,
        "[T-013] FR-011 identity: recall_at_1 must be 1.0, got {recall_at_1}"
    );
    assert!(
        (ndcg_at_10 - 1.0).abs() < f64::EPSILON,
        "[T-013] FR-011 identity: ndcg_at_10 must be 1.0, got {ndcg_at_10}"
    );
}

// T-014: t014_reverse_fixture_below_lower_bound
// FR-012: reverse_ranker fixture observed nDCG@10 ≤ observed_lower_bound × 1.05,
//         where observed_lower_bound is committed in reverse_baseline.json.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX
fn t014_reverse_fixture_below_lower_bound() {
    let baseline_text = fs::read_to_string(REVERSE_BASELINE_PATH)
        .unwrap_or_else(|e| panic!("[T-014] read {REVERSE_BASELINE_PATH}: {e}"));
    let baseline_json: serde_json::Value = serde_json::from_str(&baseline_text)
        .unwrap_or_else(|e| panic!("[T-014] parse {REVERSE_BASELINE_PATH}: {e}"));
    let observed_lower_bound = baseline_json
        .get("observed_lower_bound")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| {
            panic!(
                "[T-014] {REVERSE_BASELINE_PATH} missing observed_lower_bound: \
                 {baseline_text}"
            )
        });

    let output = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args(["evaluate", "kind=reverse"])
        .output()
        .expect("spawn eval_harness evaluate kind=reverse");
    assert_smoke_success(&output);

    let ndcg_at_10 = parse_metric_field("T-014", &output, "ndcg_at_10");

    let upper = observed_lower_bound * 1.05;
    assert!(
        ndcg_at_10 <= upper,
        "[T-014] FR-012 reverse: ndcg_at_10 ({ndcg_at_10}) must be ≤ \
         observed_lower_bound × 1.05 ({upper}); committed lower bound = \
         {observed_lower_bound}"
    );
}

// T-015: t015_single_doc_fixture_perfect_metrics
// FR-013: single_doc fixture must yield Recall@1 == 1.0 ∧ MRR == 1.0.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn t015_single_doc_fixture_perfect_metrics() {
    let output = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args(["evaluate", "kind=single_doc"])
        .output()
        .expect("spawn eval_harness evaluate kind=single_doc");
    assert_smoke_success(&output);

    let recall_at_1 = parse_metric_field("T-015", &output, "recall_at_1");
    let mrr = parse_metric_field("T-015", &output, "mrr");

    assert!(
        (recall_at_1 - 1.0).abs() < f64::EPSILON,
        "[T-015] FR-013 single_doc: recall_at_1 must be 1.0, got {recall_at_1}"
    );
    assert!(
        (mrr - 1.0).abs() < f64::EPSILON,
        "[T-015] FR-013 single_doc: mrr must be 1.0, got {mrr}"
    );
}

// T-016: t016_shuffled_ndcg_below_baseline
// FR-014: shuffling the ranking before metric computation must reduce nDCG@10
//         below the un-shuffled baseline (mutation sanity test).
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn t016_shuffled_ndcg_below_baseline() {
    let baseline = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args(["evaluate", "kind=full"])
        .output()
        .expect("spawn eval_harness evaluate kind=full");
    assert_smoke_success(&baseline);
    let baseline_ndcg = parse_metric_field("T-016 baseline", &baseline, "ndcg_at_10");

    let shuffled = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args(["evaluate", "kind=shuffled"])
        .output()
        .expect("spawn eval_harness evaluate kind=shuffled");
    assert_smoke_success(&shuffled);
    let shuffled_ndcg = parse_metric_field("T-016 shuffled", &shuffled, "ndcg_at_10");

    assert!(
        shuffled_ndcg < baseline_ndcg,
        "[T-016] FR-014 shuffle mutation: shuffled_ndcg ({shuffled_ndcg}) must be \
         strictly less than baseline_ndcg ({baseline_ndcg})"
    );
}

// T-017: t017_capture_baseline_writes_required_fields
// FR-015: capture-baseline output=<path> must produce a BaselineSnapshot
//         containing model_id, fixture_hash, global, per_category, latency
//         p50/p95.
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX (Apple Silicon)
fn t017_capture_baseline_writes_required_fields() {
    let tempdir = tempfile::tempdir().expect("create tempdir for baseline output");
    let baseline_path = tempdir.path().join("baseline.json");

    let output = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args([
            "capture-baseline",
            &format!("output={}", baseline_path.display()),
        ])
        .output()
        .expect("spawn eval_harness capture-baseline");
    assert_smoke_success(&output);

    assert!(
        baseline_path.exists(),
        "[T-017] FR-015: capture-baseline must create {} (stderr: {})",
        baseline_path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = fs::read_to_string(&baseline_path)
        .unwrap_or_else(|e| panic!("[T-017] read {}: {e}", baseline_path.display()));
    let snapshot: BaselineSnapshot = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "[T-017] FR-015: baseline.json must deserialise into BaselineSnapshot \
             ({e}), got: {text}"
        )
    });

    assert!(
        !snapshot.model_id.is_empty(),
        "[T-017] FR-015: model_id must be populated, snapshot = {snapshot:?}"
    );
    assert!(
        !snapshot.fixture_hash.is_empty(),
        "[T-017] FR-015: fixture_hash must be populated, snapshot = {snapshot:?}"
    );
    assert!(
        !snapshot.global.is_empty(),
        "[T-017] FR-015: global metrics must be present, snapshot = {snapshot:?}"
    );
    assert!(
        !snapshot.per_category.is_empty(),
        "[T-017] FR-015: per_category breakdown must be present, snapshot = {snapshot:?}"
    );
    assert!(
        snapshot.latency_p50_ms >= 0.0,
        "[T-017] FR-015: latency_p50_ms must be non-negative, got {}",
        snapshot.latency_p50_ms
    );
    assert!(
        snapshot.latency_p95_ms >= 0.0,
        "[T-017] FR-015: latency_p95_ms must be non-negative, got {}",
        snapshot.latency_p95_ms
    );
}

// T-019: t019_verify_baseline_passes_against_committed_snapshot
// FR-017: verify-baseline against the committed baseline.json must exit 0
//         with stderr containing "verify-baseline: passed".
#[test]
#[ignore] // requires ruri-v3-310m cached + MLX
fn t019_verify_baseline_passes_against_committed_snapshot() {
    let output = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .args(["verify-baseline", &format!("baseline={BASELINE_PATH}")])
        .output()
        .expect("spawn eval_harness verify-baseline");
    assert_smoke_success(&output);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("verify-baseline: passed"),
        "[T-019] FR-017: stderr must contain `verify-baseline: passed`, got: {stderr}"
    );
}

/// Parse the stdout JSON of an `eval_harness evaluate` invocation and extract
/// the `f64` value at `field`. Centralised so T-013/T-014/T-015/T-016 share a
/// single error-reporting style instead of repeating the same parse-and-extract
/// dance in every test.
fn parse_metric_field(label: &str, output: &Output, field: &str) -> f64 {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("[{label}] stdout must be JSON ({e}), got: {stdout}"));
    json.get(field)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| panic!("[{label}] missing {field}: {stdout}"))
}

// Subprocess success assertion mirroring `tests/mlx_smoke.rs::assert_smoke_success`.
// Distinguishes seatbelt-skip (panic with sandbox hint) from MLX FFI signal
// kills (panic with signal number) and ordinary failures.
fn assert_smoke_success(output: &Output) {
    if output.status.success() {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.code() == Some(SEATBELT_SKIP_EXIT) {
        panic!(
            "eval_harness skipped in Codex seatbelt sandbox; \
             run this verification outside the sandbox\nstderr: {stderr}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = output.status.signal() {
            panic!("eval_harness killed by signal {sig} (MLX FFI crash)\nstderr: {stderr}");
        }
    }
    panic!(
        "eval_harness failed with {:?}\nstderr: {stderr}",
        output.status.code()
    );
}
