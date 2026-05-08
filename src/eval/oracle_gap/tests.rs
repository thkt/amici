//! Unit tests for [`crate::eval::oracle_gap`].

use std::collections::BTreeMap;

use rurico::retrieval::HybridSearchConfig;
use rurico::storage::pre_phase_5_disabled;

use super::{
    Ac4Violation, MetricGap, OracleGapError, compute_gap, detect_ac4_violations, format_markdown,
};
use crate::eval::baseline::{BASELINE_SCHEMA_VERSION, BaselineKind, BaselineSnapshot};
use crate::eval::metrics::MetricResult;

/// Build a stub `MetricResult` for the given metric label and point estimate.
/// CI bounds are degenerate (zero width); they are not exercised by the
/// gap diff itself.
fn metric(name: &str, k: usize, point: f64) -> MetricResult {
    MetricResult {
        name: name.to_owned(),
        k,
        point_estimate: point,
        ci_lower: point,
        ci_upper: point,
        uninformative: false,
    }
}

/// Build a `BaselineSnapshot` shell with the provided `kind` /
/// `fixture_hash` / metric lists. Other provenance fields take stub
/// values that the gap computation does not inspect.
fn snapshot(
    kind: BaselineKind,
    fixture_hash: &str,
    global: Vec<MetricResult>,
    per_category: BTreeMap<String, Vec<MetricResult>>,
) -> BaselineSnapshot {
    BaselineSnapshot {
        schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
        kind,
        captured_with: "test".to_owned(),
        timestamp: "epoch:0".to_owned(),
        model_id: "test/model".to_owned(),
        model_revision: "rev".to_owned(),
        mlx_rs_version: "0.0.0".to_owned(),
        fixture_hash: fixture_hash.to_owned(),
        aggregation: "identity".to_owned(),
        merge_config: HybridSearchConfig::default(),
        normalization: pre_phase_5_disabled(),
        global,
        per_category,
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
    }
}

// T-052-401: compute_gap_rejects_baseline_with_non_forward_kind
#[test]
fn compute_gap_rejects_baseline_with_non_forward_kind() {
    let baseline = snapshot(BaselineKind::Reverse, "fnv1a64:0", vec![], BTreeMap::new());
    let oracle = snapshot(BaselineKind::Oracle, "fnv1a64:0", vec![], BTreeMap::new());
    let result = compute_gap(&baseline, &oracle);
    assert!(
        matches!(
            result,
            Err(OracleGapError::BaselineNotForward(BaselineKind::Reverse))
        ),
        "must reject non-Forward baseline kind; got: {result:?}"
    );
}

// T-052-402: compute_gap_rejects_oracle_with_non_oracle_kind
#[test]
fn compute_gap_rejects_oracle_with_non_oracle_kind() {
    let baseline = snapshot(BaselineKind::Forward, "fnv1a64:0", vec![], BTreeMap::new());
    let oracle = snapshot(BaselineKind::Forward, "fnv1a64:0", vec![], BTreeMap::new());
    let result = compute_gap(&baseline, &oracle);
    assert!(
        matches!(
            result,
            Err(OracleGapError::OracleNotOracle(BaselineKind::Forward))
        ),
        "must reject non-Oracle oracle kind; got: {result:?}"
    );
}

// T-052-403: compute_gap_rejects_fixture_hash_mismatch
//
// Two snapshots from different fixtures must not be compared — the
// per-query populations differ and the gap would be meaningless.
#[test]
fn compute_gap_rejects_fixture_hash_mismatch() {
    let baseline = snapshot(
        BaselineKind::Forward,
        "fnv1a64:aaaa",
        vec![],
        BTreeMap::new(),
    );
    let oracle = snapshot(
        BaselineKind::Oracle,
        "fnv1a64:bbbb",
        vec![],
        BTreeMap::new(),
    );
    let result = compute_gap(&baseline, &oracle);
    assert!(
        matches!(
            result,
            Err(OracleGapError::FixtureHashMismatch { ref baseline, ref oracle })
                if baseline == "fnv1a64:aaaa" && oracle == "fnv1a64:bbbb"
        ),
        "must reject differing fixture_hash; got: {result:?}"
    );
}

// T-052-404: compute_gap_emits_per_metric_diff_against_matching_fixture
#[test]
fn compute_gap_emits_per_metric_diff_against_matching_fixture() {
    let baseline = snapshot(
        BaselineKind::Forward,
        "fnv1a64:0",
        vec![
            metric("recall@5", 5, 0.6300),
            metric("recall@10", 10, 0.7415),
        ],
        BTreeMap::new(),
    );
    let oracle = snapshot(
        BaselineKind::Oracle,
        "fnv1a64:0",
        vec![
            metric("recall@5", 5, 0.7200),
            metric("recall@10", 10, 1.0000),
        ],
        BTreeMap::new(),
    );
    let gap = compute_gap(&baseline, &oracle).expect("matching kinds + fixture must succeed");
    assert_eq!(gap.global.len(), 2);
    assert!(
        (gap.global[0].diff - 0.09).abs() < 1e-9,
        "recall@5 diff = +9.0pt"
    );
    assert!(
        (gap.global[1].diff - 0.2585).abs() < 1e-9,
        "recall@10 diff = +25.85pt"
    );
}

// T-052-405: detect_ac4_violations_flags_recall_regression_per_category
//
// AC 4: oracle.recall@k must be >= baseline.recall@k for every
// category. A negative diff signals a wiring bug.
#[test]
fn detect_ac4_violations_flags_recall_regression_per_category() {
    let mut per_category: BTreeMap<String, Vec<MetricGap>> = BTreeMap::new();
    per_category.insert(
        "troubleshooting".to_owned(),
        vec![
            MetricGap {
                name: "recall@5".to_owned(),
                k: 5,
                baseline_point: 0.7,
                oracle_point: 0.65,
                diff: -0.05,
            },
            MetricGap {
                name: "ndcg@10".to_owned(),
                k: 10,
                baseline_point: 0.9,
                oracle_point: 0.85,
                diff: -0.05,
            },
        ],
    );
    let violations = detect_ac4_violations(&per_category);
    assert_eq!(
        violations,
        vec![Ac4Violation {
            category: "troubleshooting".to_owned(),
            metric_name: "recall@5".to_owned(),
            k: 5,
            baseline_point: 0.7,
            oracle_point: 0.65,
        }],
        "AC 4 covers recall metrics only; ndcg regression must not be flagged"
    );
}

// T-052-406: detect_ac4_violations_returns_empty_when_recall_holds
#[test]
fn detect_ac4_violations_returns_empty_when_recall_holds() {
    let mut per_category: BTreeMap<String, Vec<MetricGap>> = BTreeMap::new();
    per_category.insert(
        "listing".to_owned(),
        vec![MetricGap {
            name: "recall@5".to_owned(),
            k: 5,
            baseline_point: 0.5,
            oracle_point: 0.7,
            diff: 0.2,
        }],
    );
    assert!(detect_ac4_violations(&per_category).is_empty());
}

// T-052-407: format_markdown_includes_global_per_category_and_pass_banner
#[test]
fn format_markdown_includes_global_per_category_and_pass_banner() {
    let baseline = snapshot(
        BaselineKind::Forward,
        "fnv1a64:0",
        vec![metric("recall@5", 5, 0.6)],
        {
            let mut m = BTreeMap::new();
            m.insert("listing".to_owned(), vec![metric("recall@5", 5, 0.5)]);
            m
        },
    );
    let oracle = snapshot(
        BaselineKind::Oracle,
        "fnv1a64:0",
        vec![metric("recall@5", 5, 0.7)],
        {
            let mut m = BTreeMap::new();
            m.insert("listing".to_owned(), vec![metric("recall@5", 5, 0.6)]);
            m
        },
    );
    let gap = compute_gap(&baseline, &oracle).expect("compute_gap must succeed");
    let md = format_markdown(&gap);
    assert!(md.contains("# Oracle Gap"), "must contain top-level header");
    assert!(md.contains("## Global"), "must contain global section");
    assert!(
        md.contains("## Per-category"),
        "must contain per-category section"
    );
    assert!(md.contains("### listing"), "must list each category");
    assert!(
        md.contains("PASS"),
        "AC 4 must report PASS when no recall regressions; got: {md}"
    );
}

// T-052-408: format_markdown_lists_violations_under_fail_banner
#[test]
fn format_markdown_lists_violations_under_fail_banner() {
    let baseline = snapshot(BaselineKind::Forward, "fnv1a64:0", vec![], {
        let mut m = BTreeMap::new();
        m.insert("listing".to_owned(), vec![metric("recall@5", 5, 0.7)]);
        m
    });
    let oracle = snapshot(BaselineKind::Oracle, "fnv1a64:0", vec![], {
        let mut m = BTreeMap::new();
        m.insert("listing".to_owned(), vec![metric("recall@5", 5, 0.65)]);
        m
    });
    let gap = compute_gap(&baseline, &oracle).expect("compute_gap must succeed");
    let md = format_markdown(&gap);
    assert!(
        md.contains("FAIL"),
        "AC 4 must report FAIL with a regression"
    );
    assert!(md.contains("listing"), "must name the offending category");
    assert!(md.contains("recall@5"), "must name the offending metric");
}
