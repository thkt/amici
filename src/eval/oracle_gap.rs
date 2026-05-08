//! Oracle gap report (Issue #52 PR2 / AC 3, AC 4).
//!
//! Compares a Forward baseline (`capture-baseline` output) against an
//! Oracle baseline (`capture-oracle` output) captured from the same
//! fixture. Quantifies the search-side bottleneck: how much recall the
//! production retrieval is leaving on the table relative to a perfect
//! top-rank inject — the Phase 2 priority signal.
//!
//! AC 4: for every query category, `oracle.recall@k` must be at least
//! `baseline.recall@k`. A violation means the oracle inject failed to
//! land in `top-k` for some query in the category — almost certainly a
//! wiring regression (the inject is supposed to dominate Stage 2's RRF
//! score). The harness exits `EXIT_REGRESSION` so CI catches it.

use std::collections::BTreeMap;

use crate::eval::baseline::{BaselineKind, BaselineSnapshot};
use crate::eval::metrics::MetricResult;

/// Prefix on every metric name that the AC 4 gate covers.
///
/// Single source of truth so a typo (`"recall_"`) cannot silently
/// disable the gate.
const RECALL_METRIC_PREFIX: &str = "recall@";

/// Errors surfaced by [`compute_gap`].
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum OracleGapError {
    /// `baseline=<path>` did not point at a `kind=forward` snapshot.
    #[error("oracle-gap: baseline kind {0:?} is not Forward; pass capture-baseline output")]
    BaselineNotForward(BaselineKind),
    /// `oracle=<path>` did not point at a `kind=oracle` snapshot.
    #[error("oracle-gap: oracle kind {0:?} is not Oracle; pass capture-oracle output")]
    OracleNotOracle(BaselineKind),
    /// The two snapshots were captured under different schema versions.
    /// Comparing them would silently mix metric definitions.
    #[error("oracle-gap: schema_version mismatch (baseline={baseline:?}, oracle={oracle:?})")]
    SchemaVersionMismatch {
        /// Forward snapshot version label.
        baseline: String,
        /// Oracle snapshot version label.
        oracle: String,
    },
    /// The two snapshots were captured against different fixtures.
    /// Comparing the gap would mix populations.
    #[error(
        "oracle-gap: fixture_hash mismatch — comparing different fixtures \
         (baseline={baseline:?}, oracle={oracle:?})"
    )]
    FixtureHashMismatch {
        /// Forward snapshot fixture content hash.
        baseline: String,
        /// Oracle snapshot fixture content hash.
        oracle: String,
    },
    /// The two snapshots were captured under different pipeline knobs
    /// (`aggregation`, `merge_config`, `normalization`, or model
    /// identity). The gap would no longer measure the oracle under the
    /// same downstream stack — AC 4 could swing on a config typo.
    #[error(
        "oracle-gap: {field} mismatch — \
         baseline and oracle were captured under different pipeline configs \
         (baseline={baseline:?}, oracle={oracle:?})"
    )]
    PipelineConfigMismatch {
        /// Snapshot field that disagreed (`aggregation`, `merge_config`,
        /// `normalization`, `model_id`, or `model_revision`).
        field: &'static str,
        /// Forward snapshot value (debug-formatted).
        baseline: String,
        /// Oracle snapshot value (debug-formatted).
        oracle: String,
    },
    /// A category present on the baseline is absent from the oracle.
    /// AC 4 would silently skip that bucket and PASS vacuously.
    #[error(
        "oracle-gap: oracle snapshot is missing per-category bucket {category:?} \
         that baseline reports — regenerate before comparing"
    )]
    MissingOracleCategory {
        /// Category label that exists on the baseline but not the oracle.
        category: String,
    },
    /// A metric present on the baseline is absent from the oracle for a
    /// given bucket (global or per-category). AC 4 would silently skip
    /// that metric.
    #[error(
        "oracle-gap: oracle snapshot is missing metric {metric:?}@{k} in {scope} — \
         regenerate before comparing"
    )]
    MissingOracleMetric {
        /// Scope label (`"<global>"` or category name).
        scope: String,
        /// Metric label that exists on the baseline but not the oracle.
        metric: String,
        /// `k` cutoff carried by the missing metric.
        k: usize,
    },
}

/// Per-metric difference between Forward and Oracle.
///
/// `diff` is `oracle_point - baseline_point`; positive means the
/// retrieval ceiling sits above the production result for that metric.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricGap {
    /// Metric label (e.g. `recall@5`, `mrr@10`).
    pub name: String,
    /// `k` cutoff carried by the metric.
    pub k: usize,
    /// Forward `point_estimate`.
    pub baseline_point: f64,
    /// Oracle `point_estimate`.
    pub oracle_point: f64,
    /// `oracle_point - baseline_point`.
    pub diff: f64,
}

/// Single AC 4 violation: an oracle metric ranked below its baseline
/// counterpart, despite the oracle pipeline force-injecting the relevant
/// doc at rank 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Ac4Violation {
    /// Per-category label that triggered the violation. The global
    /// aggregate is reported in [`OracleGap::global`] but not gated.
    pub category: String,
    /// Metric label that regressed (e.g. `recall@5`).
    pub name: String,
    /// `k` cutoff carried by the metric.
    pub k: usize,
    /// Forward point estimate.
    pub baseline_point: f64,
    /// Oracle point estimate.
    pub oracle_point: f64,
}

/// Combined gap report consumed by [`format_markdown`] and the
/// `oracle-gap` subcommand exit code.
#[derive(Debug, Clone, PartialEq)]
pub struct OracleGap {
    /// Per-metric diff over the global aggregation.
    pub global: Vec<MetricGap>,
    /// Per-category diff. Every baseline category must have an oracle
    /// counterpart with the same metric set; mismatches surface as
    /// [`OracleGapError::MissingOracleCategory`] /
    /// [`OracleGapError::MissingOracleMetric`] so AC 4 cannot PASS
    /// vacuously on a partially-edited snapshot.
    pub per_category: BTreeMap<String, Vec<MetricGap>>,
    /// AC 4 violations detected across the per-category breakdown.
    /// `recall@k` is the only metric subject to the gate (the issue body
    /// pins the AC to recall); `mrr` and `ndcg` are reported but not
    /// gated since the production reranker can promote a relevant doc
    /// past the oracle inject when both land in the top-k tie.
    pub ac4_violations: Vec<Ac4Violation>,
}

/// Compute the gap between `baseline` (`kind=Forward`) and `oracle`
/// (`kind=Oracle`).
///
/// # Errors
///
/// Returns [`OracleGapError`] when the inputs disagree on `kind`,
/// `schema_version`, or `fixture_hash`. Each rejection prevents a
/// silently-incorrect comparison.
pub fn compute_gap(
    baseline: &BaselineSnapshot,
    oracle: &BaselineSnapshot,
) -> Result<OracleGap, OracleGapError> {
    if baseline.kind != BaselineKind::Forward {
        return Err(OracleGapError::BaselineNotForward(baseline.kind));
    }
    if oracle.kind != BaselineKind::Oracle {
        return Err(OracleGapError::OracleNotOracle(oracle.kind));
    }
    if baseline.schema_version != oracle.schema_version {
        return Err(OracleGapError::SchemaVersionMismatch {
            baseline: baseline.schema_version.clone(),
            oracle: oracle.schema_version.clone(),
        });
    }
    if baseline.fixture_hash != oracle.fixture_hash {
        return Err(OracleGapError::FixtureHashMismatch {
            baseline: baseline.fixture_hash.clone(),
            oracle: oracle.fixture_hash.clone(),
        });
    }
    enforce_pipeline_config_match(baseline, oracle)?;

    let global = diff_metric_lists(&baseline.global, &oracle.global, "<global>")?;
    let mut per_category: BTreeMap<String, Vec<MetricGap>> = BTreeMap::new();
    for (category, baseline_metrics) in &baseline.per_category {
        let oracle_metrics = oracle.per_category.get(category).ok_or_else(|| {
            OracleGapError::MissingOracleCategory {
                category: category.clone(),
            }
        })?;
        let gap = diff_metric_lists(baseline_metrics, oracle_metrics, category)?;
        per_category.insert(category.clone(), gap);
    }
    let ac4_violations = detect_ac4_violations(&per_category);
    Ok(OracleGap {
        global,
        per_category,
        ac4_violations,
    })
}

/// Reject snapshots whose pipeline knobs disagree.
///
/// `aggregation`, `merge_config`, `normalization`, `model_id`, and
/// `model_revision` define the downstream stack. A gap captured under
/// mismatched knobs no longer measures "retrieval ceiling under the
/// same rerank/aggregation" — AC 4 could swing on a config typo.
fn enforce_pipeline_config_match(
    baseline: &BaselineSnapshot,
    oracle: &BaselineSnapshot,
) -> Result<(), OracleGapError> {
    if baseline.aggregation != oracle.aggregation {
        return Err(OracleGapError::PipelineConfigMismatch {
            field: "aggregation",
            baseline: baseline.aggregation.clone(),
            oracle: oracle.aggregation.clone(),
        });
    }
    if baseline.merge_config != oracle.merge_config {
        return Err(OracleGapError::PipelineConfigMismatch {
            field: "merge_config",
            baseline: format!("{:?}", baseline.merge_config),
            oracle: format!("{:?}", oracle.merge_config),
        });
    }
    if baseline.normalization != oracle.normalization {
        return Err(OracleGapError::PipelineConfigMismatch {
            field: "normalization",
            baseline: format!("{:?}", baseline.normalization),
            oracle: format!("{:?}", oracle.normalization),
        });
    }
    if baseline.model_id != oracle.model_id {
        return Err(OracleGapError::PipelineConfigMismatch {
            field: "model_id",
            baseline: baseline.model_id.clone(),
            oracle: oracle.model_id.clone(),
        });
    }
    if baseline.model_revision != oracle.model_revision {
        return Err(OracleGapError::PipelineConfigMismatch {
            field: "model_revision",
            baseline: baseline.model_revision.clone(),
            oracle: oracle.model_revision.clone(),
        });
    }
    Ok(())
}

/// Compute per-metric `MetricGap` for two metric lists keyed by
/// `(name, k)`.
///
/// Returns [`OracleGapError::MissingOracleMetric`] when a baseline metric
/// has no oracle counterpart in the same `scope` — silently dropping it
/// would let AC 4 PASS vacuously on a partially-edited oracle snapshot.
fn diff_metric_lists(
    baseline: &[MetricResult],
    oracle: &[MetricResult],
    scope: &str,
) -> Result<Vec<MetricGap>, OracleGapError> {
    let mut out = Vec::with_capacity(baseline.len());
    for b in baseline {
        let o = oracle
            .iter()
            .find(|o| o.name == b.name && o.k == b.k)
            .ok_or_else(|| OracleGapError::MissingOracleMetric {
                scope: scope.to_owned(),
                metric: b.name.clone(),
                k: b.k,
            })?;
        out.push(MetricGap {
            name: b.name.clone(),
            k: b.k,
            baseline_point: b.point_estimate,
            oracle_point: o.point_estimate,
            diff: o.point_estimate - b.point_estimate,
        });
    }
    Ok(out)
}

/// Scan the per-category gap for `recall@k` regressions and emit one
/// [`Ac4Violation`] per offending (category, metric) pair.
///
/// The AC 4 gate covers **recall metrics only** because the oracle
/// inject specifically targets recall: forcing the relevant doc into
/// `top-k` cannot reduce the retrieval recall, so any regression
/// signals a wiring bug. `mrr` and `ndcg` involve reranker score
/// comparisons that can legitimately cause a relevant doc to slide
/// inside `top-k` after the oracle replaces a stronger natural hit;
/// reporting those as violations would be a false positive.
fn detect_ac4_violations(per_category: &BTreeMap<String, Vec<MetricGap>>) -> Vec<Ac4Violation> {
    let mut violations = Vec::new();
    for (category, gaps) in per_category {
        for gap in gaps {
            if gap.name.starts_with(RECALL_METRIC_PREFIX) && gap.diff < 0.0 {
                violations.push(Ac4Violation {
                    category: category.clone(),
                    name: gap.name.clone(),
                    k: gap.k,
                    baseline_point: gap.baseline_point,
                    oracle_point: gap.oracle_point,
                });
            }
        }
    }
    violations
}

/// Render an [`OracleGap`] as a markdown report suitable for stdout
/// inside a PR description or `oracle-gap` subcommand output.
#[must_use]
pub fn format_markdown(gap: &OracleGap) -> String {
    let mut out = String::new();
    out.push_str("# Oracle Gap\n\n## Global\n\n");
    write_metric_table(&mut out, &gap.global);
    out.push_str("\n## Per-category\n");
    for (category, gaps) in &gap.per_category {
        out.push_str(&format!("\n### {category}\n\n"));
        write_metric_table(&mut out, gaps);
    }
    out.push_str("\n## AC 4 (per-category recall@k ≥ baseline)\n\n");
    if gap.ac4_violations.is_empty() {
        out.push_str("PASS: every category satisfies oracle.recall@k ≥ baseline.recall@k.\n");
    } else {
        out.push_str("FAIL — the oracle inject regressed recall in the following slots:\n\n");
        for v in &gap.ac4_violations {
            out.push_str(&format!(
                "- `{category}` `{name}`: oracle {oracle:.4} < baseline {baseline:.4} \
                 (diff {diff:+.4})\n",
                category = v.category,
                name = v.name,
                oracle = v.oracle_point,
                baseline = v.baseline_point,
                diff = v.oracle_point - v.baseline_point,
            ));
        }
    }
    out
}

fn write_metric_table(buf: &mut String, gaps: &[MetricGap]) {
    buf.push_str("| Metric | Forward | Oracle | Gap |\n");
    buf.push_str("| --- | --- | --- | --- |\n");
    for g in gaps {
        buf.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:+.4} |\n",
            g.name, g.baseline_point, g.oracle_point, g.diff,
        ));
    }
}

#[cfg(test)]
mod tests;
