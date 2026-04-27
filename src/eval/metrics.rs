//! IR metrics and bootstrap CI (FR-001..FR-004).
//!
//! Pure functions; no I/O, no mlx dependency. Bootstrap CI uses
//! `rand_chacha::ChaCha8Rng` so identical seed produces bit-identical
//! resample sequences across runs (NFR-002).
//!
//! References:
//! - FR-001 Recall@k = |relevant ∩ top_k| / |relevant_total|, threshold rel ≥ 1
//! - FR-002 MRR@k = 1 / rank_of_first_relevant_in_top_k, 0.0 if none
//! - FR-003 nDCG@k = DCG@k / IDCG@k, DCG = Σ (2^rel - 1) / log_2(i+1)
//! - FR-004 bootstrap 95% CI over n=1000 resamples, seed-determined

use std::collections::HashMap;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Single metric outcome plus its bootstrap CI envelope.
///
/// The `uninformative` flag is set when the CI half-width exceeds 0.10
/// (BR-002 / FR-016). The flag is computed at serialization time, not
/// inside the metric functions themselves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Metric identifier, e.g. "recall@5", "ndcg@10".
    pub name: String,
    /// Cutoff used when computing the metric.
    pub k: usize,
    /// Mean over the resampled queries.
    pub point_estimate: f64,
    /// 2.5th percentile of the bootstrap distribution.
    pub ci_lower: f64,
    /// 97.5th percentile of the bootstrap distribution.
    pub ci_upper: f64,
    /// True when (`ci_upper` - `ci_lower`) / 2 > 0.10.
    pub uninformative: bool,
}

/// Recall@k with binary relevance threshold rel ≥ 1.
///
/// Returns `0.0` when the relevance map contains no relevant doc
/// (`grade ≥ 1`), since recall is undefined without a positive class.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn recall_at_k(ranked: &[&str], relevance: &HashMap<String, u8>, k: usize) -> f64 {
    let relevant_total = relevance.values().filter(|&&g| g >= 1).count();
    if relevant_total == 0 {
        return 0.0;
    }
    let intersect = ranked
        .iter()
        .take(k)
        .filter(|id| relevance.get(**id).is_some_and(|&g| g >= 1))
        .count();
    intersect as f64 / relevant_total as f64
}

/// MRR@k with binary relevance threshold rel ≥ 1.
///
/// Returns `0.0` when the top-k window contains no relevant document.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mrr_at_k(ranked: &[&str], relevance: &HashMap<String, u8>, k: usize) -> f64 {
    for (idx, id) in ranked.iter().take(k).enumerate() {
        if relevance.get(*id).is_some_and(|&g| g >= 1) {
            return 1.0 / (idx + 1) as f64;
        }
    }
    0.0
}

/// nDCG@k using graded relevance grades 0/1/2/3 with `(2^rel - 1)` gain.
///
/// Returns `0.0` when IDCG@k is `0.0` (no relevant docs in the corpus for
/// this query). DCG denominator is `log_2(rank + 1)` where `rank` is
/// 1-indexed (i.e. position 1 yields denominator `log_2(2) = 1`).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ndcg_at_k(ranked: &[&str], relevance: &HashMap<String, u8>, k: usize) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(idx, id)| {
            let rel = f64::from(relevance.get(*id).copied().unwrap_or(0));
            let denom = ((idx + 2) as f64).log2();
            (rel.exp2() - 1.0) / denom
        })
        .sum();
    let mut grades: Vec<u8> = relevance.values().copied().collect();
    grades.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = grades
        .iter()
        .take(k)
        .enumerate()
        .map(|(idx, &grade)| {
            let rel = f64::from(grade);
            let denom = ((idx + 2) as f64).log2();
            (rel.exp2() - 1.0) / denom
        })
        .sum();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// Bootstrap 95% CI for an arbitrary metric.
///
/// Returns `(point_estimate, ci_lower, ci_upper)`. The point estimate is
/// `metric(values)`; the bounds are the 2.5th and 97.5th percentiles of
/// `n_resamples` bootstrap resamples drawn with replacement using
/// `ChaCha8Rng::seed_from_u64(seed)`. BR-004 fixes the default seed at 42.
///
/// When `values` is empty or `n_resamples` is 0, returns the point estimate
/// for all three components (degenerate CI of zero width).
pub fn bootstrap_ci<F>(values: &[f64], metric: F, n_resamples: usize, seed: u64) -> (f64, f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let point_estimate = metric(values);
    if values.is_empty() || n_resamples == 0 {
        return (point_estimate, point_estimate, point_estimate);
    }
    let len = values.len();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Reuse a single scratch buffer across resamples instead of allocating a
    // fresh `Vec<f64>` per iteration. For 1000 resamples × N queries × 4
    // metrics this drops ~560k Vec allocations to one.
    let mut scratch: Vec<f64> = Vec::with_capacity(len);
    let mut samples: Vec<f64> = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        scratch.clear();
        for _ in 0..len {
            scratch.push(values[rng.random_range(0..len)]);
        }
        samples.push(metric(&scratch));
    }
    samples.sort_by(f64::total_cmp);
    let lower_idx = (n_resamples * 25) / 1000;
    let upper_idx = ((n_resamples * 975) / 1000).min(n_resamples - 1);
    (point_estimate, samples[lower_idx], samples[upper_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a relevance map from `(doc_id, grade)` pairs.
    fn relevance(entries: &[(&str, u8)]) -> HashMap<String, u8> {
        entries
            .iter()
            .map(|(id, grade)| ((*id).to_owned(), *grade))
            .collect()
    }

    /// Adopt a slice literal into the borrowed shape the metric APIs accept.
    fn ranked<'a>(ids: &[&'a str]) -> Vec<&'a str> {
        ids.to_vec()
    }

    // T-001: recall_at_k_all_relevant_in_top_k_returns_one
    // FR-001: ranked = [d1..d5], relevant = {d1, d4}, k = 5 → 1.0
    #[test]
    fn recall_at_k_all_relevant_in_top_k_returns_one() {
        let ranked = ranked(&["d1", "d2", "d3", "d4", "d5"]);
        let rel = relevance(&[("d1", 1), ("d4", 1)]);

        let result = recall_at_k(&ranked, &rel, 5);

        assert!(
            (result - 1.0).abs() < f64::EPSILON,
            "all relevant docs sit inside top-k → recall must be 1.0, got: {result}"
        );
    }

    // T-002: recall_at_k_no_relevant_in_top_k_returns_zero
    // FR-001: ranked window contains no relevant doc → 0.0
    #[test]
    fn recall_at_k_no_relevant_in_top_k_returns_zero() {
        let ranked = ranked(&["d2", "d3", "d5", "d6", "d7"]);
        let rel = relevance(&[("d1", 1), ("d4", 1)]);

        let result = recall_at_k(&ranked, &rel, 5);

        assert!(
            result.abs() < f64::EPSILON,
            "top-k window is disjoint from relevant set → recall must be 0.0, got: {result}"
        );
    }

    // T-003: mrr_at_k_first_relevant_at_rank_two_returns_half
    // FR-002: ranked = [d3, d1, d4], relevant = {d1, d4}, k = 5 → 1/2
    #[test]
    fn mrr_at_k_first_relevant_at_rank_two_returns_half() {
        let ranked = ranked(&["d3", "d1", "d4"]);
        let rel = relevance(&[("d1", 1), ("d4", 1)]);

        let result = mrr_at_k(&ranked, &rel, 5);

        assert!(
            (result - 0.5).abs() < f64::EPSILON,
            "first relevant doc is at rank 2 → MRR must be 1/2, got: {result}"
        );
    }

    // T-004: mrr_at_k_no_relevant_in_top_k_returns_zero
    // FR-002: top-k contains no relevant doc → 0.0
    #[test]
    fn mrr_at_k_no_relevant_in_top_k_returns_zero() {
        let ranked = ranked(&["d2", "d3", "d5"]);
        let rel = relevance(&[("d1", 1), ("d4", 1)]);

        let result = mrr_at_k(&ranked, &rel, 5);

        assert!(
            result.abs() < f64::EPSILON,
            "top-k window contains no relevant doc → MRR must be 0.0, got: {result}"
        );
    }

    // T-005: ndcg_at_k_perfect_graded_ordering_returns_one
    // FR-003: ranked = ideal order with grades 3,2,1 → DCG == IDCG → 1.0
    #[test]
    fn ndcg_at_k_perfect_graded_ordering_returns_one() {
        let ranked = ranked(&["d1", "d2", "d3"]);
        let rel = relevance(&[("d1", 3), ("d2", 2), ("d3", 1)]);

        let result = ndcg_at_k(&ranked, &rel, 3);

        assert!(
            (result - 1.0).abs() < f64::EPSILON,
            "ranking matches ideal graded order → nDCG must be 1.0, got: {result}"
        );
    }

    // T-006: ndcg_at_k_worst_graded_ordering_below_perfect
    // FR-003: ranked = [d3, d2, d1] is the reversed (worst) ordering of
    // relevance {d1:3, d2:2, d3:1}.
    //   DCG  = 1/1 + 3/log2(3) + 7/2  ≈ 6.3928
    //   IDCG = 7/1 + 3/log2(3) + 1/2  ≈ 9.3928
    //   nDCG = DCG / IDCG             ≈ 0.6806
    //
    // Spec T-006 originally asserted result < 0.5; FR-003's graded gain
    // dampens the lower bound to ~0.68 for these inputs, so the test
    // asserts the formula-derived value instead.
    #[test]
    fn ndcg_at_k_worst_graded_ordering_below_perfect() {
        let ranked = ranked(&["d3", "d2", "d1"]);
        let rel = relevance(&[("d1", 3), ("d2", 2), ("d3", 1)]);

        let result = ndcg_at_k(&ranked, &rel, 3);

        assert!(
            result < 1.0,
            "worst graded ordering must score below perfect → got: {result}"
        );
        let expected = 0.6806;
        assert!(
            (result - expected).abs() < 0.01,
            "expected nDCG ≈ {expected} (FR-003 formula on reversed graded inputs), got: {result}"
        );
    }

    // T-007: bootstrap_ci_is_bit_identical_for_same_seed
    // FR-004 / NFR-002: identical input + identical seed must produce
    // identical f64 values across two invocations.
    #[test]
    fn bootstrap_ci_is_bit_identical_for_same_seed() {
        let per_query_scores: Vec<f64> = vec![1.0, 0.8, 0.6, 0.4, 0.0, 1.0, 0.5, 0.75, 0.25, 0.9];
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;

        let (point_a, lower_a, upper_a) = bootstrap_ci(&per_query_scores, mean, 1000, 42);
        let (point_b, lower_b, upper_b) = bootstrap_ci(&per_query_scores, mean, 1000, 42);

        assert!(
            (point_a - point_b).abs() < f64::EPSILON,
            "FR-004: point estimate must be bit-identical across runs with seed=42 \
             (got {point_a} vs {point_b})"
        );
        assert!(
            (lower_a - lower_b).abs() < f64::EPSILON,
            "FR-004 / NFR-002: ci_lower must be bit-identical across runs with seed=42 \
             (got {lower_a} vs {lower_b})"
        );
        assert!(
            (upper_a - upper_b).abs() < f64::EPSILON,
            "FR-004 / NFR-002: ci_upper must be bit-identical across runs with seed=42 \
             (got {upper_a} vs {upper_b})"
        );
    }
}
