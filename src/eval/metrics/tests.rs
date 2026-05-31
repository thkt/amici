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

// T-061-001: hit_at_k_top_1_relevant_returns_one
// Hit@k: ranked = ["d1", "d2"], relevance = {"d1": 1}, k = 1 → 1.0
// Perspective: Equivalence (relevant doc inside top-k window).
#[test]
fn hit_at_k_top_1_relevant_returns_one() {
    let ranked = ranked(&["d1", "d2"]);
    let rel = relevance(&[("d1", 1)]);

    let result = hit_at_k(&ranked, &rel, 1);

    assert!(
        (result - 1.0).abs() < f64::EPSILON,
        "relevant doc at rank 1 within k=1 → Hit@k must be 1.0, got: {result}"
    );
}

// T-061-002: hit_at_k_relevant_at_rank_two_within_k_returns_one
// Hit@k: ranked = ["d2", "d1"], relevance = {"d1": 1}, k = 2 → 1.0
// Perspective: Boundary (relevant doc at the last position inside k).
#[test]
fn hit_at_k_relevant_at_rank_two_within_k_returns_one() {
    let ranked = ranked(&["d2", "d1"]);
    let rel = relevance(&[("d1", 1)]);

    let result = hit_at_k(&ranked, &rel, 2);

    assert!(
        (result - 1.0).abs() < f64::EPSILON,
        "relevant doc at rank 2 within k=2 → Hit@k must be 1.0, got: {result}"
    );
}

// T-061-003: hit_at_k_empty_ranked_returns_zero
// Hit@k: ranked = [], relevance = {"d1": 1}, k = 5 → 0.0 (no panic)
// Perspective: Boundary / Error (empty ranked slice must not panic).
#[test]
fn hit_at_k_empty_ranked_returns_zero() {
    let ranked: Vec<&str> = ranked(&[]);
    let rel = relevance(&[("d1", 1)]);

    let result = hit_at_k(&ranked, &rel, 5);

    assert!(
        result.abs() < f64::EPSILON,
        "empty ranked slice → Hit@k must be 0.0, got: {result}"
    );
}

// T-061-004: hit_at_k_no_relevant_in_top_k_returns_zero
// Hit@k: ranked = ["d2", "d3"], relevance = {"d1": 1}, k = 2 → 0.0
// Perspective: Branch (top-k disjoint from relevant set).
#[test]
fn hit_at_k_no_relevant_in_top_k_returns_zero() {
    let ranked = ranked(&["d2", "d3"]);
    let rel = relevance(&[("d1", 1)]);

    let result = hit_at_k(&ranked, &rel, 2);

    assert!(
        result.abs() < f64::EPSILON,
        "top-k window disjoint from relevant set → Hit@k must be 0.0, got: {result}"
    );
}

// T-061-005: hit_at_k_k_greater_than_n_returns_one
// Hit@k: ranked = ["d1"], relevance = {"d1": 1}, k = 10 → 1.0 (no panic)
// Perspective: Boundary (k > N must clamp to ranked.len() without panic).
#[test]
fn hit_at_k_k_greater_than_n_returns_one() {
    let ranked = ranked(&["d1"]);
    let rel = relevance(&[("d1", 1)]);

    let result = hit_at_k(&ranked, &rel, 10);

    assert!(
        (result - 1.0).abs() < f64::EPSILON,
        "k=10 exceeds ranked.len()=1 but relevant doc present → Hit@k must be 1.0, got: {result}"
    );
}

// T-061-006: hit_at_k_grade_zero_treated_as_irrelevant_returns_zero
// Hit@k: ranked = ["d1"], relevance = {"d1": 0}, k = 1 → 0.0
// Perspective: Boundary / Condition (grade < 1 must not count as a hit;
// mirrors the rel ≥ 1 threshold used by recall_at_k / mrr_at_k).
#[test]
fn hit_at_k_grade_zero_treated_as_irrelevant_returns_zero() {
    let ranked = ranked(&["d1"]);
    let rel = relevance(&[("d1", 0)]);

    let result = hit_at_k(&ranked, &rel, 1);

    assert!(
        result.abs() < f64::EPSILON,
        "grade 0 is below the rel ≥ 1 threshold → Hit@k must be 0.0, got: {result}"
    );
}

// T-061-007: hit_at_k_grade_three_counts_as_hit_returns_one
// Hit@k: ranked = ["d1"], relevance = {"d1": 3}, k = 1 → 1.0
// Perspective: Boundary / Condition (highest graded relevance still
// satisfies the binary rel ≥ 1 threshold).
#[test]
fn hit_at_k_grade_three_counts_as_hit_returns_one() {
    let ranked = ranked(&["d1"]);
    let rel = relevance(&[("d1", 3)]);

    let result = hit_at_k(&ranked, &rel, 1);

    assert!(
        (result - 1.0).abs() < f64::EPSILON,
        "grade 3 satisfies rel ≥ 1 → Hit@k must be 1.0, got: {result}"
    );
}
