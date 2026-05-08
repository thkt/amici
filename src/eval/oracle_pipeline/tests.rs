//! Unit tests for [`crate::eval::oracle_pipeline`].
//!
//! Split out of `oracle_pipeline.rs` (per `rules/development/STRUCTURE.md`)
//! once the test module grew past the file-line ceiling. The module itself
//! is small; tests dominate.

use std::collections::HashMap;

use rurico::retrieval::{
    Candidate, CandidateSource, HybridSearchConfig, MergeStrategy, WeightedRrf,
};
use rurico::storage::QueryNormalizationConfig;

use super::{OracleError, OracleMerge, evaluate_oracle, relevant_doc_ids};
use crate::eval::fixture::{EvalDocument, EvalQuery};
use crate::eval::pipeline::PipelineConfig;

fn fts_candidate(doc_id: &str, score: f64, rank: usize) -> Candidate {
    Candidate {
        source: CandidateSource::Fts,
        doc_id: doc_id.to_owned(),
        chunk_id: None,
        score,
        rank,
    }
}

// T-052-101: oracle_merge_injects_single_relevant_doc_at_rank_zero
//
// Even when WeightedRrf would have ranked the relevant doc lower, the
// synthetic score (anchored to inner_max + offset) keeps the slot fixed
// under any post-merge sort.
#[test]
fn oracle_merge_injects_single_relevant_doc_at_rank_zero() {
    let candidates = vec![
        fts_candidate("d1", -0.5, 0),
        fts_candidate("d2", -0.6, 1),
        fts_candidate("d3", -0.7, 2),
    ];
    let oracle = OracleMerge::new(HybridSearchConfig::default(), vec!["d3".to_owned()]);
    let merged = oracle.merge(&candidates);

    assert_eq!(
        merged[0].doc_id, "d3",
        "AC 1: relevant doc must occupy rank 0; got merged = {merged:?}"
    );
    let max_natural_score = merged
        .iter()
        .skip(1)
        .map(|h| h.score)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        merged[0].score > max_natural_score,
        "AC 1: oracle hit must outscore every natural hit so post-sort keeps it at rank 0; \
         got oracle.score = {} vs max_natural_score = {max_natural_score}",
        merged[0].score
    );
}

// T-052-102: oracle_merge_dedupes_relevant_doc_appearing_in_inner_result
//
// When a relevant doc is also produced by WeightedRrf, the oracle hit
// takes the rank-0 slot and the natural duplicate is dropped — a single
// relevant doc must not count twice in downstream metrics.
#[test]
fn oracle_merge_dedupes_relevant_doc_appearing_in_inner_result() {
    let candidates = vec![
        fts_candidate("d1", -0.5, 0),
        fts_candidate("d2", -0.6, 1),
        fts_candidate("d3", -0.7, 2),
    ];
    let oracle = OracleMerge::new(HybridSearchConfig::default(), vec!["d1".to_owned()]);
    let merged = oracle.merge(&candidates);

    assert_eq!(merged[0].doc_id, "d1", "rank 0 must be the oracle slot");
    let d1_count = merged.iter().filter(|h| h.doc_id == "d1").count();
    assert_eq!(
        d1_count, 1,
        "duplicate relevant doc must be dropped from the natural tail; got {d1_count}"
    );
}

// T-052-103: oracle_merge_preserves_natural_order_for_non_relevant_docs
//
// The contract carved in the issue body: "配置済み doc 以外の順位は
// 元の WeightedRrf 結果を維持する". After dropping the oracle slot,
// the remaining hits must appear in the same order WeightedRrf would
// have emitted them.
#[test]
fn oracle_merge_preserves_natural_order_for_non_relevant_docs() {
    let candidates = vec![
        fts_candidate("d1", -0.5, 0),
        fts_candidate("d2", -0.6, 1),
        fts_candidate("d3", -0.7, 2),
    ];
    let baseline = WeightedRrf::new(HybridSearchConfig::default()).merge(&candidates);
    let oracle = OracleMerge::new(HybridSearchConfig::default(), vec!["d3".to_owned()]);
    let merged = oracle.merge(&candidates);

    let baseline_non_d3: Vec<&str> = baseline
        .iter()
        .filter(|h| h.doc_id != "d3")
        .map(|h| h.doc_id.as_str())
        .collect();
    let oracle_non_d3: Vec<&str> = merged
        .iter()
        .filter(|h| h.doc_id != "d3")
        .map(|h| h.doc_id.as_str())
        .collect();
    assert_eq!(
        baseline_non_d3, oracle_non_d3,
        "non-relevant docs must keep their natural ordering"
    );
}

// T-052-104: oracle_merge_emits_parent_granular_hits_with_empty_source_scores
//
// Every oracle hit is parent-granular (`chunk_id = None`) so Stage 3
// aggregators that bucket by `(doc_id, chunk_id)` cannot collapse it
// against a sibling chunk. `source_scores` is empty because no real
// retriever contributed.
#[test]
fn oracle_merge_emits_parent_granular_hits_with_empty_source_scores() {
    let candidates = vec![fts_candidate("d1", -0.5, 0)];
    let oracle = OracleMerge::new(HybridSearchConfig::default(), vec!["d2".to_owned()]);
    let merged = oracle.merge(&candidates);

    assert!(
        merged[0].chunk_id.is_none(),
        "oracle hit must be parent-granular; got chunk_id = {:?}",
        merged[0].chunk_id
    );
    assert!(
        merged[0].source_scores.is_empty(),
        "oracle hit has no contributing retriever — source_scores must be empty; got {:?}",
        merged[0].source_scores
    );
}

// T-052-105: oracle_merge_orders_multiple_relevant_docs_by_input_sequence
//
// When a query has multiple relevant docs, the prepended slots use
// strictly descending synthetic scores so the input order survives
// through any post-merge sort.
#[test]
fn oracle_merge_orders_multiple_relevant_docs_by_input_sequence() {
    let candidates = vec![fts_candidate("d99", -0.5, 0)];
    let oracle = OracleMerge::new(
        HybridSearchConfig::default(),
        vec!["d1".to_owned(), "d2".to_owned(), "d3".to_owned()],
    );
    let merged = oracle.merge(&candidates);

    assert_eq!(
        merged
            .iter()
            .take(3)
            .map(|h| h.doc_id.as_str())
            .collect::<Vec<_>>(),
        vec!["d1", "d2", "d3"],
        "multi-relevant order must follow the input sequence"
    );
    assert!(
        merged[0].score > merged[1].score && merged[1].score > merged[2].score,
        "synthetic scores must be strictly descending; got {} {} {}",
        merged[0].score,
        merged[1].score,
        merged[2].score
    );
}

// T-052-106: oracle_merge_passthrough_when_no_relevant_docs
//
// Empty `relevant_docs` short-circuits to the WeightedRrf output
// verbatim — guards the `evaluate_oracle` path against a query whose
// relevance_map happens to be empty (defensive: such a query would not
// contribute to recall@k anyway, but it must not crash the pipeline).
#[test]
fn oracle_merge_passthrough_when_no_relevant_docs() {
    let candidates = vec![fts_candidate("d1", -0.5, 0), fts_candidate("d2", -0.6, 1)];
    let baseline = WeightedRrf::new(HybridSearchConfig::default()).merge(&candidates);
    let oracle = OracleMerge::new(HybridSearchConfig::default(), Vec::new());
    let merged = oracle.merge(&candidates);

    assert_eq!(
        merged, baseline,
        "empty relevant_docs must short-circuit to WeightedRrf output"
    );
}

// T-052-107: relevant_doc_ids_filters_by_grade_and_sorts_deterministically
//
// `relevant_doc_ids` drops grade=0 entries (treated as "not relevant"
// by amici metrics) and sorts the surviving ids. Sorting matters
// because HashMap iteration order is hasher-dependent — without it the
// baseline.json byte content would drift across runs.
#[test]
fn relevant_doc_ids_filters_by_grade_and_sorts_deterministically() {
    let mut relevance_map = HashMap::new();
    relevance_map.insert("z9".to_owned(), 0u8);
    relevance_map.insert("d2".to_owned(), 1u8);
    relevance_map.insert("a1".to_owned(), 3u8);
    let query = EvalQuery {
        id: "q1".to_owned(),
        text: "stub".to_owned(),
        category: "C1".to_owned(),
        relevance_map,
        annotation: "stub".to_owned(),
    };
    let ids = relevant_doc_ids(&query);
    assert_eq!(
        ids,
        vec!["a1".to_owned(), "d2".to_owned()],
        "must drop grade=0 and sort ascending"
    );
}

// T-052-201: evaluate_oracle_rejects_relevance_map_pointing_outside_corpus
//
// Issue #52 P1 (advisor): if a query's relevance_map names a doc_id
// that the corpus lacks, `apply_reranker` would silently drop it via
// `filter_map(corpus_index.get(...))`. That would let the AC 4 gate
// pass vacuously when a fixture typo dropped the answer. evaluate_oracle
// must surface this as a typed error before pipeline setup.
#[test]
fn evaluate_oracle_rejects_relevance_map_pointing_outside_corpus() {
    use rurico::embed::MockEmbedder;
    use rurico::reranker::MockReranker;
    use rurico::retrieval::IdentityAggregator;

    let corpus = vec![EvalDocument {
        id: "d1".to_owned(),
        title: "title".to_owned(),
        body: "body for d1".to_owned(),
        category_hint: None,
        source: "test".to_owned(),
    }];
    let mut relevance_map = HashMap::new();
    relevance_map.insert("d1".to_owned(), 1u8);
    relevance_map.insert("doc_that_does_not_exist".to_owned(), 1u8);
    let queries = vec![EvalQuery {
        id: "q1".to_owned(),
        text: "body".to_owned(),
        category: "C1".to_owned(),
        relevance_map,
        annotation: "stub".to_owned(),
    }];
    let embedder = MockEmbedder::default();
    let reranker: Option<&MockReranker> = None;
    let aggregator = IdentityAggregator;
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let result = evaluate_oracle(
        &corpus,
        &queries,
        &embedder,
        reranker,
        &aggregator,
        &merge_config,
        &normalization,
        &config,
    );

    assert!(
        matches!(
            result,
            Err(OracleError::UnknownRelevantDoc { ref query_id, ref doc_id })
                if query_id == "q1" && doc_id == "doc_that_does_not_exist"
        ),
        "must surface UnknownRelevantDoc {{ query_id=q1, doc_id=doc_that_does_not_exist }}; got: {result:?}"
    );
}

// T-052-202: evaluate_oracle_places_relevant_doc_at_top_under_mock_pipeline
//
// End-to-end proof that OracleMerge ↔ pipeline composition surfaces the
// relevant doc at rank 0 of `ranked_hits` even when the upstream
// retrieval order would have buried it. Uses MockEmbedder so the test
// runs in the default cargo lane (no MLX).
#[test]
fn evaluate_oracle_places_relevant_doc_at_top_under_mock_pipeline() {
    use rurico::embed::MockEmbedder;
    use rurico::reranker::MockReranker;
    use rurico::retrieval::IdentityAggregator;

    let corpus = vec![
        EvalDocument {
            id: "d1".to_owned(),
            title: "title-d1".to_owned(),
            body: "alpha document about retrieval".to_owned(),
            category_hint: None,
            source: "test".to_owned(),
        },
        EvalDocument {
            id: "d2".to_owned(),
            title: "title-d2".to_owned(),
            body: "beta document about ranking".to_owned(),
            category_hint: None,
            source: "test".to_owned(),
        },
        EvalDocument {
            id: "d3".to_owned(),
            title: "title-d3".to_owned(),
            body: "gamma document about indexing".to_owned(),
            category_hint: None,
            source: "test".to_owned(),
        },
    ];
    let mut relevance_map = HashMap::new();
    // d3 is far from "alpha retrieval" linguistically — without
    // OracleMerge it would not surface at rank 0.
    relevance_map.insert("d3".to_owned(), 1u8);
    let queries = vec![EvalQuery {
        id: "q1".to_owned(),
        text: "alpha retrieval".to_owned(),
        category: "C1".to_owned(),
        relevance_map,
        annotation: "stub".to_owned(),
    }];
    let embedder = MockEmbedder::default();
    let reranker: Option<&MockReranker> = None;
    let aggregator = IdentityAggregator;
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let results = evaluate_oracle(
        &corpus,
        &queries,
        &embedder,
        reranker,
        &aggregator,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("oracle pipeline must succeed under MockEmbedder + no reranker");

    assert_eq!(results.len(), 1, "one query in → one QueryResult out");
    let top = results[0]
        .ranked_hits
        .first()
        .expect("ranked_hits must be non-empty");
    assert_eq!(
        top.doc_id, "d3",
        "AC 1: oracle path must surface d3 at rank 0; got ranked_hits = {:?}",
        results[0].ranked_hits
    );
}

// T-052-203: evaluate_oracle_uses_per_query_relevant_docs
//
// Two queries with different relevance_maps must each surface their
// own relevant doc at rank 0 — proves the per-query OracleMerge
// construction (vs. a single shared instance) wires the right
// relevant_docs through the pipeline loop.
#[test]
fn evaluate_oracle_uses_per_query_relevant_docs() {
    use rurico::embed::MockEmbedder;
    use rurico::reranker::MockReranker;
    use rurico::retrieval::IdentityAggregator;

    let corpus = vec![
        EvalDocument {
            id: "d1".to_owned(),
            title: "t1".to_owned(),
            body: "alpha".to_owned(),
            category_hint: None,
            source: "test".to_owned(),
        },
        EvalDocument {
            id: "d2".to_owned(),
            title: "t2".to_owned(),
            body: "beta".to_owned(),
            category_hint: None,
            source: "test".to_owned(),
        },
    ];
    let queries = vec![
        EvalQuery {
            id: "qa".to_owned(),
            text: "alpha".to_owned(),
            category: "C1".to_owned(),
            relevance_map: {
                let mut m = HashMap::new();
                m.insert("d1".to_owned(), 1u8);
                m
            },
            annotation: "stub".to_owned(),
        },
        EvalQuery {
            id: "qb".to_owned(),
            text: "beta".to_owned(),
            category: "C1".to_owned(),
            relevance_map: {
                let mut m = HashMap::new();
                m.insert("d2".to_owned(), 1u8);
                m
            },
            annotation: "stub".to_owned(),
        },
    ];
    let embedder = MockEmbedder::default();
    let reranker: Option<&MockReranker> = None;
    let aggregator = IdentityAggregator;
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let results = evaluate_oracle(
        &corpus,
        &queries,
        &embedder,
        reranker,
        &aggregator,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("per-query oracle pipeline must succeed");

    assert_eq!(results[0].ranked_hits[0].doc_id, "d1", "qa → d1 at rank 0");
    assert_eq!(results[1].ranked_hits[0].doc_id, "d2", "qb → d2 at rank 0");
}
