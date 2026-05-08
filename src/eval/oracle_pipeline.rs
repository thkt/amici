//! Oracle retrieval pipeline (Issue #52).
//!
//! Reuses `pipeline::evaluate` wiring with a per-query [`OracleMerge`] that
//! force-places known-relevant documents at rank 0 of the Stage 2 output.
//! Quantifies the search-side bottleneck so amici can tell apart "rerank /
//! aggregation is leaving recall on the table" from "retrieval can't surface
//! the right doc in the first place" (CoSearch-style retrieval ceiling).
//!
//! Scope (Issue #52 AC 1, 2, 5, 6):
//! - [`OracleMerge`] implements [`MergeStrategy`] and prepends every doc in
//!   `relevant_docs` to the wrapped [`WeightedRrf`] output, dedup-merging
//!   any natural occurrences so a single relevant doc cannot count twice.
//! - [`evaluate_oracle`] constructs one [`OracleMerge`] per query from the
//!   query's `relevance_map` (graded ≥ 1) and runs the same pipeline shape
//!   as [`crate::eval::pipeline::evaluate`].
//! - Pre-validates that every relevance_map doc_id exists in the corpus so
//!   `apply_reranker`'s silent `filter_map(corpus_index.get(...))` never
//!   eats a missing answer (would otherwise let a typo in the fixture pass
//!   the AC 4 gate vacuously).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rurico::embed::Embed;
use rurico::reranker::Rerank;
use rurico::retrieval::{
    Aggregator, Candidate, HybridSearchConfig, MergeStrategy, MergedHit, WeightedRrf,
};
use rurico::storage::QueryNormalizationConfig;

use crate::eval::fixture::{EvalDocument, EvalQuery};
use crate::eval::pipeline::{
    PipelineConfig, PipelineError, QueryResult, run_single_query, setup_pipeline_connection,
};

/// Stage 2 strategy that force-injects known-relevant docs at the top of
/// the merged ranking.
///
/// Wraps [`WeightedRrf`] so the natural fusion order is preserved for every
/// non-relevant doc; the relevant doc list is **prepended** with synthetic
/// scores `inner_max + (relevant_docs.len() - i)` so the trait's
/// "score-descending sorted output" contract holds without re-sorting the
/// inner result.
///
/// Synthetic scores are anchored to `inner_max` (not a bare `f64::MAX`)
/// because subtracting small integers from `f64::MAX` saturates to the same
/// value at the precision limit — `f64::MAX - 1.0 == f64::MAX - 2.0 ==
/// f64::MAX` — which would silently collapse the relevant-doc ordering.
///
/// The synthetic [`MergedHit`]s use `chunk_id = None` (parent-granular) and
/// an empty `source_scores` map — Stage 3 aggregators that bucket by
/// `(doc_id, chunk_id)` therefore see oracle hits as pure parent rows and
/// cannot collapse them against sibling chunks. `evaluate_oracle` is the
/// only documented constructor path; no production retrieval surface uses
/// this strategy.
pub struct OracleMerge {
    inner: WeightedRrf,
    relevant_docs: Vec<String>,
}

impl OracleMerge {
    /// Build an oracle merge for one query.
    ///
    /// `relevant_docs` is the deduplicated list of `doc_id`s whose
    /// relevance grade is ≥ 1; ordering inside the slice is preserved in
    /// the emitted ranking when multiple relevant docs land at the top.
    #[must_use]
    pub fn new(merge_config: HybridSearchConfig, relevant_docs: Vec<String>) -> Self {
        Self {
            inner: WeightedRrf::new(merge_config),
            relevant_docs,
        }
    }
}

impl MergeStrategy for OracleMerge {
    fn merge(&self, candidates: &[Candidate]) -> Vec<MergedHit> {
        let inner_hits = self.inner.merge(candidates);
        if self.relevant_docs.is_empty() {
            return inner_hits;
        }
        let relevant: HashSet<&str> = self.relevant_docs.iter().map(String::as_str).collect();
        let inner_max = inner_hits.iter().map(|h| h.score).fold(0.0_f64, f64::max);
        let mut output: Vec<MergedHit> =
            Vec::with_capacity(self.relevant_docs.len() + inner_hits.len());
        // Anchor synthetic scores to `inner_max` so the smallest oracle
        // score still strictly exceeds every natural hit. The descending
        // `(len - i) + 1.0` offset keeps relevant_docs[0] highest and
        // preserves input order under any subsequent sort.
        let count = self.relevant_docs.len();
        for (i, doc_id) in self.relevant_docs.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let offset = (count - i) as f64;
            let score = inner_max + offset;
            output.push(MergedHit {
                doc_id: doc_id.clone(),
                chunk_id: None,
                score,
                source_scores: HashMap::new(),
            });
        }
        for hit in inner_hits {
            if !relevant.contains(hit.doc_id.as_str()) {
                output.push(hit);
            }
        }
        output
    }
}

/// Errors surfaced by [`evaluate_oracle`].
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum OracleError {
    /// Underlying pipeline failure (storage, embed, rerank, sanitize).
    #[error("oracle pipeline error: {0}")]
    Pipeline(#[from] PipelineError),
    /// A query's `relevance_map` referenced a `doc_id` absent from the
    /// corpus. Surfaced eagerly so the AC 4 gate ("oracle recall@k ≥
    /// baseline recall@k for every category") cannot pass vacuously when a
    /// fixture typo silently drops the answer in `apply_reranker`.
    #[error(
        "oracle: query {query_id:?} references doc_id {doc_id:?} in relevance_map but the corpus has no such document"
    )]
    UnknownRelevantDoc {
        /// The query whose relevance_map is malformed.
        query_id: String,
        /// The doc_id that does not exist in the supplied corpus.
        doc_id: String,
    },
}

/// Run the reference pipeline with a per-query [`OracleMerge`] and the
/// supplied aggregator + reranker.
///
/// Mirrors [`crate::eval::pipeline::evaluate`]'s shape but routes Stage 2
/// through [`OracleMerge::new`] so every relevant doc lands at rank 0
/// **before** Stage 3 aggregation and Stage 4 rerank. The downstream stages
/// run unmodified — measuring the post-retrieval ceiling under the
/// production rerank/aggregation behaviour rather than the
/// "everything-is-perfect" upper bound.
///
/// # Errors
///
/// Returns [`OracleError::UnknownRelevantDoc`] when any query's
/// `relevance_map` references a `doc_id` outside `corpus`, and
/// [`OracleError::Pipeline`] for downstream pipeline failures.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_oracle<E, R, A>(
    corpus: &[EvalDocument],
    queries: &[EvalQuery],
    embedder: &E,
    reranker: Option<&R>,
    aggregator: &A,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
    config: &PipelineConfig,
) -> Result<Vec<QueryResult>, OracleError>
where
    E: Embed,
    R: Rerank,
    A: Aggregator,
{
    let corpus_ids: HashSet<&str> = corpus.iter().map(|d| d.id.as_str()).collect();
    for query in queries {
        for doc_id in query.relevance_map.keys() {
            if !corpus_ids.contains(doc_id.as_str()) {
                return Err(OracleError::UnknownRelevantDoc {
                    query_id: query.id.clone(),
                    doc_id: doc_id.clone(),
                });
            }
        }
    }

    let conn = setup_pipeline_connection(corpus, embedder, normalization)?;

    let corpus_index: HashMap<&str, &str> = corpus
        .iter()
        .map(|d| (d.id.as_str(), d.body.as_str()))
        .collect();

    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        let relevant_docs = relevant_doc_ids(query);
        let oracle = OracleMerge::new(merge_config.clone(), relevant_docs);
        let started = Instant::now();
        let ranked_hits = run_single_query(
            &conn,
            query,
            embedder,
            reranker,
            aggregator,
            &oracle,
            &corpus_index,
            normalization,
            config,
        )?;
        let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        results.push(QueryResult {
            query_id: query.id.clone(),
            ranked_hits,
            latency_ms,
        });
    }
    Ok(results)
}

/// Project `query.relevance_map` to the deduplicated list of `doc_id`s
/// whose grade is ≥ 1, preserving a deterministic ordering across runs.
///
/// `HashMap` iteration order is unstable across hashers — sort by `doc_id`
/// so two captures of the same fixture produce bit-identical baselines.
fn relevant_doc_ids(query: &EvalQuery) -> Vec<String> {
    let mut ids: Vec<String> = query
        .relevance_map
        .iter()
        .filter(|(_, grade)| **grade >= 1)
        .map(|(doc_id, _)| doc_id.clone())
        .collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    use rurico::retrieval::CandidateSource;

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
    // Issue #52 AC 1: OracleMerge forces a relevant doc to rank 0 of the
    // merged output even when the underlying WeightedRrf would have ranked
    // it lower. The synthetic `score = f64::MAX` keeps the slot fixed under
    // any post-merge sort.
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
        use crate::eval::fixture::EvalQuery;
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
}
