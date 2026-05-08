//! Oracle retrieval pipeline (Issue #52).
//!
//! Reuses `pipeline::evaluate` wiring with a per-query [`OracleMerge`] that
//! force-places known-relevant documents at rank 0 of the Stage 2 output.
//! Quantifies the search-side bottleneck so amici can tell apart "rerank /
//! aggregation is leaving recall on the table" from "retrieval can't surface
//! the right doc in the first place" (CoSearch-style retrieval ceiling).

use std::collections::HashMap;
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
        let inner_max = inner_hits.iter().map(|h| h.score).fold(0.0_f64, f64::max);
        let mut output: Vec<MergedHit> =
            Vec::with_capacity(self.relevant_docs.len() + inner_hits.len());
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
        // Linear scan against `relevant_docs` (typical len 1–5) is cheaper
        // than a per-call `HashSet` allocation and keeps the merge body
        // free of intermediate state.
        for hit in inner_hits {
            if !self.relevant_docs.iter().any(|r| r == &hit.doc_id) {
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
    // Build the (doc_id → body) lookup once: it doubles as the corpus
    // membership test for `validate_relevance_map_against_corpus` and as
    // the rerank-stage body lookup for `run_single_query`. A separate
    // `HashSet<&str>` would re-walk `corpus` for the same membership check.
    let corpus_index: HashMap<&str, &str> = corpus
        .iter()
        .map(|d| (d.id.as_str(), d.body.as_str()))
        .collect();
    validate_relevance_map_against_corpus(queries, &corpus_index)?;

    let conn = setup_pipeline_connection(corpus, embedder, normalization)?;

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

/// Reject any query whose `relevance_map` references a `doc_id` absent
/// from `corpus_index`. Surfaces [`OracleError::UnknownRelevantDoc`]
/// before pipeline setup so `apply_reranker`'s silent `filter_map` cannot
/// drop the answer downstream.
fn validate_relevance_map_against_corpus(
    queries: &[EvalQuery],
    corpus_index: &HashMap<&str, &str>,
) -> Result<(), OracleError> {
    for query in queries {
        for doc_id in query.relevance_map.keys() {
            if !corpus_index.contains_key(doc_id.as_str()) {
                return Err(OracleError::UnknownRelevantDoc {
                    query_id: query.id.clone(),
                    doc_id: doc_id.clone(),
                });
            }
        }
    }
    Ok(())
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
mod tests;
