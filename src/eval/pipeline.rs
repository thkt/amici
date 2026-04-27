//! Reference search pipeline composition (FR-008..FR-010).
//!
//! Inline composition of `rurico` primitives — in-memory SQLite + FTS5 +
//! sqlite-vec + RRF + optional reranker — modeled after `recall`'s wiring
//! shape (ADR 0002). The trigram adapter `clean_for_trigram` lives in the
//! `amici` production code (`crate::storage::fts`) and is called directly,
//! so the harness baseline measures the production wiring without a mirror.

use std::collections::HashMap;
use std::time::Instant;

use rurico::embed::{EMBEDDING_DIMS, Embed, EmbedError};
use rurico::reranker::{Rerank, RerankerError};
use rurico::retrieval::{
    Aggregator, Candidate, CandidateSource, HybridSearchConfig, MergeStrategy, MergedHit,
    WeightedRrf,
};
use rurico::storage::{
    QueryNormalizationConfig, SanitizeError, ensure_sqlite_vec, f32_as_bytes, normalize_for_fts,
    prepare_match_query,
};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::eval::fixture::{EvalDocument, EvalQuery};
use crate::storage::fts::clean_for_trigram;

/// FTS5 vocab table name used by [`prepare_match_query`].
const FTS_VOCAB_TABLE: &str = "docs_vocab";

/// FTS + vec retrievals each fetch this many candidates per query before
/// RRF; matches recall's `opts.limit * 3` heuristic.
const RRF_CANDIDATE_MULTIPLIER: usize = 3;

/// One query's pipeline output: ordered hits + wall-clock latency.
///
/// `ranked_hits` reuses [`MergedHit`] — the same type passed across the Stage
/// 3 boundary — so the rerank tail no longer round-trips through a separate
/// `Hit` shape. Downstream callers should still treat the slice as sorted by
/// descending `score`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Identifier of the input [`EvalQuery`].
    pub query_id: String,
    /// Hits sorted by descending [`MergedHit::score`].
    pub ranked_hits: Vec<MergedHit>,
    /// Wall-clock latency for this single query in milliseconds.
    pub latency_ms: u64,
}

/// Tunable pipeline parameters.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Top-k cutoff after RRF merge.
    pub k: usize,
}

/// Errors surfaced by the reference pipeline.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum PipelineError {
    /// SQLite storage failure (schema build, FTS index, vec table, etc.).
    #[error("pipeline sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// Embedder failed during query or document encoding.
    #[error("pipeline embed error: {0}")]
    Embed(#[from] EmbedError),
    /// Optional reranker failed during scoring.
    #[error("pipeline rerank error: {0}")]
    Rerank(#[from] RerankerError),
    /// FTS5 query sanitization rejected the surface query.
    #[error("pipeline FTS sanitize error: {0}")]
    Sanitize(#[from] SanitizeError),
    /// `sqlite_vec` extension registration failed at process level.
    #[error("pipeline sqlite-vec load failed: {0}")]
    SqliteVec(String),
    /// Embedder returned a chunk count that does not match the corpus.
    #[error(
        "pipeline embed batch size mismatch: corpus has {corpus} docs, embedder returned {chunked}"
    )]
    ChunkCountMismatch {
        /// Number of documents in the corpus passed to the embedder.
        corpus: usize,
        /// Number of [`rurico::embed::ChunkedEmbedding`] entries returned.
        chunked: usize,
    },
    /// Embedder produced zero chunks for a corpus document.
    #[error("pipeline empty embedding for doc {doc_id:?}")]
    EmptyEmbedding {
        /// Identifier of the corpus document with no chunks.
        doc_id: String,
    },
    /// Embedder returned a [`ChunkedEmbedding`](rurico::embed::ChunkedEmbedding)
    /// whose `chunks` and `chunk_ids` vectors disagree on length.
    ///
    /// `ChunkedEmbedding::new` upholds the invariant, but the fields are
    /// `pub` and a custom impl could break it. Without this guard the
    /// per-chunk `zip` would silently stop at the shorter vector and either
    /// drop chunks (recall regression) or insert zero rows for a doc with
    /// chunks-but-no-ids (vec table mismatch).
    #[error(
        "pipeline chunk_id metadata length mismatch for doc {doc_id:?}: \
         {chunks} chunks vs {chunk_ids} chunk_ids"
    )]
    ChunkIdLengthMismatch {
        /// Identifier of the corpus document with mismatched metadata.
        doc_id: String,
        /// `chunks.len()` reported by the embedder.
        chunks: usize,
        /// `chunk_ids.len()` reported by the embedder.
        chunk_ids: usize,
    },
}

/// Run the reference pipeline on `corpus` for every query in `queries`.
///
/// Indexes `corpus` into an in-memory SQLite store with FTS5 and sqlite-vec
/// virtual tables, encodes each query with `embedder`, retrieves top hits via
/// FTS5 and vector search, merges with RRF, and optionally rescores with
/// `reranker`. Output ranked-hit count is bounded by [`PipelineConfig::k`].
///
/// # Errors
///
/// See [`PipelineError`] variants. Sqlite, embed, rerank, and sanitize errors
/// each surface their respective source via `#[from]`; sqlite-vec load
/// failure surfaces as [`PipelineError::SqliteVec`].
#[allow(clippy::too_many_arguments)]
pub fn evaluate<E, R, A>(
    corpus: &[EvalDocument],
    queries: &[EvalQuery],
    embedder: &E,
    reranker: Option<&R>,
    aggregator: &A,
    merge_config: &HybridSearchConfig,
    normalization: &QueryNormalizationConfig,
    config: &PipelineConfig,
) -> Result<Vec<QueryResult>, PipelineError>
where
    E: Embed,
    R: Rerank,
    A: Aggregator,
{
    ensure_sqlite_vec().map_err(PipelineError::SqliteVec)?;
    let conn = Connection::open_in_memory()?;
    create_schema(&conn)?;
    index_corpus(&conn, corpus, embedder, normalization)?;

    // Build (doc_id → body) lookup once before the query loop. apply_reranker
    // does O(1) hits against this map instead of an O(N) corpus.iter().find()
    // per merged hit per query.
    let corpus_index: HashMap<&str, &str> = corpus
        .iter()
        .map(|d| (d.id.as_str(), d.body.as_str()))
        .collect();

    // Build the merge strategy once outside the query loop — its config
    // does not vary per query.
    let merge_strategy = WeightedRrf::new(merge_config.clone());

    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        let started = Instant::now();
        let ranked_hits = run_single_query(
            &conn,
            query,
            embedder,
            reranker,
            aggregator,
            &merge_strategy,
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

/// Build the in-memory schema (documents + FTS5 + vec0 + fts5vocab).
///
/// `vec_docs` carries an extra `chunk_id` metadata column so chunk-level
/// vector retrieval can surface distinct child chunks of the same parent
/// doc. FTS stays parent-granular because per-chunk text is not stored on
/// [`rurico::embed::ChunkedEmbedding`]; the vector source alone is sufficient
/// to drive Stage 3 aggregation non-vacuously.
fn create_schema(conn: &Connection) -> Result<(), PipelineError> {
    conn.execute_batch(&format!(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, body TEXT NOT NULL); \
         CREATE VIRTUAL TABLE docs_fts USING fts5(doc_id UNINDEXED, body, tokenize='trigram'); \
         CREATE VIRTUAL TABLE vec_docs USING vec0(embedding FLOAT[{EMBEDDING_DIMS}], +doc_id TEXT, +chunk_id TEXT); \
         CREATE VIRTUAL TABLE {FTS_VOCAB_TABLE} USING fts5vocab(docs_fts, row);"
    ))?;
    Ok(())
}

/// Encode each document body with `embedder` and insert into all three tables.
///
/// FTS / `documents` are parent-granular (one row per [`EvalDocument`]);
/// `vec_docs` is chunk-granular — each chunk vector lands in its own row
/// tagged with `(doc_id, chunk_id)` so chunk-level retrieval can surface
/// multiple chunks of the same parent. Returns
/// [`PipelineError::ChunkCountMismatch`] when the embedder's output length
/// does not match the corpus and [`PipelineError::EmptyEmbedding`] when a
/// document yields no chunks — silent truncation of the vec index would
/// degrade recall without a visible error.
///
/// `normalization` is applied to the **FTS-indexed body only**. The
/// `documents` table keeps the original body so display surfaces see
/// un-normalized text; the embedder receives the original body because
/// SentencePiece performs NFKC internally and double application would
/// only add allocator pressure.
fn index_corpus<E: Embed>(
    conn: &Connection,
    corpus: &[EvalDocument],
    embedder: &E,
    normalization: &QueryNormalizationConfig,
) -> Result<(), PipelineError> {
    let bodies: Vec<&str> = corpus.iter().map(|d| d.body.as_str()).collect();
    let chunked = embedder.embed_documents_batch(&bodies)?;
    if chunked.len() != corpus.len() {
        return Err(PipelineError::ChunkCountMismatch {
            corpus: corpus.len(),
            chunked: chunked.len(),
        });
    }
    // Wrap inserts in a single transaction. Each chunk now generates its own
    // vec_docs row, so the implicit per-statement commit count grew from
    // `N×3` to `N×2 + N×chunks_per_doc`. Batching keeps WAL fsync overhead
    // bounded as fixture size grows.
    let tx = conn.unchecked_transaction()?;
    {
        let mut insert_doc = tx.prepare_cached("INSERT INTO documents(id, body) VALUES (?, ?)")?;
        let mut insert_fts =
            tx.prepare_cached("INSERT INTO docs_fts(doc_id, body) VALUES (?, ?)")?;
        let mut insert_vec = tx
            .prepare_cached("INSERT INTO vec_docs(embedding, doc_id, chunk_id) VALUES (?, ?, ?)")?;
        for (doc, chunked_embedding) in corpus.iter().zip(chunked.iter()) {
            insert_doc.execute(params![&doc.id, &doc.body])?;
            let fts_body = normalize_for_fts(&doc.body, normalization);
            insert_fts.execute(params![&doc.id, &fts_body])?;
            if chunked_embedding.chunks.is_empty() {
                return Err(PipelineError::EmptyEmbedding {
                    doc_id: doc.id.clone(),
                });
            }
            if chunked_embedding.chunks.len() != chunked_embedding.chunk_ids.len() {
                return Err(PipelineError::ChunkIdLengthMismatch {
                    doc_id: doc.id.clone(),
                    chunks: chunked_embedding.chunks.len(),
                    chunk_ids: chunked_embedding.chunk_ids.len(),
                });
            }
            for (chunk_vec, chunk_id) in chunked_embedding
                .chunks
                .iter()
                .zip(&chunked_embedding.chunk_ids)
            {
                insert_vec.execute(params![f32_as_bytes(chunk_vec), &doc.id, chunk_id])?;
            }
        }
    }
    tx.commit()?;
    Ok(())
}

/// Drive one `EvalQuery` through FTS + vec retrieval, RRF merge, Stage 3
/// aggregation, and (when supplied) reranker rescoring.
#[allow(clippy::too_many_arguments)]
fn run_single_query<E, R, A, M>(
    conn: &Connection,
    query: &EvalQuery,
    embedder: &E,
    reranker: Option<&R>,
    aggregator: &A,
    merge_strategy: &M,
    corpus_index: &HashMap<&str, &str>,
    normalization: &QueryNormalizationConfig,
    config: &PipelineConfig,
) -> Result<Vec<MergedHit>, PipelineError>
where
    E: Embed,
    R: Rerank,
    A: Aggregator,
    M: MergeStrategy,
{
    let candidate_limit = config.k * RRF_CANDIDATE_MULTIPLIER;
    let fts_hits = retrieve_fts(conn, &query.text, candidate_limit, normalization)?;
    let vec_hits = retrieve_vec(conn, embedder, &query.text, candidate_limit)?;
    let mut all_candidates = Vec::with_capacity(fts_hits.len() + vec_hits.len());
    all_candidates.extend(fts_hits);
    all_candidates.extend(vec_hits);
    let merged_hits = merge_strategy.merge(&all_candidates);

    // Aggregator::aggregate guarantees score-descending output (see trait
    // doc), so truncate-then-rerank does not silently drop higher-scoring
    // hits.
    let mut aggregated = aggregator.aggregate(&merged_hits);
    aggregated.truncate(config.k);

    if let Some(reranker) = reranker {
        aggregated = apply_reranker(reranker, &query.text, aggregated, corpus_index)?;
    }

    Ok(aggregated)
}

/// FTS5 retrieval. Empty / unsanitisable queries return an empty hit list
/// (mirrors recall's early-return behavior on `SanitizeError::EmptyInput`).
///
/// Returns Stage 1 [`Candidate`]s tagged with [`CandidateSource::Fts`]. The
/// `score` field carries SQLite FTS5's negative BM25 (lower magnitude is
/// better) and `rank` is 0-based — fed verbatim to [`WeightedRrf`].
///
/// FTS hits carry `chunk_id = None` because the FTS index is parent-granular
/// (per-chunk text is not stored on [`rurico::embed::ChunkedEmbedding`]).
/// Stage 2 fuses on `(doc_id, None)` for FTS contributions and
/// `(doc_id, Some(c_i))` for vector contributions, so the two sources stay
/// distinguishable in [`MergedHit::source_scores`].
fn retrieve_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
    normalization: &QueryNormalizationConfig,
) -> Result<Vec<Candidate>, PipelineError> {
    let matched = match prepare_match_query(conn, query, FTS_VOCAB_TABLE, normalization) {
        Ok(m) => m,
        Err(SanitizeError::EmptyInput | SanitizeError::NoSearchableTerms) => {
            return Ok(Vec::new());
        }
        Err(e) => return Err(e.into()),
    };
    let Some(fts_query) = clean_for_trigram(&matched) else {
        return Ok(Vec::new());
    };
    let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut stmt = conn.prepare_cached(
        "SELECT doc_id, rank FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
    )?;
    let rows = stmt.query_map(params![fts_query, limit_i64], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
    })?;
    let mut hits = Vec::new();
    for (rank, row) in rows.enumerate() {
        let (doc_id, score) = row?;
        hits.push(Candidate {
            source: CandidateSource::Fts,
            doc_id,
            chunk_id: None,
            score,
            rank,
        });
    }
    Ok(hits)
}

/// Vector retrieval via the `vec0` virtual table's KNN operator.
///
/// Uses sqlite-vec's official `AND k = ?` syntax; rows already arrive in
/// distance-ascending order, so no `ORDER BY` clause is needed.
///
/// Returns Stage 1 [`Candidate`]s tagged with [`CandidateSource::Vector`].
/// The `score` field carries the raw distance (lower is better) and `rank`
/// is 0-based — fed verbatim to [`WeightedRrf`].
///
/// Each chunk of a parent doc is indexed as its own row tagged with
/// `(doc_id, chunk_id)`, so a single query can return multiple chunks of
/// the same parent. Stage 2 keeps them distinct via the `(doc_id, chunk_id)`
/// fusion key; Stage 3 aggregators collapse to the parent on their own
/// contract.
fn retrieve_vec<E: Embed>(
    conn: &Connection,
    embedder: &E,
    query: &str,
    limit: usize,
) -> Result<Vec<Candidate>, PipelineError> {
    let embedding = embedder.embed_query(query)?;
    let bytes = f32_as_bytes(&embedding);
    let k_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
    let mut stmt = conn.prepare_cached(
        "SELECT doc_id, chunk_id, distance FROM vec_docs WHERE embedding MATCH ? AND k = ?",
    )?;
    let rows = stmt.query_map(params![bytes, k_i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;
    let mut hits = Vec::new();
    for (rank, row) in rows.enumerate() {
        let (doc_id, chunk_id, score) = row?;
        hits.push(Candidate {
            source: CandidateSource::Vector,
            doc_id,
            chunk_id: Some(chunk_id),
            score,
            rank,
        });
    }
    Ok(hits)
}

/// Rescore `merged` via `reranker.rerank(query, doc_bodies)`.
///
/// Consumes `merged` and yields a fresh `Vec<MergedHit>` with rerank scores
/// in place of RRF scores. Filters by `corpus_index` membership before
/// scoring, so the reranker's returned `index` aligns with the same filtered
/// slice and no missing-from-corpus entry can misalign rerank output with
/// merged identity. Rerank score (`f32`) widens to `f64` to keep the merged
/// list type consistent.
///
/// **Parent-context posture:** the reranker receives each hit's *parent
/// doc body* — never just the chunk body — because `corpus_index` is keyed
/// on parent `doc_id` and stores parent text. Every sibling chunk of the
/// same parent maps to the same body, giving the reranker the full
/// document context regardless of which chunk surfaced through Stage 1+2.
/// A finer-grained `chunk + window` mode would require chunk text on
/// `ChunkedEmbedding`.
///
/// `chunk_id` is preserved through rerank so chunk-level Identity output
/// keeps its child-chunk identity in the final ranking; aggregator-collapsed
/// hits (chunk_id=None) stay parent-granular.
fn apply_reranker<R: Rerank>(
    reranker: &R,
    query: &str,
    merged: Vec<MergedHit>,
    corpus_index: &HashMap<&str, &str>,
) -> Result<Vec<MergedHit>, PipelineError> {
    // `Option<ResolvedSlot>` so each slot is consumed exactly once when the
    // reranker references its index — protects against a malformed reranker
    // that emits the same index twice from re-using the same hit identity.
    struct ResolvedSlot<'a> {
        doc_id: String,
        chunk_id: Option<String>,
        body: &'a str,
        source_scores: HashMap<CandidateSource, f64>,
    }
    let mut resolved: Vec<Option<ResolvedSlot<'_>>> = merged
        .into_iter()
        .filter_map(|h| {
            corpus_index.get(h.doc_id.as_str()).map(|body| {
                Some(ResolvedSlot {
                    doc_id: h.doc_id,
                    chunk_id: h.chunk_id,
                    body,
                    source_scores: h.source_scores,
                })
            })
        })
        .collect();
    let bodies: Vec<&str> = resolved
        .iter()
        .map(|slot| slot.as_ref().map(|s| s.body).unwrap_or(""))
        .collect();
    let ranked_results = reranker.rerank(query, &bodies)?;
    let mut reranked = Vec::with_capacity(ranked_results.len());
    for r in ranked_results {
        if let Some(slot) = resolved.get_mut(r.index)
            && let Some(s) = slot.take()
        {
            reranked.push(MergedHit {
                doc_id: s.doc_id,
                chunk_id: s.chunk_id,
                score: f64::from(r.score),
                source_scores: s.source_scores,
            });
        }
    }
    Ok(reranked)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use std::collections::HashSet;

    use rurico::embed::{MockChunkedEmbedder, MockEmbedder};
    use rurico::reranker::MockReranker;
    use rurico::retrieval::{
        HybridSearchConfig, IdentityAggregator, MaxChunkAggregator, MergedHit,
    };
    use rurico::storage::QueryNormalizationConfig;

    use super::{PipelineConfig, evaluate};
    use crate::eval::fixture::{EvalDocument, EvalQuery};

    /// Build an [`EvalDocument`] with stub title / source. The body carries
    /// the surface text the FTS5 + vec wiring will see.
    fn make_document(id: &str, body: &str) -> EvalDocument {
        EvalDocument {
            id: id.to_owned(),
            title: format!("title for {id}"),
            body: body.to_owned(),
            category_hint: None,
            source: "test fixture".to_owned(),
        }
    }

    /// Build an [`EvalQuery`] with a single-doc relevance map and stub
    /// category / annotation. Distribution validation is not exercised here.
    fn make_query(id: &str, text: &str, relevant_doc: &str) -> EvalQuery {
        let mut relevance_map = HashMap::new();
        relevance_map.insert(relevant_doc.to_owned(), 1u8);
        EvalQuery {
            id: id.to_owned(),
            text: text.to_owned(),
            category: "C1".to_owned(),
            relevance_map,
            annotation: "test query".to_owned(),
        }
    }

    // T-069-001: fts5_trigram_does_not_fold_fullwidth_latin
    //
    // The `variant_notation` fixture validity rests on FTS5's trigram
    // tokenizer NOT folding fullwidth Latin to ASCII on its own. Otherwise
    // the "baseline 以上" gate passes vacuously: queries hit with
    // normalization OFF and the metrics never move.
    //
    // Pinned via `tests/fixtures/eval/queries.jsonl` q-variant-notation-* —
    // if this test starts failing (SQLite upgrades the trigram tokenizer
    // to do Unicode case folding), the fixture must be reauthored.
    #[test]
    fn fts5_trigram_does_not_fold_fullwidth_latin() {
        use rusqlite::Connection;
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING fts5(body, tokenize='trigram');
             INSERT INTO t(body) VALUES ('rust ownership and borrow checker');",
        )
        .unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM t WHERE t MATCH ?",
                ["Ｒｕｓｔ"],
                |r| r.get(0),
            )
            .unwrap_or(0);
        assert_eq!(
            count, 0,
            "FTS5 trigram tokenizer folded fullwidth Latin Ｒｕｓｔ to ASCII rust; \
             variant_notation fixture queries now match without normalization, \
             so the baseline-以上 gate passes vacuously. Reauthor the fixture."
        );
    }

    // T-011: evaluate_with_mock_embedder_returns_one_result_with_hits
    // FR-008: 5-doc corpus + 1 query + MockEmbedder + no reranker →
    //         single QueryResult whose ranked_hits is non-empty.
    // FR-009: pipeline must compose `prepare_match_query`, `rrf_merge`,
    //         `Embed::embed_query`, optional `Rerank::rerank`.
    #[test]
    fn evaluate_with_mock_embedder_returns_one_result_with_hits() {
        let corpus = vec![
            make_document("d1", "alpha document about retrieval"),
            make_document("d2", "beta document about ranking"),
            make_document("d3", "gamma document about indexing"),
            make_document("d4", "delta document about scoring"),
            make_document("d5", "epsilon document about evaluation"),
        ];
        let queries = vec![make_query("q1", "alpha retrieval", "d1")];
        let embedder = MockEmbedder::default();
        let reranker: Option<&MockReranker> = None;
        let aggregator = IdentityAggregator;
        let merge_config = HybridSearchConfig::default();
        let normalization = QueryNormalizationConfig::default();
        let config = PipelineConfig { k: 5 };

        let result = evaluate(
            &corpus,
            &queries,
            &embedder,
            reranker,
            &aggregator,
            &merge_config,
            &normalization,
            &config,
        )
        .expect("pipeline must succeed with MockEmbedder + no reranker");

        assert_eq!(
            result.len(),
            1,
            "FR-008: one query in → exactly one QueryResult out, got {} results",
            result.len()
        );
        assert!(
            !result[0].ranked_hits.is_empty(),
            "FR-008: indexed corpus + valid query → ranked_hits must be non-empty, got empty"
        );
    }

    // T-076-007: chunk_level_pipeline_yields_distinct_chunks_under_identity
    //
    // End-to-end proof of the (doc_id, chunk_id) fusion key: with
    // `MockChunkedEmbedder::new(2)` every document contributes two vector
    // chunks. The Identity pipeline must surface both chunks of the same
    // parent in `ranked_hits` instead of fusing them at Stage 2 —
    // otherwise every aggregator collapses to identity (the exact failure
    // mode rurico ADR 0004 calls out).
    #[test]
    fn chunk_level_pipeline_yields_distinct_chunks_under_identity() {
        let corpus = vec![
            make_document("d1", "alpha document about retrieval"),
            make_document("d2", "beta document about ranking"),
            make_document("d3", "gamma document about indexing"),
        ];
        let queries = vec![make_query("q1", "alpha retrieval", "d1")];
        let embedder = MockChunkedEmbedder::new(2);
        let reranker: Option<&MockReranker> = None;
        let merge_config = HybridSearchConfig::default();
        let normalization = QueryNormalizationConfig::default();
        let config = PipelineConfig { k: 10 };

        let result = evaluate(
            &corpus,
            &queries,
            &embedder,
            reranker,
            &IdentityAggregator,
            &merge_config,
            &normalization,
            &config,
        )
        .expect("identity pipeline must succeed");

        let chunked_hits: Vec<&str> = result[0]
            .ranked_hits
            .iter()
            .filter(|h| h.chunk_id.is_some())
            .map(|h| h.doc_id.as_str())
            .collect();
        assert!(
            !chunked_hits.is_empty(),
            "vector hits must carry chunk_id under chunk-level retrieval; \
             got ranked_hits={:?}",
            result[0].ranked_hits
        );

        let mut per_parent: HashMap<&str, usize> = HashMap::new();
        for doc_id in &chunked_hits {
            *per_parent.entry(*doc_id).or_default() += 1;
        }
        let max_chunks_per_parent = per_parent.values().copied().max().unwrap_or(0);
        assert!(
            max_chunks_per_parent >= 2,
            "Identity must surface ≥2 vector chunks of at least one parent doc \
             (chunks_per_doc=2 implies the index has 2 vec rows per doc); \
             got per_parent={per_parent:?}"
        );
    }

    // T-076-008: identity_and_max_chunk_produce_different_rankings
    //
    // Under chunk-level retrieval, Identity and MaxChunk MUST produce
    // different `ranked_hits`. The same fixture is fed through two
    // pipelines and the resulting (doc_id, chunk_id) sequences are
    // compared — equivalent to `eval_harness compare-baselines` on the
    // real MLX path, but MLX-free so the assertion runs on every CI build.
    #[test]
    fn identity_and_max_chunk_produce_different_rankings() {
        let corpus = vec![
            make_document("d1", "alpha document about retrieval"),
            make_document("d2", "beta document about ranking"),
            make_document("d3", "gamma document about indexing"),
        ];
        let queries = vec![make_query("q1", "alpha retrieval", "d1")];
        let embedder = MockChunkedEmbedder::new(2);
        let reranker: Option<&MockReranker> = None;
        let merge_config = HybridSearchConfig::default();
        let normalization = QueryNormalizationConfig::default();
        let config = PipelineConfig { k: 10 };

        let identity_result = evaluate(
            &corpus,
            &queries,
            &embedder,
            reranker,
            &IdentityAggregator,
            &merge_config,
            &normalization,
            &config,
        )
        .expect("identity pipeline must succeed");

        let max_chunk_result = evaluate(
            &corpus,
            &queries,
            &embedder,
            reranker,
            &MaxChunkAggregator,
            &merge_config,
            &normalization,
            &config,
        )
        .expect("max-chunk pipeline must succeed");

        let project = |hits: &[MergedHit]| -> Vec<(String, Option<String>)> {
            hits.iter()
                .map(|h| (h.doc_id.clone(), h.chunk_id.clone()))
                .collect()
        };
        let identity_ranks = project(&identity_result[0].ranked_hits);
        let max_chunk_ranks = project(&max_chunk_result[0].ranked_hits);

        assert_ne!(
            identity_ranks, max_chunk_ranks,
            "Identity and MaxChunk MUST yield different rankings on chunk-level \
             input — otherwise Stage 2 fusion is collapsing chunks before \
             aggregation has a chance to act"
        );

        // MaxChunk's contract: every emitted hit is parent-granular.
        assert!(
            max_chunk_result[0]
                .ranked_hits
                .iter()
                .all(|h| h.chunk_id.is_none()),
            "MaxChunk must strip chunk_id on every hit; got {:?}",
            max_chunk_result[0].ranked_hits
        );

        // Parent doc count: MaxChunk's set of parent doc_ids ⊆ Identity's.
        let identity_parents: HashSet<&str> = identity_result[0]
            .ranked_hits
            .iter()
            .map(|h| h.doc_id.as_str())
            .collect();
        let max_chunk_parents: HashSet<&str> = max_chunk_result[0]
            .ranked_hits
            .iter()
            .map(|h| h.doc_id.as_str())
            .collect();
        assert!(
            max_chunk_parents.is_subset(&identity_parents),
            "MaxChunk must not introduce parents Identity didn't already see; \
             max_chunk={max_chunk_parents:?} identity={identity_parents:?}"
        );
        // MaxChunk should surface fewer (or equal) entries because it collapses
        // sibling chunks of the same parent.
        assert!(
            max_chunk_result[0].ranked_hits.len() <= identity_result[0].ranked_hits.len(),
            "MaxChunk's collapsed output must be no longer than Identity's; \
             max_chunk_len={} identity_len={}",
            max_chunk_result[0].ranked_hits.len(),
            identity_result[0].ranked_hits.len()
        );
    }
}
