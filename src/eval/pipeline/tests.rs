use std::collections::{HashMap, HashSet};

use rurico::embed::{EMBEDDING_DIMS, MockChunkedEmbedder, MockEmbedder};
use rurico::reranker::MockReranker;
use rurico::retrieval::{
    HybridSearchConfig, IdentityAggregator, MaxChunkAggregator, MergedHit, WeightedRrf,
};
use rurico::storage::QueryNormalizationConfig;

use super::{
    PipelineConfig, PipelineError, evaluate, evaluate_first_search_replay, run_stage1_plus_2,
    setup_pipeline_connection,
};
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

// T-062-001: run_stage1_plus_2_returns_merged_hits_for_mock_corpus
//
// FR-001 / FR-002: helper composes retrieve_fts + retrieve_vec → merge_strategy.merge.
// Indexed corpus + valid query → non-empty Vec<MergedHit>.
#[test]
fn run_stage1_plus_2_returns_merged_hits_for_mock_corpus() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
    ];
    let query = make_query("q1", "alpha retrieval", "d1");
    let embedder = MockEmbedder::default();
    let merge_strategy = WeightedRrf::new(HybridSearchConfig::default());
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let conn = setup_pipeline_connection(&corpus, &embedder, &normalization)
        .expect("setup must succeed for MockEmbedder corpus");
    let merged = run_stage1_plus_2(
        &conn,
        &query,
        &embedder,
        &merge_strategy,
        &normalization,
        &config,
    )
    .expect("Stage 1+2 helper must succeed");

    assert!(
        !merged.is_empty(),
        "FR-001/FR-002: indexed corpus + valid query → run_stage1_plus_2 must \
             return non-empty Vec<MergedHit>, got empty"
    );
}

// T-062-002: run_stage1_plus_2_does_not_truncate_or_sort_or_aggregate
//
// FR-003: helper returns merge_strategy.merge() output verbatim. Verified by
//   1. chunk-level fixture (MockChunkedEmbedder::new(2)) → at least one
//      MergedHit retains chunk_id=Some(_); a Stage 3 aggregator (e.g.
//      MaxChunkAggregator) would collapse all chunk_ids to None.
//   2. merged.len() > config.k → no truncate to k.
#[test]
fn run_stage1_plus_2_does_not_truncate_or_sort_or_aggregate() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
    ];
    let query = make_query("q1", "alpha retrieval", "d1");
    let embedder = MockChunkedEmbedder::new(2);
    let merge_strategy = WeightedRrf::new(HybridSearchConfig::default());
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 2 };

    let conn = setup_pipeline_connection(&corpus, &embedder, &normalization)
        .expect("setup must succeed for chunk-level corpus");
    let merged = run_stage1_plus_2(
        &conn,
        &query,
        &embedder,
        &merge_strategy,
        &normalization,
        &config,
    )
    .expect("Stage 1+2 helper must succeed");

    let chunk_id_some_count = merged.iter().filter(|h| h.chunk_id.is_some()).count();
    assert!(
        chunk_id_some_count > 0,
        "FR-003: helper must NOT invoke any Stage 3 aggregator. \
             A Stage 3 aggregator would collapse chunk_id to None, but \
             retrieved chunk-level hits should be preserved verbatim. \
             Got {chunk_id_some_count} hits with chunk_id=Some, merged={merged:?}"
    );

    // candidate_limit = k * RRF_CANDIDATE_MULTIPLIER (= 2 * 3 = 6); with
    // 3 docs × 2 chunks per doc, vec retrieval contributes ≤ 6 hits.
    // After merge, length should exceed config.k (= 2) — proving the
    // helper did not truncate.
    assert!(
        merged.len() > config.k,
        "FR-003: helper must NOT truncate to k={}, got merged.len()={}",
        config.k,
        merged.len()
    );
}

// T-062-004: evaluate_first_search_replay_returns_one_result_per_query
//
// FR-006 / FR-007: Stage 1+2 + parent rollup → exactly one
// QueryResult per EvalQuery, in the same order as `queries`.
#[test]
fn evaluate_first_search_replay_returns_one_result_per_query() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
    ];
    let queries = vec![
        make_query("q1", "alpha retrieval", "d1"),
        make_query("q2", "beta ranking", "d2"),
    ];
    let embedder = MockEmbedder::default();
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 3 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    assert_eq!(
        results.len(),
        queries.len(),
        "FR-006/FR-007: one QueryResult per query, got {} results for {} queries",
        results.len(),
        queries.len()
    );
    assert_eq!(
        results[0].query_id, "q1",
        "QueryResult order must mirror input queries"
    );
    assert_eq!(results[1].query_id, "q2", "QueryResult order preserved");
}

// T-062-005 / T-062-006: evaluate_first_search_replay_signature_excludes_aggregator_and_reranker
//
// FR-008 / FR-010 compile-time guard: the signature has neither an
// `Aggregator` nor a `Reranker` parameter — verified by the call
// below compiling without either. A single passing call site pins
// the entire parameter list, so T-062-005 (no aggregator) and
// T-062-006 (no reranker) collapse into one test.
#[test]
fn evaluate_first_search_replay_signature_excludes_aggregator_and_reranker() {
    let corpus = vec![make_document("d1", "alpha")];
    let queries = vec![make_query("q1", "alpha", "d1")];
    let embedder = MockEmbedder::default();
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 1 };

    let _ = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");
}

// T-062-007: evaluate_first_search_replay_results_are_score_descending
//
// FR-007 / BR-005: ranked_hits sorted by descending score. The order
// comes from Aggregator::aggregate's trait contract (MaxChunkAggregator
// sorts unconditionally), so no explicit sort is needed in the replay
// path.
#[test]
fn evaluate_first_search_replay_results_are_score_descending() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
        make_document("d4", "delta document about scoring"),
        make_document("d5", "epsilon document about evaluation"),
    ];
    let queries = vec![make_query("q1", "alpha retrieval", "d1")];
    let embedder = MockChunkedEmbedder::new(2);
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    let scores: Vec<f64> = results[0].ranked_hits.iter().map(|h| h.score).collect();
    let mut prev = f64::INFINITY;
    for (i, score) in scores.iter().enumerate() {
        assert!(
            *score <= prev,
            "BR-005: ranked_hits must be score-descending; \
                 hit {i} has score {score} > prev {prev}, full scores={scores:?}"
        );
        prev = *score;
    }
}

// T-062-007a: evaluate_first_search_replay_results_have_unique_doc_ids
//
// FR-007a / BR-007: parent rollup collapses sibling chunks of the same
// doc_id to a single MergedHit (max score), so the top-k window
// measures unique parent docs (pgr "top-k unique docs include
// relevant" semantics, ADR-0003 第 2 deliverable).
#[test]
fn evaluate_first_search_replay_results_have_unique_doc_ids() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
    ];
    let queries = vec![make_query("q1", "alpha retrieval", "d1")];
    let embedder = MockChunkedEmbedder::new(2);
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    let doc_ids: Vec<&str> = results[0]
        .ranked_hits
        .iter()
        .map(|h| h.doc_id.as_str())
        .collect();
    let unique: HashSet<&str> = doc_ids.iter().copied().collect();
    assert_eq!(
        doc_ids.len(),
        unique.len(),
        "FR-007a/BR-007: ranked_hits must have unique doc_ids after \
             parent rollup; got duplicates in {doc_ids:?}"
    );
}

// T-062-007b: evaluate_first_search_replay_results_have_chunk_id_none
//
// BR-007: parent rollup post-condition — every entry has
// `chunk_id == None`. MaxChunkAggregator strips chunk_id when
// collapsing siblings; replay path must preserve that invariant
// through truncate-to-k.
#[test]
fn evaluate_first_search_replay_results_have_chunk_id_none() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
    ];
    let queries = vec![make_query("q1", "alpha retrieval", "d1")];
    let embedder = MockChunkedEmbedder::new(2);
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    assert!(
        results[0].ranked_hits.iter().all(|h| h.chunk_id.is_none()),
        "BR-007: parent rollup must strip chunk_id; got {:?}",
        results[0].ranked_hits
    );
}

// T-062-008: evaluate_first_search_replay_truncates_to_k
//
// FR-007: caller truncates to config.k after rollup + sort.
#[test]
fn evaluate_first_search_replay_truncates_to_k() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
        make_document("d3", "gamma document about indexing"),
        make_document("d4", "delta document about scoring"),
        make_document("d5", "epsilon document about evaluation"),
    ];
    let queries = vec![make_query("q1", "alpha retrieval", "d1")];
    let embedder = MockEmbedder::default();
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 2 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    assert!(
        results[0].ranked_hits.len() <= config.k,
        "FR-007: replay must truncate to k={}, got len={}",
        config.k,
        results[0].ranked_hits.len()
    );
}

// T-062-009: evaluate_first_search_replay_records_per_query_latency
//
// FR-009: per-query latency_ms is recorded on QueryResult.
// latency_ms is u64 by type, so its presence on QueryResult is the
// FR-009 signal — referencing the field below proves it exists.
#[test]
fn evaluate_first_search_replay_records_per_query_latency() {
    let corpus = vec![make_document("d1", "alpha")];
    let queries = vec![make_query("q1", "alpha", "d1")];
    let embedder = MockEmbedder::default();
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 1 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    let _latency: u64 = results[0].latency_ms;
}

// T-062-010: evaluate_first_search_replay_does_not_invoke_identity_aggregator
//
// FR-008 / BR-008: replay path uses MaxChunkAggregator only. Under
// chunk-level retrieval (MockChunkedEmbedder::new(2)),
// IdentityAggregator would surface per-chunk hits (chunk_id=Some).
// Replay's MaxChunkAggregator must collapse them — every result has
// chunk_id=None — proving Identity (or any aggregator that preserves
// chunk_id) was not invoked.
#[test]
fn evaluate_first_search_replay_does_not_invoke_identity_aggregator() {
    let corpus = vec![
        make_document("d1", "alpha document about retrieval"),
        make_document("d2", "beta document about ranking"),
    ];
    let queries = vec![make_query("q1", "alpha retrieval", "d1")];
    let embedder = MockChunkedEmbedder::new(2);
    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let config = PipelineConfig { k: 5 };

    let results = evaluate_first_search_replay(
        &corpus,
        &queries,
        &embedder,
        &merge_config,
        &normalization,
        &config,
    )
    .expect("replay must succeed");

    let chunk_id_some_count = results[0]
        .ranked_hits
        .iter()
        .filter(|h| h.chunk_id.is_some())
        .count();
    assert_eq!(
        chunk_id_some_count, 0,
        "FR-008/BR-008: replay must NOT invoke IdentityAggregator (or \
             any aggregator that preserves chunk_id). Got \
             {chunk_id_some_count} hits with chunk_id=Some, ranked_hits={:?}",
        results[0].ranked_hits
    );
}

// T-081-001: index_corpus_rejects_chunk_with_wrong_dimension
//
// The `&[f32; EMBEDDING_DIMS]` cast in `index_corpus` is the layout
// contract enforcement point. A chunk vector with length ≠
// `EMBEDDING_DIMS` must surface as `ChunkDimensionMismatch` instead of
// a `bytemuck` panic or silently rewritten bytes.
#[test]
fn index_corpus_rejects_chunk_with_wrong_dimension() {
    let corpus = vec![make_document("d1", "alpha document about retrieval")];
    let wrong_dims = EMBEDDING_DIMS + 1;
    let embedder = MockEmbedder::with_dims(wrong_dims);
    let normalization = QueryNormalizationConfig::default();

    let err = setup_pipeline_connection(&corpus, &embedder, &normalization)
        .expect_err("dim mismatch must surface as ChunkDimensionMismatch");

    match err {
        PipelineError::ChunkDimensionMismatch {
            doc_id,
            expected,
            actual,
        } => {
            assert_eq!(doc_id, "d1");
            assert_eq!(expected, EMBEDDING_DIMS);
            assert_eq!(actual, wrong_dims);
        }
        other => panic!("expected ChunkDimensionMismatch, got {other:?}"),
    }
}
