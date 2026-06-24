use super::*;

// T-081-003: subcommand_sets_partition_all_subcommands
//
// Issue #81 Code fix #9: ALL_SUBCOMMANDS must be the disjoint union of
// MLX_DEPENDENT_MODES and NON_MLX_MODES. The compile-time `const_assert_eq!`
// on lengths is necessary but not sufficient — it permits a swap (one entry
// moved from MLX to non-MLX with another entry going the other way). This
// runtime check pins set equality and disjointness so a misclassified
// subcommand cannot silently bypass the seatbelt-sandbox gate.
#[test]
fn subcommand_sets_partition_all_subcommands() {
    let mlx: HashSet<&str> = MLX_DEPENDENT_MODES.iter().copied().collect();
    let non_mlx: HashSet<&str> = NON_MLX_MODES.iter().copied().collect();
    let all: HashSet<&str> = ALL_SUBCOMMANDS.iter().copied().collect();
    assert!(
        mlx.is_disjoint(&non_mlx),
        "MLX-dependent and non-MLX sets must be disjoint; \
             overlap means a subcommand is both gated and ungated. mlx={mlx:?} non_mlx={non_mlx:?}"
    );
    let union: HashSet<&str> = mlx.union(&non_mlx).copied().collect();
    assert_eq!(
        union, all,
        "MLX ∪ non-MLX must equal ALL_SUBCOMMANDS; \
             missing entries can silently bypass the seatbelt classification"
    );
}

// T-004: tty_reject_message_contains_fr_011_substring
// FR-011: TTY_REJECT_MESSAGE must literally contain
//         "pipe jsonl into stdin" so an interactive run prints the
//         exact substring required by Spec FR-011.
//
// Spec Evolution 2026-05-10: integration variant infeasible —
// `cargo test` lanes inherit stdin as pipe / null, so
// `is_terminal() == true` is unreproducible across CI hosts. The
// boolean dispatch is verified separately in
// `tty_guard_returns_usage_for_terminal` so the constant + dispatch
// pair fully cover FR-011.
#[test]
fn tty_reject_message_contains_fr_011_substring() {
    assert!(
        TTY_REJECT_MESSAGE.contains("pipe jsonl into stdin"),
        "FR-011: TTY_REJECT_MESSAGE must contain 'pipe jsonl into stdin'; \
             got: {TTY_REJECT_MESSAGE}"
    );
}

// T-004: tty_guard_returns_usage_for_terminal
// FR-011: when stdin is a TTY, `check_stdin_tty_guard` returns
//         `Some(ExitCode)`; otherwise it returns `None` so
//         `run_annotate` proceeds with stdin parsing.
#[test]
fn tty_guard_returns_usage_for_terminal() {
    assert!(
        check_stdin_tty_guard(true).is_some(),
        "FR-011: TTY stdin must short-circuit run_annotate"
    );
    assert!(
        check_stdin_tty_guard(false).is_none(),
        "FR-011: piped stdin must not short-circuit run_annotate"
    );
}

// T-067-007: aggregation_kind_topk_average_roundtrips_through_storage_enum
//
// A non-default `topk_k=` passed at capture time used to be silently
// downgraded to DEFAULT_TOPK_AVERAGE_K by `from_name` at verify time.
// `AggregationKind::TopKAverage(k)` now encodes `k` in the JSON wire form
// as `"topk-average:k"`, so a round-trip through serde keeps the strategy
// bit-identical across capture/verify.
#[test]
fn aggregation_kind_topk_average_roundtrips_through_storage_enum() {
    for k in [1, 3, 5, 100] {
        let original_spec = AggregationSpec::TopKAverage(k);
        let kind: AggregationKind = original_spec.into();
        let json = serde_json::to_string(&kind).expect("serialise AggregationKind");
        assert_eq!(
            json,
            format!("\"topk-average:{k}\""),
            "TopKAverage({k}) must serialise with k encoded in the wire form"
        );
        let parsed_kind: AggregationKind =
            serde_json::from_str(&json).expect("deserialise AggregationKind");
        let parsed_spec = AggregationSpec::try_from(parsed_kind).expect("storage→runtime bridge");
        assert_eq!(
            parsed_spec, original_spec,
            "round-trip lost k for TopKAverage({k}): {parsed_spec:?}"
        );
    }
}

// T-067-008: aggregation_kind_legacy_bare_topk_average_is_rejected
//
// Pre-fix baselines that wrote bare "topk-average" without `:k` are
// explicitly rejected at deserialise time. Silently falling back to
// DEFAULT_TOPK_AVERAGE_K would re-introduce the bug that the encoded
// form fixes — verify-baseline would dispatch at the default k instead
// of the captured k. No committed fixture uses the bare form, so the
// rejection has zero footprint.
#[test]
fn aggregation_kind_legacy_bare_topk_average_is_rejected() {
    let result = serde_json::from_str::<AggregationKind>("\"topk-average\"");
    assert!(
        result.is_err(),
        "bare 'topk-average' must be rejected to prevent silent default-k fallback; \
             got {result:?}"
    );
}

// T-067-009: aggregation_kind_simple_variants_roundtrip
//
// Identity / MaxChunk / Dedupe / NotApplicable carry no `k` payload, but
// they share the round-trip contract through the storage enum.
#[test]
fn aggregation_kind_simple_variants_roundtrip() {
    for original in [
        AggregationKind::Identity,
        AggregationKind::MaxChunk,
        AggregationKind::Dedupe,
        AggregationKind::NotApplicable,
    ] {
        let json = serde_json::to_string(&original).expect("serialise");
        let parsed: AggregationKind = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(parsed, original, "round-trip mismatch for {original:?}");
    }
}

// T-067-010: aggregation_kind_not_applicable_rejects_runtime_dispatch
//
// NotApplicable is the FirstSearchReplay-only marker. The
// `TryFrom<AggregationKind> for AggregationSpec` bridge must reject it
// so verify-baseline cannot accidentally dispatch a Stage 3 strategy
// on a replay baseline.
#[test]
fn aggregation_kind_not_applicable_rejects_runtime_dispatch() {
    let result = AggregationSpec::try_from(AggregationKind::NotApplicable);
    assert!(
        result.is_err(),
        "NotApplicable must not lower to a runtime AggregationSpec; got {result:?}"
    );
}

// T-068-012: parse_merge_config_empty_kvs_returns_default
#[test]
fn parse_merge_config_empty_kvs_returns_default() {
    let kvs: HashMap<String, String> = HashMap::new();
    let parsed = parse_merge_config_from_kvs(&kvs).expect("empty kvs must parse");
    assert_eq!(parsed, HybridSearchConfig::default());
}

// T-068-013: parse_merge_config_all_overrides_applied
#[test]
fn parse_merge_config_all_overrides_applied() {
    let mut kvs: HashMap<String, String> = HashMap::new();
    kvs.insert("rrf_k".to_owned(), "30".to_owned());
    kvs.insert("fts_weight".to_owned(), "2.5".to_owned());
    kvs.insert("vector_weight".to_owned(), "0.5".to_owned());
    let parsed = parse_merge_config_from_kvs(&kvs).expect("override kvs must parse");
    assert!((parsed.rrf_k - 30.0).abs() < f64::EPSILON);
    assert!(
        (parsed.source_weights[&CandidateSource::Fts] - 2.5).abs() < f64::EPSILON,
        "fts_weight override must be applied"
    );
    assert!(
        (parsed.source_weights[&CandidateSource::Vector] - 0.5).abs() < f64::EPSILON,
        "vector_weight override must be applied"
    );
}

// T-068-014: parse_merge_config_invalid_value_returns_err
#[test]
fn parse_merge_config_invalid_value_returns_err() {
    let mut kvs: HashMap<String, String> = HashMap::new();
    kvs.insert("rrf_k".to_owned(), "not-a-number".to_owned());
    let result = parse_merge_config_from_kvs(&kvs);
    assert!(
        result.is_err(),
        "non-numeric rrf_k must surface a parse error"
    );
}

// T-068-015: parse_merge_config_partial_keeps_other_defaults
#[test]
fn parse_merge_config_partial_keeps_other_defaults() {
    let mut kvs: HashMap<String, String> = HashMap::new();
    kvs.insert("fts_weight".to_owned(), "3.0".to_owned());
    let parsed = parse_merge_config_from_kvs(&kvs).expect("partial kvs must parse");
    // fts overridden
    assert!((parsed.source_weights[&CandidateSource::Fts] - 3.0).abs() < f64::EPSILON);
    // others stay at default
    assert!((parsed.rrf_k - 60.0).abs() < f64::EPSILON);
    assert!((parsed.source_weights[&CandidateSource::Vector] - 1.0).abs() < f64::EPSILON);
}

// T-061-010: `MetricSpec::ALL` leads with HitAt1 / HitAt3.
#[test]
fn metric_spec_all_starts_with_hit_variants_in_order() {
    assert_eq!(
        MetricSpec::ALL[0],
        MetricSpec::HitAt1,
        "MetricSpec::ALL[0] must be HitAt1 so baseline.json global[0] is hit@1"
    );
    assert_eq!(
        MetricSpec::ALL[1],
        MetricSpec::HitAt3,
        "MetricSpec::ALL[1] must be HitAt3 so baseline.json global[1] is hit@3"
    );
}

// T-061-011: HitAt{1,3}.k() returns the cutoff encoded in the variant name.
#[test]
fn metric_spec_hit_variants_report_correct_k() {
    assert_eq!(
        MetricSpec::HitAt1.k(),
        1,
        "HitAt1.k() must be 1 to dispatch hit_at_k against the top-1 window"
    );
    assert_eq!(
        MetricSpec::HitAt3.k(),
        3,
        "HitAt3.k() must be 3 to dispatch hit_at_k against the top-3 window"
    );
}

// T-061-012: from_name("hit@{1,3}") resolves to HitAt{1,3}.
#[test]
fn metric_spec_from_name_resolves_hit_labels() {
    assert_eq!(
        MetricSpec::from_name("hit@1"),
        Some(MetricSpec::HitAt1),
        "from_name(\"hit@1\") must resolve to HitAt1 so verify-baseline can look up tolerance"
    );
    assert_eq!(
        MetricSpec::from_name("hit@3"),
        Some(MetricSpec::HitAt3),
        "from_name(\"hit@3\") must resolve to HitAt3 so verify-baseline can look up tolerance"
    );
}

// T-061-013: HitAt1.metric_fn() dispatches to hit_at_k.
//
// Grade-0 input is the discriminator: hit_at_k returns 0.0, while a wrong
// dispatch to recall_at_k would also return 0.0 only by coincidence (empty
// relevance set), so this case anchors the dispatch contract specifically.
#[test]
fn metric_spec_hit1_metric_fn_dispatches_to_hit_at_k() {
    let ranked: &[&str] = &["d1"];
    let relevance: HashMap<String, u8> = HashMap::from([("d1".to_owned(), 0u8)]);
    let dispatch = MetricSpec::HitAt1.metric_fn();
    let result = dispatch(ranked, &relevance, 1);
    assert!(
        result.abs() < f64::EPSILON,
        "HitAt1.metric_fn() on grade-0 input must yield 0.0; got {result}"
    );
}

// ── Issue #62 / Phase 2.3 — replay-first-search subcommand + verify-baseline 拡張 ──

// T-068-001: ADR-0005 — LazyReranker init must not fire when only the replay path runs.
#[test]
fn replay_first_search_skips_reranker_init() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rurico::embed::MockEmbedder;
    use rurico::reranker::MockReranker;
    use tempfile::tempdir;

    let dir = tempdir().expect("tempdir for replay output");
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval");
    let output_path = dir.path().join("replay.json");

    let init_calls = Arc::new(AtomicUsize::new(0));
    let reranker = LazyReranker::new({
        let init_calls = Arc::clone(&init_calls);
        move || {
            init_calls.fetch_add(1, Ordering::SeqCst);
            Ok(MockReranker::default())
        }
    });

    let ctx = EvalContext {
        fixture_dir,
        embedder: MockEmbedder::default(),
        reranker,
        timestamp: Box::new(|| "epoch:0".to_owned()),
    };

    let merge_config = HybridSearchConfig::default();
    let normalization = QueryNormalizationConfig::default();
    let _exit = run_replay_first_search_with(&ctx, &output_path, &merge_config, &normalization);

    let snapshot = read_snapshot(&output_path).unwrap_or_else(|e| panic!("T-068-001: {e}"));
    // Replay envelope contract, asserted against the real
    // `run_replay_first_search_with` output rather than a builder fed
    // hardcoded args: kind (proves the pipeline ran), captured_with
    // (FR-019 provenance), and the BR-003 aggregation marker. Checking
    // the live snapshot subsumes the former pass-through unit tests.
    assert_eq!(
        snapshot.kind,
        BaselineKind::FirstSearchReplay,
        "T-068-001: snapshot kind must be FirstSearchReplay (proves pipeline ran)"
    );
    assert_eq!(
        snapshot.captured_with, "eval_harness replay-first-search",
        "T-068-001: FR-019 — replay snapshot must record the subcommand in captured_with"
    );
    assert_eq!(
        snapshot.aggregation,
        AggregationKind::NotApplicable,
        "T-068-001: BR-003 — replay runs MaxChunkAggregator only, so the \
             aggregation marker must be NotApplicable"
    );
    assert_eq!(
        init_calls.load(Ordering::SeqCst),
        0,
        "T-068-001: replay-first-search must not invoke the reranker init closure"
    );
}

/// Build a BaselineSnapshot with the requested schema/kind so envelope
/// validation tests don't drag in MockEmbedder + fixture setup.
fn snapshot_for_envelope_test(schema: &str, kind: BaselineKind) -> BaselineSnapshot {
    BaselineSnapshot {
        schema_version: schema.to_owned(),
        kind,
        captured_with: "test".to_owned(),
        timestamp: "epoch:0".to_owned(),
        model_id: RURI_V3_310M_MODEL_ID.to_owned(),
        model_revision: "rev".to_owned(),
        mlx_rs_version: "0.0.0".to_owned(),
        fixture_hash: "fnv1a64:0".to_owned(),
        aggregation: AggregationKind::Identity,
        merge_config: HybridSearchConfig::default(),
        normalization: QueryNormalizationConfig::default(),
        global: vec![],
        per_category: BTreeMap::new(),
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
    }
}

/// Minimal (results, queries) pair for snapshot-builder tests.
/// Empty `ranked_hits` is fine — MetricSpec::ALL still iterates and
/// emits 0.0 for every metric, so ordering / presence assertions hold.
fn replay_snapshot_inputs() -> (Vec<QueryResult>, Vec<EvalQuery>) {
    let queries = vec![EvalQuery {
        id: "q1".to_owned(),
        text: "alpha".to_owned(),
        category: "test".to_owned(),
        relevance_map: HashMap::from([("d1".to_owned(), 1u8)]),
        annotation: "test".to_owned(),
    }];
    let results = vec![QueryResult {
        query_id: "q1".to_owned(),
        ranked_hits: vec![],
        latency_ms: 5,
    }];
    (results, queries)
}

fn build_test_replay_snapshot() -> BaselineSnapshot {
    let (results, queries) = replay_snapshot_inputs();
    build_baseline_snapshot(
        BaselineKind::FirstSearchReplay,
        "eval_harness replay-first-search",
        AggregationKind::NotApplicable,
        &results,
        &queries,
        "fnv1a64:0".to_owned(),
        "epoch:42".to_owned(),
        &HybridSearchConfig::default(),
        &QueryNormalizationConfig::default(),
    )
}

// T-062-016: replay_first_search_subcommand_dispatches_to_evaluate_first_search_replay
//
// FR-016 / FR-017: the subcommand kind label maps through
// `parse_baseline_kind` to `BaselineKind::FirstSearchReplay`. Pinning
// the kind string round-trip guards against an accidental rename
// breaking the dispatch wiring (main()'s match arm + verify-baseline
// kind=).
#[test]
fn replay_first_search_subcommand_dispatches_to_evaluate_first_search_replay() {
    assert_eq!(
        parse_baseline_kind("first_search_replay").expect("first_search_replay kind must parse"),
        BaselineKind::FirstSearchReplay,
        "FR-016/FR-017: subcommand kind label must map to FirstSearchReplay"
    );
}

// T-062-020: verify_baseline_dispatches_on_first_search_replay_kind
//
// FR-021 / FR-022: when committed.kind == FirstSearchReplay and the
// optional kind=first_search_replay argument matches, the envelope
// check passes — caller (`run_verify_baseline_with`) then dispatches
// to `evaluate_first_search_replay`.
#[test]
fn verify_baseline_dispatches_on_first_search_replay_kind() {
    let snap = snapshot_for_envelope_test(BASELINE_SCHEMA_VERSION, BaselineKind::FirstSearchReplay);
    let mut kvs = HashMap::new();
    kvs.insert("kind".to_owned(), "first_search_replay".to_owned());
    let result = validate_committed_baseline_envelope(&snap, &kvs);
    assert!(
        result.is_ok(),
        "FR-021/FR-022: replay kind + matching kvs must pass envelope check; got {result:?}"
    );
}

// T-062-021: verify_baseline_iterates_metric_spec_all_for_replay_kind
//
// FR-022 / NFR-005 / BR-001: replay path emits every MetricSpec::ALL
// entry to global, in the same order as forward / oracle. Ensures
// verify-baseline's `MetricSpec::ALL` iterator (Issue #61 soundness
// gap) covers FirstSearchReplay too.
#[test]
fn verify_baseline_iterates_metric_spec_all_for_replay_kind() {
    let snap = build_test_replay_snapshot();
    let names: Vec<&str> = snap.global.iter().map(|m| m.name.as_str()).collect();
    let expected: Vec<&str> = MetricSpec::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(
        names, expected,
        "FR-022/NFR-005/BR-001: replay global must mirror MetricSpec::ALL order"
    );
}

// T-062-022: verify_baseline_rejects_committed_baseline_with_schema_version_1_2
//
// FR-023 / NFR-005: stale 1.2 baseline must `EXIT_REGRESSION` (not
// silent skip). Ensures Issue #62's SCHEMA bump gate matches Issue
// #61's verify-baseline soundness contract.
#[test]
fn verify_baseline_rejects_committed_baseline_with_schema_version_1_2() {
    let snap = snapshot_for_envelope_test("1.2", BaselineKind::Forward);
    let kvs = HashMap::new();
    let result = validate_committed_baseline_envelope(&snap, &kvs);
    assert_eq!(
        result,
        Err(EXIT_REGRESSION),
        "FR-023/NFR-005: stale 1.2 baseline must EXIT_REGRESSION"
    );
}

// T-062-027: verify_baseline_rejects_committed_baseline_with_model_id_mismatch
//
// RC-002: a baseline captured with a different embedding model must
// EXIT_REGRESSION, not silently compare metrics across models. model_id
// joins schema_version / kind / metric-k as an envelope contract gate so a
// rurico model bump that changes RURI_V3_310M_MODEL_ID forces regeneration.
#[test]
fn verify_baseline_rejects_committed_baseline_with_model_id_mismatch() {
    let mut snap = snapshot_for_envelope_test(BASELINE_SCHEMA_VERSION, BaselineKind::Forward);
    snap.model_id = "cl-nagoya/ruri-v3-30m".to_owned();
    let kvs = HashMap::new();
    let result = validate_committed_baseline_envelope(&snap, &kvs);
    assert_eq!(
        result,
        Err(EXIT_REGRESSION),
        "RC-002: committed baseline from a different embedding model must EXIT_REGRESSION"
    );
}

// T-ADR0006-005: verify_baseline_rejects_committed_metric_with_k_mismatch
//
// ADR-0006 § Implementation Guidelines 第 2 項: committed metric の
// `MetricResult.k` が `MetricSpec` で宣言された k と一致しない場合は
// PIPELINE_K bump 由来の stale baseline として `EXIT_REGRESSION`。
// 実装は ADR 本文 ("recall@k キー名から逆引き") ではなく `.k` field 直読 +
// `MetricSpec.k()` 比較 (string parse 不要、rename も downstream で別途検出)。
#[test]
fn verify_baseline_rejects_committed_metric_with_k_mismatch() {
    let mut snap = snapshot_for_envelope_test(BASELINE_SCHEMA_VERSION, BaselineKind::Forward);
    snap.global.push(MetricResult {
        name: "recall@10".to_owned(),
        k: 5,
        point_estimate: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
        uninformative: false,
    });
    let kvs = HashMap::new();
    let result = validate_committed_baseline_envelope(&snap, &kvs);
    assert_eq!(
        result,
        Err(EXIT_REGRESSION),
        "ADR-0006: MetricSpec.k() と committed metric.k の不一致は EXIT_REGRESSION"
    );
}

// T-062-023: replay_first_search_records_latency_p50_ms
//
// FR-019 / NFR-004: the replay snapshot carries latency_p50_ms so the
// ADR-0003 第 4 reassessment trigger ("forward 比 2-5x 高速化に届かない")
// can be checked against committed JSON.
#[test]
fn replay_first_search_records_latency_p50_ms() {
    let snap = build_test_replay_snapshot();
    assert!(
        snap.latency_p50_ms.is_finite(),
        "FR-019/NFR-004: latency_p50_ms must be finite, got {}",
        snap.latency_p50_ms
    );
}

// T-062-024: replay_first_search_records_latency_p95_ms
//
// FR-019 / NFR-004: tail latency p95 lives next to p50 in the snapshot
// envelope so capacity drift can be tracked alongside median speedup.
#[test]
fn replay_first_search_records_latency_p95_ms() {
    let snap = build_test_replay_snapshot();
    assert!(
        snap.latency_p95_ms.is_finite(),
        "FR-019/NFR-004: latency_p95_ms must be finite, got {}",
        snap.latency_p95_ms
    );
}

// T-062-025: replay_first_search_emits_hit_at_k_in_global_metrics
//
// FR-018: replay reuses `build_global_metrics`, which iterates
// `MetricSpec::ALL` (HitAt1 / HitAt3 first per Issue #61). Pins the
// first two slots so the ADR-0003 "first-result quality" indicator
// is visible at the top of the snapshot JSON.
#[test]
fn replay_first_search_emits_hit_at_k_in_global_metrics() {
    let snap = build_test_replay_snapshot();
    assert_eq!(
        snap.global.first().map(|m| m.name.as_str()),
        Some("hit@1"),
        "FR-018: replay global[0] must be hit@1 (Issue #61 ALL ordering)"
    );
    assert_eq!(
        snap.global.get(1).map(|m| m.name.as_str()),
        Some("hit@3"),
        "FR-018: replay global[1] must be hit@3"
    );
}

// T-062-026: verify_baseline_rejects_kind_mismatch
//
// FR-024 / AC-7: `verify-baseline baseline=forward.json kind=replay`
// must `EXIT_REGRESSION`. Guards against running the replay metric
// dispatch against a forward snapshot (or vice versa).
#[test]
fn verify_baseline_rejects_kind_mismatch() {
    let snap = snapshot_for_envelope_test(BASELINE_SCHEMA_VERSION, BaselineKind::Forward);
    let mut kvs = HashMap::new();
    kvs.insert("kind".to_owned(), "first_search_replay".to_owned());
    let result = validate_committed_baseline_envelope(&snap, &kvs);
    assert_eq!(
        result,
        Err(EXIT_REGRESSION),
        "FR-024/AC-7: kind mismatch (forward baseline + kind=replay) must EXIT_REGRESSION"
    );
}

#[test]
fn fnv1a64_matches_known_answer_vectors() {
    // ADR-0011: pin the hand-rolled FNV-1a 64 against canonical test vectors so
    // a regression in the hand-roll is caught without depending on sha2.
    assert_eq!(fnv1a64_update(FNV1A64_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64_update(FNV1A64_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(
        fnv1a64_update(FNV1A64_OFFSET, b"foobar"),
        0x8594_4171_f739_67e8
    );
}

#[test]
fn fnv1a64_update_accumulates_across_chunks() {
    // hash_fixture_dir threads one hash across successive file reads; chunked
    // folding must equal hashing the concatenation (associativity).
    let split = fnv1a64_update(fnv1a64_update(FNV1A64_OFFSET, b"foo"), b"bar");
    let whole = fnv1a64_update(FNV1A64_OFFSET, b"foobar");
    assert_eq!(split, whole);
}

#[test]
fn epoch_label_formats_seconds_as_epoch_prefix() {
    // ADR-0011: pin the `epoch:N` wire shape the consumer schema parses, so a
    // drift away from the chrono-free format is caught without faking the clock.
    assert_eq!(epoch_label(0), "epoch:0");
    assert_eq!(epoch_label(42), "epoch:42");
    assert_eq!(epoch_label(1_700_000_000), "epoch:1700000000");
}
