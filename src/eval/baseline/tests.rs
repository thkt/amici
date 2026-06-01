use super::*;

// T-018: build_metric_result_wide_ci_flags_uninformative
// FR-016 / BR-002: half-width = (0.65 - 0.35) / 2 = 0.15 > 0.10 →
//                  MetricResult.uninformative == true.
#[test]
fn build_metric_result_wide_ci_flags_uninformative() {
    let result = build_metric_result("recall@5".to_owned(), 5, 0.5, 0.35, 0.65);

    assert!(
        result.uninformative,
        "FR-016: half-width 0.15 > threshold 0.10 → uninformative must be true, \
             got result = {result:?}"
    );
}

// T-018b: build_metric_result_narrow_ci_stays_informative
// FR-016 / BR-002: half-width = (0.55 - 0.45) / 2 = 0.05 < 0.10 →
//                  MetricResult.uninformative == false (negative case).
#[test]
fn build_metric_result_narrow_ci_stays_informative() {
    let result = build_metric_result("recall@5".to_owned(), 5, 0.5, 0.45, 0.55);

    assert!(
        !result.uninformative,
        "FR-016: half-width 0.05 < threshold 0.10 → uninformative must be false, \
             got result = {result:?}"
    );
}

// T-020: baseline_snapshot_round_trips_with_schema_version_and_kind
// Pins the schema_version + kind envelope so a future migration that
// drops or renames either field fails this round-trip explicitly.
#[test]
fn baseline_snapshot_round_trips_with_schema_version_and_kind() {
    let snap = BaselineSnapshot {
        schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
        kind: BaselineKind::Forward,
        captured_with: "test".to_owned(),
        timestamp: "epoch:42".to_owned(),
        model_id: "test/model".to_owned(),
        model_revision: "rev".to_owned(),
        mlx_rs_version: "0.0.0".to_owned(),
        fixture_hash: "fnv1a64:0".to_owned(),
        aggregation: default_aggregation_kind(),
        merge_config: HybridSearchConfig::default(),
        normalization: pre_phase_5_disabled(),
        global: vec![],
        per_category: BTreeMap::new(),
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
    };
    let json = serde_json::to_string(&snap).expect("serialise");
    let parsed: BaselineSnapshot = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed.schema_version, BASELINE_SCHEMA_VERSION);
    assert_eq!(parsed.kind, BaselineKind::Forward);
    assert_eq!(parsed.timestamp, "epoch:42");
}

// T-052-301: baseline_kind_oracle_serialises_as_snake_case
//
// Issue #52: BaselineKind::Oracle round-trips through serde_json as
// the snake_case label "oracle". Pinning the wire format guards
// against a future kind variant accidentally changing the discriminator
// name (which would invalidate every committed oracle_baseline.json).
#[test]
fn baseline_kind_oracle_serialises_as_snake_case() {
    let json = serde_json::to_string(&BaselineKind::Oracle).expect("serialise");
    assert_eq!(
        json, "\"oracle\"",
        "Oracle variant must serialise to \"oracle\""
    );
    let parsed: BaselineKind = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed, BaselineKind::Oracle);
}

// T-052-302: baseline_snapshot_round_trips_with_oracle_kind
//
// The full BaselineSnapshot envelope round-trips with kind=Oracle,
// proving Forward and Oracle share a body shape and only differ in
// the discriminator. Required for `capture-oracle` to reuse
// [`write_json`] without a separate serialisation path.
#[test]
fn baseline_snapshot_round_trips_with_oracle_kind() {
    let snap = BaselineSnapshot {
        schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
        kind: BaselineKind::Oracle,
        captured_with: "eval_harness capture-oracle".to_owned(),
        timestamp: "epoch:42".to_owned(),
        model_id: "test/model".to_owned(),
        model_revision: "rev".to_owned(),
        mlx_rs_version: "0.0.0".to_owned(),
        fixture_hash: "fnv1a64:0".to_owned(),
        aggregation: default_aggregation_kind(),
        merge_config: HybridSearchConfig::default(),
        normalization: pre_phase_5_disabled(),
        global: vec![],
        per_category: BTreeMap::new(),
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
    };
    let json = serde_json::to_string(&snap).expect("serialise");
    let parsed: BaselineSnapshot = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed.kind, BaselineKind::Oracle);
    assert_eq!(parsed.captured_with, "eval_harness capture-oracle");
}

// T-021: committed_baseline_json_deserialises_under_new_schema
// Default-lane guard: MLX-gated T-019 won't catch a schema migration
// that drops the committed fixture's parseability. This test runs in
// the default `cargo test` lane (no feature flag, no #[ignore]) so a
// stale fixture fails CI immediately.
#[test]
fn committed_baseline_json_deserialises_under_new_schema() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval/baseline.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let parsed: BaselineSnapshot = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "committed baseline.json must deserialise under the current schema ({e}); \
                 update the fixture when bumping {BASELINE_SCHEMA_VERSION:?}. content head: {}",
            &text.chars().take(200).collect::<String>()
        )
    });
    assert_eq!(
        parsed.schema_version, BASELINE_SCHEMA_VERSION,
        "schema_version mismatch — fixture must match {BASELINE_SCHEMA_VERSION:?}"
    );
    assert_eq!(
        parsed.kind,
        BaselineKind::Forward,
        "committed baseline.json must declare kind=forward"
    );
}

// T-062-011 / T-062-015: baseline_kind_first_search_replay_serialises_as_snake_case
//
// FR-012: BaselineKind::FirstSearchReplay round-trips through serde_json
// as the snake_case label "first_search_replay". Pinning the wire format
// guards against a future variant rename silently breaking committed
// first_search_replay_baseline.json files. A serde-attribute regression
// (e.g. dropping `rename_all = "snake_case"`) would break both serialize
// and deserialize together, so wire-format pin and enum-level round-trip
// collapse into one test (mirrors T-052-301 for Oracle).
#[test]
fn baseline_kind_first_search_replay_serialises_as_snake_case() {
    let json = serde_json::to_string(&BaselineKind::FirstSearchReplay).expect("serialise");
    assert_eq!(
        json, "\"first_search_replay\"",
        "FR-012: FirstSearchReplay variant must serialise to \"first_search_replay\""
    );
    let parsed: BaselineKind = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed, BaselineKind::FirstSearchReplay);
}

// T-062-012: baseline_snapshot_round_trips_with_first_search_replay_kind
//
// FR-013: BaselineSnapshot envelope round-trips with
// `kind=FirstSearchReplay`, proving the new variant shares Forward /
// Oracle's body shape and `replay-first-search` can reuse `write_json`
// without forking the serialisation path. `aggregation="none"` is the
// BR-003 marker for "Stage 3 ran MaxChunkAggregator only".
#[test]
fn baseline_snapshot_round_trips_with_first_search_replay_kind() {
    let snap = BaselineSnapshot {
        schema_version: BASELINE_SCHEMA_VERSION.to_owned(),
        kind: BaselineKind::FirstSearchReplay,
        captured_with: "eval_harness replay-first-search".to_owned(),
        timestamp: "epoch:42".to_owned(),
        model_id: "test/model".to_owned(),
        model_revision: "rev".to_owned(),
        mlx_rs_version: "0.0.0".to_owned(),
        fixture_hash: "fnv1a64:0".to_owned(),
        aggregation: AggregationKind::NotApplicable,
        merge_config: HybridSearchConfig::default(),
        normalization: pre_phase_5_disabled(),
        global: vec![],
        per_category: BTreeMap::new(),
        latency_p50_ms: 0.0,
        latency_p95_ms: 0.0,
    };
    let json = serde_json::to_string(&snap).expect("serialise");
    let parsed: BaselineSnapshot = serde_json::from_str(&json).expect("round-trip");
    assert_eq!(parsed.kind, BaselineKind::FirstSearchReplay);
    assert_eq!(parsed.captured_with, "eval_harness replay-first-search");
    assert_eq!(
        parsed.aggregation,
        AggregationKind::NotApplicable,
        "BR-003: aggregation marker is NotApplicable for FirstSearchReplay"
    );
}

// T-062-014: committed_baselines_deserialize_under_schema_1_3
//
// FR-014 / AC-6: every committed baseline file (forward / oracle /
// first-search replay) deserialises cleanly under schema 1.3. Pinning
// the three-file invariant in a single test surfaces a stale fixture
// (e.g. one regenerated under a different schema) in CI before
// verify-baseline hits it at runtime.
#[test]
fn committed_baselines_deserialize_under_schema_1_3() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval");
    for (name, expected_kind) in [
        ("baseline.json", BaselineKind::Forward),
        ("oracle_baseline.json", BaselineKind::Oracle),
        (
            "first_search_replay_baseline.json",
            BaselineKind::FirstSearchReplay,
        ),
    ] {
        let path = fixture_dir.join(name);
        let text =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let parsed: BaselineSnapshot = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{name} must deserialise under schema 1.3 ({e})"));
        assert_eq!(parsed.schema_version, "1.3", "{name} schema must be 1.3");
        assert_eq!(
            parsed.kind, expected_kind,
            "{name} kind must be {expected_kind:?}"
        );
    }
}

// T-081-002: aggregation_kind_roundtrips_committed_fixtures
//
// Issue #81 Code fix #10: lock in that the typed [`AggregationKind`] enum
// deserialises every committed baseline fixture to the expected variant
// **and** re-serialises to a byte-equivalent JSON literal. Catches the
// most likely regression on this PR — a serde wire-form change that
// silently rewrites the meaning of historical baselines.
#[test]
fn aggregation_kind_roundtrips_committed_fixtures() {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval");
    for (name, expected_kind) in [
        ("baseline.json", AggregationKind::Identity),
        ("oracle_baseline.json", AggregationKind::Identity),
        (
            "first_search_replay_baseline.json",
            AggregationKind::NotApplicable,
        ),
    ] {
        let path = fixture_dir.join(name);
        let text =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let snap: BaselineSnapshot =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name} must deserialise: {e}"));
        assert_eq!(
            snap.aggregation, expected_kind,
            "{name}: aggregation must deserialise to {expected_kind:?}"
        );
        let reserialized = serde_json::to_value(snap.aggregation)
            .unwrap_or_else(|e| panic!("{name}: serialise aggregation: {e}"));
        let original: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: parse as Value: {e}"));
        assert_eq!(
            reserialized, original["aggregation"],
            "{name}: aggregation field must round-trip byte-equivalent JSON"
        );
    }
}

// T-022: atomic_write_replaces_destination_on_each_call
// Verifies the temp-file + rename path overwrites cleanly across
// consecutive writes (no leftover .tmp files) and that the final
// destination matches the most recent write.
#[test]
fn atomic_write_replaces_destination_on_each_call() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("artifact.json");
    atomic_write(&path, b"first\n").expect("first write");
    atomic_write(&path, b"second\n").expect("second write");
    let content = fs::read_to_string(&path).expect("read");
    assert_eq!(content, "second\n", "atomic_write must replace contents");
    assert!(
        !dir.path().join(".artifact.json.tmp").exists(),
        "temp file must be renamed away"
    );
}

// ADR-0007 #9: historical_baseline_resolves_serde_defaults_to_pre_existing_behavior
//
// §Decision Outcome 4 + §Implementation Guidelines step 4: field-less
// historical baseline (`aggregation` / `merge_config` / `normalization` 3
// フィールドすべて欠落) を deserialize し、各 serde default が
// pre-existing behavior と等価な値に解決されることを pin。本 test が fail =
// 既存 default 関数のいずれかが pre-existing behavior と乖離した = 規律違反、
// ADR-0007 の bump policy 前提が崩れる。
#[test]
fn historical_baseline_resolves_serde_defaults_to_pre_existing_behavior() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/eval/historical/pre_serde_defaults.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let parsed: BaselineSnapshot = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "historical fixture must deserialise via serde defaults (3 missing \
                 fields each resolved by their `#[serde(default = ...)]`): {e}"
        )
    });

    assert_eq!(
        parsed.aggregation,
        AggregationKind::Identity,
        "ADR-0007: pre-aggregation historical baseline must resolve `aggregation` \
             via `default_aggregation_kind` to `AggregationKind::Identity` \
             (== pre-existing behavior, NOT a runtime default that drifted)"
    );
    assert_eq!(
        parsed.merge_config,
        HybridSearchConfig::default(),
        "ADR-0007: pre-merge-config historical baseline must resolve `merge_config` \
             via `#[serde(default)]` to `HybridSearchConfig::default()` \
             (rrf_k=60, fts/vector weights=1.0 — pre-merge-config 時代の実質挙動)"
    );
    assert_eq!(
        parsed.normalization,
        pre_phase_5_disabled(),
        "ADR-0007: pre-normalization historical baseline must resolve `normalization` \
             via `pre_phase_5_disabled` to all-OFF \
             (NOT runtime `QueryNormalizationConfig::default` which is all-ON)"
    );
}
