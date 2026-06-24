use super::*;
use rurico::storage::{QueryNormalizationConfig, prepare_match_query};
use rusqlite::Connection;

#[test]
fn distributes_or_groups() {
    // Control chars removed + sub-trigram dropped + distributed
    assert_eq!(
        clean_impl("(\"認証の\" OR \"認証\n\" OR \"認証フ\") \"フロー\"").as_deref(),
        Some("\"認証の\" \"フロー\" OR \"認証フ\" \"フロー\"")
    );

    // Multi-element group + fixed term → distributed
    assert_eq!(
        clean_impl("(\"abc\" OR \"def\") \"ghi\"").as_deref(),
        Some("\"abc\" \"ghi\" OR \"def\" \"ghi\"")
    );

    // No parens → unchanged
    assert_eq!(clean_impl("\"hello\"").as_deref(), Some("\"hello\""));

    // Single group, no fixed terms → just OR
    assert_eq!(
        clean_impl("(\"abc\" OR \"def\")").as_deref(),
        Some("\"abc\" OR \"def\"")
    );

    // Multiple OR groups → cross-product
    assert_eq!(
        clean_impl("(\"a01\" OR \"a02\") (\"b01\" OR \"b02\")").as_deref(),
        Some("\"a01\" \"b01\" OR \"a01\" \"b02\" OR \"a02\" \"b01\" OR \"a02\" \"b02\"")
    );
}

#[test]
fn returns_none_when_all_sub_trigram() {
    // All terms in the only OR-group are sub-trigram (<3 chars).
    // parse_fts_segments filters them, leaving no fixed terms → None.
    assert_eq!(clean_impl("(\"ab\" OR \"cd\")"), None);
}

#[test]
fn returns_none_for_empty_input() {
    assert_eq!(clean_impl(""), None);
}

#[test]
fn returns_none_for_sub_trigram_fixed_terms() {
    // All fixed quoted tokens are <3 chars — trigram tokenizer cannot index them.
    assert_eq!(clean_impl("\"au\""), None);
    assert_eq!(clean_impl("\"a\" \"b\""), None);
}

#[test]
fn falls_back_to_fixed_only_when_combos_exceed_max() {
    // 5 OR-groups × 3 terms = 243 combos > MAX_COMBOS (100).
    // Distribution would explode; fallback drops OR-groups and returns
    // only the fixed term.
    let input = r#"("aaa" OR "bbb" OR "ccc") ("ddd" OR "eee" OR "fff") ("ggg" OR "hhh" OR "iii") ("jjj" OR "kkk" OR "lll") ("mmm" OR "nnn" OR "ooo") "fixed""#;
    assert_eq!(
        clean_impl(input).as_deref(),
        Some("\"fixed\""),
        "OR-groups should be dropped, fixed term retained"
    );
}

#[test]
fn falls_back_to_none_when_combos_exceed_max_and_no_fixed() {
    // Same combo count, no fixed term → no indexable content remains.
    let input = r#"("aaa" OR "bbb" OR "ccc") ("ddd" OR "eee" OR "fff") ("ggg" OR "hhh" OR "iii") ("jjj" OR "kkk" OR "lll") ("mmm" OR "nnn" OR "ooo")"#;
    assert_eq!(clean_impl(input), None);
}

#[test]
fn distributes_below_combo_threshold() {
    // 4 OR-groups × 3 terms = 81 combos < MAX_COMBOS (100): distribute.
    let input = r#"("aaa" OR "bbb" OR "ccc") ("ddd" OR "eee" OR "fff") ("ggg" OR "hhh" OR "iii") ("jjj" OR "kkk" OR "lll")"#;
    let out = clean_impl(input).expect("should distribute below threshold");
    assert_eq!(out.matches(" OR ").count(), 80, "81 combos = 80 OR joins");
}

#[test]
fn distributes_at_combo_threshold() {
    // 5 × 5 × 4 = 100 combos = MAX_COMBOS: distributes (`>`, not `>=`).
    // Pins the boundary so an off-by-one slip in the guard regresses here.
    let input = r#"("a01" OR "a02" OR "a03" OR "a04" OR "a05") ("b01" OR "b02" OR "b03" OR "b04" OR "b05") ("c01" OR "c02" OR "c03" OR "c04")"#;
    let out = clean_impl(input).expect("100 combos should distribute");
    assert_eq!(out.matches(" OR ").count(), 99, "100 combos = 99 OR joins");
}

#[test]
fn accepts_live_prepare_match_query_output() {
    // Integration: rurico sanitizes, amici adapts and returns Some.
    let conn = Connection::open_in_memory().unwrap();
    let matched = prepare_match_query(
        &conn,
        "hello world",
        "nonexistent_vocab",
        &QueryNormalizationConfig::default(),
    )
    .unwrap();
    assert_eq!(
        clean_for_trigram(&matched).as_deref(),
        Some("\"hello\" \"world\"")
    );
}

#[test]
fn parse_fts_segments_recovers_rurico_or_group_wire_format() {
    // ADR-0008 round-trip: rurico is the producer-of-truth for the MATCH wire
    // format. A vocab-expanded short term ("au") makes rurico emit a
    // `( "audit" OR "authentication" OR "authorization" )` OR-group joined by
    // `" AND "` to the fixed `"login"`. amici's parse_fts_segments must recover
    // that group and the fixed term. If rurico drifts the `" OR "` separator,
    // the `( )` wrapping, or the `"..."` quoting, the recovered segments diverge
    // and this test fails on the next rev bump.
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE fts_chunks USING fts5(content);
         INSERT INTO fts_chunks(content) VALUES ('authentication login session');
         INSERT INTO fts_chunks(content) VALUES ('authorization permission role');
         INSERT INTO fts_chunks(content) VALUES ('audit logging trace');
         CREATE VIRTUAL TABLE fts_chunks_vocab USING fts5vocab(fts_chunks, row);",
    )
    .unwrap();

    let matched = prepare_match_query(
        &conn,
        "au login",
        "fts_chunks_vocab",
        &QueryNormalizationConfig::default(),
    )
    .unwrap();

    // Precondition: rurico actually emitted the OR-group wire format.
    let wire = matched.as_str();
    assert!(wire.contains(" OR "), "expected OR-group separator: {wire}");
    assert!(wire.contains('('), "expected OR-group parens: {wire}");

    let (fixed, or_groups) = parse_fts_segments(wire);

    assert_eq!(fixed, vec!["\"login\"".to_owned()]);
    assert_eq!(or_groups.len(), 1, "one expanded OR-group: {wire}");
    // Term order within the group is unspecified — vocab expansion orders by
    // `cnt DESC` then sqlite row order, so the consumer treats it as a set.
    // Sort before comparing membership.
    let mut group = or_groups[0].clone();
    group.sort();
    assert_eq!(
        group,
        vec![
            "\"audit\"".to_owned(),
            "\"authentication\"".to_owned(),
            "\"authorization\"".to_owned(),
        ]
    );
}
