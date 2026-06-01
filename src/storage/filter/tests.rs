use super::*;

// T-012: in_placeholders_numbered
#[test]
fn in_placeholders_numbered() {
    assert_eq!(in_placeholders(3), "?1, ?2, ?3");
    assert_eq!(in_placeholders(0), "");
}

// T-013: anon_placeholders_anonymous
#[test]
fn anon_placeholders_anonymous() {
    assert_eq!(anon_placeholders(3), "?, ?, ?");
    assert_eq!(anon_placeholders(0), "");
}

// T-014: as_sql_params_len_matches
#[test]
fn as_sql_params_len_matches() {
    let values = ["a", "b"];
    assert_eq!(as_sql_params(&values).len(), 2);
}

// T-015: append_eq_filter_some_appends
#[test]
fn append_eq_filter_some_appends() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_eq_filter(&mut sql, &mut params, "p.category", Some("x"));
    assert_eq!(sql, "SELECT 1 AND p.category = ?");
    assert_eq!(params.len(), 1);
}

// T-016: append_eq_filter_none_noop
#[test]
fn append_eq_filter_none_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_eq_filter(&mut sql, &mut params, "p.category", None);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-017: append_eq_filter_two_consecutive_filters
#[test]
fn append_eq_filter_two_consecutive_filters() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_eq_filter(&mut sql, &mut params, "p.category", Some("book"));
    append_eq_filter(&mut sql, &mut params, "p.lang", Some("ja"));
    assert_eq!(sql, "SELECT 1 AND p.category = ? AND p.lang = ?");
    assert_eq!(params.len(), 2);
}

// T-018: escape_like_escapes_metachars
#[test]
fn escape_like_escapes_metachars() {
    assert_eq!(escape_like("100%"), "100\\%");
    assert_eq!(escape_like("foo_bar"), "foo\\_bar");
    assert_eq!(escape_like("path\\to"), "path\\\\to");
}

// T-019: escape_like_preserves_regular_chars
#[test]
fn escape_like_preserves_regular_chars() {
    assert_eq!(escape_like("hello"), "hello");
    assert_eq!(escape_like(""), "");
    assert_eq!(escape_like("日本語"), "日本語");
}

// T-020: escape_like_order_does_not_double_escape
#[test]
fn escape_like_order_does_not_double_escape() {
    assert_eq!(escape_like("\\%"), "\\\\\\%");
}

// T-021: like_prefix_match_case_insensitive
#[test]
fn like_prefix_match_case_insensitive() {
    assert!(like_prefix_match("HelloWorld", "hello"));
    assert!(like_prefix_match("helloworld", "HELLO"));
    assert!(like_prefix_match("abc", "abc"));
}

// T-022: like_prefix_match_non_matching_returns_false
#[test]
fn like_prefix_match_non_matching_returns_false() {
    assert!(!like_prefix_match("HelloWorld", "world"));
    assert!(!like_prefix_match("abc", "xyz"));
}

// T-023: like_prefix_match_empty_prefix_always_matches
#[test]
fn like_prefix_match_empty_prefix_always_matches() {
    assert!(like_prefix_match("anything", ""));
    assert!(like_prefix_match("", ""));
}

// T-024: like_prefix_match_prefix_longer_than_value_is_false
#[test]
fn like_prefix_match_prefix_longer_than_value_is_false() {
    assert!(!like_prefix_match("ab", "abc"));
    assert!(!like_prefix_match("", "x"));
}

// T-025: append_like_prefix_filter_empty_slice_noop
#[test]
fn append_like_prefix_filter_empty_slice_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_like_prefix_filter(&mut sql, &mut params, "p.path", &[]);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-026: append_like_prefix_filter_single_prefix_appends
#[test]
fn append_like_prefix_filter_single_prefix_appends() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_like_prefix_filter(&mut sql, &mut params, "p.path", &["src/".to_owned()]);
    assert_eq!(sql, "SELECT 1 AND p.path LIKE ? ESCAPE '\\'");
    assert_eq!(params.len(), 1);
}

// T-027: append_like_prefix_filter_multiple_prefixes_or_group
#[test]
fn append_like_prefix_filter_multiple_prefixes_or_group() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_like_prefix_filter(
        &mut sql,
        &mut params,
        "p.path",
        &["src/".to_owned(), "tests/".to_owned()],
    );
    assert_eq!(
        sql,
        "SELECT 1 AND (p.path LIKE ? ESCAPE '\\' OR p.path LIKE ? ESCAPE '\\')"
    );
    assert_eq!(params.len(), 2);
}

// T-028: append_like_prefix_filter_escapes_metachars_and_appends_wildcard
// End-to-end: the helper must escape `%` in the prefix so that the trailing
// wildcard `%` is the only unescaped LIKE metachar. Verifies against a real
// SQLite connection to cover both SQL composition and parameter binding.
#[test]
fn append_like_prefix_filter_escapes_metachars_and_appends_wildcard() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (path TEXT)", []).unwrap();
    conn.execute("INSERT INTO t VALUES ('foo%bar'), ('fooxbar')", [])
        .unwrap();

    let mut sql = "SELECT path FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_like_prefix_filter(&mut sql, &mut params, "path", &["foo%".to_owned()]);

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    // Only 'foo%bar' matches: prefix "foo%" escapes to literal `foo%`, then `%` wildcards anything.
    assert_eq!(rows, vec!["foo%bar".to_owned()]);
}

// T-029: append_like_prefix_filter_consecutive_calls_preserve_state
#[test]
fn append_like_prefix_filter_consecutive_calls_preserve_state() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_like_prefix_filter(&mut sql, &mut params, "p.path", &["a/".to_owned()]);
    append_eq_filter(&mut sql, &mut params, "p.kind", Some("file"));
    assert_eq!(sql, "SELECT 1 AND p.path LIKE ? ESCAPE '\\' AND p.kind = ?");
    assert_eq!(params.len(), 2);
}

// T-030: append_in_filter_none_noop
#[test]
fn append_in_filter_none_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_in_filter::<String>(&mut sql, &mut params, "p.tag", None);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-031: append_in_filter_some_empty_produces_false_clause
#[test]
fn append_in_filter_some_empty_produces_false_clause() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let empty: &[String] = &[];
    append_in_filter(&mut sql, &mut params, "p.tag", Some(empty));
    assert_eq!(sql, "SELECT 1 AND 1 = 0");
    assert!(params.is_empty());
}

// T-032: append_in_filter_some_non_empty_appends_in_clause
#[test]
fn append_in_filter_some_non_empty_appends_in_clause() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let values = ["a".to_owned(), "b".to_owned(), "c".to_owned()];
    append_in_filter(&mut sql, &mut params, "p.tag", Some(&values));
    assert_eq!(sql, "SELECT 1 AND p.tag IN (?, ?, ?)");
    assert_eq!(params.len(), 3);
}

// T-033: append_in_filter_accepts_i64
#[test]
fn append_in_filter_accepts_i64() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let values: [i64; 2] = [42, 99];
    append_in_filter(&mut sql, &mut params, "p.id", Some(&values));
    assert_eq!(sql, "SELECT 1 AND p.id IN (?, ?)");
    assert_eq!(params.len(), 2);
}

// T-034: append_in_filter_binds_values_via_sqlite
#[test]
fn append_in_filter_binds_values_via_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (name TEXT)", []).unwrap();
    conn.execute("INSERT INTO t VALUES ('alice'), ('bob'), ('carol')", [])
        .unwrap();

    let mut sql = "SELECT name FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let picks = ["alice".to_owned(), "carol".to_owned()];
    append_in_filter(&mut sql, &mut params, "name", Some(&picks));

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows, vec!["alice".to_owned(), "carol".to_owned()]);
}

// T-035: append_exclude_ids_empty_noop
#[test]
fn append_exclude_ids_empty_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let empty: HashSet<i64> = HashSet::new();
    append_exclude_ids(&mut sql, &mut params, "p.id", &empty);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-036: append_exclude_ids_non_empty_appends_not_in
#[test]
fn append_exclude_ids_non_empty_appends_not_in() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let ids: HashSet<i64> = [1_i64, 2, 3].into_iter().collect();
    append_exclude_ids(&mut sql, &mut params, "p.id", &ids);
    assert_eq!(sql, "SELECT 1 AND p.id NOT IN (?, ?, ?)");
    assert_eq!(params.len(), 3);
}

// T-037: append_exclude_ids_consecutive_calls_preserve_state
#[test]
fn append_exclude_ids_consecutive_calls_preserve_state() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let ids: HashSet<i64> = [10_i64].into_iter().collect();
    append_exclude_ids(&mut sql, &mut params, "p.id", &ids);
    append_eq_filter(&mut sql, &mut params, "p.kind", Some("note"));
    assert_eq!(sql, "SELECT 1 AND p.id NOT IN (?) AND p.kind = ?");
    assert_eq!(params.len(), 2);
}

// T-038: append_exclude_ids_binds_via_sqlite
#[test]
fn append_exclude_ids_binds_via_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (id INTEGER)", []).unwrap();
    conn.execute("INSERT INTO t VALUES (1), (2), (3), (4)", [])
        .unwrap();

    let mut sql = "SELECT id FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let excluded: HashSet<i64> = [2_i64, 4].into_iter().collect();
    append_exclude_ids(&mut sql, &mut params, "id", &excluded);
    sql.push_str(" ORDER BY id");

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<i64> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows, vec![1_i64, 3]);
}

// T-039: append_include_ids_none_noop
#[test]
fn append_include_ids_none_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_include_ids(&mut sql, &mut params, "p.id", None);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-040: append_include_ids_some_empty_produces_false_clause
#[test]
fn append_include_ids_some_empty_produces_false_clause() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let empty = HashSet::new();
    append_include_ids(&mut sql, &mut params, "p.id", Some(&empty));
    assert_eq!(sql, "SELECT 1 AND 1 = 0");
    assert!(params.is_empty());
}

// T-041: append_include_ids_some_non_empty_appends_in_clause
#[test]
fn append_include_ids_some_non_empty_appends_in_clause() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let ids: HashSet<i64> = [7_i64, 11].into_iter().collect();
    append_include_ids(&mut sql, &mut params, "p.id", Some(&ids));
    assert_eq!(sql, "SELECT 1 AND p.id IN (?, ?)");
    assert_eq!(params.len(), 2);
}

// T-042: append_include_ids_empty_clause_filters_all_rows_via_sqlite
#[test]
fn append_include_ids_empty_clause_filters_all_rows_via_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (id INTEGER)", []).unwrap();
    conn.execute("INSERT INTO t VALUES (1), (2), (3)", [])
        .unwrap();

    let mut sql = "SELECT id FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    let empty: HashSet<i64> = HashSet::new();
    append_include_ids(&mut sql, &mut params, "id", Some(&empty));

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<i64> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    // `Some(empty)` → AND 1=0 → no rows match, per the contract.
    assert!(rows.is_empty());
}

// T-043: append_timestamp_cutoff_filter_none_noop
#[test]
fn append_timestamp_cutoff_filter_none_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_cutoff_filter(&mut sql, &mut params, "p.ts", None);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-044: append_timestamp_cutoff_filter_some_appends_ge
#[test]
fn append_timestamp_cutoff_filter_some_appends_ge() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_cutoff_filter(&mut sql, &mut params, "p.ts", Some(1_700_000_000_000));
    assert_eq!(sql, "SELECT 1 AND p.ts >= ?");
    assert_eq!(params.len(), 1);
}

// T-045: append_timestamp_cutoff_filter_binds_via_sqlite
#[test]
fn append_timestamp_cutoff_filter_binds_via_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (ts INTEGER)", []).unwrap();
    conn.execute("INSERT INTO t VALUES (100), (200), (300)", [])
        .unwrap();

    let mut sql = "SELECT ts FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_cutoff_filter(&mut sql, &mut params, "ts", Some(200));
    sql.push_str(" ORDER BY ts");

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<i64> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows, vec![200_i64, 300]);
}

// T-046: append_date_string_cutoff_filter_none_noop
#[test]
fn append_date_string_cutoff_filter_none_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_date_string_cutoff_filter(&mut sql, &mut params, "p.date", true, None);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-047: append_date_string_cutoff_filter_before_true_appends_le
#[test]
fn append_date_string_cutoff_filter_before_true_appends_le() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_date_string_cutoff_filter(&mut sql, &mut params, "p.date", true, Some("2026-04-23"));
    assert_eq!(sql, "SELECT 1 AND p.date <= ?");
    assert_eq!(params.len(), 1);
}

// T-048: append_date_string_cutoff_filter_before_false_appends_ge
#[test]
fn append_date_string_cutoff_filter_before_false_appends_ge() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_date_string_cutoff_filter(&mut sql, &mut params, "p.date", false, Some("2026-01-01"));
    assert_eq!(sql, "SELECT 1 AND p.date >= ?");
    assert_eq!(params.len(), 1);
}

// T-049: append_date_string_cutoff_filter_both_sides_compose_to_range
#[test]
fn append_date_string_cutoff_filter_both_sides_compose_to_range() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (d TEXT)", []).unwrap();
    conn.execute(
        "INSERT INTO t VALUES ('2026-01-15'), ('2026-03-01'), ('2026-05-20')",
        [],
    )
    .unwrap();

    let mut sql = "SELECT d FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_date_string_cutoff_filter(&mut sql, &mut params, "d", false, Some("2026-02-01"));
    append_date_string_cutoff_filter(&mut sql, &mut params, "d", true, Some("2026-04-30"));
    sql.push_str(" ORDER BY d");

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(rows, vec!["2026-03-01".to_owned()]);
}

// T-050: append_timestamp_day_cutoff_filter_none_noop
#[test]
fn append_timestamp_day_cutoff_filter_none_noop() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_day_cutoff_filter(&mut sql, &mut params, "p.updated_at", true, None);
    assert_eq!(sql, "SELECT 1");
    assert!(params.is_empty());
}

// T-051: append_timestamp_day_cutoff_filter_before_true_appends_lt_plus_one_day
#[test]
fn append_timestamp_day_cutoff_filter_before_true_appends_lt_plus_one_day() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_day_cutoff_filter(
        &mut sql,
        &mut params,
        "p.updated_at",
        true,
        Some("2025-03-01"),
    );
    assert_eq!(sql, "SELECT 1 AND p.updated_at < date(?, '+1 day')");
    assert_eq!(params.len(), 1);
}

// T-052: append_timestamp_day_cutoff_filter_before_false_appends_ge
#[test]
fn append_timestamp_day_cutoff_filter_before_false_appends_ge() {
    let mut sql = "SELECT 1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_day_cutoff_filter(
        &mut sql,
        &mut params,
        "p.updated_at",
        false,
        Some("2025-03-01"),
    );
    assert_eq!(sql, "SELECT 1 AND p.updated_at >= ?");
    assert_eq!(params.len(), 1);
}

// T-053: append_timestamp_day_cutoff_filter_day_inclusive_on_rfc3339_via_sqlite
// Locks in that `date(?, '+1 day')` lifts the upper bound past T-suffix rows.
#[test]
fn append_timestamp_day_cutoff_filter_day_inclusive_on_rfc3339_via_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (updated_at TEXT)", [])
        .unwrap();
    conn.execute(
        "INSERT INTO t VALUES \
             ('2025-02-28T23:59:59+00:00'), \
             ('2025-03-01'), \
             ('2025-03-01T00:00:00+00:00'), \
             ('2025-03-01T12:00:00+00:00'), \
             ('2025-03-01T23:59:59+00:00'), \
             ('2025-03-02T00:00:00+00:00')",
        [],
    )
    .unwrap();

    let mut sql = "SELECT updated_at FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_day_cutoff_filter(
        &mut sql,
        &mut params,
        "updated_at",
        true,
        Some("2025-03-01"),
    );
    sql.push_str(" ORDER BY updated_at");

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        rows,
        vec![
            "2025-02-28T23:59:59+00:00".to_owned(),
            "2025-03-01".to_owned(),
            "2025-03-01T00:00:00+00:00".to_owned(),
            "2025-03-01T12:00:00+00:00".to_owned(),
            "2025-03-01T23:59:59+00:00".to_owned(),
        ]
    );
}

// T-054: append_timestamp_day_cutoff_filter_before_false_start_inclusive_on_rfc3339_via_sqlite
// Locks in that RFC 3339 prefix sort against a bare `YYYY-MM-DD` lower bound works.
#[test]
fn append_timestamp_day_cutoff_filter_before_false_start_inclusive_on_rfc3339_via_sqlite() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (updated_at TEXT)", [])
        .unwrap();
    conn.execute(
        "INSERT INTO t VALUES \
             ('2025-02-28T23:59:59+00:00'), \
             ('2025-03-01'), \
             ('2025-03-01T00:00:00+00:00'), \
             ('2025-03-01T12:00:00+00:00'), \
             ('2025-03-02T00:00:00+00:00')",
        [],
    )
    .unwrap();

    let mut sql = "SELECT updated_at FROM t WHERE 1=1".to_owned();
    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    append_timestamp_day_cutoff_filter(
        &mut sql,
        &mut params,
        "updated_at",
        false,
        Some("2025-03-01"),
    );
    sql.push_str(" ORDER BY updated_at");

    let mut stmt = conn.prepare(&sql).unwrap();
    let rows: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params.iter()), |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        rows,
        vec![
            "2025-03-01".to_owned(),
            "2025-03-01T00:00:00+00:00".to_owned(),
            "2025-03-01T12:00:00+00:00".to_owned(),
            "2025-03-02T00:00:00+00:00".to_owned(),
        ]
    );
}
