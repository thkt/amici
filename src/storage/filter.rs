//! SQL filter construction helpers.
//!
//! Each `append_*` function takes `sql: &mut String` and
//! `params: &mut Vec<Box<dyn ToSql>>`, appending an `AND`-prefixed predicate
//! that relies on the caller's base clause (typically `WHERE 1 = 1`).
//!
//! # Security
//!
//! All `column` parameters are `&'static str`. The compiler rejects runtime
//! strings, so only compile-time literals reach the query — SQL injection via
//! column name is impossible by construction. Values are always bound as
//! positional placeholders.

use std::collections::HashSet;

use rusqlite::types::ToSql;

/// Returns numbered placeholders without parentheses: `"?1, ?2, ?3"` for len=3.
/// Returns `""` for len=0.
pub fn in_placeholders(len: usize) -> String {
    (1..=len)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Returns anonymous placeholders without parentheses: `"?, ?, ?"` for n=3.
/// Returns `""` for n=0.
///
/// Prefer over [`in_placeholders`] when multiple IN clauses share a parameter list:
/// unnamed `?` avoids index collisions that numbered `?N` would cause when params
/// are appended incrementally.
pub fn anon_placeholders(n: usize) -> String {
    vec!["?"; n].join(", ")
}

/// Returns a vec of borrowed `&dyn ToSql` references for use with rusqlite params.
pub fn as_sql_params<T: ToSql>(values: &[T]) -> Vec<&dyn ToSql> {
    values.iter().map(|v| v as &dyn ToSql).collect()
}

/// Appends ` AND {column} = ?` to `sql` and pushes the value into `params`
/// when `value` is `Some`. Does nothing when `value` is `None`.
///
/// # Security
///
/// `column` is interpolated directly into the SQL string without parameterization.
/// The `&'static str` type enforces that only compile-time string literals can be
/// passed — runtime strings and `format!(...)` results are rejected by the
/// compiler. The `value` argument is always bound as a positional placeholder and
/// is safe.
pub fn append_eq_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    value: Option<&str>,
) {
    if let Some(v) = value {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(" = ?");
        params.push(Box::new(v.to_owned()));
    }
}

/// Escapes LIKE metacharacters (`%`, `_`, `\`) in `s` for use with `ESCAPE '\'`.
///
/// Prepends a single backslash before each metacharacter in one pass, so
/// backslashes inserted by the escape never themselves become escape targets.
pub fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '\\' | '%' | '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Returns `true` iff `prefix` matches the leading bytes of `value` under
/// ASCII case-insensitive comparison, mirroring SQLite LIKE semantics.
///
/// An empty `prefix` always matches. When `prefix` is longer than `value`, the
/// result is `false`.
///
/// Use this on the Rust side when post-filtering rows already narrowed by a
/// `LIKE ? ESCAPE '\'` clause, so that SQL and Rust agree on case semantics.
pub fn like_prefix_match(value: &str, prefix: &str) -> bool {
    value
        .as_bytes()
        .get(..prefix.len())
        .is_some_and(|p| p.eq_ignore_ascii_case(prefix.as_bytes()))
}

/// Appends an `AND` clause matching `column` against any of `prefixes` via
/// `LIKE ? ESCAPE '\'`. Each prefix is [`escape_like`]-escaped before the
/// trailing `%` wildcard is appended, so metacharacters in the prefix are
/// treated as literals.
///
/// - Empty `prefixes` → no-op.
/// - Single prefix → `" AND {column} LIKE ? ESCAPE '\\'"`.
/// - Multiple prefixes → `" AND ({column} LIKE ? ESCAPE '\\' OR ... )"`.
///
/// See [`append_eq_filter`] for the `column` security contract.
pub fn append_like_prefix_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    prefixes: &[String],
) {
    if prefixes.is_empty() {
        return;
    }
    sql.push_str(" AND ");
    let multiple = prefixes.len() > 1;
    if multiple {
        sql.push('(');
    }
    for (i, prefix) in prefixes.iter().enumerate() {
        if i > 0 {
            sql.push_str(" OR ");
        }
        sql.push_str(column);
        sql.push_str(" LIKE ? ESCAPE '\\'");
        let mut pattern = escape_like(prefix);
        pattern.push('%');
        params.push(Box::new(pattern));
    }
    if multiple {
        sql.push(')');
    }
}

/// Appends `" AND {column} {op} (?, ?, ...)"` and boxes each item as a param.
///
/// Private shared tail for the public `append_in_filter` / `append_include_ids`
/// / `append_exclude_ids` helpers. Each public helper handles its own
/// `None` / empty-set contract before delegating here, so this function
/// assumes `iter` is non-empty. `op` is used verbatim ("IN" or "NOT IN").
fn append_in_clause<I>(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    op: &str,
    iter: I,
) where
    I: IntoIterator,
    I::IntoIter: ExactSizeIterator,
    I::Item: ToSql + 'static,
{
    let iter = iter.into_iter();
    sql.push_str(" AND ");
    sql.push_str(column);
    sql.push(' ');
    sql.push_str(op);
    sql.push_str(" (");
    sql.push_str(&anon_placeholders(iter.len()));
    sql.push(')');
    for v in iter {
        params.push(Box::new(v));
    }
}

/// Appends `" AND {column} IN (?, ?, ...)"` when `values` is `Some(non-empty)`.
///
/// - `None` → no-op (the filter is absent).
/// - `Some(&[])` → `" AND 1 = 0"` (the filter is present but impossible to
///   satisfy; callers passed an explicit empty set).
/// - `Some(non-empty)` → `IN` clause with one placeholder per value.
///
/// The `None` vs `Some(empty)` split mirrors [`append_include_ids`]. Callers
/// that want "no filter on empty" should pass `None`; the helper never guesses
/// that intent from an empty slice.
///
/// See [`append_eq_filter`] for the `column` security contract.
pub fn append_in_filter<T>(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    values: Option<&[T]>,
) where
    T: ToSql + Clone + 'static,
{
    let Some(values) = values else {
        return;
    };
    if values.is_empty() {
        sql.push_str(" AND 1 = 0");
        return;
    }
    append_in_clause(sql, params, column, "IN", values.iter().cloned());
}

/// Appends `" AND {column} NOT IN (?, ?, ...)"` when `exclude_ids` is non-empty.
///
/// - Empty set → no-op (excluding nothing means no filter is applied).
///
/// Iteration order over a `HashSet` is not stable, so callers must not rely on
/// a specific parameter ordering when asserting against the SQL string.
///
/// See [`append_eq_filter`] for the `column` security contract.
pub fn append_exclude_ids(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    exclude_ids: &HashSet<i64>,
) {
    if exclude_ids.is_empty() {
        return;
    }
    append_in_clause(sql, params, column, "NOT IN", exclude_ids.iter().copied());
}

/// Appends an `AND` clause restricting `column` to `include_ids`.
///
/// - `None` → no-op (no restriction requested).
/// - `Some(&empty)` → `" AND 1 = 0"` (an explicit empty allow-list matches no
///   rows; the caller asked for "only these" with an empty set).
/// - `Some(non-empty)` → `" AND {column} IN (?, ?, ...)"`.
///
/// The split between `None` and `Some(empty)` lets callers distinguish "no
/// include filter" from "include nothing". See [`append_in_filter`] for the
/// same contract applied to arbitrary values.
///
/// Iteration order over a `HashSet` is not stable. See [`append_eq_filter`] for
/// the `column` security contract.
pub fn append_include_ids(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    include_ids: Option<&HashSet<i64>>,
) {
    let Some(include_ids) = include_ids else {
        return;
    };
    if include_ids.is_empty() {
        sql.push_str(" AND 1 = 0");
        return;
    }
    append_in_clause(sql, params, column, "IN", include_ids.iter().copied());
}

/// Appends `" AND {column} >= ?"` binding `cutoff_ms` when `Some`.
///
/// The unit is milliseconds since the Unix epoch — this is recall's native
/// timestamp format. For `<=` comparisons or non-ms units, compose with
/// [`append_date_string_cutoff_filter`] or a caller-side clause.
///
/// - `None` → no-op.
///
/// See [`append_eq_filter`] for the `column` security contract.
pub fn append_timestamp_cutoff_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    cutoff_ms: Option<i64>,
) {
    if let Some(cutoff) = cutoff_ms {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(" >= ?");
        params.push(Box::new(cutoff));
    }
}

/// Appends a date cutoff comparison for textual (`ISO 8601`) date columns.
///
/// - `date_iso` `None` → no-op.
/// - `before = true`  → `" AND {column} <= ?"` (rows at or before the cutoff).
/// - `before = false` → `" AND {column} >= ?"` (rows at or after the cutoff).
///
/// Intended for SQLite `TEXT` columns storing dates like `"2026-04-23"`, where
/// lexical ordering coincides with chronological ordering. Use
/// [`append_timestamp_cutoff_filter`] for integer millisecond columns, or
/// [`append_timestamp_day_cutoff_filter`] when the column may hold RFC 3339
/// timestamps and the caller wants a day-inclusive `before`.
///
/// See [`append_eq_filter`] for the `column` security contract.
pub fn append_date_string_cutoff_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    before: bool,
    date_iso: Option<&str>,
) {
    if let Some(date) = date_iso {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(if before { " <= ?" } else { " >= ?" });
        params.push(Box::new(date.to_owned()));
    }
}

/// Appends a day-inclusive cutoff comparison for RFC 3339 timestamp columns.
///
/// - `date_iso` `None` → no-op.
/// - `before = true`  → `" AND {column} < date(?, '+1 day')"` (rows whose
///   timestamp falls on or before the cutoff day, inclusive of T-suffix
///   values that lexically follow the bare `YYYY-MM-DD` string).
/// - `before = false` → `" AND {column} >= ?"` (rows whose timestamp is on or
///   after the cutoff day; RFC 3339's `YYYY-MM-DDTHH:MM:SS±HH:MM` prefix
///   already sorts correctly against the bare date).
///
/// Intended for SQLite `TEXT` columns storing RFC 3339 timestamps such as
/// `"2025-03-01T12:00:00+00:00"`. Use [`append_date_string_cutoff_filter`]
/// when the column is guaranteed to be date-only (plain `<= ?` suffices), or
/// [`append_timestamp_cutoff_filter`] for integer millisecond columns.
///
/// See [`append_eq_filter`] for the `column` security contract.
pub fn append_timestamp_day_cutoff_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    before: bool,
    date_iso: Option<&str>,
) {
    if let Some(date) = date_iso {
        sql.push_str(" AND ");
        sql.push_str(column);
        if before {
            sql.push_str(" < date(?, '+1 day')");
        } else {
            sql.push_str(" >= ?");
        }
        params.push(Box::new(date.to_owned()));
    }
}

#[cfg(test)]
mod tests {
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
        append_date_string_cutoff_filter(
            &mut sql,
            &mut params,
            "p.date",
            false,
            Some("2026-01-01"),
        );
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
}
