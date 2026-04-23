pub mod fts;

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
/// The backslash is replaced first so the subsequent `%`/`_` replacements do
/// not double-escape already-escaped sequences.
pub fn escape_like(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
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
        // Backslash must be escaped first so that `\%` introduced by the `%`
        // replacement is not re-escaped into `\\\%`.
        assert_eq!(escape_like("%"), "\\%");
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
}
