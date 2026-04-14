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
/// Pass only compile-time-known, trusted column names — never a string derived
/// from user input. The `value` argument is always bound as a positional
/// placeholder and is safe.
pub fn append_eq_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &str,
    value: Option<&str>,
) {
    if let Some(v) = value {
        sql.push_str(" AND ");
        sql.push_str(column);
        sql.push_str(" = ?");
        params.push(Box::new(v.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-012: in_placeholders(3) → "?1, ?2, ?3"
    #[test]
    fn in_placeholders_numbered() {
        assert_eq!(in_placeholders(3), "?1, ?2, ?3");
        assert_eq!(in_placeholders(0), "");
    }

    // T-013: anon_placeholders(3) → "?, ?, ?"
    #[test]
    fn anon_placeholders_anonymous() {
        assert_eq!(anon_placeholders(3), "?, ?, ?");
        assert_eq!(anon_placeholders(0), "");
    }

    // T-014: as_sql_params — len matches input
    #[test]
    fn as_sql_params_len_matches() {
        let values = ["a", "b"];
        assert_eq!(as_sql_params(&values).len(), 2);
    }

    // T-015: append_eq_filter Some(v) → sql 追加・params 追加
    #[test]
    fn append_eq_filter_some_appends() {
        let mut sql = "SELECT 1".to_string();
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();
        append_eq_filter(&mut sql, &mut params, "p.category", Some("x"));
        assert_eq!(sql, "SELECT 1 AND p.category = ?");
        assert_eq!(params.len(), 1);
    }

    // T-016: append_eq_filter None → 変化なし
    #[test]
    fn append_eq_filter_none_noop() {
        let mut sql = "SELECT 1".to_string();
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();
        append_eq_filter(&mut sql, &mut params, "p.category", None);
        assert_eq!(sql, "SELECT 1");
        assert!(params.is_empty());
    }

    // T-017: append_eq_filter を2回連続呼び出し → SQL と params.len が両方正しく積まれる
    #[test]
    fn append_eq_filter_two_consecutive_filters() {
        let mut sql = "SELECT 1".to_string();
        let mut params: Vec<Box<dyn ToSql>> = Vec::new();
        append_eq_filter(&mut sql, &mut params, "p.category", Some("book"));
        append_eq_filter(&mut sql, &mut params, "p.lang", Some("ja"));
        assert_eq!(sql, "SELECT 1 AND p.category = ? AND p.lang = ?");
        assert_eq!(params.len(), 2);
    }
}
