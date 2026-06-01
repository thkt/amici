//! SQL filter construction helpers.
//!
//! Each `append_*` function takes `sql: &mut String` and
//! `params: &mut Vec<Box<dyn ToSql>>`, appending an `AND`-prefixed predicate
//! that relies on the caller's base clause (typically `WHERE 1 = 1`).
//!
//! # Filter contract
//!
//! `Option<T>`-receiving helpers split the empty-input case explicitly:
//!
//! - `None` → no-op (the caller did not request this filter).
//! - `Some(empty)` → `" AND 1 = 0"` (the caller asked for "match nothing").
//! - `Some(non-empty)` → the normal predicate.
//!
//! Helpers receiving a plain non-`Option` collection (e.g. `&HashSet`) treat
//! an empty collection as no-op, because "exclude nothing" or "no LIKE
//! pattern" is the only sensible reading at that call site. New helpers must
//! follow the same split before adding new variants — silently picking a
//! different convention would let callers conflate "no filter" with "match
//! nothing".
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
#[allow(dead_code)] // SQL filter utility kept for future amici-internal callers.
pub(crate) fn like_prefix_match(value: &str, prefix: &str) -> bool {
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

/// Whether [`append_in_clause`] builds an `IN` or `NOT IN` predicate.
///
/// Closed enum (rather than `op: &str`) so the SQL keyword set stays bound at
/// compile time. Matches the module's `column: &'static str` discipline — every
/// literal piece of SQL syntax is fixed at compile time, never threaded
/// through a runtime string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    In,
    NotIn,
}

impl Op {
    fn as_sql(self) -> &'static str {
        match self {
            Self::In => "IN",
            Self::NotIn => "NOT IN",
        }
    }
}

/// Appends `" AND {column} {op} (?, ?, ...)"` and boxes each item as a param.
///
/// Private shared tail for the public `append_in_filter` / `append_include_ids`
/// / `append_exclude_ids` helpers. Each public helper handles its own
/// `None` / empty-set contract before delegating here, so this function
/// assumes `iter` is non-empty.
fn append_in_clause<I>(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    op: Op,
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
    sql.push_str(op.as_sql());
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
    append_in_clause(sql, params, column, Op::In, values.iter().cloned());
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
    append_in_clause(sql, params, column, Op::NotIn, exclude_ids.iter().copied());
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
    append_in_clause(sql, params, column, Op::In, include_ids.iter().copied());
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
#[allow(dead_code)] // SQL filter utility kept for future amici-internal callers.
pub(crate) fn append_date_string_cutoff_filter(
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
mod tests;
