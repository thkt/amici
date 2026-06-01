//! Connection-bound row collectors and bulk-fetch helpers.
//!
//! Where [`crate::storage::filter`] builds the `WHERE` clause text and
//! parameter list, this module owns the *execution* side — taking a
//! [`rusqlite::Connection`], running a prepared statement, and collecting
//! rows into the caller's chosen collection type.
//!
//! Both helpers are generic over `E: From<rusqlite::Error>`, so they
//! integrate with `thiserror` enums that wrap `rusqlite::Error` via
//! `#[from]`, as well as with `anyhow::Error` (which provides
//! `From<rusqlite::Error>` out of the box) and `rusqlite::Error` itself.

use std::iter;

use rusqlite::{Connection, Row, params_from_iter, types::ToSql};

use crate::storage::filter::anon_placeholders;

/// Collects an iterator of `Result<T, rusqlite::Error>` into any
/// [`FromIterator`]-implementing collection.
///
/// Replaces the recurring `rows.collect::<Result<C, _>>().map_err(Into::into)`
/// pattern at row-collection sites.
///
/// # Examples
///
/// Use within a function whose return type fixes `E`. The compiler infers
/// `E` from the surrounding `Result<_, E>`, so callers never write the
/// turbofish themselves:
///
/// ```
/// use amici::storage::collect_rows;
/// use rusqlite::Connection;
///
/// #[derive(Debug, thiserror::Error)]
/// enum AppError {
///     #[error(transparent)]
///     Db(#[from] rusqlite::Error),
/// }
///
/// fn load_ids(conn: &Connection) -> Result<Vec<i64>, AppError> {
///     let mut stmt = conn.prepare("SELECT id FROM t ORDER BY id")?;
///     let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
///     collect_rows(rows)
/// }
///
/// let conn = Connection::open_in_memory().unwrap();
/// conn.execute("CREATE TABLE t (id INTEGER)", []).unwrap();
/// conn.execute("INSERT INTO t VALUES (1), (2), (3)", []).unwrap();
/// assert_eq!(load_ids(&conn).unwrap(), vec![1, 2, 3]);
/// ```
///
/// # Errors
///
/// Returns the first row error encountered during collection, converted into
/// `E` via [`From<rusqlite::Error>`].
pub fn collect_rows<I, T, C, E>(rows: I) -> Result<C, E>
where
    I: Iterator<Item = Result<T, rusqlite::Error>>,
    C: FromIterator<T>,
    E: From<rusqlite::Error>,
{
    rows.collect::<Result<C, _>>().map_err(E::from)
}

/// Runs a SQL `IN ({placeholders})` query against `keys` and collects rows
/// into any [`FromIterator`]-implementing collection.
///
/// `sql_template` must contain the literal token `{placeholders}`, which is
/// replaced with anonymous `?, ?, ?` matching `keys.len()` before
/// [`Connection::prepare`]. When `keys.is_empty()`, no SQL is executed and
/// an empty `C` is returned — this also avoids SQLite's syntax error on
/// `IN ()`.
///
/// # Examples
///
/// Bulk-fetch a name lookup keyed by primary key. `E` is inferred from the
/// enclosing function's return type:
///
/// ```
/// use std::collections::HashMap;
/// use amici::storage::fetch_by_in_clause;
/// use rusqlite::Connection;
///
/// #[derive(Debug, thiserror::Error)]
/// enum AppError {
///     #[error(transparent)]
///     Db(#[from] rusqlite::Error),
/// }
///
/// fn load_names(conn: &Connection, ids: &[i64]) -> Result<HashMap<i64, String>, AppError> {
///     fetch_by_in_clause(
///         conn,
///         ids,
///         "SELECT id, name FROM t WHERE id IN ({placeholders})",
///         |row| Ok((row.get(0)?, row.get(1)?)),
///     )
/// }
///
/// let conn = Connection::open_in_memory().unwrap();
/// conn.execute("CREATE TABLE t (id INTEGER, name TEXT)", []).unwrap();
/// conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')", []).unwrap();
/// let map = load_names(&conn, &[1, 3]).unwrap();
/// assert_eq!(map[&1], "a");
/// assert_eq!(map[&3], "c");
/// assert!(!map.contains_key(&2));
/// ```
///
/// # Errors
///
/// Returns any error from [`Connection::prepare`],
/// [`rusqlite::Statement::query_map`], or row collection, converted into `E`
/// via [`From<rusqlite::Error>`]. In particular, [`Connection::prepare`]
/// fails when `keys.len()` exceeds SQLite's `SQLITE_MAX_VARIABLE_NUMBER`
/// (default 32766 in SQLite ≥ 3.32.0, 999 in older builds). The caller is
/// responsible for chunking larger key sets — this helper does not split
/// internally.
pub fn fetch_by_in_clause<P, T, C, E, F>(
    conn: &Connection,
    keys: &[P],
    sql_template: &str,
    map_row: F,
) -> Result<C, E>
where
    P: ToSql,
    F: FnMut(&Row<'_>) -> Result<T, rusqlite::Error>,
    C: FromIterator<T>,
    E: From<rusqlite::Error>,
{
    if keys.is_empty() {
        return Ok(iter::empty::<T>().collect());
    }
    let placeholders = anon_placeholders(keys.len());
    let sql = sql_template.replace("{placeholders}", &placeholders);
    let mut stmt = conn.prepare(&sql).map_err(E::from)?;
    let rows = stmt
        .query_map(params_from_iter(keys.iter()), map_row)
        .map_err(E::from)?;
    collect_rows(rows)
}

#[cfg(test)]
mod tests;
