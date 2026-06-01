use std::collections::{HashMap, HashSet};

use rusqlite::Connection;

use super::*;

#[derive(Debug, thiserror::Error)]
enum TestError {
    #[error("db: {0}")]
    Db(#[from] rusqlite::Error),
}

fn setup_db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)", [])
        .unwrap();
    conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')", [])
        .unwrap();
    conn
}

// T-055: collect_rows_empty_iterator_returns_empty_collection
#[test]
fn collect_rows_empty_iterator_returns_empty_collection() {
    let rows: iter::Empty<Result<i64, rusqlite::Error>> = iter::empty();
    let v: Vec<i64> = collect_rows::<_, _, _, rusqlite::Error>(rows).unwrap();
    assert!(v.is_empty());
}

// T-056: collect_rows_collects_vec_from_query_map
#[test]
fn collect_rows_collects_vec_from_query_map() {
    let conn = setup_db();
    let mut stmt = conn.prepare("SELECT id FROM t ORDER BY id").unwrap();
    let rows = stmt.query_map([], |row| row.get::<_, i64>(0)).unwrap();
    let ids: Vec<i64> = collect_rows::<_, _, _, rusqlite::Error>(rows).unwrap();
    assert_eq!(ids, vec![1, 2, 3]);
}

// T-057: collect_rows_collects_hash_map_from_query_map
#[test]
fn collect_rows_collects_hash_map_from_query_map() {
    let conn = setup_db();
    let mut stmt = conn.prepare("SELECT id, name FROM t").unwrap();
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .unwrap();
    let map: HashMap<i64, String> = collect_rows::<_, _, _, rusqlite::Error>(rows).unwrap();
    assert_eq!(map.len(), 3);
    assert_eq!(map[&1], "a");
    assert_eq!(map[&2], "b");
    assert_eq!(map[&3], "c");
}

// T-058: collect_rows_propagates_error_via_from_impl
#[test]
fn collect_rows_propagates_error_via_from_impl() {
    let conn = setup_db();
    // 'name' column is TEXT; reading it as i64 forces a type-mismatch
    // error per row, exercising the From<rusqlite::Error> conversion.
    let mut stmt = conn.prepare("SELECT name FROM t").unwrap();
    let rows = stmt.query_map([], |row| row.get::<_, i64>(0)).unwrap();
    let result: Result<Vec<i64>, TestError> = collect_rows(rows);
    assert!(
        matches!(result, Err(TestError::Db(_))),
        "expected TestError::Db, got: {result:?}"
    );
}

// T-059: fetch_by_in_clause_empty_keys_returns_empty_collection_without_sql
#[test]
fn fetch_by_in_clause_empty_keys_returns_empty_collection_without_sql() {
    let conn = setup_db();
    // Invalid SQL — if the early return doesn't fire, prepare will error.
    let keys: &[i64] = &[];
    let map: HashMap<i64, String> = fetch_by_in_clause::<_, _, _, rusqlite::Error, _>(
        &conn,
        keys,
        "this is not valid SQL {placeholders}",
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .unwrap();
    assert!(map.is_empty());
}

// T-060: fetch_by_in_clause_non_empty_keys_collects_hash_map
#[test]
fn fetch_by_in_clause_non_empty_keys_collects_hash_map() {
    let conn = setup_db();
    let keys: &[i64] = &[1, 3];
    let map: HashMap<i64, String> = fetch_by_in_clause::<_, _, _, rusqlite::Error, _>(
        &conn,
        keys,
        "SELECT id, name FROM t WHERE id IN ({placeholders})",
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map[&1], "a");
    assert_eq!(map[&3], "c");
    assert!(!map.contains_key(&2));
}

// T-061: fetch_by_in_clause_partial_match_returns_only_existing_rows
#[test]
fn fetch_by_in_clause_partial_match_returns_only_existing_rows() {
    let conn = setup_db();
    // 5 keys, only 2 match.
    let keys: &[i64] = &[1, 99, 2, 100, 101];
    let set: HashSet<i64> = fetch_by_in_clause::<_, _, _, rusqlite::Error, _>(
        &conn,
        keys,
        "SELECT id FROM t WHERE id IN ({placeholders})",
        |row| row.get(0),
    )
    .unwrap();
    assert_eq!(set, HashSet::from([1, 2]));
}

// T-062: fetch_by_in_clause_invalid_template_propagates_prepare_error
#[test]
fn fetch_by_in_clause_invalid_template_propagates_prepare_error() {
    let conn = setup_db();
    let keys: &[i64] = &[1];
    let result: Result<Vec<(i64, String)>, TestError> =
        fetch_by_in_clause(&conn, keys, "INVALID SQL {placeholders}", |row| {
            Ok((row.get(0)?, row.get(1)?))
        });
    assert!(
        matches!(result, Err(TestError::Db(_))),
        "expected TestError::Db, got: {result:?}"
    );
}
