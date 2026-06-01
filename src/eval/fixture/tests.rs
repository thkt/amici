use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

/// Write `content` to a `NamedTempFile` and return the handle.
///
/// The handle's `path()` is passed to loaders under test; the file is
/// cleaned up when the handle drops at the end of the test.
fn write_temp_jsonl(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(content.as_bytes()).expect("write temp file");
    file.flush().expect("flush temp file");
    file
}

/// Build an [`EvalQuery`] with the given category. `id`, `text`,
/// `relevance_map`, `annotation` are stub values irrelevant to the
/// distribution check.
fn make_query_with_category(id: &str, category: &str) -> EvalQuery {
    let mut relevance_map = HashMap::new();
    relevance_map.insert("d1".to_owned(), 1u8);
    EvalQuery {
        id: id.to_owned(),
        text: "stub query text".to_owned(),
        category: category.to_owned(),
        relevance_map,
        annotation: "stub annotation".to_owned(),
    }
}

/// Build a `Vec<EvalQuery>` with C1..C6 each carrying 20 queries and
/// C7 carrying only 5 — the FR-006 violation scenario for T-009.
fn make_distribution_with_short_c7() -> Vec<EvalQuery> {
    let mut queries = Vec::with_capacity(125);
    for cat_idx in 1..=6 {
        let category = format!("C{cat_idx}");
        for q_idx in 0..20 {
            let id = format!("q-c{cat_idx}-{q_idx}");
            queries.push(make_query_with_category(&id, &category));
        }
    }
    for q_idx in 0..5 {
        let id = format!("q-c7-{q_idx}");
        queries.push(make_query_with_category(&id, "C7"));
    }
    queries
}

// T-008: load_queries_missing_relevance_map_returns_missing_field_error
// FR-005: malformed JSONL (line 2 missing `relevance_map`) →
//         FixtureError::MissingField { line: 2, field: "relevance_map" }
#[test]
fn load_queries_missing_relevance_map_returns_missing_field_error() {
    let content = concat!(
        r#"{"id":"q1","text":"alpha","category":"C1","relevance_map":{"d1":1},"annotation":"ok"}"#,
        "\n",
        r#"{"id":"q2","text":"beta","category":"C2","annotation":"missing relevance_map"}"#,
        "\n",
    );
    let file = write_temp_jsonl(content);

    let result = load_queries(file.path());

    assert!(
        matches!(
            result,
            Err(FixtureError::MissingField {
                line: 2,
                field: "relevance_map",
            })
        ),
        "expected MissingField {{ line: 2, field: \"relevance_map\" }}, got: {result:?}"
    );
}

// T-009: validate_category_distribution_short_c7_returns_distribution_error
// FR-006: C1..C6 each have 20 queries, C7 has only 5 →
//         FixtureError::CategoryDistribution { category: "C7", observed: 5, expected_min: 20 }
#[test]
fn validate_category_distribution_short_c7_returns_distribution_error() {
    let queries = make_distribution_with_short_c7();

    let result = validate_category_distribution(&queries);

    assert!(
        matches!(
            &result,
            Err(FixtureError::CategoryDistribution {
                category,
                observed: 5,
                expected_min: 20,
            }) if category == "C7"
        ),
        "expected CategoryDistribution {{ category: \"C7\", observed: 5, expected_min: 20 }}, \
             got: {result:?}"
    );
}

// T-010: load_known_answers_missing_reverse_returns_missing_kind_error
// FR-007: known_answers.jsonl contains only `identity` and `single_doc` →
//         FixtureError::MissingKnownAnswerKind(KnownAnswerKind::Reverse)
#[test]
fn load_known_answers_missing_reverse_returns_missing_kind_error() {
    let content = concat!(
        r#"{"kind":"identity","corpus":[],"queries":[]}"#,
        "\n",
        r#"{"kind":"single_doc","corpus":[],"queries":[]}"#,
        "\n",
    );
    let file = write_temp_jsonl(content);

    let result = load_known_answers(file.path());

    assert!(
        matches!(
            result,
            Err(FixtureError::MissingKnownAnswerKind(
                KnownAnswerKind::Reverse
            ))
        ),
        "expected MissingKnownAnswerKind(Reverse), got: {result:?}"
    );
}
