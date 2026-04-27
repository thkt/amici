//! Fixture loaders and validators (FR-005..FR-007).
//!
//! Loads JSONL-formatted evaluation queries, documents, and known-answer
//! fixtures. All loaders surface a typed [`FixtureError`] so callers can
//! distinguish I/O failure, malformed JSON, missing required fields,
//! category-distribution shortfall, and missing known-answer kinds.
//!
//! References:
//! - FR-005 parse `{id, text, category, relevance_map, annotation}`,
//!   fail with typed error when a required field is missing
//! - FR-006 7 categories × ≥20 queries each
//! - FR-007 known_answers.jsonl must contain identity / reverse / single_doc

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Single evaluation query plus its graded relevance map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    /// Stable identifier referenced by `relevance_map` keys in other queries.
    pub id: String,
    /// Surface query text passed to the embedder / FTS.
    pub text: String,
    /// One of seven semantic category labels (FR-006).
    pub category: String,
    /// Maps `doc_id` to graded relevance in `{0, 1, 2, 3}`.
    pub relevance_map: HashMap<String, u8>,
    /// Free-form annotation note describing the relevance judgment basis.
    pub annotation: String,
}

/// Single corpus document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalDocument {
    /// Unique identifier within the corpus.
    pub id: String,
    /// Document title (indexed alongside body).
    pub title: String,
    /// Document body text.
    pub body: String,
    /// Optional category hint emitted by the corpus author.
    pub category_hint: Option<String>,
    /// Source attribution (license + provenance).
    pub source: String,
}

/// Discriminator for a [`KnownAnswerFixture`].
///
/// JSONL serialises in `snake_case` (`identity` / `reverse` / `single_doc`)
/// to match the spec prose and shell-friendly fixture inspection via
/// `jq -r '.kind'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownAnswerKind {
    /// Query identical to the document body (top-1 must hit).
    Identity,
    /// Reverse-ordered ranking; metric should sit near its lower bound.
    Reverse,
    /// Single-document corpus (Recall@1 == MRR == 1.0).
    SingleDoc,
}

/// Mini-corpus + queries for one known-answer kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownAnswerFixture {
    /// Discriminator for this fixture's expected behavior.
    pub kind: KnownAnswerKind,
    /// Documents constituting the mini-corpus.
    pub corpus: Vec<EvalDocument>,
    /// Queries exercising the mini-corpus.
    pub queries: Vec<EvalQuery>,
}

/// Triple of known-answer fixtures required by FR-007.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownAnswerSet {
    /// `Identity` fixture.
    pub identity: KnownAnswerFixture,
    /// `Reverse` fixture.
    pub reverse: KnownAnswerFixture,
    /// `SingleDoc` fixture.
    pub single_doc: KnownAnswerFixture,
}

/// Errors surfaced by the fixture loaders and validators.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum FixtureError {
    /// Underlying I/O failure (file not found, permission denied, etc.).
    #[error("fixture I/O error: {0}")]
    Io(#[from] io::Error),
    /// JSONL line failed to parse as valid JSON.
    #[error("fixture parse error at line {line}: {source}")]
    Parse {
        /// 1-indexed line number where parsing failed.
        line: usize,
        /// Underlying serde_json error (preserved for `source()` chain).
        #[source]
        source: serde_json::Error,
    },
    /// Required field absent from a JSONL record.
    #[error("fixture missing required field {field:?} at line {line}")]
    MissingField {
        /// 1-indexed line number where the field is absent.
        line: usize,
        /// Name of the missing field.
        field: &'static str,
    },
    /// Category distribution constraint (FR-006) violated.
    #[error(
        "fixture category {category:?} has {observed} queries, expected at least {expected_min}"
    )]
    CategoryDistribution {
        /// Category whose count fell below the minimum.
        category: String,
        /// Observed query count for this category.
        observed: usize,
        /// Required minimum query count.
        expected_min: usize,
    },
    /// `known_answers.jsonl` missing one of the three required kinds.
    #[error("fixture missing known-answer kind: {0:?}")]
    MissingKnownAnswerKind(KnownAnswerKind),
}

const REQUIRED_QUERY_FIELDS: &[&str] = &["id", "text", "category", "relevance_map", "annotation"];

const REQUIRED_DOCUMENT_FIELDS: &[&str] = &["id", "title", "body", "source"];

/// FR-006: per-category query count must reach this floor.
const EXPECTED_MIN_PER_CATEGORY: usize = 20;

/// Load `queries.jsonl` into a `Vec<EvalQuery>`.
///
/// # Errors
///
/// Returns [`FixtureError::Io`] for filesystem failure,
/// [`FixtureError::Parse`] for malformed JSON, and
/// [`FixtureError::MissingField`] when a required field is absent.
pub fn load_queries(path: &Path) -> Result<Vec<EvalQuery>, FixtureError> {
    load_jsonl_records(path, REQUIRED_QUERY_FIELDS)
}

/// Load `documents.jsonl` into a `Vec<EvalDocument>`.
///
/// # Errors
///
/// See [`load_queries`] for the typed-error contract.
pub fn load_documents(path: &Path) -> Result<Vec<EvalDocument>, FixtureError> {
    load_jsonl_records(path, REQUIRED_DOCUMENT_FIELDS)
}

/// Load `known_answers.jsonl` into a [`KnownAnswerSet`].
///
/// # Errors
///
/// Returns [`FixtureError::MissingKnownAnswerKind`] when any of `Identity`,
/// `Reverse`, or `SingleDoc` is absent. See [`load_queries`] for I/O and
/// parse errors.
pub fn load_known_answers(path: &Path) -> Result<KnownAnswerSet, FixtureError> {
    let mut identity: Option<KnownAnswerFixture> = None;
    let mut reverse: Option<KnownAnswerFixture> = None;
    let mut single_doc: Option<KnownAnswerFixture> = None;
    each_jsonl_line(path, |line_no, value| {
        let fixture: KnownAnswerFixture =
            serde_json::from_value(value).map_err(|source| FixtureError::Parse {
                line: line_no,
                source,
            })?;
        match fixture.kind {
            KnownAnswerKind::Identity => identity = Some(fixture),
            KnownAnswerKind::Reverse => reverse = Some(fixture),
            KnownAnswerKind::SingleDoc => single_doc = Some(fixture),
        }
        Ok(())
    })?;
    Ok(KnownAnswerSet {
        identity: identity.ok_or(FixtureError::MissingKnownAnswerKind(
            KnownAnswerKind::Identity,
        ))?,
        reverse: reverse.ok_or(FixtureError::MissingKnownAnswerKind(
            KnownAnswerKind::Reverse,
        ))?,
        single_doc: single_doc.ok_or(FixtureError::MissingKnownAnswerKind(
            KnownAnswerKind::SingleDoc,
        ))?,
    })
}

/// Verify that each category in `queries` carries at least
/// [`EXPECTED_MIN_PER_CATEGORY`] queries (FR-006).
///
/// Counts use `BTreeMap` so the first violation reported is alphabetically
/// deterministic across runs (avoids `HashMap` random iteration order).
///
/// # Errors
///
/// Returns [`FixtureError::CategoryDistribution`] for the first category
/// found below the minimum.
pub fn validate_category_distribution(queries: &[EvalQuery]) -> Result<(), FixtureError> {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for query in queries {
        *counts.entry(query.category.as_str()).or_insert(0) += 1;
    }
    for (category, count) in &counts {
        if *count < EXPECTED_MIN_PER_CATEGORY {
            return Err(FixtureError::CategoryDistribution {
                category: (*category).to_owned(),
                observed: *count,
                expected_min: EXPECTED_MIN_PER_CATEGORY,
            });
        }
    }
    Ok(())
}

/// Load every JSONL record into a `Vec<T>`, validating that each line contains
/// every entry in `required_fields` before deserialising into `T`.
fn load_jsonl_records<T: DeserializeOwned>(
    path: &Path,
    required_fields: &[&'static str],
) -> Result<Vec<T>, FixtureError> {
    let mut records: Vec<T> = Vec::new();
    each_jsonl_line(path, |line_no, value| {
        for &field in required_fields {
            if value.get(field).is_none() {
                return Err(FixtureError::MissingField {
                    line: line_no,
                    field,
                });
            }
        }
        let record: T = serde_json::from_value(value).map_err(|source| FixtureError::Parse {
            line: line_no,
            source,
        })?;
        records.push(record);
        Ok(())
    })?;
    Ok(records)
}

/// Visit each non-empty JSONL line at `path`, parsing it into a
/// [`serde_json::Value`] and handing it to `process` along with the 1-indexed
/// line number.
fn each_jsonl_line<F>(path: &Path, mut process: F) -> Result<(), FixtureError>
where
    F: FnMut(usize, serde_json::Value) -> Result<(), FixtureError>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for (idx, line_result) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(&line).map_err(|source| FixtureError::Parse {
                line: line_no,
                source,
            })?;
        process(line_no, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
