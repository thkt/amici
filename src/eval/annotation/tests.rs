//! Unit tests for [`crate::eval::annotation`] (Issue #53 sub-PR-A and sub-PR-B).
//!
//! Sub-PR-A: T-001..T-004 in
//! `docs/spec/2026-05-08-issue-53-annotation-foundation/spec.md`. The
//! Spec's Skip rationale (FR-V002 Serialise wrapper) explicitly defers
//! that case to the implicit serde path coverage from T-002.
//!
//! Sub-PR-B: T-002 (this file's `session_write_json_round_trips_through_tempdir`
//! and `annotation_error_io_variant_is_reachable_via_from`) in
//! `.claude/workspace/planning/2026-05-09-issue-53-annotate-subcommand/spec.md`.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;

use tempfile::tempdir;

use super::*;
use crate::eval::fixture::EvalQuery;

/// Build a stub [`Provenance`] carrying the given `schema_version`. All
/// other fields take literal stub values that the test under exercise
/// does not inspect.
fn make_provenance(schema_version: &str) -> Provenance {
    Provenance {
        schema_version: schema_version.to_owned(),
        captured_with: "annotation-tests".to_owned(),
        timestamp: "epoch:0".to_owned(),
        annotator_id: "test-annotator".to_owned(),
        session_id: "test-session-0".to_owned(),
        queries_jsonl_hash: "fnv1a64:0".to_owned(),
    }
}

/// Build a stub [`Entry`] with a single relevance grade and the supplied
/// annotation note text. `mode` is fixed to [`BlockMode::Standard`] (the
/// only Phase 1 variant per FR-006).
fn make_entry(id: &str, annotation_note: &str) -> Entry {
    let mut relevance_map = BTreeMap::new();
    relevance_map.insert("d1".to_owned(), 2u8);
    Entry {
        id: id.to_owned(),
        text: "stub entry text".to_owned(),
        category: "C1".to_owned(),
        relevance_map,
        annotation_note: annotation_note.to_owned(),
        mode: BlockMode::Standard,
    }
}

/// Build a one-entry [`Session`] with provenance carrying
/// `schema_version`.
fn make_session(schema_version: &str) -> Session {
    Session {
        provenance: make_provenance(schema_version),
        entries: vec![make_entry("q1", "block-mode judgment")],
    }
}

// T-001: eval_query_annotation_and_entry_annotation_note_coexist
// FR-005: EvalQuery.annotation and Entry.annotation_note coexist in
//         shared scope without rename or shadowing.
#[test]
fn eval_query_annotation_and_entry_annotation_note_coexist() {
    let mut relevance_map = HashMap::new();
    relevance_map.insert("d1".to_owned(), 1u8);
    let query = EvalQuery {
        id: "q1".to_owned(),
        text: "alpha".to_owned(),
        category: "C1".to_owned(),
        relevance_map,
        annotation: "existing relevance note".to_owned(),
    };
    let entry = make_entry("q1", "new block-mode judgment");

    assert_eq!(
        query.annotation, "existing relevance note",
        "EvalQuery.annotation must preserve the existing relevance-rationale field semantics"
    );
    assert_eq!(
        entry.annotation_note, "new block-mode judgment",
        "Entry.annotation_note must be readable alongside EvalQuery.annotation without rename"
    );
}

// T-002: session_round_trips_through_serde_json
// FR-003: Session encodes via serde_json and decodes back to a
//         PartialEq-equal value (pins envelope shape against silent
//         field rename/drop).
#[test]
fn session_round_trips_through_serde_json() {
    let original = make_session(ANNOTATION_SCHEMA_VERSION);

    let json = serde_json::to_string(&original).expect("serialise Session to JSON");
    let parsed: Session = serde_json::from_str(&json).expect("deserialise Session from JSON");

    assert_eq!(
        parsed, original,
        "Session must round-trip through serde_json under PartialEq"
    );
}

// T-003: validate_schema_version_rejects_mismatched_label
// FR-V001: Session::validate_schema_version returns
//          AnnotationError::SchemaVersionMismatch { got, expected }
//          when provenance schema_version != ANNOTATION_SCHEMA_VERSION.
#[test]
fn validate_schema_version_rejects_mismatched_label() {
    let session = make_session("0.9");

    let result = session.validate_schema_version();

    assert!(
        matches!(
            &result,
            Err(AnnotationError::SchemaVersionMismatch { got, expected })
                if got == "0.9" && expected == "1.0"
        ),
        "FR-V001: schema_version \"0.9\" must yield SchemaVersionMismatch \
         {{ got: \"0.9\", expected: \"1.0\" }}; got: {result:?}"
    );
}

// T-004b: entry_mode_is_reachable_as_block_mode_standard
// FR-006: BlockMode::Standard is reachable as the sole Phase 1
//         authoring strategy variant via `matches!`.
#[test]
fn entry_mode_is_reachable_as_block_mode_standard() {
    let session = make_session(ANNOTATION_SCHEMA_VERSION);
    assert!(
        matches!(session.entries[0].mode, BlockMode::Standard),
        "FR-006: Entry.mode must be reachable as BlockMode::Standard; \
         got: {:?}",
        session.entries[0].mode
    );
}

// T-002 (sub-PR-B): session_write_json_round_trips_through_tempdir
// FR-004: Session::write_json writes pretty JSON via the shared
//         atomic-write path (eval::io::write_json). A stub Session
//         routed through Session::write_json must deserialise back to
//         a PartialEq-equal value.
#[test]
fn session_write_json_round_trips_through_tempdir() {
    let dir = tempdir().expect("tempdir for Session::write_json round-trip");
    let path = dir.path().join("session.json");
    let original = make_session(ANNOTATION_SCHEMA_VERSION);

    original
        .write_json(&path)
        .expect("Session::write_json must succeed");

    let body = fs::read_to_string(&path).expect("read session.json");
    let parsed: Session = serde_json::from_str(&body).expect("deserialise session.json");
    assert_eq!(
        parsed, original,
        "round-trip via Session::write_json must preserve PartialEq equality"
    );
}

// T-002 (sub-PR-B): annotation_error_io_variant_is_reachable_via_from
// FR-003: AnnotationError gains an Io variant carrying std::io::Error
//         with #[from] for `?`-propagation. Confirms the variant is
//         reachable via From conversion.
#[test]
fn annotation_error_io_variant_is_reachable_via_from() {
    let io_err = io::Error::other("stub io error");
    let err: AnnotationError = io_err.into();
    assert!(
        matches!(err, AnnotationError::Io(_)),
        "FR-003: AnnotationError::Io must be reachable via From<io::Error>; \
         got: {err:?}"
    );
}
