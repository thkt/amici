//! Subprocess integration tests for the `eval_harness annotate` subcommand
//! (Issue #53 sub-PR-B).
//!
//! Maps to spec Test Scenarios T-003..T-013 in
//! `.claude/workspace/planning/2026-05-09-issue-53-annotate-subcommand/spec.md`.
//!
//! Tests are gated behind `eval-harness`. Unlike `tests/eval_smoke.rs`, none
//! of these carry `#[ignore]`: Phase 1 block-mode authoring is MLX-free
//! (NFR-001) and runs in the default `cargo test --features eval-harness`
//! lane on any host.

#![cfg(feature = "eval-harness")]

use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use amici::eval::annotation::{ANNOTATION_SCHEMA_VERSION, Session};
use amici::eval::fixture::{EvalQuery, load_queries};
use tempfile::tempdir;

const EXIT_USAGE: i32 = 2;

/// Spawn `eval_harness annotate` with `argv` arguments, write `stdin_bytes`
/// to its stdin pipe, and capture the resulting [`std::process::Output`].
fn spawn_annotate(argv: &[&str], stdin_bytes: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_eval_harness"))
        .arg("annotate")
        .args(argv)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn eval_harness annotate");
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("annotate stdin must be piped for integration tests");
        stdin
            .write_all(stdin_bytes)
            .expect("write stdin to annotate child");
    }
    child
        .wait_with_output()
        .expect("wait for eval_harness annotate")
}

// T-011a: t011a_annotate_missing_output_argv_exits_with_usage
// FR-012: missing output= → EXIT_USAGE (2) + "annotate: output= argument required"
#[test]
fn t011a_annotate_missing_output_argv_exits_with_usage() {
    let output = spawn_annotate(&["annotator_id=thkt", "session_id=s1"], b"");
    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-011a] missing output= must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: output= argument required"),
        "[T-011a] stderr must mention output= argument required; got: {stderr}"
    );
}

// T-011b: t011b_annotate_missing_annotator_id_argv_exits_with_usage
// FR-013: missing annotator_id= → EXIT_USAGE (2) + "annotate: annotator_id= argument required"
#[test]
fn t011b_annotate_missing_annotator_id_argv_exits_with_usage() {
    let output = spawn_annotate(&["output=/tmp/_amici_t011b.json", "session_id=s1"], b"");
    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-011b] missing annotator_id= must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: annotator_id= argument required"),
        "[T-011b] stderr must mention annotator_id= argument required; got: {stderr}"
    );
}

// T-011c: t011c_annotate_missing_session_id_argv_exits_with_usage
// FR-014: missing session_id= → EXIT_USAGE (2) + "annotate: session_id= argument required"
#[test]
fn t011c_annotate_missing_session_id_argv_exits_with_usage() {
    let output = spawn_annotate(&["output=/tmp/_amici_t011c.json", "annotator_id=thkt"], b"");
    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-011c] missing session_id= must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: session_id= argument required"),
        "[T-011c] stderr must mention session_id= argument required; got: {stderr}"
    );
}

// T-003: t003_annotate_writes_session_with_provenance_envelope
// FR-010, FR-016, FR-020, FR-021, FR-022, FR-023, FR-026:
//   2 valid jsonl Entry lines pipe → exit 0; <output> deserialises as
//   Session; provenance.schema_version == "1.0", captured_with == "annotate",
//   timestamp.starts_with("epoch:"), queries_jsonl_hash.starts_with("fnv1a64:");
//   stderr contains "annotate: wrote" and "(2 entries)".
#[test]
fn t003_annotate_writes_session_with_provenance_envelope() {
    let dir = tempdir().expect("tempdir for annotate happy-path output");
    let session_path = dir.path().join("session.json");
    let stdin_jsonl = b"{\"id\":\"q1\",\"text\":\"alpha\",\"category\":\"factoid\",\"relevance_map\":{\"d1\":2},\"annotation_note\":\"first note\",\"mode\":\"standard\"}\n\
                       {\"id\":\"q2\",\"text\":\"beta\",\"category\":\"howto\",\"relevance_map\":{\"d2\":3},\"annotation_note\":\"second note\",\"mode\":\"standard\"}\n";
    let argv = [
        format!("output={}", session_path.display()),
        "annotator_id=thkt".to_owned(),
        "session_id=s1".to_owned(),
    ];
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();

    let output = spawn_annotate(&argv_refs, stdin_jsonl);

    assert_eq!(
        output.status.code(),
        Some(0),
        "[T-003] happy path must exit 0; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: wrote") && stderr.contains("(2 entries)"),
        "[T-003] FR-026: stderr must announce wrote + (2 entries); got: {stderr}"
    );

    let body = fs::read_to_string(&session_path).expect("read session.json");
    let session: Session =
        serde_json::from_str(&body).expect("[T-003] FR-010: session.json must deserialise");
    assert_eq!(
        session.provenance.schema_version, ANNOTATION_SCHEMA_VERSION,
        "[T-003] FR-023: schema_version must equal canonical constant"
    );
    assert_eq!(
        session.provenance.captured_with, "annotate",
        "[T-003] FR-020: captured_with must equal \"annotate\""
    );
    assert!(
        session.provenance.timestamp.starts_with("epoch:"),
        "[T-003] FR-021: timestamp must start with \"epoch:\"; got: {}",
        session.provenance.timestamp
    );
    assert!(
        session
            .provenance
            .queries_jsonl_hash
            .starts_with("fnv1a64:"),
        "[T-003] FR-022: queries_jsonl_hash must start with \"fnv1a64:\"; got: {}",
        session.provenance.queries_jsonl_hash
    );
    assert_eq!(
        session.entries.len(),
        2,
        "[T-003] FR-016: 2 jsonl lines must yield 2 entries; got: {:?}",
        session.entries
    );
    assert_eq!(session.entries[0].id, "q1");
    assert_eq!(session.entries[1].id, "q2");
}

/// Build the `output=` / `annotator_id=` / `session_id=` argv triple for a
/// reject-path test that does not need to inspect the written file. The
/// `output=` path resolves under `dir` so the canonicalising
/// `validate_output_path` parent check passes — without this, every reject
/// test would fail on argv parsing instead of its intended reject reason.
fn reject_argv(dir: &Path) -> [String; 3] {
    let session_path = dir.join("session.json");
    [
        format!("output={}", session_path.display()),
        "annotator_id=thkt".to_owned(),
        "session_id=s1".to_owned(),
    ]
}

// T-005: t005_annotate_empty_stdin_exits_with_usage
// FR-015: empty stdin → EXIT_USAGE (2) + "annotate: empty session, no entries written";
//         <output> must NOT be created.
#[test]
fn t005_annotate_empty_stdin_exits_with_usage() {
    let dir = tempdir().expect("tempdir for T-005");
    let argv = reject_argv(dir.path());
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let session_path = dir.path().join("session.json");

    let output = spawn_annotate(&argv_refs, b"");

    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-005] empty stdin must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: empty session, no entries written"),
        "[T-005] FR-015: stderr must mention empty session; got: {stderr}"
    );
    assert!(
        !session_path.exists(),
        "[T-005] FR-015: empty session must not create output file"
    );
}

// T-010: t010_annotate_malformed_json_exits_with_parse_error
// FR-017: malformed JSON line → EXIT_USAGE (2) + "annotate: line 1 parse error: ..."
#[test]
fn t010_annotate_malformed_json_exits_with_parse_error() {
    let dir = tempdir().expect("tempdir for T-010");
    let argv = reject_argv(dir.path());
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();

    let output = spawn_annotate(&argv_refs, b"{not valid json\n");

    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-010] malformed JSON must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: line 1 parse error"),
        "[T-010] FR-017: stderr must mention line 1 parse error; got: {stderr}"
    );
}

// T-009: t009_annotate_unknown_mode_variant_exits_with_parse_error
// FR-V004 / FR-017: mode="highlight" (Phase 1.5+ variant) → parse error path
//   with serde wording mentioning "highlight" or "unknown variant".
#[test]
fn t009_annotate_unknown_mode_variant_exits_with_parse_error() {
    let dir = tempdir().expect("tempdir for T-009");
    let argv = reject_argv(dir.path());
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let stdin_jsonl = b"{\"id\":\"q1\",\"text\":\"alpha\",\"category\":\"factoid\",\"relevance_map\":{\"d1\":2},\"annotation_note\":\"note\",\"mode\":\"highlight\"}\n";

    let output = spawn_annotate(&argv_refs, stdin_jsonl);

    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-009] unknown mode must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: line 1 parse error"),
        "[T-009] FR-V004: stderr must mention line 1 parse error; got: {stderr}"
    );
    assert!(
        stderr.contains("highlight") || stderr.contains("unknown variant"),
        "[T-009] FR-V004: stderr must mention the unknown variant name; got: {stderr}"
    );
}

// T-006: t006_annotate_invalid_category_exits_with_usage
// FR-V001: category="bogus" (not in 8 allowlist) → EXIT_USAGE (2) +
//   "annotate: line 1 invalid category" + "bogus" + at least one allowlist
//   value (e.g. "factoid")
#[test]
fn t006_annotate_invalid_category_exits_with_usage() {
    let dir = tempdir().expect("tempdir for T-006");
    let argv = reject_argv(dir.path());
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let stdin_jsonl = b"{\"id\":\"q1\",\"text\":\"alpha\",\"category\":\"bogus\",\"relevance_map\":{\"d1\":2},\"annotation_note\":\"note\",\"mode\":\"standard\"}\n";

    let output = spawn_annotate(&argv_refs, stdin_jsonl);

    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-006] invalid category must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: line 1 invalid category"),
        "[T-006] FR-V001: stderr must mention line 1 invalid category; got: {stderr}"
    );
    assert!(
        stderr.contains("bogus"),
        "[T-006] FR-V001: stderr must echo observed value; got: {stderr}"
    );
    assert!(
        stderr.contains("factoid"),
        "[T-006] FR-V001: stderr must list at least one allowlist value; got: {stderr}"
    );
}

// T-007: t007_annotate_grade_above_three_exits_with_usage
// FR-V002: relevance_map={"d1":99} (grade > 3) → EXIT_USAGE (2) +
//   "annotate: line 1 invalid grade" + "d1=99"
#[test]
fn t007_annotate_grade_above_three_exits_with_usage() {
    let dir = tempdir().expect("tempdir for T-007");
    let argv = reject_argv(dir.path());
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let stdin_jsonl = b"{\"id\":\"q1\",\"text\":\"alpha\",\"category\":\"factoid\",\"relevance_map\":{\"d1\":99},\"annotation_note\":\"note\",\"mode\":\"standard\"}\n";

    let output = spawn_annotate(&argv_refs, stdin_jsonl);

    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-007] grade > 3 must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: line 1 invalid grade"),
        "[T-007] FR-V002: stderr must mention line 1 invalid grade; got: {stderr}"
    );
    assert!(
        stderr.contains("d1=99"),
        "[T-007] FR-V002: stderr must echo doc_id=grade; got: {stderr}"
    );
}

// T-008: t008_annotate_duplicate_id_exits_with_usage
// FR-V003: 2 entries with same id → EXIT_USAGE (2) +
//   "annotate: line 2 duplicate id" + "(first seen on line 1)"
#[test]
fn t008_annotate_duplicate_id_exits_with_usage() {
    let dir = tempdir().expect("tempdir for T-008");
    let argv = reject_argv(dir.path());
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let stdin_jsonl = b"{\"id\":\"q1\",\"text\":\"alpha\",\"category\":\"factoid\",\"relevance_map\":{\"d1\":2},\"annotation_note\":\"first\",\"mode\":\"standard\"}\n\
                       {\"id\":\"q1\",\"text\":\"beta\",\"category\":\"howto\",\"relevance_map\":{\"d2\":3},\"annotation_note\":\"second\",\"mode\":\"standard\"}\n";

    let output = spawn_annotate(&argv_refs, stdin_jsonl);

    assert_eq!(
        output.status.code(),
        Some(EXIT_USAGE),
        "[T-008] duplicate id must exit {EXIT_USAGE}; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("annotate: line 2 duplicate id"),
        "[T-008] FR-V003: stderr must mention line 2 duplicate id; got: {stderr}"
    );
    assert!(
        stderr.contains("first seen on line 1"),
        "[T-008] FR-V003: stderr must include first-seen line number; got: {stderr}"
    );
}

// T-012: t012_eval_query_annotation_and_entry_annotation_note_coexist
// FR-010 + sub-PR-A FR-005: a happy-path Session whose entry id matches
//   an existing EvalQuery in queries.jsonl. Both `EvalQuery.annotation`
//   (existing field, sub-PR-A FR-005) and `Entry.annotation_note`
//   (sub-PR-A new field) must be readable in shared scope without rename.
#[test]
fn t012_eval_query_annotation_and_entry_annotation_note_coexist() {
    let queries_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eval/queries.jsonl");
    let queries: Vec<EvalQuery> = load_queries(&queries_path).expect("load queries.jsonl");
    let first_query = queries
        .first()
        .expect("[T-012] queries.jsonl must have at least one entry");

    let dir = tempdir().expect("tempdir for T-012");
    let session_path = dir.path().join("session.json");
    let stdin_jsonl = format!(
        "{{\"id\":\"{id}\",\"text\":\"alpha\",\"category\":\"factoid\",\
         \"relevance_map\":{{\"d1\":2}},\"annotation_note\":\"new block-mode note\",\
         \"mode\":\"standard\"}}\n",
        id = first_query.id,
    );
    let argv = [
        format!("output={}", session_path.display()),
        "annotator_id=thkt".to_owned(),
        "session_id=s1".to_owned(),
    ];
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();

    let output = spawn_annotate(&argv_refs, stdin_jsonl.as_bytes());
    assert_eq!(
        output.status.code(),
        Some(0),
        "[T-012] happy path must exit 0; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let body = fs::read_to_string(&session_path).expect("[T-012] read session.json");
    let session: Session = serde_json::from_str(&body).expect("[T-012] deserialise session.json");
    let entry = session
        .entries
        .first()
        .expect("[T-012] session.entries must be non-empty");
    assert_eq!(
        entry.id, first_query.id,
        "[T-012] Entry.id must mirror EvalQuery.id"
    );
    let _eval_annotation: &str = &first_query.annotation;
    let _entry_annotation_note: &str = &entry.annotation_note;
    assert_eq!(
        entry.annotation_note, "new block-mode note",
        "[T-012] FR-005: Entry.annotation_note must be the new free-form note, \
         not aliased to EvalQuery.annotation"
    );
}

// T-013: t013_adr_0004_reassessment_triggers_record_llm_provider_path
// FR-030 / FR-031 / FR-032: ADR-0004 §Reassessment Triggers must mention
//   "LlmProvider" so sub-PR-D's design step surfaces the assumption
//   (AS-005). Status header must remain `- Status: proposed` until
//   sub-PR-C lands. NFR-003 forbids post-cutoff external citations
//   (`arXiv:2602`, `AIANO`, `ADR-0021`).
#[test]
fn t013_adr_0004_reassessment_triggers_record_llm_provider_path() {
    let adr_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/decisions/0004-annotation-framework.md");
    let content = fs::read_to_string(&adr_path)
        .unwrap_or_else(|e| panic!("[T-013] read {}: {e}", adr_path.display()));

    assert!(
        content.contains("LlmProvider"),
        "[T-013] FR-030: ADR-0004 must mention LlmProvider in §Reassessment Triggers; \
         content snapshot:\n{content}"
    );

    let has_proposed_status = content
        .lines()
        .any(|line| line.trim() == "- Status: proposed");
    assert!(
        has_proposed_status,
        "[T-013] FR-031: ADR-0004 Status header must remain `- Status: proposed`; \
         content snapshot:\n{content}"
    );

    for forbidden in ["arXiv:2602", "AIANO", "ADR-0021"] {
        assert!(
            !content.contains(forbidden),
            "[T-013] FR-032 / NFR-003: ADR-0004 must not cite {forbidden:?} \
             (post-cutoff external citation); content snapshot:\n{content}"
        );
    }
}
