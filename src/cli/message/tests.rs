use super::*;

// T-043: exit_error_does_not_panic
#[test]
fn exit_error_does_not_panic() {
    exit_error("test");
}

// T-044: format_hint_arrow_joins_with_space
#[test]
fn format_hint_arrow_joins_with_space() {
    assert_eq!(
        format_hint_arrow(&["search", "query", "term"]),
        "→ search query term"
    );
}

// T-045: format_hint_arrow_accepts_string_owned
#[test]
fn format_hint_arrow_accepts_string_owned() {
    let v: Vec<String> = vec!["a".into(), "b".into()];
    assert_eq!(format_hint_arrow(&v), "→ a b");
}

// T-046: format_hint_arrow_empty_produces_arrow_only
#[test]
fn format_hint_arrow_empty_produces_arrow_only() {
    let empty: [&str; 0] = [];
    assert_eq!(format_hint_arrow(&empty), "→ ");
}

// T-047: hint_arrow_does_not_panic
#[test]
fn hint_arrow_does_not_panic() {
    hint_arrow(&["a", "b"]);
}

// T-048: info_does_not_panic
#[test]
fn info_does_not_panic() {
    info("some guidance");
}

// T-124: hint_does_not_panic
#[test]
fn hint_does_not_panic() {
    hint("run `yomu model download` to enable semantic search");
}

// T-050: deprecation_warn_does_not_panic
#[test]
fn deprecation_warn_does_not_panic() {
    deprecation_warn("--old", "--new");
}

// T-125: warning_does_not_panic
#[test]
fn warning_does_not_panic() {
    warning("model not available");
}

// T-051: format_progress_step_joins_with_em_dash_and_indents
#[test]
fn format_progress_step_joins_with_em_dash_and_indents() {
    assert_eq!(
        format_progress_step(&["page 3/10", "batch 2"]),
        "  page 3/10 — batch 2"
    );
}

// T-052: format_progress_step_single_item_indented
#[test]
fn format_progress_step_single_item_indented() {
    assert_eq!(format_progress_step(&["step 1"]), "  step 1");
}

// T-053: progress_step_does_not_panic
#[test]
fn progress_step_does_not_panic() {
    progress_step(&["a", "b"]);
}
