use super::*;

// T-018: format_schema_change_message_includes_count_when_positive
#[test]
fn format_schema_change_message_includes_count_when_positive() {
    let msg = format_schema_change_message("cached sessions", 42, "recall index");
    assert_eq!(
        msg,
        "schema changed — clearing 42 cached sessions; run `recall index` to rebuild"
    );
}

// T-019: format_schema_change_message_omits_count_when_zero
#[test]
fn format_schema_change_message_omits_count_when_zero() {
    let msg = format_schema_change_message("embeddings", 0, "sae embed");
    assert_eq!(
        msg,
        "schema changed — clearing embeddings; run `sae embed` to rebuild"
    );
}

// T-020: notify_schema_change_does_not_panic
#[test]
fn notify_schema_change_does_not_panic() {
    notify_schema_change("test", "items", 5, "test rebuild");
    notify_schema_change("test", "items", 0, "test rebuild");
}
