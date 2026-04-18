//! Schema migration notification helpers.
//!
//! Provides a unified `tracing::warn!` event for the "schema changed →
//! re-run `<tool> <cmd>`" pattern shared by sae, yomu, and recall.

use tracing::warn;

/// Emits a `warn!` event announcing that a schema migration cleared data and
/// instructing the user to re-run a rebuild command.
///
/// Structured fields `tool`, `item`, `count`, and `rebuild_cmd` are always
/// attached so log consumers can filter and aggregate without parsing the
/// message body. The human-readable message omits `count` when it is `0` —
/// use `0` when the caller does not have a meaningful row count (for example,
/// when embeddings are cleared en masse without row enumeration).
///
/// `item` is the **noun** that was cleared (e.g. `"cached sessions"`,
/// `"embeddings"`), chosen to avoid collision with the `target:` metadata
/// target key that `tracing` macros recognize as a reserved keyword.
///
/// # Examples
///
/// With a known row count:
///
/// ```no_run
/// amici::migration::notify_schema_change("recall", "cached sessions", 42, "recall index");
/// // "schema changed — clearing 42 cached sessions; run `recall index` to rebuild"
/// ```
///
/// Without a count:
///
/// ```no_run
/// amici::migration::notify_schema_change("sae", "embeddings", 0, "sae embed");
/// // "schema changed — clearing embeddings; run `sae embed` to rebuild"
/// ```
///
/// Extra fields such as `from` version or `path` should be attached via an
/// enclosing span so this helper keeps a single signature:
///
/// ```no_run
/// # use tracing::info_span;
/// # let path = std::path::Path::new("/tmp/x");
/// let _span = info_span!("migration", from = 7, to = 8, path = %path.display()).entered();
/// amici::migration::notify_schema_change("yomu", "embeddings", 0, "yomu index");
/// ```
pub fn notify_schema_change(tool: &str, item: &str, count: usize, rebuild_cmd: &str) {
    let msg = format_schema_change_message(item, count, rebuild_cmd);
    warn!(tool, item, count, rebuild_cmd, "{msg}");
}

fn format_schema_change_message(item: &str, count: usize, rebuild_cmd: &str) -> String {
    if count > 0 {
        format!("schema changed — clearing {count} {item}; run `{rebuild_cmd}` to rebuild")
    } else {
        format!("schema changed — clearing {item}; run `{rebuild_cmd}` to rebuild")
    }
}

#[cfg(test)]
mod tests {
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
}
