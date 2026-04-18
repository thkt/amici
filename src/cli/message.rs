//! CLI message helpers shared by sae / yomu / recall.
//!
//! Unifies the stderr formatting of terminal errors, shorthand expansion
//! hints, informational notices, deprecation warnings, and multi-line
//! progress updates. Matching formats across tools keeps log parsers and
//! user expectations consistent.

/// Prints a terminal error to stderr.
///
/// Use at the CLI entry point where the process is about to exit with a
/// non-zero status. Matches the `anyhow::Error` Display convention.
///
/// # Examples
///
/// ```no_run
/// if let Err(e) = run() {
///     amici::cli::exit_error(&format!("{e}"));
///     std::process::exit(1);
/// }
/// # fn run() -> Result<(), std::io::Error> { Ok(()) }
/// ```
pub fn exit_error(msg: &str) {
    eprintln!("error: {msg}");
}

/// Prints a shorthand-expansion hint to stderr.
///
/// Use after [`try_expand_shorthand`](crate::cli::try_expand_shorthand) rewrites
/// the argv, to show the user what was actually parsed. Callers convert the
/// returned `Vec<OsString>` to string slices first — typically via
/// [`OsStr::to_string_lossy`](std::ffi::OsStr::to_string_lossy) — since this
/// helper formats for display, not for round-trip shell execution. Items are
/// joined with a single space and not shell-escaped; inputs that contain
/// embedded spaces will appear without their original quoting.
///
/// # Examples
///
/// ```no_run
/// amici::cli::hint_arrow(&["search", "認証"]);
/// // stderr: → search 認証
/// ```
pub fn hint_arrow<S: AsRef<str>>(items: &[S]) {
    eprintln!("{}", format_hint_arrow(items));
}

fn format_hint_arrow<S: AsRef<str>>(items: &[S]) -> String {
    let joined = items
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<&str>>()
        .join(" ");
    format!("→ {joined}")
}

/// Prints `msg` to stderr as an informational notice.
///
/// Replaces `println!` usages where the text is CLI guidance rather than the
/// program's actual result. Output goes to stderr so stdout stays reserved
/// for pipeable data.
///
/// # Examples
///
/// ```no_run
/// amici::cli::info("no pending items; nothing to do");
/// ```
pub fn info(msg: &str) {
    eprintln!("{msg}");
}

/// Prints a deprecation warning to stderr.
///
/// Use when a CLI flag or subcommand is retained for backward compatibility
/// but callers should migrate to the new form.
///
/// # Examples
///
/// ```no_run
/// amici::cli::deprecation_warn("--legacy-flag", "--new-flag");
/// ```
pub fn deprecation_warn(old: &str, new: &str) {
    eprintln!("warning: {old} is deprecated, use {new} instead");
}

/// Prints a two-space-indented progress line to stderr.
///
/// `items` are joined with ` — ` (em dash surrounded by spaces). Use for
/// multi-field progress such as `page 3/10`, `batch 2`, etc. Intended for
/// non-TTY callers or alongside a spinner finish marker.
///
/// # Examples
///
/// ```no_run
/// amici::cli::progress_step(&["page 3/10", "batch 2"]);
/// // stderr:   page 3/10 — batch 2
/// ```
pub fn progress_step<S: AsRef<str>>(items: &[S]) {
    eprintln!("{}", format_progress_step(items));
}

fn format_progress_step<S: AsRef<str>>(items: &[S]) -> String {
    let joined = items
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<&str>>()
        .join(" — ");
    format!("  {joined}")
}

#[cfg(test)]
mod tests {
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

    // T-050: deprecation_warn_does_not_panic
    #[test]
    fn deprecation_warn_does_not_panic() {
        deprecation_warn("--old", "--new");
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
}
