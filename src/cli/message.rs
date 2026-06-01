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

/// Prints a recovery hint to stderr.
///
/// Use when an operation degraded to a fallback and the user has a concrete
/// next step (e.g. "run `<binary> model download` to enable semantic search").
/// Output is prefixed with `Hint: ` so consumers can `grep` for hints separately
/// from `warning:` / `error:` lines.
///
/// # Examples
///
/// ```no_run
/// amici::cli::hint("run `yomu model download` to enable semantic search");
/// // stderr: Hint: run `yomu model download` to enable semantic search
/// ```
pub fn hint(msg: &str) {
    eprintln!("Hint: {msg}");
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

/// Prints a generic warning to stderr.
///
/// Use when a CLI surface encounters a recoverable anomaly that the user
/// should be aware of (e.g. a model failed to load and search continues
/// with text-only fallback). Output is prefixed with `warning: ` to match
/// [`exit_error`] (`error: `) and [`deprecation_warn`] (`warning: ...`).
///
/// # Examples
///
/// ```no_run
/// amici::cli::warning("embedding model not available (probe failed)");
/// // stderr: warning: embedding model not available (probe failed)
/// ```
pub fn warning(msg: &str) {
    eprintln!("warning: {msg}");
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
mod tests;
