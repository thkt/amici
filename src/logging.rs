//! Tracing subscriber initialization for CLI binaries.
//!
//! Wraps the `tracing_subscriber::fmt` setup shared by sae, yomu, and recall
//! so each `main.rs` can initialize logging in a single call.

use std::io;

use tracing_subscriber::EnvFilter;

/// Default directive merged into every CLI's filter. See
/// [`init_subscriber`] for the merge semantics and `RUST_LOG` interaction.
const RURICO_DEFAULT_DIRECTIVE: &str = "rurico=warn";

/// Installs a `tracing_subscriber::fmt` subscriber that writes to stderr and
/// reads its filter from `RUST_LOG`, falling back to `default_filter` when the
/// environment variable is unset or unparseable.
///
/// # Examples
///
/// ```no_run
/// amici::logging::init_subscriber("yomu=warn");
/// // rest of main…
/// ```
///
/// # Migration note
///
/// Matches yomu's existing "fallback on missing `RUST_LOG`" semantics. sae's
/// previous behavior of *always* layering `sae=info` on top of `RUST_LOG` is
/// not preserved — callers who relied on implicit sae logs should export
/// `RUST_LOG=sae=info` explicitly.
///
/// `rurico=warn` is appended to `default_filter` so rurico's degraded-path
/// warnings surface in operator logs without each downstream CLI having to
/// opt in. Setting `RUST_LOG` overrides the merged default entirely; export
/// `RUST_LOG=<crate>=<level>,rurico=warn` to keep the rurico directive while
/// customizing the rest.
///
/// # Panics
///
/// Panics if `default_filter` is not a valid
/// [`tracing_subscriber::EnvFilter`] directive string. Also panics if called
/// more than once per process, since
/// [`tracing_subscriber::fmt::SubscriberBuilder::init`] installs the global
/// default subscriber.
pub fn init_subscriber(default_filter: &str) {
    let merged = merge_default_directives(default_filter);
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(resolve_env_filter(&merged))
        .init();
}

fn merge_default_directives(default_filter: &str) -> String {
    if default_filter.is_empty() {
        RURICO_DEFAULT_DIRECTIVE.to_owned()
    } else {
        format!("{default_filter},{RURICO_DEFAULT_DIRECTIVE}")
    }
}

fn resolve_env_filter(default_filter: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter))
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-021: resolve_env_filter_accepts_directive
    #[test]
    fn resolve_env_filter_accepts_directive() {
        let _filter = resolve_env_filter("yomu=warn");
    }

    // T-022: resolve_env_filter_accepts_multi_directive
    #[test]
    fn resolve_env_filter_accepts_multi_directive() {
        let _filter = resolve_env_filter("sae=info,hyper=warn");
    }

    // T-023: merge_default_directives_appends_rurico_warn
    #[test]
    fn merge_default_directives_appends_rurico_warn() {
        assert_eq!(
            merge_default_directives("yomu=warn"),
            "yomu=warn,rurico=warn"
        );
    }

    // T-024: merge_default_directives_handles_empty_default
    #[test]
    fn merge_default_directives_handles_empty_default() {
        assert_eq!(merge_default_directives(""), "rurico=warn");
    }

    // T-025: merge_default_directives_produces_parseable_filter
    #[test]
    fn merge_default_directives_produces_parseable_filter() {
        let merged = merge_default_directives("sae=info,hyper=warn");
        EnvFilter::try_new(&merged).expect("merged default filter must parse");
    }
}
