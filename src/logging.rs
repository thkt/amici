//! Tracing subscriber initialization for CLI binaries.
//!
//! Wraps the `tracing_subscriber::fmt` setup shared by sae, yomu, and recall
//! so each `main.rs` can initialize logging in a single call.

use std::io;

use tracing_subscriber::EnvFilter;

/// Default directives merged into every CLI's filter. See
/// [`init_subscriber`] for the merge semantics and `RUST_LOG` interaction.
const RURICO_DEFAULT_DIRECTIVE: &str = "rurico=warn";
const AMICI_DEFAULT_DIRECTIVE: &str = "amici=warn";

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
/// Upstream crate warn directives are appended to `default_filter` so each
/// crate's degraded-path warnings surface in operator logs without downstream
/// CLI opt-in. Setting `RUST_LOG` overrides the merged default entirely;
/// export `RUST_LOG=<crate>=<level>,rurico=warn,amici=warn` to keep the
/// upstream directives while customizing the rest.
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
    let upstream = format!("{RURICO_DEFAULT_DIRECTIVE},{AMICI_DEFAULT_DIRECTIVE}");
    if default_filter.is_empty() {
        upstream
    } else {
        format!("{default_filter},{upstream}")
    }
}

fn resolve_env_filter(default_filter: &str) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter))
}

#[cfg(test)]
mod tests;
