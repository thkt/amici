//! Tracing subscriber initialization for CLI binaries.
//!
//! Wraps the `tracing_subscriber::fmt` setup shared by sae, yomu, and recall
//! so each `main.rs` can initialize logging in a single call.

use std::io;

use tracing_subscriber::EnvFilter;

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
/// # Panics
///
/// Panics if `default_filter` is not a valid
/// [`tracing_subscriber::EnvFilter`] directive string. Also panics if called
/// more than once per process, since
/// [`tracing_subscriber::fmt::SubscriberBuilder::init`] installs the global
/// default subscriber.
pub fn init_subscriber(default_filter: &str) {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(resolve_env_filter(default_filter))
        .init();
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
}
