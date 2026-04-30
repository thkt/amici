//! Factory for env-var lookup, used by `from_env_with` constructors.
//!
//! Constructors that read environment variables should split into two: a
//! production `from_env()` that calls [`env_lookup()`], and a generic
//! `from_env_with(impl Fn(&str) -> Option<String>)` that accepts the lookup
//! as a parameter. Tests then pass a closure returning fixture values,
//! removing global env state from the test path.
//!
//! ```no_run
//! use amici::cli::env_lookup;
//!
//! pub struct Config {
//!     pub flag: bool,
//! }
//!
//! impl Config {
//!     pub fn from_env() -> Self {
//!         Self::from_env_with(env_lookup())
//!     }
//!     pub fn from_env_with<F: Fn(&str) -> Option<String>>(get: F) -> Self {
//!         Self { flag: get("MY_FLAG").as_deref() == Some("1") }
//!     }
//! }
//! ```

use std::env::var;

/// Returns the production env-var lookup: `std::env::var(k).ok()`.
///
/// Pair with `from_env_with(impl Fn(&str) -> Option<String>)` constructors.
/// Tests pass their own closure (e.g. matching keys against a fixture map)
/// instead of mutating process env.
///
/// # Examples
///
/// ```no_run
/// let get = amici::cli::env_lookup();
/// let _home = get("HOME");
/// ```
pub fn env_lookup() -> impl Fn(&str) -> Option<String> {
    |k| var(k).ok()
}
