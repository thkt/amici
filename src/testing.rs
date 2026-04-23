//! Shared test helpers exposed via the `test-support` feature.
//!
//! Items under this module are compiled only when the `test-support` feature
//! is enabled, or during in-crate `#[cfg(test)]` builds. They never enter
//! downstream production binaries.

pub mod hybrid;
