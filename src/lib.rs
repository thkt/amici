//! amici crate root.
//!
//! ADR-0010: a module that must not leak into a downstream production build is
//! gated behind a cargo feature. `cli` / `logging` / `migration` / `model` /
//! `storage` are unconditional production API. The gated modules and their
//! reasons:
//!
//! - `eval` → `eval-harness`: pulls MLX (Apple Silicon only); used by the
//!   `eval_harness` binary, never by a downstream production CLI.
//! - `testing` → `any(test, feature = "test-support")`: test helpers that must
//!   stay out of a downstream production binary.
//!
//! When adding a `pub mod`, gate it when it is eval / test / Apple-Silicon
//! only; leave it unconditional only when downstream production always needs it.

pub mod cli;
#[cfg(feature = "eval-harness")]
pub mod eval;
pub mod logging;
pub mod migration;
pub mod model;
pub mod storage;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
