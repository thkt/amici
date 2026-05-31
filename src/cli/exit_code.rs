//! Exit code conventions for CLI binaries.
//!
//! Provides:
//! - [`CliError`] trait — implement on the crate-level `Error` enum so
//!   `main.rs` can call `e.exit_code()` directly.
//! - [`codes`] module — `u8` constants from two sources, kept side-by-side
//!   so call sites do not need to import from two places:
//!   1. [sysexits.h](https://man.openbsd.org/sysexits.3) (64–78). Standard
//!      Unix exit categories — `USAGE`, `DATA_ERROR`, `SOFTWARE`,
//!      `CANT_CREAT`, `IO_ERR`, `TEMP_FAIL`.
//!   2. Project extension range (80–119). Assigned per ADR-0066 when no
//!      sysexits variant fits. Currently: `UNKNOWN` (104).
//!
//! Wrap with `ExitCode::from(codes::USAGE)` at the call site (see
//! *Why `u8`* below).
//!
//! These are **convention, not enforcement**. Each CLI is free to pick any
//! distinct schema — the requirement is only "distinct codes per error variant
//! so LLMs / scripts can branch on retry policy".
//!
//! # Group 2 baseline (ADR-0066)
//!
//! ADR-0066 partitions the CLI suite into three groups by error topology and
//! assigns each group a shared exit-code baseline. This crate is the baseline
//! for **Group 2 — local semantic search** (sae / yomu / recall / rurico).
//! Downstream CLIs alias `amici::cli::exit_code::codes` so classification
//! stays consistent across the group: a metrics dashboard tracking the rate
//! of `UNKNOWN` (104) can pivot on a single value to detect `anyhow::Error`
//! swallowing or unclassified failures regardless of which Group 2 CLI emitted
//! the code.
//!
//! # Why `u8` rather than `ExitCode` constants
//!
//! `ExitCode::from()` is not `const` (stable Rust 1.95), so
//! `pub const USAGE: ExitCode` does not compile. Exposing `u8` keeps the
//! constants usable in `const` contexts; convert at the call site.
//!
//! ```
//! use amici::cli::exit_code::{CliError, codes};
//! use std::process::ExitCode;
//!
//! enum MyError {
//!     BadArgs,
//!     IoFailure,
//!     Other,
//! }
//!
//! impl CliError for MyError {
//!     fn exit_code(&self) -> ExitCode {
//!         match self {
//!             MyError::BadArgs => ExitCode::from(codes::USAGE),
//!             MyError::IoFailure => ExitCode::from(codes::IO_ERR),
//!             MyError::Other => ExitCode::from(codes::UNKNOWN), // PJ extension (ADR-0066)
//!         }
//!     }
//! }
//! ```

use std::process::ExitCode;

/// Maps a CLI-facing error to a process exit code.
///
/// Implement on the crate-level `Error` enum (`YomuError`, `SaeError`, etc.)
/// so `main.rs` can write `return e.exit_code();` directly, replacing
/// hand-written `exit_code_for(&e)` helpers.
pub trait CliError {
    /// Returns the [`ExitCode`] this error should produce when it propagates
    /// to the CLI entry point.
    fn exit_code(&self) -> ExitCode;
}

/// `u8` exit-code constants from sysexits.h plus the ADR-0066 project
/// extension range.
///
/// Convert with `ExitCode::from(codes::CONST)` — see the module doc for why
/// these are `u8` rather than `ExitCode`.
pub mod codes {
    // ── sysexits.h (64–78) ──
    // Source: https://man.openbsd.org/sysexits.3

    /// Successful termination (`EX_OK`).
    pub const SUCCESS: u8 = 0;
    /// `EX_USAGE`. Bad command-line usage (missing arg, unknown flag, etc.).
    pub const USAGE: u8 = 64;
    /// `EX_DATAERR`. Input data was malformed — query syntax invalid,
    /// encoding error, etc. (per ADR-0066 Group 2 baseline).
    pub const DATA_ERROR: u8 = 65;
    /// `EX_SOFTWARE`. Internal software error — invariant violated.
    pub const SOFTWARE: u8 = 70;
    /// `EX_CANTCREAT`. Cannot create a (user-visible) output file or path.
    pub const CANT_CREAT: u8 = 73;
    /// `EX_IOERR`. I/O error during operation.
    pub const IO_ERR: u8 = 74;
    /// `EX_TEMPFAIL`. Transient failure — retry may succeed (downloads,
    /// optional models unavailable, etc.).
    pub const TEMP_FAIL: u8 = 75;

    // ── Project extension range (80–119) ──
    // Source: ADR-0066. Assigned when no sysexits.h variant fits.

    /// Classification fallback. Emit when an error cannot be mapped to any
    /// sysexits variant. An increase in this code's rate signals that the
    /// classification design needs review (typically an `anyhow::Error` path
    /// that should be promoted to a typed variant).
    pub const UNKNOWN: u8 = 104;

    /// Alias for [`SOFTWARE`]. Provided so call sites can use the
    /// `INTERNAL` name that ADR-0066's classification table uses, keeping
    /// the table-vs-implementation diff trivial. Numerically identical to
    /// `EX_SOFTWARE` (70).
    pub const INTERNAL: u8 = SOFTWARE;
}

#[cfg(test)]
mod tests;
