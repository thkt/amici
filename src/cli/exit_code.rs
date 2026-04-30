//! Exit code conventions for CLI binaries.
//!
//! Provides:
//! - [`CliError`] trait — implement on the crate-level `Error` enum so
//!   `main.rs` can call `e.exit_code()` directly.
//! - [`codes`] module — `u8` constants for the
//!   [sysexits.h](https://man.openbsd.org/sysexits.3) convention. Wrap with
//!   `ExitCode::from(codes::USAGE)` at the call site (see *Why `u8`* below).
//!
//! These are **convention, not enforcement**. Each CLI is free to pick any
//! distinct schema — the requirement is only "distinct codes per error variant
//! so LLMs / scripts can branch on retry policy".
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
//! }
//!
//! impl CliError for MyError {
//!     fn exit_code(&self) -> ExitCode {
//!         match self {
//!             MyError::BadArgs => ExitCode::from(codes::USAGE),
//!             MyError::IoFailure => ExitCode::from(codes::IO_ERR),
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

/// `sysexits.h`-derived `u8` exit-code constants.
///
/// Convert with `ExitCode::from(codes::CONST)` — see the module doc for why
/// these are `u8` rather than `ExitCode`.
pub mod codes {
    /// Successful termination.
    pub const SUCCESS: u8 = 0;
    /// `EX_USAGE`. Bad command-line usage (missing arg, unknown flag, etc.).
    pub const USAGE: u8 = 64;
    /// `EX_SOFTWARE`. Internal software error — invariant violated.
    pub const SOFTWARE: u8 = 70;
    /// `EX_CANTCREAT`. Cannot create a (user-visible) output file or path.
    pub const CANT_CREAT: u8 = 73;
    /// `EX_IOERR`. I/O error during operation.
    pub const IO_ERR: u8 = 74;
    /// `EX_TEMPFAIL`. Transient failure — retry may succeed (downloads,
    /// optional models unavailable, etc.).
    pub const TEMP_FAIL: u8 = 75;
}

#[cfg(test)]
mod tests {
    use super::codes;

    // sysexits.h reference values pinned here so a future refactor that
    // touches `codes::*` fails visibly instead of silently breaking
    // downstream CLIs that depend on these numbers.

    // T-122: codes_match_sysexits_constants
    #[test]
    fn codes_match_sysexits_constants() {
        assert_eq!(codes::SUCCESS, 0);
        assert_eq!(codes::USAGE, 64);
        assert_eq!(codes::SOFTWARE, 70);
        assert_eq!(codes::CANT_CREAT, 73);
        assert_eq!(codes::IO_ERR, 74);
        assert_eq!(codes::TEMP_FAIL, 75);
    }
}
