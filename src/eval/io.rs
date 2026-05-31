//! Atomic JSON write helper shared by [`crate::eval::baseline`] and
//! [`crate::eval::annotation`].
//!
//! Generic over `T: Serialize` so [`BaselineSnapshot`] and [`Session`]
//! reach a single atomic-write code path. `serde_json` failures are
//! funneled into [`io::ErrorKind::Other`] so callers' typed error
//! enums (`BaselineError`, `AnnotationError`) only need a single
//! `Io(#[from] io::Error)` variant for both serialisation and
//! filesystem failures.
//!
//! [`BaselineSnapshot`]: crate::eval::baseline::BaselineSnapshot
//! [`Session`]: crate::eval::annotation::Session

use std::io;
use std::path::Path;

use serde::Serialize;

use crate::eval::baseline::atomic_write;

/// Serialise `value` as pretty JSON terminated by a newline and
/// atomically replace the file at `path`.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] when the temp file cannot be
/// created, written, fsynced, or renamed into place. `serde_json`
/// encoding failures are wrapped via [`io::Error::other`] so callers
/// match a single variant for both serialisation and filesystem
/// faults.
pub fn write_json<T: Serialize>(value: &T, path: &Path) -> io::Result<()> {
    let mut json = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    json.push('\n');
    atomic_write(path, json.as_bytes())
}

#[cfg(test)]
mod tests;
