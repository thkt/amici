//! Production download bootstrap — the I/O composition root.
//!
//! Wires the real `rurico::embed::download_model` (network) and on-device MLX
//! probe into the unit-tested [`super::try_download_and_verify_with_fns`]. This
//! file holds only that wiring: a real network download plus a hardware probe,
//! which cannot be exercised by a unit test without a live registry and Apple
//! Silicon. It is excluded from the diff-coverage gate for the same reason test
//! scaffolding is — the logic it composes is covered via
//! `try_download_and_verify_with_fns`; only the irreducible external calls live
//! here.

use rurico::embed::{Artifacts, Embedder, ModelId, download_model};

use super::{ModelDownloadError, try_download_and_verify_with_fns};
use crate::cli::with_spinner;

/// Download the default embedding model and verify it loads correctly.
///
/// Shows a spinner on stderr during download and probe phases.
///
/// # Prerequisites
///
/// "Verify" here means **probe-load**: after the download, the fn calls
/// [`Embedder::probe`] and [`Embedder::new`] to confirm the artifacts
/// actually load on this machine. It is **not** a content-hash check —
/// artifact integrity is delegated to `rurico::embed::download_model`,
/// whose own checksum step runs before this fn's post-download probe. A
/// successful return guarantees "the model loads here and now", not
/// "the downloaded bytes match the registry manifest".
///
/// The calling binary must invoke `rurico::handle_probe_if_needed()`
/// at the very start of `main()`. Without it, the post-download probe returns
/// [`ModelDownloadError::ProbeFailed`] even when the download succeeds.
///
/// # Errors
///
/// - [`ModelDownloadError::DownloadFailed`] — the HTTP download from the model
///   registry failed.
/// - [`ModelDownloadError::BackendUnavailable`] — the hardware/OS backend
///   (e.g. MLX) is not available on this machine.
/// - [`ModelDownloadError::ProbeFailed`] — the downloaded model files could not
///   be loaded. Corrupt artifacts are deleted automatically so a subsequent
///   call can re-download. Non-corrupt probe or init failures leave artifacts
///   intact.
pub fn download_and_verify_model() -> Result<(), ModelDownloadError> {
    with_spinner(
        "Downloading model...",
        |_| "Model ready".to_owned(),
        |update| {
            try_download_and_verify_with_fns(
                || {
                    download_model(ModelId::DEFAULT).map_err(|e| {
                        tracing::error!(error = %e, "model download failed");
                        e
                    })
                },
                |e| tracing::warn!(error = %e, "failed to delete artifacts after failed probe"),
                Embedder::probe,
                Embedder::new,
                Artifacts::delete_files,
                || update("Verifying model..."),
            )
        },
    )
}
