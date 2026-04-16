pub mod embedder;
pub mod reranker;

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::io;

use rurico::embed::{
    Artifacts, Embed, EmbedInitError, Embedder, ModelId, ProbeStatus, download_model,
};

use self::embedder::try_load_embedder_with_fns;
use crate::cli::with_spinner;

/// Reason a model could not be loaded.
///
/// `Disabled` is reserved for caller-level opt-out (e.g. an environment variable);
/// the loader functions never produce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedReason {
    /// Caller explicitly disabled the model (e.g. via an environment variable or config flag).
    /// The loader functions never return this variant; it is set by the caller.
    Disabled,
    /// Model artifacts are not present in the local cache. The user must download them first.
    NotInstalled,
    /// The hardware/OS backend (e.g. MLX) is not available on this machine.
    BackendUnavailable,
    /// A cache lookup error, model-file corruption, or model-init failure occurred.
    /// The artifacts may have been deleted automatically to allow re-download.
    ProbeFailed,
}

impl fmt::Display for DegradedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DegradedReason::Disabled => write!(f, "disabled"),
            DegradedReason::NotInstalled => write!(f, "not installed"),
            DegradedReason::BackendUnavailable => write!(f, "MLX backend unavailable"),
            DegradedReason::ProbeFailed => write!(f, "probe failed"),
        }
    }
}

/// Returns a short user-facing note for a degraded model (embedder or reranker),
/// or `None` if no message should be shown (e.g. the caller explicitly disabled the model).
pub fn degraded_reason_user_note(reason: DegradedReason) -> Option<&'static str> {
    match reason {
        DegradedReason::Disabled => None,
        DegradedReason::NotInstalled => {
            Some("embedding model not installed; results from text search only")
        }
        DegradedReason::BackendUnavailable | DegradedReason::ProbeFailed => {
            Some("embedding model unavailable; results from text search only")
        }
    }
}

/// Outcome of a model-load attempt.
///
/// Callers should inspect the variant to handle the `Failed` case — dropping a
/// `Failed` value silently discards the error message.
#[must_use = "inspect the variant to handle loading failures"]
#[derive(Default)]
pub enum ModelLoad<T> {
    /// The model loaded successfully and is ready to use.
    Ready(T),
    /// No model artifacts were found; the model was never installed.
    #[default]
    Absent,
    /// The model could not be loaded. The `String` contains a human-readable error message.
    Failed(String),
}

impl<T> ModelLoad<T> {
    /// Returns `Some(&T)` when the model is [`Ready`](ModelLoad::Ready), `None` otherwise.
    pub fn as_ref(&self) -> Option<&T> {
        match self {
            Self::Ready(v) => Some(v),
            _ => None,
        }
    }

    /// Prints a user-facing hint or warning to stderr when the model is not ready.
    ///
    /// - [`Absent`](ModelLoad::Absent): prints `"Hint: {absent_hint}"`
    /// - [`Failed`](ModelLoad::Failed): prints `"Warning: {model_label} not available ({error})"`
    /// - [`Ready`](ModelLoad::Ready): no-op
    pub fn emit_load_hint(&self, absent_hint: &str, model_label: &str) {
        match self {
            Self::Absent => eprintln!("Hint: {absent_hint}"),
            Self::Failed(e) => eprintln!("Warning: {model_label} not available ({e})"),
            Self::Ready(_) => {}
        }
    }
}

impl<T> fmt::Debug for ModelLoad<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready(_) => write!(f, "Ready(...)"),
            Self::Absent => write!(f, "Absent"),
            Self::Failed(msg) => write!(f, "Failed({msg:?})"),
        }
    }
}

/// Error returned by [`download_and_verify_model`].
#[derive(Debug)]
pub enum ModelDownloadError {
    /// The HTTP download failed.
    DownloadFailed(String),
    /// The hardware/OS backend (e.g. MLX) is not available on this machine.
    BackendUnavailable,
    /// The downloaded model files could not be loaded.
    /// The inner `String` carries the probe error string; it is empty when
    /// model-file corruption prevented error capture (see `ModelCorrupt` handling).
    ProbeFailed(String),
}

impl fmt::Display for ModelDownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DownloadFailed(msg) => {
                write!(
                    f,
                    "download failed: {msg}; check your network and try again"
                )
            }
            Self::BackendUnavailable => write!(
                f,
                "MLX backend unavailable; requires Apple Silicon with macOS 14 or later"
            ),
            Self::ProbeFailed(detail) if detail.is_empty() => {
                write!(f, "model probe failed; try again or re-download the model")
            }
            Self::ProbeFailed(detail) => write!(
                f,
                "model probe failed: {detail}; try again or re-download the model"
            ),
        }
    }
}

impl Error for ModelDownloadError {}

/// Download the default embedding model and verify it loads correctly.
///
/// Shows a spinner on stderr during download and probe phases.
///
/// # Prerequisites
///
/// The calling binary must invoke `rurico::model_probe::handle_probe_if_needed()`
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
                    download_model(ModelId::default()).map_err(|e| {
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

fn try_download_and_verify_with_fns<A, E, DE>(
    download_fn: impl FnOnce() -> Result<A, DE>,
    on_delete_error: impl FnOnce(io::Error),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, EmbedInitError>,
    new_fn: impl FnOnce(&A) -> Result<E, EmbedInitError>,
    delete_fn: impl FnOnce(A) -> Result<(), io::Error>,
    on_download_complete: impl FnOnce(),
) -> Result<(), ModelDownloadError>
where
    DE: fmt::Display,
    E: Embed + 'static,
{
    let paths = download_fn().map_err(|e| ModelDownloadError::DownloadFailed(e.to_string()))?;
    on_download_complete();
    let mut probe_detail: Option<String> = None;
    try_load_embedder_with_fns(
        || Ok::<_, Infallible>(Some(paths)),
        on_delete_error,
        |e| probe_detail = Some(e.to_string()),
        probe_fn,
        new_fn,
        delete_fn,
    )
    .map(|_| ())
    .map_err(|reason| match reason {
        DegradedReason::BackendUnavailable => ModelDownloadError::BackendUnavailable,
        DegradedReason::ProbeFailed => {
            ModelDownloadError::ProbeFailed(probe_detail.unwrap_or_default())
        }
        DegradedReason::NotInstalled | DegradedReason::Disabled => {
            unreachable!(
                "loader with cache=Some cannot produce NotInstalled; Disabled is caller-only"
            )
        }
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rurico::embed::MockEmbedder;

    use super::*;

    // T-019: as_ref_returns_inner_only_for_ready
    #[test]
    fn as_ref_returns_inner_only_for_ready() {
        assert_eq!(ModelLoad::Ready(42).as_ref(), Some(&42));
        assert_eq!(ModelLoad::<i32>::Absent.as_ref(), None);
        assert_eq!(ModelLoad::<i32>::Failed("e".into()).as_ref(), None);
    }

    // T-020: debug_output_contains_variant_name
    #[test]
    fn debug_output_contains_variant_name() {
        let ready = format!("{:?}", ModelLoad::Ready(1));
        assert!(ready.contains("Ready"), "expected 'Ready', got: {ready}");
        let absent = format!("{:?}", ModelLoad::<i32>::Absent);
        assert!(
            absent.contains("Absent"),
            "expected 'Absent', got: {absent}"
        );
        let failed = format!("{:?}", ModelLoad::<i32>::Failed("msg".into()));
        assert!(
            failed.contains("Failed"),
            "expected 'Failed', got: {failed}"
        );
    }

    // T-021: emit_load_hint_does_not_panic
    #[test]
    fn emit_load_hint_does_not_panic() {
        ModelLoad::<i32>::Absent.emit_load_hint("not found", "model");
        ModelLoad::Ready(1).emit_load_hint("not found", "model");
        ModelLoad::<i32>::Failed("err".into()).emit_load_hint("not found", "model");
    }

    // T-022: download_err_returns_download_failed
    #[test]
    fn download_err_returns_download_failed() {
        let complete_called = Cell::new(false);
        let result = try_download_and_verify_with_fns::<(), MockEmbedder, _>(
            || Err::<(), _>("network timeout".to_owned()),
            |_| unreachable!("on_delete_error must not fire on download failure"),
            |_| unreachable!("probe must not be called on download failure"),
            |_| unreachable!("new must not be called on download failure"),
            |_| unreachable!("delete must not be called on download failure"),
            || complete_called.set(true),
        );
        assert!(
            matches!(result, Err(ModelDownloadError::DownloadFailed(ref msg)) if msg.contains("network timeout")),
            "expected DownloadFailed with message, got {result:?}"
        );
        assert!(
            !complete_called.get(),
            "on_download_complete must not fire on download failure"
        );
    }

    // T-023: probe_backend_unavailable_returns_backend_unavailable
    #[test]
    fn probe_backend_unavailable_returns_backend_unavailable() {
        let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
            || Ok::<_, String>(()),
            |_| {},
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!("new must not be called on BackendUnavailable"),
            |_| unreachable!("delete must not be called on BackendUnavailable"),
            || {},
        );
        assert!(
            matches!(result, Err(ModelDownloadError::BackendUnavailable)),
            "expected BackendUnavailable, got {result:?}"
        );
    }

    // T-024: probe_err_captured_in_probe_failed
    #[test]
    fn probe_err_captured_in_probe_failed() {
        let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
            || Ok::<_, String>(()),
            |_| unreachable!("on_delete_error must not fire on non-corrupt probe error"),
            |_| Err(EmbedInitError::Backend("backend down".into())),
            |_| unreachable!("new must not be called when probe fails"),
            |_| unreachable!("delete must not be called on non-corrupt probe error"),
            || {},
        );
        match result {
            Err(ModelDownloadError::ProbeFailed(detail)) => assert!(
                detail.contains("backend down"),
                "probe_detail should carry error message, got {detail:?}"
            ),
            other => panic!("expected ProbeFailed, got {other:?}"),
        }
    }

    // T-025: success_returns_ok
    #[test]
    fn success_returns_ok() {
        let complete_called = Cell::new(false);
        let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
            || Ok::<_, String>(()),
            |_| unreachable!("on_delete_error must not fire on success"),
            |_| Ok(ProbeStatus::Available),
            |_| Ok(MockEmbedder::default()),
            |_| unreachable!("delete must not be called on success"),
            || complete_called.set(true),
        );
        assert!(result.is_ok(), "expected Ok(()), got {result:?}");
        assert!(
            complete_called.get(),
            "on_download_complete must fire after successful download"
        );
    }

    // T-026: new_fn_err_captured_in_probe_failed
    #[test]
    fn new_fn_err_captured_in_probe_failed() {
        let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
            || Ok::<_, String>(()),
            |_| {},
            |_| Ok(ProbeStatus::Available),
            |_| Err(EmbedInitError::Backend("alloc failed".into())),
            |_| unreachable!("delete must not be called on non-corrupt new_fn failure"),
            || {},
        );
        match result {
            Err(ModelDownloadError::ProbeFailed(detail)) => assert!(
                detail.contains("alloc failed"),
                "probe_detail should carry new_fn error message, got {detail:?}"
            ),
            other => panic!("expected ProbeFailed, got {other:?}"),
        }
    }
}
