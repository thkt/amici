pub mod embedder;
pub mod reranker;

mod download;
pub use download::download_and_verify_model;

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::io;

use rurico::embed::Embed;
use rurico::model_init::ModelInitError;
use rurico::model_probe::ProbeStatus;

use self::embedder::try_load_embedder_with_fns;
use crate::cli::{hint, warning};

/// Reason a model could not be loaded.
///
/// `Disabled` is reserved for caller-level opt-out (e.g. an environment variable);
/// the loader functions never produce it.
///
/// ADR-0009: a typed error is converted into this enum only through
/// [`degrade_with_warn`] (cause present) or [`record_degraded`] (no cause) —
/// never a bare `.map_err(|_| DegradedReason::X)`, which erases the cause from
/// the warn event. The source scan in `tests/adr_0009_degraded_routing_gate.rs`
/// denies the bare pattern, making this the mechanical gate the ADR specifies.
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

/// A degraded embedder, wrapping the [`DegradedReason`] that caused it.
///
/// The Newtype keeps the embedder-specific note ([`EmbedderDegraded::user_note`]) from being
/// reused for a reranker by accident: `user_note` lives only on `EmbedderDegraded`, so a bare
/// [`DegradedReason`] (e.g. one returned by the reranker loader) cannot call it without an
/// explicit, greppable `EmbedderDegraded(..)` wrap. The inner field is public because
/// downstream constructs the value, so this is a visible opt-in, not a hard compile-time
/// barrier. No reranker-degraded equivalent exists yet because no consumer needs one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedderDegraded(pub DegradedReason);

impl EmbedderDegraded {
    /// Returns a short user-facing note for this degraded embedder,
    /// or `None` if no message should be shown (e.g. the caller explicitly disabled the model).
    ///
    /// `download_cmd` is interpolated into the [`DegradedReason::NotInstalled`] note so the
    /// caller can surface the binary-specific recovery action (e.g. `"yomu model download"`).
    /// Other variants ignore `download_cmd` because their cause is not addressable by
    /// re-downloading the model.
    pub fn user_note(self, download_cmd: &str) -> Option<String> {
        match self.0 {
            DegradedReason::Disabled => None,
            DegradedReason::NotInstalled => Some(format!(
                "embedding model not installed; run `{download_cmd}` to enable semantic search"
            )),
            DegradedReason::BackendUnavailable | DegradedReason::ProbeFailed => {
                Some("embedding model unavailable; results from text search only".into())
            }
        }
    }
}

/// Returns a `map_err` closure that logs the original error via `tracing::warn!`
/// and yields a [`DegradedReason`] for the degraded path.
///
/// Use this at call sites that drop a typed error in favor of a coarse
/// `DegradedReason`. The original error is preserved in the structured
/// `error` field of the warn event so log consumers can still see the cause.
///
/// The structured warn event carries `error`, `reason`, and `context` fields
/// with the message `"operation degraded"`.
///
/// # Anti-pattern
///
/// Never collapse a typed error to a [`DegradedReason`] with a bare
/// `.map_err(|_| DegradedReason::X)`: the underlying cause is dropped from
/// the warn event, so log consumers cannot tell `EmbedError::Backend` apart
/// from a cache-lookup permission error once both arrive as `ProbeFailed`.
/// PR review will not flag the regression because the function's return
/// type is unchanged. Always route the conversion through this helper (or
/// [`record_degraded`] when there is no underlying error to preserve).
///
/// # Examples
///
/// ```no_run
/// # use amici::model::{degrade_with_warn, DegradedReason};
/// # fn embed_query(_: &str) -> Result<Vec<f32>, &'static str> { Ok(vec![]) }
/// let task_emb = embed_query("hello").map_err(degrade_with_warn(
///     "brief seed inference: embed_query",
///     DegradedReason::ProbeFailed,
/// ))?;
/// # Ok::<(), DegradedReason>(())
/// ```
pub fn degrade_with_warn<E: fmt::Display>(
    context: &'static str,
    reason: DegradedReason,
) -> impl FnOnce(E) -> DegradedReason {
    move |e| {
        tracing::warn!(error = %e, ?reason, context, "operation degraded");
        reason
    }
}

/// Emits a `tracing::warn!` for an already-known [`DegradedReason`] with no
/// underlying error to preserve.
///
/// Use this when the degraded path is reached from a `Result<_, DegradedReason>`
/// (or similar) where no original error exists and only the reason needs to
/// be reported. For the `map_err` case, prefer [`degrade_with_warn`].
///
/// The structured warn event carries `reason` and `context` fields with the
/// message `"operation degraded"`.
///
/// # Anti-pattern
///
/// Do not pre-collapse a `Result<_, OtherError>` to a [`DegradedReason`]
/// via `.map_err(|_| DegradedReason::X)` and then call this fn — the
/// `error` field that would let log consumers diagnose the cause is gone
/// before the warn event is emitted. Use [`degrade_with_warn`] inside the
/// `map_err` so the original error stays in the structured log.
///
/// # Examples
///
/// ```no_run
/// # use amici::model::{record_degraded, DegradedReason};
/// # fn infer_seed_paths() -> Result<Vec<String>, DegradedReason> { Ok(vec![]) }
/// let mut degraded = false;
/// match infer_seed_paths() {
///     Ok(_paths) => {}
///     Err(reason) => {
///         record_degraded(reason, "brief: seed inference");
///         degraded = true;
///     }
/// }
/// # let _ = degraded;
/// ```
pub fn record_degraded(reason: DegradedReason, context: &str) {
    tracing::warn!(?reason, context, "operation degraded");
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
    ///
    /// # Stability
    ///
    /// The erasure to `String` is deliberate — the source error type
    /// ([`ModelDownloadError`], `ModelInitError`, or a probe failure) is
    /// flattened so callers can render the message without knowing the
    /// originating layer. Downstream CLIs (sae / yomu / recall) surface this
    /// string verbatim to the user via [`Self::emit_load_hint`], so the
    /// wording is part of the user-visible API surface. Changing the message
    /// shape is a behaviour change for every downstream consumer, not an
    /// internal log refactor.
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
    /// - [`Absent`](ModelLoad::Absent): prints `"Hint: {absent_hint}"` via [`crate::cli::hint`]
    /// - [`Failed`](ModelLoad::Failed): prints `"warning: {model_label} not available ({error})"` via [`crate::cli::warning`]
    /// - [`Ready`](ModelLoad::Ready): no-op
    pub fn emit_load_hint(&self, absent_hint: &str, model_label: &str) {
        match self {
            Self::Absent => hint(absent_hint),
            Self::Failed(e) => warning(&format!("{model_label} not available ({e})")),
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
    /// The inner `Option<String>` carries the probe error detail; it is `None`
    /// when model-file corruption prevented error capture (see
    /// [`ModelInitError::ModelCorrupt`] handling).
    ProbeFailed(Option<String>),
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
            Self::ProbeFailed(None) => {
                write!(f, "model probe failed; try again or re-download the model")
            }
            Self::ProbeFailed(Some(detail)) => write!(
                f,
                "model probe failed: {detail}; try again or re-download the model"
            ),
        }
    }
}

impl Error for ModelDownloadError {}

pub(super) fn try_download_and_verify_with_fns<A, E, DE>(
    download_fn: impl FnOnce() -> Result<A, DE>,
    on_delete_error: impl FnOnce(io::Error),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, ModelInitError>,
    new_fn: impl FnOnce(&A) -> Result<E, ModelInitError>,
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
        DegradedReason::ProbeFailed => ModelDownloadError::ProbeFailed(probe_detail),
        DegradedReason::NotInstalled | DegradedReason::Disabled => {
            unreachable!(
                "loader with cache=Some cannot produce NotInstalled; Disabled is caller-only"
            )
        }
    })
}

#[cfg(test)]
mod tests;
