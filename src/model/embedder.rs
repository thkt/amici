use std::io;
use std::sync::Arc;

use rurico::embed::{Artifacts, Embed, Embedder, ModelId, cached_artifacts};
use rurico::model_init::ModelInitError;
use rurico::model_probe::ProbeStatus;

pub use super::{DegradedReason, degraded_reason_user_note};

/// Try to load the embedding model.
///
/// # Errors
///
/// - [`DegradedReason::NotInstalled`] — `cache_check` returned `Ok(None)`.
/// - [`DegradedReason::BackendUnavailable`] — the probe reported
///   `ProbeStatus::BackendUnavailable`.
/// - [`DegradedReason::ProbeFailed`] — `cache_check` returned `Err(_)`, the
///   probe or `new_fn` reported `ModelCorrupt` (artifacts deleted before
///   returning), the probe returned another error, or `new_fn` failed.
///   `on_probe_err` is invoked for probe errors and non-corrupt `new_fn`
///   errors; it is **not** called for `ModelCorrupt`.
///
/// # Corrupt-model handling
///
/// When the probe or `new_fn` reports `ModelInitError::ModelCorrupt`, the loader
/// deletes the artifact files so a subsequent call can re-download. If deletion
/// itself fails, `on_delete_error` is invoked with the `io::Error` so the caller
/// can log or surface it — this crate never calls tracing directly.
pub fn try_load_embedder_with<CE>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_probe_err: impl FnOnce(ModelInitError),
) -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_with_fns(
        cache_check,
        on_delete_error,
        on_probe_err,
        Embedder::probe,
        Embedder::new,
        Artifacts::delete_files,
    )
}

/// Preset wrapper around [`try_load_embedder_with`] that uses the default
/// model cache lookup and a default `tracing::warn!`-based logging policy
/// for the `on_delete_error` and `on_probe_err` callbacks.
///
/// Use this when the caller has no special logging requirements. Callers
/// that need custom logging should use [`try_load_embedder_with`] directly.
///
/// # Errors
///
/// Same as [`try_load_embedder_with`].
///
/// # Examples
///
/// ```no_run
/// use amici::model::embedder::try_load_embedder_default_logging;
///
/// let embedder = try_load_embedder_default_logging()?;
/// # Ok::<(), amici::model::embedder::DegradedReason>(())
/// ```
pub fn try_load_embedder_default_logging() -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_default_logging_with_fns(
        || cached_artifacts(ModelId::default()),
        Embedder::probe,
        Embedder::new,
        Artifacts::delete_files,
    )
}

pub(super) fn try_load_embedder_default_logging_with_fns<A, E, CE>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, ModelInitError>,
    new_fn: impl FnOnce(&A) -> Result<E, ModelInitError>,
    delete_fn: impl FnOnce(A) -> Result<(), io::Error>,
) -> Result<Arc<dyn Embed>, DegradedReason>
where
    E: Embed + 'static,
{
    try_load_embedder_with_fns(
        cache_check,
        |e| tracing::warn!(error = %e, model_kind = "embed", "failed to delete corrupt model files"),
        |e| tracing::warn!(error = %e, model_kind = "embed", "embedder probe failed"),
        probe_fn,
        new_fn,
        delete_fn,
    )
}

pub(super) fn try_load_embedder_with_fns<A, E, CE>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_probe_err: impl FnOnce(ModelInitError),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, ModelInitError>,
    new_fn: impl FnOnce(&A) -> Result<E, ModelInitError>,
    delete_fn: impl FnOnce(A) -> Result<(), io::Error>,
) -> Result<Arc<dyn Embed>, DegradedReason>
where
    E: Embed + 'static,
{
    let artifacts = match cache_check() {
        Ok(Some(a)) => a,
        Ok(None) => return Err(DegradedReason::NotInstalled),
        Err(_) => return Err(DegradedReason::ProbeFailed),
    };
    match probe_fn(&artifacts) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => return Err(DegradedReason::BackendUnavailable),
        Err(ModelInitError::ModelCorrupt { .. }) => {
            if let Err(io_err) = delete_fn(artifacts) {
                on_delete_error(io_err);
            }
            return Err(DegradedReason::ProbeFailed);
        }
        Err(e) => {
            on_probe_err(e);
            return Err(DegradedReason::ProbeFailed);
        }
    }
    match new_fn(&artifacts) {
        Ok(e) => Ok(Arc::new(e) as Arc<dyn Embed>),
        Err(ModelInitError::ModelCorrupt { .. }) => {
            if let Err(io_err) = delete_fn(artifacts) {
                on_delete_error(io_err);
            }
            Err(DegradedReason::ProbeFailed)
        }
        Err(e) => {
            on_probe_err(e);
            Err(DegradedReason::ProbeFailed)
        }
    }
}

#[cfg(test)]
mod tests;
