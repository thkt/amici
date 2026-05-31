use std::io;

use rurico::model_init::ModelInitError;
use rurico::model_probe::ProbeStatus;
use rurico::reranker::{Artifacts, Rerank, Reranker};

use super::DegradedReason;

/// Try to load the reranking model.
///
/// # Errors
///
/// - [`DegradedReason::NotInstalled`] — `cache_check` returned `Ok(None)`.
/// - [`DegradedReason::BackendUnavailable`] — the probe reported
///   `ProbeStatus::BackendUnavailable`.
/// - [`DegradedReason::ProbeFailed`] — `cache_check` returned `Err(_)`, the
///   probe or `new_fn` reported `ModelCorrupt` (artifacts deleted before
///   returning), the probe returned another error, or `new_fn` failed.
///   `on_probe_err` is invoked for probe and non-corrupt `new_fn` errors; it is
///   **not** called for `ModelCorrupt`.
///
/// # Corrupt-model handling
///
/// When the probe or `new_fn` reports `ModelInitError::ModelCorrupt`, the
/// loader deletes the artifact files so a subsequent call can re-download. If
/// deletion itself fails, `on_delete_error` is invoked with the `io::Error` so
/// the caller can log or surface it — this crate never calls tracing directly.
pub fn try_load_reranker_with<CE>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_probe_err: impl FnOnce(ModelInitError),
) -> Result<Box<dyn Rerank>, DegradedReason> {
    try_load_reranker_with_fns(
        cache_check,
        on_delete_error,
        on_probe_err,
        Reranker::probe,
        Reranker::new,
        Artifacts::delete_files,
    )
}

fn try_load_reranker_with_fns<A, CE, R>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_probe_err: impl FnOnce(ModelInitError),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, ModelInitError>,
    new_fn: impl FnOnce(&A) -> Result<R, ModelInitError>,
    delete_fn: impl FnOnce(A) -> Result<(), io::Error>,
) -> Result<Box<dyn Rerank>, DegradedReason>
where
    R: Rerank + 'static,
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
        Ok(r) => Ok(Box::new(r) as Box<dyn Rerank>),
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
