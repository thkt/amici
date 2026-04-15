use std::io;

use rurico::reranker::{Artifacts, ProbeStatus, Rerank, Reranker, RerankerInitError};

use crate::model::DegradedReason;

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
///   `on_err` is invoked for probe and non-corrupt `new_fn` errors; it is
///   **not** called for `ModelCorrupt`.
///
/// # Corrupt-model handling
///
/// When the probe or `new_fn` reports `RerankerInitError::ModelCorrupt`, the
/// loader deletes the artifact files so a subsequent call can re-download. If
/// deletion itself fails, `on_delete_error` is invoked with the `io::Error` so
/// the caller can log or surface it — this crate never calls tracing directly.
pub fn try_load_reranker_with<CE>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_err: impl FnOnce(RerankerInitError),
) -> Result<Box<dyn Rerank>, DegradedReason> {
    try_load_reranker_with_fns(
        cache_check,
        on_delete_error,
        on_err,
        Reranker::probe,
        Reranker::new,
        Artifacts::delete_files,
    )
}

fn try_load_reranker_with_fns<A, CE, R>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_err: impl FnOnce(RerankerInitError),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, RerankerInitError>,
    new_fn: impl FnOnce(&A) -> Result<R, RerankerInitError>,
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
        Err(RerankerInitError::ModelCorrupt { .. }) => {
            if let Err(io_err) = delete_fn(artifacts) {
                on_delete_error(io_err);
            }
            return Err(DegradedReason::ProbeFailed);
        }
        Err(e) => {
            on_err(e);
            return Err(DegradedReason::ProbeFailed);
        }
    }
    match new_fn(&artifacts) {
        Ok(r) => Ok(Box::new(r) as Box<dyn Rerank>),
        Err(RerankerInitError::ModelCorrupt { .. }) => {
            if let Err(io_err) = delete_fn(artifacts) {
                on_delete_error(io_err);
            }
            Err(DegradedReason::ProbeFailed)
        }
        Err(e) => {
            on_err(e);
            Err(DegradedReason::ProbeFailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rurico::reranker::{RankedResult, RerankerError};

    use super::*;

    struct StubReranker;

    impl Rerank for StubReranker {
        fn score(&self, _: &str, _: &str) -> Result<f32, RerankerError> {
            Ok(0.0)
        }
        fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
            Ok(vec![0.0; pairs.len()])
        }
        fn rerank(&self, _: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
            Ok(documents
                .iter()
                .enumerate()
                .map(|(i, _)| RankedResult {
                    index: i,
                    score: 0.0,
                })
                .collect())
        }
    }

    fn cache_present() -> impl FnOnce() -> Result<Option<()>, &'static str> {
        || Ok(Some(()))
    }

    // T-006: cache_check=Ok(None) → Err(NotInstalled)
    #[test]
    fn cache_none_returns_not_installed() {
        let result = try_load_reranker_with(
            || Ok::<_, &str>(None),
            |_| unreachable!("on_delete_error must not be called"),
            |_| unreachable!("on_err must not be called"),
        );
        assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
    }

    // T-007: cache_check=Err → Err(ProbeFailed), on_err NOT called
    #[test]
    fn cache_err_returns_probe_failed() {
        let result = try_load_reranker_with(
            || Err::<Option<rurico::reranker::Artifacts>, _>("cache broken"),
            |_| unreachable!("on_delete_error must not be called on cache error"),
            |_| unreachable!("on_err must not be called on cache error"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
    }

    // T-008: probe=Available, new=Ok → Ok
    #[test]
    fn probe_available_new_ok_returns_ready() {
        let result = try_load_reranker_with_fns(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called on success"),
            |_| unreachable!("on_err must not be called on success"),
            |_| Ok(ProbeStatus::Available),
            |_| Ok(StubReranker),
            |_| unreachable!("delete must not be called on success"),
        );
        assert!(result.is_ok());
    }

    // T-009: probe=BackendUnavailable → Err(BackendUnavailable)
    #[test]
    fn probe_backend_unavailable_returns_backend_unavailable() {
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called on BackendUnavailable"),
            |_| unreachable!("on_err must not be called on BackendUnavailable"),
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!("new must not be called when backend unavailable"),
            |_| unreachable!("delete must not be called when backend unavailable"),
        );
        assert_eq!(result.err(), Some(DegradedReason::BackendUnavailable));
    }

    // T-010: probe=Err(Backend) → on_err called with error detail, Err(ProbeFailed)
    #[test]
    fn probe_err_invokes_on_err() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called on non-corrupt probe error"),
            |e| captured.set(Some(e.to_string())),
            |_| Err(RerankerInitError::Backend("probe failed".into())),
            |_| unreachable!("new must not be called when probe fails"),
            |_| unreachable!("delete must not be called on non-corrupt probe error"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        let msg = captured.into_inner().expect("on_err should fire");
        assert!(
            msg.contains("probe failed"),
            "on_err message should carry detail, got {msg:?}"
        );
    }

    // T-011: new=Err → on_err called with error detail, Err(ProbeFailed)
    #[test]
    fn new_err_invokes_on_err() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called when probe succeeds"),
            |e| captured.set(Some(e.to_string())),
            |_| Ok(ProbeStatus::Available),
            |_| Err(RerankerInitError::Backend("alloc failed".into())),
            |_| unreachable!("delete must not be called on new_fn failure"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        let msg = captured.into_inner().expect("on_err should fire");
        assert!(
            msg.contains("alloc failed"),
            "on_err message should carry detail, got {msg:?}"
        );
    }

    // T-012: probe=ModelCorrupt, delete=Ok → on_delete_error NOT called, on_err NOT called, Err(ProbeFailed)
    #[test]
    fn corrupt_delete_ok_skips_on_delete_error() {
        let on_delete_error_called = Cell::new(false);
        let on_err_called = Cell::new(false);
        let delete_called = Cell::new(false);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| on_delete_error_called.set(true),
            |_| on_err_called.set(true),
            |_| {
                Err(RerankerInitError::ModelCorrupt {
                    reason: "bad weights".into(),
                })
            },
            |_| unreachable!("new must not be called after corrupt probe"),
            |_| {
                delete_called.set(true);
                Ok(())
            },
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        assert!(delete_called.get(), "delete_fn should be called once");
        assert!(
            !on_delete_error_called.get(),
            "on_delete_error must not be called when delete succeeds"
        );
        assert!(
            !on_err_called.get(),
            "on_err must not be called on ModelCorrupt"
        );
    }

    // T-013: probe=ModelCorrupt, delete=Err(io::Error) → on_delete_error called, on_err NOT called, Err(ProbeFailed)
    #[test]
    fn corrupt_delete_err_invokes_on_delete_error() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let on_err_called = Cell::new(false);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |e| captured.set(Some(e.to_string())),
            |_| on_err_called.set(true),
            |_| {
                Err(RerankerInitError::ModelCorrupt {
                    reason: "bad weights".into(),
                })
            },
            |_| unreachable!("new must not be called after corrupt probe"),
            |_| Err(io::Error::other("disk full")),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        let msg = captured.into_inner().expect("on_delete_error should fire");
        assert!(
            msg.contains("disk full"),
            "captured error should carry io message, got {msg:?}"
        );
        assert!(
            !on_err_called.get(),
            "on_err must not be called on ModelCorrupt"
        );
    }

    // T-014: public wrapper delegates correctly — cache empty → NotInstalled (wiring test)
    #[test]
    fn public_wrapper_absent_when_cache_empty() {
        let result = try_load_reranker_with(
            || Ok::<Option<Artifacts>, &str>(None),
            |_| unreachable!("on_delete_error must not be called"),
            |_| unreachable!("on_err must not be called when cache is empty"),
        );
        assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
    }

    // T-015: probe=Available, new_fn=ModelCorrupt → delete_fn called, on_err NOT called, Err(ProbeFailed)
    // Note: Reranker::new currently documents only RerankerInitError::Backend, but the type
    // allows ModelCorrupt; this test defends against future changes in the backend.
    #[test]
    fn new_fn_corrupt_deletes_artifacts() {
        let on_delete_error_called = Cell::new(false);
        let on_err_called = Cell::new(false);
        let delete_called = Cell::new(false);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| on_delete_error_called.set(true),
            |_| on_err_called.set(true),
            |_| Ok(ProbeStatus::Available),
            |_| {
                Err(RerankerInitError::ModelCorrupt {
                    reason: "bad weights".into(),
                })
            },
            |_| {
                delete_called.set(true);
                Ok(())
            },
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        assert!(
            delete_called.get(),
            "delete_fn should be called on new_fn ModelCorrupt"
        );
        assert!(
            !on_delete_error_called.get(),
            "on_delete_error must not be called when delete succeeds"
        );
        assert!(
            !on_err_called.get(),
            "on_err must not be called on ModelCorrupt"
        );
    }
}
