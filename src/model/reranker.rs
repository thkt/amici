use std::io;

use rurico::reranker::{Artifacts, ProbeStatus, Rerank, Reranker, RerankerInitError};

use crate::model::ModelLoad;

/// Try to load the reranking model.
///
/// # Corrupt-model handling
///
/// When the probe reports `RerankerInitError::ModelCorrupt`, the loader deletes
/// the artifact files so a subsequent call can re-download. If deletion itself
/// fails, `on_delete_error` is invoked with the `io::Error` so the caller can log
/// or surface it — this crate never calls tracing directly.
pub fn try_load_reranker_with<CE>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, CE>,
    on_delete_error: impl FnOnce(io::Error),
) -> ModelLoad<Box<dyn Rerank>>
where
    CE: std::fmt::Display,
{
    try_load_reranker_with_fns(
        cache_check,
        on_delete_error,
        Reranker::probe,
        Reranker::new,
        Artifacts::delete_files,
    )
}

pub(crate) fn try_load_reranker_with_fns<A, CE, R>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, RerankerInitError>,
    new_fn: impl FnOnce(&A) -> Result<R, RerankerInitError>,
    delete_fn: impl FnOnce(A) -> Result<(), io::Error>,
) -> ModelLoad<Box<dyn Rerank>>
where
    CE: std::fmt::Display,
    R: Rerank + 'static,
{
    let artifacts = match cache_check() {
        Ok(Some(a)) => a,
        Ok(None) => return ModelLoad::Absent,
        Err(e) => return ModelLoad::Failed(e.to_string()),
    };
    match probe_fn(&artifacts) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            return ModelLoad::Failed("MLX backend is unavailable".to_string());
        }
        Err(RerankerInitError::ModelCorrupt { reason }) => {
            if let Err(io_err) = delete_fn(artifacts) {
                on_delete_error(io_err);
            }
            return ModelLoad::Failed(format!("model corrupt: {reason}"));
        }
        Err(e) => return ModelLoad::Failed(e.to_string()),
    }
    match new_fn(&artifacts) {
        Ok(r) => ModelLoad::Ready(Box::new(r) as Box<dyn Rerank>),
        Err(e) => ModelLoad::Failed(e.to_string()),
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

    fn assert_failed_containing(result: ModelLoad<Box<dyn Rerank>>, expected: &str) {
        match result {
            ModelLoad::Failed(msg) => assert!(
                msg.contains(expected),
                "expected message containing {expected:?}, got {msg:?}"
            ),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }

    // T-006: cache_check=Ok(None) → Absent
    #[test]
    fn cache_none_returns_absent() {
        let result = try_load_reranker_with(|| Ok::<_, &str>(None), |_| {});
        assert!(matches!(result, ModelLoad::Absent));
    }

    // T-007: cache_check=Err → Failed with error message
    #[test]
    fn cache_err_returns_failed() {
        let result = try_load_reranker_with(
            || Err::<Option<rurico::reranker::Artifacts>, _>("cache broken"),
            |_| unreachable!("on_delete_error must not be called on cache error"),
        );
        assert_failed_containing(result, "cache broken");
    }

    // T-008: probe=Available, new=Ok → Ready
    #[test]
    fn probe_available_new_ok_returns_ready() {
        let result = try_load_reranker_with_fns(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called on success"),
            |_| Ok(ProbeStatus::Available),
            |_| Ok(StubReranker),
            |_| unreachable!("delete must not be called on success"),
        );
        assert!(matches!(result, ModelLoad::Ready(_)));
    }

    // T-009: probe=BackendUnavailable → Failed("MLX backend is unavailable")
    #[test]
    fn probe_backend_unavailable_returns_failed() {
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called on BackendUnavailable"),
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!("new must not be called when backend unavailable"),
            |_| unreachable!("delete must not be called when backend unavailable"),
        );
        assert_failed_containing(result, "MLX backend is unavailable");
    }

    // T-010: probe=Err(Backend) → Failed with error message
    #[test]
    fn probe_err_returns_failed() {
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called on non-corrupt probe error"),
            |_| Err(RerankerInitError::Backend("probe failed".into())),
            |_| unreachable!("new must not be called when probe fails"),
            |_| unreachable!("delete must not be called on non-corrupt probe error"),
        );
        assert_failed_containing(result, "probe failed");
    }

    // T-011: new=Err → Failed with error message
    #[test]
    fn new_err_returns_failed() {
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called when probe succeeds"),
            |_| Ok(ProbeStatus::Available),
            |_| Err(RerankerInitError::Backend("alloc failed".into())),
            |_| unreachable!("delete must not be called on new_fn failure"),
        );
        assert_failed_containing(result, "alloc failed");
    }

    // T-012: probe=ModelCorrupt, delete=Ok → on_delete_error NOT called, Failed("model corrupt: ...")
    #[test]
    fn corrupt_delete_ok_skips_on_delete_error() {
        let on_delete_error_called = Cell::new(false);
        let delete_called = Cell::new(false);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |_| on_delete_error_called.set(true),
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
        assert_failed_containing(result, "bad weights");
        assert!(delete_called.get(), "delete_fn should be called once");
        assert!(
            !on_delete_error_called.get(),
            "on_delete_error must not be called when delete succeeds"
        );
    }

    // T-013: probe=ModelCorrupt, delete=Err(io::Error) → on_delete_error called, Failed
    #[test]
    fn corrupt_delete_err_invokes_on_delete_error() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let result = try_load_reranker_with_fns::<_, _, StubReranker>(
            cache_present(),
            |e| captured.set(Some(e.to_string())),
            |_| {
                Err(RerankerInitError::ModelCorrupt {
                    reason: "bad weights".into(),
                })
            },
            |_| unreachable!("new must not be called after corrupt probe"),
            |_| Err(io::Error::other("disk full")),
        );
        assert!(matches!(result, ModelLoad::Failed(_)));
        let msg = captured.into_inner().expect("on_delete_error should fire");
        assert!(
            msg.contains("disk full"),
            "captured error should carry io message, got {msg:?}"
        );
    }
}
