use rurico::reranker::{Artifacts, ProbeStatus, Rerank, Reranker, RerankerInitError};

use crate::model::{ModelLoad, try_load_model_with};

pub fn try_load_reranker_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, E>,
) -> ModelLoad<Box<dyn Rerank>> {
    try_load_reranker_with_fns(cache_check, Reranker::probe, Reranker::new)
}

pub(crate) fn try_load_reranker_with_fns<A, CE, R>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, RerankerInitError>,
    new_fn: impl FnOnce(&A) -> Result<R, RerankerInitError>,
) -> ModelLoad<Box<dyn Rerank>>
where
    CE: std::fmt::Display,
    R: Rerank + 'static,
{
    try_load_model_with(
        || cache_check().map_err(|e| e.to_string()),
        |a| match probe_fn(a) {
            Ok(ProbeStatus::Available) => Ok(()),
            Ok(ProbeStatus::BackendUnavailable) => Err("MLX backend is unavailable".to_string()),
            Err(e) => Err(e.to_string()),
        },
        |a| {
            new_fn(a)
                .map(|r| Box::new(r) as Box<dyn Rerank>)
                .map_err(|e| e.to_string())
        },
    )
}

#[cfg(test)]
mod tests {
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

    // T-006: cache_check=Ok(None) → Absent
    #[test]
    fn cache_none_returns_absent() {
        let result = try_load_reranker_with(|| Ok::<_, &str>(None));
        assert!(matches!(result, ModelLoad::Absent));
    }

    // T-007: cache_check=Err → Failed with error message
    #[test]
    fn cache_err_returns_failed() {
        let result = try_load_reranker_with(|| {
            Err::<Option<rurico::reranker::Artifacts>, _>("cache broken")
        });
        match result {
            ModelLoad::Failed(msg) => assert!(msg.contains("cache broken")),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }

    // T-008: probe=Available, new=Ok → Ready
    #[test]
    fn probe_available_new_ok_returns_ready() {
        let result = try_load_reranker_with_fns::<(), _, StubReranker>(
            || Ok::<Option<()>, &str>(Some(())),
            |_| Ok(ProbeStatus::Available),
            |_| Ok(StubReranker),
        );
        assert!(matches!(result, ModelLoad::Ready(_)));
    }

    // T-009: probe=BackendUnavailable → Failed("MLX backend is unavailable")
    #[test]
    fn probe_backend_unavailable_returns_failed() {
        let result = try_load_reranker_with_fns::<(), _, StubReranker>(
            || Ok::<Option<()>, &str>(Some(())),
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!("new must not be called when backend unavailable"),
        );
        match result {
            ModelLoad::Failed(msg) => assert!(msg.contains("MLX backend is unavailable")),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }

    // T-010: probe=Err → Failed with error message
    #[test]
    fn probe_err_returns_failed() {
        let result = try_load_reranker_with_fns::<(), _, StubReranker>(
            || Ok::<Option<()>, &str>(Some(())),
            |_| Err(RerankerInitError::Backend("probe failed".into())),
            |_| unreachable!("new must not be called when probe fails"),
        );
        match result {
            ModelLoad::Failed(msg) => assert!(msg.contains("probe failed")),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }

    // T-011: new=Err → Failed with error message
    #[test]
    fn new_err_returns_failed() {
        let result = try_load_reranker_with_fns::<(), _, StubReranker>(
            || Ok::<Option<()>, &str>(Some(())),
            |_| Ok(ProbeStatus::Available),
            |_| Err(RerankerInitError::Backend("alloc failed".into())),
        );
        match result {
            ModelLoad::Failed(msg) => assert!(msg.contains("alloc failed")),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }
}
