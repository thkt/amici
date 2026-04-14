use rurico::reranker::{Artifacts, ProbeStatus, Rerank, Reranker};

use crate::model::{ModelLoad, try_load_model_with};

pub fn try_load_reranker_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, E>,
) -> ModelLoad<Box<dyn Rerank>> {
    try_load_model_with::<_, _, String>(
        || cache_check().map_err(|e| e.to_string()),
        |a: &Artifacts| match Reranker::probe(a) {
            Ok(ProbeStatus::Available) => Ok(()),
            Ok(ProbeStatus::BackendUnavailable) => {
                Err("MLX backend is unavailable".to_string())
            }
            Err(e) => Err(e.to_string()),
        },
        |a: &Artifacts| {
            Reranker::new(a)
                .map(|r| Box::new(r) as Box<dyn Rerank>)
                .map_err(|e| e.to_string())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
