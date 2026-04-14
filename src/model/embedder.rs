use std::io;
use std::sync::Arc;

use rurico::embed::{Artifacts, Embed, EmbedInitError, Embedder, ProbeStatus};

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
///   `on_probe_err` is invoked for probe errors only; it is **not** called for
///   `ModelCorrupt` or other `new_fn` errors.
///
/// # Corrupt-model handling
///
/// When the probe or `new_fn` reports `EmbedInitError::ModelCorrupt`, the loader
/// deletes the artifact files so a subsequent call can re-download. If deletion
/// itself fails, `on_delete_error` is invoked with the `io::Error` so the caller
/// can log or surface it — this crate never calls tracing directly.
pub fn try_load_embedder_with<CE>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_probe_err: impl FnOnce(EmbedInitError),
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

fn try_load_embedder_with_fns<A, E, CE>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    on_delete_error: impl FnOnce(io::Error),
    on_probe_err: impl FnOnce(EmbedInitError),
    probe_fn: impl FnOnce(&A) -> Result<ProbeStatus, EmbedInitError>,
    new_fn: impl FnOnce(&A) -> Result<E, EmbedInitError>,
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
        Err(EmbedInitError::ModelCorrupt { .. }) => {
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
        Err(EmbedInitError::ModelCorrupt { .. }) => {
            if let Err(io_err) = delete_fn(artifacts) {
                on_delete_error(io_err);
            }
            Err(DegradedReason::ProbeFailed)
        }
        Err(_) => Err(DegradedReason::ProbeFailed),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rurico::embed::{ChunkedEmbedding, EmbedError};

    use super::*;

    struct StubEmbedder;

    impl Embed for StubEmbedder {
        fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![])
        }
        fn embed_document(&self, _: &str) -> Result<ChunkedEmbedding, EmbedError> {
            Ok(ChunkedEmbedding { chunks: vec![] })
        }
        fn embed_text(&self, _: &str, _: &str) -> Result<Vec<f32>, EmbedError> {
            Ok(vec![])
        }
    }

    fn cache_present() -> impl FnOnce() -> Result<Option<()>, &'static str> {
        || Ok(Some(()))
    }

    // T-101: cache_check=Ok(None) → Err(NotInstalled)
    #[test]
    fn t101_cache_none_returns_not_installed() {
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(None),
            |_| unreachable!("on_delete_error must not be called when cache is empty"),
            |_| unreachable!("on_probe_err must not be called when cache is empty"),
            |_| unreachable!("probe must not be called when cache is empty"),
            |_| unreachable!("new must not be called when cache is empty"),
            |_| unreachable!("delete must not be called when cache is empty"),
        );
        assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
    }

    // T-102: cache_check=Err → Err(ProbeFailed)
    #[test]
    fn t102_cache_err_returns_probe_failed() {
        let result = try_load_embedder_with_fns::<(), StubEmbedder, _>(
            || Err::<Option<()>, _>("cache broken"),
            |_| {},
            |_| unreachable!("on_probe_err must not be called when cache_check fails"),
            |_| unreachable!("probe must not be called when cache_check fails"),
            |_| unreachable!("new must not be called when cache_check fails"),
            |_| unreachable!("delete must not be called when cache_check fails"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
    }

    // T-103: probe=BackendUnavailable → Err(BackendUnavailable)
    #[test]
    fn t103_backend_unavailable_returns_backend_unavailable() {
        let result = try_load_embedder_with_fns::<_, StubEmbedder, _>(
            cache_present(),
            |_| {},
            |_| unreachable!("on_probe_err must not be called on BackendUnavailable"),
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!("new must not be called when backend unavailable"),
            |_| unreachable!("delete must not be called when backend unavailable"),
        );
        assert_eq!(result.err(), Some(DegradedReason::BackendUnavailable));
    }

    // T-104: probe=ModelCorrupt, delete=Ok → on_delete_error NOT called, Err(ProbeFailed)
    #[test]
    fn t104_corrupt_delete_ok_skips_on_delete_error() {
        let on_delete_error_called = Cell::new(false);
        let delete_called = Cell::new(false);
        let result = try_load_embedder_with_fns::<_, StubEmbedder, _>(
            cache_present(),
            |_| on_delete_error_called.set(true),
            |_| unreachable!("on_probe_err must not be called on ModelCorrupt"),
            |_| {
                Err(EmbedInitError::ModelCorrupt {
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
    }

    // T-105: probe=ModelCorrupt, delete=Err(io::Error) → on_delete_error called, Err(ProbeFailed)
    #[test]
    fn t105_corrupt_delete_err_invokes_on_delete_error() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let result = try_load_embedder_with_fns::<_, StubEmbedder, _>(
            cache_present(),
            |e| captured.set(Some(e.to_string())),
            |_| unreachable!("on_probe_err must not be called on ModelCorrupt"),
            |_| {
                Err(EmbedInitError::ModelCorrupt {
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
    }

    // T-106: cache=Some, probe=Available, new=Ok → Ok(Arc<dyn Embed>)
    #[test]
    fn t106_success_returns_arc_embed() {
        let result = try_load_embedder_with_fns(
            cache_present(),
            |_| {},
            |_| unreachable!("on_probe_err must not be called on success"),
            |_| Ok(ProbeStatus::Available),
            |_| Ok(StubEmbedder),
            |_| unreachable!("delete must not be called on success"),
        );
        let embedder = result.expect("loader should succeed");
        assert!(embedder.embed_query("hello").is_ok());
    }

    // T-107: probe=Backend err → on_probe_err called with error detail
    #[test]
    fn t107_probe_backend_err_invokes_on_probe_err() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let result = try_load_embedder_with_fns::<_, StubEmbedder, _>(
            cache_present(),
            |_| unreachable!("on_delete_error must not be called"),
            |e| captured.set(Some(e.to_string())),
            |_| Err(EmbedInitError::Backend("spawn failed".into())),
            |_| unreachable!("new must not be called when probe fails"),
            |_| unreachable!("delete must not be called on non-corrupt probe error"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        let msg = captured.into_inner().expect("on_probe_err should fire");
        assert!(
            msg.contains("spawn failed"),
            "captured error should carry backend message, got {msg:?}"
        );
    }

    // T-108: public wrapper delegates correctly — cache empty → NotInstalled (wiring test)
    #[test]
    fn t108_public_wrapper_absent_when_cache_empty() {
        let result = try_load_embedder_with(
            || Ok::<Option<Artifacts>, &str>(None),
            |_| unreachable!("on_delete_error must not be called"),
            |_| unreachable!("on_err must not be called when cache is empty"),
        );
        assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
    }

    // T-115: probe=Available, new_fn=Err → Err(ProbeFailed)
    #[test]
    fn t115_new_fn_err_returns_probe_failed() {
        let result = try_load_embedder_with_fns::<_, StubEmbedder, _>(
            cache_present(),
            |_| {},
            |_| unreachable!("on_probe_err must not be called when probe succeeds"),
            |_| Ok(ProbeStatus::Available),
            |_| Err(EmbedInitError::Backend("alloc failed".into())),
            |_| unreachable!("delete must not be called on new_fn failure"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
    }

    // T-116: probe=Available, new_fn=ModelCorrupt → delete_fn called, on_probe_err NOT called, Err(ProbeFailed)
    // Note: Embedder::new currently documents only EmbedInitError::Backend, but the type
    // allows ModelCorrupt; this test defends against future changes in the backend.
    #[test]
    fn t116_new_fn_corrupt_deletes_artifacts() {
        let on_delete_error_called = Cell::new(false);
        let delete_called = Cell::new(false);
        let result = try_load_embedder_with_fns::<_, StubEmbedder, _>(
            cache_present(),
            |_| on_delete_error_called.set(true),
            |_| unreachable!("on_probe_err must not be called when new_fn reports ModelCorrupt"),
            |_| Ok(ProbeStatus::Available),
            |_| {
                Err(EmbedInitError::ModelCorrupt {
                    reason: "bad weights".into(),
                })
            },
            |_| {
                delete_called.set(true);
                Ok(())
            },
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
        assert!(delete_called.get(), "delete_fn should be called on new_fn ModelCorrupt");
        assert!(
            !on_delete_error_called.get(),
            "on_delete_error must not be called when delete succeeds"
        );
    }
}
