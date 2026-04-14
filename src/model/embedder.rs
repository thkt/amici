use std::io;
use std::sync::Arc;

use rurico::embed::{Artifacts, Embed, EmbedInitError, Embedder, ProbeStatus};

/// Reason the embedder could not be loaded.
///
/// `Disabled` is reserved for caller-level opt-out (e.g. an environment variable);
/// [`try_load_embedder_with`] never produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedReason {
    Disabled,
    NotInstalled,
    BackendUnavailable,
    ProbeFailed,
}

/// Try to load the embedding model.
///
/// # Corrupt-model handling
///
/// When the probe reports `EmbedInitError::ModelCorrupt`, the loader deletes
/// the artifact files so a subsequent call can re-download. If deletion itself
/// fails, `on_corrupt` is invoked with the `io::Error` so the caller can log
/// or surface it — this crate never calls tracing directly.
pub fn try_load_embedder_with<CE>(
    cache_check: impl FnOnce() -> Result<Option<Artifacts>, CE>,
    on_corrupt: impl FnOnce(io::Error),
) -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_with_fns(
        cache_check,
        on_corrupt,
        |_| {},
        Embedder::probe,
        Embedder::new,
        Artifacts::delete_files,
    )
}

fn try_load_embedder_with_fns<A, E, CE>(
    cache_check: impl FnOnce() -> Result<Option<A>, CE>,
    on_corrupt: impl FnOnce(io::Error),
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
                on_corrupt(io_err);
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

    // T-101: cache_check=Ok(None) → Err(NotInstalled)
    #[test]
    fn t101_cache_none_returns_not_installed() {
        let on_corrupt_called = Cell::new(false);
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(None),
            |_| on_corrupt_called.set(true),
            |_| unreachable!("on_probe_err must not be called when cache is empty"),
            |_| unreachable!("probe must not be called when cache is empty"),
            |_| unreachable!("new must not be called when cache is empty"),
            |_| unreachable!("delete must not be called when cache is empty"),
        );
        assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
        assert!(!on_corrupt_called.get());
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
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| {},
            |_| unreachable!("on_probe_err must not be called on BackendUnavailable"),
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!("new must not be called when backend unavailable"),
            |_| unreachable!("delete must not be called when backend unavailable"),
        );
        assert_eq!(result.err(), Some(DegradedReason::BackendUnavailable));
    }

    // T-104: probe=ModelCorrupt, delete=Ok → on_corrupt NOT called, Err(ProbeFailed)
    #[test]
    fn t104_corrupt_delete_ok_skips_on_corrupt() {
        let on_corrupt_called = Cell::new(false);
        let delete_called = Cell::new(false);
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| on_corrupt_called.set(true),
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
            !on_corrupt_called.get(),
            "on_corrupt must not be called when delete succeeds"
        );
    }

    // T-105: probe=ModelCorrupt, delete=Err(io::Error) → on_corrupt called, Err(ProbeFailed)
    #[test]
    fn t105_corrupt_delete_err_invokes_on_corrupt() {
        let captured: Cell<Option<String>> = Cell::new(None);
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
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
        let msg = captured.into_inner().expect("on_corrupt should fire");
        assert!(
            msg.contains("disk full"),
            "captured error should carry io message, got {msg:?}"
        );
    }

    // T-106: cache=Some, probe=Available, new=Ok → Ok(Arc<dyn Embed>)
    #[test]
    fn t106_success_returns_arc_embed() {
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
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
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| unreachable!("on_corrupt must not be called"),
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
            |_| unreachable!("on_corrupt must not be called"),
        );
        assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
    }

    // T-113: DegradedReason has 4 compile-time exhaustive variants
    #[test]
    fn t113_degraded_reason_exhaustive_variants() {
        fn classify(r: DegradedReason) -> &'static str {
            match r {
                DegradedReason::Disabled => "disabled",
                DegradedReason::NotInstalled => "not-installed",
                DegradedReason::BackendUnavailable => "backend-unavailable",
                DegradedReason::ProbeFailed => "probe-failed",
            }
        }
        assert_eq!(classify(DegradedReason::Disabled), "disabled");
        assert_eq!(classify(DegradedReason::NotInstalled), "not-installed");
        assert_eq!(
            classify(DegradedReason::BackendUnavailable),
            "backend-unavailable"
        );
        assert_eq!(classify(DegradedReason::ProbeFailed), "probe-failed");
    }

    // T-114: loader never returns Err(Disabled) across all 6 error-returning branches
    #[test]
    fn t114_loader_never_produces_disabled() {
        // Branch A: cache Ok(None) → NotInstalled
        let a = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(None),
            |_| {},
            |_| unreachable!(),
            |_| unreachable!(),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        assert_ne!(a.err(), Some(DegradedReason::Disabled));

        // Branch B: cache Err → ProbeFailed
        let b = try_load_embedder_with_fns::<(), StubEmbedder, _>(
            || Err::<Option<()>, _>("x"),
            |_| {},
            |_| unreachable!(),
            |_| unreachable!(),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        assert_ne!(b.err(), Some(DegradedReason::Disabled));

        // Branch C: probe BackendUnavailable → BackendUnavailable
        let c = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| {},
            |_| unreachable!(),
            |_| Ok(ProbeStatus::BackendUnavailable),
            |_| unreachable!(),
            |_| unreachable!(),
        );
        assert_ne!(c.err(), Some(DegradedReason::Disabled));

        // Branch D: probe ModelCorrupt → ProbeFailed
        let d = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| {},
            |_| unreachable!(),
            |_| Err(EmbedInitError::ModelCorrupt { reason: "x".into() }),
            |_| unreachable!(),
            |_| Ok(()),
        );
        assert_ne!(d.err(), Some(DegradedReason::Disabled));

        // Branch E: probe returns non-corrupt Backend error → ProbeFailed
        let e = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| {},
            |_| {}, // on_probe_err is called here
            |_| Err(EmbedInitError::Backend("spawn failed".into())),
            |_| unreachable!("new must not be called when probe fails"),
            |_| unreachable!("delete must not be called when probe fails with non-corrupt error"),
        );
        assert_ne!(e.err(), Some(DegradedReason::Disabled));

        // Branch F: probe=Available, new_fn returns Err → ProbeFailed
        let f = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| {},
            |_| unreachable!(),
            |_| Ok(ProbeStatus::Available),
            |_| Err(EmbedInitError::Backend("alloc failed".into())),
            |_| unreachable!("delete must not be called on new_fn failure"),
        );
        assert_ne!(f.err(), Some(DegradedReason::Disabled));
    }

    // T-115: probe=Available, new_fn=Err → Err(ProbeFailed)
    #[test]
    fn t115_new_fn_err_returns_probe_failed() {
        let result = try_load_embedder_with_fns::<(), StubEmbedder, &str>(
            || Ok(Some(())),
            |_| {},
            |_| unreachable!("on_probe_err must not be called when probe succeeds"),
            |_| Ok(ProbeStatus::Available),
            |_| Err(EmbedInitError::Backend("alloc failed".into())),
            |_| unreachable!("delete must not be called on new_fn failure"),
        );
        assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
    }
}
