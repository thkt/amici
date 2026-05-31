use std::cell::Cell;

use rurico::embed::MockEmbedder;

use super::*;

fn cache_present() -> impl FnOnce() -> Result<Option<()>, &'static str> {
    || Ok(Some(()))
}

// T-101: cache_none_returns_not_installed
#[test]
fn cache_none_returns_not_installed() {
    let result = try_load_embedder_with_fns::<(), MockEmbedder, &str>(
        || Ok(None),
        |_| unreachable!("on_delete_error must not be called when cache is empty"),
        |_| unreachable!("on_probe_err must not be called when cache is empty"),
        |_| unreachable!("probe must not be called when cache is empty"),
        |_| unreachable!("new must not be called when cache is empty"),
        |_| unreachable!("delete must not be called when cache is empty"),
    );
    assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
}

// T-102: cache_err_returns_probe_failed
#[test]
fn cache_err_returns_probe_failed() {
    let result = try_load_embedder_with_fns::<(), MockEmbedder, _>(
        || Err::<Option<()>, _>("cache broken"),
        |_| {},
        |_| unreachable!("on_probe_err must not be called when cache_check fails"),
        |_| unreachable!("probe must not be called when cache_check fails"),
        |_| unreachable!("new must not be called when cache_check fails"),
        |_| unreachable!("delete must not be called when cache_check fails"),
    );
    assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
}

// T-103: backend_unavailable_returns_backend_unavailable
#[test]
fn backend_unavailable_returns_backend_unavailable() {
    let result = try_load_embedder_with_fns::<_, MockEmbedder, _>(
        cache_present(),
        |_| {},
        |_| unreachable!("on_probe_err must not be called on BackendUnavailable"),
        |_| Ok(ProbeStatus::BackendUnavailable),
        |_| unreachable!("new must not be called when backend unavailable"),
        |_| unreachable!("delete must not be called when backend unavailable"),
    );
    assert_eq!(result.err(), Some(DegradedReason::BackendUnavailable));
}

// T-104: corrupt_delete_ok_skips_on_delete_error
#[test]
fn corrupt_delete_ok_skips_on_delete_error() {
    let on_delete_error_called = Cell::new(false);
    let delete_called = Cell::new(false);
    let result = try_load_embedder_with_fns::<_, MockEmbedder, _>(
        cache_present(),
        |_| on_delete_error_called.set(true),
        |_| unreachable!("on_probe_err must not be called on ModelCorrupt"),
        |_| {
            Err(ModelInitError::ModelCorrupt {
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

// T-105: corrupt_delete_err_invokes_on_delete_error
#[test]
fn corrupt_delete_err_invokes_on_delete_error() {
    let captured: Cell<Option<String>> = Cell::new(None);
    let result = try_load_embedder_with_fns::<_, MockEmbedder, _>(
        cache_present(),
        |e| captured.set(Some(e.to_string())),
        |_| unreachable!("on_probe_err must not be called on ModelCorrupt"),
        |_| {
            Err(ModelInitError::ModelCorrupt {
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

// T-106: success_returns_arc_embed
#[test]
fn success_returns_arc_embed() {
    let result = try_load_embedder_with_fns(
        cache_present(),
        |_| {},
        |_| unreachable!("on_probe_err must not be called on success"),
        |_| Ok(ProbeStatus::Available),
        |_| Ok(MockEmbedder::default()),
        |_| unreachable!("delete must not be called on success"),
    );
    assert!(result.is_ok(), "loader should succeed");
}

// T-107: probe_backend_err_invokes_on_probe_err
#[test]
fn probe_backend_err_invokes_on_probe_err() {
    let captured: Cell<Option<String>> = Cell::new(None);
    let result = try_load_embedder_with_fns::<_, MockEmbedder, _>(
        cache_present(),
        |_| unreachable!("on_delete_error must not be called"),
        |e| captured.set(Some(e.to_string())),
        |_| {
            Err(ModelInitError::Backend {
                message: "spawn failed".into(),
                source: None,
            })
        },
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

// T-108: public_wrapper_absent_when_cache_empty
#[test]
fn public_wrapper_absent_when_cache_empty() {
    let result = try_load_embedder_with(
        || Ok::<Option<Artifacts>, &str>(None),
        |_| unreachable!("on_delete_error must not be called"),
        |_| unreachable!("on_err must not be called when cache is empty"),
    );
    assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
}

// T-115: new_fn_err_invokes_on_probe_err
#[test]
fn new_fn_err_invokes_on_probe_err() {
    let captured: Cell<Option<String>> = Cell::new(None);
    let result = try_load_embedder_with_fns::<_, MockEmbedder, _>(
        cache_present(),
        |_| {},
        |e| captured.set(Some(e.to_string())),
        |_| Ok(ProbeStatus::Available),
        |_| {
            Err(ModelInitError::Backend {
                message: "alloc failed".into(),
                source: None,
            })
        },
        |_| unreachable!("delete must not be called on new_fn failure"),
    );
    assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
    let msg = captured.into_inner().expect("on_probe_err should fire");
    assert!(
        msg.contains("alloc failed"),
        "on_probe_err message should carry detail, got {msg:?}"
    );
}

// T-116: new_fn_corrupt_deletes_artifacts
#[test]
fn new_fn_corrupt_deletes_artifacts() {
    let on_delete_error_called = Cell::new(false);
    let delete_called = Cell::new(false);
    let result = try_load_embedder_with_fns::<_, MockEmbedder, _>(
        cache_present(),
        |_| on_delete_error_called.set(true),
        |_| unreachable!("on_probe_err must not be called when new_fn reports ModelCorrupt"),
        |_| Ok(ProbeStatus::Available),
        |_| {
            Err(ModelInitError::ModelCorrupt {
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
}

// T-117: default_logging_preset_returns_arc_embed
#[test]
fn default_logging_preset_returns_arc_embed() {
    let result = try_load_embedder_default_logging_with_fns(
        cache_present(),
        |_| Ok(ProbeStatus::Available),
        |_| Ok(MockEmbedder::default()),
        |_| unreachable!("delete must not be called on success"),
    );
    assert!(result.is_ok(), "default-logging preset should succeed");
}
