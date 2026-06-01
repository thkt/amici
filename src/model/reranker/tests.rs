use std::cell::Cell;

use rurico::reranker::{Artifacts, MockReranker};

use super::*;

fn cache_present() -> impl FnOnce() -> Result<Option<()>, &'static str> {
    || Ok(Some(()))
}

// T-040: cache_none_returns_not_installed
#[test]
fn cache_none_returns_not_installed() {
    let result = try_load_reranker_with(
        || Ok::<_, &str>(None),
        |_| unreachable!("on_delete_error must not be called"),
        |_| unreachable!("on_probe_err must not be called"),
    );
    assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
}

// T-041: cache_err_returns_probe_failed
#[test]
fn cache_err_returns_probe_failed() {
    let result = try_load_reranker_with(
        || Err::<Option<Artifacts>, _>("cache broken"),
        |_| unreachable!("on_delete_error must not be called on cache error"),
        |_| unreachable!("on_probe_err must not be called on cache error"),
    );
    assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
}

// T-042: probe_available_new_ok_returns_ready
#[test]
fn probe_available_new_ok_returns_ready() {
    let result = try_load_reranker_with_fns(
        cache_present(),
        |_| unreachable!("on_delete_error must not be called on success"),
        |_| unreachable!("on_probe_err must not be called on success"),
        |_| Ok(ProbeStatus::Available),
        |_| Ok(MockReranker::default()),
        |_| unreachable!("delete must not be called on success"),
    );
    assert!(result.is_ok());
}

// T-043: probe_backend_unavailable_returns_backend_unavailable
#[test]
fn probe_backend_unavailable_returns_backend_unavailable() {
    let result = try_load_reranker_with_fns::<_, _, MockReranker>(
        cache_present(),
        |_| unreachable!("on_delete_error must not be called on BackendUnavailable"),
        |_| unreachable!("on_probe_err must not be called on BackendUnavailable"),
        |_| Ok(ProbeStatus::BackendUnavailable),
        |_| unreachable!("new must not be called when backend unavailable"),
        |_| unreachable!("delete must not be called when backend unavailable"),
    );
    assert_eq!(result.err(), Some(DegradedReason::BackendUnavailable));
}

// T-044: probe_err_invokes_on_probe_err
#[test]
fn probe_err_invokes_on_probe_err() {
    let captured: Cell<Option<String>> = Cell::new(None);
    let result = try_load_reranker_with_fns::<_, _, MockReranker>(
        cache_present(),
        |_| unreachable!("on_delete_error must not be called on non-corrupt probe error"),
        |e| captured.set(Some(e.to_string())),
        |_| {
            Err(ModelInitError::Backend {
                message: "probe failed".into(),
                source: None,
            })
        },
        |_| unreachable!("new must not be called when probe fails"),
        |_| unreachable!("delete must not be called on non-corrupt probe error"),
    );
    assert_eq!(result.err(), Some(DegradedReason::ProbeFailed));
    let msg = captured.into_inner().expect("on_probe_err should fire");
    assert!(
        msg.contains("probe failed"),
        "on_probe_err message should carry detail, got {msg:?}"
    );
}

// T-045: new_err_invokes_on_probe_err
#[test]
fn new_err_invokes_on_probe_err() {
    let captured: Cell<Option<String>> = Cell::new(None);
    let result = try_load_reranker_with_fns::<_, _, MockReranker>(
        cache_present(),
        |_| unreachable!("on_delete_error must not be called when probe succeeds"),
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

// T-046: corrupt_delete_ok_skips_on_delete_error
#[test]
fn corrupt_delete_ok_skips_on_delete_error() {
    let on_delete_error_called = Cell::new(false);
    let on_probe_err_called = Cell::new(false);
    let delete_called = Cell::new(false);
    let result = try_load_reranker_with_fns::<_, _, MockReranker>(
        cache_present(),
        |_| on_delete_error_called.set(true),
        |_| on_probe_err_called.set(true),
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
    assert!(
        !on_probe_err_called.get(),
        "on_probe_err must not be called on ModelCorrupt"
    );
}

// T-047: corrupt_delete_err_invokes_on_delete_error
#[test]
fn corrupt_delete_err_invokes_on_delete_error() {
    let captured: Cell<Option<String>> = Cell::new(None);
    let on_probe_err_called = Cell::new(false);
    let result = try_load_reranker_with_fns::<_, _, MockReranker>(
        cache_present(),
        |e| captured.set(Some(e.to_string())),
        |_| on_probe_err_called.set(true),
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
    assert!(
        !on_probe_err_called.get(),
        "on_probe_err must not be called on ModelCorrupt"
    );
}

// T-048: public_wrapper_absent_when_cache_empty
#[test]
fn public_wrapper_absent_when_cache_empty() {
    let result = try_load_reranker_with(
        || Ok::<Option<Artifacts>, &str>(None),
        |_| unreachable!("on_delete_error must not be called"),
        |_| unreachable!("on_probe_err must not be called when cache is empty"),
    );
    assert_eq!(result.err(), Some(DegradedReason::NotInstalled));
}

// T-049: new_fn_corrupt_deletes_artifacts
#[test]
fn new_fn_corrupt_deletes_artifacts() {
    let on_delete_error_called = Cell::new(false);
    let on_probe_err_called = Cell::new(false);
    let delete_called = Cell::new(false);
    let result = try_load_reranker_with_fns::<_, _, MockReranker>(
        cache_present(),
        |_| on_delete_error_called.set(true),
        |_| on_probe_err_called.set(true),
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
    assert!(
        !on_probe_err_called.get(),
        "on_probe_err must not be called on ModelCorrupt"
    );
}
