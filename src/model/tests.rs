use std::cell::Cell;
use std::io;
use std::mem;
use std::sync::{Arc, Mutex};

use rurico::embed::MockEmbedder;
use tracing::subscriber::with_default;
use tracing_subscriber::fmt::MakeWriter;

use super::*;

#[derive(Clone, Default)]
struct CapturedWriter(Arc<Mutex<Vec<u8>>>);

impl CapturedWriter {
    fn captured(&self) -> String {
        let bytes = mem::take(&mut *self.0.lock().expect("captured buffer poisoned"));
        String::from_utf8(bytes).expect("captured bytes are not UTF-8")
    }
}

impl io::Write for CapturedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("captured buffer poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for CapturedWriter {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn capture_warn(emit: impl FnOnce()) -> String {
    let writer = CapturedWriter::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer.clone())
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .finish();
    with_default(subscriber, emit);
    writer.captured()
}

// T-019: as_ref_returns_inner_only_for_ready
#[test]
fn as_ref_returns_inner_only_for_ready() {
    assert_eq!(ModelLoad::Ready(42).as_ref(), Some(&42));
    assert_eq!(ModelLoad::<i32>::Absent.as_ref(), None);
    assert_eq!(ModelLoad::<i32>::Failed("e".into()).as_ref(), None);
}

// T-020: debug_output_contains_variant_name
#[test]
fn debug_output_contains_variant_name() {
    let ready = format!("{:?}", ModelLoad::Ready(1));
    assert!(ready.contains("Ready"), "expected 'Ready', got: {ready}");
    let absent = format!("{:?}", ModelLoad::<i32>::Absent);
    assert!(
        absent.contains("Absent"),
        "expected 'Absent', got: {absent}"
    );
    let failed = format!("{:?}", ModelLoad::<i32>::Failed("msg".into()));
    assert!(
        failed.contains("Failed"),
        "expected 'Failed', got: {failed}"
    );
}

// T-021: emit_load_hint_does_not_panic
#[test]
fn emit_load_hint_does_not_panic() {
    ModelLoad::<i32>::Absent.emit_load_hint("not found", "model");
    ModelLoad::Ready(1).emit_load_hint("not found", "model");
    ModelLoad::<i32>::Failed("err".into()).emit_load_hint("not found", "model");
}

// T-022: download_err_returns_download_failed
#[test]
fn download_err_returns_download_failed() {
    let complete_called = Cell::new(false);
    let result = try_download_and_verify_with_fns::<(), MockEmbedder, _>(
        || Err::<(), _>("network timeout".to_owned()),
        |_| unreachable!("on_delete_error must not fire on download failure"),
        |_| unreachable!("probe must not be called on download failure"),
        |_| unreachable!("new must not be called on download failure"),
        |_| unreachable!("delete must not be called on download failure"),
        || complete_called.set(true),
    );
    assert!(
        matches!(result, Err(ModelDownloadError::DownloadFailed(ref msg)) if msg.contains("network timeout")),
        "expected DownloadFailed with message, got {result:?}"
    );
    assert!(
        !complete_called.get(),
        "on_download_complete must not fire on download failure"
    );
}

// T-023: probe_backend_unavailable_returns_backend_unavailable
#[test]
fn probe_backend_unavailable_returns_backend_unavailable() {
    let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
        || Ok::<_, String>(()),
        |_| {},
        |_| Ok(ProbeStatus::BackendUnavailable),
        |_| unreachable!("new must not be called on BackendUnavailable"),
        |_| unreachable!("delete must not be called on BackendUnavailable"),
        || {},
    );
    assert!(
        matches!(result, Err(ModelDownloadError::BackendUnavailable)),
        "expected BackendUnavailable, got {result:?}"
    );
}

// T-024: probe_err_captured_in_probe_failed
#[test]
fn probe_err_captured_in_probe_failed() {
    let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
        || Ok::<_, String>(()),
        |_| unreachable!("on_delete_error must not fire on non-corrupt probe error"),
        |_| {
            Err(ModelInitError::Backend {
                message: "backend down".into(),
                source: None,
            })
        },
        |_| unreachable!("new must not be called when probe fails"),
        |_| unreachable!("delete must not be called on non-corrupt probe error"),
        || {},
    );
    match result {
        Err(ModelDownloadError::ProbeFailed(Some(detail))) => assert!(
            detail.contains("backend down"),
            "probe_detail should carry error message, got {detail:?}"
        ),
        other => panic!("expected ProbeFailed, got {other:?}"),
    }
}

// T-025: success_returns_ok
#[test]
fn success_returns_ok() {
    let complete_called = Cell::new(false);
    let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
        || Ok::<_, String>(()),
        |_| unreachable!("on_delete_error must not fire on success"),
        |_| Ok(ProbeStatus::Available),
        |_| Ok(MockEmbedder::default()),
        |_| unreachable!("delete must not be called on success"),
        || complete_called.set(true),
    );
    assert!(result.is_ok(), "expected Ok(()), got {result:?}");
    assert!(
        complete_called.get(),
        "on_download_complete must fire after successful download"
    );
}

// T-026: new_fn_err_captured_in_probe_failed
#[test]
fn new_fn_err_captured_in_probe_failed() {
    let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
        || Ok::<_, String>(()),
        |_| {},
        |_| Ok(ProbeStatus::Available),
        |_| {
            Err(ModelInitError::Backend {
                message: "alloc failed".into(),
                source: None,
            })
        },
        |_| unreachable!("delete must not be called on non-corrupt new_fn failure"),
        || {},
    );
    match result {
        Err(ModelDownloadError::ProbeFailed(Some(detail))) => assert!(
            detail.contains("alloc failed"),
            "probe_detail should carry new_fn error message, got {detail:?}"
        ),
        other => panic!("expected ProbeFailed, got {other:?}"),
    }
}

fn always_fails(msg: &'static str) -> Result<(), &'static str> {
    Err(msg)
}

// T-027: degrade_with_warn_returns_closure_yielding_reason
#[test]
fn degrade_with_warn_returns_closure_yielding_reason() {
    let to_reason = degrade_with_warn("ctx", DegradedReason::ProbeFailed);
    let result = always_fails("inner err").map_err(to_reason);
    assert_eq!(result, Err(DegradedReason::ProbeFailed));
}

// T-028: degrade_with_warn_emits_warn_with_error_and_context
#[test]
fn degrade_with_warn_emits_warn_with_error_and_context() {
    let captured = capture_warn(|| {
        let to_reason = degrade_with_warn("test ctx", DegradedReason::BackendUnavailable);
        let _ = always_fails("inner err msg").map_err(to_reason);
    });
    assert!(captured.contains("WARN"), "missing WARN level: {captured}");
    assert!(
        captured.contains("operation degraded"),
        "missing message: {captured}"
    );
    assert!(
        captured.contains("inner err msg"),
        "missing original error: {captured}"
    );
    assert!(
        captured.contains("BackendUnavailable"),
        "missing reason: {captured}"
    );
    assert!(captured.contains("test ctx"), "missing context: {captured}");
}

// T-029: record_degraded_emits_warn_with_context
#[test]
fn record_degraded_emits_warn_with_context() {
    let captured = capture_warn(|| {
        record_degraded(DegradedReason::ProbeFailed, "seed inference");
    });
    assert!(captured.contains("WARN"), "missing WARN level: {captured}");
    assert!(
        captured.contains("operation degraded"),
        "missing message: {captured}"
    );
    assert!(
        captured.contains("ProbeFailed"),
        "missing reason: {captured}"
    );
    assert!(
        captured.contains("seed inference"),
        "missing context: {captured}"
    );
}

// T-030: user_note_returns_none_for_disabled
#[test]
fn user_note_returns_none_for_disabled() {
    assert_eq!(
        EmbedderDegraded(DegradedReason::Disabled).user_note("any cmd"),
        None,
        "Disabled is caller opt-out; suppress the note"
    );
}

// T-031: user_note_injects_download_cmd_for_not_installed
#[test]
fn user_note_injects_download_cmd_for_not_installed() {
    let note = EmbedderDegraded(DegradedReason::NotInstalled)
        .user_note("yomu model download")
        .expect("NotInstalled must produce a note");
    assert!(
        note.contains("`yomu model download`"),
        "download_cmd must appear backtick-quoted, got: {note}"
    );
    assert!(
        note.contains("not installed"),
        "note must explain the cause, got: {note}"
    );
}

// T-032: user_note_ignores_download_cmd_for_backend_unavailable
#[test]
fn user_note_ignores_download_cmd_for_backend_unavailable() {
    let note = EmbedderDegraded(DegradedReason::BackendUnavailable)
        .user_note("recall model download")
        .expect("BackendUnavailable must produce a note");
    assert!(
        !note.contains("recall model download"),
        "download_cmd must not leak into BackendUnavailable note, got: {note}"
    );
    assert!(
        note.contains("text search only"),
        "note must explain the fallback, got: {note}"
    );
}

// T-033: user_note_ignores_download_cmd_for_probe_failed
#[test]
fn user_note_ignores_download_cmd_for_probe_failed() {
    let note = EmbedderDegraded(DegradedReason::ProbeFailed)
        .user_note("sae model download")
        .expect("ProbeFailed must produce a note");
    assert!(
        !note.contains("sae model download"),
        "download_cmd must not leak into ProbeFailed note, got: {note}"
    );
    assert!(
        note.contains("text search only"),
        "note must explain the fallback, got: {note}"
    );
}

// T-034: probe_failed_display_distinguishes_detail_presence
#[test]
fn probe_failed_display_distinguishes_detail_presence() {
    assert_eq!(
        ModelDownloadError::ProbeFailed(None).to_string(),
        "model probe failed; try again or re-download the model",
        "None must render the no-detail wording"
    );
    assert_eq!(
        ModelDownloadError::ProbeFailed(Some("hash mismatch".into())).to_string(),
        "model probe failed: hash mismatch; try again or re-download the model",
        "Some(detail) must embed the detail in the wording"
    );
}

// T-035: new_fn_corrupt_returns_probe_failed_none
#[test]
fn new_fn_corrupt_returns_probe_failed_none() {
    let delete_attempted = Cell::new(false);
    let delete_error_seen = Cell::new(false);
    let result = try_download_and_verify_with_fns::<_, MockEmbedder, String>(
        || Ok::<_, String>(()),
        |_| delete_error_seen.set(true),
        |_| Ok(ProbeStatus::Available),
        |_| {
            Err(ModelInitError::ModelCorrupt {
                reason: "weights corrupt".into(),
            })
        },
        |_| {
            delete_attempted.set(true);
            Err(io::Error::other("disk full"))
        },
        || {},
    );
    assert!(
        matches!(result, Err(ModelDownloadError::ProbeFailed(None))),
        "ModelCorrupt must map to ProbeFailed(None) (detail not captured), got {result:?}"
    );
    assert!(
        delete_attempted.get(),
        "ModelCorrupt must attempt artifact deletion"
    );
    assert!(
        delete_error_seen.get(),
        "delete failure must reach on_delete_error"
    );
}
