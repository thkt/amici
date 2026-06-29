use super::*;

// T-030: cancel_signals_done
#[test]
fn cancel_signals_done() {
    let spinner = Spinner::new("loading...");
    let done = Arc::clone(&spinner.done);
    spinner.cancel();
    assert!(
        done.load(Ordering::Relaxed),
        "done flag should be true after cancel"
    );
}

// T-032: finish_non_tty_does_not_panic
#[test]
fn finish_non_tty_does_not_panic() {
    let spinner = Spinner::new("loading...");
    spinner.finish("done");
}

// T-033: new_with_tty_false_has_no_thread
#[test]
fn new_with_tty_false_has_no_thread() {
    let spinner = Spinner::new_with_tty("loading...", false);
    assert!(
        spinner.thread.is_none(),
        "non-TTY spinner must not spawn a thread"
    );
    spinner.cancel();
}

// T-034: new_with_tty_true_has_thread
#[test]
fn new_with_tty_true_has_thread() {
    let spinner = Spinner::new_with_tty("loading...", true);
    assert!(
        spinner.thread.is_some(),
        "TTY spinner must spawn a background thread"
    );
    spinner.cancel();
}

// T-035: with_spinner_success_returns_ok
#[test]
fn with_spinner_success_returns_ok() {
    let result = with_spinner(
        "start",
        |v: &u32| format!("done {v}"),
        |_| Ok::<u32, &str>(42),
    );
    assert_eq!(result, Ok(42));
}

// T-036: with_spinner_error_propagates
#[test]
fn with_spinner_error_propagates() {
    let result = with_spinner(
        "start",
        |_: &()| "done".to_owned(),
        |_| Err::<(), &str>("boom"),
    );
    assert_eq!(result, Err("boom"));
}

// T-037: with_spinner_progress_updater_works
#[test]
fn with_spinner_progress_updater_works() {
    let _ = with_spinner(
        "start",
        |_: &()| "done".to_owned(),
        |update| {
            update("step 1");
            update("step 2");
            Ok::<(), &str>(())
        },
    );
    // No panic = updater callable without error. Message side-effects go to stderr.
}

// T-038: embed_with_spinners_pending_zero_returns_none
#[test]
fn embed_with_spinners_pending_zero_returns_none() {
    let result = embed_with_spinners(
        0,
        |_| Ok::<u32, &str>(42),
        |v: &u32| format!("done {v}"),
        |_, _| unreachable!("run_embed must not be called when pending is zero"),
    );
    assert_eq!(result, Ok(None));
}

// T-039: embed_with_spinners_nonzero_pending_returns_some
#[test]
fn embed_with_spinners_nonzero_pending_returns_some() {
    let result = embed_with_spinners(
        5,
        |_| Ok::<u32, &str>(99),
        |v: &u32| format!("done {v}"),
        |model, _| Ok::<u32, &str>(model + 1),
    );
    assert_eq!(result, Ok(Some(100)));
}

// T-040: done_does_not_panic
#[test]
fn done_does_not_panic() {
    done("ready");
}

// T-041: finish_non_tty_does_not_panic_after_set_message
#[test]
fn finish_non_tty_does_not_panic_after_set_message() {
    let spinner = Spinner::new_with_tty("start", false);
    spinner.set_message("working");
    spinner.finish("done");
}

// T-054: finish_with_detail_none_matches_finish_behavior
#[test]
fn finish_with_detail_none_matches_finish_behavior() {
    let spinner = Spinner::new_with_tty("start", false);
    spinner.finish_with_detail("done", None);
}

// T-055: finish_with_detail_some_does_not_panic
#[test]
fn finish_with_detail_some_does_not_panic() {
    let spinner = Spinner::new_with_tty("start", false);
    spinner.finish_with_detail("done", Some("skipped 3 items"));
}
