use std::io::{IsTerminal, Write, stderr};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const TICK_MS: u64 = 80;

pub struct Spinner {
    done: Arc<AtomicBool>,
    message: Arc<Mutex<String>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    /// Creates a spinner, auto-detecting whether stderr is a TTY.
    ///
    /// When stderr is a terminal: starts a background thread that renders an animated
    /// spinner frame on each tick. When not a terminal: creates a no-op spinner that
    /// prints nothing until [`finish`](Self::finish) is called.
    pub fn new(msg: &str) -> Self {
        Self::new_with_tty(msg, stderr().is_terminal())
    }

    /// Creates a spinner with an explicit TTY decision — used in tests to exercise both paths.
    pub(super) fn new_with_tty(msg: &str, is_tty: bool) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let message = Arc::new(Mutex::new(msg.to_owned()));

        let thread = if is_tty {
            let done = Arc::clone(&done);
            let message = Arc::clone(&message);
            Some(thread::spawn(move || {
                let mut err = stderr();
                let mut i = 0;
                loop {
                    if done.load(Ordering::Relaxed) {
                        break;
                    }
                    let msg = message.lock().map(|m| m.clone()).unwrap_or_default();
                    let _ = write!(err, "\r\x1b[2K{} {}", FRAMES[i % FRAMES.len()], msg);
                    let _ = err.flush();
                    thread::sleep(Duration::from_millis(TICK_MS));
                    i += 1;
                }
            }))
        } else {
            None
        };

        Self {
            done,
            message,
            thread,
        }
    }

    /// Updates the message shown next to the spinner frame.
    pub fn set_message(&self, msg: &str) {
        if let Ok(mut m) = self.message.lock() {
            *m = msg.to_owned();
        }
    }

    /// Clears the spinner line, then prints a success marker line via [`done`].
    ///
    /// The marker is shown on both TTY and non-TTY streams so downstream log parsers
    /// see a consistent `✓ {msg}` format regardless of terminal detection.
    pub fn finish(self, msg: &str) {
        self.finish_with_detail(msg, None);
    }

    /// Finishes with the primary success line, then an optional indented detail line.
    ///
    /// `detail`, when `Some`, is printed on the following line prefixed with two
    /// spaces so it reads as a continuation of the success marker. Use for
    /// non-fatal side notes such as "skipped N items" or partial-failure summaries.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use amici::cli::Spinner;
    /// let sp = Spinner::new("Indexing...");
    /// sp.finish_with_detail(
    ///     "Indexed 100 sessions",
    ///     Some("Failed to parse 3 files — permission denied"),
    /// );
    /// ```
    pub fn finish_with_detail(self, main: &str, detail: Option<&str>) {
        // Drop first so the frame-thread clears the spinner line before `done` writes `✓`.
        drop(self);
        done(main);
        if let Some(d) = detail {
            eprintln!("  {d}");
        }
    }

    /// Stops the spinner silently by consuming it, triggering `Drop`.
    pub fn cancel(self) {}
}

/// Prints a `✓ {msg}` success line to stderr without running a spinner.
///
/// Use for standalone completion markers — for example, "nothing to do" branches
/// that skip the spinner entirely. Paired with [`Spinner::finish`] so both paths
/// produce identical output.
pub fn done(msg: &str) {
    eprintln!("\x1b[32m✓\x1b[0m {msg}");
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
            eprint!("\r\x1b[2K");
            let _ = stderr().flush();
        }
    }
}

/// Runs model load then embedding under two spinners, short-circuiting when `pending == 0`.
///
/// - `Ok(None)` — nothing to embed (pending was zero)
/// - `Ok(Some(result))` — embedding completed successfully
/// - `Err` — model load or embedding failed
pub fn embed_with_spinners<M, R, E>(
    pending: u32,
    load_model: impl FnOnce(&dyn Fn(&str)) -> Result<M, E>,
    finish_msg: impl FnOnce(&R) -> String,
    run_embed: impl FnOnce(M, &dyn Fn(&str)) -> Result<R, E>,
) -> Result<Option<R>, E> {
    if pending == 0 {
        return Ok(None);
    }
    let model = with_spinner("Loading model...", |_| "Model ready".to_owned(), load_model)?;
    let result = with_spinner(
        &format!("Embedding... 0/{pending} chunks"),
        finish_msg,
        |update| run_embed(model, update),
    )?;
    Ok(Some(result))
}

/// Runs `work` under a spinner, finishing or cancelling based on the result.
///
/// `work` receives a message-updater closure it can call to show progress.
/// On success the spinner is finished with the message returned by `finish_msg`.
/// On error the spinner is cancelled and the error is propagated.
pub fn with_spinner<T, E>(
    start: &str,
    finish_msg: impl FnOnce(&T) -> String,
    work: impl FnOnce(&dyn Fn(&str)) -> Result<T, E>,
) -> Result<T, E> {
    let spinner = Spinner::new(start);
    let result = {
        let update = |msg: &str| spinner.set_message(msg);
        work(&update)
    };
    match result {
        Ok(val) => {
            let msg = finish_msg(&val);
            spinner.finish(&msg);
            Ok(val)
        }
        Err(e) => {
            spinner.cancel();
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
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

    // T-031: set_message_updates_message
    #[test]
    fn set_message_updates_message() {
        let spinner = Spinner::new("initial");
        spinner.set_message("updated");
        let msg = spinner.message.lock().unwrap().clone();
        assert_eq!(msg, "updated");
        spinner.cancel();
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
}
