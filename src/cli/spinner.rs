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
mod tests;
