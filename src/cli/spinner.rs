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
    /// spinner frame on each tick. When not a terminal: prints `msg` to stderr immediately
    /// and skips the animation.
    pub fn new(msg: &str) -> Self {
        Self::new_with_tty(msg, stderr().is_terminal())
    }

    /// Creates a spinner with an explicit TTY decision — used in tests to exercise both paths.
    pub(crate) fn new_with_tty(msg: &str, is_tty: bool) -> Self {
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
            eprintln!("{msg}");
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

    /// Clears the spinner line, then prints a success message.
    pub fn finish(self, msg: &str) {
        let is_tty = self.thread.is_some();
        drop(self);
        if is_tty {
            eprintln!("\x1b[32m✓\x1b[0m {msg}");
        } else {
            eprintln!("{msg}");
        }
    }

    /// Stops the spinner silently by consuming it, triggering `Drop`.
    pub fn cancel(self) {}
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

#[cfg(test)]
mod tests {
    use super::*;

    // T-007: cancel() が Drop をトリガーし done フラグが true になる
    #[test]
    fn cancel_signals_done() {
        let spinner = Spinner::new("loading...");
        let done = Arc::clone(&spinner.done);
        spinner.cancel();
        assert!(done.load(Ordering::Relaxed));
    }

    // T-008: set_message() でメッセージが更新される
    #[test]
    fn set_message_updates_message() {
        let spinner = Spinner::new("initial");
        spinner.set_message("updated");
        let msg = spinner.message.lock().unwrap().clone();
        assert_eq!(msg, "updated");
        spinner.cancel();
    }

    // T-009: 非TTY 環境で finish() がパニックしない
    #[test]
    fn finish_non_tty_does_not_panic() {
        let spinner = Spinner::new("loading...");
        spinner.finish("done");
    }

    // T-010: new_with_tty(false) → thread is None (non-TTY path)
    #[test]
    fn new_with_tty_false_has_no_thread() {
        let spinner = Spinner::new_with_tty("loading...", false);
        assert!(
            spinner.thread.is_none(),
            "non-TTY spinner must not spawn a thread"
        );
        spinner.cancel();
    }

    // T-011: new_with_tty(true) → thread is Some (TTY path)
    #[test]
    fn new_with_tty_true_has_thread() {
        let spinner = Spinner::new_with_tty("loading...", true);
        assert!(
            spinner.thread.is_some(),
            "TTY spinner must spawn a background thread"
        );
        spinner.cancel();
    }
}
