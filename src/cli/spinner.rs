use std::io::{IsTerminal, Write};
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
    pub fn new(msg: &str) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let message = Arc::new(Mutex::new(msg.to_string()));

        let thread = if std::io::stderr().is_terminal() {
            let done = Arc::clone(&done);
            let message = Arc::clone(&message);
            Some(thread::spawn(move || {
                let mut err = std::io::stderr();
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

        Self { done, message, thread }
    }

    pub fn set_message(&self, msg: &str) {
        if let Ok(mut m) = self.message.lock() {
            *m = msg.to_string();
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

    /// Stops the spinner silently.
    pub fn cancel(self) {}

    /// Stops the spinner and clears the spinner line.
    /// Delegates to `Drop::drop` which signals the thread and joins it.
    pub fn stop(self) {}
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-007: stop() 後にバックグラウンドスレッドが停止し done フラグが true になる
    #[test]
    fn stop_sets_done_flag() {
        let spinner = Spinner::new("loading...");
        let done = Arc::clone(&spinner.done);
        spinner.stop();
        // done=true は「停止済み」を意味する（running フラグの否定）
        assert!(done.load(Ordering::Relaxed));
    }
}
