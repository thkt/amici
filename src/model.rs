pub mod embedder;
pub mod reranker;

/// Reason a model could not be loaded.
///
/// `Disabled` is reserved for caller-level opt-out (e.g. an environment variable);
/// the loader functions never produce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedReason {
    /// Caller explicitly disabled the model (e.g. via an environment variable or config flag).
    /// The loader functions never return this variant; it is set by the caller.
    Disabled,
    /// Model artifacts are not present in the local cache. The user must download them first.
    NotInstalled,
    /// The hardware/OS backend (e.g. MLX) is not available on this machine.
    BackendUnavailable,
    /// A cache lookup error, model-file corruption, or model-init failure occurred.
    /// The artifacts may have been deleted automatically to allow re-download.
    ProbeFailed,
}

impl std::fmt::Display for DegradedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DegradedReason::Disabled => write!(f, "disabled"),
            DegradedReason::NotInstalled => write!(f, "not installed"),
            DegradedReason::BackendUnavailable => write!(f, "MLX backend unavailable"),
            DegradedReason::ProbeFailed => write!(f, "probe failed"),
        }
    }
}

/// Returns a short user-facing note for a degraded model (embedder or reranker),
/// or `None` if no message should be shown (e.g. the caller explicitly disabled the model).
pub fn degraded_reason_user_note(reason: DegradedReason) -> Option<&'static str> {
    match reason {
        DegradedReason::Disabled => None,
        DegradedReason::NotInstalled => {
            Some("embedding model not installed; results from text search only")
        }
        DegradedReason::BackendUnavailable | DegradedReason::ProbeFailed => {
            Some("embedding model unavailable; results from text search only")
        }
    }
}

/// Outcome of a model-load attempt.
///
/// Callers should inspect the variant to handle the `Failed` case — dropping a
/// `Failed` value silently discards the error message.
#[must_use = "inspect the variant to handle loading failures"]
#[derive(Default)]
pub enum ModelLoad<T> {
    /// The model loaded successfully and is ready to use.
    Ready(T),
    /// No model artifacts were found; the model was never installed.
    #[default]
    Absent,
    /// The model could not be loaded. The `String` contains a human-readable error message.
    Failed(String),
}

impl<T> ModelLoad<T> {
    /// Returns `Some(&T)` when the model is [`Ready`](ModelLoad::Ready), `None` otherwise.
    pub fn as_ref(&self) -> Option<&T> {
        match self {
            Self::Ready(v) => Some(v),
            _ => None,
        }
    }

    /// Prints a user-facing hint or warning to stderr when the model is not ready.
    ///
    /// - [`Absent`](ModelLoad::Absent): prints `"Hint: {absent_hint}"`
    /// - [`Failed`](ModelLoad::Failed): prints `"Warning: {model_label} not available ({error})"`
    /// - [`Ready`](ModelLoad::Ready): no-op
    pub fn emit_load_hint(&self, absent_hint: &str, model_label: &str) {
        match self {
            Self::Absent => eprintln!("Hint: {absent_hint}"),
            Self::Failed(e) => eprintln!("Warning: {model_label} not available ({e})"),
            Self::Ready(_) => {}
        }
    }
}

impl<T> std::fmt::Debug for ModelLoad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(_) => write!(f, "Ready(...)"),
            Self::Absent => write!(f, "Absent"),
            Self::Failed(msg) => write!(f, "Failed({msg:?})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-019: as_ref() — Ready→Some, Absent→None, Failed→None
    #[test]
    fn as_ref_returns_inner_only_for_ready() {
        assert_eq!(ModelLoad::Ready(42).as_ref(), Some(&42));
        assert_eq!(ModelLoad::<i32>::Absent.as_ref(), None);
        assert_eq!(ModelLoad::<i32>::Failed("e".into()).as_ref(), None);
    }

    // T-020: Debug 出力が各バリアントの文字列を含む
    #[test]
    fn debug_output_contains_variant_name() {
        assert!(format!("{:?}", ModelLoad::Ready(1)).contains("Ready"));
        assert!(format!("{:?}", ModelLoad::<i32>::Absent).contains("Absent"));
        assert!(format!("{:?}", ModelLoad::<i32>::Failed("msg".into())).contains("Failed"));
    }

    // T-021: emit_load_hint — Absent→hint出力、Failed→warning出力、Ready→no-op
    // (stderr の捕捉が困難なため smoke test のみ: パニックしないことを確認)
    #[test]
    fn emit_load_hint_does_not_panic() {
        ModelLoad::<i32>::Absent.emit_load_hint("not found", "model");
        ModelLoad::Ready(1).emit_load_hint("not found", "model");
        ModelLoad::<i32>::Failed("err".into()).emit_load_hint("not found", "model");
    }
}
