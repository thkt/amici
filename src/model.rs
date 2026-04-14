pub mod embedder;
pub mod reranker;

/// Reason a model could not be loaded.
///
/// `Disabled` is reserved for caller-level opt-out (e.g. an environment variable);
/// the loader functions never produce it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedReason {
    Disabled,
    NotInstalled,
    BackendUnavailable,
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

/// Returns a short user-facing note for a degraded embedder, or `None` if no
/// message should be shown (e.g. the caller explicitly disabled embedding).
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

#[derive(Default)]
pub enum ModelLoad<T> {
    Ready(T),
    #[default]
    Absent,
    Failed(String),
}

impl<T> ModelLoad<T> {
    pub fn as_ref(&self) -> Option<&T> {
        match self {
            Self::Ready(v) => Some(v),
            _ => None,
        }
    }

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
