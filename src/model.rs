pub mod embedder;
pub mod reranker;

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

pub fn try_load_model_with<A, M, E>(
    cache_check: impl FnOnce() -> Result<Option<A>, E>,
    probe_fn: impl FnOnce(&A) -> Result<(), E>,
    new_fn: impl FnOnce(&A) -> Result<M, E>,
) -> ModelLoad<M>
where
    E: std::fmt::Display,
{
    let a = match cache_check() {
        Ok(Some(a)) => a,
        Ok(None) => return ModelLoad::Absent,
        Err(e) => return ModelLoad::Failed(e.to_string()),
    };
    if let Err(e) = probe_fn(&a) {
        return ModelLoad::Failed(e.to_string());
    }
    match new_fn(&a) {
        Ok(m) => ModelLoad::Ready(m),
        Err(e) => ModelLoad::Failed(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-001: cache Ok(Some) + probe Ok + new Ok → Ready
    #[test]
    fn all_succeed_returns_ready() {
        let result = try_load_model_with(
            || Ok::<Option<i32>, String>(Some(1)),
            |_| Ok(()),
            |_| Ok("model"),
        );
        assert!(matches!(result, ModelLoad::Ready("model")));
    }

    // T-002: cache Ok(None) → Absent、probe は呼ばれない
    #[test]
    fn cache_none_returns_absent() {
        let probe_called = std::cell::Cell::new(false);
        let result = try_load_model_with(
            || Ok::<Option<i32>, String>(None),
            |_| {
                probe_called.set(true);
                Ok(())
            },
            |_| Ok("model"),
        );
        assert!(matches!(result, ModelLoad::Absent));
        assert!(!probe_called.get());
    }

    // T-003: cache Err → Failed、probe は呼ばれない
    #[test]
    fn cache_err_returns_failed() {
        let result = try_load_model_with(
            || Err::<Option<i32>, &str>("cache error"),
            |_| Ok(()),
            |_| Ok("model"),
        );
        assert!(matches!(result, ModelLoad::Failed(ref m) if m == "cache error"));
    }

    // T-004: probe Err → Failed、new は呼ばれない
    #[test]
    fn probe_err_returns_failed() {
        let new_called = std::cell::Cell::new(false);
        let result = try_load_model_with(
            || Ok::<Option<i32>, &str>(Some(1)),
            |_| Err("probe error"),
            |_| {
                new_called.set(true);
                Ok("model")
            },
        );
        assert!(matches!(result, ModelLoad::Failed(ref m) if m == "probe error"));
        assert!(!new_called.get());
    }

    // T-005: new Err → Failed
    #[test]
    fn new_err_returns_failed() {
        let result: ModelLoad<i32> = try_load_model_with(
            || Ok::<Option<i32>, &str>(Some(1)),
            |_| Ok(()),
            |_| Err("new error"),
        );
        assert!(matches!(result, ModelLoad::Failed(ref m) if m == "new error"));
    }

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
