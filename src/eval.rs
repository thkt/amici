//! Search evaluation harness (ADR 0002).
//!
//! Composes `rurico` primitives — embedder, reranker, retrieval, storage —
//! with amici's own `crate::storage::fts::clean_for_trigram`, so the
//! baseline measures the production wiring `recall`/`sae`/`yomu` actually
//! run. Feature-gated behind `eval-harness`; default `amici` builds skip
//! the module.

pub mod baseline;
pub mod fixture;
pub mod metrics;
pub mod pipeline;
