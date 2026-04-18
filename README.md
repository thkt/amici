# amici

Shared model-loading, storage helpers, and CLI utilities for the sae/yomu/recall toolchain.

## Modules

| Module | Contents |
| ------ | -------- |
| `model` | `DegradedReason`, `degraded_reason_user_note`, `ModelLoad<T>`, `ModelDownloadError`, `download_and_verify_model` |
| `model::embedder` | `try_load_embedder_with` — loads the embedding model |
| `model::reranker` | `try_load_reranker_with` — loads the reranking model |
| `storage` | `in_placeholders`, `anon_placeholders`, `as_sql_params`, `append_eq_filter` |
| `cli` | `Spinner`, `with_spinner`, `try_expand_shorthand` |
| `migration` | `notify_schema_change` — unified `tracing::warn!` for schema-clear notices |
| `logging` | `init_subscriber` — `RUST_LOG`-aware `tracing_subscriber::fmt` setup for CLI `main.rs` |

## Usage

```toml
[dependencies]
amici = { git = "https://github.com/thkt/amici", rev = "<rev>" }
```
