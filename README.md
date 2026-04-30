# amici

Shared model-loading, storage helpers, and CLI utilities for the sae/yomu/recall toolchain.

## Modules

| Module | Contents |
| ------ | -------- |
| `model` | `DegradedReason`, `degraded_reason_user_note`, `ModelLoad<T>`, `ModelDownloadError`, `download_and_verify_model` |
| `model::embedder` | `try_load_embedder_with` — loads the embedding model |
| `model::reranker` | `try_load_reranker_with` — loads the reranking model |
| `storage::filter` | `in_placeholders`, `anon_placeholders`, `as_sql_params`, `append_eq_filter`, … — SQL `WHERE` clause and parameter builders |
| `storage::query_helpers` | `collect_rows`, `fetch_by_in_clause` — `Connection`-bound row collectors and IN-clause bulk fetch (generic over collection and error type) |
| `cli` | `Spinner`, `with_spinner`, `try_expand_shorthand` |
| `migration` | `notify_schema_change` — unified `tracing::warn!` for schema-clear notices |
| `logging` | `init_subscriber` — `RUST_LOG`-aware `tracing_subscriber::fmt` setup for CLI `main.rs` |

## Usage

```toml
[dependencies]
amici = { git = "https://github.com/thkt/amici", rev = "<rev>" }
```

## Development

### Setup

Run once after cloning:

```sh
git config --local core.hooksPath .githooks
```

This installs a pre-commit hook that runs `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings` before each commit. Violations abort the commit. To skip for one commit: `git commit --no-verify`.

### Common commands

```sh
just check                                                # test + lint + fmt-check
cargo test --all-features                                 # tests only
cargo clippy --all-targets --all-features -- -D warnings  # lint (matches CI)
cargo fmt -- --check                                      # format check
```
