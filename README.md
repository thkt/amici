# amici

Shared model-loading, storage helpers, and CLI utilities for the sae/yomu/recall toolchain.

## Modules

| Module | Contents |
| ------ | -------- |
| `model` | `DegradedReason`, `degraded_reason_user_note`, `degrade_with_warn`, `record_degraded`, `ModelLoad<T>`, `ModelDownloadError`, `download_and_verify_model` |
| `model::embedder` | `try_load_embedder_with`, `try_load_embedder_default_logging` — loads the embedding model |
| `model::reranker` | `try_load_reranker_with` — loads the reranking model |
| `storage::filter` | `in_placeholders`, `anon_placeholders`, `as_sql_params`, `append_eq_filter`, … — SQL `WHERE` clause and parameter builders |
| `storage::query_helpers` | `collect_rows`, `fetch_by_in_clause` — `Connection`-bound row collectors and IN-clause bulk fetch (generic over collection and error type) |
| `storage::fts` | `clean_for_trigram` — adapts `rurico::storage::MatchFtsQuery` for an FTS5 `trigram` tokenizer |
| `cli` | `Spinner`, `with_spinner`, `embed_with_spinners`, `done`, `try_expand_shorthand`, `env_lookup`, `CliError`, `exit_code::codes`, `exit_error`, `hint_arrow`, `info`, `deprecation_warn`, `progress_step` |
| `migration` | `notify_schema_change` — unified `tracing::warn!` for schema-clear notices |
| `logging` | `init_subscriber` — `RUST_LOG`-aware `tracing_subscriber::fmt` setup for CLI `main.rs` |
| `eval` *(feature `eval-harness`)* | Search-evaluation harness composed with `rurico` primitives — backs the `eval_harness` binary (ADR 0002). |
| `testing::hybrid` *(feature `test-support`)* | FTS↔vec symmetry contract assertion for hybrid search. |

## Usage

```toml
[dependencies]
amici = { git = "https://github.com/thkt/amici", rev = "<rev>" }
```

## Features

| Feature | Effect |
| ------- | ------ |
| `eval-harness` | Compiles `amici::eval` and the `eval_harness` binary. Required by `just eval-*` recipes. |
| `test-support` | Exposes `amici::testing::hybrid` so downstream `[dev-dependencies]` can reuse the hybrid-search contract helper. |

## CLI conventions

### Dependency injection for env-var lookup

CLIs that read environment variables in a constructor cannot be tested
deterministically — `for_test` helpers silently bypass the env path, so the
code that runs in production never sees a unit test. Split the constructor
into `from_env()` and `from_env_with(impl Fn(&str) -> Option<String>)`, and
use `amici::cli::env_lookup()` as the production lookup.

**Before** (yomu `tools.rs`):

```rust
impl Yomu {
    pub fn with_root(root: PathBuf, options: YomuOptions) -> Result<Self, YomuError> {
        // ...
        let embed_disabled = options.no_embed || env::var("YOMU_EMBED").as_deref() == Ok("0");
        let rerank_enabled = env::var("YOMU_RERANK").as_deref() == Ok("1");
        // ...
    }
}
```

**After**:

```rust
use amici::cli::env_lookup;

pub struct YomuConfig {
    pub embed_disabled: bool,
    pub rerank_enabled: bool,
}

impl YomuConfig {
    pub fn from_env() -> Self {
        Self::from_env_with(env_lookup())
    }
    pub fn from_env_with<F: Fn(&str) -> Option<String>>(get: F) -> Self {
        Self {
            embed_disabled: get("YOMU_EMBED").as_deref() == Some("0"),
            rerank_enabled: get("YOMU_RERANK").as_deref() == Some("1"),
        }
    }
}

impl Yomu {
    pub fn with_root(root: PathBuf, options: YomuOptions, config: YomuConfig) -> Result<Self, YomuError> {
        let embed_disabled = options.no_embed || config.embed_disabled;
        // ...
    }
}
```

Tests then construct `YomuConfig` directly without touching process env:

```rust
let cfg = YomuConfig::from_env_with(|k| match k {
    "YOMU_EMBED" => Some("0".into()),
    _ => None,
});
```

`env::var_os` (path resolution returning `OsString`) is out of scope — keep
those as a separate factory if your CLI mixes both.

`Option<String>` is the canonical signature for amici. Any pre-existing
`from_env_with(impl Fn(&str) -> Result<String, VarError>)` site (sae has
two: `EsaClient::from_env_with`, `data_dir_with`) is expected to be
migrated to `Option<String>` during downstream rollout — `get(k).ok()`
is the adapter when an existing site cannot move yet.

### Degraded path notification

When a typed error is collapsed into a `DegradedReason` (embedder/reranker
unavailable, probe failed, …), emit a `tracing::warn!` event so the original
error and the degraded reason are both observable in logs. Silent collapse
(`map_err(|_| DegradedReason::ProbeFailed)`) and ad-hoc inline `warn!` calls
hide regressions and drift across crates.

Use one of the two helpers in `amici::model`:

- `degrade_with_warn(context, reason)` — returns a `map_err` closure. Use when
  the call site has a typed error that needs to be collapsed.
- `record_degraded(reason, context)` — direct emit. Use when the call site
  already holds a `DegradedReason` value (no original error to preserve).

Both emit a structured warn event with the message `"operation degraded"` and
the fields `reason`, `context`, and (for `degrade_with_warn`) `error`.

**Before** (silent collapse):

```rust
let task_emb = embedder.embed_query(task)
    .map_err(|_| DegradedReason::ProbeFailed)?;
```

**After**:

```rust
use amici::model::{degrade_with_warn, DegradedReason};

let task_emb = embedder.embed_query(task)
    .map_err(degrade_with_warn(
        "brief seed inference: embed_query",
        DegradedReason::ProbeFailed,
    ))?;
```

**Before** (inline warn that drifts in wording across call sites):

```rust
match self.infer_seed_paths(/* ... */) {
    Ok(paths) => { /* ... */ }
    Err(reason) => {
        eprintln!("warning: seed inference degraded ({reason})");
        degraded = true;
    }
}
```

**After**:

```rust
use amici::model::record_degraded;

match self.infer_seed_paths(/* ... */) {
    Ok(paths) => { /* ... */ }
    Err(reason) => {
        record_degraded(reason, "brief: seed inference");
        degraded = true;
    }
}
```

The two helpers compose with `notify_schema_change` and
`try_load_embedder_with` — pick the helper whose input shape matches the call
site instead of mixing notification policies.

### Exit code convention

Implement `amici::cli::CliError` on the crate-level error enum so `main.rs`
can call `e.exit_code()` directly instead of maintaining a hand-written
`exit_code_for(&e)` helper. Distinct codes per variant let LLMs and shell
scripts branch on retry policy.

The `amici::cli::exit_code::codes` module exposes the
[sysexits.h](https://man.openbsd.org/sysexits.3) convention as `u8` constants
(see the module doc for why `u8` rather than `ExitCode`). These are
**recommended, not required** — pick any schema that gives distinct codes
per variant.

**Before** (yomu `main.rs`):

```rust
fn exit_code_for(e: &YomuError) -> ExitCode {
    match e {
        YomuError::InvalidInput(_) => ExitCode::from(2),
        YomuError::Internal(_) => ExitCode::from(4),
        YomuError::Storage(_)
        | YomuError::Io(_)
        | YomuError::Index(_)
        | YomuError::Query(_)
        | YomuError::EmbedderUnavailable(_) => ExitCode::FAILURE, // collapsed
    }
}
```

**After**:

```rust
use amici::cli::CliError;
use amici::cli::exit_code::codes;
use std::process::ExitCode;

impl CliError for YomuError {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::InvalidInput(_)        => ExitCode::from(codes::USAGE),
            Self::Internal(_)            => ExitCode::from(codes::SOFTWARE),
            Self::Storage(_)             => ExitCode::from(codes::CANT_CREAT),
            Self::Io(_)                  => ExitCode::from(codes::IO_ERR),
            Self::Index(_)               => ExitCode::from(codes::CANT_CREAT),
            Self::Query(_)               => ExitCode::from(codes::SOFTWARE),
            Self::EmbedderUnavailable(_) => ExitCode::from(codes::TEMP_FAIL),
        }
    }
}

// main.rs
return e.exit_code();
```

Document the chosen mapping in each CLI's README or `--help` output so callers
can write deterministic retry logic.

> **Migration note**: switching from a CLI's existing schema (e.g. yomu's
> `2` / `4` / `FAILURE`) to sysexits constants changes the **numeric exit
> codes** observed by callers (e.g. `2` → `64`, `4` → `70`). Shell scripts,
> CI pipelines, and parent processes that branch on a specific number must
> be updated. Announce the change in the CLI's release notes.

#### Group 2 baseline (ADR-0066)

ADR-0066 partitions the CLI suite into three groups by error topology and
assigns each group a shared exit-code baseline. amici is the baseline for
**Group 2 — local semantic search** (sae / yomu / recall / rurico).
Downstream CLIs alias `amici::cli::exit_code::codes` so classification stays
consistent across the group, and a metrics dashboard tracking the rate of
`UNKNOWN` (104) can detect `anyhow::Error` swallowing or unclassified
failures from a single value regardless of which Group 2 CLI emitted it.

The `codes` module exposes two source ranges side-by-side:

| Range  | Source                             | Codes                                                                  |
| ------ | ---------------------------------- | ---------------------------------------------------------------------- |
| 64–78  | sysexits.h                         | `USAGE`, `DATA_ERROR`, `SOFTWARE`, `CANT_CREAT`, `IO_ERR`, `TEMP_FAIL` |
| 80–119 | Project extension range (ADR-0066) | `UNKNOWN` (104)                                                        |

`codes::INTERNAL` is provided as an alias for `SOFTWARE` so call sites can use
the name that ADR-0066's classification table uses — the numeric value is
unchanged (70 = `EX_SOFTWARE`).

### Oracle mode

> Requires the `eval-harness` Cargo feature.

The eval harness ships an Oracle pipeline that measures how much recall
the production retrieval is leaving on the table. It works by forcing
every known-relevant document to rank 0 inside the merge stage, then
running the rest of the pipeline (aggregation, reranker) unchanged. The
gap between the normal "forward" baseline and this idealised oracle
baseline is the search-side improvement headroom.

```sh
# 1. Capture the oracle baseline (MLX required, Apple Silicon only).
just eval-oracle

# 2. Compare it with the forward baseline. Read-only, no MLX.
just eval-oracle-gap
```

`eval-oracle-gap` emits a markdown report with global and per-category
gap tables. It exits `1` if any category shows `oracle.recall@k <
baseline.recall@k` (the oracle path failed to land the relevant doc in
`top-k` — almost always a wiring regression). Otherwise it exits `0`.

How to read the result: a large positive `recall@k` gap means
retrieval-side investment (more rerank candidates, query rewriting,
a learned ranker) has measurable headroom. A near-zero gap means the
retrieval is already at its ceiling for this fixture, and the next
quality gain has to come from elsewhere — e.g. extending the fixture.

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
