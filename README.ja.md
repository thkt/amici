# amici

sae/yomu/recall ツールチェーンで共有するモデルローディング、ストレージヘルパー、CLI ユーティリティ。

## モジュール

| Module | Contents |
| ------ | -------- |
| `model` | `DegradedReason`, `degraded_reason_user_note`, `ModelLoad<T>`, `ModelDownloadError`, `download_and_verify_model` |
| `model::embedder` | `try_load_embedder_with` — エンベディングモデルをロード |
| `model::reranker` | `try_load_reranker_with` — リランキングモデルをロード |
| `storage` | `in_placeholders`, `anon_placeholders`, `as_sql_params`, `append_eq_filter` |
| `cli` | `Spinner`, `with_spinner`, `try_expand_shorthand` |
| `migration` | `notify_schema_change` — スキーマクリア通知用の `tracing::warn!` 統一実装 |
| `logging` | `init_subscriber` — `RUST_LOG` 対応の `tracing_subscriber::fmt` 初期化（CLI `main.rs` 用） |

## 使い方

```toml
[dependencies]
amici = { git = "https://github.com/thkt/amici", rev = "<rev>" }
```

## 開発

### セットアップ

リポジトリを clone したら一度だけ実行してください：

```sh
git config --local core.hooksPath .githooks
```

これで `git commit` 時に `cargo fmt --check` と `cargo clippy --all-targets --all-features -- -D warnings` が自動で走ります。違反があれば commit は中止されます。一時的にスキップしたい場合は `git commit --no-verify`。

### よく使うコマンド

```sh
just check                                                # test + lint + fmt-check
cargo test --all-features                                 # tests only
cargo clippy --all-targets --all-features -- -D warnings  # lint（CI と同条件）
cargo fmt -- --check                                      # フォーマット検査
```
