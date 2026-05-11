# amici

sae/yomu/recall ツールチェーンで共有するモデルローディング、ストレージヘルパー、CLI ユーティリティ。

## モジュール

| Module | Contents |
| ------ | -------- |
| `model` | `DegradedReason`, `degraded_reason_user_note`, `degrade_with_warn`, `record_degraded`, `ModelLoad<T>`, `ModelDownloadError`, `download_and_verify_model` |
| `model::embedder` | `try_load_embedder_with`, `try_load_embedder_default_logging` — エンベディングモデルをロード |
| `model::reranker` | `try_load_reranker_with` — リランキングモデルをロード |
| `storage::filter` | `in_placeholders`, `anon_placeholders`, `as_sql_params`, `append_eq_filter`, … — SQL `WHERE` 句とパラメータの構築ヘルパー |
| `storage::query_helpers` | `collect_rows`, `fetch_by_in_clause` — `Connection` を取って行を回収する実行系ヘルパー（コレクション型・エラー型を generic にサポート） |
| `storage::fts` | `clean_for_trigram` — `rurico::storage::MatchFtsQuery` を FTS5 `trigram` トークナイザ向けに整形 |
| `cli` | `Spinner`, `with_spinner`, `embed_with_spinners`, `done`, `try_expand_shorthand`, `env_lookup`, `CliError`, `exit_code::codes`, `exit_error`, `hint_arrow`, `info`, `deprecation_warn`, `progress_step` |
| `migration` | `notify_schema_change` — スキーマクリア通知用の `tracing::warn!` 統一実装 |
| `logging` | `init_subscriber` — `RUST_LOG` 対応の `tracing_subscriber::fmt` 初期化（CLI `main.rs` 用） |
| `eval` *(feature `eval-harness`)* | `rurico` プリミティブと合成した検索評価ハーネス。`eval_harness` バイナリの裏側 (ADR 0002)。 |
| `testing::hybrid` *(feature `test-support`)* | ハイブリッド検索における FTS↔vec 対称性のコントラクトアサーション。 |

## 使い方

```toml
[dependencies]
amici = { git = "https://github.com/thkt/amici", rev = "<rev>" }
```

## Features

| Feature | 効果 |
| ------- | ---- |
| `eval-harness` | `amici::eval` と `eval_harness` バイナリをビルド。`just eval-*` レシピが要求。 |
| `test-support` | `amici::testing::hybrid` を公開。下流 crate が `[dev-dependencies]` でハイブリッド検索のコントラクトヘルパーを再利用できる。 |

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
