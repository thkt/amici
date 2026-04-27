# Extract Shared Model-Loading and CLI Utilities into amici Crate

- Status: accepted
- Deciders: thkt
- Date: 2026-04-13
- Confidence: high — sae/yomuに `amici extraction target` コメントで管理済み、抽出コスト評価済み

## Context and Problem Statement

sae・yomu・recallの3クレートが `Spinner`、`try_expand_shorthand`、
`in_placeholders`/`anon_placeholders`/`as_sql_params`、`ModelLoad<T>`、
`try_load_reranker_with` を重複保持している。
現状は意図的なDRY違反として管理されているが、変更のたびに各PJへのバックポートが必要になる。

## Decision Drivers

- バックポート作業ゼロが目標Outcome
- 各PJのエラー型（`DegradedReason` 等）・スキーマ固有ロジックは各PJに残す（ADR-0037踏襲）
- rurico（低レイヤー: embedding + storage primitives）との役割分離を維持する

## Considered Options

### Option 1: amici クレートを新設して共通コードを抽出する

- Good: バックポート作業が不要になる
- Good: `try_load_model_with<A,M>` の汎用ジェネリック関数でembedder loadingフローを統一できる
- Bad: amici自体のメンテナンスコストが新たに発生する

### Option 2: 各 PJ に DRY を許容し続ける（現状）

- Good: 依存関係の追加なし
- Bad: バックポート忘れによる乖離リスクが継続する
- Bad: `amici extraction target` コメントが永続する技術的負債になる

### Option 3: rurico を拡張して共通コードを追加する

- Good: 既存の依存クレートへの統合
- Bad: ruricoは低レイヤー（embedding + storage primitives）の責務を超える
- Bad: CLI固有コード（`Spinner`、`try_expand_shorthand`）を低レイヤーに持ち込む設計上の問題

## Decision Outcome

Chosen option: Option 1（amiciクレートの新設）。バックポート作業ゼロを実現する唯一の選択肢であり、ruricoの責務を保つ。

### Positive Consequences

- sae/yomu/recallのバックポート作業がゼロになる
- `try_load_model_with<A,M>` でcache→probe→newフローを共通化し、各PJは型変換ラッパーのみ保持すればよくなる

### Negative Consequences

- amiciのgit rev更新が各PJのCargo.toml変更を伴う

## Implementation Plan

```text
amici/src/
├── lib.rs
├── cli/
│   ├── spinner.rs      (Spinner)
│   └── shorthand.rs    (try_expand_shorthand)
├── model/
│   ├── mod.rs          (ModelLoad<T>, try_load_model_with)
│   └── reranker.rs     (try_load_reranker_with)
└── storage.rs          (in_placeholders, anon_placeholders, as_sql_params, append_eq_filter)
```

## Migration Strategy

1. amiciに共通コードを実装し `rurico` を依存に追加する
2. sae/yomuでローカル定義を削除しamiciからimportするように書き換える
3. recallの `Spinner`（`Mutex<State>` 型）をamici版（`AtomicBool+Mutex<String>` 型）に差し替える
4. recallの `try_load_embedder` を `ModelLoad` パターンに書き直す

## Rollback Plan

amici依存を削除し、各PJに抽出前のコードを復元する。5カテゴリ合計で小規模なため低リスク。

## Success Criteria

- sae/yomu/recallが `cargo build` を通過する
- 抽出した5カテゴリのローカルコピーが各PJから削除されている
- `amici extraction target` コメントがsae/yomuから消えている
