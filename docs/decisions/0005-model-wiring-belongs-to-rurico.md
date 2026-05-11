# Place Model Wiring (lazy load, cache, probe) in rurico, Keep amici as Thin Composition Base

- Status: accepted
- Deciders: thkt
- Date: 2026-05-11
- Confidence: high — ADR-0001 の責務分離方針を再確認・具体化、Issue #68 設計時に確定

## Context and Problem Statement

amici は yomu / sae / recall の共通基盤として ADR-0001 で抽出された。Issue #68 (replay-first-search で reranker init を skip) の設計時、3 案 (FailingReranker / Optional<R> / LazyReranker) を比較した結果、**LazyReranker (`Rerank` trait delegate impl で `OnceLock<R>` + init closure)** が「resource lazy」 ideal に最も近いと判明した。

ここで論点が浮上した — `LazyReranker` を amici / rurico どちらに置くか。amici は yomu / sae / recall の共通基盤として薄く保ちたい。一方で `LazyReranker` の中身は `Rerank` trait と `RerankerError` (model layer primitives) に強依存しており、これらの owner は rurico 側である。判断基準を Issue #68 単体で済ませず、本 ADR で恒久的なレイヤー責務分担として明文化する。

## Decision Drivers

- amici は yomu / sae / recall 共通基盤として薄く保つ (ADR-0001 の趣旨を継続)
- model layer (`Embed` / `Rerank` trait, `ModelInitError` / `RerankerError`, `Reranker` / `Embedder` struct) は rurico 所有
- amici に model wrapper を増やすと、各 CLI が rurico を直接使う局面で同等 wrapper を再発明する誘惑が生まれる
- rurico に置けば 3 CLI + amici が同じ wrapper を共有できる

## Considered Options

### Option 1: model wiring 責務は rurico、amici は配線統合に専念

- Good: ADR-0001 の責務分離を強化、amici が薄く保たれる
- Good: yomu / sae / recall も将来 rurico の lazy wrapper を直接使える
- Good: model layer の API surface が rurico に集約され、testing seam 設計も一箇所
- Bad: rurico 側 issue / PR が必要、amici の修正は rurico merge 待ちになる
- Bad: rurico リリースサイクル (rev bump) を amici が follow する flow が必要

### Option 2: amici 側に LazyReranker を置く

- Good: amici だけで完結、rurico 触らない
- Bad: rurico の `Rerank` trait に密結合した wrapper が amici に増える、責務分散
- Bad: 同等 wrapper が yomu / sae / recall で再発明される懸念
- Bad: ADR-0001 が定めた「rurico は低レイヤー、amici は CLI 共通配線」の境界が滲む

### Option 3: 各 CLI 個別実装で amici / rurico に置かない

- Good: 各 CLI の自由度
- Bad: バックポート作業発生 (ADR-0001 の問題に逆戻り)
- Bad: model layer wrapper が 3 CLI で 3 通りに分岐する long-term リスク

## Decision Outcome

Chosen option: **Option 1** (rurico に model wiring を置き、amici は配線統合に専念)。

判定基準 (将来 amici 拡張時の指針):

| 機能カテゴリ | 配置先 |
|---|---|
| `Embed` / `Rerank` trait の wrapper (lazy / cache / retry など) | rurico |
| `ModelInitError` / `RerankerError` / `EmbedError` への variant 追加 | rurico |
| 設定 cache / probe / model load の seam | rurico |
| 上記 primitives を **組み合わせた** 業務 facets (eval harness / storage 統合 / fixture loading など) | amici |
| CLI 個別 schema / DegradedReason 等の固有エラー型 | 各 CLI (yomu / sae / recall) |

amici が新機能を考える時、**まず「rurico の primitive を組み合わせるだけで実現可能か」を問う**。新 primitive が必要なら rurico 側 issue を立てて先行 PR 化、amici は rev bump 経由で取り込む。

### Positive Consequences

- amici が共通基盤として薄く保たれる (ADR-0001 の趣旨を継続)
- yomu / sae / recall が rurico の lazy wrapper を直接使える可能性が拡がる
- amici test がモデル層を mock するパターンを単純化できる
- 将来の同種要求 (`LazyEmbedder` / `RetryReranker` 等) も同じ判断軸で rurico に集約

### Negative Consequences

- rurico 側 issue / PR が増え、amici の修正が rurico merge 待ちになる
- rurico リリースサイクル (rev bump) を amici が follow する flow が必要 (既存 `chore(deps)` bump 慣習で吸収済)
- 「rurico に置けない例外」を判断するレビューポイントが増える

## Implementation Trigger

- Issue #68 (amici): `replay-first-search` / `verify-baseline kind=first_search_replay` で reranker init を skip する要求
- → rurico Issue #154 (https://github.com/thkt/rurico/issues/154): `LazyReranker<R>` + `RerankerError::InitFailed` 追加
- → amici 側 #68: rurico merge → rev bump 後、`production_context()` で `LazyReranker::new(init_reranker)` で wrap

## Alternatives Rejected at Issue #68

| 代替案 | 棄却理由 |
|---|---|
| 案 5 (`EmbedContext` / `EvalContext` 2 型分割) | 「型レベル能力差」 ideal に近いが、本 ADR の責務分担に従えば rurico 側で resource lazy として表現する方が筋。amici 内部の dispatch 二分岐も削れる |
| 案 1A (`FailingReranker` + dummy 型) | 静的保証は dispatch logic 依存、Forward arm の monomorphize に dead code が残る。amici 内部の応急処置に近く、rurico 拡張で根本解決すべき |

## Future Implications

本 ADR は ADR-0001 の責務分離を **継続的判断軸として明文化** する。今後 amici に拡張要求が来た時、本 ADR の判定基準表を最初に当てる。「rurico に置けない理由」が示せない場合は rurico 側 issue 化が default。
