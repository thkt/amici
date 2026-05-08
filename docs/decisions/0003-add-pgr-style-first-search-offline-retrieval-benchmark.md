# Add pgr-style First-Search Offline Retrieval Benchmark

- Status: proposed
- Deciders: thkt
- Date: 2026-05-08
- Confidence: medium — pgr 記事 (entire.io 2026-05-06) で `MRR` / `Hit@1` / `Hit@3` の感度差は実証済み (fff vs baseline で Hit@1 -8 ポイント、pgr vs baseline で Hit@1 +8 ポイント)。amici の既存 fixture / Bootstrap CI / Oracle pipeline は再利用可能で実装コストは低い。一方、自社プロジェクト (yomu / sae) の ranking 改善でも同等の感度が出るかは要観測。

## Context and Problem Statement

amici の現行 eval-harness (ADR 0002) は `Recall@k`, `MRR@k`, `nDCG@k` で hybrid retrieval pipeline を評価する。これは「上位 k 件にあるか」を問う設計で、agentic search 文脈の load-bearing メトリック「first-search で関連ファイルが top に来るか」を分離計測できない。

[entire.io blog (2026-05-06)](https://entire.io/blog/improving-agentic-search-in-coding-agents) の "How We Improved Agentic Search" は first-search replay benchmark で:

- baseline (raw `ripgrep`): MRR 0.32, Hit@1 26%
- fff (高速だが ranking 改善なし): MRR 0.31 (-0.01), Hit@1 18% (-8 pt)
- pgr (definitions-first ranking): MRR 0.41 (+0.09), Hit@1 34% (+8 pt)

を観測した。fff の Hit@1 -8 ポイントは raw 速度では解決しないことを示し、pgr の +8 ポイントは ranking 戦略変更の効果を first-search 単独で捕捉している。amici の既存指標 (`Recall@10`, `MRR@10`, `nDCG@10`) はこの感度差を解像度不足で取り逃がす可能性が高い。

downstream の yomu (Issue: definitions-first / src-tier ranking 実験) や sae (現状 hybrid retrieval の質測定) を amici で評価可能にするには、first-search 単独で測れる土台が要る。

## Decision Drivers

- Goodhart 回避: 既存 `Recall@k` / `nDCG@k` は「上位 k 件にあるか」しか問わない。pgr 記事の核心 "first-result quality" を直接測れない
- 既存基盤の再利用: ADR 0002 の fixture (140+ queries × 7 categories) / Bootstrap CI (n=1000, seed=42) / Oracle pipeline (Issue #52) はそのまま流用可能
- 最小拡張: `MetricSpec` enum と `BaselineKind` enum への variant 追加で済む
- charter 整合: amici は "end-to-end retrieval-quality governance" (ADR 0002) であり、first-search も retrieval quality の一部

## Considered Options

### Option 1: `Hit@k` メトリクス追加 + `replay-first-search` サブコマンド (chosen)

- Good: `src/eval/metrics.rs` に `hit_at_k()` 追加 + `MetricSpec` enum に `Hit1` / `Hit3` variant 追加 (~20 行)
- Good: `replay-first-search` サブコマンドは既存 `evaluate()` の subset (Stage 3-5 skip) として実装可能 (~80 行)
- Good: 既存 fixture と並走可能、forward / oracle baseline に影響なし
- Good: `BaselineKind::FirstSearchReplay` variant 追加で baseline.json に kind 記録
- Bad: replay 用 trace 収集インフラは別。当面は固定 fixture から query 抽出する形

### Option 2: 既存 `Recall@k` / `MRR@k` のみで対応

- Good: ゼロ実装
- Bad: first-search 単独の挙動を分離できない。pgr 記事で fff が baseline 比 Hit@1 -8 ポイントだったケースを既存メトリクスでは捕捉しきれない可能性
- Bad: yomu の tier ranking 実験で「上位 10 件には入るが top 1 には来ない」変化が見えない

### Option 3: 完全な agentic loop simulation

- Good: end-to-end の効果も観測できる
- Bad: model API 呼び出し + 多段判断で複雑度過大
- Bad: 再現性が悪化 (model temperature, latency 変動)
- Bad: amici の charter (retrieval pipeline 評価) を逸脱

## Decision Outcome

Option 1 を採用する。

1. **`Hit@k` メトリクス追加** — `src/eval/metrics.rs` に `hit_at_k()` 関数を追加。`MetricSpec` enum に `Hit1`, `Hit3` variant を追加。`MetricSpec::tolerance` に Hit@k 用の許容値を設定。Bootstrap CI (95%, n=1000, seed=42) で算出。実装は Issue #61。

2. **`replay-first-search` サブコマンド追加** — `src/bin/eval_harness.rs` に新サブコマンドを追加。各 query について Stage 1 (FTS+Vec) + Stage 2 (RRF merge) のみ実行、Stage 3 (aggregation) / Stage 4 (rerank) / Stage 5 (final) は skip。出力は `QueryResult.ranked_hits[0..k]` のみ。実装は Issue #62。

3. **Baseline schema 更新** — `BaselineKind` enum に `FirstSearchReplay` variant を追加。`BASELINE_SCHEMA_VERSION` を 1.2 → 1.3 に bump。既存の `Forward` / `Reverse` / `Oracle` baseline には影響なし。

4. **既存基盤の再利用** — fixture は ADR 0002 で固定された `tests/fixtures/eval/` をそのまま使う。Bootstrap CI / per-category breakdown / `UNINFORMATIVE_HALF_WIDTH = 0.10` の閾値もそのまま適用。

## Reassessment Triggers

- yomu / sae の ranking 実験が complete し、Hit@k で有意な変化 (pgr 記事の +8 pt クラス) が観測されない場合 → benchmark 設計の見直し (fixture multiplicity / first-search 定義の精度)
- pgr 記事の手法を超えるベンチマークが業界標準化された場合 (例: fully agentic loop benchmark の reproducibility 解決) → Option 3 を再評価
- replay モードでの実行が forward baseline 比 2-5x の高速化に届かない場合 → Stage 3-5 の skip 漏れ調査
- Hit@k と Recall@k / MRR@k が常に一致した順序で動く場合 → Hit@k の独立な情報量がないと判断、削除を検討

## References

- [entire.io blog: How We Improved Agentic Search](https://entire.io/blog/improving-agentic-search-in-coding-agents) (2026-05-06)
- ADR 0002: Search Quality Evaluation Methodology
- Issue #61: `Hit@1` / `Hit@3` メトリクスの追加
- Issue #62: first-search replay モードの追加

## Downstream consumers (this ADR を前提として動く後続)

- [yomu#163](https://github.com/thkt/yomu/issues/163) — definitions-first / src-tier ranking の効果測定 (P1、本 ADR 完了後に着手)
- [sae#107](https://github.com/thkt/sae/issues/107) — sae の hybrid retrieval quality を amici で benchmark 化 (P3、本 ADR 完了後に着手)
- [recall#80](https://github.com/thkt/recall/issues/80) — recall README の ranking 記述と実装の整合化 (P2、独立タスク)
