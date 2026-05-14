# Pin PIPELINE_K=10 as part of baseline schema contract

- Status: accepted
- Deciders: thkt
- Date: 2026-05-14
- Scope: [rust, evaluation]
- Confidence: high — failure mode が silent (テスト不可能) であることを `/audit-undocumented` (2026-05-14) で reviewer-rust が独立同定、critic-design が keep 判定。ADR-0002 § Reassessment Triggers と recapture protocol との接続点が明確。

## Context and Problem Statement

`src/bin/eval_harness.rs:L99` の `const PIPELINE_K: usize = 10;` は ADR-0002 の評価方法論で前提とする「上位 k 件」の `k` 値を決定する。bootstrap seed (`SHUFFLE_SEED=42`, `BOOTSTRAP_SEED=42`) と並んで、baseline.json 出力 (`recall@10`, `mrr@10`, `ndcg@10`) の意味を決める load-bearing constant である。

ADR-0002 は方法論 (fixture 規約、Bootstrap CI、tolerance envelope) を定めるが PIPELINE_K の具体値は固定しない。すなわち value を変えても ADR-0002 の方法論には違反しないが、`tests/fixtures/eval/baseline.json` 等に記録された値の意味は変わる (recall@10 と recall@20 は semantic に別物)。失敗モードは silent — テストは通り、CI も通り、レビューでも見落とされうる。

type / lint で守ることもできない (constant 値の意味は外部固定値依存)。コード側コメント単独では「変更時に recapture が必要」という protocol を将来 contributor に強制できない。

## Decision Drivers

- silent invalidation 阻止: PIPELINE_K bump は `tests/fixtures/eval/*.json` 全 baseline の semantic 比較を無効化するが、テストでは検出されない
- ADR-0002 接続点の再利用: 既存の Reassessment Triggers と recapture protocol (`just eval-baseline` / `just eval-reverse` / `just eval-oracle`) をそのまま流用可能
- mechanical 不可能性: 型・lint・テストいずれも constant 意味の不変性を保証できない
- audit 由来: `/audit-undocumented` 2026-05-14 run で reviewer-rust + critic-design の独立同定により Three Strongest #1 として keep 判定

## Considered Options

* PIPELINE_K=10 を baseline schema contract の一部として本 ADR で固定
* ADR-0002 を直接編集して値を埋め込む
* sealed const generic で型システムに上げる
* ADR 起票せず、コード側コメントのみ追加

## Decision Outcome

Chosen option: "PIPELINE_K=10 を baseline schema contract の一部として本 ADR で固定する"。

1. **値固定**: `PIPELINE_K=10` を baseline.json schema 契約の load-bearing constant として宣言する (`BASELINE_SCHEMA_VERSION` と同格扱い)
2. **変更プロトコル**:
   - (a) `BASELINE_SCHEMA_VERSION` bump 必須
   - (b) `tests/fixtures/eval/*.json` 全 baseline の recapture 必須 (`just eval-baseline` / `just eval-reverse` / `just eval-oracle`)
   - (c) ADR-0002 § Reassessment Triggers に "PIPELINE_K change" 行追加
   - (d) 本 ADR を supersede する新 ADR を起票
3. **コード側マーカー** (実装 follow-up): `src/bin/eval_harness.rs:L99` の `PIPELINE_K` 定義に本 ADR を参照するコメント追加
4. **検証マーカー** (実装 follow-up): `verify-baseline` で baseline.json の `recall@k` 等のキー名から `k` を逆引きし、`PIPELINE_K` と一致しない baseline は reject

### Consequences

- Good: 将来の contributor が PIPELINE_K を bump しようとした時点で、本 ADR が PR レビューの gate になる
- Good: ADR-0002 の方法論層を破壊せず、value 変更履歴を本 ADR の supersede chain に集約できる
- Bad: ADR-0002 と本 ADR の二段で読む必要が生まれる
- Bad: type / lint で機械的に守れないため、最終的にはレビュー時の認知に依存する

### Confirmation

- `git grep "PIPELINE_K"` で定義点が `src/bin/eval_harness.rs:L99` の 1 か所のみであることを継続確認
- `BASELINE_SCHEMA_VERSION` を bump する PR が本 ADR を参照していることをレビューチェックリストに組み込む
- 上記 4 の `verify-baseline` 検証マーカーが land 後は CI が自動で gate

## Pros and Cons of the Options

### PIPELINE_K=10 を本 ADR で固定 (chosen)

* Good: silent drift の最低限の防衛ライン (PR レビュー時に本 ADR が hit する)
* Good: ADR-0002 の方法論層を壊さず、value 変更履歴を本 ADR 単独で持てる
* Bad: 型では encode できない、レビュー認知依存

### ADR-0002 を直接編集して値を埋め込む

* Good: ADR の本数を増やさない
* Bad: ADR-0002 は方法論 ADR で具体定数を持たない設計。`UNINFORMATIVE_HALF_WIDTH=0.10` 等も同様に「値そのものは ADR-0002 が抱えない」スタイルで一貫している
* Bad: 将来 K を変えた時の supersede 履歴が ADR-0002 単独になり、評価方法論の歴史と value 変更の歴史が混ざる

### sealed const generic で型システムに上げる

* Good: コード側で value を 1 か所固定できる
* Bad: 型は値の意味 (recall@10 が baseline と比較可能であること等) を保証しない、結局コメントが必要
* Bad: `MetricFn` シグネチャ等の API 変更が広範に及ぶ

### ADR 起票せず、コード側コメントのみ追加

* Good: 軽量
* Bad: コメントは「変更時の recapture プロトコル」を contributor に強制できない (review が gate にならない)
* Bad: `/audit-undocumented` で incomplete-contract pattern と判定された経緯と矛盾

## More Information

### Trade-offs

`MetricSpec::ALL` (`eval_harness.rs:L450-470`) が「misspelled metric name はゲートをすり抜けない」soundness anchor として機能しているのと同じ強度で、PIPELINE_K も schema contract の一部と宣言する。代償として「評価実装の意味は ADR-0002 と本 ADR の二段を読まないと完結しない」読みコストを許容する。

### Implementation Guidelines

- 新たな `*_at_k` メトリクス追加時 (ADR-0003 の `Hit@k` 追加 precedent 参照) は、`k` 値が `PIPELINE_K` と揃っていることを compile-time または runtime で assert
- `BaselineSnapshot::verify` 系の関数で `recall@k` キー名から K を逆引きし、`PIPELINE_K` と不一致なら `EXIT_REGRESSION` で fail

### Reassessment Triggers

- 業界標準で k≠10 (例: k=20) が agentic search の sensitivity sweet spot として確立した場合
- fixture が 168 queries × 8 categories から大幅拡張され、k=10 では信号が薄まる場合
- yomu / sae の downstream で k=10 では tier ranking 効果が見えなくなった場合
- ~~`verify-baseline` の自動 gate (上記 Implementation Guidelines) が land した場合、本 ADR の役割がレビュー gate から code-level gate へ移行 → ADR の自然消滅か Implementation Guidelines のみへ縮退を再評価~~ — **PR #94 で fired、Decision Outcome 3 (`PIPELINE_K` doc comment) + Decision Outcome 4 (`validate_committed_baseline_envelope` の K mismatch gate) が land 済み。自然消滅 / 縮退 は採用せず Accepted で維持: (a) change protocol step (d) が本 ADR の supersede chain を前提とするため自然消滅は構造的に不可、(b) MADR section 必須項目を満たすため Implementation Guidelines 単独への縮退は不可。CI gate は stale baseline を reactive に検出するが、PIPELINE_K bump 時の change protocol 4 step は ADR が prescriptive に gating する役割を保持 (2026-05-15)**

## References

- ADR-0002: Search Quality Evaluation Methodology (§ Decision Outcome / § Reassessment Triggers)
- ADR-0003: Add pgr-style First-Search Offline Retrieval Benchmark (§ Reassessment Triggers, `Hit@k` 追加 precedent)
- audit report `docs/audit/2026-05-14-undocumented-decisions.md` § Three Strongest #1 / § ADR Promotion Candidates #6
- `src/bin/eval_harness.rs:L99` (`PIPELINE_K` 定義箇所)
- `src/eval/metrics.rs` (`recall_at_k` / `mrr_at_k` / `ndcg_at_k` / `hit_at_k` — k を消費する metric 関数群)
- `tests/fixtures/eval/baseline.json` (固定対象 baseline)
