# Route model-degraded transitions through a typed contract

- Status: accepted
- Deciders: thkt
- Date: 2026-06-24
- Scope: [rust, model, observability, cross-repo]
- Confidence: high — census (2026-06-24) で reviewer が同定、critic-design が keep 判定。型システムで強制不可かつ embedder/reranker と下流 CLI を跨ぐ invariant であることをコード読解で裏付け。

## Context and Problem Statement

`src/model.rs` は model load の degraded path を 3 つの仕組みで統治する。`DegradedReason` (cache lookup / corruption / init / backend probe を 4 バケットに畳む coarse taxonomy)、`EmbedderDegraded::user_note` (embedder 専用の下流向け復旧メッセージ API)、そして typed error を `DegradedReason` に畳む際に必ず通すべき `degrade_with_warn` / `record_degraded` ヘルパである。

中核は「typed error を bare `.map_err(|_| DegradedReason::X)` で潰してはならない」というルールだ。bare 潰しは元の cause を warn event から落とし、log consumer が `EmbedError::Backend` と cache-lookup permission error を区別できなくする。doc-comment 自身が「関数の戻り型が変わらないため PR review はこの regression を flag しない」と明記する通り、型でも lint でもレビューでも守れない。`user_note` が返す文字列は下流 (sae / yomu / recall) がユーザに見せる API surface でもあり、観測性と下流 UX の両面で load-bearing な invariant が、コメント以外に拠り所を持たない。

## Decision Drivers

- mechanical 不可能性: bare `.map_err` 潰しは戻り型不変のため型・lint・レビューで検出できない (doc-comment が明言)
- 観測性の load-bearing 性: cause erasure は degraded path の根本原因を log から消し、UNKNOWN 排出率 (OUTCOME Indicator) の分析を不能にする
- cross-subsystem: embedder と reranker の両 loader、および下流 CLI の notification 表示に跨る
- census 由来: 2026-06-24 census で keep #2 同定

## Considered Options

- 本 ADR で degraded routing 契約を宣言し、`degrade_with_warn` / `record_degraded` を唯一の正規経路とする
- `DegradedReason` のコンストラクタを private 化し、ヘルパ経由しか作れない型で強制する
- clippy lint で `map_err(|_| DegradedReason` パターンを禁止する
- ADR 起票せず、model.rs のコメントのみで運用する

## Decision Outcome

Chosen option: 「本 ADR で degraded routing 契約を宣言する」。

1. **正規経路の宣言**: typed error から `DegradedReason` への変換は `degrade_with_warn` (元 error あり) または `record_degraded` (元 error なし) のみを正規経路とする。bare `.map_err(|_| DegradedReason::X)` は anti-pattern として禁止する。
2. **taxonomy と note の契約**: `DegradedReason` の 4 バケット粒度と `EmbedderDegraded::user_note` の `Option<String>` 戻り (下流が verbatim 表示する API) を契約として pin する。reranker-degraded equivalent は consumer が必要とするまで追加しない (現状の意図的非対称を維持)。
3. **下流の追従**: 下流 CLI (sae / yomu / recall) の degraded path も本契約に揃え、独自の cause-erasing 変換を持たない。

### Consequences

- Good: degraded path の cause が常に structured warn event に残り、log consumer が根本原因を追える
- Good: `user_note` の API contract が pin され、下流の verbatim 表示が安定する
- Bad: 型では強制できないため、最終防衛線はレビュー認知 + 下記 lint follow-up に依存する
- Bad: reranker-degraded の非対称が将来 consumer 登場時に再評価コストを生む

### Confirmation

- `git grep "map_err(|_| DegradedReason"` および `git grep "map_err(|_| .*Degraded"` が 0 件であることを継続確認
- clippy custom lint (実装 follow-up) で bare 潰しパターンを `deny` に昇格できれば mechanical gate へ移行
- 下流 repo で `EmbedderDegraded::user_note` 以外の degraded notification 自前実装が grep 0 件 (OEM Indicator と接続)

## Pros and Cons of the Options

### 本 ADR で routing 契約を宣言 (chosen)

- Good: cross-subsystem + cross-repo に跨る invariant を一意の文書に集約できる
- Good: lint follow-up へ昇格する受け皿になる
- Bad: 宣言時点では型強制が無く、レビュー認知依存

### コンストラクタ private 化で型強制

- Good: bare 潰しをコンパイルエラーにできる
- Bad: `degrade_with_warn` は closure を返す設計で、private 化すると call site の人間工学が悪化
- Bad: `Disabled` のような caller-only variant の生成経路と衝突する

### clippy lint で禁止

- Good: mechanical gate になりうる
- Bad: `map_err(|_| EnumVariant)` 一般を狙うと false positive が多く、custom lint 実装コストが高い
- Bad: lint だけでは「なぜ」が文書化されず、契約の所在が消える

### コメントのみ

- Good: 最軽量 (現状)
- Bad: doc-comment 自身が「レビューで flag されない」と認める通り、コメント単独では gate にならない

## More Information

### Quality Attributes

本契約は観測性 (observability) を quality attribute として優先する。degraded への fallback は許容するが、その cause を消すことは許容しない。`degrade_with_warn` の structured `error` / `reason` / `context` フィールドが、UNKNOWN 排出率を下げる (OUTCOME Indicator) ための一次データ源となる。

### Trade-offs

`EmbedderDegraded` newtype は embedder 専用 note を reranker に誤用させない greppable barrier として機能する。reranker 側に同等 note を持たせない非対称は、YAGNI に従い consumer 不在では追加しない判断であり、本 ADR はその非対称を意図的設計として記録する。

### Reassessment Triggers

- reranker-degraded notification を必要とする下流 consumer が登場した場合 (非対称の解消を再評価)
- bare 潰し禁止を強制する clippy custom lint が land した場合 (本 ADR の役割をレビュー gate から code gate へ縮退再評価)
- `DegradedReason` の 4 バケットでは区別が粗すぎる degraded 要因が出た場合 (taxonomy 拡張)

## References

- ADR-0005: Place Model Wiring in rurico, Keep amici as Thin Composition Base
- census report `docs/audit/2026-06-24-015844-adr-gaps.md` § ADR Promotion Candidates C4
- `src/model.rs` (`DegradedReason`, `EmbedderDegraded::user_note`, `degrade_with_warn`, `record_degraded`)
- `src/model/embedder.rs` / `src/model/reranker.rs` (loader の degraded path)
