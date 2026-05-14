# Pin BaselineSnapshot serde defaults to pre-existing behavior

- Status: accepted
- Deciders: thkt
- Date: 2026-05-14
- Scope: [rust, evaluation]
- Confidence: high — observable 3× in `src/eval/baseline.rs` (`default_aggregation_kind` / `pre_phase_5_disabled` / `HybridSearchConfig::default` via `#[serde(default)]`). `/audit-undocumented` 2026-05-14 で reviewer-rust + critic-design が独立 keep 判定 (Three Strongest #2)。`BASELINE_SCHEMA_VERSION` の bump policy はこの規律を前提に機能する。

## Context and Problem Statement

`BaselineSnapshot` (`src/eval/baseline.rs:L86-L150`) は新しいフィールド追加を `#[serde(default = ...)]` で吸収する設計で、historical な baseline.json を field 追加後も deserialize で round-trip できるようにしている。

このとき重要なのは **default が指す値の意味**である。`normalization` フィールド (L132-141) のコメントが明示しているように:

> Pre-normalization baselines lack this field; the serde-default points at `pre_phase_5_disabled` (all OFF), **not** at runtime `QueryNormalizationConfig::default` (all ON), so historical files replay under the behaviour they were captured with.

すなわち serde default は「runtime default」ではなく「pre-existing-behavior (そのフィールドが存在しなかった時点での実質挙動)」を指す。runtime default を割り当てると、historical な baseline.json の意味が "そのフィールドが現在の default 設定で評価された" に silently rewrite される。

この規律は 3 フィールド (`aggregation` = `AggregationKind::Identity`, `merge_config` = `HybridSearchConfig::default()`, `normalization` = `pre_phase_5_disabled`) で守られているが、module-level の規約として文書化されていない。`BASELINE_SCHEMA_VERSION` の bump policy (`pub const BASELINE_SCHEMA_VERSION: &str = "1.3";` のコメント参照) は「breaking change で bump」と定めるが、本規律 (= "default は pre-existing-behavior でなければ silently 意味が変わる") が崩れると bump policy 自体が機能しなくなる。

将来 4 つ目のフィールドを追加する contributor が runtime default を割り当てれば、historical baseline は新しい意味で読み直され、regression gate が誤検知/見逃しを起こす。型・lint で防げず、テストも難しい (historical な field-less ファイル fixture を pin する必要がある)。

## Decision Drivers

- bump policy 前提: 規律が崩れると `BASELINE_SCHEMA_VERSION` の breaking-change 判定が機能しない
- silent rewrite 防止: runtime default を当てると historical baseline の意味が静かに変わる
- type / lint 不可: serde default 関数の意味的正しさは型で表現できない
- 既存実装が示す precedent: `normalization` のコメントが暗黙の規律を露呈させている
- `/audit-undocumented` で keep 判定: Three Strongest #2

## Considered Options

* Module-doc + ADR で規律を明文化、新規フィールド追加 PR レビューの gate にする
* 各フィールド doc に runtime default 禁止コメントを per-field で繰り返す
* `BASELINE_SCHEMA_VERSION` bump を全フィールド追加で必須化し、default 規律は廃止
* 規律を文書化せず、現状の per-field コメントに任せる

## Decision Outcome

Chosen option: "Module-doc + ADR で規律を明文化、新規フィールド追加 PR レビューの gate にする"。

1. **規律宣言**: `BaselineSnapshot` の新フィールドは `#[serde(default = ...)]` を必ず付け、その default は **field 追加時点での pre-existing behavior (= そのフィールドが無かった時の実質挙動)** を表すこと。runtime default (例: `Foo::default()`) を割り当てる場合は、それが **historical baseline の実質挙動と一致する** ことを PR 説明で示すか、`BASELINE_SCHEMA_VERSION` を bump する。
2. **モジュールドキュメント追加** (実装 follow-up): `src/eval/baseline.rs` の module-doc (`//!`) に「# serde default discipline」段落を追加し、本 ADR を参照
3. **bump policy 接続**: `BASELINE_SCHEMA_VERSION` を bump せずに新フィールド追加する PR は本 ADR の規律遵守を review checklist に組み込む
4. **テスト marker** (実装 follow-up): historical baseline (field-less) を fixture として pin し、新フィールド追加時に default 経由で deserialize した結果が、historical 時代の挙動と数値一致することを検証する unit test を必須化

### Consequences

- Good: bump policy が安定して機能する (default 規律が前提条件として明示される)
- Good: silent rewrite が PR レビューで検出可能になる
- Good: ADR-0002 と本 ADR を読めば baseline schema 進化の規律が一貫して理解できる
- Bad: 新フィールド追加時に "pre-existing-behavior の値" を一度言語化する手間が発生
- Bad: type / lint で機械的に守れず、最終的にはレビュー認知に依存

### Confirmation

- 新フィールド追加 PR で `#[serde(default = ...)]` が付与され、default 値の意味 (pre-existing behavior か bump 同伴か) が PR 説明 or コードコメントで明示されていることを review checklist に追加
- 上記 4 の historical-fixture test が land 後は CI が自動 gate

## Pros and Cons of the Options

### Module-doc + ADR で規律を明文化 (chosen)

* Good: bump policy の前提条件として規律が明示される
* Good: 既存 3 フィールドの implicit precedent が後付けで explicit になる
* Bad: 規律自体が形式化されるだけで、機械的強制ではない

### per-field コメントを繰り返す

* Good: コードと最も近い距離に規律がある
* Bad: フィールドが増えるほどコメント重複が累積、コピペで「なぜ runtime default ではないか」の理由が薄まる
* Bad: 規律全体を見渡せる場所がない

### bump を全追加で必須化

* Good: 規律違反の余地が無くなる
* Bad: minor な additive 変更 (default が pre-existing behavior と完全一致) でも bump が必要になり、`BASELINE_SCHEMA_VERSION` の major/minor 意味が曖昧化
* Bad: 既存 3 フィールド追加時の "no-bump" 履歴と矛盾、retroactive 解釈が必要

### 規律を文書化せず、現状の per-field コメントに任せる

* Good: 軽量
* Bad: 4 つ目以降の contributor が runtime default を選び、historical baseline の意味が静かに rewrite される失敗モードを許容
* Bad: `/audit-undocumented` で incomplete-contract pattern と判定された経緯と矛盾

## More Information

### Trade-offs

`BaselineKind` (closed enum, `#[serde(rename_all = "snake_case")]`) と `aggregation` フィールドの typed/untyped 非対称は **PR #86 で resolved** — `aggregation` は `AggregationKind` typed enum に昇格し、wire form (`identity` / `max-chunk` / `dedupe` / `topk-average:k` / `none`) は manual `Serialize` / `Deserialize` impl で controlled (`src/eval/baseline.rs:117-171`)。audit report #56 で別途 type-system fix 候補と記録していた gap は閉じた。本 ADR は default **値の選び方** に関する規律のみを定める。

### Implementation Guidelines

新フィールド追加時の手順:

1. `#[serde(default = "fn_name")]` を付与
2. `fn_name()` は **そのフィールドが追加される前の実質挙動と等価な値** を返す
3. PR 説明に「default = `<value>` は pre-existing behavior に等しい (理由: ...)」を 1 行記載
4. 必要なら historical baseline fixture を `tests/fixtures/eval/historical/` 配下に追加し、deserialize → assert で挙動の不変性を pin
5. runtime default で十分な場合は理由を上記 3 で示すか、`BASELINE_SCHEMA_VERSION` を bump

具体例 (既存パターン):

| フィールド (追加時の文脈) | serde default | runtime default | 採用 default |
| ------------- | ------------- | --------------- | ------------ |
| `aggregation` (pre-aggregation baseline 存在) | `AggregationKind::Identity` | `AggregationKind::Identity` | `AggregationKind::Identity` (両者一致、`default_aggregation_kind`) |
| `merge_config` (pre-merge-config baseline 存在) | `HybridSearchConfig::default()` (`rrf_k=60`, weights=1.0) | 同上 | runtime default 同等 (`#[serde(default)]`) |
| `normalization` (pre-normalization baseline 存在) | `pre_phase_5_disabled` (all OFF) | `QueryNormalizationConfig::default` (all ON) | `pre_phase_5_disabled` (両者異なる、historical 側採用) |

### Reassessment Triggers

- `BaselineSnapshot` を struct 廃止し schema-bound serializer (e.g. `prost`/`schemars`-driven) に切り替える場合 → 規律の射程が変わる
- ~~historical fixture-based test が CI gate として land し、規律が機械的強制になった場合 → ADR の役割が縮退、Implementation Guidelines のみに統合再評価~~ — **PR #93 で fired、Decision Outcome 2 (`src/eval/baseline.rs` module-doc) + Decision Outcome 4 (`tests/fixtures/eval/historical/pre_serde_defaults.json` + `historical_baseline_resolves_serde_defaults_to_pre_existing_behavior` test) が land 済み。literal reading の「縮退、Implementation Guidelines のみに統合」は採用せず Accepted で維持: CI gate は既存 3 フィールド (aggregation / merge_config / normalization) の past behavior を mechanize するのみで、4 つ目以降のフィールド追加時に runtime default を割り当てるという future の失敗モードは捕捉できない (test は新規 field を assert しない)。新規フィールド追加 PR 時の「pre-existing-behavior か bump 同伴か」の判断は引き続き human cognitive gate (= ADR 本文 + review checklist) に依存する (2026-05-15)**
- ~~`aggregation: String` が typed enum に昇格した場合 → 本 ADR の例示更新~~ — **PR #86 で fired、本 ADR の line 19 / 90 / 106 例示更新済み (2026-05-14 adr-drift scan PR-A)**

## References

- ADR-0002: Search Quality Evaluation Methodology (§ Decision Outcome — `BaselineSnapshot` envelope を method-of-record として定義)
- ADR-0003: Add pgr-style First-Search Offline Retrieval Benchmark (§ Decision Outcome 3 — `BASELINE_SCHEMA_VERSION` を 1.2 → 1.3 bump した precedent)
- ADR-0004: Annotation Framework Foundation (§ Considered Options 2 — `BaselineSnapshot` envelope precedent を Provenance 型で踏襲)
- audit report `docs/audit/2026-05-14-undocumented-decisions.md` § Three Strongest #2 / § ADR Promotion Candidates #55
- `src/eval/baseline.rs:L52` (`BASELINE_SCHEMA_VERSION="1.3"` 定義)
- `src/eval/baseline.rs:L124-141` (3 つの `#[serde(default = ...)]` 用例)
- `src/eval/baseline.rs::pre_phase_5_disabled` (historical-behavior fn の precedent)
- `src/eval/baseline.rs::default_aggregation_kind` (historical-behavior fn の precedent)
