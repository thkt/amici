# Annotation Framework Foundation

- Status: proposed
- Deciders: thkt
- Date: 2026-05-08
- Confidence: medium — provenance envelope の必要性は ADR-0002 `BaselineSnapshot` 既存パターンで実証済。`eval::annotation` module path での既存 `EvalQuery.annotation` field との共存は T-001 readability test で検証可能。一方 Phase 1.5 Collaborative mode 着手時に model_id / mlx_rs_version 追加で `ANNOTATION_SCHEMA_VERSION` bump が必要となる migration debt は存在し、その時点で envelope 設計の再評価余地が残る。

## Context and Problem Statement

amici の `tests/fixtures/eval/queries.jsonl` は ADR-0002 fixture 規約に基づき 168 queries × 8 categories で運用されている。新カテゴリ追加・relevance map 拡張・annotation 文字列の手書きは継続して手作業で行われ、誰が・いつ・どの session で書いたかの provenance trail を残す手段がない。

ADR-0001 は amici の charter を「shared model-loading + CLI utilities」、ADR-0002 は「end-to-end retrieval-quality governance」と定義したが、いずれも authoring tool (queries.jsonl 拡張支援 subcommand 群) を含まない。Issue #53 で計画中の `annotate` / `annotation-stats` / `annotation-export` subcommand を charter 不変のまま追加すると charter creep となり、ADR の説明可能性 (contributor が `docs/decisions/` を読んだだけで charter scope を判断できる) が劣化する。

加えて、authoring tool が emit する session metadata は capture metadata (`BaselineSnapshot.schema_version` 等) と同様に schema-version envelope discipline を持たないと、後続 phase での field 追加 (model_id, mlx_rs_version 等) で silent drift が発生する。

## Decision Drivers

- charter clarity: authoring tool が ADR で説明可能な状態を維持し、新規 contributor が charter scope を `docs/decisions/` から判断できる
- provenance discipline: queries.jsonl 拡張に session-level provenance (annotator id / session id / fixture hash) を attach する型基盤を public API として提供
- envelope discipline: `ANNOTATION_SCHEMA_VERSION` 定数を canonical reference として固定し、schema version mismatch を runtime で typed error として検出可能にする (`oracle_gap.rs::OracleGapError::SchemaVersionMismatch` precedent)
- naming collision avoidance: 既存 `EvalQuery.annotation: String` field (`src/eval/fixture.rs:34-36`、`REQUIRED_QUERY_FIELDS:132`) と命名衝突せず両者を同一 test scope 内で参照可能にする
- 168 行 queries.jsonl の breaking change を避ける: 既存 fixture を rename / migration 不要のまま foundation を land する

## Considered Options

### Option 1: Annotation-tailored envelope + module path namespacing (chosen)

- Good: `eval::annotation::Session` / `eval::annotation::Entry` という module path で既存 `EvalQuery.annotation` field との衝突を解消、168 行の queries.jsonl rename / migration 不要
- Good: `ANNOTATION_SCHEMA_VERSION` を `BASELINE_SCHEMA_VERSION` (現 1.2) と独立に進化させられる、Phase 1.5 Collaborative mode で `model_id` / `mlx_rs_version` 追加時に baseline 側を bump せずに済む
- Good: ADR-0002 既存 envelope (`BaselineSnapshot`) precedent を踏襲した形で `Provenance` 型に schema_version / captured_with / fixture hash 等を持たせる
- Bad: capture envelope と authoring envelope の 2 envelope を維持するコスト。Phase 1.5 Collaborative mode で「authoring metadata と capture metadata を symmetric に揃えたい」という contributor 要望が出た場合、本 ADR の方針再評価が必要

### Option 2: BaselineSnapshot mirror + BaselineKind::Annotation variant

- Good: 1 envelope に統一、provenance schema が capture / annotation で symmetric
- Bad: `BASELINE_SCHEMA_VERSION` (現 1.2) を 1.3 へ bump 必要 → committed `baseline.json` fixture (T-021 で deserialise 検証中) も同時更新、Phase 1 sub-PR-A の scope を超える
- Bad: `BaselineKind::Annotation` の body shape が `BaselineSnapshot` (metrics + per_category 必須) と乖離、variant 追加時に shape divergence → optional field の serde default 設計が必要

### Option 3: rename existing `EvalQuery.annotation` field

- Good: 命名衝突なし、`Entry.annotation` という flat 命名が可能
- Bad: 168 行の queries.jsonl migration が必要 (各行の JSON key rename + fixture_hash 再計算 → `BASELINE_SCHEMA_VERSION` coordinate bump も波及)
- Bad: 既存 fixture file format の breaking change、ADR-0002 で凍結された `tests/fixtures/eval/` 規約から逸脱

## Decision Outcome

Option 1 を採用する。

1. **`eval::annotation` module 新設** — `src/eval/annotation.rs` に `Session`, `Entry`, `BlockMode`, `Provenance`, `AnnotationError`, `ANNOTATION_SCHEMA_VERSION` を pub export。tests は `src/eval/annotation/tests.rs` に分離 (LANG.md `sub.rs` + `sub/child.rs` 規約)。

2. **`ANNOTATION_SCHEMA_VERSION = "1.0"`** — `BASELINE_SCHEMA_VERSION` (現 1.2) と独立。`Provenance.schema_version` が canonical 値と不一致のとき `AnnotationError::SchemaVersionMismatch { got, expected }` を返す。

3. **Phase 1 で意図的に欠落させる field** — `Provenance` から `model_id` と `mlx_rs_version` を Phase 1 では除外。Phase 1.5 Collaborative mode (model-assisted relevance suggestion) 着手時に追加し、`ANNOTATION_SCHEMA_VERSION` を 1.0 → 1.1 へ bump する。

| Attribute | Phase 1 で欠落させる根拠 | 追加トリガと schema version bump |
| --------- | ------------------------ | ---------------------------------- |
| `model_id` | Phase 1 sub-PR-A は Block mode (手動 judgment、model 未関与) のみを foundation として扱う。`Provenance` に未使用 field を持たせると schema 欄外フィールドとして空文字列流通し、下流 consumer が「省略 OK」と誤解する | Phase 1.5 Collaborative mode (model-assisted suggestion) → `ANNOTATION_SCHEMA_VERSION` 1.0 → 1.1 bump で `Provenance.model_id: String` 追加 |
| `mlx_rs_version` | 同上、Block mode は ML runtime 未関与 | Phase 1.5 Collaborative mode → 同じ 1.0 → 1.1 bump で `Provenance.mlx_rs_version: String` 同時追加 |

4. **`AnnotationError` は `thiserror` + `#[non_exhaustive]`** — `BaselineError` / `FixtureError` / `OracleGapError` 既存 precedent を踏襲。variant 追加を非破壊で許可する forward-compatible 設計。

5. **Status: proposed で land し、PR-B / PR-C 着地後に accepted promote** — ADR-0003 precedent に従う。先行 PR-A で foundation を land、PR-B (annotate subcommand) / PR-C (annotation-stats / annotation-export) 着地時点で本 ADR を accepted へ promote する。

### Positive Consequences

- queries.jsonl 拡張に provenance trail (annotator id / session id / fixture hash) を attach する型基盤が public API として確立する
- charter scope が `docs/decisions/0004-*.md` を介して contributor に説明可能になる、authoring tool の charter creep が解消する
- schema-version envelope discipline が runtime check として enforce される (`SchemaVersionMismatch` typed error)
- 既存 `EvalQuery.annotation` field と 168 行 queries.jsonl は touch せず、breaking change を Phase 1 sub-PR-A から排除

### Negative Consequences

- capture envelope (`BaselineSnapshot`) と authoring envelope (`Session`) の 2 envelope を維持。Phase 1.5 Collaborative mode で symmetric 化を求められた場合、本 ADR の方針再評価が必要となる migration debt
- Phase 1.5 で `model_id` / `mlx_rs_version` 追加時に `ANNOTATION_SCHEMA_VERSION` bump が必須、その時点で committed annotation session fixture (もし存在すれば) も同時更新

## Reassessment Triggers

- Phase 1.5 Collaborative mode 着手時に「Annotation-tailored envelope を維持」「`BaselineSnapshot` mirror に migrate」両 option の比較判断。本 ADR §Decision Outcome 第 3 項の table が判断 anchor
- T-001 (naming collision readability) が PASS していても、実運用で `EvalQuery.annotation` と `eval::annotation::Entry` の混同による不具合が報告された場合 → §Considered Options Option 3 (rename existing field) を再評価
- contributor 数が増え authoring tool の使用頻度が baseline capture と同等以上になり、capture / authoring envelope の symmetric 化要望が複数 contributor から出た場合 → Option 2 (`BaselineSnapshot` mirror + `BaselineKind::Annotation`) を再評価
- `ANNOTATION_SCHEMA_VERSION` の bump 頻度が `BASELINE_SCHEMA_VERSION` を超える状態が継続した場合 → 2 envelope 維持の正当性 (独立進化) が機能している証左、本 ADR の方針継続を強化

## References

- ADR-0001 (amici extraction)
- ADR-0002 (eval governance methodology, `BaselineSnapshot` envelope precedent)
- ADR-0003 (Status: proposed precedent, schema version bump pattern via Issue #61 / #62)
- amici Issue #53 (this ADR's deliverable — annotation framework Phase 1)
- amici Issue #61 (`Hit@1` / `Hit@3` metrics — PR-B blocking)
- amici Issue #62 (first-search replay subcommand — PR-C blocking)
- `src/eval/baseline.rs` (`BASELINE_SCHEMA_VERSION` envelope precedent)
- `src/eval/oracle_gap.rs` (`OracleGapError::SchemaVersionMismatch` typed error precedent)
- `src/eval/fixture.rs:34-36` (existing `EvalQuery.annotation` field that motivates module-path namespacing)

## Downstream consumers (this ADR を前提として動く後続)

- amici Issue #53 PR-B — `annotate` subcommand (Block mode TTY shell + Session 永続化、本 ADR Phase 1 完了 + Issue #61 完了後に着手)
- amici Issue #53 PR-C — `annotation-stats` / `annotation-export` subcommand (Session 統計 + queries.jsonl 形式への export、本 ADR Phase 1 完了 + Issue #62 完了後に着手)
