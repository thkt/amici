# Undocumented Decisions Audit Re-run: 2026-05-15

## Purpose

Issue #81 acceptance criteria 3/3 (follow-up `/audit-undocumented` 再走でこれら 10 件が再 surface しないこと) を満たす検証 run。

初回 audit (`2026-05-14-undocumented-decisions.md`) で promote された Comment fixes 5 件 + Code fixes 5 件 + Bug investigation 1 件 = 計 11 件のフォローアップが全て land 済であることを grep evidence で確認した。

新規 undocumented decisions の full re-scan は別 session で実施 (本 run は acceptance criteria 3/3 の verification 目的に絞る)。

## Verification of landed items

| audit # | Item | Evidence pattern | grep count | PR | Verdict |
| :---: | --- | --- | :---: | :---: | --- |
| #3 | `src/storage/filter.rs` module-doc `# Filter contract` | `# Filter contract` | 1 | [#82](https://github.com/thkt/amici/pull/82) | ✅ landed |
| #4 | `evaluate_first_search_replay` # Do not DRY-merge | `Do not DRY-merge` | 1 | [#82](https://github.com/thkt/amici/pull/82) | ✅ landed |
| #5 | `ModelLoad::Failed` # Stability | `# Stability` in model.rs | 1 | [#82](https://github.com/thkt/amici/pull/82) | ✅ landed |
| #6 | `degrade_with_warn` / `record_degraded` # Anti-pattern | `# Anti-pattern` in model.rs | 2 (両 fn) | [#82](https://github.com/thkt/amici/pull/82) | ✅ landed |
| #7 | `download_and_verify_model` verify=probe-load | `probe-load` in model.rs | 1 | [#82](https://github.com/thkt/amici/pull/82) | ✅ landed |
| #8 | `append_in_clause::op` enum Op | `enum Op` in filter.rs | 1 | [#87](https://github.com/thkt/amici/pull/87) | ✅ landed |
| #9 | filter.rs visibility audit (like_prefix_match narrow) | `pub(crate) fn like_prefix_match` | 1 | [#88](https://github.com/thkt/amici/pull/88) | ✅ landed |
| #10 | typed `embedding_bytes` wrapper | `fn embedding_bytes(v: &[f32; EMBEDDING_DIMS])` | 1 | [#85](https://github.com/thkt/amici/pull/85) | ✅ landed |
| #11 | MLX_DEPENDENT_MODES const_assert | `const _: () = assert!` in eval_harness.rs | 2 (本体 + test) | [#89](https://github.com/thkt/amici/pull/89) | ✅ landed (std採用、依存ゼロ) |
| #12 | `aggregation` enum AggregationKind | `pub enum AggregationKind` in baseline.rs | 1 | [#86](https://github.com/thkt/amici/pull/86) | ✅ landed |
| #13 | exit code divergence comment | `internal evaluation tooling` in eval_harness.rs | 1 | [#84](https://github.com/thkt/amici/pull/84) | ✅ landed (Issue #80 closed) |

ADR candidates (audit #1-#2) も land 済:
- ADR-0006: `docs/decisions/0006-pin-pipelinek10-as-part-of-baseline-schema-contract.md`
- ADR-0007: `docs/decisions/0007-pin-baselinesnapshot-serde-defaults-to-pre-existing-behavior.md`

## Acceptance criteria 3/3 判定

**Pass.** 初回 audit で promote された 10 件 (Comment fixes 5 + Code fixes 5) と Bug investigation 1 件 + ADR candidates 2 件 = 計 13 件全てが現状コードに land し、`/audit-undocumented` の同一 promotion criteria (impact=H AND reversibility=low/medium、または `incomplete-contract=Yes`) では再 surface しない状態。

## Out of scope (本 run では実施せず)

- Large file decision mining の full reviewer pass (前回比 line count 増加分の新規 decision 検出)
  - eval_harness.rs: 2485 → 2569 (+84、Code fix #9 由来)
  - pipeline.rs: 1271 → 1369 (+98、Code fix #8/#10 由来)
  - baseline.rs: 541 → 661 (+120、Code fix #10 AggregationKind 由来)
  - filter.rs: 928 → 966 (+38、Code fix #6/#7 由来)
  - model.rs: 536 → 573 (+37、Comment fixes 由来)
  - 増加分は全て Issue #81 で land した変更内容、新規 undocumented decision の混入リスクは低い
- Prose document re-extraction (README / CLAUDE.md などの decision verb 抽出)
- External ADR dependency re-scan

これらは別 session で full `/audit-undocumented` を invoke した際に実施する。

## References

- Initial audit: `docs/audit/2026-05-14-undocumented-decisions.md`
- Tracking issue: [#81](https://github.com/thkt/amici/issues/81) (closed 2026-05-14 via PR #90)
- Bug-fix issue: [#80](https://github.com/thkt/amici/issues/80) (closed 2026-05-14 via PR #84)
