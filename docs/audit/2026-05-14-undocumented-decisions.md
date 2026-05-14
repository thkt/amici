# Undocumented Decisions Audit: 2026-05-14

`/audit-undocumented` run on amici crate. Auto-detected scope: 8 large files (>400 lines), 4 prose docs, 5 local ADRs, 3 external ADR references in code.

## Summary

| Metric                          | Value |
| ------------------------------- | ----- |
| Large files scanned             | 8     |
| Documents scanned               | 4     |
| Raw decision candidates         | ~150  |
| Initial promotion candidates    | 16    |
| Post-challenge keep             | 2     |
| Post-challenge downgrade        | 5     |
| Post-challenge drop             | 8     |
| Post-challenge bug-fix follow-up| 1     |

ADR worth heuristic surfaces sharply: most candidates close with a comment, a type-system tightening, or are already covered by existing ADRs / README sections.

## Large File Decisions

### `src/bin/eval_harness.rs` (2485 lines)

| #  | Line  | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ----- | -------- | ----------- | -------------------- | ------ | ------------- |
| 1  | 5     | `eval-harness` feature gates the binary so default builds skip MLX-heavy code | Yes | No | M | high |
| 2  | 7-21  | Custom `key=value` argv parser, no `clap` dep | Partial | Yes | M | medium |
| 3  | 62-81 | `EvalContext<E, R>` bundles mock-injectable seams (embedder/reranker/timestamp closure) | Yes | No | M | high |
| 4  | 78-80 | `timestamp: Box<dyn Fn() -> String>` for deterministic snapshot diffs | Yes | No | M | high |
| 5  | 87-97 | `production_context()` is sole `LazyReranker` wrap site (ADR-0005) | Yes (ADR-0005) | No | - | - |
| 6  | 99    | `PIPELINE_K=10` — bumping silently invalidates every `*.baseline.json` | Partial | Yes | **H** | **low** |
| 7  | 114-123 | `VALID_EVALUATE_KINDS` closed set; `reverse`/`shuffled` deliberately have no oracle counterpart | Yes | No | M | medium |
| 8  | 128   | `ORACLE_KIND_SUFFIX="_oracle"` — DRY anchor | Yes | No | L | high |
| 9  | 130-136 | `AGGREGATION_NONE="none"` BR-003 marker; future Stage-3 reuse rule implicit | Yes | Yes | M | medium |
| 10 | 141   | `VALID_AGGREGATION_KINDS` mirrors `AggregationKind` variants; lockstep contract implicit | Yes | Yes | M | medium |
| 11 | 149-157 | `EXIT_REGRESSION=1` / `EXIT_USAGE=2` / `EXIT_INFRA=3` **contradicts** `amici::cli::exit_code::codes::USAGE=64` (ADR-0066 Group 2 baseline) | Partial | Yes | **H** | **medium** |
| 12 | 158   | `MLX_RS_VERSION="0.25"` hardcoded; update protocol vs `Cargo.toml` unstated | No | Yes | M | high |
| 13 | 165-169 | `MetricFn` type alias for no-clone borrowed-slice contract | Yes | No | L | high |
| 14 | 213-225 | `AggregationKind::name()` encodes `TopKAverage(k)` as `"topk-average:K"` to defeat silent k-downgrade | Yes | No | M | medium |
| 15 | 258-274 | `from_name` accepts both post-fix and legacy `"topk-average"`; deprecation timeline unstated | Yes | Yes | L | high |
| 16 | 283-339 / 346-402 | `dispatch_pipeline` vs `dispatch_oracle_pipeline` — DRY-rejection rationale recorded | Yes | No | - | - |
| 17 | 450-470 | `MetricSpec::ALL` contract anchor for FR-017 soundness | Yes | No | - | - |
| 18 | 502-516 | `MetricSpec::tolerance()` rule "≥ 2× observed max drift" + 1e-3 floor | Yes (ADR-0002) | No | - | - |
| 19 | 528-539 | `MLX_DEPENDENT_MODES` sandbox-seatbelt membership; new MLX-touching subcommand bypasses sandbox silently | Yes | Yes | H | medium |
| 20 | 541-576 | `main()` dispatch; `annotate` registered but absent from usage banner | No | Yes | L | high |
| 21 | 1316-1320 | `unreachable!("Reverse rejected by validate_committed_baseline_envelope")` documented invariant | Yes | No | - | - |
| 22 | 1856-1859 | `TTY_REJECT_MESSAGE` literal pinned by FR-011 test | Yes | No | - | - |

### `src/eval/pipeline.rs` (1271 lines)

| #  | Line   | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------ | -------- | ----------- | -------------------- | ------ | ------------- |
| 23 | 1-7    | `clean_for_trigram` direct import = "production wiring without mirror"; ADR-0002 covers alternative but not forward rule | Partial | Yes | H | low |
| 24 | 30     | `FTS_VOCAB_TABLE="docs_vocab"` couples schema to `prepare_match_query` | Partial | Yes | M | medium |
| 25 | 32-34  | `RRF_CANDIDATE_MULTIPLIER=3` mirrors recall's heuristic | Partial | Yes | M | medium |
| 26 | 36-50  | `QueryResult.ranked_hits` reuses `MergedHit`; score-semantics merge across rerank silent | Yes | Yes | M | medium |
| 27 | 60-114 | `PipelineError` `#[non_exhaustive]` with 7 root-cause-keyed variants | Partial | Yes | M | medium |
| 28 | 94-113 | `ChunkIdLengthMismatch` exists because `ChunkedEmbedding` fields are `pub`; load-bearing across external `Embed` impls | Yes | Yes | M | medium |
| 29 | 129+203 | `evaluate` vs `evaluate_first_search_replay` split = FR-008/010 compile-time guard; "do not DRY-merge" rule implicit | Yes | Yes | H | low |
| 30 | 272-287 | `vec_docs` chunk-granular + FTS parent-granular; future schema change risk silent | Yes | Yes | M | medium |
| 31 | 300-304 | `normalize_for_fts` applied to FTS body only; non-SentencePiece embedder breakage silent | Yes | Yes | M | medium |
| 32 | 334-345 | empty/length-mismatch/zip ordering rule implicit | Partial | Yes | M | medium |
| 33 | 351    | `bytemuck::cast_slice` couples `f32` layout + `EMBEDDING_DIMS` + sqlite-vec schema; no type binding | No | Yes | H | low |
| 34 | 438-449 | FTS `chunk_id = None` vs vec `Some(c_i)` fusion-key asymmetry retains source attribution | Yes | Yes | M | medium |
| 35 | 463-465 | `clean_for_trigram` returning `None` mapped to empty hits, not error | No | Yes | M | medium |
| 36 | 466,509 | `i64::try_from(...).unwrap_or(i64::MAX)` saturation pattern, not `as` cast | No | Yes | M | high |
| 37 | 554-601 | `Vec<Option<ResolvedSlot>>` + `take()` defends against duplicate-index reranker emissions | Yes | Yes | M | medium |
| 38 | 582-585 | Bodies vector pads dropped slots with `""` for index alignment | No | Yes | M | high |

### `src/storage/filter.rs` (928 lines)

| #  | Line    | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------- | -------- | ----------- | -------------------- | ------ | ------------- |
| 39 | 9-12    | `&'static str` column type = compile-time SQL-injection prevention; forward rule for new helpers implicit | Partial | Yes | H | medium |
| 40 | 3-5     | Caller MUST provide `WHERE 1 = 1` base clause; "typically" wording soft | Partial | Yes | M | medium |
| 41 | 27-35   | `anon_placeholders` vs `in_placeholders` — incremental-append safety | Yes | No | - | - |
| 42 | 37-40   | `as_sql_params` borrowed `&dyn ToSql` (not `Box`) for no-clone | Partial | Yes | L | medium |
| 43 | 66-79   | `escape_like` single-pass to avoid backslash double-escape | Yes | No | - | - |
| 44 | 66-79   | `escape_like` hard-coded `ESCAPE '\\'`; divergence risk with new helpers | Partial | Yes | M | medium |
| 45 | 81-94   | `like_prefix_match` ASCII-only + `PRAGMA case_sensitive_like` desync risk | Partial | Yes | M | medium |
| 46 | 145     | `append_in_clause(op: &str)` (NOT `&'static str`) — asymmetry with module's literal-only discipline | Partial | Yes | H | high |
| 47 | 165-242 | `Option<...>` filter helpers' `None=no-op / Some(empty)=1=0` convention per-function, not module-wide | Partial | Yes | H | medium |
| 48 | 195-213 | `append_exclude_ids` `&HashSet<i64>` not Option; asymmetry vs `append_include_ids` | Yes | No | - | - |
| 49 | 244-265 | `cutoff_ms` ms-since-epoch = recall's native format; downstream coupling silent | Partial | Yes | M | medium |
| 50 | 295-328 | `date(?, '+1 day')` lift to handle T-suffix RFC 3339 timestamps | Yes | No | - | - |
| 51 | whole   | All helpers `pub`; cross-crate API contract for yomu/sae/recall unstated | No | Yes | H | medium |

### `src/eval/baseline.rs` (541 lines)

| #  | Line    | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------- | -------- | ----------- | -------------------- | ------ | ------------- |
| 52 | 24      | `UNINFORMATIVE_HALF_WIDTH=0.10`; calibration source unstated | Partial | Yes | M | high |
| 53 | 52      | `BASELINE_SCHEMA_VERSION="1.3"`; bump policy partial | Partial | Yes | M | medium |
| 54 | 66-91   | `BaselineKind` variants; `Reverse` body shape not co-located | Partial | Yes | M | medium |
| 55 | 124-141 | `serde(default = pre-existing-behavior)` discipline observable 3x but never stated | Partial | Yes | **H** | **low** |
| 56 | 125     | `aggregation: String` vs `BaselineKind: enum` asymmetry; no justification | No | Yes | M | medium |
| 57 | 159-166 | `BaselineError::Serialise` structurally unreachable through `write_json` | Yes | Yes | L | high |
| 58 | 175-192 | `build_metric_result` strict-greater-than tie-break | Yes | No | - | - |
| 59 | 206-223 | `atomic_write` POSIX-rename atomicity; Windows / non-UTF-8 unstated | Partial | Yes | M | medium |
| 60 | 208-213 | `InvalidInput` for no-filename treats caller bug as recoverable | Partial | Yes | L | high |
| 61 | 238-241 | `write_json` 1-line wrapper preserves `BaselineError::Io` envelope | Partial | Yes | M | high |

### `src/model.rs` (536 lines)

| #  | Line    | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------- | -------- | ----------- | -------------------- | ------ | ------------- |
| 62 | 20-32   | `DegradedReason::Disabled` is caller-only; loader never produces | Yes | No | - | - |
| 63 | 80-88   | `degrade_with_warn` returns `impl FnOnce(E) -> DegradedReason`; mandatory-use rule README-only | Partial | Yes | H | medium |
| 64 | 80/115  | Cross-helper field-shape contract (`?reason, context, error`) implicit | Partial | Yes | M | medium |
| 65 | 123     | `#[must_use]` + `#[derive(Default)]` on `ModelLoad<T>`; `#[default] Absent` semantics implicit | Partial | Yes | M | medium |
| 66 | 132     | `ModelLoad::Failed(String)` deliberately erases typed error; downstream UX coupled to wording | No | Yes | H | medium |
| 67 | 137-142 | `as_ref(&self) -> Option<&T>` deliberately shadows `AsRef` trait | No | Yes | M | low |
| 68 | 158-166 | Hand-rolled `Debug` because `T: ?Debug` (e.g. `Arc<dyn Embed>`); redaction choice silent | No | Yes | M | high |
| 69 | 170-179 | `ModelDownloadError::ProbeFailed(String)` empty-string corruption sentinel | Partial | Yes | M | medium |
| 70 | 207-247 | `download_and_verify_model` — "verify" = probe-load (hash check delegated to rurico); name vs semantics | Partial | Yes | H | medium |
| 71 | 249-284 | `try_download_and_verify_with_fns` re-routes through loader for corrupt-deletion reuse | Partial | Yes | M | medium |
| 72 | 278-282 | `unreachable!` carries two invariants (cache=Some implies not NotInstalled; Disabled is caller-only) | Yes | No | - | - |

### `tests/eval_annotation.rs` (450 lines)

| #  | Line    | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------- | -------- | ----------- | -------------------- | ------ | ------------- |
| 73 | 7-10    | Phase 1 block-mode MLX-free → no `#[ignore]` | Yes | No | - | - |
| 74 | 23      | Local `const EXIT_USAGE: i32 = 2` duplicates `amici::cli::exit_code::codes::USAGE` (=64); same drift class as #11 | No | Yes | M | high |
| 75 | 114-115 | Inline byte-literal JSONL deliberately avoids round-trip via `Entry::to_json` | No | Yes | M | high |
| 76 | 156-160 | `fnv1a64:` prefix-only assertion; grammar (16 hex chars) unbound | Partial | Yes | L | high |
| 77 | 406-407 | `let _: &str = ...` compile-time coexistence proof; TIDYINGS-deletion risk silent | No | Yes | M | high |
| 78 | 415-450 | `arXiv:2602`/`AIANO`/`ADR-0021` forbidden-citations list hardcoded; policy not in ADR-0004 spec field | Partial | Yes | M | high |

### `src/eval/oracle_pipeline/tests.rs` (426 lines)

| #  | Line    | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------- | -------- | ----------- | -------------------- | ------ | ------------- |
| 79 | whole   | No positive test for "`reverse`/`shuffled` have no oracle counterpart" (structural-only enforcement) | No | Yes | M | medium |
| 80 | 83-114  | Issue #52 Japanese contract phrase ("配置済み doc 以外の順位は元の WeightedRrf 結果を維持する") only in test comment | Yes | Yes | M | medium |
| 81 | 218-274 | T-052-201 attribution to "Issue #52 P1 (advisor)" preserves audit trail | Yes | No | - | - |
| 82 | 191-216 | Sort rationale ties to baseline.json byte stability | Yes | No | - | - |

### `src/eval/metrics.rs` (420 lines)

| #  | Line    | Decision | Documented? | Incomplete-contract? | Impact | Reversibility |
| -  | ------- | -------- | ----------- | -------------------- | ------ | ------------- |
| 83 | 1       | Module doc lists FR-001..FR-004 but `hit_at_k` is unanchored | Partial | Yes | L | high |
| 84 | 19-38   | `MetricResult.uninformative` computed at serialize time, not in metric fns | Yes | No | - | - |
| 85 | 20-37   | `MetricResult.name: String` stringly-typed (no `MetricKind` enum) | No | Yes | M | medium |
| 86 | 40-71   | `recall_at_k`/`mrr_at_k` 0.0 sentinel dilutes mean; aggregation filter responsibility silent | Yes | Yes | M | high |
| 87 | 106-120 | `hit_at_k` cites ADR-0003 in rustdoc — strong baseline | Yes | No | - | - |
| 88 | 154-155 | Percentile indices integer division — 95% CI guarantee degrades for small `n_resamples` | No | Yes | M | high |
| 89 | 260-277 | T-006 nDCG worst-ordering test carries formula derivation + spec-revision rationale | Yes | No | - | - |

## Prose Document Decisions

### `README.md`

| #  | Line    | Decision verb | Decision | ADR coverage |
| -  | ------- | ------------- | -------- | ------------ |
| P1 | 36-104  | adopt         | `from_env()` + `from_env_with(impl Fn(&str) -> Option<String>)` DI pattern for env-var lookup | Partial (mentioned in ADR-0001 §Implementation, but DI contract not in ADR) |
| P2 | 107-167 | adopt         | `degrade_with_warn` + `record_degraded` helpers; "silent collapse hides regressions" | None — model.rs ADR-0001 lists the API but not the mandatory-use rule |
| P3 | 174-234 | adopt         | `CliError` trait + `amici::cli::exit_code::codes` sysexits constants | Partial (external ADR-0066 covers Group 2 baseline; README amplifies it) |
| P4 | 236-255 | adopt         | Group 2 baseline (ADR-0066 import): codes 64-78 sysexits + 80-119 PJ extension; `UNKNOWN`(104) / `INTERNAL` alias | Partial — external ADR-0066 is canonical, amici-local mirror absent |
| P5 | 258-285 | adopt         | Oracle mode Pattern A (retrieval-stage only); `eval-oracle-gap` AC 4 gate exits `EXIT_REGRESSION` (1) on per-category recall regression | Yes — ADR-0002 §Oracle Mode |
| P6 | 287-307 | regulation    | `git config --local core.hooksPath .githooks` mandatory pre-commit hook | None — workflow rule, not ADR-shaped |

### `justfile`

No load-bearing decisions beyond recipes documented at point-of-use. The "read-only against fixtures unless `*-baseline` / `*-reverse`" rule (L8) is a coarse safety convention but not an ADR-shaped decision.

### `deny.toml`

| #  | Line  | Decision verb | Decision | ADR coverage |
| -  | ----- | ------------- | -------- | ------------ |
| P7 | 32-37 | deny / allow  | `unknown-registry = "deny"`, `unknown-git = "deny"`, only `crates.io-index` registry + `thkt/rurico` git allowed | None — config-as-source-of-truth |
| P8 | 3-7   | ignore        | RUSTSEC-2024-0436 ignored (transitive via rurico mlx-rs / tokenizers) | None — ignore reason in TOML comment |

### `docs/decisions/README.md`

ADR index only. Decision: MADR v4 format + `/adr` skill for creation. Self-documenting; no candidates.

## External ADR Dependencies

| ADR     | Title                                                    | Status   | Relevance to amici                                              | Verdict                                                 |
| ------- | -------------------------------------------------------- | -------- | --------------------------------------------------------------- | ------------------------------------------------------- |
| 0021    | Build Slack conversation semantic search MCP (kiku)      | proposed | Used only as "forbidden post-cutoff citation" example in T-013   | Drop — not an amici dependency, just policy fodder      |
| 0037    | sae filter helpers aligned for amici extraction          | accepted | Predecessor work for ADR-0001; explicitly "踏襲" in ADR-0001 L18 | Drop — already absorbed into amici-local ADR-0001       |
| 0066    | CLI exit code policy grouped by error topology           | accepted | **amici is the Group 2 baseline crate**; `src/cli/exit_code.rs` is the implementation | Drop — triple-documented (global ADR + README + module doc) |

## ADR Promotion Candidates (post-challenge)

| # | Candidate                                                                                  | Initial Verdict | Challenge Verdict | Final Action                                  |
| - | ------------------------------------------------------------------------------------------ | --------------- | ----------------- | --------------------------------------------- |
| 6 | `eval_harness.rs:L99` — `PIPELINE_K=10` silently invalidates every baseline if bumped       | promote         | **keep**          | **ADR**                                       |
| 55 | `baseline.rs:L124-141` — `serde(default = pre-existing-behavior)` discipline (BUMP policy enabler) | promote         | **keep**          | **ADR**                                       |
| 47 | `filter.rs:L165-242` — `Option<>` `None=no-op / Some(empty)=1=0` convention                | promote         | downgrade         | filter.rs module doc `# Filter contract` section |
| 29 | `pipeline.rs:L129+L203` — `evaluate` vs `_replay` split = FR-008/010 compile-time guard    | promote         | downgrade         | `evaluate_first_search_replay` docstring "do not DRY-merge" |
| 66 | `model.rs:L132` — `ModelLoad::Failed(String)` typed-error erasure + downstream UX coupling | promote         | downgrade         | `ModelLoad::Failed` rustdoc `# Stability` note |
| 63 | `model.rs:L80/115` — `degrade_with_warn` / `record_degraded` mandatory-use rule            | promote         | downgrade         | `degrade_with_warn` rustdoc `# Examples` warning |
| 70 | `model.rs:L207-247` — `download_and_verify_model` verify=probe-load (hash delegated to rurico) | promote         | downgrade         | `download_and_verify_model` rustdoc `# Prerequisites` clarification |
| 39 | `filter.rs:L9-12` — `&'static str` column-type forward-binding rule                         | promote         | drop              | Reconsider after #46 (`append_in_clause` op) is fixed |
| 46 | `filter.rs:L145` — `append_in_clause(op: &str)` literal-only callsite discipline           | promote         | drop              | **type-system fix**: tighten to `&'static str` or `enum Op { In, NotIn }` |
| 51 | `filter.rs` (whole) — `pub` cross-crate API surface                                        | promote         | drop              | **type-system fix**: visibility audit (`pub(crate)` where appropriate) per LANG.md |
| 23 | `pipeline.rs:L1-7` — production-wiring without mirror forward rule                          | promote         | drop              | **comment fix**: extend existing module doc with "do not introduce eval-side mirrors" |
| 33 | `pipeline.rs:L351` — `bytemuck::cast_slice` type-binding                                   | promote         | drop              | **type-system fix**: typed `fn embedding_bytes(v: &[f32; EMBEDDING_DIMS]) -> &[u8]` |
| 19 | `eval_harness.rs:L528-539` — `MLX_DEPENDENT_MODES` sandbox membership                      | promote         | drop              | **code fix**: `static_assertions::const_assert!` or `[lints.rust]` rule |
| 56 | `baseline.rs:L125` — `aggregation: String` vs `BaselineKind: enum` asymmetry                | promote         | drop              | **type-system fix**: promote to `enum AggregationKind` mirroring `BaselineKind` |
| 11 | `eval_harness.rs:L149-157` — `EXIT_REGRESSION=1/USAGE=2/INFRA=3` contradicts `codes::USAGE=64` | promote         | **bug-fix follow-up** | **Investigate intent**: pre-ADR-0066 leftover (replace with `codes::SOFTWARE`/`USAGE`/`TEMP_FAIL`) or intentional internal-tooling scheme (then add 2-line divergence comment) — **do NOT ADR wrong behavior** |
| 16 | External ADR-0066 promotion to amici-local                                                  | promote         | drop              | Triple-documented (global ADR + README + `src/cli/exit_code.rs` module doc) |

Per-file summary: `keep 2 / downgrade 5 / drop 8 / bug-fix 1 (eval_harness.rs)`.

## Three Strongest ADR Survivors

1. **`PIPELINE_K=10`** (`src/bin/eval_harness.rs:L99`) — survives because the failure mode is silent: a contributor bumps to 20 "for better recall stats" and every committed `*.baseline.json` becomes semantically incomparable (recall@10 vs recall@20) with **no test catching it**. ADR-0002 covers methodology but explicitly does not bind this constant. No mechanical guard exists (cannot be a type, cannot be a lint).

2. **`serde(default = pre-existing-behavior)` discipline** (`src/eval/baseline.rs:L124-141`) — survives because it is the unwritten convention that makes the `BASELINE_SCHEMA_VERSION` bump policy work. Observable 3× in baseline.rs (`default_aggregation_kind="identity"` not runtime default; `pre_phase_5_disabled` not runtime default; pre-Oracle blank). Runtime defaults would silently rewrite the meaning of pre-existing baseline files on read. **No type can encode "should equal historical behavior, not today's default"** — pure forward-design convention.

3. *(Bug-fix follow-up, not ADR)* **`eval_harness.rs` exit codes 1/2/3 vs `codes::USAGE=64`** — directly contradicts the Group 2 baseline this same crate exposes per ADR-0066. Either a pre-ADR-0066 leftover (fix to use `codes::*`) or intentional internal divergence (then add 2-line comment). Surfacing wrong behavior in an ADR would lock it in.

## Recommended Follow-up Issues

### ADR candidates (file via `/adr`)

1. **ADR draft**: "Pin `PIPELINE_K=10` as part of the baseline schema contract" — covers reassessment trigger (when can K change), backfill protocol for old baselines, and recapture sequence.
2. **ADR draft**: "`serde(default = ...)` historical-value discipline for `BaselineSnapshot` field additions" — covers the bump-vs-no-bump rule, the "pre-existing behavior" wording, and the linkage to `BASELINE_SCHEMA_VERSION`.

### Documentation/comment fixes (no ADR)

3. `src/storage/filter.rs` module doc → add `# Filter contract` paragraph covering `Option<>` `None=no-op / Some(empty)=1=0` rule.
4. `src/eval/pipeline.rs::evaluate_first_search_replay` docstring → add "do not DRY-merge with `evaluate`; the signature is the FR-008/010 compile-time guard".
5. `src/model.rs::ModelLoad::Failed` rustdoc → add `# Stability` note on stringly-typed payload + downstream UX coupling.
6. `src/model.rs::degrade_with_warn` rustdoc → add `# Examples` warning "never collapse with bare `.map_err(|_| Reason::X)`".
7. `src/model.rs::download_and_verify_model` rustdoc → add to `# Prerequisites` "verify means probe-load; hash check is delegated to rurico".

### Code fixes (type-system / lint)

8. `src/storage/filter.rs::append_in_clause` → tighten `op: &str` to `op: &'static str` or `enum Op { In, NotIn }`.
9. `src/storage/filter.rs` → visibility audit; narrow `pub` to `pub(crate)` per LANG.md guidance where downstream does not use the helper directly.
10. `src/eval/pipeline.rs::index_corpus` → introduce typed `fn embedding_bytes(v: &[f32; EMBEDDING_DIMS]) -> &[u8]` to replace bare `bytemuck::cast_slice` and bind the schema-layout contract.
11. `src/bin/eval_harness.rs::MLX_DEPENDENT_MODES` → wrap with `static_assertions::const_assert!` or a clippy custom rule to make membership mechanical.
12. `src/eval/baseline.rs::aggregation` → promote `String` field to `enum AggregationKind` mirroring `BaselineKind` typed/closed-set discipline.

### Bug investigation

13. `src/bin/eval_harness.rs:L149-157` — Investigate `EXIT_REGRESSION/USAGE/INFRA` vs `amici::cli::exit_code::codes` (ADR-0066 baseline). Determine if pre-ADR-0066 leftover (replace with `codes::SOFTWARE`/`USAGE`/`TEMP_FAIL`) or intentional divergence (add 2-line explicit comment). Do not ADR until intent is clarified.
