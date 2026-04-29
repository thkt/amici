# Search Quality Evaluation Methodology

- Status: accepted
- Deciders: thkt
- Date: 2026-04-27
- Confidence: high — supersedes the methodology accepted in rurico ADR 0003. Reference composition pattern is empirically established in `recall/src/hybrid.rs`; statistical significance via bootstrap CI on 140+ query fixtures is a well-known IR convention; mlx inference f32 drift across machines / mlx-rs versions is bounded by the regeneration tolerance encoded in `src/bin/eval_harness.rs::MetricSpec::tolerance`.

## Context and Problem Statement

amici hosts the production retrieval wiring (`crate::storage::fts::clean_for_trigram`) that `recall`/`sae`/`yomu` actually run. End-to-end retrieval quality evaluation must measure that wiring directly, not a mirror.

rurico ADR 0003 originally placed the harness inside rurico to evaluate primitives in a `recall`-shape composition. The cyclic-dependency constraint that justified inlining the wiring mirror does not apply to amici (`amici → rurico` is the existing direction in `Cargo.toml`). rurico ADR 0006 records the migration; this ADR re-states the methodology in amici without behavioural change — the *location* of the harness moves; the methodology does not.

## Decision Drivers

- Goodhart's-law structural avoidance: the metric must measure production wiring, not a mirror that drifts from production.
- Charter alignment: `rurico` stays at "ruri-v3 + storage primitives"; `amici` carries end-to-end retrieval-quality governance for its `sae`/`yomu`/`recall` consumers.
- Reproducibility: identical embedder/reranker primitives + identical fixture must produce metric values within the per-metric tolerance envelope across machines and over time.

## Considered Options

### Option 1: Host the harness in amici (chosen)

- Good: harness measures production wiring directly; `clean_for_trigram` exists in exactly one place after amici#24 ships the `MAX_COMBOS` guard.
- Good: no cyclic dependency (`amici → rurico` already exists).
- Bad: amici's scope expands to host the harness; mitigated by the `eval-harness` feature gate so default builds skip the module.

### Option 2: Keep the harness in rurico

- Good: zero migration cost.
- Bad: two implementations of `clean_for_trigram`; baseline measures a mirror, not production. (Detailed in rurico ADR 0006 §Options Considered Option A.)

### Option 3: Promote `clean_for_trigram` to a rurico primitive

- Good: single source of truth for the FTS5-trigram adapter.
- Bad: widens rurico's charter to include FTS5-trigram-specific knowledge; does not address the larger duplication (the entire pipeline mirror). (Detailed in rurico ADR 0006 §Options Considered Option B.)

## Decision Outcome

Adopt the four-part methodology, hosted in amici:

1. **Reference composition** — `recall`-inspired hybrid pipeline implemented inline in `src/eval/pipeline.rs` and `src/bin/eval_harness.rs`. Composes `rurico::storage::{prepare_match_query, normalize_for_fts, rrf_merge}`, `rurico::embed::Embed`, `rurico::reranker::Rerank`, and `crate::storage::fts::clean_for_trigram`. The harness owns its own schema, indexer, and orchestration; Phase 3〜6 results on this composition predict but do not guarantee `recall` outcomes.

2. **Fixture corpus and queries shipped in-repo** — `tests/fixtures/eval/` holds documents (≥50), queries (≥140 across 7 categories at ≥20 each), and three known-answer fixtures (`identity`, `reverse`, `single_doc`). Documents are sourced from openly licensed material (`tests/fixtures/eval/LICENSES.md`). No live API access at evaluation time.

3. **Statistical contract** — every reported metric (Recall@k, MRR@k, nDCG@k for k ∈ {5, 10, 50}) carries a 95% bootstrap confidence interval over n=1000 resamples with seed=42. Per-category breakdowns are emitted for inspection but the regression gate uses **global metrics only**; breakdowns whose CI half-width exceeds `UNINFORMATIVE_HALF_WIDTH = 0.10` are flagged `uninformative` in `baseline.json`.

4. **Wiring validation via known-answer fixtures** — alongside the production fixture, three deterministic micro-fixtures verify wiring correctness:
    - `identity_ranker.jsonl` — every query points at a single doc whose body matches the query verbatim; expected `nDCG@10 = 1.0` and `Recall@1 = 1.0`.
    - `reverse_ranker.jsonl` — relevance is inverse-correlated with corpus order; pipeline output should sit near the lower bound (see *Reverse fixture lower-bound protocol* in rurico ADR 0003, retained verbatim — `observed_lower_bound × 1.05` slack absorbs mlx f32 reduction drift).
    - `single_doc.jsonl` — single-document corpus; expected `Recall@1 = MRR = 1.0`.

    These three are evaluated alongside the shuffle mutation test in `tests/eval_smoke.rs`. Together they cover query/doc mix-ups, RRF argument order, and k-cutoff off-by-one.

The reproducibility contract layers two distinct tolerances:

- **Vector-level (rurico ADR 0002, FR-010 / NFR-001)** — embedder forward output between regeneration runs: `cosine_similarity ≥ 0.99999 ∧ max_abs_diff ≤ 1e-5` per workload. Verified by `mlx_smoke verify-fixture` in rurico (workloads w1 / w2 / w3); empirically holds at `max_abs_diff = 0.0` cross-process on Apple Silicon for the embedder path.
- **Metric-level (FR-017, `eval-verify` gate)** — per-metric drift envelope set empirically (`src/bin/eval_harness.rs::MetricSpec::tolerance`), keyed by `MetricSpec` variant. Reranker forward (cross-encoder) exhibits residual cross-process f32 non-determinism inherent to Apple Silicon Metal — the noise propagates into score-sensitive metrics (`recall@5`, `ndcg@10`) while presence-sensitive metrics (`recall@10`, `mrr@10`) remain bit-identical for the current fixture. The bound is chosen ≥ 2× observed max drift over N=10 captures + historical session max so >1% regression stays detectable.

`baseline.json` carries `schema_version`, `kind`, `model_id`, `model_revision`, `mlx_rs_version`, `fixture_hash`, `aggregation`, `merge_config`, `normalization`, and `captured_with` so drift drivers are explicit. `fixture_hash` is FNV-1a 64-bit over `documents.jsonl + queries.jsonl + known_answers.jsonl`.

## Reassessment Triggers

(Inherited from rurico ADR 0003 §Reassessment Triggers — handled in this location going forward.)

- `recall/src/hybrid.rs` changes its public shape → re-evaluate whether the inline composition still tracks recall closely enough.
- mlx-rs major version bump → regenerate baseline; if `cosine_similarity < 0.99999` the vector-level tolerance must be re-evaluated.
- Cross-process metric drift exceeds the per-metric envelope on a clean run → re-characterize drift with N≥10 captures and update the affected `MetricSpec::tolerance` arm; record the new bound and the trigger condition in a dedicated PR.
- A new fixture category becomes necessary → extend via fixture authoring; bump `fixture_hash` to invalidate prior baselines.
- The reverse fixture smoke assertion (T-014) exceeds `observed_lower_bound × 1.05` on a clean run → regenerate the reverse fixture and overwrite `reverse_baseline.json` in a dedicated PR; log the cause in the PR description so the protocol stays auditable.
- amici becomes large enough that its dependents want a smaller `amici-core` without the harness → split the harness into `amici-eval` (revisit rurico ADR 0006 Option D under new constraints).

## References

- ADR 0001 (amici extraction)
- rurico ADR 0002 (GPU-side pooling embed — primitive backend reproducibility, stays in rurico)
- rurico ADR 0003 (Superseded source — methodology originally accepted in rurico; this ADR continues it in amici)
- rurico ADR 0004 (Retrieval and rerank pipeline contract for rurico)
- rurico ADR 0006 (Migration record from rurico to amici)
- amici Issue #24 (`MAX_COMBOS` guard ported into `amici/src/storage/fts.rs`, prerequisite for Issue #27)
- amici Issue #27 (this ADR's deliverable — eval-harness intake into amici)
- rurico Issue #86 (rurico-side eval-harness removal; blocked by amici#27 closing)
