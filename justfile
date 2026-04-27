# amici harness shortcuts. Run `just` (or `just --list`) for the menu.
#
# 検索評価ハーネス (eval_harness) ports the retrieval-quality metrics from
# rurico to amici per ADR 0002 / Issue #27, so the baseline measures the
# production wiring (`crate::storage::fts::clean_for_trigram`) directly.
#
# All `eval-*` recipes are read-only against `tests/fixtures/eval/*.json`
# unless they end in `*-baseline` / `*-reverse` (those overwrite committed
# fixtures).

default:
    @just --list

# === Development ===

# Full local CI: cargo test + clippy + fmt --check
check: test lint fmt-check

test:
    cargo test --all-features

lint:
    cargo clippy --all-targets --all-features -- -D warnings

fmt-check:
    cargo fmt -- --check

fmt:
    cargo fmt

# === 検索評価ハーネス (eval_harness, ADR 0002) ===

# Recapture identity baseline → tests/fixtures/eval/baseline.json (MLX required)
eval-baseline:
    cargo run --bin eval_harness --features eval-harness --release -- \
      capture-baseline aggregation=identity \
      output=tests/fixtures/eval/baseline.json

# Recapture reverse baseline → tests/fixtures/eval/reverse_baseline.json
eval-reverse:
    cargo run --bin eval_harness --features eval-harness --release -- \
      capture-reverse-baseline output=tests/fixtures/eval/reverse_baseline.json

# Capture a variant baseline to /tmp (agg = identity / max-chunk / dedupe / topk-average)
eval-baseline-variant agg:
    cargo run --bin eval_harness --features eval-harness --release -- \
      capture-baseline aggregation={{agg}} \
      output=/tmp/baseline-{{agg}}.json

# Capture all 4 strategies + compare-baselines markdown table
eval-compare:
    just eval-baseline-variant identity
    just eval-baseline-variant max-chunk
    just eval-baseline-variant dedupe
    just eval-baseline-variant topk-average
    cargo run --bin eval_harness --features eval-harness --release -- \
      compare-baselines paths=/tmp/baseline-identity.json,/tmp/baseline-max-chunk.json,/tmp/baseline-dedupe.json,/tmp/baseline-topk-average.json

# Evaluate (kind = full / shuffled / identity / reverse / single_doc)
eval-evaluate kind="full":
    cargo run --bin eval_harness --features eval-harness --release -- \
      evaluate kind={{kind}}

# Verify committed baseline.json against the current pipeline (FR-017 gate)
eval-verify:
    cargo run --bin eval_harness --features eval-harness --release -- \
      verify-baseline baseline=tests/fixtures/eval/baseline.json
