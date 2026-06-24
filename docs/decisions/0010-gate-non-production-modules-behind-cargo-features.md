# Gate non-production modules behind cargo features

- Status: accepted
- Deciders: thkt
- Date: 2026-06-24
- Scope: [rust, crate-boundary, build]
- Confidence: high — census (2026-06-24) で reviewer が incomplete-contract として同定、critic-design が keep 判定。OUTCOME の library-only-except-eval 制約を cfg が encode するが forward rule がコードに無いことを lib.rs 読解で裏付け。

## Context and Problem Statement

`src/lib.rs` は amici の公開モジュール構成を決める crate root だが、module-doc が一切無い。`eval` は `#[cfg(feature = "eval-harness")]`、`testing` は `#[cfg(any(test, feature = "test-support"))]` で gate され、`cli` / `logging` / `migration` / `model` / `storage` は無条件公開という構造は、OUTCOME の「amici は `eval_harness` バイナリを除きライブラリクレートに留まる」「testing items は downstream production binary に入らない」制約を encode している。

しかし lib.rs にはこの gating の理由も、新モジュール追加時のルールも書かれていない。cfg 属性は既存モジュールの statement-of-fact に留まり、「downstream production build に漏れてはならない非production モジュールは feature gate を必須とする」という forward-looking rule が無い。将来 contributor が `pub mod` を足す際、それを production に晒すか gate するかの判断基準がコードに存在しない。`testing.rs` は自モジュールの境界を述べるが、crate 全体の no-leak ポリシーは誰も持たない。

## Decision Drivers

- downstream 漏洩の防止: eval / testing 系コードが下流 production binary の依存・コンパイル対象に入ると、MLX (Apple Silicon 必須) や test-support 依存が下流ビルドを汚染する
- forward rule の不在: 既存 cfg は statement-of-fact で、新モジュール追加時の判断基準を誰も持たない
- OUTCOME 制約の encode: library-only-except-eval は OUTCOME Constraints に明記されるが、コード側に対応する gate ルールが無い
- census 由来: 2026-06-24 census で keep #3 同定 (lib.rs の module-doc ゼロ = incomplete-contract)

## Considered Options

- 本 ADR で feature-gating ポリシーを宣言し、lib.rs に module-doc を追加する
- 非production モジュールを別 crate (amici-eval / amici-testing) に物理分離する
- ADR 起票せず、lib.rs にコメントのみ追加する

## Decision Outcome

Chosen option: 「本 ADR で feature-gating ポリシーを宣言する」。

1. **gating ポリシー宣言**: downstream production build に入ってはならないモジュールは cargo feature で gate する。現状の割当を契約として pin する: `eval` → `eval-harness` (MLX 依存、Apple Silicon 専用)、`testing` → `test(any(test, feature = "test-support"))`。`cli` / `logging` / `migration` / `model` / `storage` は無条件公開の production API。
2. **新モジュールの判断基準**: `pub mod` 追加時、それが (a) 下流 production が常時必要とするか、(b) eval / test / Apple-Silicon 専用かを判定し、後者は対応 feature で gate する。default feature には非production モジュールを入れない。
3. **module-doc マーカー** (実装 follow-up): `src/lib.rs` に本 ADR を参照する module-doc を追加し、各 gate の理由 (MLX 依存 / test 専用) を 1 行ずつ記す。

### Consequences

- Good: 新モジュール追加時に gating 判断の基準が文書化され、漏洩を未然に防ぐ
- Good: 下流 production build が MLX / test-support 依存を引かない状態を契約として保てる
- Bad: 物理 crate 分離ほど強い境界ではなく、feature 設定ミスはコンパイルが通ってしまう
- Bad: feature flag の組合せ (`--all-features` 等) でのビルド検証を CI で別途担保する必要がある

### Confirmation

- `cargo build` (default features) で `eval` / `testing` がコンパイル対象に入らないことを確認
- 下流 repo が amici を引く際、`default-features = false` または production feature のみ指定で MLX 依存を引かないことを確認
- CI が default / `--all-features` の両方でビルドを回し、feature gate の整合を担保 (Linux/Windows では eval-harness 非対象 = OUTCOME Constraints)

## Pros and Cons of the Options

### 本 ADR でポリシー宣言 + lib.rs module-doc (chosen)

- Good: 既存構造を変えずに forward rule を追加でき、低コスト
- Good: 新モジュール追加の判断基準を一意の文書に集約
- Bad: feature 設定ミスはコンパイル成功してしまう (mechanical gate ではない)

### 別 crate に物理分離

- Good: 漏洩を crate 境界で機械的に不可能にする最強の境界
- Bad: amici-eval / amici-testing への分割は workspace 再編コストが大きく、現状 1 crate の単純さを失う
- Bad: testing helper の循環依存 (amici 本体を test するのに amici-testing が要る) を招きうる

### コメントのみ

- Good: 最軽量
- Bad: lib.rs 単独のコメントは「新モジュールは gate せよ」という crate 全体ポリシーを encode する場として弱く、OUTCOME 制約との接続が消える

## More Information

### Trade-offs

feature gate は物理 crate 分離より弱い境界だが、現状 amici は単一 crate の単純さ (Occam's Razor) を outcome 上の価値として持つ。本 ADR は crate を割らずに no-leak ポリシーを文書化する中間解を採り、漏洩リスクが顕在化した時点で物理分離を Reassessment Trigger に回す。

### Reassessment Triggers

- eval / testing の依存が肥大化し、feature gate ミスが実際に下流 build を壊した場合 (物理 crate 分離を再評価)
- `cli` / `model` / `storage` のいずれかが Apple-Silicon 専用依存を持ち込み、無条件公開の前提が崩れた場合
- 下流が amici の一部モジュールのみを欲し、より細かい feature 分割が必要になった場合

## References

- ADR-0001: Extract Shared Model-Loading and CLI Utilities into amici Crate (crate 境界の先例)
- census report `docs/audit/2026-06-24-015844-adr-gaps.md` § ADR Promotion Candidates C7
- `src/lib.rs` (モジュール構成 + cfg gate)
- `src/testing.rs` (test-support 境界)
- `.claude/OUTCOME.md` § Constraints (library-only-except-eval, MLX = Apple Silicon 専用)
