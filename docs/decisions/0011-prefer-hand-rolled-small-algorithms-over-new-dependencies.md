# Prefer hand-rolled small algorithms over new dependencies

- Status: accepted
- Deciders: thkt
- Date: 2026-06-24
- Scope: [rust, dependencies, policy]
- Confidence: medium-high — census (2026-06-24) で 4+ サイトに分散する暗黙ポリシーとして同定、critic-design が keep 判定 (coordinated call sites >= 2 / domain-obvious gate を満たす)。各サイトのコメントは「X を避けた」と述べるが共有ポリシーが不在。

## Context and Problem Statement

amici には「小さなアルゴリズムは依存追加せず自前実装する」という判断が複数箇所に独立して現れる。`src/cli/shorthand.rs` の `osa_distance` (typo 検出の OSA 距離、`strsim` 等を引かず手書き)、`src/bin/eval_harness.rs` の `fnv1a64` fixture hash (`sha2` を意図的に回避)、`src/bin/eval_harness.rs` と `src/eval/baseline.rs` の `epoch:N` timestamp ラベル (`chrono` を回避)。

各サイトには「X を意図的に避けた」という局所コメントがあるが、これらを束ねる共有ポリシーが無い。結果として、将来 contributor が利便性のために `sha2` / `chrono` / Levenshtein crate を追加しようとした時、それが既存の判断と矛盾することに気付く拠り所が無い。4 箇所以上で coordinated に現れる domain-obvious な判断であり、毎回サイト単位で再議論されるのを防ぐ単一の forward rule が欠けている。

## Decision Drivers

- 依存ツリーの最小化: 小アルゴリズムのための重い依存 (`chrono` / `sha2`) はビルド時間・supply chain 面・rev pin 運用の負担を増やす
- 4+ サイトの分散: 同じ判断が独立コメントで散在し、共有ポリシーが無い (coordinated call sites >= 2 gate を満たす)
- re-litigation 防止: 共有 rule が無いと、依存追加の是非がサイトごとに毎回問い直される
- census 由来: 2026-06-24 census で keep #4 同定 (recurring policy without a single owner)

## Considered Options

- 本 ADR で「小アルゴリズムは hand-roll 優先」ポリシーを宣言し、判断基準を明示する
- ADR 起票せず、各サイトのコメントのみで運用を続ける
- `deny.toml` / `Cargo.toml` の lint で特定 crate を banned-dependencies に登録する

## Decision Outcome

Chosen option: 「本 ADR で hand-roll 優先ポリシーを宣言する」。

1. **ポリシー宣言**: 数十行で自前実装でき、正しさが自明な小アルゴリズム (string distance, 非暗号 hash, epoch timestamp 整形等) は、新規依存を追加せず hand-roll を優先する。
2. **判断基準**: 依存追加が正当化されるのは次のいずれか。(a) 自前実装の正しさ検証コストが高い (暗号 hash、Unicode 正規化、TZ 計算等)、(b) 実装が数十行を大きく超える、(c) 既に同等依存が依存ツリーに存在する。これらに該当しない限り hand-roll。
3. **既存サイトの記録**: 現存する hand-roll 判断 (`osa_distance` / `fnv1a64` / `epoch:N`) を本ポリシーの instance として記録し、各サイトのコメントから本 ADR を参照する (実装 follow-up)。

### Consequences

- Good: 依存追加の是非がサイト単位の毎回議論でなく、単一ポリシーで判断できる
- Good: 依存ツリーが小さく保たれ、rev pin 運用と supply chain 面の負担が抑えられる
- Bad: hand-roll はバグ混入リスクを自前で負う (テストで担保する必要がある)
- Bad: 「数十行を大きく超える」「正しさ検証コストが高い」の境界は判断依存で、グレーゾーンが残る

### Confirmation

- 新規依存を追加する PR が本 ADR の判断基準 (a)/(b)/(c) のいずれに該当するかをレビューで確認
- `osa_distance` / `fnv1a64` / `epoch_label` の各サイトに本ポリシーをカバーするユニットテストが存在することを確認 (hand-roll の正しさ担保)
- `cargo tree` で依存数が census 時点 (2026-06-24) から不必要に増えていないことを継続観測

## Pros and Cons of the Options

### 本 ADR でポリシー宣言 (chosen)

- Good: 分散する暗黙判断を single owner に集約し、re-litigation を防ぐ
- Good: 判断基準 (a)/(b)/(c) でグレーゾーンを最小化
- Bad: lint のような mechanical gate ではなく、レビュー認知に依存

### 各サイトのコメントのみ

- Good: 現状維持で最軽量
- Bad: 共有ポリシーが無く、依存追加の是非がサイトごとに毎回問い直される (census が同定した問題そのもの)

### banned-dependencies lint

- Good: 特定 crate (`chrono` / `sha2`) の追加を機械的に拒否できる
- Bad: ポリシーは「特定 crate 禁止」でなく「小アルゴリズムは hand-roll 優先」であり、crate 名の denylist では意図を表現できない
- Bad: 正当な理由 (暗号 hash が真に必要等) で `sha2` を入れたい時に denylist が過剰阻止する

## More Information

### Before / After comparison

- Before: `osa_distance` / `fnv1a64` / `epoch:N` が各々「X を避けた」コメントを持つが、判断の根拠が分散
- After: 本 ADR が「小アルゴリズムは hand-roll 優先、依存追加は (a)/(b)/(c) のみ」という共有 rule を持ち、各サイトが参照

### Trade-offs

YAGNI と Occam's Razor に従い、利便性のための投機的依存を排除する。代償として hand-roll の正しさを自前テストで担保する義務を負う。暗号強度・Unicode 正確性・TZ 計算など「自前実装の正しさ検証が高コスト」な領域は明示的に本ポリシーの対象外とし、過剰な hand-roll への逆振れを防ぐ。

### Reassessment Triggers

- 同等の小アルゴリズムが標準ライブラリに入り、hand-roll が冗長になった場合
- hand-roll した実装にバグが見つかり、検証コストが依存追加コストを上回ると判明した場合
- 暗号 hash や正確な日時計算が真に必要になり、判断基準 (a) に該当する依存追加が正当化される場合

## References

- census report `docs/audit/2026-06-24-015844-adr-gaps.md` § ADR Promotion Candidates C11
- `src/cli/shorthand.rs` (`osa_distance` — OSA 距離 hand-roll)
- `src/bin/eval_harness.rs` (`fnv1a64` fixture hash, `epoch:N` timestamp label)
- `src/eval/baseline.rs` (`epoch:N` capture-time label)
