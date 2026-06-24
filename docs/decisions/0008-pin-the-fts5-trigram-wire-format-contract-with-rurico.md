# Pin the FTS5 trigram wire-format contract with rurico

- Status: accepted
- Deciders: thkt
- Date: 2026-06-24
- Scope: [rust, storage, cross-repo]
- Confidence: high — census (2026-06-24) で reviewer が incomplete-contract pattern として同定、critic-design が keep 判定。amici 側のみで強制できない cross-repo invariant であることをコード読解 + test 確認で裏付け。

## Context and Problem Statement

`src/storage/fts.rs` の `parse_fts_segments` は `rurico::storage::MatchFtsQuery` が emit する MATCH 文字列を、`" OR "` 区切りと `"..."` quoting という具体的な wire-format 前提でパースする。module-doc は「`MatchFtsQuery` output のアダプタ」と述べるが、producer (rurico) と consumer (amici) のどちらにも「この serialization 形式は両者で同期し続けねばならない」という forward-looking rule が無い。

rurico 側が separator や quoting を変えても、rurico のコンパイラもテストも壊れない。amici 側の `fts/tests.rs` は amici 自身の出力形を pin するだけで、rurico を producer-of-truth として参照しない。結果として rurico の format 変更は amici のパースを silent に壊し、どちらの repo にも failing test が出ない。型でも lint でも守れない cross-repo invariant である。

## Decision Drivers

- silent breakage 阻止: rurico 側 format 変更が amici で no-failing-test のまま壊れる
- mechanical 不可能性: 両 repo のコンパイラ・lint・単体テストいずれも producer/consumer 同期を保証できない
- rev pin 運用: 配布は `git rev` pin のみ (OUTCOME Constraints) のため、format drift は rev bump 時にしか顕在化せず、検出点が一意でない
- census 由来: 2026-06-24 census で reviewer が incomplete-contract として keep #1 同定

## Considered Options

- 本 ADR で wire-format 契約を明文化し、両側にコメントマーカーを置く
- rurico に format を返す getter / typed API を生やし、amici が文字列パースをやめる
- amici 側にラウンドトリップ test (rurico producer → amici parser) を追加するのみ
- ADR 起票せず、fts.rs にコメント追記のみ

## Decision Outcome

Chosen option: 「本 ADR で wire-format 契約を明文化する」。

1. **契約宣言**: `MatchFtsQuery` の MATCH 出力形式 (`" OR "` 区切り、fixed term は `"..."` quoting、OR-group は `( ... )`) を amici↔rurico 間の wire-format 契約と宣言する。rurico 側の serialization 変更は本契約の変更であり、amici の `parse_fts_segments` 追従を伴う coordinated change を必須とする。
2. **コードマーカー** (実装 follow-up): `src/storage/fts.rs` の module-doc と `parse_fts_segments` に本 ADR を参照するコメントを追加し、rurico 側の該当 serialization 箇所にも相互参照を残す。
3. **ラウンドトリップ test** (実装 follow-up): rurico の `MatchFtsQuery` を producer として通し、amici の `parse_fts_segments` が期待 segment を復元することを確認する統合テストを追加する。これにより rurico rev bump 時に format drift が test failure として顕在化する。

### Consequences

- Good: rurico の format 変更が amici の rev bump 時にレビュー gate / test failure として顕在化する
- Good: producer/consumer の同期義務が文書化され、将来の contributor が片側だけ変えるのを防ぐ
- Bad: amici と rurico の二 repo を跨ぐ契約のため、片側 repo のレビューだけでは閉じない
- Bad: ラウンドトリップ test が land するまでは、契約はレビュー認知に依存する

### Confirmation

- `git grep "MatchFtsQuery" src/storage/fts.rs` で結合点が `fts.rs` に局在することを継続確認
- 上記 3 のラウンドトリップ test が land 後は、rurico rev bump PR の CI が format drift を自動検出
- rurico rev bump 時のレビューチェックリストに「`MatchFtsQuery` serialization 変更の有無確認」を組み込む

## Pros and Cons of the Options

### 本 ADR で wire-format 契約を明文化 (chosen)

- Good: 両 repo に相互参照マーカーを置け、同期義務が一意の文書に集約される
- Good: ラウンドトリップ test と組み合わせれば mechanical gate へ昇格できる
- Bad: 契約自体は cross-repo で、片側コンパイラでは強制不能

### rurico に typed API を生やし文字列パースをやめる

- Good: wire-format 結合そのものを消せる (最も根本的)
- Bad: rurico の公開 API 変更で、amici 以外の consumer 影響も評価が必要
- Bad: amici は rurico の下流であり、上流 API 設計を amici 都合で駆動するのは責務逆転

### ラウンドトリップ test 追加のみ

- Good: 軽量で drift を test で捉えられる
- Bad: なぜその形式なのかという契約の所在が文書化されず、test が消えると invariant も消える

### コメント追記のみ

- Good: 最軽量
- Bad: 片側コメントは cross-repo 同期義務を相手 repo の contributor に強制できない

## More Information

### Trade-offs

amici は rurico primitives の上に積む薄い composition 層 (ADR-0005) であり、rurico-internal な出力形式に依存する seam がいくつか存在する。本 ADR は fts wire-format をその代表として明文化するが、他 seam (probe sequencing 等) は別途各 ADR / コメントが扱う。代償として「fts.rs を読むだけでは契約が閉じず、rurico 側 serialization も併読が必要」な読みコストを許容する。

### Reassessment Triggers

- rurico が `MatchFtsQuery` に typed accessor を導入し、文字列パースが不要になった場合
- FTS5 trigram tokenizer 以外の tokenizer を採用し、`parse_fts_segments` の前提が変わった場合
- amici が rurico への依存をやめ、自前で MATCH 文字列を構築するようになった場合

## References

- ADR-0005: Place Model Wiring in rurico, Keep amici as Thin Composition Base (amici↔rurico 境界の先例)
- census report `docs/audit/2026-06-24-015844-adr-gaps.md` § ADR Promotion Candidates C1
- `src/storage/fts.rs` (`parse_fts_segments`, `MatchFtsQuery` アダプタ)
- `rurico::storage::MatchFtsQuery` (producer 側 serialization)
