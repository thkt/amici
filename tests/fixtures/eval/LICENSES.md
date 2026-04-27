# Licenses and Source Attributions

JSONL fixtures for the rurico search-quality evaluation harness (Issue #65, Phase 1b). All document and query prose is independently authored synthetic content; no upstream text is copied verbatim. Topic vocabulary and concept selection draw on publicly available documentation under the licenses listed below.

Per spec AS-005, only permissive (commercial-allowed, redistribution-allowed) licenses inform topic choice. Where a topic is best documented under share-alike or restrictive terms, the body is authored independently from public knowledge with the upstream origin acknowledged in the record's `source` field for transparency. The license enumeration in AS-005 (MIT / Apache 2.0 / CC0 / CC-BY) is read as a non-exhaustive permissive whitelist; equivalent permissive licenses (BSD, MPL 2.0, W3C Software Notice, IETF Trust, Python Software Foundation, PostgreSQL, public domain) are accepted under the same intent. ADR 0003 records this reading.

## documents.jsonl

Each row carries a `source` field naming the upstream documentation its topic vocabulary draws from. Synthetic paraphrase only — no direct copying.

### Permissive licenses (used freely)

| License                         | Example sources                                                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| MIT                             | rust-lang.org book / cargo, mlx, jq manual, pre-commit, jwt.io                                                             |
| Apache 2.0                      | docker, kubernetes, sqlite-vec, opentelemetry, terraform, typescript handbook, BERT / RAG / Sentence-BERT / BPE references |
| BSD-2 / BSD-3                   | nginx, redis (legacy), Go language and tour, PyTorch                                                                       |
| MPL 2.0                         | Terraform                                                                                                                  |
| Public domain                   | SQLite documentation                                                                                                       |
| CC-BY 3.0 / 4.0                 | 12factor.net, GitHub Actions docs, kubernetes.io, react.dev, HNSW paper, Annotated Transformer, GraphQL spec               |
| W3C Software Notice and License | CSP, CSS Grid specifications                                                                                               |
| IETF Trust License              | RFC 1035 / 6455 / 7519 / 7540 / 8446                                                                                       |
| Python Software Foundation      | docs.python.org tutorial / typing                                                                                          |
| PostgreSQL License              | postgresql.org documentation                                                                                               |

### Share-alike or restrictive sources (paraphrased independently)

The following sources use licenses that conflict with redistribution under permissive terms. Bodies for these topics are authored as original synthetic prose with the upstream source named for transparency. No upstream text is reproduced.

| License                | Source                                            |
| ---------------------- | ------------------------------------------------- |
| CC-BY-NC-SA            | git-scm.com Pro Git                               |
| CC-BY-SA 3.0           | Haskell wiki, learn-you-a-haskell                 |
| CC-BY-SA (OpenDSA)     | CLRS introduction summary materials               |
| LGPL-2.1+ (prose only) | systemd / freedesktop.org docs                    |
| SSPL (legacy Apache)   | MongoDB / Vitess (legacy Apache portions only)    |
| Vim license            | vim.org / Practical Vim                           |

## queries.jsonl

Queries are independently authored by Anthropic's Claude per the rurico project maintainer's direction. Each query references one or more documents in `documents.jsonl` via its `relevance_map` field. Query text is original synthetic prose; relevance judgments are AI-generated and reflect topic alignment between query and document.

Phase 4 of Issue #53 may introduce a second annotator pass; ADR 0003 records the reassessment trigger (kappa < 0.6 disagreement).

## known_answers.jsonl

Synthetic micro-fixtures used for harness wiring validation per spec AC-4 (identity / reverse / single_doc). Bodies are nonsensical phoneticisms ("alpha bravo charlie ...") and short prose fragments; no external attribution applies.

## Inventory check

```
$ wc -l tests/fixtures/eval/documents.jsonl tests/fixtures/eval/queries.jsonl tests/fixtures/eval/known_answers.jsonl
   60 documents.jsonl
  147 queries.jsonl
    3 known_answers.jsonl
```
