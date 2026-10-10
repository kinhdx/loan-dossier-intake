# 0002: Embedded vector store and hybrid search for policy questions

## Context

The policy corpus is a few dozen sections today and would be a few thousand in a real product catalogue.
Staff questions are short and in Vietnamese, often with an exact product term ("thẻ tín dụng", "trả nợ
trước hạn") next to loose wording.

## Decision

- Embeddings are computed locally with a multilingual sentence model through fastembed, so policy text is
  not sent to a third party at indexing time.
- Vectors are kept in a NumPy matrix saved to `var/index`. Search at this size is a single matrix product.
  `VectorStore` is the seam where a database store (pgvector, for example) replaces it; nothing else in the
  code depends on the storage.
- Search adds a BM25 score over diacritic-folded words to the cosine score, with weight 0.3.

## Consequences

- `intake eval retrieval` reports hit@1, hit@k and MRR for embedding-only search and for several lexical
  weights. On the current 30 in-scope questions the lexical term lifted hit@1 from 86.7% to 93.3% and hit@4
  from 96.7% to 100%.
- The weight was chosen on the same 30 questions, so that gain is optimistic. A held-out question set is
  needed before treating it as settled.
- An in-memory matrix does not cover concurrent writers, access control per document or incremental
  re-indexing. Those are the reasons to move to a database store, not search speed.
