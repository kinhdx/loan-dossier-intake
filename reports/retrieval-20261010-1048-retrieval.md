# RETRIEVAL evaluation: retrieval

Run: 20261010-1048
Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, top-k: 4, questions: 30

Lexical weight 0 is embedding search alone. The weight was chosen on this same question set, so the gain is an optimistic estimate until it is confirmed on held-out questions.

| Lexical weight | Hit@1 % | Hit@k % | MRR |
|---|---|---|---|
| 0.0 | 86.7 | 96.7 | 0.911 |
| 0.1 | 93.3 | 100.0 | 0.956 |
| 0.2 | 93.3 | 100.0 | 0.956 |
| 0.3 | 93.3 | 100.0 | 0.956 |
| 0.5 | 90.0 | 100.0 | 0.939 |
| 0.8 | 90.0 | 96.7 | 0.922 |
