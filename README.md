# loan-dossier-intake

Intake for consumer loan dossiers in Vietnamese: classify each document, read its fields, decide whether the
dossier can go straight to an underwriter, and answer staff questions about the product policy. Every number
below comes from the evaluation harness in this repository, run on synthetic documents.

The model reads. Rules decide. A dossier is sent to a reviewer whenever a value cannot be grounded in the
document, two documents disagree, the ID card fails its own consistency rules, or the request is outside the
product limits.

## What it does

```
files ──► classify (small model) ──► extract fields (model under test) ──┐   one task per file, in parallel
                                                                          ▼
                                     rule checks (checks.py) ──► ready_for_underwriting | needs_review | incomplete
```

- **Documents**: loan application form, citizen ID card (CCCD), payslip. PDFs with a text layer are sent as
  text; scanned pages are sent as images, so reading the characters and extracting the fields happen in one
  call. There is no separate OCR engine.
- **Orchestration**: a LangGraph graph fans out one read task per file and fans in to the rule checks
  (`pipeline.py`).
- **Policy questions**: hybrid retrieval (multilingual embeddings computed locally, plus BM25 over
  diacritic-folded words) over the product policy, then an answer that must cite the retrieved sections or
  say the policy does not cover the question (`rag/`).
- **API**: upload a dossier, read the result, work the review queue, ask a policy question (`api.py`).
- **Journal**: tokens, cost and latency for every model call, without document content (`store.py`).

## Gating rules

| Check | Finding | Applies to |
|---|---|---|
| Every required document type present | `MISSING_DOCUMENT` → incomplete | dossier |
| Required field present, dates and amounts parse, ID has 12 digits | `MISSING_FIELD`, `FORMAT_INVALID` | all documents |
| Extracted value occurs in the document text | `EVIDENCE_NOT_FOUND` | text documents |
| Name, ID number, date of birth agree between form and card | `NAME_MISMATCH`, `ID_MISMATCH`, `DOB_MISMATCH` | form, card |
| Payslip issued to the card holder; same employer as the form | `PAYSLIP_NAME_MISMATCH`, `EMPLOYER_MISMATCH` | payslip |
| Declared income at most 10% above payslip net | `INCOME_ABOVE_PAYSLIP` | form, payslip |
| CCCD century/sex digit and birth-year digits match the holder | `ID_NUMBER_INCONSISTENT` | card |
| Card expiry falls on the holder's 25th, 40th or 60th birthday | `ID_EXPIRY_IMPLAUSIBLE` | card |
| Card not expired | `ID_EXPIRED` | card |
| Age, amount, term, income multiple, minimum income, debt-to-income | `AGE_OUT_OF_RANGE`, `AMOUNT_OUT_OF_RANGE`, `AMOUNT_ABOVE_INCOME_MULTIPLE`, `TERM_OUT_OF_RANGE`, `INCOME_BELOW_MINIMUM`, `DTI_ABOVE_LIMIT` | form, payslip |

Scanned documents cannot be grounded in a text layer. They are covered by the cross-document rules and the
card's own number and expiry rules. Thresholds live in `policy/thresholds.json`; the same numbers are written
in prose in `policy/*.md`, which is what the question-answering index is built from. Rationale:
[docs/decisions/0001](docs/decisions/0001-rules-decide-the-route.md).

## Results

All runs on 10/10/2026. Classification uses `claude-haiku-5-5` in every run; the column names the extraction
model. Reports: `reports/idp-20261010-1100-*`, `-1101-*`, `-1102-*`, `reports/rag-20261010-*`,
`reports/retrieval-20261010-1048-*`.

**Document intake, 40 synthetic dossiers.** The standard set (30) has clean scans and text PDFs in two
layouts. The stress set (10) has 100 dpi blurred ID card scans, scanned forms with a struck-through and
corrected amount, and payslips with the year-to-date column before the monthly one. Opus was run on the stress
set only, because Haiku and Sonnet already score 100% on the standard set.

| Standard set (30) | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|
| Decision accuracy | 100% | 100% |
| Planted defects caught | 100% | 100% |
| Clean dossiers sent to review | 0% | 0% |
| Passed as ready with a misread field | 0 | 0 |
| Field accuracy, text / scanned | 100% / 100% | 100% / 100% |
| Cost per dossier | $0.0013 | $0.0185 |
| Latency p50 / p95 | 4.2 s / 6.6 s | 4.5 s / 8.2 s |

| Stress set (10) | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| Decision accuracy | 50% | 70% | 70% |
| Planted defects caught | 100% | 100% | 100% |
| Clean dossiers sent to review (of 5) | 100% | 60% | 60% |
| Passed as ready with a misread field | 0 | 0 | 0 |
| Field accuracy, text / scanned | 100% / 83.6% | 100% / 92.9% | 100% / 92.1% |
| Cost per dossier | $0.0024 | $0.0240 | $0.0481 |
| Latency p50 / p95 | 8.5 s / 9.0 s | 7.6 s / 8.6 s | 10.0 s / 11.6 s |

Latency is model time on the critical path; the three files of a dossier are read in parallel.

What the numbers say:

- On text documents and clean scans the smallest model is as accurate as the others at a fourteenth of
  Sonnet's cost per dossier.
- On degraded scans the larger models read more fields correctly, but all three read the name "THÙY" as
  "THÚY" or "THỦY" on blurred cards, and all three misread some dates of birth. None of those dossiers passed
  as ready: the misread value contradicted the form, the ID number digits or the expiry rule, and the rules
  sent the dossier to review.
- Opus costs twice as much as Sonnet on the stress set with no gain in decision accuracy.

**What the evaluation changed.** Two rules exist because an earlier run showed a misread passing as ready:

- `ID_EXPIRY_IMPLAUSIBLE`: Haiku read a blurred expiry date 10/01/2044 as 10/01/2046, and nothing
  contradicted it. A citizen ID card expires on the holder's 25th, 40th or 60th birthday, so any other date is a
  misread.
- `NAME_DIACRITICS_DIFFER`: Sonnet and Opus read "THÙY" as "THÚY" on two clean stress dossiers, and the
  name check ignored diacritics. Names written with diacritics on both documents must now match them exactly;
  a form typed without diacritics is still accepted.

Both rules trade more reviews on poor scans for zero silent errors in this set.

**Policy questions, 36 questions** (30 answerable from the policy, 6 out of scope), hybrid retrieval, top 4.

| | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| Gold section ranked first / in top 4 | 93.3% / 100% | 93.3% / 100% | 93.3% / 100% |
| Answers stating every key fact | 100% | 100% | 100% |
| Answers citing a gold section | 100% | 100% | 100% |
| Out-of-scope questions declined | 6 / 6 | 6 / 6 | 6 / 6 |
| In-scope questions wrongly declined | 0 | 0 | 0 |
| Cost per question | $0.00018 | $0.0038 | $0.0083 |
| Latency p50 | 2.4 s | 2.2 s | 2.8 s |

On a corpus this small and well structured, the answer model makes no difference and retrieval is what
matters. Retrieval with and without the lexical term (`intake eval retrieval`, no model calls):

| Lexical weight | Hit@1 | Hit@4 | MRR |
|---|---|---|---|
| 0 (embeddings only) | 86.7% | 96.7% | 0.911 |
| 0.3 (used) | 93.3% | 100% | 0.956 |

The weight was chosen on the same questions, so the gain is optimistic until checked on held-out questions.

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Put an Anthropic API key in `.env`
(see `.env.example`).

```bash
uv sync
uv run pytest -q                                  # offline tests, no API key needed
uv run intake process fixtures/dossiers/D001      # one dossier
uv run intake index                               # build the policy index (downloads a ~220 MB embedding model once)
uv run intake ask "Tất toán ở tháng thứ 8 thì mất phí bao nhiêu?"
uv run intake eval idp --model claude-haiku-5-5   # writes reports/idp-*.md and .json; responses cached in var/cache
uv run intake eval rag --model claude-haiku-5-5
uv run intake eval retrieval                      # retrieval only, no model calls
uv run intake serve                               # API on http://127.0.0.1:8000/docs
uv run python fixtures/generate.py                # regenerate the synthetic dossiers
```

## Layout

```
src/loan_intake/
  documents.py     text layer or page image per page
  extract.py       classification and field extraction prompts and schemas
  checks.py        gating rules
  pipeline.py      LangGraph fan-out / fan-in
  rag/             chunking, embeddings + BM25 search, cited answers
  evaluation/      IDP and RAG scoring, reports
  api.py, cli.py, store.py, llm.py
policy/            sample product policy (fictional) and thresholds
fixtures/          dossier generator and the generated dossiers with expected.json
eval/              policy question set with gold sections and key facts
reports/           evaluation reports referenced above
docs/decisions/    design decisions
```

## What it does not do

- It does not make credit decisions. `ready_for_underwriting` means the data is complete and consistent, not
  that the loan should be granted.
- It has only seen synthetic documents with three layouts. Real scans (glare, folds, handwriting, the back of
  the card) will lower every number above.
- No authentication or per-user authorisation on the API; it binds to localhost.
- The vector store is an in-process matrix. A database-backed store is the next step for concurrent writers
  and incremental updates ([docs/decisions/0002](docs/decisions/0002-embedded-store-and-hybrid-search.md)).
- Fields are read from the front of the ID card only.

## Data

Every person, employer, ID number and policy figure in this repository is invented. ID numbers are masked in
API listings and the model-call journal stores no document content.
