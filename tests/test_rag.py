from __future__ import annotations

import numpy as np

from loan_intake.evaluation.rag import facts_present, load_questions
from loan_intake.policy import POLICY_DIR
from loan_intake.rag.answer import ask
from loan_intake.rag.chunking import chunk_markdown, load_corpus
from loan_intake.rag.index import MatrixStore
from loan_intake.schemas import Usage


def test_chunks_follow_sections_and_ids_are_stable():
    chunks = chunk_markdown("doc", "# Title\n\nintro\n\n## A\none\n\n## B\ntwo\n")
    assert [c.id for c in chunks] == ["doc#1", "doc#2"]
    assert chunks[1].text.startswith("Title > B")


def test_every_gold_chunk_in_the_question_set_exists():
    ids = {c.id for c in load_corpus(POLICY_DIR)}
    for q in load_questions():
        assert set(q["gold_chunks"]) <= ids, q["id"]


def test_facts_match_ignores_case_and_diacritics():
    assert facts_present("Thu nhập tối thiểu là 8.000.000 đồng", [["8.000.000", "8 triệu"]])
    assert not facts_present("Thu nhập tối thiểu là 7 triệu", [["8.000.000", "8 triệu"]])


class KeywordEmbedder:
    """Deterministic bag-of-words embedder, enough to exercise the answer path offline."""

    model_name = "keyword-test"

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def embed(self, texts):
        rows = np.array([[t.lower().count(w) for w in self.vocabulary] for t in texts], dtype=np.float32) + 1e-6
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class CitingClient:
    def __init__(self, answerable=True, cite=True):
        self.answerable, self.cite = answerable, cite

    def complete(self, *, model, system, content, schema, step, effort="low"):
        allowed = schema["properties"]["citations"]["items"]["enum"]
        data = {"answerable": self.answerable, "answer": "Lãi suất 18%/năm.",
                "citations": allowed[:1] if self.cite else []}
        return data, Usage(model=model, step=step)


def _store():
    chunks = load_corpus(POLICY_DIR)
    embedder = KeywordEmbedder(["lãi suất", "tuổi", "phí", "hồ sơ", "thu nhập", "dữ liệu"])
    return MatrixStore.build(chunks, embedder), embedder


def test_answer_cites_only_retrieved_sections():
    store, embedder = _store()
    answer = ask("Lãi suất là bao nhiêu?", store, embedder, CitingClient(), "test-model")
    assert answer.answerable
    assert set(answer.citations) <= {cid for cid, _ in answer.retrieved}


def test_answer_without_citation_is_treated_as_unanswerable():
    store, embedder = _store()
    answer = ask("Lãi suất là bao nhiêu?", store, embedder, CitingClient(cite=False), "test-model")
    assert not answer.answerable
    assert answer.citations == []
