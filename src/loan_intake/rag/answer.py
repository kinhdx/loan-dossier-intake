"""Answers staff questions about the loan product from the retrieved policy sections only.

The answer schema forces the model to say whether the retrieved sections answer the question and to cite
them by ID. Citations are restricted to the IDs that were retrieved, and an answer marked answerable
with no citation is downgraded to unanswerable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from loan_intake.llm import ModelClient, ModelError
from loan_intake.rag.index import Embedder, VectorStore
from loan_intake.schemas import Usage

TOP_K = 4

SYSTEM = """You answer questions from bank staff about a consumer loan product, in Vietnamese.
Use only the policy sections provided. If they do not contain the answer, set answerable to false and say
briefly that the policy does not cover it; do not answer from general knowledge.
Keep amounts, percentages and time limits exactly as the policy states them. Cite every section you used."""


@dataclass
class Answer:
    question: str
    answerable: bool
    text: str
    citations: list[str]
    retrieved: list[tuple[str, float]]
    usage: list[Usage] = field(default_factory=list)


def answer_schema(chunk_ids: list[str]) -> dict:
    return {"type": "object",
            "properties": {"answerable": {"type": "boolean"},
                           "answer": {"type": "string"},
                           "citations": {"type": "array", "items": {"type": "string", "enum": chunk_ids}}},
            "required": ["answerable", "answer", "citations"], "additionalProperties": False}


def ask(question: str, store: VectorStore, embedder: Embedder, client: ModelClient, model: str,
        k: int = TOP_K) -> Answer:
    hits = store.search(embedder.embed([question])[0], k, question)
    retrieved = [(c.id, round(score, 3)) for c, score in hits]
    sections = "\n\n".join(f"[{c.id}]\n{c.text}" for c, _ in hits)
    content = [{"type": "text", "text": f"Policy sections:\n\n{sections}\n\nQuestion: {question}"}]
    try:
        data, usage = client.complete(model=model, system=SYSTEM, content=content,
                                      schema=answer_schema([c.id for c, _ in hits]), step="answer")
    except ModelError as e:
        return Answer(question, False, f"error: {e}", [], retrieved)
    citations = list(dict.fromkeys(data["citations"]))
    answerable = bool(data["answerable"]) and bool(citations)
    return Answer(question, answerable, data["answer"].strip(), citations if answerable else [], retrieved, [usage])
