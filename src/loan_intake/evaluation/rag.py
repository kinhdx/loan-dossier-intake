"""Scores retrieval and answers against eval/rag_questions.jsonl.

Each question names the policy section(s) that answer it and the key facts a correct answer must state
(each fact is a list of accepted phrasings). Out-of-scope questions name no section; the expected
behaviour is to say the policy does not cover them.

Scores:
- retrieval hit@1 / hit@k and mean reciprocal rank of the first gold section
- key-fact accuracy on in-scope questions (every fact present in the answer)
- citation accuracy: in-scope answers that cite at least one gold section
- correct refusals on out-of-scope questions, false refusals on in-scope ones
- cost and latency per question

Key-fact matching is a floor: it catches missing or wrong numbers, not every badly phrased answer.
"""
from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loan_intake.llm import ModelClient
from loan_intake.normalize import fold
from loan_intake.rag.answer import TOP_K, ask
from loan_intake.rag.index import Embedder, VectorStore

QUESTIONS = Path(__file__).resolve().parents[3] / "eval" / "rag_questions.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def facts_present(answer: str, facts: list[list[str]]) -> bool:
    text = fold(answer)
    return all(any(fold(alt) in text for alt in group) for group in facts)


def run(store: VectorStore, embedder: Embedder, client: ModelClient, model: str, workers: int = 4) -> dict:
    questions = load_questions()

    def one(q: dict) -> dict:
        a = ask(q["question"], store, embedder, client, model)
        seconds = sum(u.seconds for u in a.usage)  # model time; local embedding and search take milliseconds
        retrieved = [cid for cid, _ in a.retrieved]
        gold = set(q["gold_chunks"])
        rank = next((i + 1 for i, cid in enumerate(retrieved) if cid in gold), None)
        in_scope = bool(gold)
        return {
            "id": q["id"], "question": q["question"], "in_scope": in_scope, "answerable": a.answerable,
            "answer": a.text, "citations": a.citations, "retrieved": a.retrieved, "gold": sorted(gold),
            "rank": rank, "facts_ok": in_scope and a.answerable and facts_present(a.text, q["must_include"]),
            "cites_gold": bool(gold & set(a.citations)),
            "cost_usd": round(sum(u.cost_usd for u in a.usage), 5), "seconds": round(seconds, 2),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, questions))
    return {"model": model, "embedding_model": getattr(embedder, "model_name", ""), "top_k": TOP_K,
            "rows": rows, "summary": summarise(rows)}


def retrieval_sweep(store, embedder: Embedder, weights=(0.0, 0.1, 0.2, 0.3, 0.5, 0.8), k: int = TOP_K) -> dict:
    """Retrieval only, no model calls: hit@1, hit@k and MRR for each lexical weight (0 = embeddings only)."""
    questions = [q for q in load_questions() if q["gold_chunks"]]
    vectors = embedder.embed([q["question"] for q in questions])
    original = store.lexical_weight
    rows = []
    for w in weights:
        store.lexical_weight = w
        ranks = []
        for q, v in zip(questions, vectors, strict=True):
            hits = [c.id for c, _ in store.search(v, k, q["question"])]
            ranks.append(next((i + 1 for i, h in enumerate(hits) if h in set(q["gold_chunks"])), None))
        rows.append({"lexical_weight": w, "hit_at_1_pct": _pct(sum(r == 1 for r in ranks), len(ranks)),
                     "hit_at_k_pct": _pct(sum(r is not None for r in ranks), len(ranks)),
                     "mrr": round(statistics.mean(1 / r if r else 0 for r in ranks), 3)})
    store.lexical_weight = original
    return {"model": "retrieval", "embedding_model": getattr(embedder, "model_name", ""), "top_k": k,
            "questions": len(questions), "rows": rows}


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def summarise(rows: list[dict]) -> dict:
    inside = [r for r in rows if r["in_scope"]]
    outside = [r for r in rows if not r["in_scope"]]
    seconds = sorted(r["seconds"] for r in rows)
    return {
        "questions": len(rows), "in_scope": len(inside), "out_of_scope": len(outside),
        "hit_at_1_pct": _pct(sum(r["rank"] == 1 for r in inside), len(inside)),
        "hit_at_k_pct": _pct(sum(r["rank"] is not None for r in inside), len(inside)),
        "mrr": round(statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in inside), 3) if inside else None,
        "key_fact_accuracy_pct": _pct(sum(r["facts_ok"] for r in inside), len(inside)),
        "citation_accuracy_pct": _pct(sum(r["cites_gold"] for r in inside), len(inside)),
        "false_refusal_pct": _pct(sum(not r["answerable"] for r in inside), len(inside)),
        "correct_refusal_pct": _pct(sum(not r["answerable"] for r in outside), len(outside)),
        "cost_per_question_usd": round(statistics.mean(r["cost_usd"] for r in rows), 5) if rows else None,
        "latency_p50_s": round(statistics.median(seconds), 2) if seconds else None,
        "latency_p95_s": round(seconds[max(0, int(len(seconds) * 0.95) - 1)], 2) if seconds else None,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }
