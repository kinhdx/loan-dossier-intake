"""Runs the intake pipeline over the synthetic dossiers and scores it against fixtures/dossiers/*/expected.json.

Scores:
- decision accuracy: dossier routed to the expected outcome (ready / needs review / incomplete)
- defect recall: planted defects that produced the expected finding code
- false review rate: clean dossiers that were sent to review anyway
- ready with wrong data: dossiers passed as ready although a field was misread (the silent-error count;
  the gating rules exist to keep this at zero)
- field accuracy: extracted values equal to the values written on the document, after normalisation,
  reported separately for text documents and scanned ID cards
- cost per dossier, and latency as model time on the critical path (files are read in parallel)

Scores are reported for all dossiers and separately for the standard (D*) and stress (H*) sets.
"""
from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from loan_intake.checks import AMOUNT_FIELDS, DATE_FIELDS
from loan_intake.documents import load_dir
from loan_intake.normalize import digits_only, parse_amount, parse_date, parse_term_months, squash
from loan_intake.pipeline import Settings, build_graph, process
from loan_intake.schemas import DossierResult

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "dossiers"


def same_value(field: str, got: str, want: str) -> bool:
    if field in DATE_FIELDS:
        return parse_date(got) is not None and parse_date(got) == parse_date(want)
    if field in AMOUNT_FIELDS:
        return parse_amount(got) is not None and parse_amount(got) == parse_amount(want)
    if field in {"id_number", "phone"}:
        return digits_only(got) == digits_only(want)
    if field == "term_months":
        return parse_term_months(got) is not None and parse_term_months(got) == parse_term_months(want)
    # Names and employers: case-insensitive but diacritics count, so "Năm" read as "Nam" is a miss.
    return squash(got).casefold() == squash(want).casefold()


def score_dossier(result: DossierResult, expected: dict) -> dict:
    codes = {f.code for f in result.findings if f.severity.value != "info"}
    want = set(expected["expected_findings"])
    fields_total = {"text": 0, "scanned": 0}
    fields_ok = {"text": 0, "scanned": 0}
    misses, classify_ok = [], 0
    for doc in result.documents:
        truth = expected["documents"].get(doc.name)
        if truth is None:
            continue
        classify_ok += doc.doc_type.value == truth["doc_type"]
        kind = "scanned" if doc.scanned else "text"
        for field, value in truth["fields"].items():
            fields_total[kind] += 1
            if same_value(field, doc.get(field), value):
                fields_ok[kind] += 1
            else:
                misses.append({"document": doc.name, "field": field, "expected": value, "got": doc.get(field)})
    return {
        "dossier": result.dossier_id, "defect": expected["defect"],
        "expected_decision": expected["expected_decision"], "decision": result.decision.value,
        "decision_ok": result.decision.value == expected["expected_decision"],
        "expected_findings": sorted(want), "findings": sorted(codes),
        "defect_found": bool(want) and want <= codes, "unexpected_findings": sorted(codes - want),
        "documents": len(expected["documents"]), "classified_ok": classify_ok,
        "fields_total": fields_total, "fields_ok": fields_ok, "field_misses": misses,
        "cost_usd": round(sum(u.cost_usd for u in result.usage), 5),
        # Model time on the critical path: files are read in parallel, so the slowest file sets the latency.
        # Taken from the recorded call durations, so a run served from the response cache reports the same value.
        "seconds": round(max((sum(u.seconds for u in d.usage) for d in result.documents), default=0.0), 2),
        "tokens_in": sum(u.input_tokens for u in result.usage), "tokens_out": sum(u.output_tokens for u in result.usage),
    }


def run(settings: Settings, limit: int | None = None, workers: int = 4, subset: str | None = None) -> dict:
    prefix = {"standard": "D", "stress": "H"}.get(subset or "", "")
    folders = sorted(p for p in FIXTURES.iterdir() if p.is_dir() and p.name.startswith(prefix))[:limit]
    graph = build_graph(settings)

    def one(folder: Path) -> dict:
        expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
        result = process(folder.name, load_dir(folder), settings, date.fromisoformat(expected["as_of"]), graph)
        return score_dossier(result, expected)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, folders))
    subsets = {"all": rows, "standard": [r for r in rows if r["dossier"].startswith("D")],
               "stress": [r for r in rows if r["dossier"].startswith("H")]}
    return {"model": settings.extract_model, "classify_model": settings.classify_model, "rows": rows,
            "summary": {name: summarise(subset) for name, subset in subsets.items() if subset}}


def _pct(n: int, d: int) -> float | None:
    return round(100 * n / d, 1) if d else None


def summarise(rows: list[dict]) -> dict:
    clean = [r for r in rows if r["defect"] == "clean"]
    defects = [r for r in rows if r["defect"] != "clean"]
    seconds = sorted(r["seconds"] for r in rows)
    text_total = sum(r["fields_total"]["text"] for r in rows)
    scan_total = sum(r["fields_total"]["scanned"] for r in rows)
    return {
        "dossiers": len(rows),
        "decision_accuracy_pct": _pct(sum(r["decision_ok"] for r in rows), len(rows)),
        "defect_recall_pct": _pct(sum(r["defect_found"] for r in defects), len(defects)),
        "false_review_rate_pct": _pct(sum(r["decision"] != "ready_for_underwriting" for r in clean), len(clean)),
        "ready_with_wrong_data": sum(r["decision"] == "ready_for_underwriting" and bool(r["field_misses"])
                                    for r in rows),
        "classification_accuracy_pct": _pct(sum(r["classified_ok"] for r in rows), sum(r["documents"] for r in rows)),
        "field_accuracy_text_pct": _pct(sum(r["fields_ok"]["text"] for r in rows), text_total),
        "field_accuracy_scanned_pct": _pct(sum(r["fields_ok"]["scanned"] for r in rows), scan_total),
        "cost_per_dossier_usd": round(statistics.mean(r["cost_usd"] for r in rows), 4) if rows else None,
        "latency_p50_s": round(statistics.median(seconds), 2) if seconds else None,
        "latency_p95_s": round(seconds[max(0, int(len(seconds) * 0.95) - 1)], 2) if seconds else None,
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }
