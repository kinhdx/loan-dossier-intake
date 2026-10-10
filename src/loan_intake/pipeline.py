"""The intake workflow as a LangGraph graph.

    START --fan-out--> read_document (one task per file, run in parallel) --fan-in--> check --> END

Each read_document task classifies and extracts one file. `check` runs once all files are read: it applies
the deterministic rules in checks.py and sets the dossier decision.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from loan_intake.checks import decide, run_checks
from loan_intake.extract import read_document
from loan_intake.llm import ModelClient
from loan_intake.policy import Thresholds, load_thresholds
from loan_intake.schemas import DossierResult, ExtractedDocument, Finding, SourceDocument

DEFAULT_EXTRACT_MODEL = "claude-opus-5-5"
DEFAULT_CLASSIFY_MODEL = "claude-haiku-5-5"


@dataclass
class Settings:
    client: ModelClient
    extract_model: str = DEFAULT_EXTRACT_MODEL
    classify_model: str = DEFAULT_CLASSIFY_MODEL
    limits: Thresholds | None = None


class DossierState(TypedDict, total=False):
    dossier_id: str
    as_of: date
    sources: list[SourceDocument]
    extracted: Annotated[list[ExtractedDocument], operator.add]
    findings: list[Finding]
    metrics: dict


class ReadTask(TypedDict):
    source: SourceDocument


def build_graph(settings: Settings):
    limits = settings.limits or load_thresholds()

    def fan_out(state: DossierState) -> list[Send]:
        return [Send("read_document", {"source": s}) for s in state["sources"]]

    def read(task: ReadTask) -> dict:
        doc = read_document(task["source"], settings.client, settings.extract_model, settings.classify_model)
        return {"extracted": [doc]}

    def check(state: DossierState) -> dict:
        docs = sorted(state.get("extracted", []), key=lambda d: d.name)
        findings, metrics = run_checks(docs, limits, state["as_of"])
        return {"findings": findings, "metrics": metrics}

    graph = StateGraph(DossierState)
    graph.add_node("read_document", read)
    graph.add_node("check", check)
    graph.add_conditional_edges(START, fan_out, ["read_document"])
    graph.add_edge("read_document", "check")
    graph.add_edge("check", END)
    return graph.compile()


def process(dossier_id: str, sources: list[SourceDocument], settings: Settings, as_of: date | None = None,
            graph=None) -> DossierResult:
    graph = graph or build_graph(settings)
    state = graph.invoke({"dossier_id": dossier_id, "as_of": as_of or date.today(), "sources": sources})
    docs = sorted(state["extracted"], key=lambda d: d.name)
    return DossierResult(dossier_id=dossier_id, decision=decide(state["findings"]), documents=docs,
                         findings=state["findings"], metrics=state["metrics"])
