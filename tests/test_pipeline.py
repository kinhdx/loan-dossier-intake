from __future__ import annotations

import base64
import threading
from datetime import date

from conftest import load_expected

from loan_intake.documents import load_dir
from loan_intake.evaluation.idp import score_dossier
from loan_intake.pipeline import Settings, process
from loan_intake.schemas import Usage


class FixtureClient:
    """Plays back the values written on each fixture document, without network calls.

    The document is identified from the content it is sent: the first text line for text documents,
    an image block for the scanned ID card. Evidence is the line of the document that holds the value,
    as a well-behaved model would quote it.
    """

    def __init__(self, folder):
        self.expected = load_expected(folder)
        self.sources = {d.name: d for d in load_dir(folder)}
        self.lock = threading.Lock()
        self.steps: list[str] = []

    def _document(self, content: list[dict]) -> str:
        first = content[0]
        if first["type"] == "image":
            data = base64.b64decode(first["source"]["data"])
            return next(name for name, d in self.sources.items() if d.scanned and d.pages[0].image_png == data)
        body = first["text"].split("\n", 1)[1]
        return next(name for name, d in self.sources.items() if not d.scanned and d.text.startswith(body[:60]))

    def complete(self, *, model, system, content, schema, step, effort="low"):
        name = self._document(content)
        truth = self.expected["documents"][name]
        with self.lock:
            self.steps.append(step)
        usage = Usage(model=model, step=step, input_tokens=1000, output_tokens=100, cost_usd=0.002, seconds=0.01)
        if step == "classify":
            return {"doc_type": truth["doc_type"]}, usage
        lines = self.sources[name].text.splitlines()
        out = {}
        for field in schema["properties"]:
            value = truth["fields"].get(field, "")
            evidence = next((line for line in lines if value and value in line), value)
            out[field] = {"value": value, "evidence": evidence}
        return out, usage


def test_pipeline_reads_every_document_and_reaches_expected_decision(fixture_dirs):
    for folder in fixture_dirs:
        client = FixtureClient(folder)
        expected = client.expected
        result = process(folder.name, load_dir(folder), Settings(client), as_of=date.fromisoformat(expected["as_of"]))
        score = score_dossier(result, expected)
        assert score["decision_ok"], (folder.name, score["findings"])
        assert sorted(score["findings"]) == sorted(expected["expected_findings"]), folder.name
        assert client.steps.count("classify") == len(expected["documents"])
        assert score["fields_ok"] == score["fields_total"]
