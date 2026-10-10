from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from loan_intake.policy import load_thresholds
from loan_intake.schemas import DocType, ExtractedDocument, ExtractedField

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dossiers"


def load_expected(folder: Path) -> dict:
    return json.loads((folder / "expected.json").read_text(encoding="utf-8"))


def truth_documents(expected: dict, scanned: bool = True) -> list[ExtractedDocument]:
    """The values written on each document, as a perfect extractor would return them."""
    docs = []
    for name, d in expected["documents"].items():
        fields = {k: ExtractedField(v, v) for k, v in d["fields"].items()}
        docs.append(ExtractedDocument(name, DocType(d["doc_type"]), scanned, fields))
    return docs


@pytest.fixture
def limits():
    return load_thresholds()


@pytest.fixture
def as_of():
    return date(2026, 10, 1)


@pytest.fixture
def fixture_dirs():
    return sorted(p for p in FIXTURES.iterdir() if p.is_dir())
