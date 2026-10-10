"""Classifies one document and reads its fields.

Two calls per document. Classification is a short, closed-choice task and runs on a small model by
default; extraction runs on the model under evaluation. Text-layer documents are sent as text, scanned
documents as page images.
"""
from __future__ import annotations

import base64

from loan_intake.llm import ModelClient, ModelError
from loan_intake.schemas import FIELDS, DocType, ExtractedDocument, ExtractedField, SourceDocument

CLASSIFY_SYSTEM = """You sort documents submitted with consumer loan applications in Vietnam.
Answer with the document type:
- application_form: the loan application form (Đơn đề nghị vay vốn) filled in by the applicant
- national_id: a citizen identity card (Căn cước công dân / CCCD)
- payslip: a monthly payslip or salary statement (Phiếu lương / Bảng lương)
- unknown: anything else"""

EXTRACT_SYSTEM = """You copy fields from a Vietnamese loan document into a structured record.

For every field return:
- value: the value exactly as written, including dots in amounts and the date format used on the document.
  Do not convert units, compute totals or correct spelling.
- evidence: the shortest exact span of the document that contains the value, usually the label and the value
  on that line, copied character for character.
If a field does not appear on the document, return empty strings for both. Never fill a field from another
field or from general knowledge."""


def page_content(doc: SourceDocument) -> list[dict]:
    content: list[dict] = []
    for page in doc.pages:
        if page.image_png is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.b64encode(page.image_png).decode("ascii")}})
        else:
            content.append({"type": "text", "text": f"--- page {page.number} ---\n{page.text}"})
    return content


def classify_schema() -> dict:
    return {"type": "object", "properties": {"doc_type": {"type": "string", "enum": [t.value for t in DocType]}},
            "required": ["doc_type"], "additionalProperties": False}


def extract_schema(doc_type: DocType) -> dict:
    field = {"type": "object", "properties": {"value": {"type": "string"}, "evidence": {"type": "string"}},
             "required": ["value", "evidence"], "additionalProperties": False}
    props = {name: {**field, "description": desc} for name, desc in FIELDS[doc_type].items()}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def read_document(doc: SourceDocument, client: ModelClient, extract_model: str, classify_model: str
                  ) -> ExtractedDocument:
    result = ExtractedDocument(name=doc.name, doc_type=DocType.UNKNOWN, scanned=doc.scanned, source_text=doc.text)
    content = page_content(doc)
    try:
        data, usage = client.complete(model=classify_model, system=CLASSIFY_SYSTEM,
                                      content=content + [{"type": "text", "text": "Which document type is this?"}],
                                      schema=classify_schema(), step="classify")
        result.usage.append(usage)
        result.doc_type = DocType(data["doc_type"])
        if result.doc_type == DocType.UNKNOWN:
            return result
        data, usage = client.complete(model=extract_model, system=EXTRACT_SYSTEM,
                                      content=content + [{"type": "text",
                                                          "text": f"Document type: {result.doc_type}. Copy its fields."}],
                                      schema=extract_schema(result.doc_type), step="extract")
        result.usage.append(usage)
        result.fields = {k: ExtractedField(v["value"], v["evidence"]) for k, v in data.items()}
    except ModelError as e:
        result.error = str(e)
    return result
