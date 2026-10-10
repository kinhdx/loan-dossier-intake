"""Document types, the fields read from each, and the result objects passed through the pipeline.

Every extracted field carries the value and the text it was read from ("evidence"). On documents with a
text layer the evidence is checked against that text; on scanned documents it cannot be, so those fields
rely on cross-document agreement instead (see checks.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DocType(StrEnum):
    APPLICATION_FORM = "application_form"
    NATIONAL_ID = "national_id"
    PAYSLIP = "payslip"
    UNKNOWN = "unknown"


# Field name -> description given to the model. Order is the order of the output schema.
FIELDS: dict[DocType, dict[str, str]] = {
    DocType.APPLICATION_FORM: {
        "full_name": "applicant's full name",
        "date_of_birth": "applicant's date of birth",
        "id_number": "citizen ID number (CCCD), 12 digits",
        "phone": "applicant's mobile number",
        "employer": "current employer",
        "declared_monthly_income": "net monthly income declared by the applicant",
        "existing_monthly_debt": "monthly repayments on existing loans and cards",
        "requested_amount": "loan amount requested",
        "term_months": "loan term in months",
    },
    DocType.NATIONAL_ID: {
        "full_name": "holder's full name",
        "date_of_birth": "holder's date of birth",
        "id_number": "card number, 12 digits",
        "sex": "sex as printed (Nam / Nữ)",
        "expiry_date": "date the card expires",
    },
    DocType.PAYSLIP: {
        "employee_name": "employee's full name",
        "employer": "employer name",
        "pay_period": "month the payslip covers",
        "gross_income": "gross monthly income",
        "net_income": "net amount paid to the employee",
    },
}

REQUIRED_DOCS = (DocType.APPLICATION_FORM, DocType.NATIONAL_ID, DocType.PAYSLIP)


class Severity(StrEnum):
    INCOMPLETE = "incomplete"  # the dossier cannot be assessed until a document is supplied
    REVIEW = "review"          # a person must look at it before underwriting
    INFO = "info"              # recorded, no action needed


class Decision(StrEnum):
    READY = "ready_for_underwriting"
    REVIEW = "needs_review"
    INCOMPLETE = "incomplete"


@dataclass
class Page:
    number: int
    text: str
    image_png: bytes | None = None  # set when the page has no usable text layer


@dataclass
class SourceDocument:
    name: str
    pages: list[Page]

    @property
    def scanned(self) -> bool:
        return any(p.image_png is not None for p in self.pages)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages)


@dataclass
class ExtractedField:
    value: str
    evidence: str


@dataclass
class Usage:
    model: str
    step: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


@dataclass
class ExtractedDocument:
    name: str
    doc_type: DocType
    scanned: bool
    fields: dict[str, ExtractedField] = field(default_factory=dict)
    source_text: str = ""
    usage: list[Usage] = field(default_factory=list)
    error: str = ""

    def get(self, name: str) -> str:
        f = self.fields.get(name)
        return f.value.strip() if f else ""


@dataclass
class Finding:
    code: str
    severity: Severity
    message: str
    document: str = ""
    field: str = ""


@dataclass
class DossierResult:
    dossier_id: str
    decision: Decision
    documents: list[ExtractedDocument]
    findings: list[Finding]
    metrics: dict[str, float | int | None] = field(default_factory=dict)

    @property
    def usage(self) -> list[Usage]:
        return [u for d in self.documents for u in d.usage]
