"""Deterministic checks that decide whether a dossier goes straight to underwriting or to a person first.

The model's job ends at reading documents. Whether a dossier is complete, consistent and inside the
product limits is decided here, by rules that can be read, tested and audited:

1. completeness   - every required document type is present
2. field checks   - required fields present and parseable (12-digit ID, valid dates, amounts)
3. grounding      - on text documents, the value (or the span quoted for it) must occur in the text
4. cross-document - name, ID number and date of birth agree between form and ID card (names ignore case, and
                    ignore diacritics only when one side was written without them);
                    employer and income agree between form and payslip
5. ID card rules  - the century/sex digit and birth-year digits of the CCCD match the holder; the expiry date
                    falls on the holder's 25th, 40th or 60th birthday; the card has not expired
6. product limits - age, amount, income multiple, term, minimum income and debt-to-income
                    from policy/thresholds.json

A check that fails produces a Finding. The dossier decision is the most severe finding.
"""
from __future__ import annotations

from datetime import date

from loan_intake.normalize import (
    digits_only,
    fold,
    has_diacritics,
    parse_amount,
    parse_date,
    parse_term_months,
    squash,
)
from loan_intake.policy import Thresholds, monthly_installment
from loan_intake.schemas import (
    FIELDS,
    REQUIRED_DOCS,
    Decision,
    DocType,
    ExtractedDocument,
    Finding,
    Severity,
)

REVIEW, INCOMPLETE, INFO = Severity.REVIEW, Severity.INCOMPLETE, Severity.INFO

DATE_FIELDS = {"date_of_birth", "expiry_date"}
AMOUNT_FIELDS = {"declared_monthly_income", "existing_monthly_debt", "requested_amount", "gross_income",
                 "net_income"}
OPTIONAL_FIELDS = {"existing_monthly_debt", "phone", "pay_period", "gross_income"}
ID_VALIDITY_AGES = {25, 40, 60}


def run_checks(docs: list[ExtractedDocument], limits: Thresholds, as_of: date) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    by_type: dict[DocType, ExtractedDocument] = {}
    for d in docs:
        if d.error:
            findings.append(Finding("EXTRACTION_FAILED", REVIEW, d.error, d.name))
        elif d.doc_type == DocType.UNKNOWN:
            findings.append(Finding("UNKNOWN_DOCUMENT", INFO, "document is not one of the expected types", d.name))
        elif d.doc_type in by_type:
            findings.append(Finding("DUPLICATE_DOCUMENT", REVIEW, f"second {d.doc_type} in the dossier", d.name))
        else:
            by_type[d.doc_type] = d

    for t in REQUIRED_DOCS:
        if t not in by_type:
            findings.append(Finding("MISSING_DOCUMENT", INCOMPLETE, f"no {t} in the dossier"))

    for d in by_type.values():
        findings += _field_checks(d)

    form, card, slip = (by_type.get(t) for t in REQUIRED_DOCS)
    findings += _cross_checks(form, card, slip, limits)
    if card:
        findings += _id_card_checks(card, as_of)
    metrics: dict = {}
    if form:
        f, metrics = _product_checks(form, slip, limits, as_of)
        findings += f
    return findings, metrics


def decide(findings: list[Finding]) -> Decision:
    severities = {f.severity for f in findings}
    if Severity.INCOMPLETE in severities:
        return Decision.INCOMPLETE
    if Severity.REVIEW in severities:
        return Decision.REVIEW
    return Decision.READY


def _field_checks(d: ExtractedDocument) -> list[Finding]:
    out: list[Finding] = []
    source = squash(d.source_text)
    for name in FIELDS[d.doc_type]:
        value = d.get(name)
        if not value:
            if name not in OPTIONAL_FIELDS:
                out.append(Finding("MISSING_FIELD", REVIEW, f"{name} not found", d.name, name))
            continue
        if name in DATE_FIELDS and parse_date(value) is None:
            out.append(Finding("FORMAT_INVALID", REVIEW, f"{name} is not a date: {value!r}", d.name, name))
        if name in AMOUNT_FIELDS and parse_amount(value) is None:
            out.append(Finding("FORMAT_INVALID", REVIEW, f"{name} is not an amount: {value!r}", d.name, name))
        if name == "id_number" and len(digits_only(value)) != 12:
            out.append(Finding("FORMAT_INVALID", REVIEW, "ID number must have 12 digits", d.name, name))
        if not d.scanned:
            evidence = squash(d.fields[name].evidence)
            if squash(value) not in source and (not evidence or evidence not in source):
                out.append(Finding("EVIDENCE_NOT_FOUND", REVIEW, f"value for {name} does not occur in the document text",
                                   d.name, name))
    return out


def _cross_checks(form, card, slip, limits: Thresholds) -> list[Finding]:
    out: list[Finding] = []
    if form and card:
        out += _compare_names(form.get("full_name"), card.get("full_name"), "NAME_MISMATCH",
                              "name on the form differs from the ID card", form.name, "full_name")
        if digits_only(form.get("id_number")) and digits_only(form.get("id_number")) != digits_only(card.get("id_number")):
            out.append(Finding("ID_MISMATCH", REVIEW, "ID number on the form differs from the ID card", form.name, "id_number"))
        dob_form, dob_card = parse_date(form.get("date_of_birth")), parse_date(card.get("date_of_birth"))
        if dob_form and dob_card and dob_form != dob_card:
            out.append(Finding("DOB_MISMATCH", REVIEW, "date of birth on the form differs from the ID card", form.name,
                               "date_of_birth"))
    if card and slip:
        out += _compare_names(card.get("full_name"), slip.get("employee_name"), "PAYSLIP_NAME_MISMATCH",
                              "payslip is issued to a different person", slip.name, "employee_name")
    if form and slip:
        if form.get("employer") and slip.get("employer") and fold(form.get("employer")) != fold(slip.get("employer")):
            out.append(Finding("EMPLOYER_MISMATCH", REVIEW, "employer on the form differs from the payslip", form.name,
                               "employer"))
        declared, net = parse_amount(form.get("declared_monthly_income")), parse_amount(slip.get("net_income"))
        if declared and net and declared > net * (1 + limits.income_tolerance):
            out.append(Finding("INCOME_ABOVE_PAYSLIP", REVIEW,
                               f"declared income {declared:,} exceeds payslip net {net:,} by more than "
                               f"{limits.income_tolerance:.0%}", form.name, "declared_monthly_income"))
    return out


def _compare_names(a: str, b: str, code: str, message: str, document: str, field: str) -> list[Finding]:
    """Names must agree ignoring case. Diacritics are compared only when both sides carry them: a form typed
    without accents ("Nguyen Van A") is accepted, but "Thùy" against "Thúy" is a misread or a different person."""
    if not a or not b:
        return []
    if fold(a) != fold(b):
        return [Finding(code, REVIEW, message, document, field)]
    if has_diacritics(a) and has_diacritics(b) and squash(a).casefold() != squash(b).casefold():
        return [Finding("NAME_DIACRITICS_DIFFER", REVIEW, f"names differ in diacritics: {a!r} / {b!r}", document,
                        field)]
    return []


def _id_card_checks(card: ExtractedDocument, as_of: date) -> list[Finding]:
    """CCCD layout: 3-digit province code, 1 century/sex digit, 2 birth-year digits, 6 serial digits."""
    out: list[Finding] = []
    number = digits_only(card.get("id_number"))
    dob = parse_date(card.get("date_of_birth"))
    sex = fold(card.get("sex"))
    if len(number) == 12 and dob:
        century_digit = int(number[3])
        female = century_digit % 2 == 1
        century = 1900 + 100 * (century_digit // 2)
        year_ok = century + int(number[4:6]) == dob.year
        sex_ok = not sex or (sex == "nu") == female
        if not (year_ok and sex_ok):
            out.append(Finding("ID_NUMBER_INCONSISTENT", REVIEW,
                               "century/sex or birth-year digits of the ID number do not match the holder",
                               card.name, "id_number"))
    expiry = parse_date(card.get("expiry_date"))
    if expiry and expiry < as_of:
        out.append(Finding("ID_EXPIRED", REVIEW, f"ID card expired on {expiry:%d/%m/%Y}", card.name, "expiry_date"))
    # A card is valid until the holder's 25th, 40th or 60th birthday, so any other expiry date was misread.
    if expiry and dob and ((expiry.month, expiry.day) != (dob.month, dob.day)
                           or expiry.year - dob.year not in ID_VALIDITY_AGES):
        out.append(Finding("ID_EXPIRY_IMPLAUSIBLE", REVIEW, "expiry date is not the holder's 25th, 40th or 60th "
                           "birthday", card.name, "expiry_date"))
    return out


def _product_checks(form: ExtractedDocument, slip, limits: Thresholds, as_of: date) -> tuple[list[Finding], dict]:
    out: list[Finding] = []
    amount = parse_amount(form.get("requested_amount"))
    term = parse_term_months(form.get("term_months"))
    dob = parse_date(form.get("date_of_birth"))
    debt = parse_amount(form.get("existing_monthly_debt")) or 0
    net = parse_amount(slip.get("net_income")) if slip else None
    metrics: dict = {"requested_amount": amount, "term_months": term, "net_income": net}

    if amount is not None and not limits.min_amount <= amount <= limits.max_amount:
        out.append(Finding("AMOUNT_OUT_OF_RANGE", REVIEW, f"requested amount {amount:,} is outside "
                           f"{limits.min_amount:,}-{limits.max_amount:,}", form.name, "requested_amount"))
    if amount and net and amount > limits.max_income_multiple * net:
        out.append(Finding("AMOUNT_ABOVE_INCOME_MULTIPLE", REVIEW, f"requested amount is more than "
                           f"{limits.max_income_multiple} times net income", form.name, "requested_amount"))
    if term is not None and not limits.min_term_months <= term <= limits.max_term_months:
        out.append(Finding("TERM_OUT_OF_RANGE", REVIEW, f"term {term} months is outside "
                           f"{limits.min_term_months}-{limits.max_term_months}", form.name, "term_months"))
    if dob:
        age = as_of.year - dob.year - ((as_of.month, as_of.day) < (dob.month, dob.day))
        metrics["age"] = age
        age_at_maturity = age + (term or 0) / 12
        if age < limits.min_age or age_at_maturity > limits.max_age_at_maturity:
            out.append(Finding("AGE_OUT_OF_RANGE", REVIEW, f"age {age} (at maturity {age_at_maturity:.1f}) is outside "
                               "the product range", form.name, "date_of_birth"))
    if net is not None and net < limits.min_net_income:
        out.append(Finding("INCOME_BELOW_MINIMUM", REVIEW, f"net income {net:,} is below {limits.min_net_income:,}",
                           slip.name, "net_income"))
    if net and amount and term:
        installment = monthly_installment(amount, term, limits.annual_rate)
        dti = (debt + installment) / net
        metrics.update(installment=round(installment), dti=round(dti, 3))
        if dti > limits.max_dti:
            out.append(Finding("DTI_ABOVE_LIMIT", REVIEW, f"debt-to-income {dti:.0%} is above {limits.max_dti:.0%}",
                               form.name, "requested_amount"))
    return out, metrics
