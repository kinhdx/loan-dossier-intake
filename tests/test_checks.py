from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import FIXTURES, load_expected, truth_documents

from loan_intake.checks import decide, run_checks
from loan_intake.schemas import Decision, DocType, ExtractedDocument, ExtractedField


@pytest.mark.parametrize("folder", sorted(p for p in FIXTURES.iterdir() if p.is_dir()), ids=lambda p: p.name)
def test_rules_flag_exactly_the_planted_defect(folder, limits, as_of):
    """With perfect extraction, the rules raise the planted defect and nothing else."""
    expected = load_expected(folder)
    findings, _ = run_checks(truth_documents(expected), limits, as_of)
    assert sorted({f.code for f in findings}) == sorted(expected["expected_findings"])
    assert decide(findings).value == expected["expected_decision"]


def _form(**fields) -> ExtractedDocument:
    return ExtractedDocument("form.pdf", DocType.APPLICATION_FORM, False,
                             {k: ExtractedField(v, f"label: {v}") for k, v in fields.items()},
                             source_text="\n".join(f"label: {v}" for v in fields.values()))


def test_value_not_in_text_document_is_flagged(limits, as_of):
    doc = _form(full_name="Lê Văn Nam")
    doc.fields["full_name"] = ExtractedField("Lê Văn Năm", "Họ và tên: Lê Văn Năm")
    findings, _ = run_checks([doc], limits, as_of)
    assert any(f.code == "EVIDENCE_NOT_FOUND" and f.field == "full_name" for f in findings)


def test_scanned_documents_skip_evidence_check(limits, as_of):
    expected = load_expected(sorted(FIXTURES.iterdir())[0])
    docs = truth_documents(expected, scanned=True)
    for d in docs:
        d.fields = {k: replace(v, evidence="not in any text") for k, v in d.fields.items()}
    findings, _ = run_checks(docs, limits, as_of)
    assert not any(f.code == "EVIDENCE_NOT_FOUND" for f in findings)


def test_diacritics_and_case_do_not_count_as_name_mismatch(limits, as_of):
    clean = next(load_expected(p) for p in sorted(FIXTURES.iterdir()) if load_expected(p)["defect"] == "clean")
    docs = truth_documents(clean)
    slip = next(d for d in docs if d.doc_type == DocType.PAYSLIP)
    slip.fields["employee_name"] = ExtractedField("NGUYEN VAN A", "")
    card = next(d for d in docs if d.doc_type == DocType.NATIONAL_ID)
    card.fields["full_name"] = ExtractedField("NGUYỄN VĂN A", "")
    form = next(d for d in docs if d.doc_type == DocType.APPLICATION_FORM)
    form.fields["full_name"] = ExtractedField("Nguyễn Văn A", "")
    findings, _ = run_checks(docs, limits, as_of)
    assert not {"NAME_MISMATCH", "PAYSLIP_NAME_MISMATCH"} & {f.code for f in findings}


def test_names_differing_only_in_tone_marks_go_to_review(limits, as_of):
    clean = next(load_expected(p) for p in sorted(FIXTURES.iterdir()) if load_expected(p)["defect"] == "clean")
    docs = truth_documents(clean)
    form = next(d for d in docs if d.doc_type == DocType.APPLICATION_FORM)
    card = next(d for d in docs if d.doc_type == DocType.NATIONAL_ID)
    slip = next(d for d in docs if d.doc_type == DocType.PAYSLIP)
    form.fields["full_name"] = ExtractedField("Ngô Thùy Mai", "")
    slip.fields["employee_name"] = ExtractedField("Ngô Thùy Mai", "")
    card.fields["full_name"] = ExtractedField("NGÔ THÚY MAI", "")  # tone mark misread on a blurred scan
    findings, _ = run_checks(docs, limits, as_of)
    assert decide(findings) == Decision.REVIEW
    assert {f.code for f in findings} >= {"NAME_DIACRITICS_DIFFER"}


def test_missing_required_field_goes_to_review(limits, as_of):
    clean = next(load_expected(p) for p in sorted(FIXTURES.iterdir()) if load_expected(p)["defect"] == "clean")
    docs = truth_documents(clean)
    card = next(d for d in docs if d.doc_type == DocType.NATIONAL_ID)
    card.fields["expiry_date"] = ExtractedField("", "")
    findings, _ = run_checks(docs, limits, as_of)
    assert decide(findings) == Decision.REVIEW
    assert any(f.code == "MISSING_FIELD" and f.field == "expiry_date" for f in findings)


def test_extraction_error_is_never_ready(limits, as_of):
    clean = next(load_expected(p) for p in sorted(FIXTURES.iterdir()) if load_expected(p)["defect"] == "clean")
    docs = truth_documents(clean)
    docs[0].error = "the model declined the request"
    findings, _ = run_checks(docs, limits, as_of)
    assert decide(findings) != Decision.READY


@pytest.mark.parametrize("term, code", [("ba mươi sáu tháng", "FORMAT_INVALID"), ("3 năm 6 tháng", "FORMAT_INVALID"),
                                        ("7 tháng", "TERM_OUT_OF_RANGE")])
def test_unreadable_or_off_step_term_goes_to_review(term, code, limits, as_of):
    """An unparsed term would otherwise skip the term, age-at-maturity and debt-to-income checks."""
    clean = next(load_expected(p) for p in sorted(FIXTURES.iterdir()) if load_expected(p)["defect"] == "clean")
    docs = truth_documents(clean, scanned=True)
    form = next(d for d in docs if d.doc_type == DocType.APPLICATION_FORM)
    form.fields["term_months"] = ExtractedField(term, "")
    findings, _ = run_checks(docs, limits, as_of)
    assert decide(findings) == Decision.REVIEW
    assert any(f.code == code and f.field == "term_months" for f in findings)


@pytest.mark.parametrize("id_number, sex, consistent", [
    ("001203012345", "Nam", True),    # 2: male, born 2000-2099
    ("001303012345", "Nữ", True),     # 3: female, born 2000-2099
    ("001103012345", "Nữ", False),    # 1 means born 1900-1999, so 1903, not 2003
    ("001203012345", "Nữ", False),    # 2 is the male digit
])
def test_id_number_century_digit_for_holders_born_from_2000(id_number, sex, consistent, limits, as_of):
    fields = {"full_name": "TRẦN MINH ANH", "date_of_birth": "15/04/2003", "id_number": id_number, "sex": sex,
              "expiry_date": "15/04/2028"}
    card = ExtractedDocument("cccd.pdf", DocType.NATIONAL_ID, True,
                             {k: ExtractedField(v, "") for k, v in fields.items()})
    findings, _ = run_checks([card], limits, as_of)
    assert ("ID_NUMBER_INCONSISTENT" in {f.code for f in findings}) is not consistent
