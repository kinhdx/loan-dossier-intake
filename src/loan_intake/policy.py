"""Underwriting thresholds for the sample product, loaded from policy/thresholds.json.

The same numbers are written out in prose in the policy documents that the question-answering index
is built from, so the intake checks and the answers staff get from /ask come from one source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parents[2] / "policy"


@dataclass(frozen=True)
class Thresholds:
    min_age: int
    max_age_at_maturity: int
    min_net_income: int
    max_dti: float
    min_amount: int
    max_amount: int
    max_income_multiple: int
    min_term_months: int
    max_term_months: int
    annual_rate: float
    income_tolerance: float  # declared income may exceed the payslip by this fraction before it is flagged


def load_thresholds(path: Path | None = None) -> Thresholds:
    data = json.loads((path or POLICY_DIR / "thresholds.json").read_text(encoding="utf-8"))
    return Thresholds(**data)


def monthly_installment(amount: int, term_months: int, annual_rate: float) -> float:
    """Equal monthly instalment (annuity) for the requested loan."""
    r = annual_rate / 12
    if r == 0:
        return amount / term_months
    return amount * r / (1 - (1 + r) ** -term_months)
