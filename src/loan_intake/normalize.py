"""Parsing helpers for values as they are written on Vietnamese loan documents.

Amounts use '.' as the thousands separator ("15.000.000 đ"), dates are day-first ("05/03/1990").
Every parser returns None instead of raising, so a value that cannot be read becomes a finding
rather than an exception.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

_AMOUNT_NOISE = re.compile(r"(vnđ|vnd|đồng|đ|dong)", re.IGNORECASE)
_MILLIONS = re.compile(r"^([\d.,]+)\s*(trieu|tr)")  # matched on folded text
_DATE = re.compile(r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})")


def parse_amount(raw: str | None) -> int | None:
    """'15.000.000 đ' -> 15000000, '15,5 triệu' -> 15500000.

    Without a unit word, VND has no decimals, so ',' and '.' are both thousands separators.
    """
    if not raw:
        return None
    text = _AMOUNT_NOISE.sub("", raw).strip().lower()
    millions = _MILLIONS.search(fold(text))
    if millions:
        number = millions.group(1).replace(".", "").replace(",", ".")
        try:
            return round(float(number) * 1_000_000)
        except ValueError:
            return None
    digits = re.sub(r"[.,\s]", "", text)
    return int(digits) if digits.isdigit() else None


def parse_int(raw: str | None) -> int | None:
    if not raw:
        return None
    match = re.search(r"\d+", raw)
    return int(match.group()) if match else None


def parse_term_months(raw: str | None) -> int | None:
    """'36 tháng' -> 36, '3 năm' -> 36."""
    number = parse_int(raw)
    if number is None:
        return None
    return number * 12 if fold(raw).split()[-1:] == ["nam"] else number


def parse_date(raw: str | None) -> date | None:
    """Day-first dates: 05/03/1990, 5-3-1990, 05.03.1990."""
    if not raw:
        return None
    match = _DATE.search(raw)
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def fold(text: str | None) -> str:
    """Lower-case, strip Vietnamese diacritics and collapse whitespace, for tolerant name comparison."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(stripped.lower().split())


def has_diacritics(text: str | None) -> bool:
    return bool(text) and fold(text) != " ".join(text.lower().split())


def squash(text: str | None) -> str:
    """NFC + collapsed whitespace, used to check that a quoted value really occurs in the source text."""
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def digits_only(text: str | None) -> str:
    return re.sub(r"\D", "", text or "")


def mask_id(id_number: str | None) -> str:
    """001190012345 -> 001******345, for logs and API listings."""
    value = digits_only(id_number)
    if len(value) < 7:
        return "*" * len(value)
    return value[:3] + "*" * (len(value) - 6) + value[-3:]
