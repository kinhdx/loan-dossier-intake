"""Turns uploaded files into pages: text where the PDF has a text layer, a PNG render where it does not.

A page counts as scanned when its text layer holds fewer than MIN_TEXT_CHARS characters. Scanned pages
are sent to the model as images, so reading the characters and extracting the fields happen in one call.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from loan_intake.schemas import Page, SourceDocument

MIN_TEXT_CHARS = 40
RENDER_DPI = 150
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load(name: str, data: bytes) -> SourceDocument:
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        pix = pymupdf.Pixmap(data)
        return SourceDocument(name, [Page(1, "", pix.tobytes("png"))])
    if suffix != ".pdf":
        raise ValueError(f"unsupported file type: {name}")
    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_CHARS:
                pages.append(Page(i, text))
            else:
                pages.append(Page(i, "", page.get_pixmap(dpi=RENDER_DPI).tobytes("png")))
    return SourceDocument(name, pages)


def load_dir(folder: Path) -> list[SourceDocument]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES | {".pdf"})
    return [load(p.name, p.read_bytes()) for p in files]
