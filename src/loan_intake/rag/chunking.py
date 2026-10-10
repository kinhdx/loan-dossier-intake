"""Splits the policy documents into retrievable chunks, one per '##' section.

Chunk IDs are '<file stem>#<section number>', stable as long as sections are not reordered, so the
question set can name the chunk that holds each answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


def chunk_markdown(stem: str, markdown: str) -> list[Chunk]:
    lines = markdown.splitlines()
    doc_title = next((line[2:].strip() for line in lines if line.startswith("# ")), stem)
    chunks: list[Chunk] = []
    heading, body = None, []

    def flush():
        if heading is not None and any(b.strip() for b in body):
            title = f"{doc_title} > {heading}"
            chunks.append(Chunk(f"{stem}#{len(chunks) + 1}", title, f"{title}\n" + "\n".join(body).strip()))

    for line in lines:
        if line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        elif heading is not None:
            body.append(line)
    flush()
    return chunks


def load_corpus(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks += chunk_markdown(path.stem, path.read_text(encoding="utf-8"))
    return chunks
