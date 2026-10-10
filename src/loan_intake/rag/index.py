"""Embedding and vector search for the policy corpus.

Embeddings are computed locally with a multilingual sentence model (fastembed, ONNX runtime), so policy
text never leaves the machine at indexing time. The store is an in-process matrix persisted to disk,
which is enough for a corpus of a few thousand chunks; `VectorStore` is the seam for a database-backed
store such as pgvector.

Search is hybrid: cosine similarity plus a BM25 score over diacritic-folded words, weighted by
LEXICAL_WEIGHT. The lexical part recovers sections that share an exact term with the question
("thẻ tín dụng") but are phrased differently enough that the sentence embedding ranks them low.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from loan_intake.normalize import fold
from loan_intake.rag.chunking import Chunk

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LEXICAL_WEIGHT = 0.3


def words(text: str) -> list[str]:
    return re.findall(r"\w+", fold(text))


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = [Counter(words(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        terms = set(words(query))
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, (doc, length) in enumerate(zip(self.docs, self.lengths, strict=True)):
            for t in terms & doc.keys():
                tf = doc[t]
                out[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
        return out


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class VectorStore(Protocol):
    def search(self, query_vector: np.ndarray, k: int, query_text: str = "") -> list[tuple[Chunk, float]]: ...


class MatrixStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, model_name: str,
                 lexical_weight: float = LEXICAL_WEIGHT):
        self.chunks, self.vectors, self.model_name = chunks, vectors, model_name
        self.lexical_weight = lexical_weight
        self.bm25 = BM25([c.text for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> MatrixStore:
        return cls(chunks, embedder.embed([c.text for c in chunks]), embedder.model_name)

    def search(self, query_vector: np.ndarray, k: int, query_text: str = "") -> list[tuple[Chunk, float]]:
        scores = self.vectors @ query_vector
        if query_text and self.lexical_weight:
            lexical = self.bm25.scores(query_text)
            if lexical.max() > 0:
                scores = scores + self.lexical_weight * lexical / lexical.max()
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"model": self.model_name, "chunks": [c.__dict__ for c in self.chunks]}
        (folder / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> MatrixStore:
        meta = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in meta["chunks"]], np.load(folder / "vectors.npy"), meta["model"])
