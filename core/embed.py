"""Text embeddings (all-MiniLM-L6-v2 via fastembed / ONNX, no PyTorch).

Every vector is float32 and L2-normalised, so a dot product is a cosine
similarity. Vectors are stored in SQLite as raw float32 bytes.

Two backends:
  * "minilm" (default) -- the real model, downloaded once into data/models/.
  * "hash"             -- a dependency-free bag-of-words hashing embedder used
                          by the tests and as an offline fallback for the demo.
                          Set FRONTPAGE_EMBEDDER=hash to force it.
The backend that produced the stored vectors is recorded in the kv table;
vectors from different backends are never mixed (see ensure_backend).
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from typing import Protocol

import numpy as np

from . import config, db

DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_WORDS = 500  # title + first ~500 words of text
BATCH_SIZE = 16


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows; zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 through fastembed (ONNX Runtime, CPU)."""

    name = "minilm"

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        # threads=2 keeps the fetch job from hogging all four Pi cores.
        self._model = TextEmbedding(MODEL_NAME, cache_dir=str(config.models_dir()), threads=2)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        vectors = list(self._model.embed(texts, batch_size=BATCH_SIZE))
        return normalize(np.vstack(vectors))


class HashEmbedder:
    """Deterministic bag-of-words vectors: each word maps to a fixed random direction.

    Texts sharing words get similar vectors, which is enough for tests and for
    looking at the UI offline. It is not a real semantic model.
    """

    name = "hash"

    def _word_vector(self, word: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in set(re.findall(r"[a-z0-9]+", text.lower())):
                if len(word) > 2:
                    out[i] += self._word_vector(word)
        return normalize(out)


_embedder: Embedder | None = None


def get_embedder(allow_fallback: bool = False) -> Embedder:
    """The process-wide embedder (loaded lazily; the web app never needs it).

    allow_fallback=True returns the hash embedder if the real model cannot be
    loaded (e.g. first run with no network) instead of raising.
    """
    global _embedder
    if _embedder is None:
        if os.environ.get("FRONTPAGE_EMBEDDER") == "hash":
            _embedder = HashEmbedder()
        else:
            try:
                _embedder = MiniLMEmbedder()
            except Exception:
                if not allow_fallback:
                    raise
                _embedder = HashEmbedder()
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Override the embedder (tests)."""
    global _embedder
    _embedder = embedder


def article_text(title: str, text: str | None) -> str:
    """What we embed for an article: its title plus the first ~500 words."""
    words = (text or "").split()[:EMBED_WORDS]
    return f"{title}. {' '.join(words)}".strip()


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def ensure_backend(conn: sqlite3.Connection, embedder: Embedder) -> None:
    """Drop stored vectors if they came from a different backend.

    The caller then re-embeds everything with embed_pending, so vectors from
    two models are never compared with each other.
    """
    stored = db.kv_get(conn, "embedder")
    stored_name = stored.decode() if isinstance(stored, bytes) else stored
    if stored_name != embedder.name:
        conn.execute("UPDATE articles SET embedding = NULL")
        conn.execute("DELETE FROM kv WHERE key LIKE 'interests:%' OR key = 'model'")
        db.kv_set(conn, "embedder", embedder.name)
        conn.commit()


def embed_pending(conn: sqlite3.Connection, embedder: Embedder) -> list[int]:
    """Embed every article that has no vector yet. Returns their ids."""
    rows = conn.execute(
        "SELECT id, title, COALESCE(content_text, summary, '') AS body FROM articles WHERE embedding IS NULL"
    ).fetchall()
    ids: list[int] = []
    for start in range(0, len(rows), 64):
        chunk = rows[start : start + 64]
        vectors = embedder.embed([article_text(r["title"], r["body"]) for r in chunk])
        conn.executemany(
            "UPDATE articles SET embedding = ? WHERE id = ?",
            [(to_blob(v), r["id"]) for v, r in zip(vectors, chunk)],
        )
        conn.commit()
        ids.extend(r["id"] for r in chunk)
    return ids
