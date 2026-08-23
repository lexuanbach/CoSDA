from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


@dataclass
class HashEmbedder:
    dim: int = 768
    char_min: int = 3
    char_max: int = 5

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        rows = [self._encode_one(text or "") for text in texts]
        return np.vstack(rows).astype(np.float32) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        padded = f"  {text.lower()}  "
        for n in range(self.char_min, self.char_max + 1):
            if len(padded) < n:
                continue
            for i in range(len(padded) - n + 1):
                gram = padded[i : i + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                idx = value % self.dim
                sign = 1.0 if (value >> 63) else -1.0
                vec[idx] += sign
        norm = math.sqrt(float(vec @ vec))
        if norm:
            vec /= norm
        return vec


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        device = os.environ.get("COSDA_EMBEDDING_DEVICE") or None
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True),
            dtype=np.float32,
        )


def make_embedder(name: str = "hash", dim: int = 768):
    if name == "hash":
        return HashEmbedder(dim=dim)
    if name in {"labse", "sentence-transformers/LaBSE"}:
        return SentenceTransformerEmbedder("sentence-transformers/LaBSE")
    if name.startswith("sentence-transformers/"):
        return SentenceTransformerEmbedder(name)
    raise ValueError(f"Unknown embedder: {name}")
