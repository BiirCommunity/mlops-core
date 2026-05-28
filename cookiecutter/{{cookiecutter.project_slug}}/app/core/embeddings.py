"""Multilingual sentence embeddings for drift monitoring."""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class PromptEmbedder:
    """Lightweight multilingual embedder for prompt/response vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "DRIFT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        device_str = device or os.environ.get("DRIFT_EMBEDDING_DEVICE", "cpu")
        self.model = SentenceTransformer(self.model_name, device=device_str)

    def embed(self, text: str) -> np.ndarray:
        stripped = (text or "").strip()
        if not stripped:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros(dim, dtype=np.float32)
        vector = self.model.encode(
            stripped,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        cleaned = [(text or "").strip() for text in texts]
        if not cleaned:
            return []
        vectors = self.model.encode(
            cleaned,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [np.asarray(row, dtype=np.float32) for row in vectors]
