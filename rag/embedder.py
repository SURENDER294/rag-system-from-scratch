import os
import time
from typing import List

import numpy as np

from rag.chunker import Chunk


class Embedder:
    """
    Converts text chunks into dense vector embeddings.

    Supports:
        - OpenAI text-embedding-ada-002  (default)
        - Any sentence-transformers model (local, no API key needed)

    The embeddings are L2-normalised before being returned so that a
    dot-product search is equivalent to cosine similarity. This is a
    common trick that simplifies vector store operations downstream.

    Args:
        model:       Model name. Use 'openai:<model>' for OpenAI or
                     'local:<model_name>' for sentence-transformers.
        batch_size:  Number of texts per API call. Larger = faster but
                     may hit token-per-minute limits.
        api_key:     OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model: str = "openai:text-embedding-ada-002",
        batch_size: int = 100,
        api_key: str = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self._backend, self._model_name = self._parse_model(model)
        self._local_model = None  # lazy-loaded on first use

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Embed a list of Chunk objects.

        Returns a numpy array of shape (len(chunks), embedding_dim).
        Rows are in the same order as the input chunks.
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a plain list of strings."""
        if not texts:
            return np.array([])

        if self._backend == "openai":
            vectors = self._embed_openai(texts)
        elif self._backend == "local":
            vectors = self._embed_local(texts)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

        # L2-normalise so downstream cosine similarity is just a dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid divide-by-zero
        return vectors / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a 1-D numpy array."""
        result = self.embed_texts([query])
        return result[0]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        try:
            import openai  # type: ignore
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        openai.api_key = self.api_key
        all_vectors = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            # Retry once on rate-limit errors
            for attempt in range(2):
                try:
                    response = openai.Embedding.create(
                        model=self._model_name, input=batch
                    )
                    break
                except Exception as exc:
                    if attempt == 0 and "rate" in str(exc).lower():
                        time.sleep(5)
                    else:
                        raise

            batch_vectors = [item["embedding"] for item in response["data"]]
            all_vectors.extend(batch_vectors)

        return np.array(all_vectors, dtype=np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Use a sentence-transformers model running locally."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                )
            self._local_model = SentenceTransformer(self._model_name)

        vectors = self._local_model.encode(texts, convert_to_numpy=True)
        return vectors.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model(model: str):
        """Split 'backend:model_name' into (backend, model_name)."""
        if ":" in model:
            backend, name = model.split(":", 1)
            return backend.lower(), name
        # Default to OpenAI if no prefix given
        return "openai", model
