import pickle
from typing import List, Tuple

import numpy as np

from rag.chunker import Chunk


class VectorStore:
    """
    An in-memory vector store backed by FAISS.

    Why FAISS and not a pure-numpy brute-force search?
    For small corpora (< 10k chunks) brute-force is fine.
    At 50k+ chunks an IVF or HNSW index makes a real difference.
    We default to FAISS IndexFlatIP (exact inner-product search)
    which works correctly on L2-normalised vectors and gives exact
    cosine similarities without approximation.

    The store keeps a parallel list of Chunk objects so we can return
    the original text alongside similarity scores.
    """

    def __init__(self):
        self._index = None          # FAISS index
        self._chunks = []           # parallel list of Chunk objects
        self._dim = None            # embedding dimension (set on first add)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks:     List of Chunk objects.
            embeddings: numpy array of shape (len(chunks), embedding_dim).
                        Vectors should already be L2-normalised.
        """
        import faiss  # type: ignore

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        dim = embeddings.shape[1]

        if self._index is None:
            # First time: build the index
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)  # inner-product = cosine on normalised vecs
        elif dim != self._dim:
            raise ValueError(
                f"Embedding dimension mismatch: store has {self._dim}, got {dim}"
            )

        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Find the top_k most similar chunks to the query vector.

        Returns a list of (Chunk, score) tuples sorted by score descending.
        Scores are cosine similarities in the range [-1, 1].
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_vector.astype(np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS uses -1 for unfilled slots
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the store to disk using pickle."""
        import faiss

        # FAISS indices can't be pickled directly; serialise separately
        index_bytes = faiss.serialize_index(self._index) if self._index else None
        payload = {
            "index_bytes": index_bytes,
            "chunks": self._chunks,
            "dim": self._dim,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a previously saved VectorStore from disk."""
        import faiss

        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        store = cls()
        store._chunks = payload["chunks"]
        store._dim = payload["dim"]

        if payload["index_bytes"] is not None:
            store._index = faiss.deserialize_index(payload["index_bytes"])

        return store

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"VectorStore(chunks={len(self._chunks)}, dim={self._dim})"
