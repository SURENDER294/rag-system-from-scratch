import numpy as np
from typing import List, Tuple, Optional

from .vector_store import VectorStore
from .embedder import Embedder


class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.

    Uses cosine similarity (via FAISS dot-product on normalized vectors)
    to rank and return top-k chunks from the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """
        Args:
            vector_store: A loaded/populated VectorStore instance.
            embedder: The same Embedder used when building the index.
            top_k: Number of top results to return.
            score_threshold: If set, only return results with similarity >= threshold.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for the given query.

        Args:
            query: The user's natural language question.
            top_k: Override the default top_k for this call.

        Returns:
            A list of (chunk_text, similarity_score) tuples sorted by score desc.
        """
        k = top_k if top_k is not None else self.top_k

        # Embed the query using the same model as the documents
        query_embedding = self.embedder.embed_query(query)
        query_vec = np.array([query_embedding], dtype=np.float32)

        # Search the FAISS index
        scores, indices = self.vector_store.search(query_vec, k=k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 when fewer than k results exist
                continue

            chunk_text = self.vector_store.get_chunk(idx)

            if self.score_threshold is not None and score < self.score_threshold:
                # Stop adding results once we fall below the threshold
                break

            results.append((chunk_text, float(score)))

        return results

    def retrieve_texts(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Convenience method — returns only the chunk texts, no scores.

        Args:
            query: The user's natural language question.
            top_k: Override the default top_k for this call.

        Returns:
            A list of chunk text strings.
        """
        pairs = self.retrieve(query, top_k=top_k)
        return [text for text, _ in pairs]

    def __repr__(self) -> str:
        return (
            f"Retriever(top_k={self.top_k}, "
            f"score_threshold={self.score_threshold}, "
            f"embedder={self.embedder.__class__.__name__})"
        )
