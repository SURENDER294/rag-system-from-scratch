import os
import logging
from pathlib import Path
from typing import List, Optional, Union

from .document_loader import DocumentLoader
from .chunker import TextChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .prompt_builder import PromptBuilder
from .generator import Generator, GenerationResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Two modes of operation:
      1. Build mode  — ingest documents, embed them, save the index.
      2. Query mode  — load a saved index, retrieve context, generate an answer.

    Both modes can be used in the same session (build then query).
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        index_dir: Optional[str] = None,
    ):
        """
        Args:
            openai_api_key: OpenAI key. Falls back to OPENAI_API_KEY env var.
            embedding_model: OpenAI embedding model name.
            llm_model: OpenAI chat model for generation.
            chunk_size: Token/char size of each document chunk.
            chunk_overlap: Overlap between consecutive chunks.
            top_k: Number of chunks to retrieve per query.
            score_threshold: Minimum similarity score for retrieved chunks.
            temperature: LLM sampling temperature.
            max_tokens: Max tokens in LLM response.
            index_dir: Directory to save/load FAISS index. Defaults to ./rag_index.
        """
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Pass openai_api_key= or set OPENAI_API_KEY."
            )

        self.index_dir = Path(index_dir) if index_dir else Path("rag_index")

        # --- Component wiring ---
        self._loader = DocumentLoader()
        self._chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self._embedder = Embedder(model=embedding_model, api_key=api_key)
        self._vector_store = VectorStore()
        self._retriever = Retriever(
            vector_store=self._vector_store,
            embedder=self._embedder,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        self._prompt_builder = PromptBuilder(max_context_chunks=top_k)
        self._generator = Generator(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        logger.info(
            "RAGPipeline initialized | embed=%s | llm=%s | top_k=%d",
            embedding_model,
            llm_model,
            top_k,
        )

    # ------------------------------------------------------------------
    # Build / Ingest
    # ------------------------------------------------------------------

    def ingest(self, paths: Union[str, List[str]], save: bool = True) -> int:
        """
        Load documents, chunk them, embed, and populate the vector store.

        Args:
            paths: A file path, directory path, or list of paths.
            save: If True, persist the FAISS index to disk after ingestion.

        Returns:
            Number of chunks indexed.
        """
        if isinstance(paths, str):
            paths = [paths]

        all_chunks: List[str] = []
        for p in paths:
            logger.info("Loading documents from: %s", p)
            docs = self._loader.load(p)
            for doc_text in docs:
                chunks = self._chunker.split(doc_text)
                all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No text chunks were produced from the given paths.")

        logger.info("Embedding %d chunks...", len(all_chunks))
        embeddings = self._embedder.embed_documents(all_chunks)

        self._vector_store.add(embeddings, all_chunks)
        logger.info("Indexed %d chunks into FAISS.", len(all_chunks))

        if save:
            self.save_index()

        return len(all_chunks)

    def save_index(self) -> None:
        """Persist the FAISS index and chunk texts to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vector_store.save(str(self.index_dir))
        logger.info("Index saved to %s", self.index_dir)

    def load_index(self) -> None:
        """Load a previously saved FAISS index from disk."""
        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {self.index_dir}. "
                "Run ingest() first to build the index."
            )
        self._vector_store.load(str(self.index_dir))
        logger.info("Index loaded from %s", self.index_dir)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ) -> Union[str, dict]:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The natural language question to answer.
            top_k: Override the default number of chunks to retrieve.
            return_sources: If True, also return the retrieved chunks and scores.

        Returns:
            If return_sources=False: the answer string.
            If return_sources=True: a dict with 'answer', 'sources', and 'metadata'.
        """
        logger.info("Query: %s", question)

        # Step 1: Retrieve relevant context
        retrieved = self._retriever.retrieve(question, top_k=top_k)
        if not retrieved:
            logger.warning("No relevant chunks found for the query.")

        # Step 2: Build the prompt
        prompt = self._prompt_builder.build_from_pairs(question, retrieved)

        # Step 3: Generate the answer
        result: GenerationResult = self._generator.generate(prompt)

        logger.info(
            "Generated answer | tokens=%d | latency=%.0fms",
            result.total_tokens,
            result.latency_ms,
        )

        if return_sources:
            return {
                "answer": result.answer,
                "sources": [{"text": t, "score": s} for t, s in retrieved],
                "metadata": {
                    "model": result.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_ms": round(result.latency_ms, 2),
                },
            }

        return result.answer

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RAGPipeline("
            f"embed={self._embedder.model!r}, "
            f"llm={self._generator.model!r}, "
            f"top_k={self._retriever.top_k})"
        )
