"""
examples/basic_rag.py
---------------------
Minimal end-to-end demo of the RAG pipeline.

This script shows the three-step workflow:
  1. Ingest a local directory of documents.
  2. Ask a question.
  3. Print the answer with sources.

Usage:
    # Make sure OPENAI_API_KEY is set in your environment or a .env file
    python examples/basic_rag.py
"""

import os
import sys
import textwrap
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import RAGPipeline  # noqa: E402


def main():
    # ------------------------------------------------------------------
    # 1.  Configuration
    # ------------------------------------------------------------------
    # By default we index the sample_docs/ directory that sits next to
    # this script.  You can point it at any folder of PDFs / TXTs / MDs.
    docs_dir = Path(__file__).parent / "sample_docs"

    if not docs_dir.exists():
        print(f"[demo] Creating sample_docs/ directory at {docs_dir}")
        docs_dir.mkdir(parents=True)
        _write_sample_doc(docs_dir)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "Error: OPENAI_API_KEY is not set.\n"
            "Export it in your shell or create a .env file with:\n"
            "  OPENAI_API_KEY=sk-..."
        )

    # ------------------------------------------------------------------
    # 2.  Build the pipeline
    # ------------------------------------------------------------------
    print("[demo] Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        openai_api_key=api_key,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        chunk_size=400,
        chunk_overlap=50,
        top_k=3,
    )

    # ------------------------------------------------------------------
    # 3.  Ingest documents
    # ------------------------------------------------------------------
    print(f"[demo] Ingesting documents from: {docs_dir}")
    num_chunks = pipeline.ingest(str(docs_dir), save=True)
    print(f"[demo] Indexed {num_chunks} chunks.\n")

    # ------------------------------------------------------------------
    # 4.  Ask questions
    # ------------------------------------------------------------------
    questions = [
        "What is Retrieval-Augmented Generation?",
        "How does the chunking step work?",
        "What embedding models are supported?",
    ]

    for question in questions:
        print(f"Question: {question}")
        result = pipeline.query(question, return_sources=True)

        # Pretty-print the answer
        answer = result["answer"]  # type: ignore
        wrapped = textwrap.fill(answer, width=80)
        print(f"Answer:\n{wrapped}\n")

        # Show top source chunk
        sources = result["sources"]  # type: ignore
        if sources:
            top = sources[0]
            preview = top["text"][:120].replace("\n", " ")
            print(f"Top source (score={top['score']:.3f}): {preview!r}")

        meta = result["metadata"]  # type: ignore
        print(
            f"[tokens: {meta['prompt_tokens']}+{meta['completion_tokens']} "
            f"| latency: {meta['latency_ms']:.0f}ms]\n"
        )
        print("-" * 70)


def _write_sample_doc(docs_dir: Path) -> None:
    """Create a tiny sample document so the demo works out of the box."""
    content = textwrap.dedent("""
        # RAG System From Scratch

        ## What is Retrieval-Augmented Generation?

        Retrieval-Augmented Generation (RAG) is an AI architecture that improves
        large language model responses by fetching relevant context from an external
        knowledge base before generating an answer.

        Instead of relying solely on knowledge baked into model weights, RAG:
        1. Embeds a user query into a vector.
        2. Searches a vector store for the closest document chunks.
        3. Injects those chunks into the LLM prompt as grounding context.
        4. Generates an answer that cites the retrieved material.

        ## How does chunking work?

        Long documents are split into overlapping chunks so that each piece fits
        within the embedding model's context window.  The chunker in this repo uses
        a recursive character-level splitter with a configurable chunk_size and
        overlap parameter so boundary information is not lost between chunks.

        ## Supported embedding models

        - **OpenAI** (default): `text-embedding-3-small`, `text-embedding-3-large`,
          `text-embedding-ada-002`.
        - **Local / offline**: Any `sentence-transformers` model, e.g.
          `all-MiniLM-L6-v2` for fast CPU inference.
    """)
    (docs_dir / "intro.md").write_text(content.strip())
    print(f"[demo] Created sample document: {docs_dir / 'intro.md'}")


if __name__ == "__main__":
    main()
