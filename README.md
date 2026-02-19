# RAG System From Scratch 🔍

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

> **Build a Retrieval-Augmented Generation (RAG) pipeline from the ground up — no LangChain, no magic wrappers. Pure Python 3.8. Every line explained.**

Most RAG tutorials hand you a 10-line LangChain snippet and call it a day. This repo does the opposite: we build every component — document loader, text chunker, embedding engine, vector store, retriever, and LLM generator — from scratch, so you actually understand what's happening under the hood.

---

## Why This Exists

Every AI Engineer interview asks: *"Explain how RAG works."*

If your answer is *"LangChain handles it"*, you're not getting the job.

This project was built to demonstrate deep understanding of the RAG stack — the kind Fortune 500 engineering teams expect when they're paying for someone who can debug retrieval quality at 2am.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Document │───▶│  Chunker │───▶│ Embedder │───▶│  Vector  │  │
│  │  Loader  │    │          │    │          │    │  Store   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                        │        │
│                                                        ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Answer  │◀───│Generator │◀───│  Prompt  │◀───│Retriever │  │
│  │          │    │  (LLM)   │    │ Builder  │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag-system-from-scratch/
├── rag/
│   ├── __init__.py
│   ├── document_loader.py     # Load PDF, TXT, Markdown docs
│   ├── chunker.py             # Recursive + semantic chunking
│   ├── embedder.py            # Sentence embeddings via API or local
│   ├── vector_store.py        # Hand-rolled FAISS-backed vector store
│   ├── retriever.py           # Cosine similarity retrieval
│   ├── prompt_builder.py      # Context-aware prompt construction
│   ├── generator.py           # LLM generation (OpenAI / local)
│   └── pipeline.py            # End-to-end RAG orchestration
├── examples/
│   ├── basic_rag.py           # Quick start example
│   └── advanced_rag.py        # Multi-doc, re-ranking, streaming
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/SURENDER294/rag-system-from-scratch.git
cd rag-system-from-scratch
pip install -r requirements.txt
```

```python
from rag.pipeline import RAGPipeline

# Initialize the full pipeline
pipeline = RAGPipeline(
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-3.5-turbo",
    chunk_size=512,
    chunk_overlap=50,
    top_k=5
)

# Index your documents
pipeline.index_documents(["docs/handbook.pdf", "docs/faq.txt"])

# Ask anything
answer = pipeline.query("What is the refund policy?")
print(answer)
```

---

## Core Components

### 1. Document Loader
Handles PDF, plain text, and Markdown. Strips noise, normalizes whitespace, preserves structure.

### 2. Text Chunker
Recursive character-level splitting with overlap. Respects sentence boundaries so chunks stay semantically coherent.

### 3. Embedder
Works with OpenAI `text-embedding-ada-002` out of the box. Swap to any local model (sentence-transformers) with one line.

### 4. Vector Store
FAISS-backed in-memory store with serialization to disk. Add/delete documents without full re-indexing.

### 5. Retriever
Cosine similarity search with optional MMR (Maximal Marginal Relevance) for diversity.

### 6. Generator
OpenAI ChatCompletion with streaming support. Pluggable — swap in Anthropic, local llama.cpp, or Ollama.

---

## What You'll Learn

- How chunking strategy directly impacts retrieval quality
- Why cosine similarity beats dot product for normalized embeddings
- How to build a FAISS index from scratch without the wrapper
- Prompt engineering for grounded, citation-aware answers
- How to measure RAG quality (faithfulness, relevance, groundedness)

---

## Requirements

- Python 3.8+
- OpenAI API key (or swap to local model)
- faiss-cpu
- numpy
- tiktoken
- PyPDF2

---

## Contributing

PRs welcome. If you find a bug or want to add a feature (re-ranking, hybrid search, multi-modal), open an issue first.

---

## License

MIT — use it, fork it, learn from it.
