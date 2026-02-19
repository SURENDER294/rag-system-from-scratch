"""
tests/test_rag.py
-----------------
Unit tests for each RAG component.

These tests are designed to run WITHOUT an OpenAI API key by mocking all
network calls.  Only the pure-Python logic (chunking, prompting, etc.) is
tested against real data.

Run with:
    pytest tests/ -v
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow imports from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import TextChunker
from rag.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# TextChunker tests
# ---------------------------------------------------------------------------

class TestTextChunker(unittest.TestCase):
    """Tests for the TextChunker component."""

    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, overlap=20)

    def test_short_text_returns_single_chunk(self):
        text = "Hello world. This is a short document."
        chunks = self.chunker.split(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text_is_split(self):
        # Generate text longer than chunk_size
        text = "word " * 50  # ~250 chars
        chunks = self.chunker.split(text)
        self.assertGreater(len(chunks), 1)

    def test_all_chunks_are_non_empty(self):
        text = "sentence. " * 30
        chunks = self.chunker.split(text)
        for chunk in chunks:
            self.assertTrue(len(chunk.strip()) > 0, "Empty chunk found")

    def test_overlap_causes_shared_text(self):
        # With overlap > 0, adjacent chunks should share some content
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 5
        chunker = TextChunker(chunk_size=50, overlap=20)
        chunks = chunker.split(text)
        if len(chunks) > 1:
            # Last part of chunk[0] should appear at start of chunk[1]
            tail = chunks[0][-15:].strip()
            self.assertIn(tail, chunks[1])

    def test_empty_input_returns_empty_list(self):
        chunks = self.chunker.split("")
        self.assertEqual(chunks, [])

    def test_whitespace_only_returns_empty_list(self):
        chunks = self.chunker.split("   \n\t  ")
        self.assertEqual(chunks, [])


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder(unittest.TestCase):
    """Tests for the PromptBuilder component."""

    def setUp(self):
        self.builder = PromptBuilder(max_context_chunks=3)

    def test_basic_build(self):
        prompt = self.builder.build(
            question="What is RAG?",
            context_chunks=["RAG stands for Retrieval-Augmented Generation."],
        )
        self.assertIn("What is RAG?", prompt)
        self.assertIn("Retrieval-Augmented Generation", prompt)

    def test_context_chunks_are_numbered(self):
        chunks = ["Chunk A", "Chunk B", "Chunk C"]
        prompt = self.builder.build("Question?", chunks)
        self.assertIn("[1]", prompt)
        self.assertIn("[2]", prompt)
        self.assertIn("[3]", prompt)

    def test_max_context_chunks_cap(self):
        # Provide more chunks than the cap allows
        chunks = [f"Chunk {i}" for i in range(10)]
        prompt = self.builder.build("Question?", chunks)
        # Only first 3 chunks should appear
        self.assertIn("Chunk 0", prompt)
        self.assertIn("Chunk 2", prompt)
        self.assertNotIn("Chunk 3", prompt)

    def test_build_from_pairs(self):
        pairs = [("Context text", 0.95)]
        prompt = self.builder.build_from_pairs("My question?", pairs)
        self.assertIn("Context text", prompt)
        self.assertIn("My question?", prompt)

    def test_custom_template(self):
        template = "Context: {context}\nQ: {question}\nA:"
        builder = PromptBuilder(template=template)
        prompt = builder.build("test question", ["test context"])
        self.assertTrue(prompt.startswith("Context:"))

    def test_invalid_template_raises(self):
        with self.assertRaises(ValueError):
            PromptBuilder(template="This template has no placeholders")

    def test_empty_context_chunks(self):
        # Should not crash with empty context
        prompt = self.builder.build("Question with no context?", [])
        self.assertIn("Question with no context?", prompt)


# ---------------------------------------------------------------------------
# Chunker edge cases
# ---------------------------------------------------------------------------

class TestTextChunkerEdgeCases(unittest.TestCase):
    """Additional edge-case tests for TextChunker."""

    def test_chunk_size_equals_text_length(self):
        text = "a" * 200
        chunker = TextChunker(chunk_size=200, overlap=0)
        chunks = chunker.split(text)
        # Should be exactly one chunk
        self.assertEqual(len(chunks), 1)

    def test_repr_contains_chunk_size(self):
        chunker = TextChunker(chunk_size=256, overlap=32)
        r = repr(chunker)
        self.assertIn("256", r)


class TestPromptBuilderRepr(unittest.TestCase):
    """Test string representations."""

    def test_repr(self):
        builder = PromptBuilder(max_context_chunks=7)
        r = repr(builder)
        self.assertIn("7", r)


if __name__ == "__main__":
    unittest.main()
