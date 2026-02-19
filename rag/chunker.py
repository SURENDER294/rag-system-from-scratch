from typing import List

from rag.document_loader import Document


class Chunk:
    """A piece of a document produced by the chunker."""

    def __init__(self, text: str, source: str, chunk_index: int, page: int = None):
        self.text = text
        self.source = source
        self.chunk_index = chunk_index
        self.page = page

    def __repr__(self):
        preview = self.text[:50].replace("\n", " ")
        return (
            f"Chunk(source='{self.source}', index={self.chunk_index}, "
            f"text='{preview}...')"
        )


class TextChunker:
    """
    Splits Document objects into smaller, overlapping Chunk objects.

    Strategy: recursive character splitting.

    We try to split on paragraph breaks first (\n\n), then single newlines,
    then sentences, and finally hard-split on character count if nothing else
    works. This gives natural-feeling chunks that respect document structure.

    Args:
        chunk_size:    Target chunk size in characters.
        chunk_overlap: How many characters from the end of one chunk to
                       carry over into the start of the next. This keeps
                       context from bleeding across chunk boundaries.
    """

    # Ordered from least to most aggressive split point.
    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of Documents into Chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self._split_text(doc.text)
            for idx, chunk_text in enumerate(chunks):
                all_chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=doc.source,
                        chunk_index=idx,
                        page=doc.page,
                    )
                )
        return all_chunks

    # ------------------------------------------------------------------
    # Internal splitting logic
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        """Recursively split text, trying each separator in order."""
        return self._recursive_split(text, self.SEPARATORS)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        # Base case: text already fits in one chunk
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = self._pick_separator(text, separators)
        splits = text.split(separator)

        # Merge small splits back together until we reach chunk_size,
        # then start a new chunk (with overlap from the previous one).
        chunks = []
        current_parts = []
        current_length = 0

        for part in splits:
            part_len = len(part) + len(separator)

            if current_length + part_len > self.chunk_size and current_parts:
                # Flush the current chunk
                chunk_text = separator.join(current_parts)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())

                # Keep the tail of the current chunk as overlap
                overlap_text = self._build_overlap(current_parts, separator)
                current_parts = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) + len(separator) if overlap_text else 0

            current_parts.append(part)
            current_length += part_len

        # Flush whatever is left
        if current_parts:
            tail = separator.join(current_parts).strip()
            if tail:
                chunks.append(tail)

        return chunks

    def _pick_separator(self, text: str, separators: List[str]) -> str:
        """Return the first separator that actually appears in the text."""
        for sep in separators:
            if sep and sep in text:
                return sep
        return separators[-1]  # empty string = hard character split

    def _build_overlap(self, parts: List[str], separator: str) -> str:
        """
        Take enough tail parts from the current chunk to fill chunk_overlap
        characters. This is the context we carry into the next chunk.
        """
        overlap_parts = []
        overlap_len = 0
        for part in reversed(parts):
            part_len = len(part) + len(separator)
            if overlap_len + part_len > self.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_len += part_len
        return separator.join(overlap_parts)
