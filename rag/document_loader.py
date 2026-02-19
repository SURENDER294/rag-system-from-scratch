import os
import re
from typing import List, Optional


class Document:
    """Represents a loaded document with text content and metadata."""

    def __init__(self, text: str, source: str, page: Optional[int] = None):
        self.text = text
        self.source = source
        self.page = page

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Document(source='{self.source}', page={self.page}, text='{preview}...')"


class DocumentLoader:
    """
    Loads documents from various file formats into a flat list of Document objects.

    Supported formats:
        - .txt  plain text
        - .md   markdown
        - .pdf  PDF (requires PyPDF2)
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def load(self, path: str) -> List[Document]:
        """Load a single file or every file inside a directory."""
        if os.path.isdir(path):
            return self._load_directory(path)
        return self._load_file(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_directory(self, directory: str) -> List[Document]:
        docs = []
        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, filename)
                    docs.extend(self._load_file(full_path))
        return docs

    def _load_file(self, path: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._load_pdf(path)
        elif ext in (".txt", ".md"):
            return self._load_text(path)
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

    def _load_text(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()
        text = self._clean_text(raw)
        return [Document(text=text, source=path)]

    def _load_pdf(self, path: str) -> List[Document]:
        # We import here so the module still works when PyPDF2 is not installed
        # and the user only works with text files.
        try:
            import PyPDF2  # type: ignore
        except ImportError:
            raise ImportError(
                "PyPDF2 is required to load PDF files. "
                "Install it with: pip install PyPDF2"
            )

        docs = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page_num, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                text = self._clean_text(raw)
                if text:  # skip blank pages
                    docs.append(Document(text=text, source=path, page=page_num))
        return docs

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and remove junk characters."""
        # Collapse runs of whitespace (but keep paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
