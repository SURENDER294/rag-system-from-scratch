from typing import List, Optional


# Default prompt template — works well for most QA use cases
_DEFAULT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


class PromptBuilder:
    """
    Builds a prompt string by injecting retrieved context chunks
    and a user question into a template.

    Keeps the interface simple so it's easy to swap templates later
    without touching the rest of the pipeline.
    """

    def __init__(
        self,
        template: Optional[str] = None,
        context_separator: str = "\n\n---\n\n",
        max_context_chunks: int = 5,
    ):
        """
        Args:
            template: A format string with {context} and {question} placeholders.
                      Defaults to the built-in QA template.
            context_separator: String used to join multiple context chunks.
            max_context_chunks: Hard cap on how many chunks go into the prompt.
        """
        self.template = template or _DEFAULT_TEMPLATE
        self.context_separator = context_separator
        self.max_context_chunks = max_context_chunks

        # Validate placeholders at construction time so we fail fast
        if "{context}" not in self.template or "{question}" not in self.template:
            raise ValueError(
                "Prompt template must contain both {context} and {question} placeholders."
            )

    def build(
        self,
        question: str,
        context_chunks: List[str],
    ) -> str:
        """
        Build the final prompt string.

        Args:
            question: The user's question.
            context_chunks: Ordered list of relevant text chunks from the retriever.

        Returns:
            A formatted prompt string ready to be sent to the LLM.
        """
        # Respect the max_context_chunks ceiling
        chunks = context_chunks[: self.max_context_chunks]

        # Number each chunk so the model can reference them if needed
        numbered = [
            f"[{i + 1}] {chunk.strip()}" for i, chunk in enumerate(chunks)
        ]
        context_block = self.context_separator.join(numbered)

        return self.template.format(
            context=context_block,
            question=question.strip(),
        )

    def build_from_pairs(
        self,
        question: str,
        retrieved_pairs: List[tuple],
    ) -> str:
        """
        Convenience method for when the retriever returns (text, score) tuples.

        Args:
            question: The user's question.
            retrieved_pairs: List of (chunk_text, score) from Retriever.retrieve().

        Returns:
            A formatted prompt string.
        """
        chunks = [text for text, _ in retrieved_pairs]
        return self.build(question, chunks)

    def __repr__(self) -> str:
        return (
            f"PromptBuilder("
            f"max_context_chunks={self.max_context_chunks}, "
            f"context_separator={self.context_separator!r})"
        )
