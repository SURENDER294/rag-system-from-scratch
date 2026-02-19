import os
import time
from typing import List, Optional, Dict, Any

try:
    import openai
except ImportError:
    openai = None  # type: ignore


class GenerationResult:
    """Simple container for the LLM response and metadata."""

    def __init__(
        self,
        answer: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
    ):
        self.answer = answer
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.latency_ms = latency_ms

    def __repr__(self) -> str:
        return (
            f"GenerationResult(model={self.model!r}, "
            f"tokens={self.total_tokens}, "
            f"latency_ms={self.latency_ms:.1f})"
        )


class Generator:
    """
    Wraps an OpenAI Chat Completion call to generate answers from prompts.

    Keeps things simple: one method to call, one result object back.
    Supports streaming=False for now (streaming can be added later).
    """

    # Models we know work well for RAG
    SUPPORTED_MODELS = (
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        request_timeout: int = 60,
    ):
        """
        Args:
            model: OpenAI model name.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Max tokens in the completion.
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            system_message: Optional system-level instruction for the model.
            request_timeout: Timeout in seconds for the API call.
        """
        if openai is None:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.system_message = system_message or (
            "You are a precise, helpful assistant that answers questions "
            "strictly based on the provided context."
        )

        # Key resolution: explicit > env var
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY."
            )

        self._client = openai.OpenAI(api_key=resolved_key)

    def generate(self, prompt: str) -> GenerationResult:
        """
        Send the prompt to the LLM and return a structured result.

        Args:
            prompt: The full formatted prompt (including context and question).

        Returns:
            A GenerationResult with the answer text and usage metadata.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        return GenerationResult(
            answer=answer.strip(),
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
        )

    def batch_generate(
        self, prompts: List[str], delay_between: float = 0.5
    ) -> List[GenerationResult]:
        """
        Generate answers for a list of prompts sequentially.

        A small delay is added between calls to respect rate limits.

        Args:
            prompts: List of prompt strings.
            delay_between: Seconds to wait between API calls.

        Returns:
            List of GenerationResult objects in the same order.
        """
        results = []
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt)
            results.append(result)
            if i < len(prompts) - 1:
                time.sleep(delay_between)
        return results

    def __repr__(self) -> str:
        return (
            f"Generator(model={self.model!r}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )
