"""LLM Client using litellm for unified provider access."""

import time
import litellm
from typing import List, Dict, Optional
from scr.api.llm_api.config import (
    get_litellm_model_string,
    get_litellm_kwargs,
    validate_provider,
)
from scr.api.llm_api.providers.completion_result import CompletionResult
from scr.utils.logger import get_logger

logger = get_logger(__name__)


def _response_metadata(response, provider):
    finish = getattr(response.choices[0], "finish_reason", None)
    raw = response.model_dump() if provider == "llama_cpp" and hasattr(response, "model_dump") else None
    return {"finish_reason": finish if isinstance(finish, str) else None,
            "raw_response": raw if isinstance(raw, dict) else None}

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    """Client for interacting with LLM providers via litellm."""

    def __init__(self, provider: str):
        validate_provider(provider)
        self.provider = provider
        self.extra_kwargs = get_litellm_kwargs(provider)

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs,
    ) -> CompletionResult:
        litellm_model = get_litellm_model_string(self.provider, model)

        merged_kwargs = {**self.extra_kwargs, **kwargs}
        merged_kwargs.pop("on_token", None)

        t0 = time.time()
        response = litellm.completion(
            model=litellm_model,
            messages=messages,
            stream=False,
            **merged_kwargs,
        )
        duration_s = time.time() - t0

        if not response.choices:
            raise RuntimeError("Received empty response from LLM API")

        message = response.choices[0].message
        content = message.content or ""

        reasoning = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning = message.reasoning_content

        # Extract token counts
        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", None)
            output_tokens = getattr(response.usage, "completion_tokens", None)

        logger.info(f"LLM response received from {self.provider}/{model}")
        return CompletionResult(
            content=content,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=getattr(response, "model", litellm_model),
            duration_s=duration_s,
            **_response_metadata(response, self.provider),
        )

    async def async_get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> CompletionResult:
        litellm_model = get_litellm_model_string(self.provider, model)

        merged_kwargs = {**self.extra_kwargs, **kwargs}
        merged_kwargs.pop("on_token", None)

        t0 = time.time()
        response = await litellm.acompletion(
            model=litellm_model,
            messages=messages,
            stream=False,
            **merged_kwargs,
        )
        duration_s = time.time() - t0

        if not response.choices:
            raise RuntimeError("Received empty response from LLM API")

        message = response.choices[0].message
        content = message.content or ""

        reasoning = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning = message.reasoning_content

        # Extract token counts
        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", None)
            output_tokens = getattr(response.usage, "completion_tokens", None)

        logger.info(f"Async LLM response received from {self.provider}/{model}")
        return CompletionResult(
            content=content,
            reasoning=reasoning,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=getattr(response, "model", litellm_model),
            duration_s=duration_s,
            **_response_metadata(response, self.provider),
        )


def get_client(provider: str) -> LLMClient:
    """Get a client instance for the specified provider."""
    if provider == "claude":
        from scr.api.llm_api.claude_cli_client import ClaudeCLIClient
        return ClaudeCLIClient()
    if provider == "gemini":
        from scr.api.llm_api.gemini_cli_client import GeminiCLIClient
        return GeminiCLIClient()
    if provider == "codex":
        from scr.api.llm_api.codex_cli_client import CodexCLIClient
        return CodexCLIClient()
    return LLMClient(provider)
