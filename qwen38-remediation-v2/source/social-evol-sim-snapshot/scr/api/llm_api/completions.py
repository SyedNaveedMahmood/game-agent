"""Completion functions wrapping the LLM client."""

from scr.api.llm_api.providers.completion_result import CompletionResult
from scr.models.prompt_manager.messages import Messages
from typing import Optional, Dict


def get_completions(
    client,
    messages: Messages,
    model: str,
    stream: bool = False,
    response_format: Optional[Dict[str, str]] = None,
    **kwargs,
) -> CompletionResult:
    if response_format is not None:
        kwargs["response_format"] = response_format

    return client.get_completion(messages.messages, model, stream=stream, **kwargs)


async def async_get_completions(
    client,
    messages: Messages,
    model: str,
    response_format: Optional[Dict[str, str]] = None,
    **kwargs,
) -> CompletionResult:
    if response_format is not None:
        kwargs["response_format"] = response_format

    return await client.async_get_completion(messages.messages, model, **kwargs)
