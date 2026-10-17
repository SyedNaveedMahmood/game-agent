"""The local provider uses the existing LiteLLM path and JSON response format."""
from unittest.mock import AsyncMock, Mock

import pytest

from scr.api.llm_api.client import LLMClient
from scr.api.llm_api.config import get_litellm_model_string, validate_provider


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.asyncio
async def test_local_completion(monkeypatch, asynchronous):
    # An ambient paid key must never replace the local SDK placeholder.
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-used')
    response = Mock(choices=[Mock(message=Mock(content='{"ok":true}', reasoning_content=''))],
                    usage=Mock(prompt_tokens=10, completion_tokens=5), model='Qwen3.8-27B-Q4_K_M')
    completion = AsyncMock(return_value=response) if asynchronous else Mock(return_value=response)
    method = 'acompletion' if asynchronous else 'completion'
    monkeypatch.setattr(f'scr.api.llm_api.client.litellm.{method}', completion)
    client = LLMClient('llama_cpp')
    messages = [{'role': 'user', 'content': 'Return JSON'}]
    kwargs = dict(response_format={'type': 'json_object'})
    if asynchronous:
        result = await client.async_get_completion(messages, 'Qwen3.8-27B-Q4_K_M', **kwargs)
    else:
        result = client.get_completion(messages, 'Qwen3.8-27B-Q4_K_M', **kwargs)
    call = completion.call_args.kwargs
    assert call['api_base'] == 'http://127.0.0.1:8080/v1'
    assert call['api_key'] == 'local'
    assert call['model'] == 'openai/Qwen3.8-27B-Q4_K_M'
    assert call['response_format'] == {'type': 'json_object'}
    assert call['timeout'] == 1800.0
    assert call['messages'] == messages
    assert result.content == '{"ok":true}'


def test_local_provider_is_registered():
    validate_provider('llama_cpp')
    assert get_litellm_model_string('llama_cpp', 'model') == 'openai/model'
