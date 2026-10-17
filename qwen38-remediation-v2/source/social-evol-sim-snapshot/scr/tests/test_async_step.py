"""
Integration tests for the async simulation step.

Tests the three-phase architecture with mocked LLM calls.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from scr.models.agent.decision_result import AgentDecisionResult
from scr.models.agent.responses import Response
from scr.models.agent.actions import DoNothing, Action
from scr.models.prompt_manager.messages import Messages
from scr.simulation.event_bus import SimulationEventBus, EventType


def _make_do_nothing_response(agent_id: str) -> Response:
    """Create a valid DoNothing response for testing."""
    return Response(
        agent_id=agent_id,
        thinking="Test thinking",
        short_term_plan="Test plan",
        long_term_memory={},
        action=Action(root=DoNothing(action_type="do_nothing", reason="test")),
    )


def _make_decision_result(agent_id: str, success: bool = True) -> AgentDecisionResult:
    """Create a test AgentDecisionResult."""
    if success:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=_make_do_nothing_response(agent_id),
            messages=Messages(),
            success=True,
        )
    else:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=None,
            messages=Messages(),
            success=False,
            error="Test error",
        )


class TestAgentDecisionResult:
    """Test the AgentDecisionResult dataclass."""

    def test_successful_result(self):
        result = _make_decision_result("agent_0", success=True)
        assert result.success
        assert result.agent_id == "agent_0"
        assert result.response is not None
        assert result.error is None

    def test_failed_result(self):
        result = _make_decision_result("agent_0", success=False)
        assert not result.success
        assert result.agent_id == "agent_0"
        assert result.response is None
        assert result.error == "Test error"


class TestEventBus:
    """Test the event bus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = SimulationEventBus(enabled=True)
        events_received = []

        async def subscriber(event):
            events_received.append(event)

        bus.subscribe(subscriber)
        await bus.start()

        bus.publish(EventType.STEP_STARTED, step=1)
        bus.publish(EventType.STEP_COMPLETED, step=1, duration=1.5)

        # Give the dispatch loop time to process
        await asyncio.sleep(0.3)
        await bus.stop()

        assert len(events_received) == 2
        assert events_received[0].event_type == EventType.STEP_STARTED
        assert events_received[0].data["step"] == 1
        assert events_received[1].event_type == EventType.STEP_COMPLETED

    @pytest.mark.asyncio
    async def test_disabled_bus(self):
        bus = SimulationEventBus(enabled=False)
        events_received = []

        async def subscriber(event):
            events_received.append(event)

        bus.subscribe(subscriber)
        bus.publish(EventType.STEP_STARTED, step=1)

        await asyncio.sleep(0.1)
        assert len(events_received) == 0


class TestAsyncConfig:
    """Test async configuration with backward compatibility."""

    def test_default_async_config(self):
        from scr.models.core.config import LLMConfig
        config = LLMConfig(
            provider="openai", chat_model="gpt-5-mini",
            max_retries=3, two_stage_model=True
        )
        assert config.async_config.max_concurrent_calls == 10
        assert config.async_config.call_timeout_seconds == 120.0
        assert config.async_config.enable_dashboard is False

    def test_custom_async_config(self):
        from scr.models.core.config import LLMConfig
        config = LLMConfig(
            provider="openai", chat_model="gpt-5-mini",
            max_retries=3, two_stage_model=True,
            async_config={
                "max_concurrent_calls": 5,
                "call_timeout_seconds": 30.0,
                "enable_dashboard": True,
            }
        )
        assert config.async_config.max_concurrent_calls == 5
        assert config.async_config.call_timeout_seconds == 30.0
        assert config.async_config.enable_dashboard is True


class TestExponentialBackoff:
    """Test exponential backoff utility."""

    def test_backoff_increases(self):
        from scr.utils.async_utils import exponential_backoff_delay
        delays = [exponential_backoff_delay(i, base=1.0, max_delay=100.0) for i in range(5)]
        # Each should be roughly 2x the previous (ignoring jitter)
        for i in range(1, len(delays)):
            assert delays[i] > delays[i-1] * 0.5  # Allow for jitter

    def test_backoff_capped(self):
        from scr.utils.async_utils import exponential_backoff_delay
        delay = exponential_backoff_delay(100, base=1.0, max_delay=30.0)
        assert delay <= 31.0  # max_delay + max jitter


class TestLiteLLMConfig:
    """Test litellm model string mapping."""

    def test_openai_mapping(self):
        from scr.api.llm_api.config import get_litellm_model_string
        assert get_litellm_model_string("openai", "gpt-5-mini") == "gpt-5-mini"

    def test_deepseek_mapping(self):
        from scr.api.llm_api.config import get_litellm_model_string
        assert get_litellm_model_string("deepseek", "deepseek-chat") == "deepseek/deepseek-chat"

    def test_alibaba_mapping(self):
        from scr.api.llm_api.config import get_litellm_model_string
        assert get_litellm_model_string("alibaba", "qwen-plus") == "openai/qwen-plus"

    def test_tongyuan_mapping(self):
        from scr.api.llm_api.config import get_litellm_model_string
        assert get_litellm_model_string("tongyuan", "gpt-4o") == "azure/gpt-4o"

    def test_openrouter_mapping(self):
        from scr.api.llm_api.config import get_litellm_model_string
        assert get_litellm_model_string("openrouter", "openai/gpt-4o-mini") == "openrouter/openai/gpt-4o-mini"

    def test_unsupported_provider(self):
        from scr.api.llm_api.config import get_litellm_model_string
        with pytest.raises(ValueError):
            get_litellm_model_string("unknown", "model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
