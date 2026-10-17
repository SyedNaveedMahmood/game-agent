"""Tests for the unified JSONL event logger (sim_logger)."""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

import pytest

from scr.utils import sim_logger
from scr.utils.sim_logger import JSONStdlibHandler, jsonl_event_bus_subscriber
from scr.api.llm_api.providers.completion_result import CompletionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_run_dir(tmp_path):
    """Provide a temporary run directory and ensure sim_logger is cleaned up."""
    sim_logger.init(tmp_path)
    yield tmp_path
    sim_logger.close()


def _read_events(run_dir: Path):
    """Read all JSONL lines from events.jsonl."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmit:
    def test_writes_valid_jsonl(self, tmp_run_dir):
        sim_logger.emit("test_event", type="test", level="info", foo="bar")
        events = _read_events(tmp_run_dir)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "test_event"
        assert e["type"] == "test"
        assert e["level"] == "info"
        assert e["data"]["foo"] == "bar"
        assert "ts" in e
        assert "id" in e

    def test_multiple_events(self, tmp_run_dir):
        for i in range(5):
            sim_logger.emit(f"event_{i}", type="test")
        events = _read_events(tmp_run_dir)
        assert len(events) == 5

    def test_no_crash_when_closed(self, tmp_run_dir):
        sim_logger.close()
        # Should not raise
        sim_logger.emit("after_close", type="test")


class TestBind:
    def test_context_propagation(self, tmp_run_dir):
        tokens = sim_logger.bind(run_id="run123", step=5, agent_id="agent_A")
        sim_logger.emit("ctx_event", type="test")
        sim_logger.unbind(tokens)

        events = _read_events(tmp_run_dir)
        assert len(events) == 1
        e = events[0]
        assert e["run_id"] == "run123"
        assert e["step"] == 5
        assert e["agent_id"] == "agent_A"

    def test_unbind_restores_defaults(self, tmp_run_dir):
        tokens = sim_logger.bind(agent_id="temp_agent")
        sim_logger.unbind(tokens)
        sim_logger.emit("after_unbind", type="test")

        events = _read_events(tmp_run_dir)
        assert events[0]["agent_id"] == ""

    @pytest.mark.asyncio
    async def test_asyncio_gather_isolation(self, tmp_run_dir):
        """Verify that asyncio.gather tasks get isolated context copies."""
        results = []

        async def task(agent_id):
            sim_logger.bind(agent_id=agent_id)
            await asyncio.sleep(0.01)  # yield to event loop
            sim_logger.emit("task_event", type="test")

        await asyncio.gather(task("A"), task("B"), task("C"))

        events = _read_events(tmp_run_dir)
        agent_ids = {e["agent_id"] for e in events}
        # Each task should have emitted with its own agent_id
        assert len(events) == 3
        assert agent_ids == {"A", "B", "C"}


class TestJSONStdlibHandler:
    def test_converts_log_records(self, tmp_run_dir):
        handler = JSONStdlibHandler()
        handler.setLevel(logging.DEBUG)

        test_logger = logging.getLogger("test.sim_logger.handler")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False

        try:
            test_logger.info("Hello from stdlib")

            events = _read_events(tmp_run_dir)
            assert len(events) == 1
            e = events[0]
            assert e["event"] == "log"
            assert e["type"] == "log"
            assert e["data"]["message"] == "Hello from stdlib"
            assert e["data"]["logger"] == "test.sim_logger.handler"
            assert e["level"] == "info"
        finally:
            test_logger.removeHandler(handler)


class TestEventBusSubscriber:
    @pytest.mark.asyncio
    async def test_translates_events(self, tmp_run_dir):
        from scr.simulation.event_bus import SimulationEvent, EventType

        event = SimulationEvent(
            event_type=EventType.STEP_STARTED,
            data={"step": 1, "num_agents": 5},
        )
        await jsonl_event_bus_subscriber(event)

        events = _read_events(tmp_run_dir)
        assert len(events) == 1
        e = events[0]
        assert e["event"] == "step_started"
        assert e["type"] == "event_bus"
        assert e["data"]["step"] == 1
        assert e["data"]["num_agents"] == 5


class TestCompletionResultFields:
    def test_new_fields_serialize(self):
        cr = CompletionResult(
            content="hello",
            reasoning="because",
            input_tokens=100,
            output_tokens=50,
            model="gpt-4o",
            duration_s=1.23,
        )
        d = cr.model_dump()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["model"] == "gpt-4o"
        assert d["duration_s"] == 1.23

    def test_none_fields_excluded(self):
        cr = CompletionResult(content="hello")
        d = cr.model_dump()
        assert "input_tokens" not in d
        assert "output_tokens" not in d
        assert "model" not in d
        assert "duration_s" not in d

    def test_json_roundtrip(self):
        cr = CompletionResult(
            content="test",
            input_tokens=10,
            output_tokens=20,
            model="test-model",
            duration_s=0.5,
        )
        json_str = cr.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["input_tokens"] == 10
        assert parsed["model"] == "test-model"
