"""
Tests for the retry error classification system.

Verifies:
- classify_root_cause() matches every known action handler error pattern
- ValidationResult.error_type property returns correct values per stage
- validate_llm_response() returns the correct ValidationStage on failure
"""

import json
import pytest

from scr.simulation.agent_decision.retry_tracker import (
    RetryTracker,
    ValidationResult,
    ValidationStage,
    classify_root_cause,
)


# ---------------------------------------------------------------------------
# classify_root_cause — pattern coverage
# ---------------------------------------------------------------------------


class TestClassifyRootCause:
    """classify_root_cause should match real error messages from action handlers."""

    @pytest.mark.parametrize(
        "message, expected_hint",
        [
            # insufficient_resources — Allocate HP, Collect resource, Rob
            ("Agent_0 has insufficient hp to fight", "insufficient_resources"),
            ("Insufficient quantity: tried 50, available 10", "insufficient_resources"),
            ("Not enough of resource to allocate", "insufficient_resources"),
            ("No resources available for collection", "insufficient_resources"),
            # invalid_action — phase mismatch, age, message length, self-fight
            ("cannot fight yourself", "invalid_action"),
            ("Agent_1 is not old enough to reproduce", "invalid_action"),
            ("Agent can only choose to communicate, allocate, fight, rob or doNothing on this time step", "invalid_action"),
            ("Resource 'gold' cannot be collected in this biome", "invalid_action"),
            ("Message exceeds maximum length of 500 characters", "invalid_action"),
            ("Agent name cannot be empty", "invalid_action"),
            # state_hallucination — target not found
            ("Agent Agent_99 not found in checkpoint", "state_hallucination"),
            ("Prey prey_42 does not exist", "state_hallucination"),
            ("No such agent in the environment", "state_hallucination"),
            ("Invalid agent id: Agent_999", "state_hallucination"),
            # stale_state — dead target
            ("Target agent is dead", "stale_state"),
            ("Agent_3 is not alive", "stale_state"),
            ("Agent already dead before fight", "stale_state"),
            ("Stale action: prey was already killed", "stale_state"),
            # response_too_long — litellm streaming
            ("separator chunk longer than limit", "response_too_long"),
            ("chunk limit exceeded", "response_too_long"),
            # json_syntax — JSON parse errors
            ("Invalid JSON syntax: Expecting value", "json_syntax"),
            ("Failed to parse response", "json_syntax"),
            ("Could not decode the output", "json_syntax"),
            ("Unexpected token in JSON at position 5", "json_syntax"),
            # schema_mismatch — Pydantic errors
            ("Your response failed to match the expected schema", "schema_mismatch"),
            ("Field ('action',) - extra fields not permitted (error type: extra_forbidden)", "schema_mismatch"),
            # rate_limit
            ("Rate limit exceeded", "rate_limit"),
            ("429 Too Many Requests", "rate_limit"),
            ("too many requests, please slow down", "rate_limit"),
            # timeout
            ("Timeout after 120s", "timeout"),
            ("Operation timed out", "timeout"),
            # unknown — nothing matches
            ("Some completely unrecognized error", "unknown"),
        ],
    )
    def test_pattern_match(self, message, expected_hint):
        assert classify_root_cause(message) == expected_hint


# ---------------------------------------------------------------------------
# ValidationResult.error_type
# ---------------------------------------------------------------------------


class TestValidationResult:
    """ValidationResult.error_type should derive from the stage."""

    def test_error_type_json(self):
        r = ValidationResult(success=False, errors=["bad json"], stage=ValidationStage.JSON)
        assert r.error_type == "validation_json"

    def test_error_type_schema(self):
        r = ValidationResult(success=False, errors=["schema fail"], stage=ValidationStage.SCHEMA)
        assert r.error_type == "validation_schema"

    def test_error_type_contextual(self):
        r = ValidationResult(success=False, errors=["ctx fail"], stage=ValidationStage.CONTEXTUAL)
        assert r.error_type == "validation_contextual"

    def test_error_type_no_stage(self):
        r = ValidationResult(success=False, errors=["mystery"])
        assert r.error_type == "unknown"

    def test_error_message_joins(self):
        r = ValidationResult(success=False, errors=["err1", "err2"])
        assert r.error_message == "err1; err2"

    def test_error_message_empty(self):
        r = ValidationResult(success=True)
        assert r.error_message == ""

    def test_success_result(self):
        r = ValidationResult(success=True, response="ok")
        assert r.success is True
        assert r.response == "ok"
        assert r.error_message == ""


# ---------------------------------------------------------------------------
# RetryTracker summary - event log de-duplication
# ---------------------------------------------------------------------------


class TestRetryTrackerSummary:
    """Retry summaries should count canonical retry events, not event bus mirrors."""

    def test_summary_prefers_canonical_retry_records(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_id = "test-run"
        events_dir = tmp_path / "data" / run_id
        events_dir.mkdir(parents=True)

        retry_data = {
            "agent_id": "agent_1",
            "step": 4,
            "attempt": 1,
            "error_type": "timeout",
            "error_message": "Timeout after 120.0s",
            "root_cause_hint": "timeout",
        }
        events = [
            {"ts": "2026-05-11T00:00:00", "event": "retry", "type": "retry", "data": retry_data},
            {"ts": "2026-05-11T00:00:00", "event": "retry", "type": "event_bus", "data": retry_data},
        ]
        for step in range(1, 5):
            events.append(
                {
                    "ts": "2026-05-11T00:00:00",
                    "event": "llm_request",
                    "type": "decision",
                    "data": {"agent_id": "agent_1", "step": step},
                }
            )
        with open(events_dir / "events.jsonl", "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        summary = RetryTracker(run_id).summary_for_run()

        assert summary["total_retries"] == 1
        assert summary["decision_count"] == 4
        assert summary["retried_decision_count"] == 1
        assert summary["retry_rate"] == 0.25
        assert summary["error_breakdown"] == {"timeout": 1}

    def test_summary_falls_back_to_event_bus_records(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_id = "test-run"
        events_dir = tmp_path / "data" / run_id
        events_dir.mkdir(parents=True)

        event = {
            "ts": "2026-05-11T00:00:00",
            "event": "retry",
            "type": "event_bus",
            "data": {
                "agent_id": "agent_1",
                "step": 4,
                "attempt": 1,
                "error_type": "validation_contextual",
                "error_message": "Target agent 'all' not found",
                "root_cause_hint": "state_hallucination",
            },
        }
        with open(events_dir / "events.jsonl", "w") as f:
            f.write(json.dumps(event) + "\n")

        summary = RetryTracker(run_id).summary_for_run()

        assert summary["total_retries"] == 1
        assert summary["error_breakdown"] == {"validation_contextual": 1}


# ---------------------------------------------------------------------------
# validate_llm_response — stage assignment
# ---------------------------------------------------------------------------


class TestValidateLLMResponseStage:
    """validate_llm_response should set the correct ValidationStage on failure."""

    def test_json_failure_stage(self):
        from scr.simulation.act_manager.validator.validator import validate_llm_response
        from scr.models.simulation.checkpoint import Checkpoint

        checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")
        result = validate_llm_response("not json at all {{{", checkpoint, agent_id="Agent_0")
        assert not result.success
        assert result.stage == ValidationStage.JSON
        assert result.error_type == "validation_json"

    def test_schema_failure_stage(self):
        from scr.simulation.act_manager.validator.validator import validate_llm_response
        from scr.models.simulation.checkpoint import Checkpoint

        checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")
        # Valid JSON but invalid schema (missing required fields)
        bad_schema = json.dumps({"not_a_real_field": 123})
        result = validate_llm_response(bad_schema, checkpoint, agent_id="Agent_0")
        assert not result.success
        assert result.stage == ValidationStage.SCHEMA
        assert result.error_type == "validation_schema"

    def test_contextual_failure_stage(self):
        from scr.simulation.act_manager.validator.validator import validate_llm_response
        from scr.models.simulation.checkpoint import Checkpoint

        checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")
        agent_id = checkpoint.metadata.execution_queue[0]

        # Build a valid-schema response that targets a nonexistent agent
        response_dict = {
            "agent_id": agent_id,
            "thinking": "I will fight a ghost",
            "short_term_plan": "fight",
            "long_term_memory": {},
            "action": {
                "action_type": "fight",
                "target_agent_id": "Agent_NONEXISTENT",
                "fight_reason_label": "retaliate_against_attack",
                "reason": "test",
            },
        }
        raw = json.dumps(response_dict)
        result = validate_llm_response(raw, checkpoint, agent_id=agent_id)
        assert not result.success
        assert result.stage == ValidationStage.CONTEXTUAL
        assert result.error_type == "validation_contextual"

    def test_success_returns_no_stage(self):
        from scr.simulation.act_manager.validator.validator import validate_llm_response
        from scr.models.simulation.checkpoint import Checkpoint

        checkpoint = Checkpoint.initialize_from_config("configA_z8_easyHunting_visible")
        agent_id = checkpoint.metadata.execution_queue[0]

        response_dict = {
            "agent_id": agent_id,
            "thinking": "I will do nothing",
            "short_term_plan": "rest",
            "long_term_memory": {},
            "action": {
                "action_type": "do_nothing",
                "reason": "testing",
            },
        }
        raw = json.dumps(response_dict)
        result = validate_llm_response(raw, checkpoint, agent_id=agent_id)
        assert result.success
        assert result.stage is None
        assert result.response is not None
