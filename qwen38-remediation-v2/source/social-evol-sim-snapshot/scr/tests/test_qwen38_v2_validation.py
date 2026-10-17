"""Regression checks for audited V2 failure modes; no inference required."""
import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import scr.simulation.agent_decision
from scr.models.agent.responses import Response
from scr.simulation.act_manager.validator.deep_memory import (
    validate_memory_structure, validate_memory_fidelity, MAJOR_SECTIONS,
)
from scr.simulation.agent_decision.retry_tracker import ValidationResult, ValidationStage
from qwen38_v2_infrastructure import (
    budget_check, ContextBudgetExceeded, context_truncated,
    allocation_measurement, watchdog_classification, action_dict,
)


def memory():
    return {MAJOR_SECTIONS[0]: {}, "Agent_Specific_Memory": {
        "agent_2": {"important_interaction_history": {
            "what_i_did_to_him": {}, "what_he_did_to_me": {
                "time_step_1": 1, "action_type": "allocate", "if_success": True,
                "reason": "Allocated 3 HP to me.", "target_moral_type": "universal_group_focused_moral"}},
            "thinking": "", "moral_type": "reciprocal_group_focused_moral", "relationship": "ally", "agreement": "", "plan": ""}},
        "Family_Plan": {}, "Plan_For_Reproduction": {"thinking": "", "preconditions_and_subgoals": "", "estimated_time_to_produce_next_child": 0}, "Strategies": ""}


def test_original_flat_memory_schema_and_no_mutation():
    original = memory()
    before = copy.deepcopy(original)
    assert validate_memory_structure(original)["valid"]
    assert original == before


def test_nested_events_supported_but_not_required():
    m = memory()
    h = m["Agent_Specific_Memory"]["agent_2"]["important_interaction_history"]
    flat = h["what_he_did_to_me"]
    h["what_he_did_to_me"] = {"time_step_1": {"time_step": 1, **{k:v for k,v in flat.items() if k != "time_step_1"}}}
    assert validate_memory_structure(m)["valid"]


def test_sections_and_event_fields_are_separate_action_diagnostics():
    m = memory()
    del m["Family_Plan"]
    del m["Agent_Specific_Memory"]["agent_2"]["important_interaction_history"]["what_he_did_to_me"]["if_success"]
    result = validate_memory_structure(m)
    assert not result["valid"]
    assert any("Family_Plan" in e["path"] for e in result["errors"])
    assert any("if_success" in e["path"] for e in result["errors"])
    response = Response(agent_id="agent_1", thinking="", short_term_plan="", long_term_memory=m,
                        action={"action_type": "do_nothing", "reason": "valid action, invalid memory"})
    assert response.action.root.action_type == "do_nothing"


def test_factual_omission_wrong_participant_and_contradiction():
    before = memory()
    m = copy.deepcopy(before)
    event = m["Agent_Specific_Memory"]["agent_2"]["important_interaction_history"]["what_he_did_to_me"]
    event["if_success"] = False
    fact = {"id": "cooperation", "entity": "agent_2", "description": "Three HP allocation", "patterns": [r"3 HP", r"allocate"]}
    result = validate_memory_fidelity(m, before, focal_id="agent_1", facts=[fact], known_agents=["agent_1", "agent_2"], current_step=4,
        complete_event_steps=[1], events=[{"step": 1, "actor": "agent_2", "recipient": "agent_1", "action_type": "allocate", "success": True}])
    assert not result["deterministic_consistent"]
    assert any(e["kind"] == "contradictory_event_outcome" for e in result["findings"])
    m["Agent_Specific_Memory"]["agent_99"] = m["Agent_Specific_Memory"].pop("agent_2")
    result = validate_memory_fidelity(m, before, focal_id="agent_1", facts=[fact], known_agents=["agent_1", "agent_2"])
    assert result["fact_hits"] == 0
    assert result["facts"][0]["other_entities_matching"] == ["agent_99"]
    assert any(e["kind"] == "incorrect_participant" for e in result["findings"])


def test_nested_truncation_field_and_explicit_context_budget():
    assert context_truncated({"server_context": {"truncated": True}})
    assert not context_truncated({"server_context": {"truncated": False}, "input_tokens": 48000, "output_tokens": 6000})
    with pytest.raises(ContextBudgetExceeded):
        budget_check(50000, "first_stage")
    assert budget_check(40000, "first_stage")["fits"]


def test_actual_allocation_capped_recipient_and_full_plan():
    a = {"action_type": "allocate", "allocation_plan": {"agent_2": 5, "agent_3": 2}}
    result = allocation_measurement(a, {"agent_1": 20, "agent_2": 38, "agent_3": 2},
                                   {"agent_1": 13, "agent_2": 40, "agent_3": 4}, "agent_1")
    assert result["offered_hp"] == result["source_spent_hp"] == 7
    assert result["received_hp"] == 4
    assert result["source_matches_plan"]
    assert action_dict({"action": {"root": a}}) == a


def test_original_allocation_execution_measured():
    import qwen38_capability_evaluation as cap
    from scr.models.agent.actions import Allocate
    cp = cap.new_checkpoint(701, "v2-regression-allocation")
    cp.social_environment.get_agent_by_id("agent_2").state.hp = 2
    before = {a.id:a.state.hp for a in cp.social_environment.agents}
    action = Allocate(action_type="allocate", reason="Known transfer", allocation_reason_label="sharing_with_other", allocation_plan={"agent_2": 5})
    cap.apply_fixture_action(cp, "agent_1", action)
    after = {a.id:a.state.hp for a in cp.social_environment.agents}
    measured = allocation_measurement(action.model_dump(), before, after, "agent_1")
    assert measured["received_hp"] == measured["source_spent_hp"] == 5
    assert measured["recipients"][0]["recipient"] == "agent_2"


@pytest.mark.parametrize("age", [301, 1200, 3500])
def test_watchdog_never_restarts_healthy_slow_inference(age):
    result = watchdog_classification(healthy=True, process_alive=True, active=True,
                                    idle_seconds=0, progress_age=3, request_age=age)
    assert not result["restart"]


def test_watchdog_requires_failure_evidence():
    assert watchdog_classification(healthy=False, process_alive=False, active=False, idle_seconds=400, progress_age=400, request_age=400)["restart"]
    assert watchdog_classification(healthy=True, process_alive=True, active=True, idle_seconds=0, progress_age=250, request_age=3700)["restart"]
    assert not watchdog_classification(healthy=True, process_alive=True, active=False, idle_seconds=4000, progress_age=4000, request_age=0)["restart"]


def test_seed_updates_previously_imported_rng():
    from scr.utils.random_utils import shared_random, set_random_seed
    set_random_seed(123)
    first = shared_random.random()
    set_random_seed(123)
    assert shared_random.random() == first


def test_paired_fixture_changes_only_focal_moral_type_and_run_id(tmp_path,monkeypatch):
    import qwen38_v2_fixtures as f
    monkeypatch.setattr(f,"OUT",tmp_path)
    cp,initial=f.build_a("adult_offspring")
    snapshot=json.loads(cp.model_dump_json())
    clones=[f.clone_role(snapshot,role,"pair") for role in f.ROLES]
    assert len({f.physical_pair_hash(c) for c in clones})==1
    for clone in clones:
        focal=clone.social_environment.get_agent_by_id("agent_1")
        child=clone.social_environment.get_agent_by_id("agent_8")
        assert focal.id=="agent_1" and focal.state.hp==20 and focal.state.age==18
        assert child.state.hp==2 and child.state.age==10
        assert child.family.parent_id==focal.id and child.id in focal.family.children_ids


def test_validator_field_names_match_original_reflection_prompt():
    import inspect
    from scr.simulation.agent_decision import async_response_processor
    from scr.simulation.act_manager.validator.deep_memory import EVENT_FIELDS,AGENT_FIELDS,PREY_FIELDS
    original=inspect.getsource(async_response_processor)
    for name in tuple(MAJOR_SECTIONS)+EVENT_FIELDS+AGENT_FIELDS+tuple(PREY_FIELDS):
        assert name in original
    assert '"time_step_<int>": "int"' in original


def test_kin_stranger_counterparts_match_identity_age_hp_and_resources(tmp_path,monkeypatch):
    import qwen38_v2_fixtures as f
    monkeypatch.setattr(f,"OUT",tmp_path)
    kin,_=f.build_a("adult_offspring")
    stranger,_=f.build_a("stranger")
    assert kin.physical_environment.model_dump()==stranger.physical_environment.model_dump()
    for aid in ("agent_1","agent_8"):
        a=kin.social_environment.get_agent_by_id(aid)
        b=stranger.social_environment.get_agent_by_id(aid)
        assert a.state.model_dump()==b.state.model_dump()
        assert a.type==b.type
    assert len(kin.social_environment.agents)==len(stranger.social_environment.agents)==8


def test_context_validation_preserves_checkpoint_and_all_rng_states():
    import random,pickle,numpy as np
    import qwen38_capability_evaluation as cap
    from scr.utils.random_utils import shared_random
    from scr.simulation.act_manager.validator.validator import validate_llm_response
    cp=cap.new_checkpoint(12319,"validation-purity-regression")
    response=Response(agent_id="agent_1",thinking="",short_term_plan="",long_term_memory={},
        action={"action_type":"fight","reason":"Legality check must be pure","target_agent_id":"agent_2","fight_reason_label":"retaliate_against_attack"})
    before=cp.model_dump_json()
    py=random.getstate();env=shared_random.getstate();npy=pickle.dumps(np.random.get_state())
    assert validate_llm_response(response.model_dump_json(),cp,agent_id="agent_1").success
    assert random.getstate()==py and shared_random.getstate()==env
    assert pickle.dumps(np.random.get_state())==npy
    assert cp.model_dump_json()==before


def test_deceased_parent_is_valid_genealogy_not_missing_relationship():
    import qwen38_capability_evaluation as cap
    from scr.models.agent.agent import Agent
    from qwen38_v2_infrastructure import family_consistent
    cp=cap.new_checkpoint(12321,"family-cemetery-regression")
    parent=cp.social_environment.get_agent_by_id("agent_1")
    child=Agent.initialize(parent.type,"agent_1_child",cp.configuration,hp=3,age=0)
    child.family.parent_id=parent.id;parent.family.children_ids.append(child.id)
    cp.social_environment.agents.append(child);cp.metadata.execution_queue.append(child.id)
    parent.state.hp=0
    cp.remove_dead_agents("controlled regression death")
    assert family_consistent(cp,require_inheritance=True)
    child.family.parent_id="agent_unknown"
    assert not family_consistent(cp)


def test_local_retry_retains_original_evidence_and_bounds_feedback():
    from scr.models.prompt_manager import Messages
    from scr.simulation.agent_decision import async_retry
    messages = Messages(messages=[{"role": "system", "content": "original rules"}, {"role": "user", "content": "irreplaceable observations and memory"}])
    originals = copy.deepcopy(messages.messages)
    captured = []
    response = Response(agent_id="agent_1", thinking="", short_term_plan="", long_term_memory=memory(), action={"action_type":"do_nothing", "reason":""})

    async def failing(**kwargs):
        captured.append(copy.deepcopy(kwargs["messages"].messages))
        kwargs["messages"].append("assistant", "x"*20000)
        kwargs["messages"].append("user", "old reflection"*500)
        if len(captured) < 3:
            return ValidationResult(success=False, errors=["illegal action"], stage=ValidationStage.CONTEXTUAL)
        return ValidationResult(success=True, response=response)

    config = SimpleNamespace(chat_model="local", reasoning_model=None, provider="llama_cpp", two_stage_model=True,
                             async_config=SimpleNamespace(call_timeout_seconds=30, retry_backoff_base=0, retry_backoff_max=0))
    cp = SimpleNamespace(metadata=SimpleNamespace(current_time_step=1))
    with patch.object(async_retry, "async_process_llm_response", side_effect=failing), patch.object(async_retry, "save_debug_messages"):
        success, _ = asyncio.run(async_retry.async_process_llm_response_with_retries(None, messages, cp, config, "agent_1", max_retries=2))
    assert success and len(captured) == 3
    assert all(c[:2] == originals for c in captured)
    assert [len(c) for c in captured] == [2, 3, 3]
    assert "Last chance to generate a valid response." in captured[-1][-1]["content"]
