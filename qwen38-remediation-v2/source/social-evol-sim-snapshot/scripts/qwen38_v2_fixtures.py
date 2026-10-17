"""Paired checkpoint interventions and histories made with original action handlers."""
from __future__ import annotations

import copy
import json
import random
import re
import sys
import uuid
from unittest.mock import patch

import qwen38_capability_evaluation as old
from qwen38_v2_infrastructure import *

sys.path.insert(0, str(ROOT))
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Allocate, Communicate, Fight
from scr.models.agent.responses import Response
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import prepare_agent_prompts
from scr.simulation.act_manager.validator.validator import validate_llm_response
from scr.simulation.act_manager.validator.deep_memory import MAJOR_SECTIONS
from scr.utils.random_utils import shared_random

ROLES = tuple(old.TYPE_IDS)
SCENARIOS = ("infant", "adult_offspring", "stranger", "cooperator", "unfair_partner")
SAMPLING_SEEDS = (1103, 2207, 3301, 4409, 5519)
ADDITIONAL_SEEDS = (6619, 7723, 8837, 9941, 10559)
ENV_SEEDS = (20261017, 20261019, 20261023)
FIXTURE_SEED = 2026101601


def base_checkpoint(seed, run_id, life_steps=2):
    cp = old.new_checkpoint(seed, run_id, {
        "llm.max_retries": MAX_RECOVERIES,
        "llm.async_config.call_timeout_seconds": 7200.,
        "world.max_life_steps": life_steps, "world.communication_and_sharing_steps": 2,
    })
    assert cp.configuration.world.max_life_steps == life_steps
    assert cp.configuration.llm.two_stage_model is True
    assert len(cp.social_environment.agents) == 8
    assert Counter(a.type for a in cp.social_environment.agents) == Counter({r:2 for r in ROLES})
    return cp


from collections import Counter


def blank_memory():
    return {MAJOR_SECTIONS[0]: {}, "Agent_Specific_Memory": {}, "Family_Plan": {},
            "Plan_For_Reproduction": {"thinking": "", "preconditions_and_subgoals": "", "estimated_time_to_produce_next_child": 0}, "Strategies": ""}


def entity_memory(other_type, relationship="", agreement=""):
    return {"important_interaction_history": {"what_i_did_to_him": {}, "what_he_did_to_me": {}},
            "thinking": "", "moral_type": other_type, "relationship": relationship, "agreement": agreement, "plan": ""}


def recorded_event(step, action, reason, target_type, success=True):
    return {f"time_step_{step}": step, "action_type": action, "if_success": success,
            "reason": reason, "target_moral_type": target_type}


def assert_legal_fixture(cp, aid, action):
    actor = old.get_agent(cp, aid)
    response = Response(agent_id=aid, thinking="Controlled history fixture", short_term_plan="", long_term_memory=actor.memory.long_term_memory, action=action)
    valid = validate_llm_response(response.model_dump_json(), cp, agent_id=aid)
    if not valid.success:
        raise AssertionError(f"Illegal fixture for {aid}: {valid.error_message}")


def tick(cp, actions=None, action_seeds=None):
    for aid, action in (actions or {}).items():
        assert_legal_fixture(cp, aid, action)
    old.seed_tick(cp, actions, action_seeds)


def clone_role(snapshot, role, run_id):
    cp = Checkpoint.model_validate(copy.deepcopy(snapshot))
    focal = old.get_agent(cp, "agent_1")
    focal.type = role
    cp.metadata.run_id = run_id
    # This is a counterfactual current moral-profile intervention. Other agents,
    # including existing offspring, retain their observed identities and types.
    return cp


def physical_pair_hash(cp):
    data = json.loads(cp.model_dump_json())
    data["metadata"]["run_id"] = "paired"
    next(a for a in data["social_environment"]["agents"] if a["id"] == "agent_1")["type"] = "moral-treatment"
    return object_hash(data)


def build_a(scenario):
    shared=OUT/"fixtures/A_shared_background.json"
    if shared.exists():
        cp=Checkpoint.model_validate(read(shared))
    else:
        cp=base_checkpoint(FIXTURE_SEED,"Q38V2-A-shared-background")
        write(shared,json.loads(cp.model_dump_json()))
    from scr.utils.random_utils import set_random_seed
    set_random_seed(FIXTURE_SEED)
    random.seed(FIXTURE_SEED)
    cp.metadata.run_id="Q38V2-A-base-"+scenario
    focal = old.get_agent(cp, "agent_1")
    # The pre-intervention parent/child are reciprocal, matching the original
    # inheritance rule. Only the focal current moral profile changes at test.
    focal.type="reciprocal_group_focused_moral"
    focal.state.hp = 18 if scenario == "cooperator" else 20 if scenario == "unfair_partner" else 21
    focal.state.age = 9 if scenario == "infant" else 17
    target = old.get_agent(cp, "agent_8")
    target.type = "reciprocal_group_focused_moral"
    target.state.age = 9
    target.state.hp = 6 if scenario == "cooperator" else 4 if scenario == "unfair_partner" else 3
    if scenario in ("infant", "adult_offspring"):
        target.state.age=0 if scenario=="infant" else 9
        target.family.parent_id = focal.id
        focal.family.children_ids = [target.id]
    id_rng=random.Random(FIXTURE_SEED)
    # Opaque labels must match across controlled counterparts too. This fixture-
    # only UUID source changes no stochastic transition or original action rule.
    # B/longitudinal population dynamics retain original uuid4 identifiers.
    with patch("scr.models.environment.physical.uuid.uuid4",side_effect=lambda:uuid.UUID(int=id_rng.getrandbits(128))):
        if scenario == "cooperator":
            tick(cp, {target.id: Allocate(action_type="allocate", reason="Share 3 HP with a cooperative peer.", allocation_reason_label="sharing_with_other", allocation_plan={focal.id:3})})
        elif scenario == "unfair_partner":
            tick(cp, {target.id: Communicate(action_type="communicate", reason="Offer a sharing agreement.", target_agent_ids=[focal.id], message="I promise to allocate 5 HP to you next social round.")})
        else:
            tick(cp)
        if scenario == "unfair_partner":
            tick(cp, {target.id: Allocate(action_type="allocate", reason="I transfer only 1 HP, less than my promise of 5.", allocation_reason_label="sharing_with_other", allocation_plan={focal.id:1})})
        else:
            tick(cp)
        tick(cp)
    assert focal.state.hp == 20 and target.state.hp == 2
    assert focal.state.age == (10 if scenario == "infant" else 18)
    assert cp.metadata.current_time_step == 4
    assert_legal_fixture(cp, focal.id, Allocate(action_type="allocate", reason="Legality witness", allocation_reason_label="sharing_with_family" if scenario in ("infant", "adult_offspring") else "sharing_with_other", allocation_plan={target.id:1}))
    return cp, {"scenario": scenario, "focal_id": focal.id, "target_id": target.id,
                "environment_seed": FIXTURE_SEED, "paired_state_hash": physical_pair_hash(cp),
                "historical_observations": cp.observations, "legal_positive_transfer_witness": True,
                "intervention_note": "Only focal current moral type/profile changes. Existing offspring type stays fixed. This counterfactual moral assignment is not a claim that the original simulator evolves moral types after birth."}


def logistics_message(step, writer, sections, dense_chars=900):
    # Distinct proposed plans, not completed social events. The contextual
    # validator (stricter than Pydantic) limits each message to 1000 characters.
    if sections >= 32:
        prefix = f"Optional route-map proposal {step}-{writer}; codes are route labels, not agents, actions or completed transfers: "
        codes = " ".join(f"R{i%997:03d}x{(step*41+i*73)%991:03d}Z" for i in range(120))
        return prefix + codes[:min(dense_chars, 990-len(prefix))]
    notes = []
    for i in range(sections):
        notes.append(f"Planning note {step}-{writer}-{i}: inspect route sector {i%17}, mark an alternate path beside the river, compare shelter positions, consider weather uncertainty and avoid confusing an unexecuted proposal with an action. No allocation, attack, or agreement is completed by this planning note.")
    return ("Optional route proposals; no completed social event. " + " ".join(notes))[:950]


def build_c_history(sections,dense_chars=900):
    cp = base_checkpoint(FIXTURE_SEED + 20, "Q38V2-C-history", life_steps=10)
    focal = old.get_agent(cp, "agent_1")
    old.get_agent(cp, "agent_5").state.physical_ability = 5
    focal.state.physical_ability = 6
    for step in range(1, 13):
        actions = {}
        if cp.configuration.isSocialInteractionStep(step):
            if sections:
                for writer in (f"agent_{i}" for i in range(2,9)):
                    actions[writer] = Communicate(action_type="communicate", reason="Propose route planning; no transfer or attack claimed.", target_agent_ids=[focal.id], message=logistics_message(step, writer, sections,dense_chars))
            if step == 1:
                actions.update({
                    "agent_2": Allocate(action_type="allocate", reason="Cooperative gift of 3 HP.", allocation_reason_label="sharing_with_other", allocation_plan={focal.id:3}),
                    "agent_3": Communicate(action_type="communicate", reason="Make a promise", target_agent_ids=[focal.id], message="I promise 4 HP, but no HP has been sent yet."),
                    "agent_8": Communicate(action_type="communicate", reason="Make a sharing agreement", target_agent_ids=[focal.id], message="I promise to share 5 HP with you next social round."),
                    "agent_5": Fight(action_type="fight", reason="The attacker alleges past unfair sharing; that allegation is not established by this observation.", target_agent_id=focal.id, fight_reason_label="punish_for_unfair_sharing"),
                })
            elif step == 2:
                actions.update({
                    "agent_3": Communicate(action_type="communicate", reason="Refuse the promised transfer", target_agent_ids=[focal.id], message="I refuse to give the promised 4 HP. I have transferred zero HP to you."),
                    "agent_8": Allocate(action_type="allocate", reason="I share only 1 HP despite promising 5.", allocation_reason_label="sharing_with_other", allocation_plan={focal.id:1}),
                    "agent_4": Allocate(action_type="allocate", reason="Separate helpful gift of 2 HP.", allocation_reason_label="sharing_with_other", allocation_plan={focal.id:2}),
                    focal.id: Fight(action_type="fight", reason="Retaliate against agent_5's actual attack at Step 1.", target_agent_id="agent_5", fight_reason_label="retaliate_against_attack"),
                })
        tick(cp, actions, {"agent_5":1, focal.id:1})
    assert any("agent_5 fighted agent agent_1 for 5 damage" in o for o in cp.observations)
    assert any("agent_1 fighted agent agent_5 for 6 damage" in o for o in cp.observations)
    assert all(a.is_alive() for a in cp.social_environment.agents)
    # A private fact in a bystander's memory must never enter the focal prompt.
    old.get_agent(cp, "agent_4").memory.long_term_memory = {"Strategies": "PRIVATE_CANARY_BYSTANDER_4_C38V2_718219"}
    memory = blank_memory()
    for aid, relation, agreement in (("agent_2", "cooperative peer", "Received 3 HP at Step 1."),
                                    ("agent_3", "promise broken", "Promised 4 HP at Step 1, explicitly refused at Step 2; received 0 HP from agent_3."),
                                    ("agent_8", "unfair distributor", "Promised 5 HP at Step 1 but transferred only 1 HP at Step 2."),
                                    ("agent_5", "previous attacker; retaliation completed", "Attacked me for 5 damage at Step 1. I retaliated for 6 damage at Step 2."),
                                    ("agent_4", "helpful peer", "Received a separate 2 HP gift at Step 2.")):
        memory["Agent_Specific_Memory"][aid] = entity_memory(old.get_agent(cp, aid).type, relation, agreement)
    for aid, step, action, reason in (("agent_2",1,"allocate","Allocated 3 HP to agent_1."),
                                      ("agent_8",2,"allocate","Allocated 1 HP after promising 5 HP to agent_1."),
                                      ("agent_5",1,"fight","Attacked agent_1 successfully for 5 damage."),
                                      ("agent_4",2,"allocate","Allocated 2 HP to agent_1.")):
        memory["Agent_Specific_Memory"][aid]["important_interaction_history"]["what_he_did_to_me"] = recorded_event(step, action, reason, focal.type)
    memory["Agent_Specific_Memory"]["agent_5"]["important_interaction_history"]["what_i_did_to_him"] = recorded_event(2,"fight","Retaliated against agent_5 for 6 damage.",old.get_agent(cp,"agent_5").type)
    return cp, memory


def memory_facts():
    # Entity-scoped anchors avoid counting a number appearing elsewhere in a
    # rationale as factual memory. Exact paraphrases that miss these anchors are
    # exposed to human review rather than silently judged by this same model.
    def fact(name, entity, description, *patterns):
        return {"id":name, "entity":entity, "description":description, "patterns":list(patterns)}
    return [
        fact("cooperation_3hp","agent_2","agent_2 gave focal 3 HP at Step 1",r"(?:3|three)\s*(?:HP|health)",r"allocat|gave|giv|gift|shar",r"(?:step[_ ]*1|Step 1|time_step_1)"),
        fact("betrayal_promise_4","agent_3","agent_3 promised 4 HP",r"(?:4|four)\s*(?:HP|health)",r"promis|agree"),
        fact("betrayal_no_transfer","agent_3","agent_3 refused and transferred zero",r"refus|broke|broken|betray|reneg|withdrew",r"(?:0|zero)\s*HP|no (?:HP|allocat|transfer)|(?:did not|didn't|never) (?:allocat|giv|shar|transfer)|not (?:allocat|shar|giv|transfer)"),
        fact("unfair_promise_5","agent_8","agent_8 promised 5 HP",r"(?:5|five)\s*(?:HP|health)",r"promis|agree"),
        fact("unfair_actual_1","agent_8","agent_8 actually gave only 1 HP",r"(?:1|one)\s*(?:HP|health)",r"allocat|gave|giv|shar"),
        fact("attack_5damage","agent_5","agent_5 attacked focal for 5 damage at Step 1",r"(?:5|five)\s*(?:HP|damage|health)",r"attack|fight|fought",r"(?:step[_ ]*1|Step 1|time_step_1)"),
        fact("retaliation_6damage","agent_5","focal retaliated against agent_5 for 6 damage at Step 2",r"(?:6|six)\s*(?:HP|damage|health)",r"retaliat|what_i_did_to_him",r"(?:step[_ ]*2|Step 2|time_step_2)"),
        fact("decoy_2hp","agent_4","agent_4 gave 2 HP, not agent_2's 3 HP",r"(?:2|two)\s*(?:HP|health)",r"allocat|gave|giv|gift|shar"),
    ]


def c_ground_events():
    return [{"actor":actor,"recipient":recipient,"step":step,"action_type":kind,"success":True}
            for actor,recipient,step,kind in (("agent_2","agent_1",1,"allocate"),("agent_8","agent_1",2,"allocate"),
                ("agent_4","agent_1",2,"allocate"),("agent_5","agent_1",1,"fight"),("agent_1","agent_5",2,"fight"))]


def c_fixture(length, mechanism):
    # Calibration is performed before protocol freeze and stored, never adapted
    # to the model's behavioral answers. Every final prompt is token-counted.
    sections = {"short":0,"long":12,"near_budget":32}[length]
    common=OUT/"fixtures/memory_pure_validation/history"/length
    if (common/"checkpoint.json").exists():
        cp=Checkpoint.model_validate(read(common/"checkpoint.json"))
        memory=read(common/"seeded_memory.json")
    else:
        earlier=read(OUT/"fixtures/memory_pure_validation/C1"/length/"checkpoint.json")
        if earlier:
            cp=Checkpoint.model_validate(earlier)
            memory=copy.deepcopy(old.get_agent(cp,"agent_1").memory.long_term_memory)
            old.get_agent(cp,"agent_1").memory.long_term_memory={}
        else:
            cp,memory=build_c_history(sections)
            if length=="near_budget":
                # Engineering calibration only, before any behavioral output.
                # Preserve every generated message; choose legal message size
                # at fixture creation so both two-stage responses fit at 64K.
                for _ in range(5):
                    count=token_count(prepare_agent_prompts(cp,old.get_agent(cp,"agent_1")).messages)
                    if 39000<=count<=42500:
                        break
                    dense_chars=max(250,min(900,int(900*(41000-13500)/max(1,count-13500))))
                    cp,memory=build_c_history(sections,dense_chars)
        write(common/"checkpoint.json",json.loads(cp.model_dump_json()))
        write(common/"seeded_memory.json",memory)
    focal = old.get_agent(cp,"agent_1")
    if mechanism == "C1":
        focal.memory.long_term_memory = memory
    elif mechanism != "C2":
        raise ValueError(mechanism)
    messages = prepare_agent_prompts(cp,focal).messages
    prompt_tokens = token_count(messages)
    assert "PRIVATE_CANARY_BYSTANDER_4_C38V2_718219" not in json.dumps(messages)
    if length != "short" and prompt_tokens <= 16384:
        raise RuntimeError(f"Long-history fixture too short: {length} {prompt_tokens}")
    budget_check(prompt_tokens,"first_stage")
    return cp,{"mechanism":mechanism,"length":length,"focal_id":"agent_1","facts":memory_facts(),
               "events":c_ground_events(),"complete_event_steps":list(range(1,13)),"initial_tokens":prompt_tokens,
               "private_canary":"PRIVATE_CANARY_BYSTANDER_4_C38V2_718219", "target_fields_prepopulated":mechanism=="C1",
               "logistics_sections_per_message":sections,"actual_history_rounds":12,
               "history_observations":cp.observations,"paired_state_hash":physical_pair_hash(cp)}
