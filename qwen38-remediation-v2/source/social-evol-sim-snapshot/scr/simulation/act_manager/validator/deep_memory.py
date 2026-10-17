"""Read-only audits of the original reflection memory format.

These diagnostics deliberately do NOT change action validation, reject an action,
repair a memory, or trigger corrective prompting. Factual checks report their
coverage; absence of an automated finding is not proof of semantic correctness.
"""
from __future__ import annotations

import json
import re
from typing import Any

MAJOR_SECTIONS = (
    "Prey_Hunting_Collaboration_Distribution_Retaliation_Memory_And_Planning",
    "Agent_Specific_Memory", "Family_Plan", "Plan_For_Reproduction", "Strategies",
)
PREY_FIELDS = {
    "hunt_fact_history_of_this_prey": (),
    "communication_and_planning_before_killing_prey": (
        "amount_of_reward", "who_communicated_to_hunt_together", "who_I_want_to_collaborate",
        "mutually_confirmed_agents_for_collaboration", "anyone_wants_me_to_not_hunt_this_prey",
        "my_own_distribution_plan"),
    "distribution_after_killing_prey": (
        "time_step_killed_prey", "winner", "reward_redistributed_yet", "time_passed_unallocated",
        "judge_if_winner_still_planning_to_share", "actual_reward_allocation_by_winner",
        "evaluating_the_redistribution", "is_fair_allocation_by_winner", "free_rider_winner"),
    "plan_next": ("thinking", "stage", "plan", "retaliation_plan"),
    "afterward_happenings": ("thinking", "retaliation_events", "other_events"),
    "lessons_learned": (),
}
AGENT_FIELDS = ("important_interaction_history", "thinking", "moral_type", "relationship", "agreement", "plan")
EVENT_FIELDS = ("action_type", "if_success", "reason", "target_moral_type")
TIME_KEY = re.compile(r"^time_step_(\d+)$")


def event_records(history: Any):
    """Yield (step, event) for flat ORIGINAL records and optional keyed records.

    Original example: {"time_step_3": 3, "action_type": "allocate", ...}.
    A nested {"time_step_3": {"time_step": 3, ...}} is also interpretable;
    nesting is never required. Empty history is allowed when nothing occurred.
    """
    if not isinstance(history, dict) or not history:
        return
    keys = [(k, TIME_KEY.fullmatch(k)) for k in history if isinstance(k, str)]
    keys = [(k, m) for k, m in keys if m]
    for key, match in keys:
        value = history[key]
        yield int(match.group(1)), value if isinstance(value, dict) else history
    if not keys and "time_step" in history:
        yield history.get("time_step"), history


def validate_memory_structure(memory: Any) -> dict:
    errors: list[dict] = []

    def error(path, issue):
        errors.append({"path": path, "issue": issue})

    def fields(value, required, path):
        # The original reflection explicitly permits blank strings for no content.
        if value == "":
            return False
        if not isinstance(value, dict):
            error(path, "expected object or an explicitly blank no-content string")
            return False
        for name in required:
            if name not in value:
                error(f"{path}.{name}", "required field missing")
        return True

    def typed(record, names, expected, path, choices=None):
        if not isinstance(record, dict):
            return
        for name in names:
            if name not in record or record[name] == "":
                continue
            value = record[name]
            if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
                error(path + "." + name, "incorrect field type: expected " + expected.__name__)
            elif choices is not None and value not in choices:
                error(path + "." + name, "value outside reflection prompt's allowed choices")

    if not isinstance(memory, dict):
        return {"valid": False, "errors": [{"path": "long_term_memory", "issue": "expected object"}]}
    fields(memory, MAJOR_SECTIONS, "long_term_memory")
    for section in MAJOR_SECTIONS[:3]:
        if section in memory and memory[section] != "" and not isinstance(memory[section], dict):
            error(section, "expected object or blank no-content string")
    agents = memory.get("Agent_Specific_Memory")
    if isinstance(agents, dict):
        for aid, record in agents.items():
            path = f"Agent_Specific_Memory.{aid}"
            if not fields(record, AGENT_FIELDS, path):
                continue
            typed(record, ("thinking", "moral_type", "relationship", "agreement", "plan"), str, path)
            history = record.get("important_interaction_history", "")
            if fields(history, ("what_i_did_to_him", "what_he_did_to_me"), path + ".important_interaction_history"):
                for direction in ("what_i_did_to_him", "what_he_did_to_me"):
                    container = history.get(direction, "")
                    if container in ({}, ""):
                        continue
                    ep = path + ".important_interaction_history." + direction
                    if not isinstance(container, dict):
                        error(ep, "expected flat event object, keyed events, or empty history")
                        continue
                    events = list(event_records(container))
                    if not events:
                        error(ep, "nonempty event has no time_step_<int> field")
                    for step, event in events:
                        fields(event, EVENT_FIELDS, ep)
                        if not isinstance(step, int) or isinstance(step, bool):
                            error(ep, "event time must be an integer")
                        flat_time = container.get(f"time_step_{step}")
                        if flat_time is not None and not isinstance(flat_time, dict):
                            if not isinstance(flat_time, int) or isinstance(flat_time, bool) or flat_time != step:
                                error(ep, "flat time_step_<int> value must equal the step integer")
                        if "if_success" in event and not isinstance(event["if_success"], bool):
                            error(ep + ".if_success", "expected boolean")
                        for name in ("action_type", "reason", "target_moral_type"):
                            if name in event and not isinstance(event[name], str):
                                error(ep + "." + name, "expected string")
                        typed(event, ("action_type",), str, ep, ("fight", "rob", "allocate"))
    prey = memory.get(MAJOR_SECTIONS[0])
    if isinstance(prey, dict):
        for pid, record in prey.items():
            path = MAJOR_SECTIONS[0] + "." + pid
            if not fields(record, PREY_FIELDS, path):
                continue
            for section, required in PREY_FIELDS.items():
                if required:
                    fields(record.get(section, ""), required, path + "." + section)
            hunting = record.get("hunt_fact_history_of_this_prey")
            if isinstance(hunting, dict):
                for aid, event in hunting.items():
                    fields(event, ("time_step", "result", "damage", "if_killed"), path + ".hunt_fact_history_of_this_prey." + aid)
                    typed(event, ("time_step", "damage"), int, path + ".hunt_fact_history_of_this_prey." + aid)
                    typed(event, ("if_killed",), bool, path + ".hunt_fact_history_of_this_prey." + aid)
            plan = record.get("plan_next")
            if isinstance(plan, dict) and "retaliation_plan" in plan:
                fields(plan["retaliation_plan"], ("collaboration_plan", "retaliation_method", "retaliation_goal"), path + ".plan_next.retaliation_plan")
                typed(plan["retaliation_plan"], ("collaboration_plan",), list, path + ".plan_next.retaliation_plan")
                typed(plan["retaliation_plan"], ("retaliation_method",), str, path + ".plan_next.retaliation_plan", ("rob", "fight"))
                typed(plan["retaliation_plan"], ("retaliation_goal",), str, path + ".plan_next.retaliation_plan")
                typed(plan, ("thinking", "plan"), str, path + ".plan_next")
                typed(plan, ("stage",), str, path + ".plan_next", ("closed_with_fair_share", "keep_hunting", "wait_and_ask_for_sharing", "warn_and_plan_for_retaliation", "execute_retaliation", "finished_retaliation", "give_up_retaliation"))
            communication = record.get("communication_and_planning_before_killing_prey")
            if isinstance(communication, dict):
                typed(communication, ("amount_of_reward",), int, path + ".communication_and_planning_before_killing_prey")
                typed(communication, ("who_communicated_to_hunt_together", "who_I_want_to_collaborate", "mutually_confirmed_agents_for_collaboration"), list, path + ".communication_and_planning_before_killing_prey")
                distribution = communication.get("my_own_distribution_plan", "")
                fields(distribution, ("thinking", "share_method"), path + ".communication_and_planning_before_killing_prey.my_own_distribution_plan")
                typed(distribution, ("thinking",), str, path + ".communication_and_planning_before_killing_prey.my_own_distribution_plan")
                typed(distribution, ("share_method",), str, path + ".communication_and_planning_before_killing_prey.my_own_distribution_plan", ("fair_to_all_collaborator", "only_to_my_allies_in_this_hunt", "all_to_self"))
                objections = communication.get("anyone_wants_me_to_not_hunt_this_prey", {})
                if isinstance(objections, dict):
                    for aid, objection in objections.items():
                        fields(objection, ("why", "ignore_or_follow", "if_he_hunted_do_I_share"), path + ".communication_and_planning_before_killing_prey.anyone_wants_me_to_not_hunt_this_prey." + aid)
                        typed(objection, ("why", "ignore_or_follow"), str, path + ".communication_and_planning_before_killing_prey.anyone_wants_me_to_not_hunt_this_prey." + aid)
                        typed(objection, ("if_he_hunted_do_I_share",), bool, path + ".communication_and_planning_before_killing_prey.anyone_wants_me_to_not_hunt_this_prey." + aid)
            distribution = record.get("distribution_after_killing_prey")
            dp = path + ".distribution_after_killing_prey"
            typed(distribution, ("time_step_killed_prey", "time_passed_unallocated"), int, dp)
            typed(distribution, ("reward_redistributed_yet",), bool, dp)
            typed(distribution, ("winner", "judge_if_winner_still_planning_to_share", "evaluating_the_redistribution"), str, dp)
            typed(distribution, ("is_fair_allocation_by_winner", "free_rider_winner"), str, dp, ("true", "false", "NA"))
            typed(distribution, ("actual_reward_allocation_by_winner",), dict, dp)
            afterward = record.get("afterward_happenings")
            typed(afterward, ("thinking", "other_events"), str, path + ".afterward_happenings")
            typed(afterward, ("retaliation_events",), dict, path + ".afterward_happenings")
            typed(record, ("lessons_learned",), str, path)
    family = memory.get("Family_Plan")
    if isinstance(family, dict):
        for aid, record in family.items():
            fields(record, ("status", "plan"), "Family_Plan." + aid)
            typed(record, ("status", "plan"), str, "Family_Plan." + aid)
    if "Plan_For_Reproduction" in memory:
        reproduction = memory["Plan_For_Reproduction"]
        fields(reproduction, ("thinking", "preconditions_and_subgoals", "estimated_time_to_produce_next_child"), "Plan_For_Reproduction")
        if isinstance(reproduction, dict) and "estimated_time_to_produce_next_child" in reproduction:
            value = reproduction["estimated_time_to_produce_next_child"]
            if value != "" and (not isinstance(value, int) or isinstance(value, bool)):
                error("Plan_For_Reproduction.estimated_time_to_produce_next_child", "expected integer or blank")
    if "Strategies" in memory and not isinstance(memory["Strategies"], str):
        error("Strategies", "expected string")
    return {"valid": not errors, "errors": errors, "schema_source": "original async_response_processor reflection prompt; flat events accepted"}


def memory_events(memory, focal_id):
    entities = memory.get("Agent_Specific_Memory", {}) if isinstance(memory, dict) else {}
    if not isinstance(entities, dict):
        return
    for aid, record in entities.items():
        if not isinstance(record, dict):
            continue
        history = record.get("important_interaction_history", {})
        if not isinstance(history, dict):
            continue
        for direction in ("what_i_did_to_him", "what_he_did_to_me"):
            for step, event in event_records(history.get(direction, {})):
                yield {"step": step, "actor": focal_id if direction == "what_i_did_to_him" else aid,
                       "recipient": aid if direction == "what_i_did_to_him" else focal_id,
                       "action_type": event.get("action_type"), "success": event.get("if_success"),
                       "record": event, "entity": aid, "direction": direction}


def validate_memory_fidelity(memory, previous, *, focal_id, events=(), facts=(), known_agents=(), current_step=None,
                             complete_event_steps=(), family_ids=()):
    """Separate deterministic factual checks from candidates needing human review.

    `facts` are predeclared regex/field anchors, scoped to a specific participant.
    Exact preservation is a measurement, not an instruction sent to the model.
    """
    memory = memory if isinstance(memory, dict) else {}
    previous = previous if isinstance(previous, dict) else {}
    findings, candidates, scored = [], [], []
    entities = memory.get("Agent_Specific_Memory", {})
    if not isinstance(entities, dict):
        entities = {}
    known = set(known_agents)
    records = list(memory_events(memory, focal_id))
    for record in records:
        if known and (record["actor"] not in known or record["recipient"] not in known):
            findings.append({"kind": "incorrect_participant", "event": record})
        if current_step is not None and isinstance(record["step"], int) and record["step"] >= current_step:
            findings.append({"kind": "invented_or_premature_completed_event", "event": record})
        matches = [e for e in events if e.get("step") == record["step"] and e.get("actor") == record["actor"]
                   and e.get("recipient") == record["recipient"] and e.get("action_type") == record["action_type"]]
        if not matches and record["step"] in complete_event_steps:
            findings.append({"kind": "unsupported_event_or_wrong_attribution", "event": record})
        elif matches and all(e.get("success") is not None and e["success"] != record["success"] for e in matches):
            findings.append({"kind": "contradictory_event_outcome", "event": record})
    prior_records = list(memory_events(previous, focal_id))
    for prior in prior_records:
        matched = [r for r in records if all(r[k] == prior[k] for k in ("actor", "recipient", "step", "action_type"))]
        if not matched:
            candidates.append({"kind": "dropped_structured_event_check_paraphrase", "event": prior})
        elif any(r["success"] != prior["success"] for r in matched):
            findings.append({"kind": "changed_previous_event_outcome", "previous": prior, "updated": matched})
    prior_entities = previous.get("Agent_Specific_Memory", {})
    for aid, record in entities.items():
        if known and aid not in known:
            findings.append({"kind": "invented_participant", "entity": aid})
        if not isinstance(record, dict):
            continue
        relationship = str(record.get("relationship", ""))
        if re.search(r"\bfamily\b|\b(?:my )?(?:child|parent|offspring)\b", relationship, re.I) and aid not in set(family_ids):
            candidates.append({"kind": "unsupported_family_relationship_check_language", "entity": aid, "relationship": relationship})
        old = prior_entities.get(aid, {}) if isinstance(prior_entities, dict) else {}
        if isinstance(old, dict) and old.get("relationship") and old.get("relationship") != relationship:
            relevant = any(aid in (e.get("actor"), e.get("recipient")) for e in events)
            if not relevant:
                candidates.append({"kind": "unsupported_relationship_update", "entity": aid, "before": old.get("relationship"), "after": relationship})
    for fact in facts:
        entity = fact.get("entity")
        scope = entities.get(entity, {}) if entity else memory
        text = json.dumps(scope, ensure_ascii=False)
        expressions = fact.get("patterns", [])
        hits = [bool(re.search(pattern, text, re.I | re.S)) for pattern in expressions]
        preserved = bool(hits) and all(hits)
        alternatives = []
        if entity and not preserved:
            for other, value in entities.items():
                if other != entity and all(re.search(p, json.dumps(value), re.I | re.S) for p in expressions):
                    alternatives.append(other)
        scored.append({"fact_id": fact["id"], "expected": fact.get("description"), "entity": entity,
                       "automated_anchor_present": preserved, "matched_patterns": hits,
                       "other_entities_matching": alternatives, "evidence": text[:2400],
                       "judgment_type": "predeclared factual anchor; ambiguous paraphrases require blinded human review"})
    return {"deterministic_consistent": not findings, "findings": findings, "review_candidates": candidates,
            "facts": scored, "fact_hits": sum(f["automated_anchor_present"] for f in scored), "fact_total": len(scored),
            "structured_events_before": len(prior_records), "structured_events_after": len(records),
            "coverage": "structured event/participant/outcome checks and declared factual anchors; semantic hallucinations not exhaustively decidable"}
