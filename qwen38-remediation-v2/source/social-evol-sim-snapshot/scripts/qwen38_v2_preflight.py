"""Executed engineering checks and independent recount of historical artifacts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from qwen38_v2_infrastructure import *

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import scr.simulation.agent_decision  # Initialize the repository's import cycle in its normal order.
from scr.simulation.act_manager.validator.deep_memory import validate_memory_structure


def historical_audit():
    legacy = ROOT / "data/qwen38_capability_evaluation_20261016"
    decisions = []
    for exp in ("A", "C"):
        for path in (legacy / exp).glob("*/result.json"):
            row = read(path, {})
            decision = row.get("decision", {})
            decisions.append((str(path.relative_to(ROOT)), decision))
    transfers = []
    for path in (legacy / "B/rounds").glob("step_*/result.json"):
        row = read(path, {})
        for agent in row.get("agent_decisions", []):
            decision = agent.get("decision", {})
            decisions.append((str(path.relative_to(ROOT)) + ":" + agent["agent_id"], decision))
            action = action_dict(decision.get("response"))
            if action.get("action_type") == "allocate":
                transfers.append({"step": row["step"], "actor": agent["agent_id"], "plan": action.get("allocation_plan"),
                                  "observation_evidence": [o for o in row.get("new_observations", []) if agent["agent_id"] + " allocated" in o]})
    structures = []
    legacy_c=[]
    import qwen38_capability_evaluation as old
    for path in (legacy/"C").glob("*/result.json"):
        row=read(path,{})
        score=old.score_memory_facts(row.get("decision",{}),row.get("initial",{}))
        legacy_c.append({"path":str(path.relative_to(ROOT)),"correct":score.get("correct",0),"total":score.get("total",4),
                         "slots":score.get("slots",{}),"note":"Recomputed historical heuristic, not independent semantic truth or acquisition accuracy"})
    for path, decision in decisions:
        response = decision.get("response")
        if response:
            structures.append({"path": path, "action_success": decision.get("success"),
                               **validate_memory_structure(response.get("long_term_memory"))})
    errors = jsonlines(legacy / "telemetry/call_errors.jsonl")
    context = [e for e in errors if any(x in str(e.get("error", "")).lower() for x in ("context size", "context length", "too many tokens", "n_ctx"))]
    result = {"at": now(), "decision_units": len(decisions), "memories_audited": len(structures),
              "memory_structure_valid": sum(x["valid"] for x in structures), "memory_structures": structures,
              "allocations": transfers, "allocation_actions": len(transfers),
              "planned_allocated_hp": sum(sum(t["plan"].values()) for t in transfers),
              "context_exceptions": len(context), "context_units": sorted({x.get("unit_id") for x in context}),
              "missing_major_section_units":sum(any(e["path"].startswith("long_term_memory.") for e in s["errors"]) for s in structures),
              "incomplete_agent_record_units":sum(any(e["path"].startswith("Agent_Specific_Memory.") and e["issue"]=="required field missing" for e in s["errors"]) for s in structures),
              "legacy_C_heuristic_recomputed":legacy_c,
              "legacy_C_heuristic_hits":sum(x["correct"] for x in legacy_c),"legacy_C_heuristic_total":sum(x["total"] for x in legacy_c),
              "focal_ids_by_moral_type": {"universal": "agent_1", "reciprocal": "agent_3", "kin": "agent_5", "selfish": "agent_7"},
              "note": "Historical recount is separate from corrected results. Full-round net HP is not an execution delta."}
    write(OUT / "historical_audit.json", result)
    return result


def freeze_historical_hashes():
    destination = OUT / "historical_sha256.json"
    if destination.exists():
        return
    roots = [ROOT / "data/qwen38_capability_evaluation_20261016", ROOT / "data/qwen38_full_baseline_20261016"]
    files = {str(p.relative_to(ROOT)): sha(p) for root in roots for p in root.rglob("*") if p.is_file()}
    write(destination, {"at": now(), "files": files})


def run_preflight():
    ensure_server()
    samples, stopping = [], threading.Event()

    def sample():
        while not stopping.is_set():
            try:
                samples.append({"at": now(), **gpu()})
            except Exception:
                pass
            stopping.wait(2)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    trials = []
    try:
        for mode in ("off", "low"):
            path = OUT / f"preflight/native_{mode}.json"
            if path.exists() and read(path, {}).get("passed"):
                trials.append(read(path))
                continue
            messages = [{"role": "system", "content": "Return a JSON object with answer and explanation. Put internal thinking in the native reasoning channel."},
                        {"role": "user", "content": "Three hunters split 17 HP fairly with integer shares. Give a valid split summing to 17. Explain the one-HP imbalance."}]
            request = {"model": "Qwen3.8-27B-Q4_K_M", "messages": messages,
                       "response_format": {"type": "json_object"}, "max_tokens": 3072,
                       "seed": 1701, "temperature": .7, "top_p": .9, "top_k": 20, **reasoning_settings(mode)}
            started = time.time()
            response = api("/v1/chat/completions", request, timeout=3600)
            message = response["choices"][0]["message"]
            parsed = json.loads(message["content"])
            reasoning = message.get("reasoning_content") or ""
            result = {"at": now(), "mode": mode, "request": request, "raw_response": response,
                      "elapsed_seconds": time.time()-started, "json_parsed": parsed,
                      "native_reasoning_present": bool(reasoning),
                      "reasoning_separate_from_json": "<think>" not in message["content"],
                      "passed": bool(parsed) and (bool(reasoning) == (mode == "low"))}
            write(path, result)
            trials.append(result)
        for target in (48000, 60000):
            path = OUT / f"preflight/context_{target}.json"
            if path.exists() and read(path, {}).get("passed"):
                trials.append(read(path))
                continue
            header = "Return JSON containing first_marker, middle_marker, last_marker copied from the records. first_marker=amber_93817.\n"
            filler = "\n".join(f"Neutral ledger row {i}: plain stone recorded near the river; no social interaction or resource transfer." for i in range(6000))
            low, high = 1, len(filler)
            while high-low > 200:
                middle = (low+high)//2
                content = header + filler[:middle//2] + "\nmiddle_marker=cedar_41629\n" + filler[middle//2:middle] + "\nlast_marker=violet_27503"
                messages = [{"role": "user", "content": content}]
                count = token_count(messages)
                if count < target:
                    low = middle
                else:
                    high = middle
            prompt = header + filler[:low//2] + "\nmiddle_marker=cedar_41629\n" + filler[low//2:low] + "\nlast_marker=violet_27503"
            request = {"model": "Qwen3.8-27B-Q4_K_M", "messages": [{"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}, "max_tokens": 128, "seed": 1701,
                       "temperature": .7, "top_p": .9, "top_k": 20, **reasoning_settings("off")}
            estimate = token_count(request["messages"])
            start = time.time()
            before = metrics()
            response = api("/v1/chat/completions", request, timeout=3600)
            parsed = json.loads(response["choices"][0]["message"]["content"])
            result = {"at": now(), "target_prompt_tokens": target, "estimated_tokens": estimate,
                      "request": request, "raw_response": response, "elapsed_seconds": time.time()-start,
                      "metrics_before": before, "metrics_after": metrics(), "slots": api("/slots"),
                      "passed": parsed == {"first_marker": "amber_93817", "middle_marker": "cedar_41629", "last_marker": "violet_27503"},
                      "gpu": gpu()}
            write(path, result)
            trials.append(result)
    finally:
        stopping.set()
        sampler.join(5)
        write(OUT / "preflight/gpu_samples.json", samples)
    write(OUT / "preflight/summary.json", {"at": now(), "context": CONTEXT, "trials": [{k:v for k,v in t.items() if k not in ("request", "raw_response")} for t in trials],
                                           "all_passed": all(t["passed"] for t in trials)})
    return trials


if __name__ == "__main__":
    freeze_historical_hashes()
    audit = historical_audit()
    print(json.dumps({"historical_units": audit["decision_units"], "allocated_hp": audit["planned_allocated_hp"],
                      "allocation_actions": audit["allocation_actions"], "context_errors": audit["context_exceptions"]}), flush=True)
    run_preflight()
    print("Preflight complete; see preflight/summary.json", flush=True)
