"""Offline integrity pass for the local Qwen capability-evaluation artifacts.

The supervising runner can be restarted while work is in progress. This final
pass reloads durable request/response pairs, normalizes draft/reflection stage
labels from the actual prompts, rebuilds derived results, and reruns the audit.
It performs no model inference.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "qwen38_capability_evaluation", ROOT / "scripts" / "qwen38_capability_evaluation.py"
)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

module.REPORT_TEMPLATE = module.REPORT_PATH.read_text(encoding="utf-8")
module.STATE = module.read_json(module.STATE_PATH, {}) or {}
completion_statuses = {
    "complete", "complete_with_blockers", "complete_with_integrity_warnings",
    "blocked_with_unresolved_units",
}
counts_at_start = module.progress_counts(module.STATE)
all_units_resolved_at_start = (
    counts_at_start["A_complete"] + counts_at_start["A_blocked"] == 40
    and counts_at_start["B_rounds_complete"] + counts_at_start["B_rounds_blocked"] == 6
    and counts_at_start["C_complete"] + counts_at_start["C_blocked"] == 8
)
if module.STATE.get("status") not in completion_statuses and not (
        module.STATE.get("status") in {"running", "crashed_recoverable"} and all_units_resolved_at_start):
    raise SystemExit(f"Evaluation is not finalized yet (status={module.STATE.get('status')!r}).")

updated = {"A": 0, "B_rounds": 0, "C": 0}

# The runner's one-second monitor can be delayed by a long report render. The
# detached watchdog remains independent and supplies the missing VRAM samples.
existing_gpu_samples = {
    (sample.get("at"), sample.get("used_mib"))
    for sample in module.read_jsonl(module.GPU_JSONL)
}
merged_gpu_samples = 0
for sample in module.read_jsonl(module.OUT / "telemetry" / "independent_monitor.jsonl"):
    gpu = sample.get("gpu") or {}
    if sample.get("event") != "heartbeat" or not isinstance(gpu.get("used_mib"), (int, float)):
        continue
    record = {
        "at": sample.get("at"), "used_mib": gpu.get("used_mib"),
        "total_mib": gpu.get("total_mib"), "utilization_percent": gpu.get("utilization_percent"),
        "power_watts": gpu.get("power_watts"), "source": "independent_watchdog",
    }
    key = (record["at"], record["used_mib"])
    if key not in existing_gpu_samples:
        module.append_jsonl(module.GPU_JSONL, record)
        existing_gpu_samples.add(key)
        merged_gpu_samples += 1

for experiment, results in (("A", module.read_a_results()), ("C", module.read_c_results())):
    for item in results:
        decision = item.get("decision")
        if not isinstance(decision, dict):
            continue
        initial = item.get("initial", {})
        case_id = initial.get("case_id")
        if not case_id:
            continue
        decision["calls"] = module.unit_call_records(experiment, case_id)
        item["outcome"] = module.decision_outcome(decision, initial)
        if experiment == "C":
            item["outcome"]["memory_facts"] = module.score_memory_facts(decision, initial)
        module.write_json(module.OUT / experiment / case_id / "result.json", item)
        updated[experiment] += 1

for result in module.read_b_rounds():
    step = result.get("step")
    if not isinstance(step, int):
        continue
    for row in result.get("agent_decisions", []):
        decision = row.get("decision") or {}
        agent_id = row.get("agent_id")
        if not agent_id:
            continue
        decision["calls"] = module.unit_call_records("B", f"B-round-{step}-agent-{agent_id}")
        row["decision"] = decision
    successes = [row.get("decision", {}) for row in result.get("agent_decisions", [])
                 if row.get("decision", {}).get("success")]
    result["reflection_stages_present"] = all(
        {"first_stage", "reflection"}.issubset({c.get("stage") for c in d.get("calls", [])})
        for d in successes
    )
    module.write_json(module.OUT / "B" / "rounds" / f"step_{step}" / "result.json", result)
    updated["B_rounds"] += 1

module.write_blinded_review_tables()
# Match the evaluation preflight's import order; the repository validator and
# response processor otherwise encounter their existing circular import.
import scr.simulation.agent_decision  # noqa: F401
audit = module.audit_evaluation_artifacts()
module.write_json(module.OUT / "postfinalization_audit.json", {
    "updated_result_groups": updated,
    "normalized_call_count": len(module.read_call_results()),
    "merged_independent_gpu_samples": merged_gpu_samples,
    "artifact_audit": audit,
})

counts = module.progress_counts(module.STATE)
all_resolved = (
    counts["A_complete"] + counts["A_blocked"] == 40
    and counts["B_rounds_complete"] + counts["B_rounds_blocked"] == 6
    and counts["C_complete"] + counts["C_blocked"] == 8
)
has_blockers = bool(counts["A_blocked"] or counts["B_rounds_blocked"] or counts["C_blocked"])
if all_resolved:
    module.STATE["status"] = (
        "complete_with_blockers" if has_blockers else
        "complete" if audit.get("passed") else
        "complete_with_integrity_warnings"
    )
    finalized_at = module.utc_now()
    module.STATE.update({
        "phase": "complete", "current_unit": None, "current_call": None,
        "current_call_started_epoch": None, "finished_at": finalized_at,
        "updated_at": finalized_at, "last_activity_at": finalized_at,
        "offline_finalization_audit_at": audit.get("at"),
    })
    module.write_json(module.OUT / "heartbeat.json", {
        "at": finalized_at, "event": "offline_finalization", "status": module.STATE["status"],
        "phase": "complete", "current_unit": None,
    })
module.write_json(module.STATE_PATH, module.STATE)
module.refresh_report()
print({"updated": updated, "normalized_calls": len(module.read_call_results()),
       "audit_passed": audit.get("passed"), "audit_errors": audit.get("errors"),
       "status": module.STATE.get("status")})
