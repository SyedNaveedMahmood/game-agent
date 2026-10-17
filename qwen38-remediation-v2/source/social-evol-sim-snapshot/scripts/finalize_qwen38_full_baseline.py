"""Offline integrity and inference-stage audit for the long Qwen baseline run."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "qwen38_full_baseline_20261016"
REPORT = ROOT / "reports" / "QWEN38_CAPABILITY_EVALUATION.md"
START = "<!-- QWEN38_FULL_BASELINE_START -->"
END = "<!-- QWEN38_FULL_BASELINE_END -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


def locate_request(record: dict[str, Any]) -> Path | None:
    match = re.fullmatch(r"B-round-(\d+)-agent-(.+)", str(record.get("unit_id", "")))
    if not match:
        return None
    tick, agent_id = int(match.group(1)), match.group(2)
    seq = int(record.get("sequence", 0))
    outer = int(record.get("outer_attempt", 1))
    retry = int(record.get("retry_attempt", 1))
    base = OUT / "ticks" / f"step_{tick:03d}" / f"agent_{agent_id}"
    pattern = f"round_attempt_*/calls/call_{seq:03d}_outer_{outer}_retry_{retry}_*_request.json"
    matches = list(base.glob(pattern))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def classify_stage(request: dict[str, Any] | None) -> str | None:
    if not request:
        return None
    messages = request.get("messages", [])
    # Retry calls include earlier reflection prompts in the conversation history.
    # Classify this request by the latest user instruction, not any prior turn.
    last_user = next((message for message in reversed(messages if isinstance(messages, list) else [])
                      if message.get("role") == "user"), None)
    content = last_user.get("content", "") if last_user else ""
    if isinstance(content, str) and content.lstrip().startswith("Reflect:"):
        return "reflection"
    return "first_stage"


def memory_independent(checkpoint) -> bool:
    agents = checkpoint.social_environment.agents
    return (len({id(a.memory) for a in agents}) == len(agents)
            and len({id(a.memory.long_term_memory) for a in agents}) == len(agents))


def relationships_consistent(checkpoint) -> bool:
    agents = {a.id: a for a in checkpoint.social_environment.agents}
    for agent in agents.values():
        parent_id = agent.family.parent_id
        if parent_id and (parent_id not in agents or agent.id not in agents[parent_id].family.children_ids):
            return False
        for child_id in agent.family.children_ids:
            if child_id not in agents or agents[child_id].family.parent_id != agent.id:
                return False
    return True


def audit_server_log_window() -> dict[str, Any]:
    """Count llama-server truncation releases from this run's log-time window.

    llama.cpp log timestamps are monotonic ``minutes.seconds.millis.micros``.
    The run-start manifest maps that clock to the simulation start. If the
    server restarted, this single-process mapping is incomplete and is reported
    as such rather than treating a partial log as proof of no truncation.
    """
    manifest = read_json(OUT / "telemetry" / "server_log_window.json", {}) or {}
    state = read_json(OUT / "progress.json", {}) or {}
    log_path = ROOT / manifest.get("log_path", ".local/llama-server.log")
    try:
        start_elapsed = float(manifest["log_elapsed_seconds_at_run_start"])
    except (KeyError, TypeError, ValueError):
        start_elapsed = None
    complete = bool(start_elapsed is not None and log_path.is_file()
                    and int(state.get("server_restarts", 0)) == 0)
    releases = truncated = 0
    if start_elapsed is not None and log_path.is_file():
        stamp_re = re.compile(r"^(\d+)\.(\d{1,2})\.(\d{3})\.(\d{3})\b")
        release_re = re.compile(r"stop processing: n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*(\d+)")
        try:
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                stamp = stamp_re.match(line)
                release = release_re.search(line)
                if not stamp or not release:
                    continue
                minutes, seconds, millis, micros = (int(part) for part in stamp.groups())
                elapsed = minutes * 60 + seconds + millis / 1000 + micros / 1_000_000
                if elapsed < start_elapsed:
                    continue
                releases += 1
                truncated += int(release.group(2) != "0")
        except OSError:
            complete = False
    return {"complete": complete, "release_count": releases,
            "truncated_release_count": truncated,
            "server_restarts": int(state.get("server_restarts", 0)),
            "log_path": str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),
            "window_start_elapsed_seconds": start_elapsed}


def audit() -> dict[str, Any]:
    from scr.models.agent.responses import Response
    from scr.models.simulation.checkpoint import Checkpoint
    from scr.simulation.act_manager.validator.validator import validate_llm_response

    state = read_json(OUT / "progress.json", {}) or {}
    terminal = state.get("status") in {"complete", "blocked"} or (OUT / "supervisor_blocker.json").exists()
    if not terminal:
        raise RuntimeError(f"Refusing final audit while runner status is {state.get('status')!r}.")

    errors: list[str] = []
    tick_results = []
    for path in sorted((OUT / "ticks").glob("step_*/result.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
        result = read_json(path)
        if result:
            tick_results.append((path, result))

    raw_calls = read_jsonl(OUT / "telemetry" / "completions.jsonl")
    classified = []
    missing_prompts = []
    calls_by_unit_attempt: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in raw_calls:
        request_path = locate_request(record)
        request = read_json(request_path) if request_path else None
        stage = classify_stage(request)
        row = dict(record)
        row["stage_recorded_by_hook"] = record.get("stage")
        row["stage"] = stage or "unresolved"
        row["raw_request_path"] = str(request_path.relative_to(ROOT)) if request_path else None
        if not request_path:
            missing_prompts.append({"unit_id": record.get("unit_id"), "sequence": record.get("sequence"),
                                    "outer_attempt": record.get("outer_attempt"), "retry_attempt": record.get("retry_attempt")})
        classified.append(row)
        calls_by_unit_attempt[(str(record.get("unit_id")), int(record.get("outer_attempt", 1)))].append(row)

    successes = failures = response_revalidated = reflection_pair_missing = 0
    action_counts = Counter()
    persistent_failures = []
    for tick_path, result in tick_results:
        tick = int(result["tick"])
        before_path = tick_path.parent / "before.json"
        after_path = tick_path.parent / "after.json"
        try:
            before = Checkpoint.model_validate_json(before_path.read_text(encoding="utf-8"))
            after = Checkpoint.model_validate_json(after_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Tick {tick}: checkpoint parse failed: {type(exc).__name__}: {exc}")
            continue
        if after.metadata.current_time_step != tick + 1:
            errors.append(f"Tick {tick}: after-checkpoint is at step {after.metadata.current_time_step}.")
        if not memory_independent(after):
            errors.append(f"Tick {tick}: memory objects are not independent.")
        if not relationships_consistent(after):
            errors.append(f"Tick {tick}: relationship links are inconsistent.")
        by_id = {a.id: a for a in before.social_environment.agents}
        for decision in result.get("decisions", []):
            if not decision.get("success"):
                failures += 1
                persistent_failures.append({"tick": tick, "agent_id": decision.get("agent_id"), "error": decision.get("error")})
                continue
            successes += 1
            agent_id = decision.get("agent_id")
            response_dump = decision.get("response")
            if not response_dump:
                errors.append(f"Tick {tick}, agent {agent_id}: successful decision is missing response JSON.")
                continue
            try:
                response = Response.model_validate(response_dump)
                action_counts[response.action.root.action_type] += 1
                serialized = response.model_dump(mode="json")
                matching = []
                attempt = int(decision.get("outer_attempt", 1))
                unit_id = f"B-round-{tick}-agent-{agent_id}"
                for call in calls_by_unit_attempt.get((unit_id, attempt), []):
                    if call.get("stage") != "reflection" or not call.get("validation", {}).get("valid"):
                        continue
                    try:
                        raw = json.loads(call.get("content", ""))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if raw == serialized:
                        matching.append(call)
                if not matching:
                    errors.append(f"Tick {tick}, agent {agent_id}: no saved valid reflection output matches final action response.")
                    reflection_pair_missing += 1
                else:
                    call = matching[-1]
                    validation = validate_llm_response(call["content"], before, agent_id=agent_id)
                    if not validation.success:
                        errors.append(f"Tick {tick}, agent {agent_id}: offline action validation failed: {validation.error_message}")
                    else:
                        response_revalidated += 1
            except Exception as exc:
                errors.append(f"Tick {tick}, agent {agent_id}: Pydantic response parse failed: {type(exc).__name__}: {exc}")

    stage_counts = Counter(call.get("stage", "unresolved") for call in classified)
    invalid_calls = [call for call in classified if not call.get("validation", {}).get("valid")]
    context_truncations = [call for call in classified if call.get("server_context", {}).get("truncated") is True]
    call_exceptions = []
    for path in (OUT / "ticks").glob("step_*/agent_*/round_attempt_*/calls/*_exception.json"):
        item = read_json(path, {}) or {}
        call_exceptions.append(item)
    context_errors = [x for x in call_exceptions if "ContextWindowExceeded" in str(x.get("error", ""))]
    gpu = [x for x in read_jsonl(OUT / "telemetry" / "gpu_samples.jsonl") if x.get("used_mib") is not None]
    output_tokens = sum(int(x.get("output_tokens") or 0) for x in classified)
    input_tokens = sum(int(x.get("input_tokens") or 0) for x in classified)
    gen_seconds = sum(float(x.get("reported_duration_seconds") or x.get("observed_duration_seconds") or 0) for x in classified)
    output_tps = output_tokens / gen_seconds if gen_seconds else None
    context_rows = []
    for call in classified:
        context = call.get("server_context") or {}
        try:
            capacity = int(context["n_ctx"])
            prompt_tokens = int(call["input_tokens"])
            generated_tokens = int(call.get("output_tokens") or 0)
            slot_sequence_tokens = int(context["n_prompt_tokens"])
        except (KeyError, TypeError, ValueError):
            continue
        if prompt_tokens >= 0 and generated_tokens >= 0 and capacity > 0:
            context_rows.append((prompt_tokens, generated_tokens, slot_sequence_tokens, capacity))
    sequence_utilization = [(slot_tokens / capacity) for _, _, slot_tokens, capacity in context_rows]
    context_server_count_discrepancies = sum(
        abs(slot_tokens - (prompt_tokens + generated_tokens)) > 2
        for prompt_tokens, generated_tokens, slot_tokens, _ in context_rows
    )
    server_log_audit = audit_server_log_window()
    wall_runtime_seconds = None
    try:
        started = float(state["started_epoch"])
        finished = datetime.fromisoformat(state["finished_at"].replace("Z", "+00:00")).timestamp()
        wall_runtime_seconds = max(0.0, finished - started)
    except (KeyError, TypeError, ValueError, OSError):
        pass
    final_cp_path = OUT / "final.json"
    if final_cp_path.exists():
        try:
            final_cp = Checkpoint.model_validate_json(final_cp_path.read_text(encoding="utf-8"))
            final_population = dict(Counter(a.type for a in final_cp.social_environment.agents))
            final_agent_count = len(final_cp.social_environment.agents)
            final_step = final_cp.metadata.current_time_step
        except Exception as exc:
            errors.append(f"Final checkpoint parse failed: {type(exc).__name__}: {exc}")
            final_population, final_agent_count, final_step = {}, None, None
    else:
        final_population, final_agent_count, final_step = {}, None, None

    total_decision_slots = sum(int(r.get("decision_slots", 0)) for _, r in tick_results)
    completed_ticks = len(tick_results)
    if state.get("status") == "complete" and completed_ticks != int(state.get("completed_ticks", 0)):
        errors.append(f"Tick result count {completed_ticks} != progress committed count {state.get('completed_ticks')}.")
    committed = read_json(OUT / "last_committed.json")
    if state.get("status") == "complete" and committed:
        try:
            cp = Checkpoint.model_validate(committed["checkpoint"])
            if cp.metadata.current_time_step != final_step:
                errors.append("Final checkpoint and last committed checkpoint disagree on current step.")
        except Exception as exc:
            errors.append(f"Last committed checkpoint parse failed: {type(exc).__name__}: {exc}")

    result = {
        "audited_at": utc_now(), "status": state.get("status"), "ticks_expected": 240,
        "ticks_committed": completed_ticks, "decision_slots": total_decision_slots,
        "successful_decisions": successes, "persistent_failed_decisions": failures,
        "responses_revalidated_from_reflection": response_revalidated,
        "successes_missing_matching_reflection": reflection_pair_missing,
        "llm_calls": len(classified), "first_stage_calls": stage_counts.get("first_stage", 0),
        "reflection_calls": stage_counts.get("reflection", 0),
        "unclassified_calls": stage_counts.get("unresolved", 0), "missing_raw_prompt_files": missing_prompts,
        "invalid_generation_calls": len(invalid_calls), "context_window_exceptions": len(context_errors),
        "response_metadata_truncation_flags": len(context_truncations),
        "server_log_context_audit": server_log_audit,
        "context_measurement_calls": len(context_rows),
        "context_input_prompt_tokens_max": max((prompt for prompt, _, _, _ in context_rows), default=None),
        "context_input_prompt_tokens_mean": round(sum(prompt for prompt, _, _, _ in context_rows) / len(context_rows), 1) if context_rows else None,
        "context_sequence_tokens_max": max((slot for _, _, slot, _ in context_rows), default=None),
        "context_window_tokens": sorted({capacity for _, _, _, capacity in context_rows}),
        "context_sequence_occupancy_max_percent": round(100 * max(sequence_utilization), 2) if sequence_utilization else None,
        "context_calls_at_or_above_90_percent": sum(value >= 0.9 for value in sequence_utilization),
        "context_server_count_discrepancies": context_server_count_discrepancies,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "generation_seconds": round(gen_seconds, 2),
        "output_tokens_per_second": round(output_tps, 2) if output_tps else None,
        "wall_runtime_seconds": round(wall_runtime_seconds, 2) if wall_runtime_seconds is not None else None,
        "gpu_samples": len(gpu), "gpu_used_min_mib": min((int(x["used_mib"]) for x in gpu), default=None),
        "gpu_used_max_mib": max((int(x["used_mib"]) for x in gpu), default=None),
        "gpu_used_mean_mib": round(sum(int(x["used_mib"]) for x in gpu) / len(gpu), 1) if gpu else None,
        "actions": dict(action_counts), "final_agent_count": final_agent_count,
        "final_population_by_type": final_population, "final_checkpoint_step": final_step,
        "outer_tick_recoveries": state.get("outer_recoveries", 0), "server_restarts": state.get("server_restarts", 0),
        "persistent_failed_units": persistent_failures, "integrity_errors": errors,
    }
    write_json(OUT / "audit.json", result)
    (OUT / "telemetry" / "classified_completions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in classified), encoding="utf-8",
    )
    update_report(result)
    return result


def update_report(audit: dict[str, Any]) -> None:
    text = REPORT.read_text(encoding="utf-8")
    left = text.find(START)
    right = text.find(END, left + len(START)) if left >= 0 else -1
    if left < 0 or right < 0:
        raise RuntimeError("Full-baseline report section markers are missing.")
    block = text[left:right + len(END)]
    values = [
        f"- Offline post-run audit: **{audit['ticks_committed']}/{audit['ticks_expected']} ticks**, {audit['successful_decisions']}/{audit['decision_slots']} successful decisions, {audit['persistent_failed_decisions']} persistent failures; {audit['responses_revalidated_from_reflection']} final responses revalidated against the original validator; {len(audit['integrity_errors'])} integrity errors.",
        f"- Two-stage trace audit from saved prompts: {audit['first_stage_calls']} first-stage calls and {audit['reflection_calls']} reflection calls; {audit['unclassified_calls']} unresolved stage labels, {len(audit['missing_raw_prompt_files'])} missing raw request files, {audit['successes_missing_matching_reflection']} successful decisions without a matching valid reflection output.",
        f"- Retry/context audit: {audit['outer_tick_recoveries']} outer tick recoveries; {audit['server_restarts']} server restarts; {audit['invalid_generation_calls']} invalid raw generations; {audit['context_window_exceptions']} context-window request exceptions; {audit['response_metadata_truncation_flags']} response-metadata truncation flags. llama-server log audit: {audit['server_log_context_audit']['truncated_release_count']} truncated releases among {audit['server_log_context_audit']['release_count']} run-window releases; coverage complete: {audit['server_log_context_audit']['complete']}.",
        f"- Context/runtime audit: context measurements for {audit['context_measurement_calls']} calls; input-prompt tokens max/mean {audit['context_input_prompt_tokens_max'] or 'unavailable'}/{audit['context_input_prompt_tokens_mean'] or 'unavailable'}; max finished sequence occupancy {audit['context_sequence_tokens_max'] or 'unavailable'}/{audit['context_window_tokens'] or 'unavailable'} tokens ({audit['context_sequence_occupancy_max_percent'] or 'unavailable'}%); calls at or above 90%: {audit['context_calls_at_or_above_90_percent']}; server-count mismatches above 2 tokens: {audit['context_server_count_discrepancies']}; end-to-end runtime {round(audit['wall_runtime_seconds'] / 3600, 2) if audit['wall_runtime_seconds'] is not None else 'unavailable'} hours.",
        f"- Final population: {audit['final_agent_count']} agents at checkpoint step {audit['final_checkpoint_step']}, by type `{audit['final_population_by_type']}`. Valid action totals: `{audit['actions']}`.",
        f"- Offline token/throughput totals: {audit['input_tokens']} prompt tokens, {audit['output_tokens']} generated tokens, {audit['output_tokens_per_second'] or 'unavailable'} output tokens/s across {audit['generation_seconds']} generation seconds.",
        f"- GPU sample range: {audit['gpu_used_min_mib'] or 'unavailable'}–{audit['gpu_used_max_mib'] or 'unavailable'} MiB used (mean {audit['gpu_used_mean_mib'] or 'unavailable'} MiB, {audit['gpu_samples']} samples). Full audit: `data/qwen38_full_baseline_20261016/audit.json`; reclassified call evidence: `data/qwen38_full_baseline_20261016/telemetry/classified_completions.jsonl`.",
    ]
    insertion = "\n\n".join(values)
    marker = "The runner calls the repository's original `simulation_step.step()`"
    if "- Offline post-run audit:" not in block:
        position = block.find(marker)
        if position < 0:
            raise RuntimeError("Could not locate the full-baseline methodology text in the report block.")
        block = block[:position] + insertion + "\n\n" + block[position:]
    else:
        start = block.index("- Offline post-run audit:")
        end = block.find("\n\nThe runner calls", start)
        if end < 0:
            raise RuntimeError("Could not locate the previous offline audit block.")
        block = block[:start] + insertion + block[end:]
    updated = text[:left] + block + text[right + len(END):]
    temp = REPORT.with_name(REPORT.name + ".tmp")
    temp.write_text(updated, encoding="utf-8")
    os.replace(temp, REPORT)


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not report["integrity_errors"] else 2)
