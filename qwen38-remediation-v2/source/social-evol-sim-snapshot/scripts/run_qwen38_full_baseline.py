"""Run one resumable 80-life-step Qwen baseline through the original step()."""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update({
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONUTF8": "1",
})

import qwen38_capability_evaluation as cap

OUT = ROOT / "data" / "qwen38_full_baseline_20261016"
TICKS = OUT / "ticks"
REPORT = ROOT / "reports" / "QWEN38_CAPABILITY_EVALUATION.md"
SEED = 20261016
CONFIG_DIR = "configZ_major_v3"
RUN_ID = "qwen38-full-baseline-20261016"
MAX_LIFE_STEPS = 80
COMMUNICATION_STEPS = 2
MAX_TICKS = MAX_LIFE_STEPS * (COMMUNICATION_STEPS + 1)
MAX_ATTEMPTS = 6  # one original attempt plus five outer recoveries
BACKOFF = (5, 10, 20, 40, 60)
START = "<!-- QWEN38_FULL_BASELINE_START -->"
END = "<!-- QWEN38_FULL_BASELINE_END -->"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False) + "\n")
        handle.flush()


def configure_capture() -> None:
    """Reuse the already-tested local server, GPU, token, and prompt capture hooks."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "telemetry").mkdir(parents=True, exist_ok=True)
    TICKS.mkdir(parents=True, exist_ok=True)
    cap.OUT = OUT
    cap.STATE_PATH = OUT / "progress.json"
    cap.CALLS_JSONL = OUT / "telemetry" / "completions.jsonl"
    cap.HEARTBEAT_JSONL = OUT / "telemetry" / "heartbeat.jsonl"
    cap.GPU_JSONL = OUT / "telemetry" / "gpu_samples.jsonl"
    cap.SERVER_JSONL = OUT / "telemetry" / "server_samples.jsonl"
    cap.REPORT_PATH = REPORT
    cap.progress_counts = lambda state=None: {
        "ticks_completed": int((state or cap.STATE).get("completed_ticks", 0)),
        "ticks_total": MAX_TICKS,
        "successful_decisions": int((state or cap.STATE).get("successful_decisions", 0)),
        "failed_decisions": int((state or cap.STATE).get("failed_decisions", 0)),
    }
    cap.refresh_report = update_report
    cap.note_progress = note_progress
    cap.STATE = read_json(cap.STATE_PATH, {}) or {}
    cap.LOCK = __import__("threading").RLock()
    cap.MONITOR_STOP = __import__("threading").Event()
    cap.install_instrumentation()


def note_progress(message: str) -> None:
    cap.STATE["last_event"] = {"at": now(), "message": message}
    cap.STATE["last_activity_at"] = now()
    cap.STATE["updated_at"] = now()
    write_json(cap.STATE_PATH, cap.STATE)


def display_actions(actions: dict[str, int]) -> str:
    return ", ".join(f"{name}:{count}" for name, count in sorted(actions.items())) or "none"


def tick_results() -> list[dict[str, Any]]:
    rows = []
    if not TICKS.exists():
        return rows
    for path in sorted(TICKS.glob("step_*/result.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
        result = read_json(path)
        if result:
            rows.append(result)
    return rows


def telemetry_summary() -> dict[str, Any]:
    calls = []
    try:
        for line in cap.CALLS_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    output_tokens = sum(int(x.get("output_tokens") or 0) for x in calls)
    input_tokens = sum(int(x.get("input_tokens") or 0) for x in calls)
    generation_seconds = sum(float(x.get("reported_duration_seconds") or x.get("observed_duration_seconds") or 0) for x in calls)
    output_tps = output_tokens / generation_seconds if generation_seconds else None
    stage_counts = Counter(x.get("stage", "unknown") for x in calls)
    gpus = []
    try:
        for line in cap.GPU_JSONL.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("used_mib") is not None:
                gpus.append(int(item["used_mib"]))
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "calls": len(calls), "first_stage_calls": stage_counts.get("first_stage", 0),
        "reflection_calls": stage_counts.get("reflection", 0),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "generation_seconds": round(generation_seconds, 2),
        "output_tokens_per_second": round(output_tps, 2) if output_tps else None,
        "gpu_samples": len(gpus), "gpu_used_min_mib": min(gpus) if gpus else None,
        "gpu_used_max_mib": max(gpus) if gpus else None,
        "gpu_used_mean_mib": round(sum(gpus) / len(gpus), 1) if gpus else None,
    }


def update_report() -> None:
    state = cap.STATE or {}
    rows = tick_results()
    summary = telemetry_summary()
    elapsed = max(0.0, time.time() - float(state.get("started_epoch", time.time())))
    status = state.get("status", "running")
    completed = int(state.get("completed_ticks", len(rows)))
    lines = [
        START,
        "## Experiment D — Full 80-life-step baseline",
        "",
        f"**Status: {status}.** This follow-on run uses one visible-type baseline setting and one fixed environment seed; it is not the paper's full 20-run, four-setting study. The paper reports results across 20 runs/four settings ([ACL paper](https://aclanthology.org/2026.acl-long.1339/)).",
        "",
        f"- Started: {state.get('started_at', 'pending')}; last update: {now()}; run ID: `{state.get('run_id', RUN_ID)}`.",
        f"- Progress: **{completed}/{MAX_TICKS} simulation ticks** ({completed / (COMMUNICATION_STEPS + 1):.2f}/{MAX_LIFE_STEPS} life-step cycles); initial agents: 8, two per active moral type; current tick: {state.get('current_tick', 'pending')}.",
        f"- Config: `{CONFIG_DIR}` prompts/rules, `max_life_steps={MAX_LIFE_STEPS}`, `communication_and_sharing_steps={COMMUNICATION_STEPS}`, `two_stage_model=true`, local `Qwen3.8-27B-Q4_K_M`, `127.0.0.1:8080/v1`, single inference slot, seed `{SEED}`.",
        f"- Decision totals so far: {state.get('successful_decisions', 0)} successful, {state.get('failed_decisions', 0)} persistent failed decisions; outer tick recoveries: {state.get('outer_recoveries', 0)}; server restarts: {state.get('server_restarts', 0)}.",
        f"- Elapsed wall time: {elapsed / 3600:.2f} h. Raw prompts/responses, checkpoints, tick results, and telemetry: `data/qwen38_full_baseline_20261016/`; original file checkpoints/logs: `data/{state.get('run_id', RUN_ID)}/`.",
        f"- Captured calls: {summary['calls']} ({summary['first_stage_calls']} first-stage, {summary['reflection_calls']} reflection); prompt tokens {summary['input_tokens']}; generated tokens {summary['output_tokens']}; measured generation throughput {summary['output_tokens_per_second'] or 'pending'} tokens/s.",
        f"- RTX 3090 board memory samples: {summary['gpu_samples']}; used VRAM {summary['gpu_used_min_mib'] or 'pending'}–{summary['gpu_used_max_mib'] or 'pending'} MiB (mean {summary['gpu_used_mean_mib'] or 'pending'} MiB).",
        "",
        "The runner calls the repository's original `simulation_step.step()` and action/environment code. It keeps successful per-agent decisions cached across failed tick attempts, commits a checkpoint and random-generator state after every completed tick, records unresolved decision failures, and does not alter prompts or reduce agents. One environment seed is fixed; llama-server's default random generation seed remains nondeterministic.",
        "",
        "| Tick | Cycle / phase | Decisions | Alive after | Population by type | Actions proposed | Persistent failures | Tick seconds |",
        "|---:|---|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('tick')} | {row.get('cycle')} / {row.get('phase')} | "
            f"{row.get('successful_decisions', 0)}/{row.get('decision_slots', 0)} | {row.get('alive_after')} | "
            f"{row.get('population_by_type', {})} | {display_actions(row.get('actions', {}))} | "
            f"{len(row.get('failed_agent_ids', []))} | {row.get('elapsed_seconds', '—')} |"
        )
    if state.get("blocker"):
        lines += ["", f"**Blocker:** {state['blocker']}"]
    if status == "complete":
        lines += ["", f"**Completion:** {state.get('finished_at')}; stopping reason: {state.get('stop_reason', 'maximum tick count reached')}."]
    lines.append(END)
    block = "\n".join(lines)
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Qwen3.8 Capability Evaluation\n"
    if START in existing and END in existing:
        left = existing.index(START)
        right = existing.index(END, left) + len(END)
        updated = existing[:left] + block + existing[right:]
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    historical = "No part of the paper's 80-step evolutionary experiment is included."
    if historical in updated:
        updated = updated.replace(
            historical,
            "At pilot finalization, no part of the paper's 80-step evolutionary experiment was included; the follow-on single-baseline run is documented below.",
            1,
        )
    temp = REPORT.with_name(REPORT.name + ".tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(updated, encoding="utf-8")
    os.replace(temp, REPORT)


def tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tuple_tree(x) for x in value)
    return value


def rng_state() -> dict[str, Any]:
    import numpy as np
    from scr.utils.random_utils import shared_random
    nstate = np.random.get_state()
    return {
        "python": jsonable(random.getstate()),
        "shared": jsonable(shared_random.getstate()),
        "numpy": [nstate[0], nstate[1].tolist(), int(nstate[2]), int(nstate[3]), float(nstate[4])],
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    from scr.utils.random_utils import shared_random
    random.setstate(tuple_tree(state["python"]))
    shared_random.setstate(tuple_tree(state["shared"]))
    nstate = state["numpy"]
    np.random.set_state((nstate[0], np.asarray(nstate[1], dtype="uint32"), int(nstate[2]), int(nstate[3]), float(nstate[4])))


def init_state() -> dict[str, Any]:
    existing = read_json(OUT / "progress.json", {}) or {}
    if existing:
        return existing
    return {
        "status": "starting", "run_id": RUN_ID, "config_dir": CONFIG_DIR,
        "seed": SEED, "max_life_steps": MAX_LIFE_STEPS,
        "communication_and_sharing_steps": COMMUNICATION_STEPS,
        "ticks_total": MAX_TICKS, "completed_ticks": 0,
        "successful_decisions": 0, "failed_decisions": 0,
        "outer_recoveries": 0, "server_restarts": 0,
        "started_at": now(), "started_epoch": time.time(),
        "updated_at": now(), "current_tick": None,
    }


def validate_config(checkpoint) -> None:
    from collections import Counter
    config = checkpoint.configuration
    assert config.llm.provider == "llama_cpp"
    assert config.llm.chat_model == "Qwen3.8-27B-Q4_K_M"
    assert config.llm.two_stage_model is True
    assert config.world.max_life_steps == MAX_LIFE_STEPS
    assert config.world.communication_and_sharing_steps == COMMUNICATION_STEPS
    assert config.llm.async_config.max_concurrent_calls == 1
    assert len(checkpoint.social_environment.agents) == 8
    assert Counter(a.type for a in checkpoint.social_environment.agents) == Counter({name: 2 for name in cap.TYPE_IDS})


def initial_checkpoint():
    import numpy as np
    from scr.models.simulation.checkpoint import Checkpoint
    from scr.utils.random_utils import shared_random
    random.seed(SEED)
    np.random.seed(SEED)
    # Seed the shared Random instance in place because simulation modules import it directly.
    shared_random.seed(SEED)
    overrides = {
        "world.max_life_steps": MAX_LIFE_STEPS,
        "world.communication_and_sharing_steps": COMMUNICATION_STEPS,
        "llm.provider": "llama_cpp",
        "llm.chat_model": "Qwen3.8-27B-Q4_K_M",
        "llm.two_stage_model": True,
        "llm.async_config.max_concurrent_calls": 1,
        "llm.async_config.call_timeout_seconds": 3600.0,
        "random_seed": SEED,
    }
    checkpoint = Checkpoint.initialize_from_config(CONFIG_DIR, overrides)
    checkpoint.metadata.run_id = RUN_ID
    validate_config(checkpoint)
    return checkpoint


def initial_manifest(checkpoint) -> dict[str, Any]:
    import subprocess
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_commit = "unavailable"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits", "--id=0"],
                            capture_output=True, text=True, check=True, timeout=8).stdout.strip()
    except Exception as exc:
        gpu = f"unavailable: {exc}"
    try:
        props = cap.api_json("/props", timeout=10)
    except Exception as exc:
        props = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "started_at": now(), "git_commit": git_commit, "run_id": RUN_ID,
        "config_dir": CONFIG_DIR, "config": checkpoint.configuration.model_dump(mode="json"),
        "seed": SEED, "actual_runner_ticks": MAX_TICKS,
        "model": "Qwen3.8-27B-Q4_K_M", "endpoint": cap.API + "/v1",
        "llama_server_props": props, "gpu": gpu,
        "protocol_note": "One baseline setting and one seed, not the paper's 20-run/four-setting study.",
    }


def memory_integrity(before: dict[str, Any], decisions: dict[str, dict], checkpoint) -> dict[str, Any]:
    agents_before = {x["id"]: x for x in before["social_environment"]["agents"]}
    after = {a.id: a for a in checkpoint.social_environment.agents}
    updates = 0
    checked = 0
    for agent_id, decision in decisions.items():
        if not decision.get("success") or not decision.get("response") or agent_id not in after:
            continue
        prior = agents_before.get(agent_id, {}).get("memory", {})
        response = decision["response"]
        expected_ltm = dict(prior.get("long_term_memory", {}))
        expected_ltm.update(response.get("long_term_memory") or {})
        expected_plan = response.get("short_term_plan") or prior.get("short_term_plan", "")
        agent = after[agent_id]
        checked += 1
        response_saved = bool(agent.response_history) and agent.response_history[-1].model_dump(mode="json") == response
        memory_saved = agent.memory.long_term_memory == expected_ltm and agent.memory.short_term_plan == expected_plan
        if response_saved and memory_saved:
            updates += 1
    return {
        "checked_successful_decisions": checked, "memory_updates_verified": updates,
        "memory_objects_independent": cap.check_memory_independence(checkpoint),
        "relationships_consistent": cap.relationship_consistency(checkpoint),
    }


def population_counts(checkpoint) -> dict[str, int]:
    counts = Counter(a.type for a in checkpoint.social_environment.agents)
    return dict(sorted(counts.items()))


async def run() -> int:
    configure_capture()
    if os.environ.get("QWEN_FULL_VALIDATE_ONLY") == "1":
        cp = initial_checkpoint()
        validate_config(cp)
        print(json.dumps({"config": CONFIG_DIR, "ticks": MAX_TICKS, "agents": len(cp.social_environment.agents), "two_stage": cp.configuration.llm.two_stage_model}))
        return 0

    lock_path = OUT / "runner.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
    except FileExistsError:
        try:
            pid = int(lock_path.read_text(encoding="ascii").strip())
            os.kill(pid, 0)
            raise RuntimeError(f"Another full baseline runner appears active (PID {pid}).")
        except (OSError, ValueError):
            lock_path.unlink(missing_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
    atexit.register(lambda: lock_path.unlink(missing_ok=True))

    monitor = None
    try:
        from scr.models.simulation.checkpoint import Checkpoint
        from scr.simulation.event_bus import SimulationEventBus
        from scr.simulation.runner import simulation_step
        from scr.simulation.runner.stopping_criteria import check_stopping_criteria
        from scr.utils import sim_logger
        from scr.utils.checkpoint.save_checkpoint import save_checkpoint
        from scr.utils.logger import init_run_logger
        from scr.api.db_api.connection import disable_db
    except ModuleNotFoundError:
        raise

    try:
        disable_db()
    except Exception:
        pass

    cap.STATE = init_state()
    if cap.STATE.get("status") == "complete":
        update_report()
        return 0
    cap.STATE["status"] = "running"
    cap.STATE["runner_pid"] = os.getpid()
    cap.STATE["updated_at"] = now()

    committed_path = OUT / "last_committed.json"
    committed = read_json(committed_path)
    if committed:
        checkpoint = Checkpoint.model_validate(committed["checkpoint"])
        restore_rng(committed["rng_state"])
        validate_config(checkpoint)
    else:
        checkpoint = initial_checkpoint()
        validate_config(checkpoint)
        rng0 = rng_state()
        start_json = json.loads(checkpoint.model_dump_json())
        write_json(OUT / "initial.json", start_json)
        write_json(committed_path, {"step": 0, "checkpoint": start_json, "rng_state": rng0})
        write_json(OUT / "manifest.json", initial_manifest(checkpoint))
        save_checkpoint(checkpoint, str(ROOT / "data"))

    committed_ticks = max(0, checkpoint.metadata.current_time_step - 1)
    if int(cap.STATE.get("completed_ticks", 0)) != committed_ticks:
        append_jsonl(OUT / "recovery.jsonl", {
            "at": now(), "action": "reconciled_progress_with_committed_checkpoint",
            "previous_progress_ticks": cap.STATE.get("completed_ticks", 0),
            "committed_checkpoint_ticks": committed_ticks,
        })
        cap.STATE["completed_ticks"] = committed_ticks

    init_run_logger(ROOT / "data" / RUN_ID)
    sim_logger.bind(run_id=RUN_ID)
    cap.ensure_server()
    cap.STATE["server_health"] = cap.health()
    cap.STATE["started_at"] = cap.STATE.get("started_at") or now()
    cap.STATE["started_epoch"] = float(cap.STATE.get("started_epoch") or time.time())
    cap.STATE["last_activity_at"] = now()
    write_json(cap.STATE_PATH, cap.STATE)
    update_report()
    monitor = cap.start_monitor()

    event_bus = SimulationEventBus(enabled=True)
    event_bus.subscribe(sim_logger.jsonl_event_bus_subscriber)
    await event_bus.start()
    try:
        while int(cap.STATE.get("completed_ticks", 0)) < MAX_TICKS:
            if check_stopping_criteria(checkpoint):
                cap.STATE["stop_reason"] = "all_agents_dead"
                break
            expected_tick = checkpoint.metadata.current_time_step
            if expected_tick > MAX_TICKS:
                break
            round_dir = TICKS / f"step_{expected_tick:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            before_path = round_dir / "before.json"
            if not before_path.exists():
                write_json(before_path, json.loads(checkpoint.model_dump_json()))
            before = read_json(before_path)
            if before is None:
                raise RuntimeError(f"Missing valid before-checkpoint for tick {expected_tick}.")
            expected_agents = [a.id for a in Checkpoint.model_validate(before).social_environment.agents if a.is_alive()]
            cache_dir = OUT / "decision_cache" / f"step_{expected_tick:03d}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            start_rng = rng_state()
            tick_started = time.time()
            cap.STATE.update({"phase": "simulation_tick", "current_tick": expected_tick,
                              "current_unit": f"tick_{expected_tick}", "last_activity_at": now()})
            write_json(cap.STATE_PATH, cap.STATE)
            outer_errors: list[dict[str, Any]] = []
            final_working = None
            latest_runtime: dict[str, Any] | None = None
            messages = {}
            failed_ids: list[str] = []
            attempts_used = 0
            for attempt in range(1, MAX_ATTEMPTS + 1):
                attempts_used = attempt
                working = Checkpoint.model_validate(before)
                restore_rng(start_rng)
                runtime = {
                    "step": expected_tick, "round_attempt": attempt, "run_id": RUN_ID,
                    "round_dir": round_dir, "cache_dir": cache_dir, "decisions": {},
                }
                token = cap.B_RUNTIME.set(runtime)
                try:
                    messages = await simulation_step.step(
                        working, str(ROOT / "data"), max_concurrent=1, event_bus=event_bus,
                    )
                    latest_runtime, final_working = runtime, working
                    failed_ids = [agent_id for agent_id in expected_agents
                                  if not runtime["decisions"].get(agent_id, {}).get("success")]
                except Exception as exc:
                    failure = {"attempt": attempt, "at": now(), "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()}
                    outer_errors.append(failure)
                    append_jsonl(OUT / "failures.jsonl", {"tick": expected_tick, **failure})
                    if not cap.health().get("ok"):
                        try:
                            cap.restart_server(f"full baseline tick {expected_tick} raised {type(exc).__name__}")
                            cap.STATE["server_restarts"] = int(cap.STATE.get("server_restarts", 0)) + 1
                        except Exception as restart_exc:
                            append_jsonl(OUT / "recovery.jsonl", {"at": now(), "tick": expected_tick,
                                                                  "action": "server_restart_failed", "error": str(restart_exc)})
                finally:
                    cap.B_RUNTIME.reset(token)
                if final_working is not None and not failed_ids:
                    break
                if attempt < MAX_ATTEMPTS:
                    cap.STATE["outer_recoveries"] = int(cap.STATE.get("outer_recoveries", 0)) + 1
                    note_progress(f"Tick {expected_tick}: retrying failed agents {failed_ids or 'step exception'} ({attempt}/{MAX_ATTEMPTS - 1} recoveries used).")
                    if not cap.health().get("ok"):
                        try:
                            cap.restart_server(f"tick {expected_tick} recovery attempt {attempt}")
                            cap.STATE["server_restarts"] = int(cap.STATE.get("server_restarts", 0)) + 1
                        except Exception as restart_exc:
                            append_jsonl(OUT / "recovery.jsonl", {"at": now(), "tick": expected_tick,
                                                                  "action": "server_restart_failed", "error": str(restart_exc)})
                    await asyncio.sleep(BACKOFF[attempt - 1])

            if final_working is None or latest_runtime is None:
                reason = f"Original step failed for tick {expected_tick} after {MAX_ATTEMPTS} attempts; later ticks depend on the unresolved state."
                cap.STATE.update({"status": "blocked", "blocker": reason, "current_tick": expected_tick,
                                  "finished_at": now(), "outer_errors": outer_errors})
                write_json(OUT / "blocker.json", {"at": now(), "tick": expected_tick,
                                                   "reason": reason, "errors": outer_errors,
                                                   "checkpoint_before": before})
                write_json(cap.STATE_PATH, cap.STATE)
                update_report()
                return 2

            decision_rows = []
            action_counts = Counter()
            for agent_id in expected_agents:
                data = latest_runtime["decisions"].get(agent_id) or read_json(cache_dir / f"{agent_id}.json") or {}
                decision_rows.append({"agent_id": agent_id, "success": bool(data.get("success")),
                                      "error": data.get("error"), "response": data.get("response"),
                                      "calls": data.get("calls", []), "outer_attempt": data.get("outer_attempt", attempts_used)})
                if data.get("success") and data.get("response"):
                    response = cap.deserialize_decision(data).response
                    action = response.action.root.model_dump(mode="json")
                    action_counts[action.get("action_type", "unknown")] += 1

            working_after = json.loads(final_working.model_dump_json())
            integrity = memory_integrity(before, {r["agent_id"]: r for r in decision_rows}, final_working)
            alive_after = sum(1 for a in final_working.social_environment.agents if a.is_alive())
            all_ids = [a.id for a in final_working.social_environment.agents]
            state_consistent = (
                final_working.metadata.current_time_step == expected_tick + 1
                and len(all_ids) == len(set(all_ids))
                and len(final_working.metadata.execution_queue) == len(set(final_working.metadata.execution_queue))
                and all(0 <= a.state.hp <= a.state.max_hp for a in final_working.social_environment.agents)
            )
            (round_dir / "messages.json").write_text(
                json.dumps(jsonable({k: v.messages for k, v in messages.items()}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reloaded = Checkpoint.model_validate_json(json.dumps(working_after))
            reload_ok = reloaded.metadata.current_time_step == expected_tick + 1
            persistent = bool(failed_ids)
            result = {
                "status": "complete_with_persistent_failures" if persistent else "complete",
                "tick": expected_tick, "cycle": (expected_tick - 1) // 3 + 1,
                "phase": "production/action" if expected_tick % 3 == 0 else "social/coordination",
                "attempts": attempts_used, "outer_errors": outer_errors,
                "decision_slots": len(expected_agents),
                "successful_decisions": sum(1 for x in decision_rows if x["success"]),
                "failed_agent_ids": failed_ids, "decisions": decision_rows,
                "actions": dict(action_counts), "alive_after": alive_after,
                "population_by_type": population_counts(final_working),
                "state_after": {a.id: {"hp": a.state.hp, "age": a.state.age, "alive": a.is_alive(), "type": a.type}
                                for a in final_working.social_environment.agents},
                "new_observations": [x for x in final_working.observations if x not in before.get("observations", [])],
                "resources_after": final_working.physical_environment.model_dump(mode="json"),
                "integrity": integrity, "state_consistent": state_consistent,
                "checkpoint_reload_verified": reload_ok,
                "elapsed_seconds": round(time.time() - tick_started, 2), "completed_at": now(),
            }
            write_json(round_dir / "after.json", working_after)
            write_json(round_dir / "result.json", result)
            rng_after = rng_state()
            # One atomic record keeps the checkpoint and its random state in sync across recovery.
            write_json(committed_path, {"step": expected_tick, "checkpoint": working_after, "rng_state": rng_after})
            save_checkpoint(final_working, str(ROOT / "data"))
            checkpoint = final_working
            cap.STATE["completed_ticks"] = int(cap.STATE.get("completed_ticks", 0)) + 1
            cap.STATE["successful_decisions"] = int(cap.STATE.get("successful_decisions", 0)) + result["successful_decisions"]
            cap.STATE["failed_decisions"] = int(cap.STATE.get("failed_decisions", 0)) + len(failed_ids)
            cap.STATE.update({"current_tick": expected_tick, "last_completed_tick": expected_tick,
                              "last_tick_status": result["status"], "last_tick_seconds": result["elapsed_seconds"],
                              "last_tick_actions": dict(action_counts), "updated_at": now()})
            write_json(cap.STATE_PATH, cap.STATE)
            append_jsonl(OUT / "telemetry" / "tick_commits.jsonl", {
                "at": now(), "tick": expected_tick, "completed_ticks": cap.STATE["completed_ticks"],
                "status": result["status"], "decisions": result["successful_decisions"],
                "decision_slots": result["decision_slots"], "elapsed_seconds": result["elapsed_seconds"],
            })
            update_report()
            note_progress(f"Tick {expected_tick}/{MAX_TICKS} committed; {result['successful_decisions']}/{result['decision_slots']} decisions valid.")

        if cap.STATE.get("status") != "blocked":
            cap.STATE["status"] = "complete"
            cap.STATE["finished_at"] = now()
            cap.STATE["stop_reason"] = cap.STATE.get("stop_reason") or "maximum tick count reached"
            cap.STATE["final_checkpoint"] = str(OUT / "final.json")
            write_json(OUT / "final.json", json.loads(checkpoint.model_dump_json()))
            write_json(cap.STATE_PATH, cap.STATE)
            update_report()
        return 0
    except Exception as exc:
        cap.STATE.update({"status": "process_error", "last_error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(), "updated_at": now()})
        write_json(OUT / "process_error.json", {"at": now(), "error": cap.STATE["last_error"],
                                                 "traceback": cap.STATE["traceback"]})
        write_json(cap.STATE_PATH, cap.STATE)
        update_report()
        raise
    finally:
        await event_bus.stop()
        cap.MONITOR_STOP.set()
        if monitor is not None:
            monitor.join(timeout=10)
        sim_logger.close()
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run())
    except Exception:
        traceback.print_exc()
        exit_code = 1
    raise SystemExit(exit_code)
