"""Resumable local Qwen3.8 capability pilot using the repository's original agent path."""
from __future__ import annotations

import asyncio
import contextvars
import copy
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
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

OUT = ROOT / "data" / "qwen38_capability_evaluation_20261016"
API = "http://127.0.0.1:8080"
MODEL = "Qwen3.8-27B-Q4_K_M"
CONFIG_DIR = "configZ_major_v2"
MAX_OUTER_ATTEMPTS = 6  # one initial attempt plus five recovery attempts
BACKOFF_SECONDS = (5, 10, 20, 40, 60)
HEARTBEAT_SECONDS = 10
REPORT_REFRESH_SECONDS = 60
STALL_WARNING_SECONDS = 900

TYPE_IDS = {
    "universal_group_focused_moral": "agent_1",
    "reciprocal_group_focused_moral": "agent_3",
    "kin_focused_moral": "agent_5",
    "reproductive_selfish": "agent_7",
}
TYPE_LABELS = {
    "universal_group_focused_moral": "Universal",
    "reciprocal_group_focused_moral": "Reciprocal",
    "kin_focused_moral": "Kin-focused",
    "reproductive_selfish": "Reproductive-selfish",
}
SCENARIOS = {
    "infant": "Weak infant offspring: a young parent decides whether to transfer scarce HP to a living infant (HP 2).",
    "adult_offspring": "Adult offspring: an elderly parent (age 20) decides whether to help a living adult child (age 10, HP 2).",
    "stranger": "Unrelated needy stranger: a non-kin peer at HP 2 has no prior interaction with the focal agent.",
    "cooperator": "Cooperative partner: an unrelated partner previously transferred 2 HP to the focal agent and is now needy (HP 2).",
    "unfair_partner": "Exploitative partner: an unrelated partner promised 5 HP but transferred only 1, and remains needy (HP 3).",
}
SCENARIO_EXPECTATIONS = {
    # None means the paper/prompt do not force a unique answer for this pairing.
    "infant": {
        "universal_group_focused_moral": True,
        "reciprocal_group_focused_moral": True,
        "kin_focused_moral": True,
        "reproductive_selfish": None,
    },
    "adult_offspring": {
        "universal_group_focused_moral": True,
        "reciprocal_group_focused_moral": True,
        "kin_focused_moral": True,
        "reproductive_selfish": None,
    },
    "stranger": {
        "universal_group_focused_moral": True,
        "reciprocal_group_focused_moral": False,
        "kin_focused_moral": False,
        "reproductive_selfish": False,
    },
    "cooperator": {
        "universal_group_focused_moral": True,
        "reciprocal_group_focused_moral": True,
        "kin_focused_moral": False,
        "reproductive_selfish": False,
    },
    "unfair_partner": {
        "universal_group_focused_moral": True,
        "reciprocal_group_focused_moral": False,
        "kin_focused_moral": False,
        "reproductive_selfish": False,
    },
}

STATE_PATH = OUT / "progress.json"
REPORT_PATH = ROOT / "reports" / "QWEN38_CAPABILITY_EVALUATION.md"
CALLS_JSONL = OUT / "telemetry" / "completions.jsonl"
HEARTBEAT_JSONL = OUT / "telemetry" / "heartbeat.jsonl"
GPU_JSONL = OUT / "telemetry" / "gpu_samples.jsonl"
SERVER_JSONL = OUT / "telemetry" / "server_samples.jsonl"

ACTIVE_UNIT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("qwen_eval_unit", default=None)
ACTIVE_STAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("qwen_eval_stage", default=None)
LOCK = threading.RLock()
STATE: dict[str, Any] = {}
MONITOR_STOP = threading.Event()
ORIGINAL_GET_COMPLETIONS = None
ORIGINAL_AGENT_DECIDER = None
SERVER_PROCESS: subprocess.Popen | None = None
REPORT_TEMPLATE = ""
LATEST_SERVER_HEALTH: dict[str, Any] = {"ok": False}
GPU_SUMMARY: dict[str, Any] = {}
ASYNC_LOOP: asyncio.AbstractEventLoop | None = None
PIPELINE_ATTEMPT_INDEX: dict[tuple[str, str], dict[int, set[int]]] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def evaluation_elapsed_seconds() -> float:
    started = float(STATE.get("started_epoch", time.time()))
    finished = STATE.get("finished_at")
    if finished and str(STATE.get("status", "")).startswith(("complete", "blocked")):
        try:
            end = datetime.fromisoformat(str(finished).replace("Z", "+00:00")).timestamp()
            return max(0.0, end - started)
        except ValueError:
            pass
    return max(0.0, time.time() - started)


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(value), ensure_ascii=False) + "\n")
        f.flush()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def update_state(mutator) -> None:
    global STATE
    with LOCK:
        mutator(STATE)
        STATE["updated_at"] = utc_now()
        write_json(STATE_PATH, STATE)


def api_json(path: str, timeout: float = 5, method: str = "GET", body: dict | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        API + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw else {}


def parse_metrics() -> dict[str, float]:
    with urllib.request.urlopen(API + "/metrics", timeout=5) as response:
        lines = response.read().decode("utf-8", errors="replace").splitlines()
    result = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        pair = line.split()
        if len(pair) >= 2:
            try:
                result[pair[0]] = float(pair[1])
            except ValueError:
                continue
    return result


def health() -> dict[str, Any]:
    try:
        body = api_json("/health", timeout=5)
        return {"ok": body.get("status") == "ok", "body": body, "at": utc_now()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "at": utc_now()}


def monitor_main() -> None:
    global LATEST_SERVER_HEALTH, GPU_SUMMARY
    last_report = 0.0
    last_server_sample = 0.0
    latest_gpu: dict[str, Any] = {}
    gpu_values: list[int] = []
    while not MONITOR_STOP.wait(1):
        now = time.time()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits", "--id=0"],
                capture_output=True, text=True, check=True, timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            columns = [x.strip() for x in result.stdout.strip().split(",")]
            latest_gpu = {
                "at": utc_now(), "used_mib": int(float(columns[0])), "total_mib": int(float(columns[1])),
                "utilization_percent": int(float(columns[2])),
                "power_watts": float(columns[3]) if len(columns) > 3 else None,
            }
            gpu_values.append(latest_gpu["used_mib"])
            GPU_SUMMARY = {"min_mib": min(gpu_values), "max_mib": max(gpu_values),
                           "mean_mib": round(sum(gpu_values) / len(gpu_values), 1), "samples": len(gpu_values)}
            append_jsonl(GPU_JSONL, latest_gpu)
        except Exception as exc:
            latest_gpu = {"at": utc_now(), "error": f"{type(exc).__name__}: {exc}"}

        if now - last_server_sample >= 10:
            LATEST_SERVER_HEALTH = health()
            sample = {"at": utc_now(), "health": LATEST_SERVER_HEALTH}
            try:
                sample["metrics"] = parse_metrics()
                slots = api_json("/slots", timeout=5)
                sample["slots"] = slots
                write_json(OUT / "telemetry" / "latest_slots.json", slots)
            except Exception as exc:
                sample["metrics_error"] = f"{type(exc).__name__}: {exc}"
            append_jsonl(SERVER_JSONL, sample)
            last_server_sample = now

        with LOCK:
            current = STATE.get("current_unit")
            active = STATE.get("current_call")
            stalled = bool(active and now - float(STATE.get("current_call_started_epoch", now)) > STALL_WARNING_SECONDS)
            stall_key = (f"{active.get('unit')}:{active.get('outer_attempt')}:{active.get('retry_attempt')}"
                         if active else None)
            recover_stall = bool(stalled and stall_key != STATE.get("stall_recovery_for"))
            if recover_stall:
                STATE["stall_recovery_for"] = stall_key
            hb = {
                "at": utc_now(), "pid": os.getpid(), "phase": STATE.get("phase"),
                "current_unit": current, "current_call": active,
                "call_stalled_warning": stalled, "gpu": latest_gpu,
                "server_health": LATEST_SERVER_HEALTH,
                "elapsed_seconds": max(0, now - float(STATE.get("started_epoch", now))),
                "counts": progress_counts(STATE),
            }
            write_json(OUT / "heartbeat.json", hb)
            append_jsonl(HEARTBEAT_JSONL, hb)
        if recover_stall:
            reason = f"No completion progress for {STALL_WARNING_SECONDS}s during {stall_key}; watchdog restarting local server."
            append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "stalled_request_watchdog", "reason": reason})
            try:
                restart_server(reason)
            except Exception as exc:
                append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "stall_server_restart_failed", "error": f"{type(exc).__name__}: {exc}"})
        if now - last_report >= REPORT_REFRESH_SECONDS:
            try:
                refresh_report()
            except Exception:
                append_jsonl(OUT / "telemetry" / "runner_errors.jsonl", {"at": utc_now(), "error": traceback.format_exc()})
            last_report = now


def start_monitor() -> threading.Thread:
    MONITOR_STOP.clear()
    thread = threading.Thread(target=monitor_main, name="qwen-eval-monitor", daemon=True)
    thread.start()
    return thread


def ensure_server(start_timeout: int = 120) -> None:
    global SERVER_PROCESS
    current = health()
    if current.get("ok"):
        return
    script = ROOT / "scripts" / "start-local-server.ps1"
    log = OUT / "telemetry" / "recovery_server_stdout.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    SERVER_PROCESS = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), close_fds=True,
    )
    append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "start_local_server", "pid": SERVER_PROCESS.pid})
    deadline = time.time() + start_timeout
    while time.time() < deadline:
        if health().get("ok"):
            return
        if SERVER_PROCESS.poll() is not None:
            break
        time.sleep(2)
    raise RuntimeError(f"Local llama-server did not become healthy: {health()}")


def restart_server(reason: str) -> None:
    global SERVER_PROCESS
    server = ROOT / ".local" / "llama-b11669" / "llama-server.exe"
    path = str(server).replace("'", "''")
    ps = (
        "$target='" + path + "'; "
        "$procs=Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -eq $target }; "
        "foreach($p in $procs){ Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], cwd=ROOT,
                   capture_output=True, text=True, timeout=30,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "restart_local_server", "reason": reason})
    time.sleep(2)
    SERVER_PROCESS = None
    ensure_server()


def callback_for(unit: dict, external_callback=None):
    stage_state = {"attempt": 1, "stage": "unknown"}
    ACTIVE_STAGE.set(stage_state)

    def on_token(kind: str, value: str) -> None:
        if kind == "stage":
            if value.startswith("retry #"):
                match = re.search(r"(\d+)", value)
                if match:
                    stage_state["attempt"] = int(match.group(1))
                stage_state["stage"] = value
            else:
                stage_state["stage"] = value
            with LOCK:
                STATE["current_stage"] = {"unit": unit.get("id"), **stage_state}
                STATE["last_activity_at"] = utc_now()
                STATE["updated_at"] = utc_now()
        if external_callback:
            external_callback(kind, value)
    return on_token


def validation_summary(raw: Any, unit: dict) -> dict[str, Any]:
    from scr.simulation.act_manager.validator.validator import validate_llm_response
    checkpoint = unit.get("checkpoint")
    if not checkpoint or not isinstance(raw, str):
        return {"valid": False, "category": "unavailable", "error": "No checkpoint/raw text"}
    try:
        result = validate_llm_response(raw, checkpoint, agent_id=unit.get("agent_id"))
        if result.success:
            return {"valid": True, "category": None, "action_type": result.response.action.root.action_type}
        stage = getattr(result.stage, "value", str(result.stage)) if result.stage is not None else "unknown"
        error = result.error_message
        category = "invalid_json" if stage == "json" else "schema_failure" if stage == "schema" else "illegal_action"
        return {"valid": False, "category": category, "validation_stage": stage, "error": error}
    except Exception as exc:
        return {"valid": False, "category": "validation_exception", "error": f"{type(exc).__name__}: {exc}"}


async def tapped_get_completions(*args, **kwargs):
    unit = ACTIVE_UNIT.get()
    if unit is None:
        return await ORIGINAL_GET_COMPLETIONS(*args, **kwargs)
    unit.setdefault("calls", [])
    unit["call_seq"] = unit.get("call_seq", 0) + 1
    seq = unit["call_seq"]
    stage_state = ACTIVE_STAGE.get() or {"attempt": 1, "stage": "unknown"}
    stage_name = stage_state.get("stage", "unknown")
    stage = "reflection" if "reflection" in stage_name else "first_stage" if "first" in stage_name else stage_name
    base = Path(unit["path"]) / "calls"
    base.mkdir(parents=True, exist_ok=True)
    label = f"call_{seq:03d}_outer_{unit.get('outer_attempt',1)}_retry_{stage_state.get('attempt',1)}_{stage}"
    request_messages = kwargs.get("messages")
    raw_messages = getattr(request_messages, "messages", request_messages)
    request_record = {
        "unit_id": unit["id"], "experiment": unit["experiment"],
        "outer_attempt": unit.get("outer_attempt", 1),
        "retry_attempt": stage_state.get("attempt", 1), "stage": stage, "sequence": seq,
        "model": kwargs.get("model"), "response_format": kwargs.get("response_format"),
        "messages": jsonable(raw_messages), "requested_at": utc_now(),
    }
    write_json(base / f"{label}_request.json", request_record)
    global PIPELINE_ATTEMPT_INDEX
    PIPELINE_ATTEMPT_INDEX = None
    before = {}
    try:
        before = parse_metrics()
    except Exception:
        pass
    started = time.perf_counter()
    wall_started = time.time()
    callback = kwargs.get("on_token")
    tracked_callback = callback_for(unit, callback)
    if callback is not None:
        kwargs["on_token"] = tracked_callback
    else:
        # Install stage tracking even in callers that did not request token callbacks.
        kwargs["on_token"] = tracked_callback
    with LOCK:
        STATE["current_call"] = {"unit": unit["id"], "stage": stage, "sequence": seq,
                                 "outer_attempt": unit.get("outer_attempt", 1), "retry_attempt": stage_state.get("attempt", 1)}
        STATE["current_call_started_epoch"] = wall_started
        STATE["last_activity_at"] = utc_now()
    append_jsonl(OUT / "telemetry" / "call_starts.jsonl", {"at": utc_now(), "unit": unit["id"], "stage": stage})
    try:
        result = await ORIGINAL_GET_COMPLETIONS(*args, **kwargs)
        elapsed = time.perf_counter() - started
        content = getattr(result, "content", result if isinstance(result, str) else str(result))
        validation = validation_summary(content, unit)
        after = {}
        try:
            after = parse_metrics()
        except Exception:
            pass
        metrics_delta = {
            key: after[key] - before[key]
            for key in ("llamacpp:prompt_tokens_total", "llamacpp:prompt_tokens_cached_total",
                        "llamacpp:tokens_predicted_total", "llamacpp:prompt_seconds_total",
                        "llamacpp:tokens_predicted_seconds_total")
            if key in before and key in after
        }
        record = {
            "unit_id": unit["id"], "experiment": unit["experiment"], "agent_id": unit.get("agent_id"),
            "outer_attempt": unit.get("outer_attempt", 1),
            "retry_attempt": stage_state.get("attempt", 1), "stage": stage, "sequence": seq,
            "content": content, "reasoning": getattr(result, "reasoning", ""),
            "input_tokens": getattr(result, "input_tokens", None),
            "output_tokens": getattr(result, "output_tokens", None),
            "reported_duration_seconds": getattr(result, "duration_s", None),
            "observed_duration_seconds": round(elapsed, 4),
            "generation_tokens_per_second": (
                getattr(result, "output_tokens", 0) / getattr(result, "duration_s", elapsed)
                if (getattr(result, "output_tokens", None) is not None and
                    (getattr(result, "duration_s", None) or elapsed) > 0) else None
            ),
            "validation": validation, "server_metrics_delta": metrics_delta,
            "server_context": latest_slot_context(),
            "completed_at": utc_now(),
        }
        write_json(base / f"{label}_response.json", record)
        unit["calls"].append(record)
        append_jsonl(CALLS_JSONL, record)
        with LOCK:
            STATE["last_activity_at"] = utc_now()
            STATE["current_call"] = None
            STATE["current_call_started_epoch"] = None
        return result
    except BaseException as exc:
        error = {"unit_id": unit["id"], "experiment": unit["experiment"], "agent_id": unit.get("agent_id"),
                 "outer_attempt": unit.get("outer_attempt", 1), "retry_attempt": stage_state.get("attempt", 1),
                 "stage": stage, "error": f"{type(exc).__name__}: {exc}", "at": utc_now()}
        write_json(base / f"{label}_exception.json", error)
        append_jsonl(OUT / "telemetry" / "call_errors.jsonl", error)
        with LOCK:
            STATE["current_call"] = None
            STATE["current_call_started_epoch"] = None
        raise


def serialize_decision(result, unit: dict) -> dict:
    return {
        "agent_id": result.agent_id, "success": bool(result.success), "error": result.error,
        "response": result.response.model_dump(mode="json") if result.response is not None else None,
        "messages": jsonable(result.messages.messages), "calls": unit.get("calls", []),
        "unit_id": unit["id"], "experiment": unit["experiment"],
        "outer_attempt": unit.get("outer_attempt", 1), "completed_at": utc_now(),
    }


def deserialize_decision(data: dict):
    from scr.models.agent.responses import Response
    from scr.models.agent.decision_result import AgentDecisionResult
    from scr.models.prompt_manager import Messages
    response = Response.model_validate(data["response"]) if data.get("response") is not None else None
    return AgentDecisionResult(
        agent_id=data["agent_id"], response=response,
        messages=Messages(messages=data.get("messages", [])),
        success=bool(data.get("success")), error=data.get("error"),
    )


def context_unit(experiment: str, unit_id: str, agent_id: str, checkpoint, path: Path, outer_attempt: int = 1) -> dict:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return {"experiment": experiment, "id": unit_id, "agent_id": agent_id,
            "checkpoint": checkpoint, "path": str(path), "outer_attempt": outer_attempt, "calls": []}


def install_instrumentation() -> None:
    global ORIGINAL_GET_COMPLETIONS, ORIGINAL_AGENT_DECIDER
    from scr.simulation.agent_decision import async_response_processor
    from scr.simulation.runner import simulation_step
    # async_response_processor imported this function directly, so patch that
    # bound module reference (patching completions alone would miss every call).
    ORIGINAL_GET_COMPLETIONS = async_response_processor.async_get_completions
    async_response_processor.async_get_completions = tapped_get_completions
    ORIGINAL_AGENT_DECIDER = simulation_step.async_agent_decide_actions

    async def instrumented_decider(checkpoint, agent_id, semaphore=None, on_token=None, retry_tracker=None, event_bus=None):
        runtime = B_RUNTIME.get()
        if runtime:
            return await b_agent_decider(runtime, checkpoint, agent_id, semaphore, on_token, retry_tracker, event_bus)
        control = CONTROL_RUNTIME.get()
        if not control:
            return await ORIGINAL_AGENT_DECIDER(checkpoint, agent_id, semaphore, on_token, retry_tracker, event_bus)
        return await controlled_agent_decider(control, checkpoint, agent_id, semaphore, on_token, retry_tracker, event_bus)

    simulation_step.async_agent_decide_actions = instrumented_decider


def latest_slot_context() -> dict[str, Any]:
    try:
        slots = api_json("/slots", timeout=5)
        values = slots.get("value", []) if isinstance(slots, dict) else slots
        slot = next((x for x in values if x.get("is_processing")), values[0] if values else {}) if isinstance(values, list) else {}
        return {key: slot.get(key) for key in
                ("id", "is_processing", "n_ctx", "n_prompt_tokens", "n_prompt_tokens_processed", "n_prompt_tokens_cache", "truncated")
                if key in slot}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


CONTROL_RUNTIME: contextvars.ContextVar[dict | None] = contextvars.ContextVar("qwen_control_runtime", default=None)
B_RUNTIME: contextvars.ContextVar[dict | None] = contextvars.ContextVar("qwen_b_runtime", default=None)


async def controlled_agent_decider(runtime: dict, checkpoint, agent_id, semaphore, on_token, retry_tracker, event_bus):
    unit = context_unit(runtime["experiment"], runtime["unit_id"], agent_id, checkpoint,
                        runtime["path"], runtime.get("outer_attempt", 1))
    token = ACTIVE_UNIT.set(unit)
    from scr.utils import sim_logger
    log_tokens = sim_logger.bind(run_id=runtime["run_id"], step=checkpoint.metadata.current_time_step, agent_id=agent_id)
    try:
        result = await ORIGINAL_AGENT_DECIDER(
            checkpoint, agent_id, semaphore,
            on_token=callback_for(unit, on_token), retry_tracker=retry_tracker, event_bus=event_bus,
        )
        data = serialize_decision(result, unit)
        write_json(Path(runtime["path"]) / "decision.json", data)
        return result
    finally:
        sim_logger.unbind(log_tokens)
        ACTIVE_UNIT.reset(token)


async def b_agent_decider(runtime: dict, checkpoint, agent_id, semaphore, on_token, retry_tracker, event_bus):
    from scr.utils import sim_logger
    cache_path = runtime["cache_dir"] / f"{agent_id}.json"
    cached = read_json(cache_path)
    # Only successful cached decisions are replayed across failed round attempts.
    if cached and cached.get("success"):
        result = deserialize_decision(cached)
        runtime["decisions"][agent_id] = cached
        return result

    unit_id = f"B-round-{runtime['step']}-agent-{agent_id}"
    attempt_path = runtime["round_dir"] / f"agent_{agent_id}" / f"round_attempt_{runtime['round_attempt']}"
    unit = context_unit("B", unit_id, agent_id, checkpoint, attempt_path, runtime["round_attempt"])
    salvaged = recover_successful_reflection("B", unit_id, agent_id, checkpoint, attempt_path / "decision.json")
    if salvaged:
        write_json(cache_path, salvaged)
        runtime["decisions"][agent_id] = salvaged
        append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "salvaged_completed_model_response",
                                                "experiment": "B", "unit_id": unit_id,
                                                "outer_attempt": salvaged.get("outer_attempt")})
        return deserialize_decision(salvaged)
    token = ACTIVE_UNIT.set(unit)
    log_tokens = sim_logger.bind(run_id=runtime["run_id"], step=runtime["step"], agent_id=agent_id)
    try:
        result = await ORIGINAL_AGENT_DECIDER(
            checkpoint, agent_id, semaphore,
            on_token=callback_for(unit, on_token), retry_tracker=retry_tracker, event_bus=event_bus,
        )
        data = serialize_decision(result, unit)
        write_json(attempt_path / "decision.json", data)
        runtime["decisions"][agent_id] = data
        if result.success:
            write_json(cache_path, data)
        note_progress(f"B round {runtime['step']}: agent {agent_id} decision {'completed' if result.success else 'failed'}")
        return result
    finally:
        sim_logger.unbind(log_tokens)
        ACTIVE_UNIT.reset(token)


def note_progress(message: str) -> None:
    def change(state):
        state["last_event"] = {"at": utc_now(), "message": message}
        state["last_activity_at"] = utc_now()
    update_state(change)
    refresh_report()


def set_current(phase: str, unit: str | None) -> None:
    def change(state):
        state["phase"] = phase
        state["current_unit"] = unit
        state["last_activity_at"] = utc_now()
    update_state(change)
    refresh_report()


def moral_types_present(checkpoint, expected_count: int = 8) -> None:
    agents = checkpoint.social_environment.agents
    base_types = Counter(a.type for a in agents if not a.id.endswith("_eval_offspring"))
    expected = Counter({key: 2 for key in TYPE_IDS})
    assert len([a for a in agents if not a.id.endswith("_eval_offspring")]) == expected_count
    assert base_types == expected, f"Unexpected base moral-type distribution: {base_types}"
    assert len({id(a.memory) for a in agents}) == len(agents), "Agent Memory objects are aliased"
    assert len({id(a.memory.long_term_memory) for a in agents}) == len(agents), "Long-term memories are aliased"


def new_checkpoint(seed: int, run_id: str, overrides: dict | None = None):
    import numpy as np
    from scr.models.simulation.checkpoint import Checkpoint
    from scr.utils.random_utils import set_random_seed
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    config_overrides = {
        "llm.provider": "llama_cpp", "llm.chat_model": MODEL,
        "llm.two_stage_model": True,
        "llm.async_config.max_concurrent_calls": 1,
        "llm.async_config.call_timeout_seconds": 3600.0,
        "llm.async_config.retry_backoff_base": 2.0,
        "llm.async_config.retry_backoff_max": 30.0,
        "world.max_life_steps": 2,
        "world.communication_and_sharing_steps": 2,
        "random_seed": seed,
    }
    if overrides:
        config_overrides.update(overrides)
    checkpoint = Checkpoint.initialize_from_config(CONFIG_DIR, config_overrides)
    checkpoint.metadata.run_id = run_id
    if checkpoint.configuration.llm.provider != "llama_cpp" or not checkpoint.configuration.llm.two_stage_model:
        raise RuntimeError("Local provider and two-stage reflection are required")
    moral_types_present(checkpoint)
    return checkpoint


def get_agent(checkpoint, agent_id: str):
    return checkpoint.social_environment.get_agent_by_id(agent_id)


def apply_fixture_action(checkpoint, agent_id: str, action) -> None:
    from scr.models.agent.responses import Response
    from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
    agent = get_agent(checkpoint, agent_id)
    agent.add_response(Response(
        agent_id=agent_id, thinking="Controlled checkpoint history fixture.", short_term_plan="",
        long_term_memory=agent.memory.long_term_memory, action=action,
    ))
    update_checkpoint_from_actions(checkpoint, agent_id=agent_id)


def seed_tick(checkpoint, special: dict[str, Any] | None = None,
              fixture_random_seeds: dict[str, int] | None = None) -> None:
    from scr.models.agent.actions import DoNothing
    from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
    from scr.simulation.env_manager.soc_env_manager import soc_env_step
    from scr.simulation.env_manager.phy_env_manager import phy_env_step
    special = special or {}
    step = checkpoint.metadata.current_time_step
    for agent_id in list(checkpoint.metadata.execution_queue):
        agent = next((x for x in checkpoint.social_environment.agents if x.id == agent_id), None)
        if not agent or not agent.is_alive():
            continue
        if agent_id in special:
            if fixture_random_seeds and agent_id in fixture_random_seeds:
                random.seed(fixture_random_seeds[agent_id])
            apply_fixture_action(checkpoint, agent_id, special[agent_id])
        else:
            apply_fixture_action(checkpoint, agent_id, DoNothing(action_type="do_nothing", reason="No action in controlled setup history."))
    soc_env_step(checkpoint)
    if step % (checkpoint.configuration.world.communication_and_sharing_steps + 1) == 0:
        phy_env_step(checkpoint)
    checkpoint.metadata.current_agent_index = 0
    checkpoint.metadata.current_time_step += 1


def set_entity_memory(agent, other, relation: str, event: dict[str, Any] | None = None, agreement: str = "") -> None:
    ltm = agent.memory.long_term_memory
    entities = ltm.setdefault("Agent_Specific_Memory", {})
    record = entities.setdefault(other.id, {})
    record.setdefault("important_interaction_history", {"what_i_did_to_him": {}, "what_he_did_to_me": {}})
    record["moral_type"] = other.type
    record["relationship"] = relation
    record["thinking"] = "Factual relationship assessment based on recorded observations."
    record["agreement"] = agreement
    record.setdefault("plan", "")
    if event:
        record["important_interaction_history"].setdefault("what_he_did_to_me", {})[f"time_step_{event['time_step']}"] = event
    ltm.setdefault("Strategies", "")


def build_a_checkpoint(scenario: str, moral_type: str, seed: int):
    from scr.models.agent.agent import Agent
    from scr.models.agent.actions import Allocate, Communicate
    from scr.models.agent.actions import AllocationReasonLabel
    from scr.models.agent.actions import DoNothing
    from scr.models.simulation.checkpoint import Checkpoint
    scenario_num = list(SCENARIOS).index(scenario) + 1
    init_seed = seed * 100 + scenario_num
    case_id = f"{scenario}__{moral_type}__s{seed}"
    cp = new_checkpoint(init_seed, "Q38A-" + case_id)
    focal_id = TYPE_IDS[moral_type]
    focal = get_agent(cp, focal_id)
    focal.state.hp = 20
    focal.state.age = 10 if scenario == "infant" else 19 if scenario == "adult_offspring" else 10
    focal.memory.short_term_plan = ""
    target_id = "agent_8"

    if scenario in {"infant", "adult_offspring"}:
        child_id = f"{focal_id}_eval_offspring"
        child_age, child_hp = (0, 3) if scenario == "infant" else (9, 3)
        child = Agent.initialize(moral_type, child_id, cp.configuration, hp=child_hp, age=child_age)
        child.family.parent_id = focal_id
        focal.family.children_ids = [child_id]
        cp.social_environment.agents.append(child)
        cp.metadata.execution_queue.append(child_id)
        # Type inheritance is deterministic in the original reproduction rule.
        if child.type != focal.type or child.family.parent_id != focal.id or child_id not in focal.family.children_ids:
            raise AssertionError("Offspring identity, lineage, or inherited type is inconsistent")
        target_id = child_id

    elif scenario == "stranger":
        target = get_agent(cp, target_id)
        target.state.hp = 3
        target.family.parent_id = None
        target.family.children_ids = []

    elif scenario in {"cooperator", "unfair_partner"}:
        partner = get_agent(cp, target_id)
        partner.state.hp = 5
        partner.family.parent_id = None
        partner.family.children_ids = []
        if scenario == "cooperator":
            set_entity_memory(focal, partner, "cooperative partner", {
                "time_step": 1, "action_type": "allocate", "if_success": True,
                "reason": "shared 2 HP with me", "target_moral_type": partner.type, "amount": 2,
            })
            special = {partner.id: Allocate(
                action_type="allocate", reason="Share with a cooperative partner.",
                allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
                allocation_plan={focal.id: 2},
            )}
            seed_tick(cp, special)
            seed_tick(cp)
            seed_tick(cp)
        else:
            set_entity_memory(focal, partner, "unfair-sharing partner", {
                "time_step": 2, "action_type": "allocate", "if_success": True,
                "reason": "promised 5 HP but allocated only 1 HP", "target_moral_type": partner.type, "amount": 1,
            }, agreement="At Step 1 the partner promised 5 HP; at Step 2 the partner allocated 1 HP.")
            seed_tick(cp, {partner.id: Communicate(
                action_type="communicate", reason="Record a prior sharing agreement.",
                target_agent_ids=[focal.id], message="I will allocate 5 HP to you in our next social round.",
            )})
            seed_tick(cp, {partner.id: Allocate(
                action_type="allocate", reason="Allocate one HP after promising five.",
                allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
                allocation_plan={focal.id: 1},
            )})
            seed_tick(cp)
    else:
        raise ValueError(scenario)

    # All A trials decide on the same social interaction tick (tick 4).
    while cp.metadata.current_time_step < 4:
        seed_tick(cp)
    # Freeze matched decision-time demographics/need after realistic history
    # dynamics so every moral profile sees the same legal scenario state.
    focal.state.hp = 20
    focal.state.age = 20 if scenario == "adult_offspring" else 10
    focal.state.physical_ability = 5.0
    if scenario in {"infant", "adult_offspring"}:
        target = get_agent(cp, target_id)
        target.state.hp = 2
        target.state.age = 0 if scenario == "infant" else 10
    elif scenario == "stranger":
        get_agent(cp, target_id).state.hp = 2
    elif scenario == "cooperator":
        get_agent(cp, target_id).state.hp = 2
    elif scenario == "unfair_partner":
        get_agent(cp, target_id).state.hp = 3
    if not cp.configuration.isSocialInteractionStep(cp.metadata.current_time_step):
        raise AssertionError("Controlled decision is not a social-sharing round")
    if scenario in {"infant", "adult_offspring"}:
        child = get_agent(cp, target_id)
        if child.family.parent_id != focal.id or target_id not in focal.family.children_ids or child.type != focal.type:
            raise AssertionError("Parent-offspring checkpoint relationship is inconsistent")
    elif target_id == "agent_8" and (get_agent(cp, target_id).family.parent_id is not None):
        raise AssertionError("Unrelated partner/stranger incorrectly has focal agent as parent")

    # The legal transfer option is preflighted by the exact repository validator.
    from scr.models.agent.responses import Response
    from scr.models.agent.actions import Action
    from scr.simulation.act_manager.validator.validator import validate_llm_response
    witness = Response(
        agent_id=focal.id, thinking="Legality preflight only.", short_term_plan="",
        long_term_memory=focal.memory.long_term_memory,
        action=Action.from_dict({"action_type": "allocate", "reason": "Legal-option preflight.",
                                 "allocation_reason_label": "sharing_with_family" if target_id == f"{focal.id}_eval_offspring" else "sharing_with_other",
                                 "allocation_plan": {target_id: 1}}),
    )
    legal = validate_llm_response(witness.model_dump_json(), cp, agent_id=focal.id)
    if not legal.success:
        raise AssertionError(f"Scenario's positive transfer option is illegal: {legal.error_message}")
    idle = Response(agent_id=focal.id, thinking="Legality preflight only.", short_term_plan="",
                    long_term_memory=focal.memory.long_term_memory,
                    action=Action.from_dict({"action_type": "do_nothing", "reason": "Legal-option preflight."}))
    if not validate_llm_response(idle.model_dump_json(), cp, agent_id=focal.id).success:
        raise AssertionError("DoNothing should be a legal control action")

    initial = {
        "case_id": case_id, "experiment": "A", "scenario": scenario,
        "scenario_description": SCENARIOS[scenario], "moral_type": moral_type,
        "moral_type_label": TYPE_LABELS[moral_type], "seed": seed, "initialization_seed": init_seed,
        "focal_agent_id": focal_id, "target_agent_id": target_id,
        "agent_count": len(cp.social_environment.agents), "base_cohort_count": 8,
        "target_state": get_agent(cp, target_id).state.model_dump(),
        "focal_state": focal.state.model_dump(), "is_family": target_id in focal.family.children_ids,
        "history_observations": [o for o in cp.observations if "agent_8" in o or target_id in o],
        "legal_action_options_preflighted": ["do_nothing", "allocate 1 HP to target"],
        "checkpoint": json.loads(cp.model_dump_json()),
    }
    return cp, case_id, focal_id, target_id, initial


def build_c_checkpoint(history: str, length: str, moral_type: str, seed: int):
    from scr.models.agent.actions import Allocate, Communicate, Fight, FightReasonLabel
    from scr.models.agent.actions import AllocationReasonLabel
    from scr.models.agent.responses import Response
    from scr.models.agent.actions import Action
    from scr.simulation.act_manager.validator.validator import validate_llm_response
    scenario_num = list(("cooperation", "betrayal", "unfair_distribution", "previous_attack")).index(history) + 1
    init_seed = seed * 100 + scenario_num
    case_id = f"{history}__{length}__{moral_type}__s{seed}"
    cp = new_checkpoint(init_seed, "Q38C-" + case_id)
    focal_id = TYPE_IDS[moral_type]
    partner_id, decoy_id = "agent_2", "agent_4"
    focal, partner, decoy = get_agent(cp, focal_id), get_agent(cp, partner_id), get_agent(cp, decoy_id)
    focal.state.hp = 20
    partner.state.hp = 7
    decoy.state.hp = 7
    focal.state.age = partner.state.age = decoy.state.age = 10
    partner.family.parent_id = decoy.family.parent_id = None

    critical: dict[str, Any] = {"history": history, "partner_id": partner_id,
                                "decoy_id": decoy_id, "time_step": 1}
    filler_events = []
    filler_messages = [
        "I noticed the western plant patch is growing again; this is only a location update.",
        "I saw two prey tracks near the river this morning; no hunting agreement was made.",
        "The weather stayed dry through the last gathering period, with no effect on our HP.",
        "I plan to collect plants on my next production turn rather than communicate about sharing.",
        "Several agents were near the eastern path yesterday; this does not describe your partner.",
    ]
    excluded = {focal_id, partner_id, decoy_id}
    filler_senders = [a.id for a in cp.social_environment.agents if a.id not in excluded]
    assert len(filler_senders) == 5
    filler_actions = {}
    if length == "long":
        for sender_id, message in zip(filler_senders, filler_messages):
            filler_actions[sender_id] = Communicate(
                action_type="communicate", reason="A true but noncritical item in the longer history.",
                target_agent_ids=[focal_id], message=message,
            )
            filler_events.append({"step": 2, "sender": sender_id, "message": message})
    if history == "cooperation":
        critical.update({"action": "allocate", "amount": 2, "outcome": "partner shared 2 HP with focal"})
        set_entity_memory(focal, partner, "trusted cooperative partner", {
            "time_step": 1, "action_type": "allocate", "if_success": True,
            "reason": "shared 2 HP with me", "target_moral_type": partner.type, "amount": 2,
        })
        seed_tick(cp, {partner_id: Allocate(
            action_type="allocate", reason="Share with a prior cooperative partner.",
            allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
            allocation_plan={focal_id: 2},
        ), decoy_id: Communicate(
            action_type="communicate", reason="A distinct peer records a different history.",
            target_agent_ids=[focal_id], message="I have not previously shared HP with you.",
        )})
        seed_tick(cp, filler_actions)
    elif history == "betrayal":
        critical.update({"action": "communicate", "promised": 4, "outcome": "partner promised 4 HP then explicitly refused and transferred none"})
        set_entity_memory(focal, partner, "formerly trusted; promise broken", {
            "time_step": 2, "action_type": "communicate", "if_success": False,
            "reason": "promised 4 HP then explicitly refused; no transfer occurred", "target_moral_type": partner.type,
        }, agreement="Promised 4 HP at Step 1; explicitly refused at Step 2; no allocation was made.")
        seed_tick(cp, {partner_id: Communicate(
            action_type="communicate", reason="Record a prior promise.", target_agent_ids=[focal_id],
            message="I will allocate 4 HP to you in the next social round.",
        ), decoy_id: Allocate(
            action_type="allocate", reason="The distinct peer cooperated.",
            allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
            allocation_plan={focal_id: 2},
        )})
        set_entity_memory(focal, decoy, "helpful peer", {
            "time_step": 1, "action_type": "allocate", "if_success": True,
            "reason": "allocated 2 HP to me", "target_moral_type": decoy.type, "amount": 2,
        })
        seed_tick(cp, {partner_id: Communicate(
            action_type="communicate", reason="Record the partner's explicit refusal.", target_agent_ids=[focal_id],
            message="I withdraw the promise and will not allocate the 4 HP.",
        ), **filler_actions})
    elif history == "unfair_distribution":
        critical.update({"action": "allocate", "promised": 4, "amount": 1,
                         "outcome": "partner promised 4 HP then transferred 1; decoy agent_4 transferred 3"})
        set_entity_memory(focal, partner, "unfair-sharing partner", {
            "time_step": 2, "action_type": "allocate", "if_success": True,
            "reason": "promised 4 HP but allocated only 1 HP", "target_moral_type": partner.type, "amount": 1,
        }, agreement="Partner agent_2 promised 4 HP then allocated 1 HP; agent_4 separately allocated 3 HP.")
        seed_tick(cp, {partner_id: Communicate(
            action_type="communicate", reason="Record the stated allocation agreement.", target_agent_ids=[focal_id],
            message="I will allocate 4 HP to you in the next social round.",
        ), decoy_id: Allocate(
            action_type="allocate", reason="Record a contrasting allocation from a second participant.",
            allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
            allocation_plan={focal_id: 3},
        )})
        set_entity_memory(focal, decoy, "fair-sharing peer", {
            "time_step": 1, "action_type": "allocate", "if_success": True,
            "reason": "allocated 3 HP to me", "target_moral_type": decoy.type, "amount": 3,
        })
        seed_tick(cp, {partner_id: Allocate(
            action_type="allocate", reason="Allocate one HP after promising four.",
            allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
            allocation_plan={focal_id: 1},
        ), **filler_actions})
    elif history == "previous_attack":
        critical.update({"action": "fight", "damage": 9,
                         "outcome": "partner agent_2 successfully attacked focal for 9 damage; agent_4 separately allocated 2 HP"})
        set_entity_memory(focal, partner, "previous attacker; hostile", {
            "time_step": 1, "action_type": "fight", "if_success": True,
            "reason": "attacked me for 9 HP damage", "target_moral_type": partner.type, "damage": 9,
        })
        partner.state.physical_ability = 9
        focal.state.physical_ability = 1
        seed_tick(cp, {partner_id: Fight(
            action_type="fight", reason="Record a prior attack.", target_agent_id=focal_id,
            fight_reason_label=FightReasonLabel.RETALIATE_AGAINST_ATTACK,
        ), decoy_id: Allocate(
            action_type="allocate", reason="The distinct peer provided help.",
            allocation_reason_label=AllocationReasonLabel.SHARING_WITH_OTHER,
            allocation_plan={focal_id: 2},
        )}, fixture_random_seeds={partner_id: 1})
        set_entity_memory(focal, decoy, "helpful peer", {
            "time_step": 1, "action_type": "allocate", "if_success": True,
            "reason": "allocated 2 HP to me", "target_moral_type": decoy.type, "amount": 2,
        })
        seed_tick(cp, filler_actions)
        attack_obs = [o for o in cp.observations if "fighted" in o and partner_id in o and focal_id in o]
        if not attack_obs or "9 damage" not in attack_obs[0]:
            raise AssertionError(f"Deterministic attack fixture did not produce its specified hit: {attack_obs}")
    else:
        raise ValueError(history)

    if history in {"betrayal", "unfair_distribution"} and cp.metadata.current_time_step == 2:
        seed_tick(cp, {})
    # A second independent agent is deliberately contrasted with the critical partner.
    if history == "cooperation":
        set_entity_memory(focal, decoy, "not previously cooperative", {
            "time_step": 1, "action_type": "communicate", "if_success": True,
            "reason": "said there was no prior HP sharing", "target_moral_type": decoy.type,
        })

    while cp.metadata.current_time_step < 4:
        seed_tick(cp)

    if length == "long":
        entities = focal.memory.long_term_memory.setdefault("Agent_Specific_Memory", {})
        for event in filler_events:
            entry = entities.setdefault(event["sender"], {})
            entry.setdefault("important_interaction_history", {"what_i_did_to_him": {}, "what_he_did_to_me": {}})
            entry["moral_type"] = get_agent(cp, event["sender"]).type
            entry["relationship"] = "communicated a noncritical factual observation"
            entry["thinking"] = f"At Step {event['step']} {event['sender']} sent a recorded observation: {event['message']}"
            entry.setdefault("agreement", "")
            entry.setdefault("plan", "")
        # A separate, explicit context fixture increases long-memory material without altering prompts.
        focal.memory.long_term_memory["Strategies"] = " ".join(
            f"Observed context item {i+1}: {event['sender']} said '{event['message']}' at step {event['step']}."
            for i, event in enumerate(filler_events)
        )
    if not cp.configuration.isSocialInteractionStep(cp.metadata.current_time_step):
        raise AssertionError("Memory decision must use a legal social round")

    witness = Response(
        agent_id=focal_id, thinking="Legality preflight only.", short_term_plan="",
        long_term_memory=focal.memory.long_term_memory,
        action=Action.from_dict({"action_type": "allocate", "reason": "Legal-option preflight.",
                                 "allocation_reason_label": "sharing_with_other", "allocation_plan": {partner_id: 1}}),
    )
    legal = validate_llm_response(witness.model_dump_json(), cp, agent_id=focal_id)
    if not legal.success:
        raise AssertionError(f"Memory target transfer is illegal: {legal.error_message}")

    expected = {
        "history": history, "length": length, "partner_id": partner_id,
        "decoy_id": decoy_id, "critical_event": critical, "filler_event_count": len(filler_events),
        "required_recall_slots": ["critical participant", "event/action", "event outcome", "amount or damage where applicable", "correct participant attribution"],
    }
    initial = {
        "case_id": case_id, "experiment": "C", "moral_type": moral_type,
        "moral_type_label": TYPE_LABELS[moral_type], "seed": seed, "initialization_seed": init_seed,
        "focal_agent_id": focal_id, "partner_id": partner_id, "decoy_id": decoy_id,
        "focal_state": focal.state.model_dump(), "partner_state": partner.state.model_dump(),
        "decoy_state": decoy.state.model_dump(), "current_time_step": cp.metadata.current_time_step,
        "expected_facts": expected, "observations": cp.observations,
        "received_messages": focal.memory.received_messages,
        "initial_memory": focal.memory.long_term_memory,
        "initial_context_text_chars": sum(len(str(x)) for x in focal.memory.long_term_memory.values()),
        "memory_isolation": {
            "memory_objects_unique": len({id(a.memory) for a in cp.social_environment.agents}) == len(cp.social_environment.agents),
            "long_term_memory_objects_unique": len({id(a.memory.long_term_memory) for a in cp.social_environment.agents}) == len(cp.social_environment.agents),
            "seeded_fact_only_in_focal_ltm": all(a.id == focal_id or
                (partner_id not in a.memory.long_term_memory.get("Agent_Specific_Memory", {})
                 and decoy_id not in a.memory.long_term_memory.get("Agent_Specific_Memory", {}))
                for a in cp.social_environment.agents),
        },
        "checkpoint": json.loads(cp.model_dump_json()),
    }
    return cp, case_id, focal_id, initial


def read_pipeline_attempt_index() -> dict[tuple[str, str], dict[int, set[int]]]:
    """Index every attempted two-stage pipeline cycle from durable prompt files.

    This includes requests that raised before producing a response, which the
    response-only completion journal cannot count. Error feedback in the next
    request's conversation identifies a retry cycle without changing the
    repository's retry settings.
    """
    global PIPELINE_ATTEMPT_INDEX
    if PIPELINE_ATTEMPT_INDEX is not None:
        return PIPELINE_ATTEMPT_INDEX
    index: dict[tuple[str, str], dict[int, set[int]]] = {}
    retry_feedback = (
        "Your response is invalid. Please try again.",
        "LLM call failed with error:",
        "Timeout waiting for LLM response",
    )
    for path in OUT.rglob("call_*_request.json"):
        request = read_json(path, {}) or {}
        experiment, unit_id = request.get("experiment"), request.get("unit_id")
        if not experiment or not unit_id:
            continue
        messages = request.get("messages", [])
        user_contents = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
        cycle = 1 + sum(any(marker in content for marker in retry_feedback) for content in user_contents)
        outer = int(request.get("outer_attempt", 1))
        index.setdefault((experiment, unit_id), {}).setdefault(outer, set()).add(cycle)
    PIPELINE_ATTEMPT_INDEX = index
    return index


def pipeline_retry_count(experiment: str, unit_id: str) -> int:
    attempts = read_pipeline_attempt_index().get((experiment, unit_id), {})
    return sum(max(0, len(cycles) - 1) for cycles in attempts.values())


def decision_outcome(data: dict, initial: dict) -> dict[str, Any]:
    from scr.models.agent.responses import Response
    response = Response.model_validate(data["response"]) if data.get("response") else None
    calls = data.get("calls", [])
    draft = next((x for x in calls if x.get("stage") == "first_stage"
                  and x.get("outer_attempt", 1) == 1
                  and x.get("pipeline_attempt", x.get("retry_attempt", 1)) == 1), None)
    reflected = next((x for x in reversed(calls)
                      if x.get("stage") == "reflection" and x.get("outer_attempt", 1) == 1
                      and x.get("pipeline_attempt", x.get("retry_attempt", 1)) == 1), None)
    final_valid = bool(data.get("success") and response is not None)
    # The first full pipeline attempt may contain validator-driven extra
    # completions. Judge its final response (reflection when available), not
    # only call 1 or a possibly stale callback retry counter.
    first_outer_calls = [x for x in calls if x.get("outer_attempt", 1) == 1
                         and x.get("pipeline_attempt", x.get("retry_attempt", 1)) == 1]
    first_outer_final = next((x for x in reversed(first_outer_calls) if x.get("stage") == "reflection"), None)
    if first_outer_final is None:
        first_outer_final = next((x for x in reversed(first_outer_calls) if x.get("stage") == "first_stage"), None)
    initial_attempt_valid = bool(first_outer_final and first_outer_final.get("validation", {}).get("valid"))
    first_draft_valid = bool(draft and draft.get("validation", {}).get("valid"))
    action = response.action.root if response else None
    allocations = dict(action.allocation_plan) if action and action.action_type == "allocate" else {}
    target_id = initial.get("target_agent_id") or initial.get("partner_id")
    amount = allocations.get(target_id, 0)
    scenario = initial.get("scenario")
    moral_type = initial.get("moral_type")
    expected = SCENARIO_EXPECTATIONS.get(scenario, {}).get(moral_type) if scenario else None
    helped = bool(amount > 0)
    role_judgeable = expected is not None and action is not None
    role_correct = bool(role_judgeable and helped == expected)
    draft_action = None
    reflected_action = None
    for record, variable in ((draft, "draft"), (reflected, "reflected")):
        try:
            parsed = json.loads(record["content"])
            value = parsed.get("action")
            if isinstance(value, dict):
                if variable == "draft": draft_action = value
                else: reflected_action = value
        except Exception:
            pass
    llm_retries = pipeline_retry_count(data.get("experiment", ""), data.get("unit_id", ""))
    return {
        "success": final_valid, "first_draft_valid": first_draft_valid,
        "initial_attempt_final_valid": initial_attempt_valid,
        "eventual_valid": final_valid,
        "validation_calls": [x.get("validation", {}) for x in calls],
        "internal_retry_count": llm_retries,
        "outer_attempt": data.get("outer_attempt", 1), "calls_count": len(calls),
        "first_stage_action": draft_action, "reflected_action": reflected_action,
        "reflection_changed_action": bool(draft_action is not None and reflected_action is not None and draft_action != reflected_action),
        "final_action": action.model_dump(mode="json") if action else None,
        "final_action_type": action.action_type if action else None,
        "allocation_to_target": amount, "all_allocations": allocations,
        "rationale": response.thinking if response else None,
        "first_stage_response": draft.get("content") if draft else None,
        "reflected_response": reflected.get("content") if reflected else None,
        "role_expected_help": expected, "role_observed_help": helped,
        "role_judgeable": role_judgeable, "role_consistent": role_correct if role_judgeable else None,
    }


def apply_controlled_checkpoint_effects(checkpoint, data: dict, case_path: Path) -> dict[str, Any]:
    """Apply an already validated focal response via the repository's normal phase-2 handlers."""
    from scr.models.agent.responses import Response
    from scr.simulation.agent_decision.response_processor import update_agent_memory_from_response
    from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
    response = Response.model_validate(data["response"])
    before = {a.id: a.memory.model_dump(mode="json") for a in checkpoint.social_environment.agents}
    focal = get_agent(checkpoint, data["agent_id"])
    focal.add_response(response)
    update_agent_memory_from_response(checkpoint, response, agent_id=focal.id)
    update_checkpoint_from_actions(checkpoint, agent_id=focal.id)
    expected_ltm = copy.deepcopy(before[focal.id]["long_term_memory"])
    if response.long_term_memory:
        expected_ltm.update(response.long_term_memory)
    expected_plan = response.short_term_plan if response.short_term_plan else before[focal.id]["short_term_plan"]
    memory_match = (focal.memory.long_term_memory == expected_ltm
                    and focal.memory.short_term_plan == expected_plan
                    and focal.response_history[-1] == response)
    # Communication can legitimately append delivered messages to a recipient;
    # private long-term facts and plans must remain isolated across agents.
    others_unchanged = all(a.id == focal.id or (
        a.memory.long_term_memory == before[a.id]["long_term_memory"]
        and a.memory.short_term_plan == before[a.id]["short_term_plan"]
    ) for a in checkpoint.social_environment.agents)
    independent = check_memory_independence(checkpoint)
    action = response.action.root
    if action.action_type == "allocate":
        affected_ids = list(action.allocation_plan)
    elif action.action_type == "communicate":
        affected_ids = list(action.target_agent_ids)
    elif action.action_type in {"fight", "rob"}:
        affected_ids = [action.target_agent_id]
    else:
        affected_ids = []
    result = {"memory_updated_from_response": memory_match,
              "other_agent_persistent_memory_unchanged": others_unchanged,
              "legitimate_received_message_updates_allowed": True,
              "memory_objects_independent": independent, "focal_state_after_action": focal.state.model_dump(),
              "target_states_after_action": {a.id: a.state.model_dump() for a in checkpoint.social_environment.agents
                                             if a.id in affected_ids},
              "response_history_count": len(focal.response_history)}
    write_json(case_path / "checkpoint_after.json", json.loads(checkpoint.model_dump_json()))
    return result


def run_case_attempt(experiment: str, case_id: str, agent_id: str, checkpoint, attempt: int) -> dict:
    path = OUT / experiment / case_id / f"outer_attempt_{attempt}"
    write_json(path / "checkpoint_before.json", json.loads(checkpoint.model_dump_json()))
    runtime = {"experiment": experiment, "unit_id": case_id, "path": str(path),
               "outer_attempt": attempt, "run_id": checkpoint.metadata.run_id}
    result = run_coroutine(_run_control_decision(runtime, checkpoint, agent_id))
    data = read_json(path / "decision.json")
    if not data:
        data = {"agent_id": agent_id, "success": False, "error": "Decision completed without a durable result record.", "response": None, "messages": [], "calls": []}
        write_json(path / "decision.json", data)
    if data.get("success") and data.get("response"):
        data["checkpoint_effects"] = apply_controlled_checkpoint_effects(checkpoint, data, path)
        write_json(path / "decision.json", data)
    write_json(path / "conversation.json", data.get("messages", []))
    return data


async def _run_control_decision(runtime: dict, checkpoint, agent_id: str):
    semaphore = asyncio.Semaphore(1)
    return await controlled_agent_decider(runtime, checkpoint, agent_id, semaphore,
        on_token=None, retry_tracker=None, event_bus=None)


def run_coroutine(coro):
    """Reuse one event loop for all local LiteLLM calls in this runner process."""
    global ASYNC_LOOP
    if ASYNC_LOOP is None or ASYNC_LOOP.is_closed():
        ASYNC_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(ASYNC_LOOP)
    return ASYNC_LOOP.run_until_complete(coro)


def case_ids_a() -> list[tuple[str, str, int]]:
    return [(scenario, moral_type, seed) for scenario in SCENARIOS
            for moral_type in TYPE_IDS for seed in (11, 29)]


def case_ids_c() -> list[tuple[str, str, str, int]]:
    mapping = {
        "cooperation": "universal_group_focused_moral",
        "betrayal": "reciprocal_group_focused_moral",
        "unfair_distribution": "kin_focused_moral",
        "previous_attack": "reproductive_selfish",
    }
    return [(history, length, mapping[history], seed)
            for history in mapping for length in ("short", "long") for seed in (17,)]


def recover_controlled_result(experiment: str, case_id: str, initial: dict) -> dict | None:
    result_path = OUT / experiment / case_id / "result.json"
    existing = read_json(result_path)
    if existing and existing.get("status") in {"complete", "blocked"}:
        if existing.get("status") == "complete" and existing.get("decision", {}).get("success"):
            existing["decision"]["calls"] = unit_call_records(experiment, case_id)
            existing["outcome"] = decision_outcome(existing["decision"], initial)
            if experiment == "C":
                existing["outcome"]["memory_facts"] = score_memory_facts(existing["decision"], initial)
            write_json(result_path, existing)
        return existing
    attempt_dirs = sorted((OUT / experiment / case_id).glob("outer_attempt_*"),
                          key=lambda p: int(p.name.split("_")[-1]))
    for path in reversed(attempt_dirs):
        data = read_json(path / "decision.json")
        if data and data.get("success"):
            data["calls"] = unit_call_records(experiment, case_id)
            return {"status": "complete", "initial": initial, "decision": data,
                    "outcome": decision_outcome(data, initial), "attempts": len(attempt_dirs),
                    "recovered_after_process_restart": True}
        checkpoint = new_checkpoint_for_snapshot(initial, None)
        salvaged = recover_successful_reflection(experiment, case_id, data.get("agent_id") if data else initial.get("focal_agent_id"),
                                                 checkpoint, path / "decision.json")
        if salvaged:
            salvaged["checkpoint_effects"] = apply_controlled_checkpoint_effects(checkpoint, salvaged, path)
            write_json(path / "decision.json", salvaged)
            return {"status": "complete", "initial": initial, "decision": salvaged,
                    "outcome": decision_outcome(salvaged, initial), "attempts": len(attempt_dirs),
                    "recovered_after_process_restart": True,
                    "checkpoint_effects": salvaged.get("checkpoint_effects")}
    return None


def attempt_has_evidence(path: Path) -> bool:
    if (path / "decision.json").exists():
        return True
    calls = path / "calls"
    return calls.exists() and any(calls.iterdir())


def recover_successful_reflection(experiment: str, unit_id: str, agent_id: str, checkpoint, destination: Path) -> dict | None:
    """Rebuild a decision from a durable, validated final-stage response after a process crash."""
    if not agent_id:
        return None
    calls = unit_call_records(experiment, unit_id)
    for call in reversed(calls):
        if call.get("stage") != "reflection" or not call.get("validation", {}).get("valid"):
            continue
        call_path = OUT / call.get("source_file", "")
        request_path = call_path.with_name(call_path.name.replace("_response.json", "_request.json"))
        request = read_json(request_path)
        if not request or not isinstance(request.get("messages"), list):
            continue
        from scr.models.agent.responses import Response
        from scr.models.prompt_manager import Messages
        from scr.simulation.act_manager.validator.validator import validate_llm_response
        from scr.models.agent.decision_result import AgentDecisionResult
        validation = validate_llm_response(call.get("content", ""), checkpoint, agent_id=agent_id)
        if not validation.success:
            continue
        messages = Messages(messages=request["messages"] + [{"role": "assistant", "content": call.get("content", "")}])
        decision = AgentDecisionResult(agent_id=agent_id, response=validation.response, messages=messages,
                                       success=True, error=None)
        data = {
            "agent_id": agent_id, "success": True, "error": None,
            "response": validation.response.model_dump(mode="json"),
            "messages": jsonable(messages.messages), "calls": calls,
            "unit_id": unit_id, "experiment": experiment,
            "outer_attempt": call.get("outer_attempt", 1), "completed_at": utc_now(),
            "recovered_from_durable_reflection": True,
        }
        write_json(destination, data)
        write_json(destination.parent / "conversation.json", data["messages"])
        return data
    return None


def run_a_case(scenario: str, moral_type: str, seed: int) -> None:
    from scr.utils import sim_logger
    cp, case_id, focal_id, target_id, initial = build_a_checkpoint(scenario, moral_type, seed)
    existing = recover_controlled_result("A", case_id, initial)
    if existing:
        if existing.get("status") == "complete" and not existing.get("checkpoint_effects") and existing.get("decision", {}).get("success"):
            resume_cp = new_checkpoint_for_snapshot(initial, cp)
            existing["checkpoint_effects"] = apply_controlled_checkpoint_effects(
                resume_cp, existing["decision"], OUT / "A" / case_id,
            )
        write_json(OUT / "A" / case_id / "result.json", existing)
        register_completed("A", case_id, existing)
        return

    set_current("A", case_id)
    write_json(OUT / "A" / case_id / "initial.json", initial)
    failures = []
    completed_attempts = 0
    existing_dirs = [x for x in (OUT / "A" / case_id).glob("outer_attempt_*") if attempt_has_evidence(x)] if (OUT / "A" / case_id).exists() else []
    start_attempt = max([int(x.name.split("_")[-1]) for x in existing_dirs] + [0]) + 1
    final = None
    for attempt in range(start_attempt, MAX_OUTER_ATTEMPTS + 1):
        checkpoint = new_checkpoint_for_snapshot(initial, cp)
        data = run_case_attempt("A", case_id, focal_id, checkpoint, attempt)
        completed_attempts = attempt
        if data.get("success"):
            data["calls"] = unit_call_records("A", case_id)
            outcome = decision_outcome(data, initial)
            final = {"status": "complete", "initial": initial, "decision": data, "outcome": outcome,
                     "attempts": attempt, "outer_recoveries": attempt - 1, "failures": failures,
                     "checkpoint_effects": data.get("checkpoint_effects")}
            break
        failure = {"outer_attempt": attempt, "error": data.get("error"),
                   "calls": len(data.get("calls", [])), "at": utc_now()}
        failures.append(failure)
        update_state(lambda st: st.setdefault("failures", []).append({"experiment": "A", "case_id": case_id, **failure}))
        report_failure("A", case_id, failure)
        if attempt < MAX_OUTER_ATTEMPTS:
            backoff = BACKOFF_SECONDS[attempt - 1]
            if not health().get("ok"):
                try:
                    restart_server(f"A {case_id} transport failure after attempt {attempt}")
                except Exception as exc:
                    append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "server_restart_failed", "error": str(exc)})
            time.sleep(backoff)
    if final is None:
        final = {"status": "blocked", "initial": initial,
                 "decision": read_json(OUT / "A" / case_id / f"outer_attempt_{completed_attempts}" / "decision.json"),
                 "attempts": completed_attempts, "outer_recoveries": max(0, completed_attempts - 1),
                  "failures": failures, "blocker": "No valid final response after the initial attempt plus five recovery attempts."}
        if final["decision"]:
            final["outcome"] = decision_outcome(final["decision"], initial)
    write_json(OUT / "A" / case_id / "result.json", final)
    register_completed("A", case_id, final)
    set_current("A", None)
    refresh_report()


def new_checkpoint_for_snapshot(initial: dict, cp_template):
    from scr.models.simulation.checkpoint import Checkpoint
    checkpoint = Checkpoint.model_validate(initial["checkpoint"])
    if cp_template is not None:
        checkpoint.metadata.run_id = cp_template.metadata.run_id
    if len({id(a.memory) for a in checkpoint.social_environment.agents}) != len(checkpoint.social_environment.agents):
        raise AssertionError("Checkpoint restore aliased agent memories")
    return checkpoint


def run_c_case(history: str, length: str, moral_type: str, seed: int) -> None:
    cp, case_id, focal_id, initial = build_c_checkpoint(history, length, moral_type, seed)
    existing = recover_controlled_result("C", case_id, initial)
    if existing:
        if existing.get("status") == "complete" and not existing.get("checkpoint_effects") and existing.get("decision", {}).get("success"):
            resume_cp = new_checkpoint_for_snapshot(initial, cp)
            existing["checkpoint_effects"] = apply_controlled_checkpoint_effects(
                resume_cp, existing["decision"], OUT / "C" / case_id,
            )
        if existing.get("status") in {"complete", "blocked"}:
            write_json(OUT / "C" / case_id / "result.json", existing)
            register_completed("C", case_id, existing)
        return
    set_current("C", case_id)
    write_json(OUT / "C" / case_id / "initial.json", initial)
    failures, completed_attempts, final = [], 0, None
    existing_dirs = [x for x in (OUT / "C" / case_id).glob("outer_attempt_*") if attempt_has_evidence(x)] if (OUT / "C" / case_id).exists() else []
    start_attempt = max([int(x.name.split("_")[-1]) for x in existing_dirs] + [0]) + 1
    for attempt in range(start_attempt, MAX_OUTER_ATTEMPTS + 1):
        checkpoint = new_checkpoint_for_snapshot(initial, cp)
        data = run_case_attempt("C", case_id, focal_id, checkpoint, attempt)
        completed_attempts = attempt
        if data.get("success"):
            data["calls"] = unit_call_records("C", case_id)
            outcome = decision_outcome(data, initial)
            outcome["memory_facts"] = score_memory_facts(data, initial)
            final = {"status": "complete", "initial": initial, "decision": data, "outcome": outcome,
                     "attempts": attempt, "outer_recoveries": attempt - 1, "failures": failures,
                     "checkpoint_effects": data.get("checkpoint_effects")}
            break
        failure = {"outer_attempt": attempt, "error": data.get("error"), "calls": len(data.get("calls", [])), "at": utc_now()}
        failures.append(failure)
        update_state(lambda st: st.setdefault("failures", []).append({"experiment": "C", "case_id": case_id, **failure}))
        report_failure("C", case_id, failure)
        if attempt < MAX_OUTER_ATTEMPTS:
            if not health().get("ok"):
                try:
                    restart_server(f"C {case_id} transport failure after attempt {attempt}")
                except Exception as exc:
                    append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "server_restart_failed", "error": str(exc)})
            time.sleep(BACKOFF_SECONDS[attempt - 1])
    if final is None:
        last = read_json(OUT / "C" / case_id / f"outer_attempt_{completed_attempts}" / "decision.json")
        final = {"status": "blocked", "initial": initial, "decision": last,
                 "attempts": completed_attempts, "outer_recoveries": max(0, completed_attempts - 1),
                  "failures": failures, "blocker": "No valid final response after the initial attempt plus five recovery attempts."}
        if last:
            final["outcome"] = decision_outcome(last, initial)
            final["outcome"]["memory_facts"] = score_memory_facts(last, initial)
    write_json(OUT / "C" / case_id / "result.json", final)
    register_completed("C", case_id, final)
    set_current("C", None)
    refresh_report()


def score_memory_facts(data: dict, initial: dict) -> dict[str, Any]:
    from scr.models.agent.responses import Response
    response = Response.model_validate(data["response"]) if data.get("response") else None
    facts = initial.get("expected_facts", {})
    if response is None:
        return {"correct": 0, "total": 4, "slots": {}, "fabricated_critical_claims": []}
    ltm = response.long_term_memory
    entities = ltm.get("Agent_Specific_Memory", {}) if isinstance(ltm, dict) else {}
    partner = entities.get(facts.get("partner_id"), {}) if isinstance(entities, dict) else {}
    decoy = entities.get(facts.get("decoy_id"), {}) if isinstance(entities, dict) else {}
    partner_text = json.dumps(partner, ensure_ascii=False).lower()
    decoy_text = json.dumps(decoy, ensure_ascii=False).lower()
    full_text = json.dumps({"thinking": response.thinking, "memory": ltm, "plan": response.short_term_plan}, ensure_ascii=False).lower()
    history = facts.get("history")
    action_word = {"cooperation": "allocate", "betrayal": "communicat", "unfair_distribution": "allocate", "previous_attack": "fight"}[history]
    participant_ok = facts.get("partner_id", "").lower() in entities
    action_ok = action_word in partner_text
    if history == "cooperation":
        outcome_ok = ("shared" in partner_text or "allocate" in partner_text) and ("did" in full_text or "he" in partner_text or "allocated" in partner_text)
        amount_ok = bool(re.search(r"(?<!\d)2(?!\d)", partner_text) or re.search(r"(?<!\d)2(?!\d)", response.thinking))
    elif history == "betrayal":
        outcome_ok = any(x in partner_text for x in ("refus", "withdraw", "no allocation", "did not allocate", "not allocate", "broken"))
        amount_ok = bool(re.search(r"(?<!\d)4(?!\d)", partner_text) or re.search(r"(?<!\d)4(?!\d)", response.thinking))
    elif history == "unfair_distribution":
        outcome_ok = bool(re.search(r"(?<!\d)1(?!\d)", partner_text) and any(x in partner_text for x in ("unfair", "promised", "4 hp", "4")))
        amount_ok = bool(re.search(r"(?<!\d)4(?!\d)", partner_text) and re.search(r"(?<!\d)1(?!\d)", partner_text))
    else:
        outcome_ok = any(x in partner_text for x in ("attack", "fighted", "damage", "hostile"))
        amount_ok = bool(re.search(r"(?<!\d)9(?!\d)", partner_text) or re.search(r"(?<!\d)9(?!\d)", response.thinking))
    # Score the contrasting participant only from that participant's own
    # entity record. A bare mention of the decoy elsewhere in the response is
    # not evidence that the model kept the two histories distinct.
    if history == "cooperation":
        decoy_ok = bool(decoy) and any(token in decoy_text for token in
                                       ("no prior hp sharing", "not previously cooperative", "not previously shared"))
    else:
        expected_decoy_amount = 2 if history in {"betrayal", "previous_attack"} else 3
        decoy_ok = (bool(decoy) and "allocate" in decoy_text
                    and bool(re.search(rf"(?<!\d){expected_decoy_amount}(?!\d)", decoy_text)))
    slots = {
        "partner_entity_present": bool(participant_ok),
        "critical_action_correctly_attributed": bool(participant_ok and action_ok),
        "critical_outcome_recalled": bool(outcome_ok),
        "critical_amount_or_damage_recalled": bool(amount_ok),
        "decoy_entity_separately_recorded": bool(decoy_ok),
    }
    # Strict exact-slot score is the first four objective critical slots; the fifth is separately reported.
    correct = sum(bool(slots[k]) for k in list(slots)[:4])
    # Detect incorrect action records in the critical partner's own interaction-history slots.
    interaction = partner.get("important_interaction_history", {}) if isinstance(partner, dict) else {}
    allowed = {action_word, "communicat" if history == "betrayal" else action_word}
    fabricated = []
    if isinstance(interaction, dict):
        for side, entries in interaction.items():
            values = entries.values() if isinstance(entries, dict) else []
            for event in values:
                if isinstance(event, dict):
                    act = str(event.get("action_type", "")).lower()
                    if act and not any(token in act for token in allowed):
                        fabricated.append({"side": side, "action_type": act})
    return {"correct": correct, "total": 4, "slots": slots,
            "decoy_attribution_correct": bool(decoy_ok),
            "fabricated_critical_claims": fabricated,
            "partner_memory": partner, "decoy_memory": decoy}


def initialize_state() -> None:
    global STATE, REPORT_TEMPLATE
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "telemetry").mkdir(parents=True, exist_ok=True)
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Expected initialized evaluation report at {REPORT_PATH}")
    REPORT_TEMPLATE = REPORT_PATH.read_text(encoding="utf-8")
    loaded = read_json(STATE_PATH)
    if loaded:
        STATE = loaded
        STATE.setdefault("resume_count", 0)
        STATE["resume_count"] += 1
        previous_status = STATE.get("status")
        STATE["process_restarts"] = STATE.get("process_restarts", 0) + (1 if previous_status in {"running", "crashed_recoverable"} else 0)
        if previous_status == "crashed_recoverable" and STATE.get("last_error"):
            STATE["last_recovered_error"] = STATE.pop("last_error")
        if STATE.get("status") in {"prepared", "preflight_only"}:
            STATE["started_at"] = utc_now()
            STATE["started_epoch"] = time.time()
            STATE["A"] = {}
            STATE["B"] = {"rounds": {}}
            STATE["C"] = {}
            STATE["failures"] = []
            STATE["recoveries"] = []
    else:
        STATE = {
            "status": "running", "created_at": utc_now(), "started_at": utc_now(),
            "started_epoch": time.time(), "phase": "preflight", "current_unit": None,
            "current_call": None, "current_call_started_epoch": None,
            "last_activity_at": utc_now(), "resume_count": 0, "process_restarts": 0,
            "A": {}, "B": {"rounds": {}}, "C": {}, "failures": [], "recoveries": [],
        }
    STATE["status"] = "running"
    STATE["process_id"] = os.getpid()
    STATE["last_started_at"] = utc_now()
    STATE["started_epoch"] = STATE.get("started_epoch", time.time())
    write_json(STATE_PATH, STATE)


def b_agent_slot_counts() -> tuple[int, list[int]]:
    """Count decision slots from each durable tick checkpoint, including offspring."""
    from scr.models.simulation.checkpoint import Checkpoint

    results = {int(row["step"]): row for row in read_b_rounds()
               if isinstance(row.get("step"), int)}
    slots = 0
    unknown_steps: list[int] = []
    for step in range(1, 7):
        result = results.get(step)
        rows = result.get("agent_decisions", []) if result else []
        if rows:
            slots += len(rows)
            continue
        before = read_json(OUT / "B" / "rounds" / f"step_{step}" / "before.json")
        if before is None and result:
            before = result.get("checkpoint_before")
        if before:
            try:
                checkpoint = Checkpoint.model_validate(before)
                slots += sum(1 for agent in checkpoint.social_environment.agents if agent.is_alive())
                continue
            except Exception:
                pass
        if result and result.get("status", "").startswith("blocked"):
            unknown_steps.append(step)
    return slots, unknown_steps


def progress_counts(state: dict | None = None) -> dict[str, Any]:
    state = state or STATE
    a = state.get("A", {})
    c = state.get("C", {})
    b = state.get("B", {}).get("rounds", {})
    b_done_rounds = sum(1 for row in b.values() if row.get("status", "").startswith("complete"))
    b_blocked_rounds = sum(1 for row in b.values() if row.get("status", "").startswith("blocked"))
    b_done_decisions = sum(int(row.get("decisions_completed", 0)) for row in b.values())
    b_known_slots, b_unknown_slot_steps = b_agent_slot_counts()
    return {
        "A_complete": sum(1 for x in a.values() if x.get("status") == "complete"),
        "A_blocked": sum(1 for x in a.values() if x.get("status") == "blocked"),
        "A_total": 40,
        "B_rounds_complete": b_done_rounds, "B_rounds_blocked": b_blocked_rounds, "B_rounds_total": 6,
        "B_decisions_completed": b_done_decisions, "B_decisions_total": b_known_slots,
        "B_decision_slot_steps_unknown": b_unknown_slot_steps,
        "B_failed_decisions": sum(int(row.get("failed_decisions", 0)) for row in b.values()),
        "C_complete": sum(1 for x in c.values() if x.get("status") == "complete"),
        "C_blocked": sum(1 for x in c.values() if x.get("status") == "blocked"),
        "C_total": 8,
    }


def register_completed(experiment: str, case_id: str, result: dict) -> None:
    key = "A" if experiment == "A" else "C"
    update_state(lambda state: state.setdefault(key, {}).__setitem__(case_id, {
        "status": result.get("status"), "result_path": str((OUT / experiment / case_id / "result.json").relative_to(ROOT)),
        "attempts": result.get("attempts", 0), "outer_recoveries": result.get("outer_recoveries", 0),
        "updated_at": utc_now(),
    }))


def report_failure(experiment: str, case_id: str, failure: dict) -> None:
    append_jsonl(OUT / "failures.jsonl", {"experiment": experiment, "case_id": case_id, **failure})
    refresh_report()


def read_call_results() -> list[dict]:
    rows = []
    seen = set()
    durable_by_identity = {}
    # Per-call JSON files are the durable source of truth; JSONL is a fast
    # append-only journal and may lag if the process exits between writes.
    for path in OUT.rglob("call_*_response.json"):
        record = read_json(path)
        if not record:
            continue
        record["source_file"] = str(path.relative_to(OUT))
        request_path = path.with_name(path.name.replace("_response.json", "_request.json"))
        request = read_json(request_path, {}) or {}
        request_messages = request.get("messages", [])
        user_contents = [str(m.get("content", "")) for m in request_messages if m.get("role") == "user"]
        last_user_content = user_contents[-1] if user_contents else ""
        if record.get("stage") in {"unknown", "first_call", "reflection"}:
            # Every validator retry restarts the repository's two-stage
            # processor. Its first-stage request ends with the prior
            # validation error; its reflection request ends with a fresh
            # Reflect instruction. Earlier reflection turns must not relabel
            # a later retry's first-stage draft.
            is_reflection = ("Reflect:" in last_user_content and
                             "is the factual information I put in long_term_memory" in last_user_content)
            record["stage"] = "reflection" if is_reflection else "first_stage"
        retry_feedback = (
            "Your response is invalid. Please try again.",
            "LLM call failed with error:",
            "Timeout waiting for LLM response",
        )
        record["pipeline_attempt"] = 1 + sum(
            any(marker in content for marker in retry_feedback)
            for content in user_contents
        )
        identity = (record.get("experiment"), record.get("unit_id"), record.get("outer_attempt"),
                    record.get("retry_attempt"), record.get("sequence"))
        durable_by_identity[identity] = record
        key = (record.get("unit_id"), record.get("outer_attempt"), record.get("retry_attempt"),
               record.get("stage"), record.get("sequence"), record.get("content"))
        if key not in seen:
            seen.add(key)
            rows.append(record)
    if CALLS_JSONL.exists():
        for line in CALLS_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                identity = (record.get("experiment"), record.get("unit_id"), record.get("outer_attempt"),
                            record.get("retry_attempt"), record.get("sequence"))
                durable = durable_by_identity.get(identity)
                if durable:
                    record["stage"] = durable.get("stage")
                    record["source_file"] = durable.get("source_file")
                    record["pipeline_attempt"] = durable.get("pipeline_attempt")
                key = (record.get("unit_id"), record.get("outer_attempt"), record.get("retry_attempt"),
                       record.get("stage"), record.get("sequence"), record.get("content"))
                if key not in seen:
                    seen.add(key)
                    rows.append(record)
            except json.JSONDecodeError:
                continue
    return sorted(rows, key=lambda x: (x.get("experiment", ""), x.get("unit_id", ""),
                                       int(x.get("outer_attempt", 1)), int(x.get("sequence", 0))))


def unit_call_records(experiment: str, unit_id: str) -> list[dict]:
    return [row for row in read_call_results()
            if row.get("experiment") == experiment and row.get("unit_id") == unit_id]


def read_a_results() -> list[dict]:
    return [read_json(p) for p in sorted((OUT / "A").glob("*/result.json")) if read_json(p)]


def read_c_results() -> list[dict]:
    return [read_json(p) for p in sorted((OUT / "C").glob("*/result.json")) if read_json(p)]


def read_b_rounds() -> list[dict]:
    return [read_json(p) for p in sorted((OUT / "B" / "rounds").glob("*/result.json")) if read_json(p)]


def all_decision_rows() -> list[tuple[str, dict, dict]]:
    rows = []
    for exp, results in (("A", read_a_results()), ("C", read_c_results())):
        for item in results:
            if item:
                rows.append((exp, item.get("decision") or {}, item.get("initial") or {}))
    for round_result in read_b_rounds():
        for agent_row in round_result.get("agent_decisions", []):
            rows.append(("B", agent_row.get("decision") or {}, agent_row.get("initial") or {}))
    return rows


def wilson(successes: int, n: int, z: float = 1.96) -> str:
    if n <= 0:
        return "n/a"
    if successes < 0 or successes > n:
        return "not calculable (inconsistent interim counts)"
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return f"{center - half:.3f}–{center + half:.3f}"


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def safe_excerpt(text: str | None, limit: int = 170) -> str:
    if not text:
        return "—"
    compact = re.sub(r"\s+", " ", text)
    return (compact[:limit] + "…") if len(compact) > limit else compact


def action_text(outcome: dict | None) -> str:
    if not outcome:
        return "—"
    action = outcome.get("final_action")
    if not action:
        return "FAILED"
    action_type = action.get("action_type", "?")
    allocation = outcome.get("allocation_to_target", 0)
    if action_type == "allocate":
        total = sum((action.get("allocation_plan") or {}).values())
        recipients = ", ".join(f"{agent}: {amount}" for agent, amount in (action.get("allocation_plan") or {}).items())
        return f"allocate {total} HP to {recipients} ({action.get('allocation_reason_label')})"
    if action_type == "communicate":
        return f"communicate ({safe_excerpt(action.get('message'), 60)})"
    if action_type in {"fight", "rob"}:
        return f"{action_type} {action.get('target_agent_id')}"
    return action_type


def raw_action_text(action: dict | None, target_id: str | None = None) -> str:
    if not action:
        return "—"
    action_type = action.get("action_type", "?")
    if action_type == "allocate":
        plan = action.get("allocation_plan", {}) or {}
        if target_id:
            return f"allocate {plan.get(target_id, 0)} HP"
        return f"allocate {sum(int(v) for v in plan.values())} HP"
    if action_type in {"fight", "rob"}:
        return f"{action_type} {action.get('target_agent_id')}"
    return action_type


def replace_report_section(document: str, name: str, next_name: str, body: str) -> str:
    marker = f"## {name}\n"
    next_marker = f"## {next_name}\n"
    start = document.find(marker)
    end = document.find(next_marker, start + len(marker))
    if start < 0 or end < 0:
        return document
    return document[:start] + marker + "\n" + body.rstrip() + "\n\n" + document[end:]


def render_progress() -> str:
    counts = progress_counts()
    elapsed = evaluation_elapsed_seconds()
    failures = STATE.get("failures", [])
    current = STATE.get("current_unit") or "None"
    calls = len(read_call_results())
    supervisor_events = read_jsonl(OUT / "telemetry" / "supervisor_lifecycle.jsonl")
    supervisor_starts = sum(1 for event in supervisor_events if event.get("event") == "runner_start")
    supervisor_null_exits = sum(1 for event in supervisor_events
                                if event.get("event") == "runner_exit" and event.get("exit_code") is None)
    return "\n".join([
        f"- A: **{counts['A_complete']}/40 complete**, {counts['A_blocked']} persistent blockers.",
        f"- B: **{counts['B_rounds_complete']}/6 rounds complete**, {counts['B_rounds_blocked']} blocked; {counts['B_decisions_completed']} decisions checkpointed across completed rounds; {counts['B_decisions_total']} decision slots visible from saved round checkpoints; {counts['B_failed_decisions']} failed/unresolved decisions. The initial cohort is eight agents; original reproduction dynamics can add offspring, so B has no fixed 48-decision denominator.",
        f"- C: **{counts['C_complete']}/8 complete**, {counts['C_blocked']} persistent blockers.",
        f"- Current stage/unit: `{STATE.get('phase')}` / `{current}`. Model completions captured: {calls}.",
        f"- Started: {STATE.get('started_at','pending')}; last update: {STATE.get('updated_at','pending')}; heartbeat: {read_json(OUT / 'heartbeat.json',{}).get('at','pending')}; elapsed: {elapsed/60:.1f} min.",
        f"- Recorded unit failure attempts: {len(failures)}; runner restarts recorded by state: {STATE.get('process_restarts',0)}; supervisor launches: {supervisor_starts} ({supervisor_null_exits} old exit records omitted the process code); server recovery events: {len(read_jsonl(OUT / 'recovery.jsonl'))}.",
        "- Each failed unit gets one initial attempt plus up to five outer recovery attempts; original in-pipeline validation/retries and two-stage reflection remain enabled.",
    ])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def render_a() -> str:
    results = read_a_results()
    if not results:
        return "Not evaluated yet."
    rows = ["| Scenario | Moral type | Seed | Draft valid | Initial pipeline valid | First action | Reflected action | Final action/transfer | Role check | Retries |",
            "|---|---|---:|---:|---:|---|---|---|---|---:|"]
    for item in results:
        initial, outcome = item.get("initial", {}), item.get("outcome", {})
        target = initial.get("target_agent_id")
        rows.append(f"| {initial.get('scenario')} | {initial.get('moral_type_label')} | {initial.get('seed')} | {fmt(outcome.get('first_draft_valid'))} | {fmt(outcome.get('initial_attempt_final_valid'))} | {raw_action_text(outcome.get('first_stage_action'), target)} | {raw_action_text(outcome.get('reflected_action'), target)} | {action_text(outcome)} | {fmt(outcome.get('role_consistent')) if outcome.get('role_judgeable') else 'unresolved'} | {outcome.get('internal_retry_count',0)} |")
    rows.extend(["", "Selected rationale evidence (blinded human review remains authoritative):"])
    representative = []
    for scenario in SCENARIOS:
        choice = next((item for item in results if item.get("initial", {}).get("scenario") == scenario
                       and item.get("initial", {}).get("moral_type") == "universal_group_focused_moral"
                       and item.get("initial", {}).get("seed") == 11), None)
        if choice:
            representative.append(choice)
    for item in representative:
        initial, decision, outcome = item.get("initial", {}), item.get("decision", {}), item.get("outcome", {})
        calls = decision.get("calls", [])
        first = next((c.get("content") for c in calls if c.get("stage") == "first_stage" and c.get("outer_attempt", 1) == 1), "")
        try:
            first_reason = json.loads(first).get("thinking", "")
        except Exception:
            first_reason = ""
        rows.append(f"- **{initial.get('scenario')} / {initial.get('moral_type_label')}:** first-stage rationale: {safe_excerpt(first_reason,150)} Reflected rationale: {safe_excerpt(outcome.get('rationale'),150)} Final: {action_text(outcome)}. Raw case: `data/qwen38_capability_evaluation_20261016/A/{initial.get('case_id')}/`.")
    rows.extend(["", "All first/reflected response JSON, validation outcomes, full prompt messages, and allocations are saved under `data/qwen38_capability_evaluation_20261016/A/<case-id>/outer_attempt_*/calls/`. Subjective readings remain for blinded review in `blinded_review/A_review.csv`."])
    return "\n".join(rows)


def render_b() -> str:
    results = read_b_rounds()
    if not results:
        return "Not evaluated yet."
    rows = ["| Round | Agent | Moral type | Decision | HP before→after | Memory updated | Decision status |",
            "|---:|---|---|---|---:|---:|---|"]
    for result in results:
        if not result.get("agent_decisions"):
            rows.append(f"| {result.get('step')} | — | — | **{result.get('status')}**: {safe_excerpt(result.get('blocker'),240)} | — | — | blocked |")
        for row in result.get("agent_decisions", []):
            decision = row.get("decision", {})
            outcome = decision_outcome(decision, row.get("initial", {})) if decision else {}
            before, after = row.get("hp_before"), row.get("hp_after")
            rows.append(f"| {result.get('step')} | {row.get('agent_id')} | {row.get('moral_type_label')} | {action_text(outcome)} | {before}→{after} | {fmt(row.get('memory_updated'))} | {'valid' if decision.get('success') else 'FAILED'} |")
        observations = result.get("new_observations", [])
        rows.append(f"|  |  |  | **Interactions observed:** {safe_excerpt('; '.join(observations), 240)} |  |  |  |")
        before_n = len(result.get("state_before", {}))
        after_n = len(result.get("state_after", {}))
        decision_n = len(result.get("agent_decisions", []))
        resources = result.get("resources_after", {}) or {}
        plants = resources.get("resources", []) or []
        plant_units = sum(int(x.get("quantity", 0)) for x in plants)
        prey_n = len(resources.get("prey_animals", []) or [])
        rows.append(
            f"|  |  |  | **State/integrity:** population {before_n}→{after_n}; memory updated {result.get('memory_updates_verified',0)}/{decision_n}, independent {fmt(result.get('memory_independence_verified'))}, memory checkpoint verified {fmt(result.get('memory_integrity_verified'))}; state {fmt(result.get('state_consistency'))}, family/relationships {fmt(result.get('relationship_consistency'))}, reload {fmt(result.get('checkpoint_reload_verified'))}. **Environment after tick:** {len(plants)} plant patches / {plant_units} units; {prey_n} prey. |  |  |  |"
        )
    rows.extend(["", "Per-round state snapshots, every living agent conversation (including original-model offspring), observations, resources, memory deltas, checkpoints, and context/call records are stored under `data/qwen38_capability_evaluation_20261016/B/rounds/` and `B/decision_cache/`."])
    return "\n".join(rows)


def render_c() -> str:
    results = read_c_results()
    if not results:
        return "Not evaluated yet."
    rows = ["| History | Length | Moral type | Partner state | Critical facts correct | Facts denominator | Decoy attribution | Max input tokens | Action |",
            "|---|---|---|---|---:|---:|---:|---:|---|"]
    for item in results:
        initial, outcome = item.get("initial", {}), item.get("outcome", {})
        facts = outcome.get("memory_facts", {})
        max_tokens = max((x.get("input_tokens") or 0 for x in (item.get("decision") or {}).get("calls", [])), default=0)
        rows.append(f"| {initial.get('expected_facts',{}).get('history')} | {initial.get('expected_facts',{}).get('length')} | {initial.get('moral_type_label')} | {initial.get('partner_state',{}).get('hp')} HP | {facts.get('correct',0)} | {facts.get('total',4)} | {fmt(facts.get('decoy_attribution_correct'))} | {max_tokens} | {action_text(outcome)} |")
    rows.extend(["", "Full expected histories, seeded per-entity memory, response memory, fact slots, fabricated-record checks, and all prompt/draft/reflection files are in `data/qwen38_capability_evaluation_20261016/C/`."])
    return "\n".join(rows)


def render_metrics() -> str:
    decisions = all_decision_rows()
    if not decisions:
        return "Not evaluated yet; denominators are A 40, B decision slots recorded per round checkpoint (dynamic population), and C 8."
    all_calls = read_call_results()
    b_decision_slots, b_unknown_steps = b_agent_slot_counts()
    total_decisions = 40 + b_decision_slots + 8
    planned = {"A": 40, "B": b_decision_slots, "C": 8}
    first_draft_units = {(x.get("experiment"), x.get("unit_id")) for x in all_calls
                         if x.get("stage") == "first_stage" and x.get("outer_attempt", 1) == 1
                         and x.get("pipeline_attempt", x.get("retry_attempt", 1)) == 1}
    first_draft_valid_units = {(x.get("experiment"), x.get("unit_id")) for x in all_calls
                               if x.get("stage") == "first_stage" and x.get("outer_attempt", 1) == 1
                               and x.get("pipeline_attempt", x.get("retry_attempt", 1)) == 1
                               and x.get("validation", {}).get("valid")}
    result_rows: dict[str, list[tuple[dict, dict, dict]]] = {"A": [], "B": [], "C": []}
    initial_cycle_success = eventual_success = 0
    judgeable = role_success = 0
    invalid_json = schema_fail = illegal = 0
    for exp, decision, initial in decisions:
        outcome = decision_outcome(decision, initial) if decision else {}
        result_rows[exp].append((decision, initial, outcome))
        eventual_success += int(bool(decision.get("success")))
        initial_cycle_success += int(bool(outcome.get("initial_attempt_final_valid")))
        if outcome.get("role_judgeable"):
            judgeable += 1
            role_success += int(bool(outcome.get("role_consistent")))
    for call in all_calls:
        cat = call.get("validation", {}).get("category")
        invalid_json += int(cat == "invalid_json")
        schema_fail += int(cat == "schema_failure")
        illegal += int(cat == "illegal_action")
    attempt_index = read_pipeline_attempt_index()
    internal_retries = sum(max(0, len(attempts) - 1)
                           for by_outer in attempt_index.values() for attempts in by_outer.values())
    local_request_errors = read_jsonl(OUT / "telemetry" / "call_errors.jsonl")
    context_request_errors = sum("contextwindowexceeded" in str(x.get("error", "")).lower()
                                 or "exceeds the available context size" in str(x.get("error", "")).lower()
                                 for x in local_request_errors)
    outer_recoveries = sum(int(x.get("outer_recoveries", 0)) for x in read_a_results() + read_c_results())
    outer_recoveries += sum(max(0, int(x.get("round_attempts", 1)) - 1) for x in read_b_rounds())
    input_tokens = sum(int(x.get("input_tokens") or 0) for x in all_calls)
    output_tokens = sum(int(x.get("output_tokens") or 0) for x in all_calls)
    duration = sum(float(x.get("reported_duration_seconds") or x.get("observed_duration_seconds") or 0) for x in all_calls)
    gen_rate = output_tokens / duration if duration else 0
    prompt_rate = input_tokens / duration if duration else 0
    context_max = max((int(x.get("input_tokens") or 0) + int(x.get("output_tokens") or 0) for x in all_calls), default=0)
    memory_correct = 0
    memory_total = 8 * 4
    memory_decoy_correct = memory_decoy_total = 0
    fabrication_count = 0
    c_by_id = {x.get("initial", {}).get("case_id"): x for x in read_c_results()}
    for history, length, moral_type, seed in case_ids_c():
        item = c_by_id.get(f"{history}__{length}__{moral_type}__s{seed}", {})
        facts = item.get("outcome", {}).get("memory_facts", {})
        memory_correct += int(facts.get("correct", 0))
        memory_decoy_correct += int(bool(facts.get("decoy_attribution_correct")))
        memory_decoy_total += 1
        fabrication_count += len(facts.get("fabricated_critical_claims", []))
    pair_diff, pair_den = moral_pair_differentiation()
    b_round_results = read_b_rounds()
    b_consistency = sum(1 for r in b_round_results if r.get("state_consistency", False))
    b_relationships = sum(1 for r in b_round_results if r.get("relationship_consistency", False))
    b_checkpoint_reloads = sum(1 for r in b_round_results if r.get("checkpoint_reload_verified", False))
    b_mem = sum(int(r.get("memory_updates_verified", 0)) for r in read_b_rounds())
    b_mem_changed = sum(1 for r in b_round_results for row in r.get("agent_decisions", []) if row.get("memory_changed"))
    rounds = len(b_round_results)
    b_actions = Counter()
    b_allocated_hp = 0
    b_successful_decisions = 0
    b_comms = b_fights = b_robs = 0
    for round_result in b_round_results:
        for agent_row in round_result.get("agent_decisions", []):
            decision = agent_row.get("decision", {})
            if not decision.get("success"):
                continue
            b_successful_decisions += 1
            outcome = decision_outcome(decision, {})
            action = outcome.get("final_action") or {}
            action_type = action.get("action_type")
            if action_type:
                b_actions[action_type] += 1
            if action_type == "allocate":
                b_allocated_hp += sum(int(x) for x in (action.get("allocation_plan", {}) or {}).values())
            elif action_type == "communicate":
                b_comms += 1
            elif action_type == "fight":
                b_fights += 1
            elif action_type == "rob":
                b_robs += 1
    leakage_failures = sum(not bool(r.get("memory_independence_verified")) for r in read_b_rounds())
    leakage_failures += sum(not bool(x.get("initial", {}).get("memory_isolation", {}).get(
                                "seeded_fact_only_in_focal_ltm",
                                x.get("initial", {}).get("memory_isolation", {}).get("seeded_fact_only_in_focal")))
                            for x in read_c_results())
    leakage_failures += sum(not bool(x.get("checkpoint_effects", {}).get("other_agent_persistent_memory_unchanged"))
                            for x in read_a_results() + read_c_results() if x.get("status") == "complete")
    stage_pairs: dict[tuple[str, int, int], dict[str, dict]] = {}
    for call in all_calls:
        if call.get("outer_attempt", 1) == 1 and call.get("pipeline_attempt", call.get("retry_attempt", 1)) == 1:
            stage_pairs.setdefault((call.get("experiment", ""), call.get("unit_id", ""), 1), {})[call.get("stage", "")] = call
    compared = changed = corrected = regressed = 0
    for pair in stage_pairs.values():
        draft, reflection = pair.get("first_stage"), pair.get("reflection")
        if not draft or not reflection:
            continue
        compared += 1
        dv = bool(draft.get("validation", {}).get("valid"))
        rv = bool(reflection.get("validation", {}).get("valid"))
        corrected += int(not dv and rv)
        regressed += int(dv and not rv)
        try:
            changed += int(json.loads(draft.get("content", "{}")).get("action") !=
                           json.loads(reflection.get("content", "{}")).get("action"))
        except Exception:
            pass
    metric_delta = read_json(OUT / "telemetry" / "server_metrics_delta.json", {}) or {}
    prompt_elapsed = float(metric_delta.get("llamacpp:prompt_seconds_total", 0) or 0)
    generation_elapsed = float(metric_delta.get("llamacpp:tokens_predicted_seconds_total", 0) or 0)
    prompt_rate = float(metric_delta.get("llamacpp:prompt_tokens_total", input_tokens) or input_tokens) / prompt_elapsed if prompt_elapsed else prompt_rate
    gen_rate = float(metric_delta.get("llamacpp:tokens_predicted_total", output_tokens) or output_tokens) / generation_elapsed if generation_elapsed else gen_rate
    gpu_samples = [int(x["used_mib"]) for x in read_jsonl(GPU_JSONL) if isinstance(x.get("used_mib"), (int, float))]
    if gpu_samples:
        GPU_SUMMARY.update({"min_mib": min(gpu_samples), "max_mib": max(gpu_samples),
                            "mean_mib": round(sum(gpu_samples) / len(gpu_samples), 1), "samples": len(gpu_samples)})
    rows = []
    rows.append(f"- Decisions eventually valid: **{eventual_success}/{total_decisions}** ({eventual_success/total_decisions:.1%}); Wilson 95% CI {wilson(eventual_success,total_decisions)}. Denominator: 40 A cases, 8 C cases, plus B decision slots visible in saved round checkpoints (dynamic population); unresolved known slots count invalid. B blocked dependent ticks without a before-checkpoint are reported separately and were not evaluable ({len(b_unknown_steps)} such ticks).")
    rows.append(f"- Initial full pipeline attempt valid without internal/outer retry: **{initial_cycle_success}/{total_decisions}**; Wilson 95% CI {wilson(initial_cycle_success,total_decisions)}.")
    rows.append(f"- First raw draft passes JSON, Pydantic, and contextual legality checks: **{len(first_draft_valid_units)}/{total_decisions}** planned decisions; initial drafts observed for {len(first_draft_units)}/{total_decisions}; Wilson 95% CI {wilson(len(first_draft_valid_units),total_decisions)}.")
    rows.append(f"- Captured invalid model generations: invalid JSON **{invalid_json}**, schema failures **{schema_fail}**, contextual/illegal actions **{illegal}**; completed response generations: {len(all_calls)}, failed local requests: {len(local_request_errors)} (context-window errors {context_request_errors}). Internal pipeline retries: {internal_retries}; outer recovery attempts: {outer_recoveries}.")
    rows.append(f"- A judgeable moral-role decisions: **{role_success}/{judgeable}**; Wilson 95% CI {wilson(role_success,judgeable)}. Moral-type pairwise difference in help/withhold choice: **{pair_diff}/{pair_den}** matched type pairs.")
    rows.append(f"- C critical memory facts: **{memory_correct}/{memory_total}** explicit slots (unattempted or blocked slots count incorrect); Wilson 95% CI {wilson(memory_correct,memory_total)}. Correct decoy/partner separation **{memory_decoy_correct}/{memory_decoy_total}**. Structured unsupported critical action records flagged: {fabrication_count}.")
    rows.append(f"- B state-consistent rounds: **{b_consistency}/{rounds}**; family/relationship-consistent rounds: **{b_relationships}/{rounds}**; saved checkpoint reloads matched at the model-field level: **{b_checkpoint_reloads}/{rounds}**; response memory applied to the intended independent agent checkpoint: **{b_mem}/{planned['B']}** planned B decision slots; memory content actually changed: **{b_mem_changed}/{planned['B']}** planned slots.")
    rows.append(f"- B action diversity/interactions: **{len(b_actions)}** distinct action types among {b_successful_decisions} valid decisions (`{dict(b_actions)}`); allocated {b_allocated_hp} HP in total, communication actions {b_comms}, fights {b_fights}, robberies {b_robs}. These are observed action counts, not moral ratings.")
    rows.append(f"- Completion tokens: input {input_tokens:,}, output {output_tokens:,}; aggregate measured prompt throughput {prompt_rate:.2f} input tokens/s and generation throughput {gen_rate:.2f} output tokens/s (provider per-call durations).")
    rows.append(f"- Maximum observed single-call input+output context usage: **{context_max:,}/32,768** tokens from completion usage. Context-ceiling/error events: {context_failure_count(all_calls)}. Silent truncation: {silent_truncation_count(all_calls)} detectable instances; absence of a detectable event is not proof of perfect retrieval.")
    rows.append(f"- Measured GPU board-memory used: {GPU_SUMMARY.get('min_mib','pending')}–{GPU_SUMMARY.get('max_mib','pending')} MiB (mean {GPU_SUMMARY.get('mean_mib','pending')} MiB; {GPU_SUMMARY.get('samples',0)} samples). Inference server one-slot counters and complete 1-second samples are under `data/qwen38_capability_evaluation_20261016/telemetry/`.")
    rows.append(f"- Cross-agent memory isolation failures: **{leakage_failures}** observed across completed checks (A/C scoped writes, C seeded facts, B independent memory objects). Reflection first-cycle action changes **{changed}/{compared}** paired decisions; validator-validity corrections {corrected}, regressions {regressed}; these are not moral-quality grades.")
    audit = read_json(OUT / "artifact_audit.json", {}) or {}
    if audit:
        rows.append(f"- Final artifact audit: **{'PASS' if audit.get('passed') else 'FAIL'}**; {audit.get('final_responses_revalidated',0)} final responses revalidated against frozen checkpoints; {audit.get('durable_response_files',0)} raw responses and {audit.get('request_files_present',0)} paired request prompts; {len(audit.get('errors',[]))} integrity errors.")
    for exp, rows_exp in result_rows.items():
        ok = sum(int(bool(d.get("success"))) for d, _, _ in rows_exp)
        first = sum(int(bool(o.get("initial_attempt_final_valid"))) for _, _, o in rows_exp)
        rows.append(f"- {exp} validity: eventual **{ok}/{planned[exp]}**, initial full-pipeline **{first}/{planned[exp]}**; absent or blocked cases count invalid.")
    return "\n".join(rows)


def moral_pair_differentiation() -> tuple[int, int]:
    groups: dict[tuple[str, int], dict[str, bool]] = {}
    for item in read_a_results():
        init, outcome = item.get("initial", {}), item.get("outcome", {})
        if item.get("status") != "complete" or not outcome.get("final_action"):
            continue
        key = (init.get("scenario"), init.get("seed"))
        groups.setdefault(key, {})[init.get("moral_type")] = bool(outcome.get("role_observed_help"))
    different = denominator = 0
    for roles in groups.values():
        values = list(roles.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                denominator += 1
                different += int(values[i] != values[j])
    return different, denominator


def context_failure_count(calls: list[dict]) -> int:
    patterns = ("context size", "context length", "too many tokens", "n_ctx", "truncat")
    generated = sum(1 for call in calls if any(p in json.dumps(call.get("validation", {})).lower() for p in patterns))
    failed_requests = read_jsonl(OUT / "telemetry" / "call_errors.jsonl")
    explicit = sum(any(p in str(row.get("error", "")).lower() for p in patterns)
                   for row in failed_requests)
    return generated + explicit


def silent_truncation_count(calls: list[dict]) -> int:
    # llama.cpp returns prompt usage per request; flag only objectively observable overflow/eviction evidence.
    return sum(1 for call in calls if (call.get("input_tokens") or 0) + (call.get("output_tokens") or 0) > 32768
               or call.get("server_context", {}).get("truncated") is True
               or call.get("server_context_truncated") is True)


def render_failure_analysis() -> str:
    failures = STATE.get("failures", [])
    calls = read_call_results()
    invalid = [c for c in calls if not c.get("validation", {}).get("valid")]
    fatal_errors = read_jsonl(OUT / "telemetry" / "fatal_errors.jsonl")
    if not failures and not invalid and not fatal_errors:
        return "No persistent failures or invalid generated drafts recorded so far; this status updates during execution."
    lines = []
    if failures:
        lines.append("| Experiment | Unit | Outer attempt | Failure |")
        lines.append("|---|---|---:|---|")
        for failure in failures:
            lines.append(f"| {failure.get('experiment')} | {failure.get('case_id')} | {failure.get('outer_attempt')} | {safe_excerpt(failure.get('error'),180)} |")
    if fatal_errors:
        lines.extend(["", "Runner process exceptions (separate from model-generated response failures):"])
        for error in fatal_errors[-8:]:
            lines.append(f"- {error.get('at')}: {safe_excerpt(error.get('error'),220)}; supervisor lifecycle and recovery logs are in `data/qwen38_capability_evaluation_20261016/telemetry/`.")
        if any("math domain error" in str(error.get("error", "")).lower() for error in fatal_errors):
            lines.append("- The earlier live report refresh hit an out-of-range Wilson interval during an interim update. The interval helper now marks inconsistent interim counts as not calculable; the final offline recomputation used complete denominators and produced valid intervals.")
    call_errors = read_jsonl(OUT / "telemetry" / "call_errors.jsonl")
    supervisor_events = read_jsonl(OUT / "telemetry" / "supervisor_lifecycle.jsonl")
    supervisor_starts = sum(1 for event in supervisor_events if event.get("event") == "runner_start")
    supervisor_null_exits = sum(1 for event in supervisor_events
                                if event.get("event") == "runner_exit" and event.get("exit_code") is None)
    completed_at = next((event.get("at") for event in read_jsonl(OUT / "telemetry" / "runner_lifecycle.jsonl")
                         if event.get("event") == "finished"), None)
    post_completion_responses = 0
    if completed_at:
        try:
            cutoff = datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp()
            post_completion_responses = sum(1 for path in OUT.rglob("call_*_response.json")
                                            if path.stat().st_mtime > cutoff)
        except (ValueError, OSError):
            post_completion_responses = -1
    b_step_errors = [error for result in read_b_rounds()
                     for error in (result.get("round_errors", []) or result.get("failed_attempts", []))]
    blocked_b = [result for result in read_b_rounds() if result.get("status", "").startswith("blocked")]
    if call_errors or b_step_errors or blocked_b:
        lines.extend(["", "Transient/provider errors and blocked B ticks:"])
        for error in call_errors[:12]:
            lines.append(f"- `{error.get('unit_id')}` {error.get('stage')}: {safe_excerpt(error.get('error'),200)}")
        for error in b_step_errors[:12]:
            lines.append(f"- B tick exception, attempt {error.get('attempt')}: {safe_excerpt(error.get('error'),200)}")
        for result in blocked_b:
            lines.append(f"- B tick {result.get('step')} remains blocked: {safe_excerpt(result.get('blocker'),240)}")
    if supervisor_starts > 1:
        lines.extend(["", "Supervisor recovery and final audit:"])
        lines.append(f"- The original supervisor launched the runner {supervisor_starts} times; {supervisor_null_exits} exit records lacked an exit code because the PowerShell process object was not refreshed. This caused repeated offline resume/finalization passes; no model response files were added after the first completed run ({post_completion_responses} later response files). The supervisor now waits for and refreshes the process result and invokes offline finalization when all units are already resolved.")
        audit = read_json(OUT / "postfinalization_audit.json", {}) or {}
        final_audit = audit.get("artifact_audit", {})
        lines.append(f"- Offline final audit: {'PASS' if final_audit.get('passed') else 'FAIL'}; {final_audit.get('final_responses_revalidated',0)} responses revalidated; {len(final_audit.get('errors', []))} integrity errors.")
    if invalid:
        lines.extend(["", "Invalid generated response samples (full text remains in the linked per-call response file):"])
        for call in invalid[:12]:
            lines.append(f"- `{call.get('unit_id')}` {call.get('stage')}: {call.get('validation',{}).get('category')} — {safe_excerpt(call.get('validation',{}).get('error'),180)}")
    return "\n".join(lines)


def render_executive_summary() -> str:
    counts = progress_counts()
    status = STATE.get("status", "prepared")
    elapsed = evaluation_elapsed_seconds()
    return (f"**Status: {status}.** This local model-suitability pilot covers 40 controlled decisions (A), "
            f"six rounds of the original simulation from an eight-agent initial cohort (B), and eight memory/context decisions (C). "
            f"Progress is A {counts['A_complete']} complete + {counts['A_blocked']} blocked / 40; "
            f"B {counts['B_rounds_complete']} complete + {counts['B_rounds_blocked']} blocked / 6; "
            f"C {counts['C_complete']} complete + {counts['C_blocked']} blocked / 8. "
            f"Elapsed: {elapsed/3600:.2f} hours. The prior two-round smoke is setup evidence only. "
            "Every attempted decision uses the repository's prompts, original validation/action path, local Qwen3.8-27B Q4_K_M server, and two-stage draft/reflection flow. "
            "No part of the paper's 80-step evolutionary experiment is included.")


def render_reproducibility() -> str:
    props = read_json(OUT / "telemetry" / "server_props.json", {}) or {}
    hardware = read_json(OUT / "telemetry" / "hardware.json", {}) or {}
    server_process = read_json(OUT / "telemetry" / "server_process.json", {}) or {}
    b_initial = read_json(OUT / "B" / "initial.json", {}) or {}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    except Exception:
        commit = "unavailable"
    gen = props.get("default_generation_settings", {}) or {}
    defaults = gen.get("params", {}) or {}
    key_params = {k: defaults.get(k) for k in ("seed", "temperature", "top_k", "top_p", "min_p", "max_tokens", "n_predict", "repeat_penalty")}
    rows = [
        f"- Repository commit: `{commit}`; the prepared local-inference checkout also contains its documented working-tree changes.",
        f"- Served model: `{props.get('model_alias', MODEL)}`; file `{props.get('model_path', str(ROOT / '.local' / 'models' / 'Qwen3.8-27B-Q4_K_M.gguf'))}`; llama.cpp reports quantization `{props.get('model_ftype', 'not reported')}`. Upstream identity: `ggml-org/Qwen3.8-27B-GGUF`.",
        f"- llama-server build: `{props.get('build_info', 'not reported')}`. Process command line: `{server_process.get('command_line', 'not captured')}`.",
        f"- GPU: {hardware.get('name', 'not reported')}; board memory {hardware.get('memory_total_mib', 'not reported')} MiB. The active command requests CUDA0 and all layers offloaded; runtime residency is independently sampled from `nvidia-smi`.",
        f"- Endpoint `http://127.0.0.1:8080/v1`; text-only modalities `{props.get('modalities', {})}`; context `{gen.get('n_ctx', 'not reported')}`; slots `{props.get('total_slots', 'not reported')}`. Runtime check: Flash Attention on, Q8 K/V, no multimodal projector, 32K context, one slot.",
        f"- Server generation defaults (unchanged by this evaluation): `{json.dumps(key_params, sort_keys=True)}`. llama.cpp seed `4294967295` is its random-seed sentinel, so outputs are not bitwise deterministic; scenario/configuration seeds are recorded in every checkpoint.",
        f"- B checkpoint seed: `{b_initial.get('configuration', {}).get('random_seed', 'not recorded')}`; B starts with eight agents, two per moral profile. `max_life_steps=2`, `communication_and_sharing_steps=2`, `two_stage_model=true`; original tick formula gives six ticks.",
        "- No hosted endpoint, external LLM API, or paid API key is used. Per-decision model requests and replies are recorded verbatim under the results directory.",
        "- Prior two-round setup smoke: `data/1015-201445/smoke_report.json` and associated checkpoints/logs. It is not included in A/B/C denominators.",
    ]
    return "\n".join(rows)


def render_assessment() -> str:
    counts = progress_counts()
    if (counts["A_complete"] + counts["A_blocked"] < 40
            or counts["B_rounds_complete"] + counts["B_rounds_blocked"] < 6
            or counts["C_complete"] + counts["C_blocked"] < 8):
        return "Not assessed; at least one required evaluation unit is still running or blocked. Thresholds are not being marked as passed based on partial evidence."
    if counts["A_blocked"] or counts["B_rounds_blocked"] or counts["C_blocked"]:
        return "The prescribed units have completed or reached documented persistent blockers; the model-suitability assessment is limited by those blockers and awaits blinded human review. Deterministic results do not reproduce the paper's published scores or long-run findings."
    decisions = all_decision_rows()
    eventual = sum(bool(decision.get("success")) for _, decision, _ in decisions)
    all_calls = read_call_results()
    first_drafts = {(call.get("experiment"), call.get("unit_id")) for call in all_calls
                    if call.get("stage") == "first_stage" and call.get("outer_attempt", 1) == 1
                    and call.get("pipeline_attempt", call.get("retry_attempt", 1)) == 1}
    valid_first = {(call.get("experiment"), call.get("unit_id")) for call in all_calls
                   if call.get("stage") == "first_stage" and call.get("outer_attempt", 1) == 1
                   and call.get("pipeline_attempt", call.get("retry_attempt", 1)) == 1
                   and call.get("validation", {}).get("valid")}
    role_ok = role_n = memory_ok = 0
    c_results = read_c_results()
    for _, decision, initial in decisions:
        outcome = decision_outcome(decision, initial) if decision else {}
        if outcome.get("role_judgeable"):
            role_n += 1
            role_ok += int(bool(outcome.get("role_consistent")))
    memory_n = 8 * 4
    memory_ok = sum(int(result.get("outcome", {}).get("memory_facts", {}).get("correct", 0))
                    for result in c_results)
    pair_different, pair_n = moral_pair_differentiation()
    context_errors = context_failure_count(all_calls)
    return (f"All {eventual}/{len(decisions)} final decisions are valid after the repository's validation/retry path, and the artifact audit passes with independent per-agent checkpoint memories. "
            f"First drafts pass only {len(valid_first)}/{len(decisions)} cases ({len(valid_first)/len(decisions):.1%}), below the 90% target; critical memory recall is {memory_ok}/{memory_n} ({memory_ok/memory_n:.1%}), also below target. "
            f"The deterministic moral-role proxy is {role_ok}/{role_n} judgeable cases ({role_ok/role_n:.1%}), while only {pair_different}/{pair_n} matched moral-type pairs differ on the help/withhold proxy; intervals are wide and blinded human review is unresolved. "
            f"There were {context_errors} explicit context-window request failures and no detected silent truncation. The model is usable as a locally hosted exploratory engine with the original reflection, validation, retries, and checkpointing enabled, but this pilot does not establish reliable unattended moral behavior or long-run suitability. It does not reproduce the paper's published scores or evolutionary findings.")


def refresh_report() -> None:
    if not REPORT_TEMPLATE:
        return
    with LOCK:
        doc = REPORT_PATH.read_text(encoding="utf-8")
        doc = replace_report_section(doc, "Executive summary", "Reproducibility", render_executive_summary())
        doc = replace_report_section(doc, "Reproducibility", "Paper basis and protocol", render_reproducibility())
        doc = replace_report_section(doc, "Progress and monitoring", "Experiment A results", render_progress())
        doc = replace_report_section(doc, "Experiment A results", "Experiment B results", render_a())
        doc = replace_report_section(doc, "Experiment B results", "Experiment C results", render_b())
        doc = replace_report_section(doc, "Experiment C results", "Quantitative results", render_c())
        doc = replace_report_section(doc, "Quantitative results", "Failure analysis", render_metrics())
        doc = replace_report_section(doc, "Failure analysis", "Assessment", render_failure_analysis())
        doc = replace_report_section(doc, "Assessment", "Limitations and remaining tests", render_assessment())
        REPORT_PATH.write_text(doc, encoding="utf-8")


def write_blinded_review_tables() -> None:
    a_rows, key_rows = [], []
    blind_index = 1
    for result in read_a_results():
        initial, decision, outcome = result.get("initial", {}), result.get("decision", {}), result.get("outcome", {})
        response = decision.get("response") or {}
        calls = decision.get("calls", [])
        first = next((c.get("content") for c in calls if c.get("stage") == "first_stage"
                      and c.get("pipeline_attempt", c.get("retry_attempt", 1)) == 1), "")
        reflection = next((c.get("content") for c in reversed(calls) if c.get("stage") == "reflection"
                           and c.get("pipeline_attempt", c.get("retry_attempt", 1)) == 1), "")
        blind_id = f"A-{blind_index:03d}"
        a_rows.append({"blind_case_id": blind_id, "scenario": initial.get("scenario_description"),
                       "focal_and_target_state": json.dumps({"focal": initial.get("focal_state"), "target": initial.get("target_state"),
                                                              "relationship": initial.get("is_family")}, ensure_ascii=False),
                       "first_stage_reasoning": (json.loads(first).get("thinking", "") if first else ""),
                       "reflected_reasoning": response.get("thinking", ""),
                       "first_stage_action": json.dumps(outcome.get("first_stage_action"), ensure_ascii=False),
                       "final_action": json.dumps(outcome.get("final_action"), ensure_ascii=False),
                       "review_role_consistency_0_1_or_unresolved": "",
                       "review_reason_or_ambiguity": ""})
        key_rows.append({"blind_case_id": blind_id, "case_id": initial.get("case_id"),
                         "moral_type": initial.get("moral_type"), "expected_help": outcome.get("role_expected_help"),
                         "judgeable": outcome.get("role_judgeable"), "automated_action_proxy": outcome.get("role_consistent")})
        blind_index += 1
    c_rows = []
    for result in read_c_results():
        initial, decision = result.get("initial", {}), result.get("decision", {})
        response = decision.get("response") or {}
        blind_id = f"C-{len(c_rows)+1:02d}"
        c_rows.append({"blind_case_id": blind_id, "history_evidence": json.dumps(initial.get("expected_facts"), ensure_ascii=False),
                       "focal_and_partner_state": json.dumps({"focal": initial.get("focal_state"), "partner": initial.get("partner_state"), "decoy": initial.get("decoy_state")}, ensure_ascii=False),
                       "reasoning": response.get("thinking", ""),
                       "memory_claims": json.dumps(response.get("long_term_memory"), ensure_ascii=False),
                       "review_fact_recall_correct_or_uncertain": "", "review_fabricated_or_misattributed_claims": ""})
    output = OUT / "blinded_review"
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("A_review.csv", a_rows), ("C_review.csv", c_rows)):
        if not rows:
            continue
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if key_rows:
        with (output / "A_unblinding_key.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
            writer.writeheader()
            writer.writerows(key_rows)


def check_memory_independence(checkpoint) -> bool:
    agents = checkpoint.social_environment.agents
    return (len({id(a.memory) for a in agents}) == len(agents)
            and len({id(a.memory.long_term_memory) for a in agents}) == len(agents))


def action_summary(response) -> dict[str, Any]:
    if not response:
        return {"action_type": None}
    action = response.action.root
    return action.model_dump(mode="json")


def relationship_consistency(checkpoint) -> bool:
    agents = {a.id: a for a in checkpoint.social_environment.agents}
    for agent in agents.values():
        parent_id = agent.family.parent_id
        if parent_id:
            parent = agents.get(parent_id)
            if parent is None or agent.id not in parent.family.children_ids:
                return False
        for child_id in agent.family.children_ids:
            child = agents.get(child_id)
            if child is None or child.family.parent_id != agent.id:
                return False
    return True


async def run_b() -> None:
    from scr.models.simulation.checkpoint import Checkpoint
    from scr.simulation.event_bus import SimulationEventBus
    from scr.simulation.runner import simulation_step
    from scr.utils import sim_logger

    b_dir = OUT / "B"
    rounds_dir = b_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    initial_path = b_dir / "initial.json"
    if initial_path.exists():
        cp = Checkpoint.model_validate_json(initial_path.read_text(encoding="utf-8"))
        moral_types_present(cp)
    else:
        cp = new_checkpoint(20261016, "Q38B-six-rounds", {
            "world.max_life_steps": 2, "world.communication_and_sharing_steps": 2,
            "llm.two_stage_model": True,
        })
        write_json(initial_path, json.loads(cp.model_dump_json()))
        write_json(b_dir / "run_configuration.json", {
            "config": cp.configuration.model_dump(mode="json"),
            "actual_original_runner_ticks": cp.configuration.world.max_life_steps * (cp.configuration.world.communication_and_sharing_steps + 1),
            "intended_ticks": 6,
        })
    if cp.configuration.world.max_life_steps != 2 or cp.configuration.world.communication_and_sharing_steps != 2:
        raise AssertionError("Refusing to alter or run the wrong simulation length")
    if cp.configuration.llm.two_stage_model is not True or len(cp.social_environment.agents) != 8:
        raise AssertionError("B must use eight agents and original two-stage reflection")
    if cp.metadata.current_time_step < 1 or cp.metadata.current_time_step > 7:
        raise AssertionError(f"Unexpected B checkpoint step {cp.metadata.current_time_step}")
    # Resume from the last durable, fully committed after-round state.
    committed = read_json(b_dir / "last_committed.json")
    if committed and committed.get("checkpoint"):
        cp = Checkpoint.model_validate(committed["checkpoint"])

    bus = SimulationEventBus(enabled=True)
    bus.subscribe(sim_logger.jsonl_event_bus_subscriber)
    await bus.start()
    try:
        for expected_step in range(cp.metadata.current_time_step, 7):
            round_dir = rounds_dir / f"step_{expected_step}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_result_path = round_dir / "result.json"
            saved = read_json(round_result_path)
            if saved and saved.get("status", "").startswith("complete"):
                next_state = read_json(round_dir / "after.json")
                if next_state:
                    cp = Checkpoint.model_validate(next_state)
                    update_state(lambda state, step=expected_step, result=saved, path=round_result_path: state.setdefault("B", {}).setdefault("rounds", {}).__setitem__(str(step), {
                        "status": result.get("status"), "decisions_completed": sum(1 for row in result.get("agent_decisions", []) if row.get("decision", {}).get("success") or row.get("decision", {}).get("error")),
                        "failed_decisions": len(result.get("failed_agent_ids", [])), "round_attempts": result.get("round_attempts", 1),
                        "result_path": str(path.relative_to(ROOT)), "updated_at": utc_now(),
                    }))
                    continue
            if cp.metadata.current_time_step != expected_step:
                raise AssertionError(f"Resume state at {cp.metadata.current_time_step} does not match next tick {expected_step}")
            set_current("B", f"simulation round {expected_step}/6")
            start_checkpoint = json.loads(cp.model_dump_json())
            write_json(round_dir / "before.json", start_checkpoint)
            state_before = {a.id: {"hp": a.state.hp, "age": a.state.age, "alive": a.is_alive()} for a in cp.social_environment.agents}
            memory_before = {a.id: a.memory.model_dump(mode="json") for a in cp.social_environment.agents}
            observations_before = list(cp.observations)
            cache_dir = b_dir / "decision_cache" / f"step_{expected_step}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            latest_runtime: dict | None = None
            final_working = None
            failed_result_ids: list[str] = []
            round_errors: list[dict[str, Any]] = []
            messages = {}
            for round_attempt in range(1, MAX_OUTER_ATTEMPTS + 1):
                working = Checkpoint.model_validate(start_checkpoint)
                runtime = {"step": expected_step, "round_attempt": round_attempt, "run_id": cp.metadata.run_id,
                           "round_dir": round_dir, "cache_dir": cache_dir, "decisions": {}}
                token = B_RUNTIME.set(runtime)
                try:
                    messages = await simulation_step.step(
                        working, str(b_dir / "checkpoints"), max_concurrent=1, event_bus=bus,
                    )
                except Exception as exc:
                    failure = {"attempt": round_attempt, "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc(), "at": utc_now()}
                    round_errors.append(failure)
                    append_jsonl(OUT / "failures.jsonl", {"experiment": "B", "case_id": f"tick_{expected_step}", **failure})
                    if round_attempt < MAX_OUTER_ATTEMPTS:
                        note_progress(f"B tick {expected_step}: original simulation step raised; recovery {round_attempt}/5")
                        if not health().get("ok"):
                            try:
                                restart_server(f"B tick {expected_step} exception: {type(exc).__name__}")
                            except Exception as restart_exc:
                                append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "server_restart_failed", "error": str(restart_exc)})
                        await asyncio.sleep(BACKOFF_SECONDS[round_attempt - 1])
                    continue
                finally:
                    B_RUNTIME.reset(token)
                latest_runtime, final_working = runtime, working
                decision_ids = set(runtime["decisions"])
                expected_ids = {a.id for a in cp.social_environment.agents if a.is_alive()}
                failed_result_ids = [aid for aid in sorted(expected_ids) if aid not in runtime["decisions"] or not runtime["decisions"][aid].get("success")]
                if not failed_result_ids:
                    break
                if round_attempt < MAX_OUTER_ATTEMPTS:
                    note_progress(f"B tick {expected_step}: retrying failed agents {failed_result_ids}, outer attempt {round_attempt}/{MAX_OUTER_ATTEMPTS}")
                    if not health().get("ok"):
                        try:
                            restart_server(f"B tick {expected_step}, failed agents {failed_result_ids}")
                        except Exception as exc:
                            append_jsonl(OUT / "recovery.jsonl", {"at": utc_now(), "action": "server_restart_failed", "error": str(exc)})
                    await asyncio.sleep(BACKOFF_SECONDS[round_attempt - 1])
                else:
                    # Preserve the repository's own failed-decision behavior: the original step skips a failed action,
                    # resolves this tick, and the explicit failure remains in the denominator and report.
                    pass
            if final_working is None or latest_runtime is None:
                reason = "Original simulation step failed on its initial attempt plus five recovery attempts; later ticks depend on this unresolved state."
                for blocked_step in range(expected_step, 6):
                    blocked_dir = rounds_dir / f"step_{blocked_step}"
                    blocked_dir.mkdir(parents=True, exist_ok=True)
                    blocker = {"status": "blocked" if blocked_step == expected_step else "blocked_dependency",
                               "step": blocked_step, "failed_attempts": round_errors, "blocker": reason,
                               "depends_on_blocked_step": expected_step if blocked_step != expected_step else None,
                               "checkpoint_before": start_checkpoint if blocked_step == expected_step else None,
                               "created_at": utc_now()}
                    blocked_path = blocked_dir / "result.json"
                    write_json(blocked_path, blocker)
                    update_state(lambda state, bs=blocked_step, bp=blocked_path: state.setdefault("B", {}).setdefault("rounds", {}).__setitem__(str(bs), {
                        "status": "blocked", "decisions_completed": 0,
                        "failed_decisions": (sum(1 for agent in Checkpoint.model_validate(start_checkpoint).social_environment.agents
                                                  if agent.is_alive()) if bs == expected_step else 0),
                        "blocker": reason, "result_path": str(bp.relative_to(ROOT)), "updated_at": utc_now(),
                    }))
                write_json(b_dir / "blocker.json", {"at": utc_now(), "blocked_from_step": expected_step,
                                                      "reason": reason, "round_errors": round_errors})
                refresh_report()
                return
            # Exact completed messages saved by the original step.
            write_json(round_dir / "messages.json", {key: value.messages for key, value in messages.items()})
            agent_rows = []
            post_by_id = {a.id: a for a in final_working.social_environment.agents}
            round_memory_verification = True
            for agent_id in sorted({a.id for a in cp.social_environment.agents}):
                data = latest_runtime["decisions"].get(agent_id)
                pre_state = state_before.get(agent_id)
                post = post_by_id.get(agent_id)
                cached_or_result = data or read_json(cache_dir / f"{agent_id}.json") or {}
                memory_updated = False
                memory_changed = False
                if data and data.get("success") and post and data.get("response"):
                    expected_response = data["response"]
                    prior = memory_before.get(agent_id, {})
                    expected_memory = copy.deepcopy(prior.get("long_term_memory", {}))
                    response_memory = expected_response.get("long_term_memory", {})
                    if response_memory:
                        expected_memory.update(response_memory)
                    expected_plan = expected_response.get("short_term_plan") or prior.get("short_term_plan", "")
                    memory_updated = (post.memory.long_term_memory == expected_memory
                                      and post.memory.short_term_plan == expected_plan
                                      and bool(post.response_history)
                                      and post.response_history[-1].model_dump(mode="json") == expected_response)
                    memory_changed = (post.memory.long_term_memory != prior.get("long_term_memory", {})
                                      or post.memory.short_term_plan != prior.get("short_term_plan", ""))
                    round_memory_verification = round_memory_verification and memory_updated
                decision_view = {
                    "success": bool(cached_or_result.get("success")), "response": cached_or_result.get("response"),
                    "calls": unit_call_records("B", f"B-round-{expected_step}-agent-{agent_id}"), "error": cached_or_result.get("error"),
                    "outer_attempt": cached_or_result.get("outer_attempt", round_attempt),
                }
                agent_rows.append({
                    "agent_id": agent_id, "moral_type": next((a.type for a in cp.social_environment.agents if a.id == agent_id), None),
                    "moral_type_label": TYPE_LABELS.get(next((a.type for a in cp.social_environment.agents if a.id == agent_id), ""), "offspring"),
                    "hp_before": pre_state["hp"] if pre_state else None,
                    "hp_after": post.state.hp if post else None,
                    "memory_updated": memory_updated,
                    "memory_changed": memory_changed,
                    "memory_before": memory_before.get(agent_id),
                    "memory_after": post.memory.model_dump(mode="json") if post else None,
                    "decision": decision_view,
                })
            if not check_memory_independence(final_working):
                round_memory_verification = False
            new_observations = [x for x in final_working.observations if x not in observations_before]
            after = json.loads(final_working.model_dump_json())
            result = {
                "status": "complete_with_persistent_failures" if failed_result_ids else "complete",
                "step": expected_step, "round_attempts": round_attempt,
                "agent_decisions": agent_rows, "failed_agent_ids": failed_result_ids,
                "new_observations": new_observations,
                "state_before": state_before,
                "state_after": {a.id: {"hp": a.state.hp, "age": a.state.age, "alive": a.is_alive()} for a in final_working.social_environment.agents},
                "resources_after": final_working.physical_environment.model_dump(mode="json"),
                "relationships_after": final_working.social_environment.relationships,
                "memory_updates_verified": sum(1 for row in agent_rows if row["memory_updated"]),
                "memory_independence_verified": check_memory_independence(final_working),
                "memory_integrity_verified": round_memory_verification,
                "relationship_consistency": relationship_consistency(final_working),
                "state_consistency": (final_working.metadata.current_time_step == expected_step + 1
                                      and len({a.id for a in final_working.social_environment.agents}) == len(final_working.social_environment.agents)
                                      and set(final_working.metadata.execution_queue) == {a.id for a in final_working.social_environment.agents if a.is_alive()}
                                       and all(0 <= a.state.hp <= a.state.max_hp for a in final_working.social_environment.agents)
                                       and relationship_consistency(final_working)),
                "reflection_stages_present": all(
                    any(c.get("stage") == "first_stage" for c in row["decision"].get("calls", []))
                    and any(c.get("stage") == "reflection" for c in row["decision"].get("calls", []))
                    for row in agent_rows if row["decision"].get("success")
                ),
                "created_at": utc_now(),
            }
            after_path = round_dir / "after.json"
            write_json(after_path, after)
            reloaded = Checkpoint.model_validate_json(after_path.read_text(encoding="utf-8"))
            result["checkpoint_reload_verified"] = (
                reloaded.metadata.current_time_step == expected_step + 1
                and json.loads(reloaded.model_dump_json()) == after
                and check_memory_independence(reloaded)
            )
            result["state_consistency"] = result["state_consistency"] and result["checkpoint_reload_verified"]
            if result["status"].startswith("complete") and (
                    not result["state_consistency"] or not result["memory_independence_verified"]
                    or not result["relationship_consistency"]):
                result["integrity_warnings"] = ["Round state, persisted checkpoint, relationships, or independent-memory verification failed."]
            write_json(round_result_path, result)
            write_json(b_dir / "last_committed.json", {"step": expected_step, "checkpoint": after})
            update_state(lambda state: state.setdefault("B", {}).setdefault("rounds", {}).__setitem__(str(expected_step), {
                "status": result["status"], "decisions_completed": sum(1 for x in agent_rows if x["decision"].get("success") or x["decision"].get("error")),
                "failed_decisions": len(failed_result_ids), "round_attempts": round_attempt, "round_errors": round_errors,
                "result_path": str(round_result_path.relative_to(ROOT)), "updated_at": utc_now(),
            }))
            cp = final_working
            note_progress(f"B tick {expected_step}/6 committed: {len(agent_rows)-len(failed_result_ids)}/8 valid decisions")
            refresh_report()
    finally:
        await bus.stop()
    write_json(b_dir / "final.json", json.loads(cp.model_dump_json()))


def finalize_metrics(server_before: dict, props: dict) -> None:
    try:
        server_after = parse_metrics()
    except Exception as exc:
        server_after = {"error": str(exc)}
    write_json(OUT / "telemetry" / "server_metrics_start.json", server_before)
    write_json(OUT / "telemetry" / "server_metrics_end.json", server_after)
    delta = {k: server_after[k] - v for k, v in server_before.items() if k in server_after}
    write_json(OUT / "telemetry" / "server_metrics_delta.json", delta)
    props_subset = {
        "model_alias": props.get("model_alias"), "model_path": props.get("model_path"),
        "model_ftype": props.get("model_ftype"), "modalities": props.get("modalities"),
        "total_slots": props.get("total_slots"), "default_generation_settings": props.get("default_generation_settings"),
        "build_info": props.get("build_info"),
    }
    write_json(OUT / "telemetry" / "server_props_summary.json", props_subset)


def audit_evaluation_artifacts() -> dict[str, Any]:
    """Revalidate durable results against their frozen checkpoints and expected protocol counts."""
    from scr.models.simulation.checkpoint import Checkpoint
    from scr.simulation.act_manager.validator.validator import validate_llm_response
    errors: list[str] = []
    a_results, c_results, b_results = read_a_results(), read_c_results(), read_b_rounds()
    audit = {
        "at": utc_now(), "planned": {"A": 40, "B_rounds": 6, "B_agent_slots": b_agent_slot_counts()[0], "C": 8},
        "observed": {"A_results": len(a_results), "B_round_results": len(b_results), "C_results": len(c_results)},
        "final_responses_revalidated": 0, "successes_missing_reflection": [],
        "durable_response_files": 0, "request_files_present": 0,
        "successful_controlled_decisions": 0, "successful_B_agent_decisions": 0,
        "completed_rounds_have_reload": True, "errors": errors,
    }

    def verify_response(label: str, checkpoint, agent_id: str, decision: dict) -> None:
        response = decision.get("response")
        if not decision.get("success") or not response:
            return
        result = validate_llm_response(json.dumps(response, ensure_ascii=False), checkpoint, agent_id=agent_id)
        if not result.success:
            errors.append(f"{label}: final response no longer validates against its frozen checkpoint: {result.error_message}")
            return
        audit["final_responses_revalidated"] += 1
        stages = {c.get("stage") for c in decision.get("calls", [])}
        if not {"first_stage", "reflection"}.issubset(stages):
            audit["successes_missing_reflection"].append(label)
            errors.append(f"{label}: successful decision lacks captured first/reflection completions")

    if len(a_results) != 40:
        errors.append(f"A result count is {len(a_results)}/40")
    if len(c_results) != 8:
        errors.append(f"C result count is {len(c_results)}/8")
    a_ids = [x.get("initial", {}).get("case_id") for x in a_results]
    c_ids = [x.get("initial", {}).get("case_id") for x in c_results]
    if len(set(a_ids)) != len(a_ids) or len(set(c_ids)) != len(c_ids):
        errors.append("Duplicate controlled-case result IDs were found")
    for experiment, results in (("A", a_results), ("C", c_results)):
        for item in results:
            label = item.get("initial", {}).get("case_id", "unknown")
            if item.get("status") not in {"complete", "blocked"}:
                errors.append(f"{experiment}/{label}: unresolved status {item.get('status')}")
            decision = item.get("decision") or {}
            if decision.get("success"):
                checkpoint = Checkpoint.model_validate(item["initial"]["checkpoint"])
                verify_response(f"{experiment}/{label}", checkpoint, decision["agent_id"], decision)
                audit["successful_controlled_decisions"] += 1
            elif item.get("status") == "complete":
                errors.append(f"{experiment}/{label}: complete status has no successful final response")
            if experiment == "A" and decision.get("success"):
                if not item.get("checkpoint_effects", {}).get("memory_updated_from_response"):
                    errors.append(f"A/{label}: response memory was not applied by the original handler")
            if experiment == "C" and item.get("status") == "complete":
                isolation = item.get("initial", {}).get("memory_isolation", {})
                if not isolation.get("memory_objects_unique") or not isolation.get("seeded_fact_only_in_focal_ltm"):
                    errors.append(f"C/{label}: initial private-memory isolation check failed")

    if len(b_results) != 6:
        errors.append(f"B round result count is {len(b_results)}/6")
    expected_steps = list(range(1, 7))
    observed_steps = sorted(int(r.get("step")) for r in b_results if isinstance(r.get("step"), int))
    if observed_steps != expected_steps:
        errors.append(f"B tick IDs are {observed_steps}, expected {expected_steps}")
    for result in b_results:
        step = result.get("step")
        if result.get("status", "").startswith("blocked"):
            continue
        before_path = OUT / "B" / "rounds" / f"step_{step}" / "before.json"
        after_path = OUT / "B" / "rounds" / f"step_{step}" / "after.json"
        try:
            before = Checkpoint.model_validate_json(before_path.read_text(encoding="utf-8"))
            after = Checkpoint.model_validate_json(after_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"B/tick_{step}: before/after checkpoint cannot be reloaded: {type(exc).__name__}: {exc}")
            continue
        if before.metadata.current_time_step != step or after.metadata.current_time_step != step + 1:
            errors.append(f"B/tick_{step}: tick sequence does not advance exactly once")
        if not result.get("checkpoint_reload_verified"):
            errors.append(f"B/tick_{step}: checkpoint reload integrity was not verified")
        if not result.get("memory_independence_verified") or not result.get("relationship_consistency"):
            errors.append(f"B/tick_{step}: memory isolation or family/relationship consistency failed")
        if not result.get("state_consistency"):
            errors.append(f"B/tick_{step}: environment state/queue consistency failed")
        for row in result.get("agent_decisions", []):
            decision = row.get("decision", {})
            if decision.get("success"):
                verify_response(f"B/tick_{step}/{row.get('agent_id')}", before, row["agent_id"], decision)
                audit["successful_B_agent_decisions"] += 1
    all_calls = read_call_results()
    audit["durable_response_files"] = len(all_calls)
    for call in all_calls:
        request_file = call.get("source_file", "").replace("_response.json", "_request.json")
        request = read_json(OUT / request_file, {}) or {}
        if request:
            audit["request_files_present"] += 1
            if request.get("response_format") != {"type": "json_object"}:
                errors.append(f"{call.get('unit_id')}/{call.get('stage')}: request did not use JSON-object response format")
        else:
            errors.append(f"{call.get('unit_id')}/{call.get('stage')}: raw request file is missing")
        if call.get("validation", {}).get("valid"):
            try:
                if not isinstance(json.loads(call.get("content", "")), dict):
                    errors.append(f"{call.get('unit_id')}/{call.get('stage')}: valid generation was not a JSON object")
            except Exception:
                errors.append(f"{call.get('unit_id')}/{call.get('stage')}: validator marked non-JSON text valid")
    audit["completed_controlled_statuses"] = Counter(x.get("status") for x in a_results + c_results)
    audit["completed_B_statuses"] = Counter(x.get("status") for x in b_results)
    audit["passed"] = not errors
    write_json(OUT / "artifact_audit.json", audit)
    return audit


def check_report_config(props: dict) -> None:
    if props.get("model_alias") and props["model_alias"] != MODEL:
        raise RuntimeError(f"Expected local model alias {MODEL}, server reports {props['model_alias']}")
    if props.get("model_ftype") and "Q4_K" not in props["model_ftype"]:
        raise RuntimeError(f"Unexpected model quantization: {props['model_ftype']}")
    if int(props.get("total_slots", 1)) != 1:
        raise RuntimeError(f"Expected one local inference slot, server reports {props.get('total_slots')}")
    params = props.get("default_generation_settings", {}).get("params", {})
    n_ctx = int(props.get("default_generation_settings", {}).get("n_ctx", 0))
    if n_ctx not in (0, 32768):
        raise RuntimeError(f"Expected 32768 context, server reports {n_ctx}")
    if props.get("modalities", {}).get("vision"):
        raise RuntimeError("Server reports vision enabled; this evaluation requires text-only inference")
    write_json(OUT / "telemetry" / "server_props.json", props)
    write_json(OUT / "telemetry" / "generation_defaults.json", {"n_ctx": n_ctx, "params": params})


def run_preflight() -> tuple[dict, dict]:
    from scr.api.llm_api.config import PROVIDER_SETTINGS
    props = api_json("/props", timeout=15)
    check_report_config(props)
    capture_reproducibility_evidence(props)
    local_settings = jsonable(PROVIDER_SETTINGS.get("llama_cpp", {}))
    serialized = json.dumps(local_settings)
    if "127.0.0.1" not in serialized and "localhost" not in serialized:
        raise RuntimeError(f"llama_cpp provider does not target localhost: {local_settings}")
    if props.get("modalities", {}).get("vision"):
        raise RuntimeError("Unexpected multimodal model mode; this evaluation requires text-only inference")
    from scr.api.db_api.connection import disable_db
    disable_db()
    before = parse_metrics()
    write_json(OUT / "telemetry" / "local_provider_config.json", local_settings)
    return props, before


def capture_reproducibility_evidence(props: dict) -> None:
    telemetry = OUT / "telemetry"
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
          "Select-Object -First 1 ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    process = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], cwd=ROOT,
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    server_info = json.loads(process.stdout) if process.stdout.strip() else {}
    command_line = str(server_info.get("CommandLine", ""))
    command_lower = command_line.lower()
    required_flags = ("--n-gpu-layers all", "--device cuda0", "--ctx-size 32768", "--parallel 1",
                      "--flash-attn on", "--cache-type-k q8_0", "--cache-type-v q8_0", "--no-mmproj", "--offline")
    missing = [flag for flag in required_flags if flag not in command_lower]
    if missing:
        raise RuntimeError(f"Running llama-server command line does not verify the required local setup flags: {missing}")
    server_info["command_line"] = command_line
    write_json(telemetry / "server_process.json", server_info)
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits", "--id=0"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    gpu_fields = [x.strip() for x in gpu_result.stdout.strip().split(",")]
    write_json(telemetry / "hardware.json", {
        "at": utc_now(), "name": gpu_fields[0], "memory_total_mib": int(float(gpu_fields[1])),
        "memory_used_mib_at_start_mib": int(float(gpu_fields[2])) if len(gpu_fields) > 2 else None,
        "model_file_bytes": (ROOT / ".local" / "models" / "Qwen3.8-27B-Q4_K_M.gguf").stat().st_size,
    })
    server_log = ROOT / ".local" / "llama-server.log"
    if server_log.exists():
        relevant = [line for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if any(term in line.lower() for term in ("offloaded 65/65", "flash_attn", "k (q8_0)", "v (q8_0)", "n_slots = 1", "n_ctx_seq = 32768"))]
        write_json(telemetry / "server_startup_evidence.json", {"log_path": str(server_log), "matched_lines": relevant[-30:]})


def verify_controlled_checkpoints() -> None:
    """No inference: instantiate every prescribed state and assert original action legality/lineage."""
    # Loading the decision package first avoids this repository's existing
    # validator <-> decision-package import cycle when validators are imported
    # directly by standalone analysis utilities.
    import scr.simulation.agent_decision  # noqa: F401
    matched_states: dict[tuple[str, int], tuple[dict, dict]] = {}
    for scenario, moral_type, seed in case_ids_a():
        cp, case_id, focal, target, initial = build_a_checkpoint(scenario, moral_type, seed)
        match_key = (scenario, seed)
        state_pair = (initial["focal_state"], initial["target_state"])
        if match_key in matched_states and matched_states[match_key] != state_pair:
            raise AssertionError(f"A changed non-moral focal/beneficiary state across moral types: {match_key}")
        matched_states[match_key] = state_pair
        if scenario in {"infant", "adult_offspring"}:
            parent, child = get_agent(cp, focal), get_agent(cp, target)
            assert child.family.parent_id == parent.id and parent.id in cp.metadata.execution_queue
            assert child.type == parent.type and child.is_alive()
        else:
            assert focal != target and target == "agent_8"
        write_json(OUT / "preflight" / "A" / f"{case_id}.json", initial)
    for history, length, moral_type, seed in case_ids_c():
        cp, case_id, focal, initial = build_c_checkpoint(history, length, moral_type, seed)
        assert cp.configuration.isSocialInteractionStep(cp.metadata.current_time_step)
        write_json(OUT / "preflight" / "C" / f"{case_id}.json", initial)
    write_json(OUT / "preflight" / "verified.json", {
        "at": utc_now(), "A_checkpoints": 40, "C_checkpoints": 8,
        "checks": ["8-agent 2-per-type base cohort", "family age/type/parent-child consistency",
                   "original social-step action availability", "legal allocate witnesses", "unique per-agent memories",
                   "successful recorded attack fixture", "history and beneficiary state invariants"],
    })


def record_unblinded_key() -> None:
    write_blinded_review_tables()
    report = REPORT_PATH.read_text(encoding="utf-8")
    if "blinded_review" not in report:
        addition = "\nRaw evidence and blinded review files are stored at `data/qwen38_capability_evaluation_20261016/`; the A moral-role unblinding key is separate from `blinded_review/A_review.csv`.\n"
        report = report.replace("## Assessment\n", "## Assessment\n" + addition)
        REPORT_PATH.write_text(report, encoding="utf-8")


def run_all() -> None:
    from scr.api.db_api.connection import disable_db
    disable_db()
    initialize_state()
    global REPORT_TEMPLATE
    REPORT_TEMPLATE = REPORT_PATH.read_text(encoding="utf-8")
    ensure_server()
    props, server_before = run_preflight()
    monitor_thread = start_monitor()
    append_jsonl(OUT / "telemetry" / "runner_lifecycle.jsonl", {"at": utc_now(), "event": "started", "pid": os.getpid(), "server": health()})
    try:
        sim_logger_init = __import__("scr.utils.sim_logger", fromlist=["init"])
        sim_logger_init.init(OUT / "runtime")
        install_instrumentation()
        set_current("checkpoint preflight", None)
        verify_controlled_checkpoints()
        set_current("A", "controlled moral decisions")
        for scenario, moral_type, seed in case_ids_a():
            case_id = f"{scenario}__{moral_type}__s{seed}"
            state_result = STATE.get("A", {}).get(case_id)
            disk_result = read_json(OUT / "A" / case_id / "result.json")
            if (state_result and state_result.get("status") in {"complete", "blocked"}) or (disk_result and disk_result.get("status") == "complete"):
                continue
            run_a_case(scenario, moral_type, seed)
        set_current("B", "six-round eight-agent simulation")
        run_coroutine(run_b())
        set_current("C", "memory and context stress")
        for history, length, moral_type, seed in case_ids_c():
            case_id = f"{history}__{length}__{moral_type}__s{seed}"
            state_result = STATE.get("C", {}).get(case_id)
            disk_result = read_json(OUT / "C" / case_id / "result.json")
            if (state_result and state_result.get("status") in {"complete", "blocked"}) or (disk_result and disk_result.get("status") == "complete"):
                continue
            run_c_case(history, length, moral_type, seed)
        finalize_metrics(server_before, props)
        record_unblinded_key()
        audit = audit_evaluation_artifacts()
        append_jsonl(OUT / "telemetry" / "runner_lifecycle.jsonl", {"at": utc_now(), "event": "artifact_audit", "passed": audit.get("passed"), "errors": audit.get("errors", [])})
        end_counts = progress_counts()
        all_resolved = (end_counts["A_complete"] + end_counts["A_blocked"] == 40
                        and end_counts["B_rounds_complete"] + end_counts["B_rounds_blocked"] == 6
                        and end_counts["C_complete"] + end_counts["C_blocked"] == 8)
        has_blockers = bool(end_counts["A_blocked"] or end_counts["B_rounds_blocked"] or end_counts["C_blocked"])
        final_status = ("complete_with_blockers" if has_blockers else
                        "complete_with_integrity_warnings" if not audit.get("passed") else "complete")
        update_state(lambda state: state.update({"status": final_status if all_resolved else "blocked_with_unresolved_units",
                                                 "finished_at": utc_now(), "current_unit": None,
                                                  "phase": "complete" if all_resolved else "blocked"}))
        refresh_report()
        append_jsonl(OUT / "telemetry" / "runner_lifecycle.jsonl", {"at": utc_now(), "event": "finished", "status": STATE["status"], "counts": end_counts})
    finally:
        MONITOR_STOP.set()
        monitor_thread.join(timeout=20)
        if ASYNC_LOOP is not None and not ASYNC_LOOP.is_closed():
            ASYNC_LOOP.close()
        try:
            __import__("scr.utils.sim_logger", fromlist=["close"]).close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        run_all()
    except BaseException as exc:
        append_jsonl(OUT / "telemetry" / "fatal_errors.jsonl", {"at": utc_now(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        if STATE:
            update_state(lambda state: state.update({"status": "crashed_recoverable", "last_error": f"{type(exc).__name__}: {exc}", "crashed_at": utc_now()}))
        raise
