"""Small local-only persistence, measurement, context and watchdog utilities."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/qwen38_remediation_v2_20261016"
API = "http://127.0.0.1:8080"
CONTEXT = 65536
OUTPUT_RESERVE = 8192
REFLECTION_OVERHEAD_RESERVE = 2304
MAX_RECOVERIES = 5
FILE_LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FILE_LOCK:
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(temp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.1)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FILE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def write_jsonlines(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value="".join(json.dumps(row,ensure_ascii=False,default=str)+"\n" for row in rows)
    with FILE_LOCK:
        if path.exists() and path.read_text(encoding="utf-8")==value:
            return
        if path.exists():
            backup=path.with_name(path.stem+f".previous.{time.time_ns()}"+path.suffix)
            backup.write_bytes(path.read_bytes())
        temp=path.with_name(path.name+".tmp")
        temp.write_text(value,encoding="utf-8");os.replace(temp,path)


def jsonlines(path):
    try:
        result = []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return result
    except OSError:
        return []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def api(path, body=None, timeout=10):
    if not path.startswith("/"):
        raise ValueError("Only local llama-server paths are allowed")
    request = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read().decode()
        return json.loads(data)


def metrics():
    with urllib.request.urlopen(API + "/metrics", timeout=5) as response:
        text = response.read().decode()
    result = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    result[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return result


def gpu():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                                      "--format=csv,noheader,nounits", "--id=0"], text=True, timeout=8)
    values = [float(x.strip()) for x in output.split(",")]
    return dict(zip(("used_mib", "total_mib", "utilization_percent", "power_watts"), values))


def reasoning_settings(mode):
    if mode not in ("off", "low"):
        raise ValueError(mode)
    return {"reasoning_effort": "none" if mode == "off" else "low",
            "chat_template_kwargs": {"enable_thinking": mode != "off"}, "reasoning_format": "deepseek"}


def token_count(messages, mode="off"):
    rendered = api("/apply-template", {"messages": messages, **reasoning_settings(mode)}, timeout=30)
    result = api("/tokenize", {"content": rendered["prompt"], "add_special": False}, timeout=30)
    return len(result["tokens"])


class ContextBudgetExceeded(RuntimeError):
    pass


def budget_check(prompt_tokens, stage, context=CONTEXT, output_reserve=OUTPUT_RESERVE):
    future = output_reserve + REFLECTION_OVERHEAD_RESERVE if stage == "first_stage" else 0
    needed = prompt_tokens + output_reserve + future + 64  # tokenizer/template boundary margin
    record = {"estimated_prompt_tokens": prompt_tokens, "output_reserve": output_reserve,
              "future_reflection_reserve": future, "boundary_margin": 64,
              "required_context": needed, "configured_context": context, "fits": needed <= context}
    if not record["fits"]:
        raise ContextBudgetExceeded("Explicit context preflight; original observations/memories retained: " + json.dumps(record))
    return record


def context_truncated(record, context=CONTEXT):
    nested = record.get("server_context", {}) or {}
    usage = (record.get("input_tokens") or 0) + (record.get("output_tokens") or 0)
    return bool(record.get("server_context_truncated") or nested.get("truncated") or usage > context)


def action_dict(response):
    if response is None:
        return {}
    if hasattr(response, "action"):
        return response.action.root.model_dump(mode="json")
    action = response.get("action", {})
    return action.get("root", action)


def allocation_measurement(action, before, after, actor):
    """Measure immediate execution deltas, before subsequent agents/metabolism.

    `offered_hp` can exceed received HP at the original max-HP cap. We report
    both and never infer transfers from full-round net HP changes.
    """
    action = action.get("root", action) if isinstance(action, dict) else action
    if not action or action.get("action_type") != "allocate":
        return {"allocation_action": False, "offered_hp": 0, "source_spent_hp": 0, "received_hp": 0, "recipients": []}
    plan = action.get("allocation_plan", {})
    recipients = [{"recipient": aid, "planned_hp": amount, "hp_before": before.get(aid),
                   "hp_after": after.get(aid), "received_hp": max(0, (after.get(aid) or 0) - (before.get(aid) or 0))}
                  for aid, amount in plan.items()]
    spent = (before.get(actor) or 0) - (after.get(actor) or 0)
    return {"allocation_action": True, "offered_hp": sum(plan.values()), "source_spent_hp": spent,
            "received_hp": sum(r["received_hp"] for r in recipients), "recipients": recipients,
            "executed": spent > 0, "source_matches_plan": spent == sum(plan.values())}


def family_consistent(checkpoint,require_inheritance=False):
    living=list(checkpoint.social_environment.agents)
    deceased=[record.agent for record in checkpoint.social_environment.cemetery]
    all_agents=living+deceased
    known={a.id:a for a in all_agents}
    if len(known)!=len(all_agents):
        return False
    for agent in all_agents:
        parent=known.get(agent.family.parent_id) if agent.family.parent_id else None
        if agent.family.parent_id and (parent is None or agent.id not in parent.family.children_ids):
            return False
        if parent and require_inheritance and parent.type!=agent.type:
            return False
        for child_id in agent.family.children_ids:
            child=known.get(child_id)
            if child is None or child.family.parent_id!=agent.id:
                return False
    return True


def watchdog_classification(*, healthy, process_alive, active, idle_seconds, progress_age,
                            request_age, timeout=3600, oom=False):
    if not process_alive:
        return {"state": "crashed", "restart": True}
    if oom:
        return {"state": "explicit_gpu_oom", "restart": True}
    if active and request_age < timeout and (progress_age < 180 or healthy):
        # A responsive server with an active slot is not proven stalled just
        # because prompt processing or native thinking takes several minutes.
        return {"state": "healthy_active_slow" if request_age >= 300 else "healthy_active", "restart": False}
    if active and request_age >= timeout and progress_age >= 180:
        return {"state": "confirmed_request_timeout_without_progress", "restart": True}
    if not healthy and idle_seconds >= 180 and progress_age >= 180:
        return {"state": "unresponsive_without_progress", "restart": True}
    return {"state": "healthy_idle" if healthy else "unresponsive_grace_period", "restart": False}


def owned_server_alive():
    owned = read(OUT / "server/owned_server.json", {})
    pid = owned.get("pid")
    if not pid:
        return False
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; if($p -and $p.ExecutablePath -eq '{str(ROOT / '.local/llama-b11669/llama-server.exe')}'){{exit 0}}else{{exit 1}}"],
        capture_output=True, timeout=15)
    return result.returncode == 0


def ensure_server():
    try:
        if api("/health").get("status") == "ok":
            if not owned_server_alive():
                raise RuntimeError("Port 8080 belongs to a server not owned by this evaluation")
            return
    except (OSError, TimeoutError):
        pass
    if not owned_server_alive():
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                 str(ROOT / "scripts/start-qwen38-v2-server.ps1"), "-ContextSize", str(CONTEXT)],
                                text=True, capture_output=True, timeout=45)
        append(OUT / "telemetry/recovery.jsonl", {"at": now(), "event": "start_owned_server", "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(result.stderr)
    for _ in range(60):
        try:
            if api("/health").get("status") == "ok":
                return
        except (OSError, TimeoutError):
            pass
        time.sleep(2)
    raise RuntimeError("Owned llama-server did not become healthy within 120s")


def restart_owned_server(reason):
    active=read(OUT / "active_call.json", {})
    unit=active.get("unit") or "idle_recovery"
    with FILE_LOCK:
        counts=read(OUT / "server/recovery_counts.json", {})
        if counts.get(unit,0)>=MAX_RECOVERIES:
            raise RuntimeError("Persistent server recovery budget exhausted for " + unit)
        counts[unit]=counts.get(unit,0)+1
        write(OUT / "server/recovery_counts.json", counts)
    owned = read(OUT / "server/owned_server.json", {})
    if owned_server_alive():
        subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {int(owned['pid'])} -Force"],
                       capture_output=True, timeout=15, check=True)
    append(OUT / "telemetry/recovery.jsonl", {"at": now(), "event": "restart_owned_server", "reason": reason, "previous": owned})
    time.sleep(2)
    ensure_server()


def log_token_progress(log_path):
    try:
        with Path(log_path).open("rb") as handle:
            handle.seek(max(0, Path(log_path).stat().st_size - 16000))
            tail = handle.read().decode("utf-8", errors="replace")
        lines = [s for s in tail.splitlines() if any(x in s for x in ("n_gen =", "prompt processing progress", "stop processing:", "prompt done"))]
        return lines[-1] if lines else None
    except OSError:
        return None


class Monitor(threading.Thread):
    def __init__(self, state, report_callback):
        super().__init__(daemon=True)
        self.state, self.report_callback = state, report_callback
        self.stop_event = threading.Event()
        self.last_progress = time.monotonic()
        self.last_signature = None
        self.failed_since = None
        self.restart_counts = {}

    def run(self):
        while not self.stop_event.is_set():
            sample = {"at": now(), "pid": os.getpid(), "unit": self.state.get("current_unit")}
            try:
                sample["gpu"] = gpu()
                sample["health"] = api("/health", timeout=5)
                sample["metrics"] = metrics()
                slots = api("/slots", timeout=5)
                sample["slots"] = slots
                active = any(s.get("is_processing") for s in slots)
                signature = (log_token_progress(read(OUT / "server/owned_server.json", {}).get("log", "")),
                             [(s.get("n_prompt_tokens"), s.get("n_prompt_tokens_processed"), s.get("n_decoded")) for s in slots])
                if signature != self.last_signature:
                    self.last_progress, self.last_signature = time.monotonic(), signature
                self.failed_since = None
                call = read(OUT / "active_call.json", {})
                age = max(0, time.time() - call.get("started_epoch", time.time())) if call.get("active") else 0
                classification = watchdog_classification(healthy=True, process_alive=True, active=active,
                    idle_seconds=0, progress_age=time.monotonic() - self.last_progress, request_age=age)
                sample.update(classification)
                sample["token_progress"] = signature[0]
                if classification["restart"]:
                    unit = self.state.get("current_unit", "unknown")
                    count = self.restart_counts.get(unit, 0)
                    if count < MAX_RECOVERIES:
                        self.restart_counts[unit] = count + 1
                        restart_owned_server(classification["state"])
            except Exception as exc:
                sample["monitor_error"] = f"{type(exc).__name__}: {exc}"
                if self.failed_since is None:
                    self.failed_since = time.monotonic()
                # The requesting unit controls crash recovery. This thread must
                # remain alive while health probes fail and never kill others.
            append(OUT / "telemetry/monitor.jsonl", sample)
            write(OUT / "heartbeat.json", {"at": now(), "pid": os.getpid(), "state": self.state, "monitor": sample})
            try:
                self.report_callback()
            except Exception as exc:
                append(OUT / "telemetry/report_errors.jsonl", {"at": now(), "error": str(exc)})
            self.stop_event.wait(15)
