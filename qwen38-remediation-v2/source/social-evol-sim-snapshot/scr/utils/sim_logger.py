"""
Unified JSONL event logger for the simulation.

Provides a single writer to ``data/<run_id>/events.jsonl`` that all
subsystems funnel into: stdlib logging (via JSONStdlibHandler), the
event bus (via jsonl_event_bus_subscriber), and direct emit() calls.
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Context variables (asyncio.gather copies context per task automatically)
# ---------------------------------------------------------------------------

_run_id: ContextVar[str] = ContextVar("sim_run_id", default="")
_step: ContextVar[Optional[int]] = ContextVar("sim_step", default=None)
_agent_id: ContextVar[str] = ContextVar("sim_agent_id", default="")
_parent_id: ContextVar[str] = ContextVar("sim_parent_id", default="")

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_file = None
_file_path: Optional[Path] = None


def bind(**kwargs) -> list:
    """Set context vars, return list of reset tokens."""
    tokens = []
    mapping = {
        "run_id": _run_id,
        "step": _step,
        "agent_id": _agent_id,
        "parent_id": _parent_id,
    }
    for key, value in kwargs.items():
        var = mapping.get(key)
        if var is not None:
            tokens.append(var.set(value))
    return tokens


def unbind(tokens: list) -> None:
    """Reset context vars from tokens returned by bind()."""
    for token in tokens:
        token.var.reset(token)


# ---------------------------------------------------------------------------
# Init / close
# ---------------------------------------------------------------------------


def init(run_dir: Path) -> None:
    """Open the events.jsonl file for writing."""
    global _file, _file_path
    if _file is not None:
        close()
    _file_path = run_dir / "events.jsonl"
    _file_path.parent.mkdir(parents=True, exist_ok=True)
    _file = open(_file_path, "a", buffering=1)  # line-buffered


def close() -> None:
    """Flush and close the events.jsonl file."""
    global _file, _file_path
    if _file is not None:
        try:
            _file.flush()
            _file.close()
        except Exception:
            pass
        _file = None
        _file_path = None


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------

# Guard against recursion: emit() must never trigger stdlib logging
_emitting = False


def emit(
    event: str,
    *,
    type: str = "log",
    level: str = "info",
    **data: Any,
) -> None:
    """Write one JSONL line with auto-populated context fields.

    This function does raw file.write() and never calls stdlib logger
    to prevent infinite recursion when used from JSONStdlibHandler.
    """
    global _emitting
    if _file is None or _emitting:
        return

    _emitting = True
    try:
        step_val = _step.get()
        record: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "id": uuid.uuid4().hex[:12],
            "parent_id": _parent_id.get(),
            "run_id": _run_id.get(),
            "step": step_val if step_val is not None else -1,
            "agent_id": _agent_id.get(),
            "type": type,
            "event": event,
            "level": level,
            "data": data,
        }
        _file.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # never let logging failures crash the simulation
    finally:
        _emitting = False


# ---------------------------------------------------------------------------
# Event bus subscriber
# ---------------------------------------------------------------------------


async def jsonl_event_bus_subscriber(event) -> None:
    """Async subscriber that translates event bus events into emit() calls.

    Replaces the old ``logging_subscriber``.
    """
    emit(
        event.event_type.value,
        type="event_bus",
        level="info",
        **event.data,
    )


# ---------------------------------------------------------------------------
# Stdlib logging handler
# ---------------------------------------------------------------------------


class JSONStdlibHandler(logging.Handler):
    """Bridges stdlib logger.info() calls into sim_logger.emit().

    Attached to scr.* loggers so that existing 179+ logger.info() calls
    automatically flow into events.jsonl without modifying call sites.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Use the module-level emit function (not self.emit — that's this method)
        _emit_from_handler(record)


def _emit_from_handler(record: logging.LogRecord) -> None:
    """Translate a logging.LogRecord into a sim_logger.emit() call."""
    from scr.utils.sim_logger import emit as sim_emit

    sim_emit(
        "log",
        type="log",
        level=record.levelname.lower(),
        message=record.getMessage(),
        logger=record.name,
    )
