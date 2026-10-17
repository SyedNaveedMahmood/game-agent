"""
Retry tracking module for the Morality-AI Simulation.

This module provides structured logging and analysis of LLM retry attempts,
including JSONL logging, summary generation, and diagnosis prompt creation.
"""

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from scr.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Validation stage & result
# ---------------------------------------------------------------------------


class ValidationStage(Enum):
    """Which validation stage failed."""
    JSON = "json"
    SCHEMA = "schema"
    CONTEXTUAL = "contextual"


@dataclass
class ValidationResult:
    """Structured result flowing from validate_llm_response through processors to retry loops."""
    success: bool
    response: object = None          # Response on success, None on failure
    errors: List[str] = field(default_factory=list)
    stage: Optional[ValidationStage] = None
    error_type_override: Optional[str] = None  # Set by exception handlers to bypass stage-based derivation

    @property
    def error_type(self) -> str:
        """Derive error_type from override, validation stage, or fallback to unknown."""
        if self.error_type_override is not None:
            return self.error_type_override
        if self.stage is not None:
            return f"validation_{self.stage.value}"
        return "unknown"

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors) if self.errors else ""


# ---------------------------------------------------------------------------
# Error-type vocabulary
# ---------------------------------------------------------------------------

ERROR_TYPES = {
    "validation_json",
    "validation_schema",
    "validation_contextual",
    "llm_exception",
    "timeout",
    "empty_response",
    "rate_limit",
    "connection_error",
    "subprocess_error",
    "auth_error",
    "unknown",
}

# ---------------------------------------------------------------------------
# Root-cause hint auto-classification
# ---------------------------------------------------------------------------

_ROOT_CAUSE_PATTERNS: List[tuple] = [
    # Resource / HP errors (action handlers)
    (r"(?i)(insufficient\s+(hp|quantity)|not enough of resource|no resources available)", "insufficient_resources"),
    # Invalid action constraints
    (r"(?i)(cannot fight yourself|not old enough|can only choose to|cannot be collected|exceeds maximum length|cannot be empty)", "invalid_action"),
    # Target not found / hallucinated IDs
    (r"(?i)(not found|does not exist|no such|invalid.*id)", "state_hallucination"),
    # Dead / stale targets
    (r"(?i)(is dead|not alive|already dead|stale)", "stale_state"),
    # litellm streaming chunk limit
    (r"(?i)(separator.*chunk.*longer|chunk.*limit)", "response_too_long"),
    # Context length / token limit
    (r"(?i)(context.?length|too.?long|max.?tokens|token.?limit|content.?size)", "context_length"),
    # JSON parse errors
    (r"(?i)(invalid json|parse|decode|unexpected token)", "json_syntax"),
    # Pydantic schema errors
    (r"(?i)(failed to match.*schema|field.*error type:)", "schema_mismatch"),
    # Rate limiting
    (r"(?i)(rate.?limit|429|too many requests)", "rate_limit"),
    # Timeout
    (r"(?i)(timeout|timed?\s*out|probable timeout)", "timeout"),
    # Empty / no response
    (r"(?i)(empty response|no output|no content|no response)", "empty_response"),
    # Connection / network errors
    (r"(?i)(connection|network|refused|unreachable|dns)", "connection_error"),
    # Auth errors
    (r"(?i)(unauthorized|forbidden|401|403|auth.*fail|invalid.*key)", "auth_error"),
    # Subprocess / process errors
    (r"(?i)(failed \(exit|subprocess|process.*fail|exit code)", "subprocess_error"),
]


def classify_root_cause(error_message: str) -> str:
    """Classify an error message into a root-cause hint via regex matching."""
    for pattern, hint in _ROOT_CAUSE_PATTERNS:
        if re.search(pattern, error_message):
            return hint
    return "unknown"


def classify_llm_exception(exc: Exception) -> str:
    """Classify an LLM-related exception into an error_type string.

    Used by response processors to set ``error_type_override`` on ValidationResult
    so the retry tracker records a precise error type instead of ``"unknown"``.
    """
    import asyncio as _asyncio

    if isinstance(exc, (_asyncio.TimeoutError, TimeoutError)):
        return "timeout"

    error_str = str(exc).lower()

    if "empty response" in error_str or "no output" in error_str or "no content" in error_str:
        return "empty_response"
    if "429" in str(exc) or "rate" in error_str and "limit" in error_str:
        return "rate_limit"
    if "401" in str(exc) or "403" in str(exc) or "unauthorized" in error_str or "forbidden" in error_str:
        return "auth_error"
    if isinstance(exc, (ConnectionError, OSError)) and "timeout" not in error_str:
        return "connection_error"
    if "connection" in error_str or "network" in error_str or "refused" in error_str:
        return "connection_error"
    if "failed (exit" in error_str or "subprocess" in error_str:
        return "subprocess_error"
    if "timeout" in error_str or "timed out" in error_str:
        return "timeout"

    return "llm_exception"


def reclassify_error(
    error_type: str,
    error_message: str,
    duration: float,
    call_timeout: float,
    llm_raw_output: str,
) -> tuple:
    """Safety-net reclassification for errors that slipped through primary classification.

    Returns ``(error_type, error_message)`` — possibly adjusted.
    """
    # Near-timeout with no LLM output → treat as timeout regardless of original label
    if duration >= call_timeout * 0.9 and not llm_raw_output.strip():
        if error_type in ("unknown", "llm_exception"):
            return "timeout", f"Probable timeout ({duration:.1f}s, no LLM output)"

    # Generic "Validation failed" with no detail → make the message more informative
    if error_type == "unknown" and error_message in ("Validation failed", ""):
        if duration >= call_timeout * 0.9:
            return "timeout", f"Probable timeout ({duration:.1f}s)"
        return "llm_exception", "LLM returned empty or invalid response"

    return error_type, error_message


# ---------------------------------------------------------------------------
# RetryRecord dataclass
# ---------------------------------------------------------------------------


@dataclass
class RetryRecord:
    agent_id: str
    step: int
    attempt: int
    max_attempts: int
    timestamp: float
    error_type: str
    error_message: str
    root_cause_hint: str
    action_type: str
    model: str
    two_stage: bool
    duration_seconds: float
    run_id: str
    agent_state: dict = field(default_factory=dict)
    available_targets: list = field(default_factory=list)
    llm_raw_output: str = ""
    prompt_tail: str = ""


# ---------------------------------------------------------------------------
# RetryTracker
# ---------------------------------------------------------------------------


class RetryTracker:
    """Accumulates RetryRecords and writes JSONL logs, summaries, and diagnosis prompts."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._records: List[RetryRecord] = []

    # -- recording ----------------------------------------------------------

    def record(self, **kwargs) -> RetryRecord:
        """Create and accumulate a RetryRecord."""
        rec = RetryRecord(run_id=self.run_id, **kwargs)
        self._records.append(rec)
        return rec

    # -- persistence --------------------------------------------------------

    def flush(self) -> None:
        """No-op — retry events are now written to events.jsonl via sim_logger.emit()."""
        self._records.clear()

    # -- analysis -----------------------------------------------------------

    def _read_log(self) -> List[dict]:
        """Read retry records, preferring events.jsonl over debug/retry_log.jsonl."""
        events_path = Path(f"data/{self.run_id}/events.jsonl")
        if events_path.exists():
            primary_records: List[dict] = []
            fallback_records: List[dict] = []
            with open(events_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") != "retry":
                        continue
                    # sim_logger emits canonical retry records with type="retry".
                    # The event bus may also mirror the same retry with type="event_bus";
                    # keep those only as a fallback for older logs that lack canonical records.
                    is_primary = event.get("type") == "retry"
                    # Extract the data dict and merge top-level fields
                    rec = dict(event.get("data", {}))
                    # Ensure required fields are present (fall back to event-level)
                    if "step" not in rec and "step" in event:
                        rec["step"] = event["step"]
                    if "agent_id" not in rec and "agent_id" in event:
                        rec["agent_id"] = event["agent_id"]
                    # Map timestamp from event ts (ISO string → epoch float)
                    if "timestamp" not in rec and "ts" in event:
                        try:
                            import datetime
                            rec["timestamp"] = datetime.datetime.fromisoformat(event["ts"]).timestamp()
                        except (ValueError, TypeError):
                            rec["timestamp"] = 0.0
                    # Provide safe defaults for fields used by summary/diagnosis
                    rec.setdefault("error_message", rec.get("error_type", ""))
                    rec.setdefault("attempt", 1)
                    rec.setdefault("max_attempts", 1)
                    if is_primary:
                        primary_records.append(rec)
                    else:
                        fallback_records.append(rec)
            return primary_records or fallback_records
        return []

    def _read_decision_count(self) -> int:
        """Count unique agent decisions from decision-request events."""
        events_path = Path(f"data/{self.run_id}/events.jsonl")
        if not events_path.exists():
            return 0

        decisions = set()
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "llm_request" or event.get("type") != "decision":
                    continue
                data = event.get("data", {})
                agent_id = data.get("agent_id") or event.get("agent_id")
                step = data.get("step", event.get("step"))
                if agent_id and step is not None:
                    decisions.add((agent_id, step))

        return len(decisions)

    def summary_for_run(self) -> dict:
        """Read JSONL log and write ``data/<run_id>/debug/retry_summary.json``.

        Returns the summary dict.
        """
        records = self._read_log()
        if not records:
            summary: dict = {
                "run_id": self.run_id,
                "total_retries": 0,
                "decision_count": self._read_decision_count(),
                "retried_decision_count": 0,
                "error_breakdown": {},
                "root_cause_breakdown": {},
                "hotspot_agents": [],
                "hotspot_steps": [],
                "retry_rate": 0.0,
                "patterns": [],
                "worst_case": None,
            }
        else:
            total = len(records)

            # Error breakdown
            error_counter: Counter = Counter(r["error_type"] for r in records)
            root_cause_counter: Counter = Counter(
                r["root_cause_hint"] for r in records
            )

            # Hotspot agents (top 5)
            agent_counter: Counter = Counter(r["agent_id"] for r in records)
            hotspot_agents = [
                {"agent_id": aid, "retry_count": cnt}
                for aid, cnt in agent_counter.most_common(5)
            ]

            # Hotspot steps (top 5)
            step_counter: Counter = Counter(r["step"] for r in records)
            hotspot_steps = [
                {"step": step, "retry_count": cnt}
                for step, cnt in step_counter.most_common(5)
            ]

            # Retry rate: retries / all unique decision requests when available.
            retried_decision_count = len(
                {(r["agent_id"], r["step"]) for r in records}
            )
            decision_count = self._read_decision_count()
            retry_denominator = max(decision_count, retried_decision_count)
            retry_rate = total / retry_denominator if retry_denominator else 0.0

            # Pattern detection
            patterns: List[str] = []
            if error_counter.get("validation_json", 0) > total * 0.5:
                patterns.append(
                    "Majority of retries are JSON parse failures - "
                    "consider strengthening output format instructions."
                )
            if root_cause_counter.get("state_hallucination", 0) > total * 0.3:
                patterns.append(
                    "Significant state hallucination - agents reference "
                    "nonexistent targets. Consider injecting target lists "
                    "into prompts."
                )
            if root_cause_counter.get("rate_limit", 0) > total * 0.2:
                patterns.append(
                    "Rate limiting is a notable contributor. "
                    "Consider reducing concurrency or adding backoff."
                )
            dominant_error = error_counter.most_common(1)[0]
            if dominant_error[1] > total * 0.7:
                patterns.append(
                    f"Error type '{dominant_error[0]}' dominates "
                    f"({dominant_error[1]}/{total}). Focus debugging there."
                )

            # Worst case: record with highest attempt number
            worst = max(records, key=lambda r: r["attempt"])

            summary = {
                "run_id": self.run_id,
                "total_retries": total,
                "decision_count": decision_count,
                "retried_decision_count": retried_decision_count,
                "error_breakdown": dict(error_counter),
                "root_cause_breakdown": dict(root_cause_counter),
                "hotspot_agents": hotspot_agents,
                "hotspot_steps": hotspot_steps,
                "retry_rate": round(retry_rate, 3),
                "patterns": patterns,
                "worst_case": {
                    "agent_id": worst["agent_id"],
                    "step": worst["step"],
                    "attempt": worst["attempt"],
                    "error_type": worst["error_type"],
                    "error_message": worst["error_message"],
                },
            }

        path = Path(f"data/{self.run_id}/debug/retry_summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug("Retry summary written to %s", path)
        return summary

    def generate_diagnosis_prompt(self) -> str:
        """Write ``data/<run_id>/debug/diagnosis_prompt.md`` with an AI-friendly analysis prompt.

        Returns the markdown content.
        """
        summary = self.summary_for_run()
        records = self._read_log()

        # Time range
        if records:
            ts_min = min(r["timestamp"] for r in records)
            ts_max = max(r["timestamp"] for r in records)
            time_range = f"{ts_max - ts_min:.1f}s"
        else:
            time_range = "N/A"

        lines: List[str] = []
        lines.append("# Retry Diagnosis Prompt")
        lines.append("")
        lines.append("## Context")
        lines.append("")
        lines.append(f"- **Run ID**: `{self.run_id}`")
        lines.append(f"- **Total retries**: {summary['total_retries']}")
        lines.append(f"- **Time range**: {time_range}")
        lines.append(f"- **Retry rate** (retries per decision): {summary['retry_rate']}")
        lines.append("")

        # Summary table - error breakdown
        lines.append("## Error Breakdown")
        lines.append("")
        lines.append("| Error Type | Count |")
        lines.append("|---|---|")
        for etype, count in sorted(
            summary["error_breakdown"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {etype} | {count} |")
        lines.append("")

        # Root cause breakdown
        lines.append("## Root Cause Hints")
        lines.append("")
        lines.append("| Root Cause | Count |")
        lines.append("|---|---|")
        for cause, count in sorted(
            summary["root_cause_breakdown"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {cause} | {count} |")
        lines.append("")

        # Patterns
        if summary["patterns"]:
            lines.append("## Detected Patterns")
            lines.append("")
            for p in summary["patterns"]:
                lines.append(f"- {p}")
            lines.append("")

        # Top 3 issues with raw output samples
        lines.append("## Top Issues with Raw Output Samples")
        lines.append("")
        error_counter: Counter = Counter(r["error_type"] for r in records)
        for i, (etype, _count) in enumerate(error_counter.most_common(3)):
            samples = [r for r in records if r["error_type"] == etype][:2]
            lines.append(f"### {i + 1}. {etype}")
            lines.append("")
            for s in samples:
                lines.append(
                    f"**Agent {s['agent_id']}, step {s['step']}, "
                    f"attempt {s['attempt']}**"
                )
                lines.append("")
                lines.append(f"Error: `{s['error_message']}`")
                lines.append("")
                raw = s.get("llm_raw_output", "")
                if raw:
                    # Truncate for readability
                    snippet = raw[:500]
                    lines.append("```")
                    lines.append(snippet)
                    lines.append("```")
                    lines.append("")
            lines.append("")

        # Closing question
        lines.append("## Analysis Request")
        lines.append("")
        lines.append(
            "Based on these patterns, what changes to prompts, validation, "
            "or retry logic would reduce the retry rate?"
        )
        lines.append("")

        content = "\n".join(lines)

        path = Path(f"data/{self.run_id}/debug/diagnosis_prompt.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        logger.debug("Diagnosis prompt written to %s", path)
        return content
