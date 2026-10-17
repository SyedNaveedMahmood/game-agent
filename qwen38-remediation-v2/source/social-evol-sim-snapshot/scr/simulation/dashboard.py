"""
Rich Live Dashboard for real-time simulation monitoring.

Subscribes to the event bus and renders a Rich Live table.
"""

import asyncio
import sys
import time
from typing import Dict, Optional
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from scr.simulation.event_bus import SimulationEvent, EventType, SimulationEventBus
from scr.utils.logger import suppress_console_logging

console = Console()


class SimulationDashboard:
    """Real-time dashboard using Rich Live display."""

    def __init__(self, event_bus: SimulationEventBus):
        self.event_bus = event_bus
        self._live: Optional[Live] = None
        self._current_step: int = 0
        self._current_phase: str = "Initializing"
        self._num_agents: int = 0
        self._step_start_time: float = time.time()
        self._agent_states: Dict[str, Dict] = {}
        self._total_errors: int = 0
        self._total_retries: int = 0
        self._llm_times: list = []
        self._error_breakdown: Dict[str, int] = {}
        self._run_id: str = ""
        self._enabled: bool = True

        # Auto-detect non-TTY
        if not sys.stdout.isatty():
            self._enabled = False

    async def start(self) -> None:
        """Start the dashboard display and subscribe to events."""
        if not self._enabled:
            return

        suppress_console_logging()
        self.event_bus.subscribe(self._handle_event)
        self._live = Live(
            self._render(),
            refresh_per_second=4,
            console=console,
        )
        self._live.start()

    async def stop(self) -> None:
        """Stop the dashboard display."""
        if self._live:
            self._live.stop()
            self._live = None

    def _render(self) -> Panel:
        """Render the current dashboard state."""
        elapsed = time.time() - self._step_start_time

        # Header
        header = Text()
        header.append(f"[Step {self._current_step}] ", style="bold cyan")
        header.append(f"[{self._current_phase}] ", style="bold yellow")
        header.append(f"[Elapsed: {elapsed:.1f}s]", style="bold green")

        # Agent table
        table = Table(show_header=True, header_style="bold magenta", expand=True, padding=(0, 1, 1, 0))
        table.add_column("Agent", style="cyan", width=10)
        table.add_column("Stage", width=12)
        table.add_column("Tokens", justify="right", width=7)
        table.add_column("Time(s)", justify="right", width=7)
        table.add_column("Action", width=13)
        table.add_column("Retries", width=15)
        table.add_column("Preview", no_wrap=True, ratio=1)

        for agent_id in sorted(self._agent_states.keys()):
            state = self._agent_states[agent_id]
            stage = state.get("stage", "waiting")
            streaming = state.get("streaming", "")  # "thinking" or "text" or ""
            agent_time = state.get("time", 0.0)
            action = state.get("action", "...")
            thinking_chars = state.get("thinking_chars", 0)
            text_chars = state.get("text_chars", 0)
            total_tokens = (thinking_chars + text_chars) // 4  # approx tokens (chars/4)

            # Color-code stage by streaming sub-state
            if stage == "done":
                stage_style = "green"
            elif stage == "failed":
                stage_style = "bold red"
            elif stage.startswith("retry"):
                stage_style = "red"
            elif streaming == "thinking":
                stage_style = "magenta"
            elif streaming == "text":
                stage_style = "yellow"
            else:
                stage_style = "white"

            # Build preview — single line, show newlines as ↵
            raw_preview = ""
            if streaming == "thinking":
                raw_preview = state.get("thinking_preview", "")
            elif streaming == "text":
                raw_preview = state.get("text_preview", "")
            preview = raw_preview.replace("\n", "↵").replace("\r", "")

            # Token count display
            token_str = str(total_tokens) if total_tokens > 0 else "..."

            # Build retry info string
            retry_count = state.get("retry_count", 0)
            last_error = state.get("last_error_type", "")
            if retry_count > 0:
                retry_str = f"\u00d7{retry_count} {last_error}"
                retry_style = "bold red" if retry_count >= 3 else "yellow"
            else:
                retry_str = ""
                retry_style = "dim"

            table.add_row(
                agent_id,
                Text(stage, style=stage_style),
                token_str,
                f"{agent_time:.1f}" if agent_time > 0 else "...",
                action,
                Text(retry_str, style=retry_style),
                Text(preview, style="dim"),
            )

        # Footer stats
        avg_llm = sum(self._llm_times) / len(self._llm_times) if self._llm_times else 0.0
        footer = Text()
        footer.append(f"[Errors: {self._total_errors}] ", style="red" if self._total_errors > 0 else "green")
        footer.append(f"[Retries: {self._total_retries}] ", style="yellow" if self._total_retries > 0 else "green")
        footer.append(f"[Avg LLM: {avg_llm:.1f}s]", style="blue")
        if self._error_breakdown:
            breakdown_parts = [f"{k}: {v}" for k, v in sorted(self._error_breakdown.items(), key=lambda x: -x[1])]
            footer.append(f" [{' | '.join(breakdown_parts)}]", style="dim yellow")

        return Panel(
            Group(header, table, footer),
            title=f"Simulation Dashboard — {self._run_id}" if self._run_id else "Simulation Dashboard",
            border_style="blue",
        )

    async def _handle_event(self, event: SimulationEvent) -> None:
        """Handle incoming events and update state."""
        data = event.data
        etype = event.event_type

        if etype == EventType.STEP_STARTED:
            self._current_step = data.get("step", 0)
            self._num_agents = data.get("num_agents", 0)
            if not self._run_id:
                self._run_id = data.get("run_id", "")
            self._step_start_time = time.time()
            self._agent_states.clear()
            self._llm_times.clear()
            self._error_breakdown.clear()

        elif etype == EventType.PHASE_STARTED:
            phase = data.get("phase", 0)
            phase_names = {1: "Phase 1: LLM Decisions", 2: "Phase 2: Actions", 3: "Phase 3: Environment"}
            self._current_phase = phase_names.get(phase, f"Phase {phase}")

        elif etype == EventType.PHASE_COMPLETED:
            pass

        elif etype == EventType.AGENT_DECISION_STARTED:
            agent_id = data.get("agent_id", "")
            self._agent_states[agent_id] = {
                "stage": "waiting",
                "streaming": "",
                "start_time": time.time(),
                "time": 0.0,
                "action": "...",
                "thinking_chars": 0,
                "text_chars": 0,
                "thinking_preview": "",
                "text_preview": "",
            }

        elif etype == EventType.AGENT_DECISION_COMPLETED:
            agent_id = data.get("agent_id", "")
            success = data.get("success", False)
            action_type = data.get("action_type", "")
            state = self._agent_states.get(agent_id, {})
            call_time = time.time() - state.get("start_time", time.time())
            self._llm_times.append(call_time)

            if success:
                state["stage"] = "done"
            else:
                state["stage"] = "failed"
                self._total_errors += 1

            state["streaming"] = ""
            state["time"] = call_time
            if action_type:
                state["action"] = action_type

        elif etype == EventType.ACTION_APPLIED:
            agent_id = data.get("agent_id", "")
            action_type = data.get("action_type", "?")
            if agent_id in self._agent_states:
                self._agent_states[agent_id]["action"] = action_type

        elif etype == EventType.RETRY:
            self._total_retries += 1
            agent_id = data.get("agent_id", "")
            error_type = data.get("error_type", "unknown")
            root_cause = data.get("root_cause_hint", "")
            attempt = data.get("attempt", 0)

            # Track per-agent retry info
            if agent_id in self._agent_states:
                state = self._agent_states[agent_id]
                state["retry_count"] = state.get("retry_count", 0) + 1
                state["last_error_type"] = error_type
                state["last_root_cause"] = root_cause

            # Track error type breakdown
            self._error_breakdown[error_type] = self._error_breakdown.get(error_type, 0) + 1

        elif etype == EventType.TOKEN_RECEIVED:
            agent_id = data.get("agent_id", "")
            token_type = data.get("token_type", "")
            text = data.get("text", "")
            state = self._agent_states.get(agent_id)
            if state:
                if token_type == "stage":
                    state["stage"] = text
                    state["streaming"] = ""
                    # Reset token counts for new stage
                    state["thinking_chars"] = 0
                    state["text_chars"] = 0
                    state["thinking_preview"] = ""
                    state["text_preview"] = ""
                elif token_type == "thinking":
                    state["thinking_chars"] = state.get("thinking_chars", 0) + len(text)
                    state["streaming"] = "thinking"
                    prev = state.get("thinking_preview", "")
                    state["thinking_preview"] = (prev + text)[-80:]
                elif token_type == "text":
                    state["text_chars"] = state.get("text_chars", 0) + len(text)
                    state["streaming"] = "text"
                    prev = state.get("text_preview", "")
                    state["text_preview"] = (prev + text)[-80:]

        elif etype == EventType.ERROR:
            self._total_errors += 1

        elif etype == EventType.STEP_COMPLETED:
            self._current_phase = "Completed"

        # Update display
        if self._live and self._enabled:
            self._live.update(self._render())
