"""
Event bus for decoupling simulation logic from monitoring/logging.

Uses asyncio.Queue for fire-and-forget event publishing.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable
from scr.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    AGENT_DECISION_STARTED = "agent_decision_started"
    AGENT_DECISION_COMPLETED = "agent_decision_completed"
    LLM_CALL_COMPLETED = "llm_call_completed"
    RETRY = "retry"
    ACTION_APPLIED = "action_applied"
    ERROR = "error"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    TOKEN_RECEIVED = "token_received"


@dataclass
class SimulationEvent:
    """A simulation event."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


# Type alias for subscriber callbacks
Subscriber = Callable[[SimulationEvent], Awaitable[None]]


class SimulationEventBus:
    """
    Async event bus using asyncio.Queue.

    Supports publish/subscribe pattern with fire-and-forget publishing.
    """

    def __init__(self, enabled: bool = True, max_queue_size: int = 1000):
        self.enabled = enabled
        self._subscribers: List[Subscriber] = []
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._dispatch_task: Optional[asyncio.Task] = None

    def subscribe(self, callback: Subscriber) -> None:
        """Register a subscriber callback."""
        self._subscribers.append(callback)

    def publish(self, event_type: EventType, **data) -> None:
        """
        Fire-and-forget event publishing.

        Non-blocking: uses put_nowait. If queue is full, logs warning and drops.
        """
        if not self.enabled:
            return

        event = SimulationEvent(event_type=event_type, data=data)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Event bus queue full, dropping event: {event_type}")

    async def start(self) -> None:
        """Start the event dispatch loop."""
        if not self.enabled or self._running:
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop the event dispatch loop and drain remaining events."""
        self._running = False
        if self._dispatch_task:
            # Put a sentinel to unblock the dispatch loop
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            await self._dispatch_task
            self._dispatch_task = None

    async def _dispatch_loop(self) -> None:
        """Internal loop that dispatches events to subscribers."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if event is None:  # sentinel
                break

            for subscriber in self._subscribers:
                try:
                    await subscriber(event)
                except Exception as e:
                    logger.error(f"Event subscriber error: {e}")

        # Drain remaining events
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                if event is None:
                    continue
                for subscriber in self._subscribers:
                    try:
                        await subscriber(event)
                    except Exception:
                        pass
            except asyncio.QueueEmpty:
                break


