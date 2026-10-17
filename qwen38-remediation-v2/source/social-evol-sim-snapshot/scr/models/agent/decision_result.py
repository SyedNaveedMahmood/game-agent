"""Agent decision result dataclass for decoupled decision-making."""

from dataclasses import dataclass, field
from typing import Optional
from scr.models.agent.responses import Response
from scr.models.prompt_manager.messages import Messages


@dataclass
class AgentDecisionResult:
    """Result of an agent's decision-making phase (pure, no side effects)."""
    agent_id: str
    response: Optional[Response]  # validated parsed response
    messages: Messages            # conversation history for saving
    success: bool
    error: Optional[str] = None
