"""
DoNothing Action Handler Module.

This module contains the handler for the DoNothing action, which allows an agent
to skip their turn and simply observe the environment.
"""

from typing import Optional
from scr.models.agent import Agent
from scr.models.agent.actions import DoNothing
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger

logger = get_logger(__name__)

def do_nothing(checkpoint: Checkpoint, agent: Agent, action: DoNothing) -> Checkpoint:
    """
    Handle a DoNothing action where the agent chooses to skip their turn.
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the action
        action (DoNothing): The action being performed
    
    Returns:
        Checkpoint: The updated simulation checkpoint
    """
        
    # Log the observation
    logger.observation(
        "Agent %s %s",
        agent.id, "chose to do nothing"
    )
    
    # Record the observation in the checkpoint
    checkpoint.add_observation(
        step=checkpoint.metadata.current_time_step,
        agent_id=agent.id,
        details="chose to do nothing"
    )
    
    return checkpoint 