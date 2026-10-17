"""
Communicate action handler for the simulation.

This module handles communication between agents.
"""

from typing import Dict, Any, List
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent, Memory
from scr.models.agent.actions import Communicate
from scr.models.environment.plant import PlantNode
from scr.utils.logger import get_logger

logger = get_logger(__name__)

def communicate(checkpoint: Checkpoint, agent: Agent, action: Communicate) -> Checkpoint:
    """
    Handle the communicate action for an agent.
    
    This function:
    1. Validates the communication requirements
    2. Finds all target agents
    3. Records the communication in all agents' memories
    4. Updates the checkpoint with the observation
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the communication
        action (Communicate): The communicate action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate required fields
    if not action.target_agent_ids:
        raise ValueError("Communicate action requires at least one target_agent_id")
    
    if not action.message:
        raise ValueError("Communicate action requires a message")
        
    # Validate message length
    if len(action.message) > 1000:  # Maximum message length
        raise ValueError("Message exceeds maximum length of 1000 characters")
        
    # Validate message content
    if not action.message.strip():
        raise ValueError("Message cannot be empty or contain only whitespace")
    
    # Find all target agents
    target_agents = []
    for target_id in action.target_agent_ids:
        target_agent = next(
            (a for a in checkpoint.social_environment.agents if a.id == target_id),
            None
        )
        
        if not target_agent:
            raise ValueError(f"Target agent '{target_id}' not found")
            
        # Check if target agent is alive
        if not target_agent.is_alive():
            raise ValueError(f"Target agent '{target_id}' is dead")
            
        target_agents.append(target_agent)
    
    # Record the communication in all agents' memories
    for target_agent in target_agents:
        communication_record = {
            "type": "communication",
            "from_agent": agent.id,
            "to_agent": target_agent.id,
            "message": action.message,
            "timestamp": checkpoint.metadata.current_time_step
        }
        
        target_agent.memory.received_messages.append(str(communication_record))
        
        # Log the observation
        logger.observation(
            "Agent %s communicated to agent %s: %s",
            agent.id, target_agent.id, action.message
        )
        
    # Record the observation in the checkpoint
    details = f"communicated to {', '.join([target_agent.id for target_agent in target_agents])}: {action.message}"
    checkpoint.add_observation(
        step=checkpoint.metadata.current_time_step,
        agent_id=agent.id,
        details=details
    )
    
    return checkpoint
