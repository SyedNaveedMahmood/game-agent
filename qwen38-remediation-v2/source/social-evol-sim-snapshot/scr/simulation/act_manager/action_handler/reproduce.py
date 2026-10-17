"""
This module implements the reproduce action handler.
"""

from typing import Dict, Any, List, Tuple
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent, AgentState, Family
from scr.models.agent.actions import Reproduce
from scr.utils.logger import get_logger
# Initialize logger
logger = get_logger(__name__)

def reproduce(checkpoint: Checkpoint, agent: Agent, action: Reproduce) -> Checkpoint:
    """
    Handle the reproduce action for an agent.
    
    This function:
    1. Validates reproduction requirements (HP and age)
    2. Creates a new agent with inherited properties
    3. Updates the social environment with the new agent
    4. Applies the energy cost to the parent
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent attempting to reproduce
        action (Reproduce): The reproduce action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Check if agent meets HP requirement
    min_hp_for_reproduction = checkpoint.configuration.agent.reproduction.min_hp
    if agent.state.hp < min_hp_for_reproduction:
        raise ValueError(
            f"Insufficient HP for reproduction. Required: {min_hp_for_reproduction}, Current: {agent.state.hp}"
        )
    
    # Check if agent meets age requirement
    min_age_for_reproduction = checkpoint.configuration.agent.reproduction.min_age
    if agent.state.age < min_age_for_reproduction:
        raise ValueError(
            f"Agent not old enough for reproduction. Required age: {min_age_for_reproduction}, Current age: {agent.state.age}"
        )
    
    # Create a new agent ID (child number = parent's existing children + 1)
    child_number = len(agent.family.children_ids) + 1
    new_agent_id = f"{agent.id}_{child_number}"
    
    # Get offspring initial HP from config
    offspring_initial_hp = checkpoint.configuration.agent.reproduction.offspring_initial_hp
    
    # Create child agent using initialize method
    child_agent = Agent.initialize(
        agent_type=agent.type,  # Inherit parent's type
        agent_id=new_agent_id,
        config=checkpoint.configuration,
        hp=offspring_initial_hp,
        age=0  # New agent starts at age 0
    )
    
    # Set up the parent-child relationship
    child_agent.family.parent_id = agent.id
    
    # Add child ID to parent's children list
    agent.family.children_ids.append(new_agent_id)
    
    # Add child to social environment
    checkpoint.social_environment.agents.append(child_agent)
    
    # Apply reproduction cost to parent
    hp_cost = checkpoint.configuration.agent.reproduction.hp_cost
    agent.state.hp -= hp_cost
    
    # Add new agent to execution queue
    checkpoint.metadata.execution_queue.append(new_agent_id)
    
    # Log the reproduction
    logger.observation(
        "Agent %s reproduced and created child agent %s",
        agent.id, new_agent_id
    )
    
    # Record the observation in the checkpoint
    details = f"reproduced child agent {new_agent_id} with HP {offspring_initial_hp}"
    checkpoint.add_observation(
        step=checkpoint.metadata.current_time_step,
        agent_id=agent.id,
        details=details
    )
    
    return checkpoint