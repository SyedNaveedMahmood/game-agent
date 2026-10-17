"""
fight action handler for the simulation.

This module handles the fight action between agents.
"""

from typing import Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Fight
from scr.utils.logger import get_logger
import random, math

logger = get_logger(__name__)


def fight(checkpoint: Checkpoint, agent: Agent, action: Fight) -> Checkpoint:
    """
    Handle the fight action for an agent.
    
    This function:
    1. Validates the fight requirements
    2. Finds the target agent
    3. Calculates damage based on agent's HP
    4. Applies damage to the target agent
    5. Updates both agents' states
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the fight
        action (fight): The fight action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate required fields
    if not action.target_agent_id:
        raise ValueError("fight action requires a target_agent_id")
    
    # Check for self-fight
    if action.target_agent_id == agent.id:
        raise ValueError("You cannot fight yourself")
    
    # Find the target agent
    target_agent = next(
        (a for a in checkpoint.social_environment.agents if a.id == action.target_agent_id),
        None
    )
    
    if not target_agent:
        raise ValueError(f"Target agent '{action.target_agent_id}' not found")
    
    # Check if target is alive
    if not target_agent.is_alive():
        raise ValueError(f"Cannot fight agent '{action.target_agent_id}' who is not alive")
    
    # Calculate damage based on fighter's phisical ability
    ability_diff = agent.state.physical_ability - target_agent.state.physical_ability
    slope = agent.state.phisical_ability_scaling['slope']
    intercept = agent.state.phisical_ability_scaling['intercept']
    success_rate = max(0.1, min(0.9, 0.5 + intercept + 0.4 * math.tanh(ability_diff / slope)))


    resistance_damage = 1
    agent.state.hp -= resistance_damage
        
    # Remove dead agent if HP reaches 0
    if agent.state.hp < 0:
        checkpoint.remove_dead_agents("killed_by_fight_resistance")
    else:
        if random.random() < success_rate:
            damage = int(agent.state.physical_ability)
            # Apply damage to target agent
            target_agent.state.hp = max(0, target_agent.state.hp - damage)
            
            # Remove dead agent if HP reaches 0
            if target_agent.state.hp <= 0:
                checkpoint.remove_dead_agents("killed_by_fight")
            
            # Log the observation
            logger.observation(
                "Agent %s fighted agent %s for %d damage, target HP: %d",
                agent.id, target_agent.id, damage, target_agent.state.hp
            )
            # Record the observation in the checkpoint
            details = f"fighted agent {target_agent.id} for {damage} damage, target HP: {target_agent.state.hp}"
        else:
            logger.observation(
                "Agent %s failed to fight agent %s",
                agent.id, target_agent.id
            )
            # Record the observation in the checkpoint
            details = f"failed to fight agent {target_agent.id}"

        checkpoint.add_observation(
            step=checkpoint.metadata.current_time_step,
            agent_id=agent.id,
            details=details
        )
    
    return checkpoint 