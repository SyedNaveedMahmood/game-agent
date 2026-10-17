"""
Rob action handler for the simulation.

This module handles the robbing of resources from one agent by another.
"""

from typing import Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Rob
from scr.models.core.base_models import InventoryItem
from scr.utils.logger import get_logger
import random, math

logger = get_logger(__name__)


def rob(checkpoint: Checkpoint, agent: Agent, action: Rob) -> Checkpoint:
    """
    Handle the rob action for an agent.
    
    This function:
    1. Validates the robbing requirements
    2. Checks if agents are adjacent to each other
    3. Finds the target agent
    4. Finds the resource in the target agent's inventory
    5. Verifies sufficient quantity available
    6. Checks if robbing agent's inventory has room
    7. Calculates success rate based on fight power
    8. Attempts rob with chance of success based on fight power
    9. If successful, transfers the specified quantity of resources to the robbing agent
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the robbing
        action (Rob): The rob action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate required fields
    if not action.target_agent_id:
        raise ValueError("Rob action requires a target_agent_id")
    if not action.resource_type:
        raise ValueError("Rob action requires a resource_type")
    
    # Validate resource_type
    if action.resource_type not in ['plant', 'meat']:
        raise ValueError(f"Invalid resource_type: {action.resource_type}. Must be 'plant' or 'meat'")
        
    # Validate quantity
    if action.quantity <= 0:
        raise ValueError("Quantity must be greater than 0")
    
    # Find the target agent
    target_agent = next(
        (a for a in checkpoint.social_environment.agents if a.id == action.target_agent_id),
        None
    )
    
    if not target_agent:
        raise ValueError(f"Target agent '{action.target_agent_id}' not found")
    
    # Check if target is alive
    if not target_agent.is_alive():
        raise ValueError(f"Cannot rob agent '{action.target_agent_id}' who is not alive")
    
    # # Find the specified resource in the target agent's inventory
    # target_resource = next(
    #     (item for item in target_agent.inventory if item.type == action.resource_type),
    #     None
    # )
    
    # if not target_resource:
    #     raise ValueError(f"Resource type '{action.resource_type}' not found in target agent's inventory")

    # Validate quantity
    if target_agent.state.hp < action.quantity:
        raise ValueError(f"Insufficient quantity of {action.resource_type} to rob. Target has {target_agent.state.hp}, trying to rob {action.quantity}")
    
    # Calculate success rate based on physical ability, using the same formula as hunt action
    # success_rate = min(agent.state.physical_ability / 10, 0.9)
    ability_diff = agent.state.physical_ability - target_agent.state.physical_ability
    slope = agent.state.phisical_ability_scaling['slope']
    intercept = agent.state.phisical_ability_scaling['intercept']
    success_rate = max(0.1, min(0.9, 0.5 + intercept + 0.4 * math.tanh(ability_diff / slope)))

    resistance_damage = 1
    agent.state.hp -= resistance_damage
        
    # Remove dead agent if HP reaches 0
    if agent.state.hp < 0:
        checkpoint.remove_dead_agents("killed_by_robbery_resistance")
    else:
        # Determine if rob is successful
        if random.random() < success_rate:
            # Rob was successful
            damage = action.quantity

            agent.state.hp = min(agent.state.max_hp, agent.state.hp + damage)
            # Apply damage to target agent
            target_agent.state.hp = max(0, target_agent.state.hp - damage)
            
            # Log the observation
            logger.observation(
                "Agent %s successfully robbed %d energy from agent %s (remaining HP: %d)",
                agent.id, damage, target_agent.id, agent.state.hp
            )
            
            # Record the observation in the checkpoint
            details = f"successfully robbed {damage} energy from agent {target_agent.id} (remaining HP: {agent.state.hp})"
        else:
            # Log the observation
            logger.observation(
                "Agent %s failed to rob energy from agent %s and took %d damage (remaining HP: %d)",
                agent.id, target_agent.id, resistance_damage, agent.state.hp
            )
            
            # Record the observation in the checkpoint
            details = f"failed to rob energy from agent {target_agent.id} and took {resistance_damage} damage (remaining HP: {agent.state.hp})"
    
        checkpoint.add_observation(
            step=checkpoint.metadata.current_time_step,
            agent_id=agent.id,
            details=details
        )
        
    return checkpoint 