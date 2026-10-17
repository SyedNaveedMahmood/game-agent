"""
Collect action handler for the simulation.

This module handles the collection of resources by agents.
"""

from typing import Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Collect
from scr.models.environment.plant import PlantNode
from scr.utils.logger import get_logger
import uuid

logger = get_logger(__name__)


def collect(checkpoint: Checkpoint, agent: Agent, action: Collect) -> Checkpoint:
    """
    Handle the collect action for an agent.
    
    This function:
    1. Validates the collection requirements
    2. Ensures there's sufficient quantity available
    3. Adds HP directly to the agent based on the resource's nutrition value
    4. Updates the resource node's quantity
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the collection
        action (Collect): The collect action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate required fields
    if not action.resource_id:
        raise ValueError("Collect action requires a resource_id")
    
    # Check if the requested resource exists
    resources = [
        r for r in checkpoint.physical_environment.resources 
        if r.id == action.resource_id
    ]
    
    if not resources:
        raise ValueError(
            f"Agent '{agent.id}' attempted to collect resource '{action.resource_id}', but it does not exist"
        )
    
    # Get the resource
    resource = resources[0]
        
    # Verify resource is a plant
    if not isinstance(resource, PlantNode):
        raise ValueError(
            f"Resource '{action.resource_id}' is not a plant and cannot be collected directly. Only plants can be collected."
        )
        
    # Check if there's enough of the resource left to collect
    if resource.quantity < action.quantity:
        raise ValueError(
            f"Not enough of resource '{action.resource_id}' to collect (requested: {action.quantity}, available: {resource.quantity})"
        )
    
    # Determine collection amount (limited by what's available)
    max_collect = checkpoint.configuration.agent.max_collect_quantity
    requested_quantity = action.quantity
    available_quantity = resource.quantity
    
    # Ensure we don't collect more than what's available
    collected_quantity = min(requested_quantity, available_quantity, max_collect)
    
    if collected_quantity <= 0:
        raise ValueError(f"No resources available to collect")
    
    # Calculate HP gain from the collected resources
    hp_gain = collected_quantity * resource.nutrition
    
    # Add HP to agent (capped at max_hp)
    agent.state.hp = min(agent.state.max_hp, agent.state.hp + hp_gain)
    
    # Only reduce resource quantity if successfully collected
    resource.quantity -= collected_quantity
    
    # Log the observation
    logger.observation(
        "Agent %s collected %d units of plant for %d HP gain, current HP: %d",
        agent.id, collected_quantity, hp_gain, agent.state.hp
    )
    
    # Record the observation in the checkpoint
    details = f"collected {collected_quantity} plant for {hp_gain} HP gain, current HP: {agent.state.hp}"
    checkpoint.add_observation(
        step=checkpoint.metadata.current_time_step,
        agent_id=agent.id,
        details=details
    )

    return checkpoint