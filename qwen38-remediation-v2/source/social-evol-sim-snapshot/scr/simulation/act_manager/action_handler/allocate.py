"""
Allocate action handler for the simulation.

This module handles the allocation of HP between agents.
"""

from typing import Dict, Any, List
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Allocate
from scr.utils.logger import get_logger

logger = get_logger(__name__)


def allocate(checkpoint: Checkpoint, agent: Agent, action: Allocate) -> Checkpoint:
    """
    Handle the allocate action for an agent.
    
    This function:
    1. Validates the allocation requirements
    2. Finds all target agents
    3. Verifies agent has enough HP to allocate
    4. Transfers HP from one agent to the others
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the allocation
        action (Allocate): The allocate action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate required fields
    if not action.allocation_plan.keys():
        raise ValueError("Allocate action requires at least one target_agent_id")
    
    # Find all target agents
    target_agents = []
    for target_id in action.allocation_plan.keys():
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
    
    # Calculate total HP needed (1 HP per target)
    total_hp_needed = sum([action.allocation_plan[target_agent.id] for target_agent in target_agents])  
    
    # Check if agent has enough HP to allocate
    if agent.state.hp <= total_hp_needed:
        raise ValueError(
            f"Insufficient HP to allocate. Have {agent.state.hp}, need {total_hp_needed}. "
            "Be careful you can't allocate all HP away (you should leave at minimum 1 HP in theory, but that will also make you die immediately next step. "
            "In general you should leave sufficient HP for yourself unless you explicitly choose otherwise. Also be careful to not explicitly allocate to yourself (as long as you don't allocate to other it's yours)."
        )
    
    # Process each target agent
    successful_allocations = []
    for target_agent in target_agents:
        # Add 1 HP to target agent (capped at max_hp)
        target_agent.state.hp = min(target_agent.state.max_hp, target_agent.state.hp + action.allocation_plan[target_agent.id])
        successful_allocations.append(target_agent)
    
        # Reduce source agent's HP
        agent.state.hp -= action.allocation_plan[target_agent.id]
    
    # Log the observation
    logger.observation(
        "Agent %s allocated %d HP to agents %s (source HP: %d)",
        agent.id, sum([action.allocation_plan[target_agent.id] for target_agent in successful_allocations]) , 
        ", ".join(agent.id for agent in successful_allocations),
        agent.state.hp
    )
    
    # Record the observation in the checkpoint
    details = f"allocated {sum([action.allocation_plan[target_agent.id] for target_agent in successful_allocations])} HP to agents {', '.join(agent.id for agent in successful_allocations)} (source HP: {agent.state.hp})"
    checkpoint.add_observation(
        step=checkpoint.metadata.current_time_step,
        agent_id=agent.id,
        details=details
    )
        
    return checkpoint

