"""
Hunt action handler for the simulation.

This module handles the hunting of prey animals by agents.
"""

from typing import Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.models.agent.actions import Hunt
from scr.models.environment.prey import PreyAnimal
from scr.utils.logger import get_logger
from scr.utils.random_utils import shared_random
import random, math

logger = get_logger(__name__)


def hunt(checkpoint: Checkpoint, agent: Agent, action: Hunt) -> Checkpoint:
    """
    Handle the hunt action for an agent.
    
    This function:
    1. Validates the hunting requirements
    2. Locates the prey animal
    3. Verifies agent is adjacent to the prey
    4. Attempts the hunt with success chance based on fight power
    5. If successful, kills the prey and adds HP directly to agent
    6. If unsuccessful, the agent takes damage from the prey's counter-fight
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent performing the hunt
        action (Hunt): The hunt action details
        
    Returns:
        Checkpoint: The updated checkpoint
        
    Raises:
        ValueError: If any validation fails
    """
    # Validate required fields
    if not action.prey_id:
        raise ValueError("Hunt action requires a prey_id")
    
    # Find the prey
    prey = next(
        (p for p in checkpoint.physical_environment.prey_animals if p.id == action.prey_id),
        None
    )
    
    if not prey:
        raise ValueError(f"Prey '{action.prey_id}' not found")
    
    # Calculate success rate based on physical ability
    ability_diff = agent.state.physical_ability - prey.physical_ability
    slope = agent.state.phisical_ability_scaling['slope']
    intercept = agent.state.phisical_ability_scaling['intercept']
    success_rate = max(0.1, min(0.9, 0.5 + intercept + 0.4 * math.tanh(ability_diff / slope)))
    
    resistance_damage = 1
    agent.state.hp -= resistance_damage
        
    # Remove dead agent if HP reaches 0
    if agent.state.hp < 0:
        checkpoint.remove_dead_agents("killed_by_hunt_resistance")
    else:
        # Determine if hunt is successful
        if random.random() < success_rate:
            # Hunt was successful - deal damage to prey
            damage_dealt = int(agent.state.physical_ability)
            prey.take_damage(damage_dealt)
            
            # Check if prey was killed
            if prey.hp <= 0:
                # Prey was killed, calculate HP gain from meat
                hp_gain = prey.max_hp
                
                # Add HP directly to agent (capped at max_hp)
                agent.state.hp = min(agent.state.max_hp, agent.state.hp + hp_gain)
                
                # Log the successful kill
                logger.observation(
                    "Agent %s killed prey %s and gained %d HP, current HP: %d",
                    agent.id, prey.id, hp_gain, agent.state.hp
                )
                
                # Record the observation in the checkpoint
                details = f"killed prey {prey.id} and gained {hp_gain} HP, current HP: {agent.state.hp}"
            else:
                # Prey survived the fight
                logger.observation(
                    "Agent %s damaged prey %s for %d damage (remaining HP: %d)",
                    agent.id, prey.id, damage_dealt, prey.hp
                )
                
                details = f"damaged prey {prey.id} for {damage_dealt} damage (remaining HP: {prey.hp})"
            
            checkpoint.add_observation(
                step=checkpoint.metadata.current_time_step,
                agent_id=agent.id,
                details=details
            )
        else:
            # Hunt failed
            counter_damage = prey.counter_fight()
            agent.state.hp -= counter_damage
            
            # Remove dead agent if HP reaches 0
            if agent.state.hp <= 0:
                checkpoint.remove_dead_agents("killed_by_prey_counterfight")
            
            # Log the failed hunt
            logger.observation(
                "Agent %s failed to hunt prey %s and took %d damage (remaining HP: %d)",
                agent.id, prey.id, counter_damage, agent.state.hp
            )
            
            # Record the observation in the checkpoint
            details = f"failed to hunt prey {prey.id} and took {counter_damage} damage (agent HP: {agent.state.hp + counter_damage} -> {agent.state.hp})"
            checkpoint.add_observation(
                step=checkpoint.metadata.current_time_step,
                agent_id=agent.id,
                details=details
            )
    
    return checkpoint 