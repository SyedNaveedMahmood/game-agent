"""
Physical environment manager for the simulation.

This module handles updates to the physical environment during each simulation step,
including plant growth and resource regeneration.
"""

from scr.models.environment.physical import PlantNode
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger

logger = get_logger(__name__)

def phy_env_step(checkpoint: Checkpoint):
    """
    Update resources in the physical environment for each step.
    
    This function:
    1. Processes plant growth and regrowth
    2. Handles resource decay
    3. Manages plant respawning
    4. Handles prey respawning
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
    """
    # Track plants that need respawning
    plants_to_respawn = []
    
    for resource in checkpoint.physical_environment.resources:
        if isinstance(resource, PlantNode):
            # Track the state before growth for logging
            old_quantity = resource.quantity
            old_depletion_turn = resource.depletion_turn
            
            # Advance the plant's growth with current turn
            resource.advance_growth(current_turn=checkpoint.metadata.current_time_step)
            
            # Log growth if it occurred
            if resource.quantity > old_quantity:
                logger.observation(f"Plant {resource.id} grew from {old_quantity} to {resource.quantity} units")
            
            # Log death if it just occurred (not already logged and not ready to respawn)
            if resource.is_dead and not resource.has_logged_death and resource.depletion_turn != -1:
                logger.observation(f"Plant {resource.id} has died and will respawn in {resource.respawn_delay} turns")
                resource.has_logged_death = True
                
            # Check if plant is ready to respawn (depletion_turn = -1 means ready to respawn)
            if resource.is_dead and resource.depletion_turn == -1:
                plants_to_respawn.append(resource)
    
    # Handle respawning
    for plant in plants_to_respawn:
        # Update plant state
        plant.quantity = plant.capacity  # Start with full capacity
        plant.depletion_turn = -1  # Reset depletion turn
        plant.has_logged_death = False  # Reset death logging for next time
        
        # Add respawn observation
        details = f"Plant {plant.id} respawned with {plant.quantity} units"
        checkpoint.add_observation(
            step=checkpoint.metadata.current_time_step,
            agent_id="environment",  # Use "environment" as the agent ID for environment events
            details=details
        )
        logger.observation(f"Plant {plant.id} respawned with {plant.quantity} units")
    
    # Handle prey respawning
    try:
        old_prey_count = len(checkpoint.physical_environment.prey_animals)
        checkpoint.physical_environment.spawn_new_prey()
        new_prey_count = len(checkpoint.physical_environment.prey_animals)
        
        if new_prey_count > old_prey_count:
            # A new prey was spawned
            if not checkpoint.physical_environment.prey_animals:
                logger.warning("Prey list is empty after spawn attempt")
                return checkpoint
                
            new_prey = checkpoint.physical_environment.prey_animals[-1]  # Get the last added prey
            # Create detailed prey information string
            prey_details = (
                f"Prey {new_prey.id} spawned "
                f"with HP={new_prey.hp}, physical_ability={new_prey.physical_ability}, "
                f"reward_hp={new_prey.get_reward_hp()}, "
                f"num_agents_to_kill={new_prey.num_agents_to_kill}"
            )
            checkpoint.add_observation(
                step=checkpoint.metadata.current_time_step,
                agent_id="environment",
                details=prey_details
            )
            logger.observation(
                f"Prey {new_prey.id} spawned with HP={new_prey.hp}, "
                f"physical_ability={new_prey.physical_ability}, "
                f"reward_hp={new_prey.get_reward_hp()}, "
                f"num_agents_to_kill={new_prey.num_agents_to_kill}"
            )
    except Exception as e:
        logger.error(f"Error during prey spawn: {str(e)}")
    
    logger.info("Physical environment updated")
    return checkpoint
