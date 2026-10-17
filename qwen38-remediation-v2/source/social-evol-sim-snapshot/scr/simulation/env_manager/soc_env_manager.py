"""
Social Environment Manager Module.

This module contains functions for managing the social environment in the simulation.
"""

from typing import List, Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.utils.logger import get_logger
from scr.utils import sim_logger

logger = get_logger(__name__)

def soc_env_step(checkpoint: Checkpoint) -> None:
    """
    Update the social environment for a single step.
    
    This includes:
    1. Processing agent interactions
    2. Updating agent states
    3. Managing agent deaths
    4. Updating the execution queue
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
    """
    # Process agent interactions
    process_interactions(checkpoint)
    
    # Update agent states
    update_agent_states(checkpoint)
    
    # Remove dead agents
    checkpoint.remove_dead_agents("natural_causes")
    
    # Update execution queue
    update_execution_queue(checkpoint)

def process_interactions(checkpoint: Checkpoint) -> None:
    """
    Process interactions between agents.
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
    """
    # TODO: Implement agent interaction logic
    pass

def update_agent_states(checkpoint: Checkpoint) -> None:
    """
    Update the states of all agents.
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
    """
    for agent in checkpoint.social_environment.agents:
        if agent.is_alive() and checkpoint.metadata.current_time_step % (checkpoint.configuration.world.communication_and_sharing_steps + 1) == 0:
            # Update agent state
            agent.state.age += 1
            
            # Get max age from configuration
            max_age = checkpoint.configuration.agent.age.max
            
            # Apply aging effects
            if agent.state.age > max_age:
                agent.state.hp = 0  # Agent dies of old age
                logger.info(f"Agent {agent.id} died of old age at {agent.state.age}")
                sim_logger.emit("agent_died", type="lifecycle", agent_id=agent.id, cause="old_age", age=agent.state.age)
                # Add observation for death by old age
                details = f"died of old age at age {agent.state.age} (max age: {max_age})"
                checkpoint.add_observation(
                    step=checkpoint.metadata.current_time_step,
                    agent_id=agent.id,
                    details=details
                )
            else:
                agent.state.hp -= 1
                # Check if agent died from HP loss
                if agent.state.hp <= 0:
                    logger.info(f"Agent {agent.id} died from HP loss at age {agent.state.age}")
                    sim_logger.emit("agent_died", type="lifecycle", agent_id=agent.id, cause="hp_loss", age=agent.state.age)
                    # Add observation for death by HP loss
                    details = f"died from HP loss at age {agent.state.age}"
                    checkpoint.add_observation(
                        step=checkpoint.metadata.current_time_step,
                        agent_id=agent.id,
                        details=details
                    )
        logger.debug(f"Agent {agent.id} state updated: Age {agent.state.age}, HP {agent.state.hp}")

def update_execution_queue(checkpoint: Checkpoint) -> None:
    """
    Update the execution queue based on agent states.
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
    """
    # Remove dead agents from execution queue
    checkpoint.metadata.execution_queue = [
        agent_id for agent_id in checkpoint.metadata.execution_queue
        if next((a for a in checkpoint.social_environment.agents if a.id == agent_id), None) and 
        next((a for a in checkpoint.social_environment.agents if a.id == agent_id), None).is_alive()
    ]
    
    logger.debug(f"Execution queue: {checkpoint.metadata.execution_queue}")
    
    # Reset current agent index if needed
    if checkpoint.metadata.current_agent_index >= len(checkpoint.metadata.execution_queue):
        checkpoint.metadata.current_agent_index = 0