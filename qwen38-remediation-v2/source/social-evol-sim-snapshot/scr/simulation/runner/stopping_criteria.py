"""
Stopping Criteria Module.

This module contains functions for determining when a simulation should stop.
"""

from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
from scr.utils import sim_logger

logger = get_logger(__name__)

def check_stopping_criteria(checkpoint: Checkpoint) -> bool:
    """
    Check if any stopping criteria are met for the simulation.
    
    Current stopping criteria include:
    1. Maximum time steps reached
    2. All agents are dead
    
    Args:
        checkpoint (Checkpoint): The current checkpoint.
        
    Returns:
        bool: True if any stopping criteria are met, False otherwise.
    """
    # Check if all agents are dead
    if len(checkpoint.social_environment.agents) == 0:
        logger.info("Stopping criteria met: All agents are dead.")
        sim_logger.emit("stopping_criteria_met", type="lifecycle", criteria="all_agents_dead")
        return True
    
    # Check if there's a specific termination condition in the configuration
    if (checkpoint.configuration and 
        checkpoint.configuration.termination_conditions and 
        hasattr(checkpoint.configuration.termination_conditions, 'end_if_all_agents_dead') and
        checkpoint.configuration.termination_conditions.end_if_all_agents_dead):
        
        # Count agents by type
        moral_agents = sum(1 for agent in checkpoint.social_environment.agents if agent.type == "moral")
        immoral_agents = sum(1 for agent in checkpoint.social_environment.agents if agent.type == "immoral")
        
        # Check if one type of agent is extinct
        if moral_agents == 0:
            logger.info("Stopping criteria met: All moral agents are dead.")
            sim_logger.emit("stopping_criteria_met", type="lifecycle", criteria="all_moral_agents_dead")
            return True
        if immoral_agents == 0:
            logger.info("Stopping criteria met: All immoral agents are dead.")
            sim_logger.emit("stopping_criteria_met", type="lifecycle", criteria="all_immoral_agents_dead")
            return True
    
    return False 