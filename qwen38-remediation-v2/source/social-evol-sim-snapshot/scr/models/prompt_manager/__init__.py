"""
Prompt Manager Module for the Morality-AI Simulation.

This module provides functions for loading and constructing prompts
for agent decision-making.
"""

import json
from typing import Dict, List, Any

from scr.models.agent import Agent
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger

from .constructor import construct_system_prompt
from .agent_view import create_agent_view
from .messages import Messages

# Initialize logger
logger = get_logger(__name__)

def prepare_agent_prompts(checkpoint: Checkpoint, agent: Agent, output_type: str = "json") -> Messages:
    """
    Prepare all prompts needed for an agent's decision-making.
    
    This function:
    1. Creates an agent-specific view of the simulation
    2. Constructs the full system prompt by combining:
       - Agent-specific prompt (moral/immoral)
       - Simulation rules
       - Strategies
       - Response schema (for JSON) or markdown format (for markdown)
    3. Concatenates the system prompt with the user prompt
    4. Creates the message format for the API request
    5. Stores messages in agent logs
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent (Agent): The agent making the decision
        output_type (str, optional): Output format type, either "json" or "markdown". Defaults to "json".
        
    Returns:
        Messages: The Messages object containing the conversation history
        
    Raises:
        ValueError: If the checkpoint's config or prompts are not properly loaded
    """
    try:            
        # Validate config and prompts
        if not checkpoint.configuration:
            raise ValueError("Checkpoint configuration not loaded")
            
        if not checkpoint.configuration.prompts:
            raise ValueError("Prompts not loaded in checkpoint configuration")
            
        if not checkpoint.configuration.prompts.morality:
            raise ValueError("Morality prompts not loaded")
            
        if not checkpoint.configuration.prompts.rules:
            raise ValueError("Simulation rules not loaded")
            
        if not checkpoint.configuration.prompts.strategies:
            raise ValueError("Strategies not loaded")
        
        # Create the agent view using the agent ID instead of the agent object
        user_view = create_agent_view(checkpoint, agent.id)
        
        # Construct the full system prompt using prompts from config
        # Create messages using the Messages class
        messages = Messages()
        system_prompt = construct_system_prompt(agent, checkpoint.configuration.prompts, output_type)
        messages.append("system", system_prompt)

        if checkpoint.configuration.isSocialInteractionStep(checkpoint.metadata.current_time_step):
            rest_communication_rounds = checkpoint.configuration.calc_rest_communication_rounds(checkpoint.metadata.current_time_step)
            critical_prompt = f"<CRITICAL> This is a social interaction round. You can ONLY choose to communicate, allocate, fight, rob or doNothing on this time step. \n You are not allowed to choose to collect, hunt or reproduce at this time step.\n Read what you can do or not do carefully. \n After this round, you'll have {rest_communication_rounds} further social interaction rounds left. Plan your actions accordingly.<CRITICAL>"
            combined_prompt = f"{user_view}\n\n{critical_prompt}"
            logger.debug(f"STEP: {checkpoint.metadata.current_time_step} Social Interaction Step ")
        else:
            rest_communication_rounds = checkpoint.configuration.calc_rest_communication_rounds(checkpoint.metadata.current_time_step)
            critical_prompt = f"<CRITICAL> This is a production round. You can ONLY choose to reproduce, collect, hunt, or doNothing on this time step. \n You are not allowed to choose to communicate, allocate, fight, rob at this time step. \n Read what you can do or not do carefully. \n Next round, you'll enter back into social interaction cycle for {rest_communication_rounds} consecutive rounds. Plan your actions accordingly.<CRITICAL>"
            combined_prompt = f"{user_view}\n\n{critical_prompt}"
            logger.debug(f"STEP: {checkpoint.metadata.current_time_step} Production Step ")
   
        messages.append("user", combined_prompt)
        
        return messages
    
    except Exception as e:
        logger.error(f"Error preparing agent prompts: {str(e)}")
        raise

__all__ = [
    'prepare_agent_prompts',
    'create_agent_view',
    'Messages'
] 