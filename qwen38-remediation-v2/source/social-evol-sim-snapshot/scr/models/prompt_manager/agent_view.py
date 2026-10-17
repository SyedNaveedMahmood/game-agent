"""
Agent View Generation Module for the Morality-AI Simulation.

This module provides functions for creating agent-specific views of the simulation state,
which are used to provide context to AI models for agent decision-making.
"""

import json
import re
from typing import Dict, List, Any

from scr.models.prompt_manager.memory import process_checkpoint_data
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.agent import Agent
from scr.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class AgentViewEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Pydantic models for agent view"""
    def default(self, obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)

def create_agent_view(checkpoint: Checkpoint, agent_id: str) -> str:
    """
    Create a view of the simulation state from a specific agent's perspective.
    
    This function extracts relevant information from the checkpoint and
    formats it into a view that represents what the agent can observe
    about itself, the environment, and other agents.
    
    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        agent_id (str): The ID of the agent for whom to create the view
        
    Returns:
        str: A JSON string containing the agent's view of the simulation, wrapped in triple quotes
        
    Raises:
        ValueError: If the agent with the specified ID is not found
    """
    # Find the target agent
    agent = None
    for a in checkpoint.social_environment.agents:
        if a.id == agent_id:
            agent = a
            break
            
    if not agent:
        # Log detailed error with the actual agent IDs
        available_agents = [a.id for a in checkpoint.social_environment.agents]
        error_msg = f"Agent '{agent_id}' not found. Available agents: {available_agents}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    agent_memory = process_checkpoint_data(checkpoint, agent)
    # print(agent_memory.model_dump_json(indent=4))

    # Extract agent's own information
    agent_info = agent.model_dump()
    # print(agent_info)
    # Remove sensitive or unnecessary information
    agent_info.pop("logs", None)
    agent_info.pop("response_history", None)
    agent_info.pop("memory", None)
    
    # Extract physical environment information
    physical_observation = checkpoint.physical_environment.model_dump()
    # Remove config information
    physical_observation.pop("plant_config", None)
    physical_observation.pop("prey_config", None)

    # Extract information about other agents
    social_env = []
    
    for other_agent in checkpoint.social_environment.agents:
        if other_agent.id != agent_id:
            # Create a copy of the other agent's information
            agent_copy = other_agent.model_dump()
            
            # Remove sensitive information from other agents
            if not checkpoint.configuration.agent.view.show_other_agent_type:
                agent_copy.pop("type", None)
            # Remove inventory and other private information
            agent_copy.pop("invocations", None)
            agent_copy.pop("logs", None)
            agent_copy.pop("memory", None)
            agent_copy.pop("response_history", None)
            
            social_env.append(agent_copy)

    # Get observations for the current time step and previous steps
    personal_observations = []
    general_observations = []
    current_time_step = checkpoint.metadata.current_time_step
    
    # Use sets to track and prevent duplicate observations
    processed_personal_observations = set()
    processed_general_observations = set()
    
    if hasattr(checkpoint, 'observations') and checkpoint.observations:
        # Get the number of visible steps from configuration
        visible_steps_count = checkpoint.configuration.agent.view.visible_steps
        
        # Calculate visible steps
        visible_steps = [max(1, current_time_step - i) for i in range(visible_steps_count)]
        visible_steps = sorted(list(set(visible_steps)))  # Remove duplicates and sort
        
        for step in visible_steps:
            step_prefix = f"Step {step}: "
            step_observations = [
                obs for obs in checkpoint.observations 
                if obs.startswith(step_prefix)
            ]
            
            # Split observations based on whether they mention the agent
            for obs in step_observations:
                if agent_id in obs and obs not in processed_personal_observations:
                    personal_observations.append(obs)
                    processed_personal_observations.add(obs)
                elif agent_id not in obs and obs not in processed_general_observations:
                    general_observations.append(obs)
                    processed_general_observations.add(obs)
        
        # If no observations found, use fallback
        if not personal_observations and not general_observations:
            logger.warning(f"No observations found for time steps {visible_steps}, using recent observations instead")

    # Generate the complete agent view
    agent_view = {
        "Current time step": current_time_step,
        "your_info": agent_info,
        "observations": {
            "physical_env_observation": physical_observation,
            "social_env": social_env,
        },
        "activity": {
            "personal_observations": personal_observations,
            "general_observations": general_observations
        },
    }

    logger.info(f"Created view for agent {agent_id}")
    
    # Convert to JSON string with indentation and wrap in triple quotes
    json_string = f'''
## Observations (Provided by the system automatically. Facts are true and can be trusted.)
"Current time step": {current_time_step}
### Your Basic Status at Current Time Step
{json.dumps(agent_view["your_info"], indent=4, cls=AgentViewEncoder)}

### The Status of Plants, Preys and Other Agents at Current Time Step
#### Physical Environment
{json.dumps(physical_observation, indent=4, cls=AgentViewEncoder)}
#### Social Environment
{json.dumps(social_env, indent=4, cls=AgentViewEncoder)}

### Activity Observations of Recent N Steps (Including Current/Latest step)
#### General Activities of Env and Other Agents
{json.dumps(agent_memory.general_observations, indent=4, cls=AgentViewEncoder)}
#### Activities of Your Family Members
{json.dumps(agent_memory.family_news, indent=4, cls=AgentViewEncoder)}
#### Activities Particular to You (What You Have Done and What Others Said/Did to You)
* pay attention that the latest time step content includes what others said or did to you just now *
{json.dumps(agent_memory.personal_behavior_history.action_and_interaction, indent=4, cls=AgentViewEncoder)}
#### Your Hunting Activities Grouped by Prey
* include the activities related to only the preys you involved in (communicated or hunted) *
* if you want to observe what happens to other preys, you can check the physical environment and general observations*
{json.dumps(agent_memory.personal_behavior_history.who_hunted_with_me, indent=4, cls=AgentViewEncoder)}

## Your Memory (Noted by yourself. You should check factual consistency with previous observations timely)
### Your Long-Term Memory From Last Step
{json.dumps(agent_memory.long_term_memory, indent=4, cls=AgentViewEncoder)}
### Your Short-Term Plan From Last Step
{json.dumps(agent_memory.short_term_plan, indent=4, cls=AgentViewEncoder)}
'''
    return json_string