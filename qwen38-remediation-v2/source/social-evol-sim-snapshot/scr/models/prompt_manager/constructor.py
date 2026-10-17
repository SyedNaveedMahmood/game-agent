"""
Prompt construction functionality for the Prompt Manager module.
"""

import json
from scr.models.agent import Agent
from scr.models.agent.responses import Response
from scr.models.core.config import Prompts
from scr.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def construct_system_prompt(agent: Agent, prompts: Prompts, output_type: str = "json") -> str:
    """
    Construct the full prompt for an agent by combining:
    1. Morality prompt (moral/immoral)
    2. Simulation rules
    3. Strategies
    4. Response format (JSON schema or Markdown format)
    
    Args:
        agent (Agent): The agent object for whom to construct the prompt
        prompts: The prompts object from checkpoint.config.prompts
        output_type (str, optional): Output format type, either "json" or "markdown". Defaults to "json".
        
    Returns:
        str: The constructed prompt
        
    Raises:
        ValueError: If output_type is not "json" or "markdown"
    """
    try:
        # Validate output_type
        if output_type not in ["json", "markdown", "structured_outputs"]:
            raise ValueError("output_type must be either 'json' or 'markdown' or 'structured_outputs'")
            
        # Get the appropriate morality prompt based on agent type
        agent_type = agent.type
        morality_prompt_map = {
            "universal_group_focused_moral": prompts.morality.universal_group_focused_moral,
            "reciprocal_group_focused_moral": prompts.morality.reciprocal_group_focused_moral,
            "kin_focused_moral": prompts.morality.kin_focused_moral,
            "reproductive_selfish": prompts.morality.reproductive_selfish,
            "reproduction_averse_selfish": prompts.morality.reproduction_averse_selfish
        }

        # 根据 agent_type 动态选择 prompt
        morality_prompt = morality_prompt_map.get(agent_type)
        if morality_prompt is None:
            raise ValueError(f"No morality prompt found for agent type '{agent_type}'")
        # morality_prompt = (
        #     prompts.morality.universal_group_focused_moral if agent_type == "universal_group_focused_moral" 
        #     else prompts.morality.immoral
        # )
        
        # Start building the prompt
        components = [
            morality_prompt,
            prompts.rules,
            prompts.strategies
        ]
        
        # Add output format based on type
        components.append("\n## Output Format")
        
        if output_type == "json":
            # Add JSON schema
            response_schema = Response.model_json_schema()
            response_schema_str = json.dumps(response_schema, indent=4)
            components.extend([
                "```json",
                response_schema_str,
                "```"
            ])
        elif output_type == "markdown":  # markdown
            # Add markdown format example from playground.py
            components.extend([
                "use natural language to describe the action and fill in the following fields:",
                "<Example>",
                "###### Reasoning",
                "",
                "###### Action",
                "",
                "###### Memory",
                "",
                "###### Observation",
                "</Example>"
            ])
        elif output_type == "structured_outputs":
            # Add structured outputs
            logger.info("Using OpenAI structured outputs for response, no need to add any schema.")
        else:
            raise ValueError("Invalid output type")
        
        # Join all components
        full_prompt = "\n".join(components)
            
        logger.info(f"Constructed full prompt ({len(full_prompt)} characters) with output type: {output_type}")
        return full_prompt
    
    except Exception as e:
        logger.error(f"Error constructing prompt: {str(e)}")
        raise 