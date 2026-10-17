import re
import json
from typing import Dict, Any, Optional

def extract_agent_view(message_content: str) -> str:
    """
    Extract the agent view from a message content by finding text from 
    "Current time step" to the end of the message.
    
    Args:
        message_content (str): The full message content containing the agent view
        
    Returns:
        str: The extracted agent view, or empty string if not found
    """
    # Use regex to find content starting with "Current time step" to the end
    match = re.search(r'"Current time step".*', message_content, re.DOTALL)
    
    if match:
        return match.group(0)
    return ""

def parse_agent_view(agent_view_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the extracted agent view text into a structured dictionary.
    
    Args:
        agent_view_text (str): The extracted agent view text
        
    Returns:
        Optional[Dict[str, Any]]: Parsed agent view as a dictionary, or None if parsing fails
    """
    try:
        # Extract current time step
        time_step_match = re.search(r'"Current time step":\s*(\d+)', agent_view_text)
        current_time_step = int(time_step_match.group(1)) if time_step_match else None
        
        # Extract JSON blocks
        json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', agent_view_text)
        
        # Parse each block based on the preceding header
        result = {"Current time step": current_time_step}
        
        # Find agent info (first JSON block)
        if len(json_blocks) > 0:
            result["your_info"] = json.loads(json_blocks[0])
        
        # Find personal observations
        personal_obs_match = re.search(r'Personal Observations:\s*```json\s*([\s\S]*?)\s*```', agent_view_text)
        if personal_obs_match and len(json_blocks) > 1:
            result["personal_observations"] = json.loads(json_blocks[1])
        
        # Find general observations
        general_obs_match = re.search(r'General Observations:\s*```json\s*([\s\S]*?)\s*```', agent_view_text)
        if general_obs_match and len(json_blocks) > 2:
            result["general_observations"] = json.loads(json_blocks[2])
        
        # Find observations (physical and social environment)
        if len(json_blocks) > 3:
            result["observations"] = json.loads(json_blocks[3])
            
        return result
    except Exception as e:
        print(f"Error parsing agent view: {e}")
        return None

def extract_and_parse_agent_view(message_content: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the agent view from a message content.
    
    Args:
        message_content (str): The full message content containing the agent view
        
    Returns:
        Optional[Dict[str, Any]]: Parsed agent view as a dictionary, or None if extraction or parsing fails
    """
    agent_view_text = extract_agent_view(message_content)
    if not agent_view_text:
        return None
        
    return parse_agent_view(agent_view_text) 