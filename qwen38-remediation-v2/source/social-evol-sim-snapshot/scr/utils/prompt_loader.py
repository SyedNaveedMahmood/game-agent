import json
from scr.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def load_prompts(prompt_path: str):
    """
    Loads the prompts from a JSON or text file.
    
    Args:
        prompt_path (str): Path to the prompt file
        
    Returns:
        str or dict: The loaded prompt content
        
    Raises:
        ValueError: If the file format is not supported
    """
    try:
        with open(prompt_path, 'r') as f:
            if prompt_path.endswith('.json'):
                return json.load(f)
            elif prompt_path.endswith('.txt'):
                return f.read()
            else:
                raise ValueError("Unsupported file format. Only .json and .txt are supported.")
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_path}: {e}")
        raise 