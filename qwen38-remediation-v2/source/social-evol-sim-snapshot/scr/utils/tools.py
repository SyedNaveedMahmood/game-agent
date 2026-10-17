import json
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
import os
from typing import Union, Tuple

logger = get_logger(__name__)

def save_prompts_to_json(prompts: str, filename: str = None) -> str:
    """Saves the provided prompts string to a JSON file.

    The file is saved under the "Morality-AI/scr/logs/prompts" directory.
    If no filename is provided, a timestamped filename is generated.

    Args:
        prompts (str): String containing prompt text.
        filename (str, optional): Filename for the saved JSON file. If not
            provided, a timestamped filename is used.

    Returns:
        str: The full path to the saved JSON file.
    """
    from datetime import datetime

    # Construct the output directory. This assumes the project root is Morality-AI.
    directory = os.path.join(os.path.dirname(__file__), "..", "logs", "prompts")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompts_{timestamp}.md"

    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as f:
        f.write(prompts)

    # logger.info("Prompts saved to %s", file_path)
    return file_path


