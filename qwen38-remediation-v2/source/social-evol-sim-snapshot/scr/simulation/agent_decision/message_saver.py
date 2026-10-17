"""
Message saving module for the Morality-AI Simulation.

This module handles saving failed messages for debugging purposes.
"""

from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages
from scr.utils.logger import get_logger
import os
from pathlib import Path

# Initialize logger
logger = get_logger(__name__)

def save_debug_messages(messages: Messages, checkpoint: Checkpoint, agent_id: str, retry_count: int = 0) -> None:
    """
    Save failed messages to a file for debugging, using the same structure as checkpoint saving.
    
    Args:
        messages (Messages): The conversation messages
        checkpoint (Checkpoint): The current simulation state
        agent_id (str): The ID of the agent
        retry_count (int): The number of retries attempted (default: 0)
    """
    metadata = checkpoint.metadata
    run_id = metadata.run_id
    current_time_step = metadata.current_time_step

    # Base directory for failed messages: data/<run_id>/debug/failed_messages/
    base_dir = Path("data") / run_id / "debug" / "failed_messages"

    # Create subdirectory for retry count if applicable
    if retry_count > 0:
        base_dir = base_dir / f"retry_{retry_count}"

    base_dir.mkdir(parents=True, exist_ok=True)

    # Save messages directly (not via save_with_checkpoint_structure since
    # the run_id is already encoded in the directory path)
    try:
        step_dir = base_dir / f"step_{current_time_step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        filename = f"messages_t{current_time_step}_{agent_id}"
        messages.save(filename=filename, output_dir=str(step_dir))
        
        logger.info(f"Failed messages saved to {step_dir}")
    except Exception as e:
        logger.error(f"Failed to save debug messages: {str(e)}")

        # Fallback to regular save method
        fallback_dir = str(base_dir)
        os.makedirs(fallback_dir, exist_ok=True)

        messages.save(
            filename=f"messages_{current_time_step}_{agent_id}",
            output_dir=fallback_dir
        )
        logger.warning(f"Used fallback method to save messages to {fallback_dir}") 