"""
Checkpoint saving utilities.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
from scr.api.db_api.checkpoint import insert_checkpoint_to_db
from scr.api.db_api.connection import is_db_disabled

logger = get_logger(__name__)

def save_checkpoint(checkpoint: Checkpoint, output_dir: str = "./data") -> Tuple[str, str]:
    """
    Save a checkpoint to a file and optionally to the database.
    Saves in both agent-based and step-based directory structures.
    
    Args:
        checkpoint (Checkpoint): The checkpoint to save
        output_dir (str): Base directory to save the checkpoint file
        agent_id (str): ID of the agent for the checkpoint
        
    Returns:
        Tuple[str, str]: Paths to the saved checkpoint files (agent-based, step-based)
        
    Raises:
        OSError: If there's an error creating directories or writing files
    """
    run_id = checkpoint.metadata.run_id
    time_step = checkpoint.metadata.current_time_step
    agent_id = checkpoint.metadata.get_current_agent_id()
    # Convert to Path objects for better path handling
    # Structure: data/<run_id>/checkpoints/
    base_path = Path(output_dir)
    run_path = base_path / run_id / "checkpoints"
    
    # Create base run directory
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Define paths for both structures
    agent_path = run_path / "agent-base" / agent_id
    step_path = run_path / "step-base" / f"step_{time_step}"
    
    # Create directories
    agent_path.mkdir(parents=True, exist_ok=True)
    step_path.mkdir(parents=True, exist_ok=True)
    
    # Generate consistent filenames
    filename = f"checkpoint_t{time_step}_{agent_id}.json"
    agent_output_path = agent_path / filename
    step_output_path = step_path / filename
    
    # Serialize checkpoint once
    checkpoint_json = checkpoint.model_dump_json(indent=2)
    
    # Save to both locations with error handling
    try:
        agent_output_path.write_text(checkpoint_json)
        step_output_path.write_text(checkpoint_json)
        logger.info(f"Successfully saved checkpoint to {agent_output_path} and {step_output_path}")
    except OSError as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        raise
    
    # Insert to database only when all agents in the current step have been processed
    if not is_db_disabled():
        if insert_checkpoint_to_db(checkpoint):
            logger.info(f"Successfully saved checkpoint to database for run {run_id}")
        else:
            logger.warning(f"Failed to save checkpoint to database for run {run_id}")

    return str(agent_output_path), str(step_output_path)