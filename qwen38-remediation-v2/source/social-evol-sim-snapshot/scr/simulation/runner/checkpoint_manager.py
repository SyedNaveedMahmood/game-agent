"""
Checkpoint Manager Module.

This module contains functions for managing simulation checkpoints.
"""

import os
from typing import Optional, List
from scr.utils.logger import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = "scr/data/snapshot"

def find_latest_checkpoint(run_id: str) -> Optional[str]:
    """
    Find the latest checkpoint file for a specific run ID.
    
    This function:
    1. Locates the run directory
    2. Finds all step directories
    3. Identifies the latest step directory
    4. Finds all checkpoint files in that directory
    5. Returns the most recently modified checkpoint file
    
    Args:
        run_id (str): The run ID to find checkpoints for.
        
    Returns:
        Optional[str]: Path to the latest checkpoint file, or None if not found.
    """
    run_dir = os.path.join(CHECKPOINT_DIR, f"run-{run_id}")
    if not os.path.exists(run_dir):
        logger.error(f"Run directory not found: {run_dir}")
        return None
    
    # Find all step directories
    step_dirs = []
    for item in os.listdir(run_dir):
        if item.startswith("step-") and os.path.isdir(os.path.join(run_dir, item)):
            step_dirs.append(item)
    
    if not step_dirs:
        logger.error(f"No step directories found in {run_dir}")
        return None
    
    # Sort step directories by step number
    step_dirs.sort(key=lambda x: int(x.split("-")[1]))
    latest_step_dir = os.path.join(run_dir, step_dirs[-1])
    
    # Find all checkpoint files in the latest step directory
    checkpoint_files = []
    for file in os.listdir(latest_step_dir):
        if file.startswith("checkpoint-") and file.endswith(".json"):
            checkpoint_files.append(os.path.join(latest_step_dir, file))
    
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {latest_step_dir}")
        return None
    
    # Return the latest checkpoint file by modification time
    return max(checkpoint_files, key=os.path.getmtime)

def get_all_run_ids() -> List[str]:
    """
    Get a list of all run IDs from the checkpoint directory.
    
    Returns:
        List[str]: List of run IDs.
    """
    if not os.path.exists(CHECKPOINT_DIR):
        logger.error(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return []
    
    run_ids = []
    for item in os.listdir(CHECKPOINT_DIR):
        if item.startswith("run-") and os.path.isdir(os.path.join(CHECKPOINT_DIR, item)):
            run_ids.append(item[4:])  # Remove "run-" prefix
    
    return run_ids 