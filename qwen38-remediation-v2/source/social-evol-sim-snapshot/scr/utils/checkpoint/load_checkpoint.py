# scr/utils/checkpoint/load_checkpoint.py
"""
Checkpoint Loading Module.

This module provides functions for loading checkpoints from files or the database.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import glob

from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
from scr.api.db_api.checkpoint import fetch_checkpoint_from_db, get_available_run_ids

logger = get_logger(__name__)

# Default checkpoint directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "checkpoints")

def load_checkpoint(checkpoint_path: str = None, run_id: str = None, time_step: int = None):
    """
    Load a checkpoint from file or database.
    
    Args:
        checkpoint_path (str, optional): Path to the checkpoint file
        run_id (str, optional): Run ID to fetch checkpoint from database. Used if checkpoint_path is None.
        time_step (int, optional): Specific time step to fetch from database
        
    Returns:
        Checkpoint: The loaded checkpoint
        
    Raises:
        ValueError: If database connection fails
    """
    # If run_id is provided, try to load from database
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        step_base_path = os.path.join(checkpoint_path, run_id, "checkpoints", "step-base")
        if not os.path.exists(step_base_path):
            raise FileNotFoundError(f"Path not found: {step_base_path}")

        # 找到所有 step_XX 文件夹，按数字排序
        step_dirs = [d for d in os.listdir(step_base_path) if d.startswith("step_")]
        if not step_dirs:
            raise FileNotFoundError("No step directories found in step-base")

        # 提取 step 数字进行排序
        step_dirs_sorted = sorted(step_dirs, key=lambda x: int(x.split("_")[1]))
        latest_step_dir = os.path.join(step_base_path, step_dirs_sorted[-1])

        # 找到该目录下最新（按修改时间）生成的 .json 文件
        json_files = glob.glob(os.path.join(latest_step_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {latest_step_dir}")

        latest_json = max(json_files, key=os.path.getmtime)
        return load_checkpoint_from_file(latest_json)
    
    if run_id:
        checkpoint_data = fetch_checkpoint_from_db(run_id, time_step)
        if checkpoint_data:
            try:
                # # Convert dictionary to JSON string if needed
                # if isinstance(checkpoint_data, dict):
                #     checkpoint_data = json.dumps(checkpoint_data)
                return Checkpoint(**checkpoint_data)
            except Exception as e:
                logger.error(f"Error parsing checkpoint data from database: {str(e)}")
                # Fall through to file loading if database data is invalid
        
        logger.info(f"Loaded checkpoint for run_id '{run_id}' from database")
    
    # If no specific path or run_id, or if database fetch failed, find the newest file
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint()
    
    # Load from file
    return load_checkpoint_from_file(checkpoint_path)

def load_checkpoint_from_file(checkpoint_path: str):
    """
    Load a checkpoint from a file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        Checkpoint: The loaded checkpoint
    """
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = f.read()
        checkpoint_data = json.loads(checkpoint_data)
        return Checkpoint(**checkpoint_data)
    except Exception as e:
        logger.error(f"Error loading checkpoint from file {checkpoint_path}: {str(e)}")
        raise

def find_latest_checkpoint() -> str:
    """
    Find the latest checkpoint file in the checkpoint directory.
    
    Returns:
        str: Path to the latest checkpoint file
    """
    checkpoint_files = []
    for root, _, files in os.walk(CHECKPOINT_DIR):
        for file in files:
            if file.startswith("checkpoint-") and file.endswith(".json"):
                checkpoint_files.append(os.path.join(root, file))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
    
    return checkpoint_path