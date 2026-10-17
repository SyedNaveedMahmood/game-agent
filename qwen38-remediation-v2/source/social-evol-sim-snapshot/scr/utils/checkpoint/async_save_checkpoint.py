"""
Async checkpoint saving utilities.
"""

import aiofiles
from pathlib import Path
from typing import Tuple
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
from scr.api.db_api.checkpoint import insert_checkpoint_to_db
from scr.api.db_api.connection import is_db_disabled

logger = get_logger(__name__)


async def async_save_checkpoint(checkpoint: Checkpoint, output_dir: str = "./data") -> Tuple[str, str]:
    """
    Async version of save_checkpoint. Saves checkpoint to files asynchronously.

    Args:
        checkpoint: The checkpoint to save
        output_dir: Base data directory (checkpoints stored under <output_dir>/<run_id>/checkpoints/)

    Returns:
        Tuple of paths (agent-based, step-based)
    """
    run_id = checkpoint.metadata.run_id
    time_step = checkpoint.metadata.current_time_step
    agent_id = checkpoint.metadata.get_current_agent_id()

    # Structure: data/<run_id>/checkpoints/
    base_path = Path(output_dir)
    run_path = base_path / run_id / "checkpoints"

    agent_path = run_path / "agent-base" / agent_id
    step_path = run_path / "step-base" / f"step_{time_step}"

    agent_path.mkdir(parents=True, exist_ok=True)
    step_path.mkdir(parents=True, exist_ok=True)

    filename = f"checkpoint_t{time_step}_{agent_id}.json"
    agent_output_path = agent_path / filename
    step_output_path = step_path / filename

    checkpoint_json = checkpoint.model_dump_json(indent=2)

    try:
        async with aiofiles.open(str(agent_output_path), "w") as f:
            await f.write(checkpoint_json)
        async with aiofiles.open(str(step_output_path), "w") as f:
            await f.write(checkpoint_json)
        logger.info(f"Async saved checkpoint to {agent_output_path} and {step_output_path}")
    except OSError as e:
        logger.error(f"Failed to async save checkpoint: {str(e)}")
        raise

    # DB insert is sync for now (psycopg doesn't support async without psycopg[async])
    if not is_db_disabled():
        if insert_checkpoint_to_db(checkpoint):
            logger.info(f"Successfully saved checkpoint to database for run {run_id}")
        else:
            logger.warning(f"Failed to save checkpoint to database for run {run_id}")

    return str(agent_output_path), str(step_output_path)
