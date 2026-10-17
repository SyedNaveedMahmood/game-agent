"""
Database operations for checkpoints.
"""

from typing import Optional, List, Dict
from datetime import datetime
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
from scr.api.db_api.connection import get_db_connection, is_db_disabled

logger = get_logger(__name__)

def insert_checkpoint_to_db(checkpoint: Checkpoint) -> bool:
    """
    Insert a checkpoint into the database.

    Args:
        checkpoint (Checkpoint): The checkpoint to insert

    Returns:
        bool: True if successful, False otherwise
    """
    if is_db_disabled():
        return False
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Insert or update the simulation run with the checkpoint data
                cur.execute(
                    """
                    INSERT INTO "SimulationRun" ("runId", checkpoint)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (
                        checkpoint.metadata.run_id,
                        checkpoint.model_dump_json()
                    )
                )
                conn.commit()
                return True
                
    except Exception as e:
        logger.error(f"Error inserting checkpoint to database: {str(e)}")
        return False

def fetch_checkpoint_from_db(run_id: str, time_step: Optional[int] = None) -> Optional[str]:
    """
    Fetch a checkpoint from the database.

    Args:
        run_id (str): The run ID to fetch
        time_step (Optional[int]): This parameter is kept for backward compatibility but is no longer used

    Returns:
        Optional[str]: The checkpoint data as a JSON string, or None if not found
    """
    if is_db_disabled():
        return None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT checkpoint
                    FROM "SimulationRun"
                    WHERE "runId" = %s AND visible = true
                    ORDER BY "createdAt" DESC
                    LIMIT 1
                    """,
                    (run_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None
                
    except Exception as e:
        logger.error(f"Error fetching checkpoint from database: {str(e)}")
        return None

def get_available_run_ids() -> List[str]:
    """
    Get a list of all available run IDs from the database.

    Returns:
        List[str]: List of run IDs
    """
    if is_db_disabled():
        return []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT "runId" 
                    FROM "SimulationRun" 
                    WHERE visible = true
                    ORDER BY "createdAt" DESC
                    """
                )
                results = cur.fetchall()
                return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Error fetching run IDs from database: {str(e)}")
        return []

def get_checkpoints_by_run_id(run_id: str) -> List[dict]:
    """
    Fetch a list of checkpoints for a simulation, one per unique time step, ordered by time step.

    Args:
        run_id (str): The run ID to fetch checkpoints for

    Returns:
        List[dict]: A list of checkpoint dictionaries, one per unique time step
    """
    if is_db_disabled():
        return []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    WITH ranked_checkpoints AS (
                        SELECT 
                            checkpoint,
                            (checkpoint->'metadata'->>'current_time_step')::int AS time_step,
                            ROW_NUMBER() OVER (PARTITION BY (checkpoint->'metadata'->>'current_time_step')::int ORDER BY "createdAt" DESC) AS rn
                        FROM "SimulationRun"
                        WHERE "runId" = %s
                    )
                    SELECT checkpoint
                    FROM ranked_checkpoints
                    WHERE rn = 1
                    ORDER BY time_step
                    ''',
                    (run_id,)
                )
                results = cur.fetchall()
                return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Error fetching checkpoints for simulation {run_id}: {str(e)}")
        return []


def get_last_checkpoint_by_run_id(run_id: str) -> Optional[dict]:
    """
    Get the most recent checkpoint for a specific run ID as a dictionary.
    Args:
        run_id (str): The run ID to fetch the last checkpoint for
    Returns:
        Optional[dict]: The most recent checkpoint as a dictionary, or None if not found
    """
    if is_db_disabled():
        return None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    '''
                    SELECT checkpoint
                    FROM "SimulationRun"
                    WHERE "runId" = %s AND visible = true
                    ORDER BY "createdAt" DESC
                    LIMIT 1
                    ''',
                    (run_id,)
                )
                result = cur.fetchone()
                if result and result[0]:
                    return result[0]
                return None
    except Exception as e:
        logger.error(f"Error fetching last checkpoint for run ID {run_id}: {str(e)}")
        return None