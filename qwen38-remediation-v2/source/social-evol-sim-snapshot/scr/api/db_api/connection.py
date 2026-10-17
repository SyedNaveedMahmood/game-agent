"""
Database Connection Module.

This module provides functions for connecting to the database.
"""

import os
import psycopg
from typing import Optional
from scr.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level flag to disable all database operations
_db_disabled = False


def disable_db() -> None:
    """Disable all database operations for this process."""
    global _db_disabled
    _db_disabled = True


def is_db_disabled() -> bool:
    """Return True if database operations are disabled."""
    return _db_disabled


def get_db_connection(db_url: Optional[str] = None):
    """
    Get a connection to the database.

    Args:
        db_url (str, optional): The database URL. If not provided, the environment
            variable 'DATABASE_URL' is used.

    Returns:
        psycopg.Connection: A database connection, or None if DB is disabled

    Raises:
        ValueError: If no database URL is provided and not found in environment variables
    """
    if _db_disabled:
        return None

    if not db_url:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL not provided and not found in ENV variables.")
            raise ValueError("DATABASE_URL not provided and not found in ENV variables.")

    try:
        return psycopg.connect(db_url)
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise