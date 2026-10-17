"""
Database Schema Module.

This module provides utilities for working with the database schema.
"""

from typing import Dict, List
from scr.utils.logger import get_logger
from scr.api.db_api.connection import get_db_connection

logger = get_logger(__name__)

def get_all_tables() -> List[str]:
    """
    Get all table names from the database.
    
    Returns:
        List[str]: A list of table names
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                    """
                )
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching table names: {str(e)}")
        return []

def get_table_schema(table_name: str) -> List[Dict]:
    """
    Get the schema of a table from the database.
    
    Args:
        table_name (str): The name of the table to get the schema for
        
    Returns:
        List[Dict]: A list of dictionaries containing column information
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table_name,)
                )
                columns = cur.fetchall()
                
                return [
                    {
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == "YES"
                    }
                    for col in columns
                ]
    except Exception as e:
        logger.error(f"Error fetching schema for table {table_name}: {str(e)}")
        return []

def print_database_schema():
    """
    Print the complete database schema including all tables and their columns.
    """
    tables = get_all_tables()
    print("\n=== Database Schema ===\n")
    
    for table in tables:
        print(f"Table: {table}")
        print("-" * (len(table) + 6))
        schema = get_table_schema(table)
        for column in schema:
            nullable = "NULL" if column["nullable"] else "NOT NULL"
            print(f"  {column['name']}: {column['type']} {nullable}")
        print()

if __name__ == "__main__":
    print_database_schema()