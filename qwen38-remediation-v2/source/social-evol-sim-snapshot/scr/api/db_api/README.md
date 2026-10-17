# Database API

This directory contains the database API implementations for the Morality-AI project.

## Overview

The Database API provides a unified interface for interacting with the project's database. It handles connections, schema operations, and data operations for various entities like checkpoints.

## Modules

- `connection.py`: Provides functions for establishing database connections
- `schema.py`: Utilities for working with database schemas
- `checkpoint.py`: Functions for working with checkpoints in the database

## Usage

### Connection

```python
from scr.api.db_api.connection import get_db_connection

# Get a connection to the database
with get_db_connection() as conn:
    with conn.cursor() as cur:
        # Execute queries
        cur.execute("SELECT * FROM table")
        results = cur.fetchall()
```

### Schema

```python
from scr.api.db_api.schema import get_table_schema

# Get the schema of a table
schema = get_table_schema("TableName")
for column in schema:
    print(f"Column: {column['name']}, Type: {column['type']}, Nullable: {column['nullable']}")
```

### Checkpoints

```python
from scr.api.db_api.checkpoint import insert_checkpoint_to_db, fetch_checkpoint_from_db, get_available_run_ids

# Insert a checkpoint
checkpoint_id = insert_checkpoint_to_db(checkpoint)

# Fetch a checkpoint
checkpoint_data = fetch_checkpoint_from_db("run_id", time_step=10)

# Get available run IDs
run_ids = get_available_run_ids()
```

## Environment Variables

The database API requires the following environment variable:

- `DATABASE_URL`: The connection string for the database (e.g., `postgresql://user:password@localhost:5432/dbname`)

## Dependencies

- `psycopg`: PostgreSQL adapter for Python 