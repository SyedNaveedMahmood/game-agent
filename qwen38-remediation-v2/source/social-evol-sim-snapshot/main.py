"""
Main module for running the Resource and Morality Simulation.

This module serves as the entry point for the simulation, handling command-line arguments
and delegating to the appropriate components.
"""

# Load .env file before any other imports
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()

from scr.simulation.cli import (
    parse_cli_args,
    list_available_runs,
    execute_run_simulation,
    execute_resume_simulation,
)
from scr.api.db_api.connection import disable_db
from scr.utils.logger import get_logger, set_global_log_level, suppress_console_logging
from scr.utils.token_estimator import format_estimate
import logging
from typing import Dict, Tuple, Optional
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages

import sys

logger = get_logger(__name__)

# Suppress console logging before anything else when dashboard is active
if "--dashboard" in sys.argv:
    suppress_console_logging()


async def main() -> Optional[Tuple[Checkpoint, Dict[int, Dict[str, Messages]]]]:
    """
    Main function to parse arguments and run the simulation.

    Returns:
        Optional[Tuple[Checkpoint, Dict[int, Dict[str, Messages]]]]: If running a simulation,
            returns a tuple of (final checkpoint, messages by time step).
            If just listing runs, returns None.
    """
    args, config_overrides = parse_cli_args()

    enable_dashboard = args.get("dashboard", False)

    # Disable DB if explicitly requested or DATABASE_URL is not set
    if args.get("no_db") or not os.environ.get("DATABASE_URL"):
        disable_db()
        if args.get("no_db"):
            logger.info("Database disabled via --no_db flag (file-only checkpoints)")
        else:
            logger.info("DATABASE_URL not set, running without database (file-only checkpoints)")

    # Set logging level if specified
    if args["log_level"]:
        log_level = getattr(logging, args["log_level"].upper())
        set_global_log_level(log_level)
        logger.info(f"Log level set to {args['log_level'].upper()}")

    # Log loaded API keys (after dashboard suppression so it only goes to file)
    api_keys = [k for k in os.environ.keys() if k.endswith('_API_KEY')]
    logger.info(f"Loaded environment variables: {', '.join(api_keys)}")

    result = None
    mode = args["mode"]

    if mode == "estimate-cost":
        from scr.models.core.config import Config

        config = Config.load_from_dir(args["config_dir"])
        config.apply_overrides(config_overrides)
        print(format_estimate(
            steps=config.world.max_life_steps,
            agents=config.agent.initial_count,
            two_stage=config.llm.two_stage_model,
            visible_window=config.agent.view.visible_steps,
        ))
        return None

    if mode == "list-runs":
        list_available_runs()
    elif mode == "resume":
        result = await execute_resume_simulation(
            args["run_id"],
            args["checkpoint_dir"],
            config_overrides,
            time_step=args["time_step"],
            enable_dashboard=enable_dashboard,
        )
    elif mode == "run":
        result = await execute_run_simulation(
            args["config_dir"],
            args["checkpoint_dir"],
            config_overrides,
            enable_dashboard=enable_dashboard,
        )

    return result


if __name__ == "__main__":
    asyncio.run(main())
