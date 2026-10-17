"""
CLI Run Commands Module.

This module contains functions for executing simulation commands from the CLI.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
from scr.utils.checkpoint.load_checkpoint import get_available_run_ids
from scr.simulation.runner import run_simulation, resume_simulation
from scr.simulation.event_bus import SimulationEventBus
from scr.simulation.dashboard import SimulationDashboard
from scr.utils.logger import get_logger
from scr.utils import sim_logger
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages

logger = get_logger(__name__)


def _get_local_run_ids(data_dir: str = "./data") -> list[str]:
    """Scan data directory for run IDs (directories containing a checkpoints/ sub-dir)."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return []
    return sorted(
        [d.name for d in data_path.iterdir() if d.is_dir() and (d / "checkpoints").is_dir()],
        reverse=True,
    )


def list_available_runs() -> None:
    """List all available simulation runs from the database, with local file fallback."""
    run_ids = get_available_run_ids()

    if run_ids:
        logger.info("Available simulation runs (database):")
        for run_id in run_ids:
            logger.info(f"  - {run_id}")
        return

    # Fallback: scan local checkpoint directories
    local_ids = _get_local_run_ids()
    if local_ids:
        logger.info("Available simulation runs (local checkpoints):")
        for run_id in local_ids:
            logger.info(f"  - {run_id}")
        return

    logger.info("No simulation runs found.")


async def _setup_event_bus(enable_dashboard: bool) -> Tuple[Optional[SimulationEventBus], Optional[SimulationDashboard]]:
    """Create event bus and optionally attach dashboard."""
    event_bus = SimulationEventBus(enabled=True)
    dashboard = None

    if enable_dashboard:
        dashboard = SimulationDashboard(event_bus)
        await dashboard.start()
    else:
        event_bus.subscribe(sim_logger.jsonl_event_bus_subscriber)

    return event_bus, dashboard


async def execute_run_simulation(
    config_path: str,
    checkpoint_dir: str,
    config_overrides: dict = None,
    enable_dashboard: bool = False,
) -> Tuple[Checkpoint, Dict[int, Dict[str, Messages]]]:
    """Execute a new simulation run."""
    config_overrides = config_overrides or {}
    logger.info(f"Starting new simulation with config: {config_path}")

    event_bus, dashboard = await _setup_event_bus(enable_dashboard)

    try:
        final_checkpoint, all_messages = await run_simulation(
            config_path, checkpoint_dir,
            config_overrides=config_overrides,
            event_bus=event_bus,
        )
    finally:
        sim_logger.close()
        if dashboard:
            await dashboard.stop()

    logger.info("Simulation completed successfully")
    return final_checkpoint, all_messages


async def execute_resume_simulation(
    run_id: str,
    checkpoint_dir: str = None,
    config_overrides: dict = None,
    time_step: int = None,
    enable_dashboard: bool = False,
) -> Optional[Tuple[Checkpoint, Dict[int, Dict[str, Messages]]]]:
    """Execute a resumed simulation run."""
    config_overrides = config_overrides or {}
    logger.info(f"Resuming simulation with run_id: {run_id}")

    event_bus, dashboard = await _setup_event_bus(enable_dashboard)

    try:
        result = await resume_simulation(
            run_id, checkpoint_dir,
            config_overrides=config_overrides,
            time_step=time_step,
            event_bus=event_bus,
        )
        logger.info("Resumed simulation completed successfully")
        return result
    except FileNotFoundError as e:
        logger.error(f"Failed to resume simulation: {str(e)}")
        logger.info("Use --list_runs to see available run IDs")
        return None
    finally:
        sim_logger.close()
        if dashboard:
            await dashboard.stop()
