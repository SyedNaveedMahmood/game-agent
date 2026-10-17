"""
Simulation Runner Module.

This module contains functions for running simulations from start to finish.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
from scr.models.simulation.checkpoint import Checkpoint
from scr.simulation.runner.simulation_step import step
from scr.simulation.runner.stopping_criteria import check_stopping_criteria
from scr.simulation.event_bus import SimulationEventBus
from scr.utils.logger import get_logger, init_run_logger
from scr.utils import sim_logger
from scr.models.prompt_manager import Messages
from scr.api.llm_api.config import PROVIDER_SETTINGS

logger = get_logger(__name__)


async def run_simulation(
    config_dir: str,
    checkpoint_dir: str,
    run_id: Optional[str] = None,
    config_overrides: Optional[dict] = None,
    enable_event_bus: bool = True,
    event_bus: Optional[SimulationEventBus] = None,
) -> Tuple[Checkpoint, Dict[int, Dict[str, Messages]]]:
    """
    Run a simulation from start to finish.

    Args:
        config_dir: Name of the config directory to use
        checkpoint_dir: Directory to save checkpoints
        run_id: ID for this simulation run
        config_overrides: Dict of dotted config keys to override values
        enable_event_bus: Whether to enable the event bus
        event_bus: Optional pre-configured event bus

    Returns:
        Tuple of (final checkpoint, messages by time step)
    """
    config_overrides = config_overrides or {}

    # Auto-select model when provider is overridden but model is not
    if "llm.provider" in config_overrides and "llm.chat_model" not in config_overrides:
        provider = config_overrides["llm.provider"]
        if provider in PROVIDER_SETTINGS:
            config_overrides["llm.chat_model"] = PROVIDER_SETTINGS[provider]["models"][0]
            logger.info(f"Auto-selected default model for {provider}: {config_overrides['llm.chat_model']}")

    # Generate the initial checkpoint with overrides applied BEFORE
    # environments/agents are created (so initial_count etc. take effect)
    checkpoint = Checkpoint.initialize_from_config(config_dir, config_overrides)

    for key, value in config_overrides.items():
        logger.info(f"Override {key}: {value}")

    # Initialise per-run file logging under data/<run_id>/logs/
    run_dir = Path(checkpoint_dir) / checkpoint.metadata.run_id
    init_run_logger(run_dir)

    sim_logger.bind(run_id=checkpoint.metadata.run_id)

    logger.info(f"Starting simulation with run_id: {checkpoint.metadata.run_id}")
    logger.debug(f"Run config:\n{checkpoint.configuration.model_dump_json(indent=2)}")
    sim_logger.emit(
        "simulation_started", type="lifecycle",
        run_id=checkpoint.metadata.run_id,
        config_dir=config_dir,
        num_agents=len(checkpoint.social_environment.agents),
        max_concurrent=checkpoint.configuration.llm.async_config.max_concurrent_calls,
    )

    max_concurrent = checkpoint.configuration.llm.async_config.max_concurrent_calls

    num_steps = checkpoint.configuration.world.max_life_steps * (checkpoint.configuration.world.communication_and_sharing_steps + 1)

    # Set up event bus
    if event_bus is None and enable_event_bus:
        event_bus = SimulationEventBus(enabled=True)
        event_bus.subscribe(sim_logger.jsonl_event_bus_subscriber)

    if event_bus:
        await event_bus.start()

    all_messages: Dict[int, Dict[str, Messages]] = {}

    logger.info(f"Configured to run for {num_steps} steps (max_concurrent: {max_concurrent})")

    try:
        steps_run = 0
        while steps_run < num_steps:
            if check_stopping_criteria(checkpoint):
                logger.info("Stopping criteria met. Ending simulation.")
                break

            current_step = checkpoint.metadata.current_time_step
            step_messages = await step(
                checkpoint, checkpoint_dir,
                max_concurrent=max_concurrent,
                event_bus=event_bus,
            )
            all_messages[current_step] = step_messages

            steps_run += 1
            logger.info(f"Completed step {current_step}, {steps_run}/{num_steps} steps run")
    finally:
        if event_bus:
            await event_bus.stop()

    logger.info(f"Simulation completed after {steps_run} steps")
    sim_logger.emit(
        "simulation_completed", type="lifecycle",
        run_id=checkpoint.metadata.run_id,
        steps_run=steps_run,
        agents_alive=len([a for a in checkpoint.social_environment.agents if a.is_alive()]),
    )
    return checkpoint, all_messages
