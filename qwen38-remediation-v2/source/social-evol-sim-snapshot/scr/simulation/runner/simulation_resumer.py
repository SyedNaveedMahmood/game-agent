"""
Simulation Resumer Module.

This module contains functions for resuming simulations from checkpoints.
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.checkpoint.load_checkpoint import load_checkpoint
from scr.simulation.runner.simulation_step import step
from scr.simulation.runner.stopping_criteria import check_stopping_criteria
from scr.simulation.event_bus import SimulationEventBus
from scr.utils.logger import get_logger, init_run_logger
from scr.utils import sim_logger
from scr.models.prompt_manager import Messages
from scr.api.llm_api.config import PROVIDER_SETTINGS

logger = get_logger(__name__)


async def resume_simulation(
    run_id: str,
    checkpoint_dir: str,
    config_overrides: Optional[dict] = None,
    time_step: Optional[int] = None,
    enable_event_bus: bool = True,
    event_bus: Optional[SimulationEventBus] = None,
) -> Tuple[Checkpoint, Dict[int, Dict[str, Messages]]]:
    """
    Resume a simulation from a checkpoint in the database.

    Args:
        run_id: The run ID to resume from.
        checkpoint_dir: Directory for checkpoints.
        config_overrides: Dict of dotted config keys to override values.
        time_step: Specific time step to resume from.
        enable_event_bus: Whether to enable the event bus.
        event_bus: Optional pre-configured event bus.

    Returns:
        Tuple of (final checkpoint, messages by time step)
    """
    config_overrides = config_overrides or {}

    logger.info(f"Saving checkpoints to {checkpoint_dir}")
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_dir, run_id=run_id, time_step=time_step)
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found for run ID: {run_id}")

    checkpoint.configuration.llm.max_retries = 10

    # Auto-select model when provider is overridden but model is not
    if "llm.provider" in config_overrides and "llm.chat_model" not in config_overrides:
        provider = config_overrides["llm.provider"]
        if provider in PROVIDER_SETTINGS:
            config_overrides["llm.chat_model"] = PROVIDER_SETTINGS[provider]["models"][0]
            logger.info(f"Auto-selected default model for {provider}: {config_overrides['llm.chat_model']}")

    checkpoint.configuration.apply_overrides(config_overrides)

    # Initialise per-run file logging under data/<run_id>/logs/
    run_dir = Path(checkpoint_dir) / checkpoint.metadata.run_id
    init_run_logger(run_dir)

    for key, value in config_overrides.items():
        logger.info(f"Override {key}: {value}")

    sim_logger.bind(run_id=checkpoint.metadata.run_id)

    logger.info(f"Resumed simulation from run ID: {run_id}, time step: {checkpoint.metadata.current_time_step}")
    logger.debug(f"Run config:\n{checkpoint.configuration.model_dump_json(indent=2)}")
    sim_logger.emit(
        "simulation_started", type="lifecycle",
        run_id=checkpoint.metadata.run_id,
        resumed_from_step=checkpoint.metadata.current_time_step,
        num_agents=len(checkpoint.social_environment.agents),
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

    try:
        steps_run = checkpoint.metadata.current_time_step
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

    logger.info(f"Resumed simulation completed after {steps_run} steps")
    sim_logger.emit(
        "simulation_completed", type="lifecycle",
        run_id=checkpoint.metadata.run_id,
        steps_run=steps_run,
        agents_alive=len([a for a in checkpoint.social_environment.agents if a.is_alive()]),
    )
    return checkpoint, all_messages
