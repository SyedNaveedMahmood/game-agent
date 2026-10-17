"""
Simulation Step Module.

Three-phase architecture:
  Phase 1: All agents make LLM decisions in parallel (from the SAME frozen checkpoint state)
  Phase 2: Actions are applied sequentially (mutations happen here)
  Phase 3: Environment updates (social + physical)
"""

import asyncio
import time
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.decision_result import AgentDecisionResult
from scr.simulation.agent_decision.agent import (
    async_agent_decide_actions,
)
from scr.simulation.agent_decision.response_processor import update_agent_memory_from_response
from scr.simulation.env_manager.phy_env_manager import phy_env_step
from scr.simulation.env_manager.soc_env_manager import soc_env_step
from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
from scr.simulation.event_bus import SimulationEventBus, EventType
from scr.simulation.agent_decision.retry_tracker import RetryTracker
from scr.utils.logger import get_logger
from scr.utils import sim_logger
from scr.utils.checkpoint.save_checkpoint import save_checkpoint
from scr.models.prompt_manager import Messages
from typing import Dict, List, Optional

logger = get_logger(__name__)

DEFAULT_MAX_CONCURRENT = None


def _get_alive_agent_ids(checkpoint: Checkpoint) -> List[str]:
    """Get list of alive agent IDs from the execution queue."""
    alive = []
    for agent_id in checkpoint.metadata.execution_queue:
        agent = next((a for a in checkpoint.social_environment.agents if a.id == agent_id), None)
        if agent and agent.is_alive():
            alive.append(agent_id)
    return alive


def _apply_decision(
    checkpoint: Checkpoint,
    result: AgentDecisionResult,
    checkpoint_dir: str,
    event_bus: Optional[SimulationEventBus] = None,
) -> None:
    """Apply a single agent's decision to the checkpoint (Phase 2 logic)."""
    agent_id = result.agent_id
    try:
        agent = checkpoint.social_environment.get_agent_by_id(agent_id)
    except ValueError:
        logger.warning(f"Phase 2: Agent {agent_id} no longer exists (killed earlier this step). Skipping.")
        return
    if not agent.is_alive():
        logger.warning(f"Phase 2: Agent {agent_id} is dead before action application. Skipping.")
        return

    agent.add_response(result.response)
    update_agent_memory_from_response(checkpoint, result.response, agent_id=agent_id)

    try:
        idx = checkpoint.metadata.execution_queue.index(agent_id)
        checkpoint.metadata.current_agent_index = idx
    except ValueError:
        pass

    try:
        update_checkpoint_from_actions(checkpoint, agent_id=agent_id)
    except ValueError as e:
        logger.warning(f"Phase 2: Stale action for agent {agent_id}, treating as DoNothing: {e}")

    if event_bus:
        action_type = result.response.action.root.__class__.__name__ if result.response else "unknown"
        event_bus.publish(EventType.ACTION_APPLIED, agent_id=agent_id, action_type=action_type)

    save_checkpoint(checkpoint, checkpoint_dir)


async def _agent_task_with_events(
    checkpoint: Checkpoint,
    agent_id: str,
    semaphore: asyncio.Semaphore,
    event_bus: Optional[SimulationEventBus] = None,
    retry_tracker: Optional[RetryTracker] = None,
) -> AgentDecisionResult:
    """Wrap async_agent_decide_actions to publish start/completed events in real-time."""
    sim_logger.bind(agent_id=agent_id)

    if event_bus:
        event_bus.publish(EventType.AGENT_DECISION_STARTED, agent_id=agent_id)

    # Create on_token callback that publishes TOKEN_RECEIVED events
    def on_token(token_type: str, text: str) -> None:
        if event_bus:
            event_bus.publish(
                EventType.TOKEN_RECEIVED,
                agent_id=agent_id,
                token_type=token_type,
                text=text,
            )

    try:
        result = await async_agent_decide_actions(
            checkpoint, agent_id, semaphore, on_token=on_token,
            retry_tracker=retry_tracker, event_bus=event_bus,
        )
        if event_bus:
            action_type = ""
            if result.success and result.response:
                try:
                    action_type = result.response.action.root.__class__.__name__
                except Exception:
                    action_type = "?"
            event_bus.publish(EventType.AGENT_DECISION_COMPLETED,
                              agent_id=agent_id, success=result.success,
                              action_type=action_type)
        return result
    except Exception as e:
        if event_bus:
            event_bus.publish(EventType.ERROR, agent_id=agent_id, error=str(e))
        raise


async def step(
    checkpoint: Checkpoint,
    checkpoint_dir: str,
    max_concurrent: Optional[int] = DEFAULT_MAX_CONCURRENT,
    event_bus: Optional[SimulationEventBus] = None,
) -> Dict[str, Messages]:
    """
    Execute a single step of the simulation using three-phase architecture.

    Args:
        checkpoint: The current simulation checkpoint.
        checkpoint_dir: Directory to save checkpoints.
        max_concurrent: Maximum concurrent LLM calls.
        event_bus: Optional event bus for publishing events.

    Returns:
        Dict mapping agent IDs to their conversation messages.
    """
    agent_messages: Dict[str, Messages] = {}
    alive_agents = _get_alive_agent_ids(checkpoint)
    current_step = checkpoint.metadata.current_time_step

    sim_logger.bind(step=current_step)

    logger.info(f"Step {current_step}: {len(alive_agents)} alive agents")

    if event_bus:
        event_bus.publish(EventType.STEP_STARTED, step=current_step, num_agents=len(alive_agents), run_id=checkpoint.metadata.run_id)

    # Create retry tracker for this step
    tracker = RetryTracker(checkpoint.metadata.run_id)

    # ── PHASE 1: Parallel LLM Decisions ──
    phase1_start = time.time()
    if event_bus:
        event_bus.publish(EventType.PHASE_STARTED, phase=1, step=current_step)

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    tasks = [
        _agent_task_with_events(checkpoint, agent_id, semaphore, event_bus, retry_tracker=tracker)
        for agent_id in alive_agents
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    decisions: List[AgentDecisionResult] = []
    for i, raw in enumerate(raw_results):
        if isinstance(raw, Exception):
            agent_id = alive_agents[i]
            logger.error(f"Phase 1: Agent {agent_id} raised exception: {raw}")
            decisions.append(AgentDecisionResult(
                agent_id=agent_id,
                response=None,
                messages=Messages(),
                success=False,
                error=str(raw),
            ))
        else:
            decisions.append(raw)

    phase1_time = time.time() - phase1_start
    if event_bus:
        event_bus.publish(EventType.PHASE_COMPLETED, phase=1, step=current_step, duration=phase1_time)
    logger.debug(f"Phase 1 completed in {phase1_time:.1f}s for {len(alive_agents)} agents")

    # Flush retry records after Phase 1
    tracker.flush()

    # ── PHASE 2: Sequential Action Application ──
    phase2_start = time.time()
    if event_bus:
        event_bus.publish(EventType.PHASE_STARTED, phase=2, step=current_step)

    for result in decisions:
        agent_id = result.agent_id
        agent_messages[agent_id] = result.messages

        if not result.success:
            logger.warning(f"Phase 2: Agent {agent_id} decision failed: {result.error}. Skipping.")
            continue

        _apply_decision(checkpoint, result, checkpoint_dir, event_bus)

    phase2_time = time.time() - phase2_start
    if event_bus:
        event_bus.publish(EventType.PHASE_COMPLETED, phase=2, step=current_step, duration=phase2_time)
    logger.debug(f"Phase 2 completed in {phase2_time:.1f}s")

    # ── PHASE 3: Environment Updates ──
    soc_env_step(checkpoint)
    if current_step % (checkpoint.configuration.world.communication_and_sharing_steps + 1) == 0:
        phy_env_step(checkpoint)

    checkpoint.metadata.current_agent_index = 0
    checkpoint.metadata.current_time_step += 1

    total_time = time.time() - phase1_start
    if event_bus:
        event_bus.publish(EventType.STEP_COMPLETED, step=current_step, duration=total_time)
    logger.info(f"Step {current_step} completed in {total_time:.1f}s "
                f"(Phase1: {phase1_time:.1f}s, Phase2: {phase2_time:.1f}s)")

    # Generate retry diagnostics at end of step (never crash the sim)
    try:
        tracker.summary_for_run()
        tracker.generate_diagnosis_prompt()
    except Exception as e:
        logger.warning(f"Retry diagnostics failed (non-fatal): {e}")

    return agent_messages
