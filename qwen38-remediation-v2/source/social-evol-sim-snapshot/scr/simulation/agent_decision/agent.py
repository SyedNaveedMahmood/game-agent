"""
Main agent decision module for the Morality-AI Simulation.

This module handles the main flow of agent decision-making in the simulation.
"""

import asyncio
import os
from typing import Callable, Optional, Tuple

from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import prepare_agent_prompts, Messages
from scr.models.agent.decision_result import AgentDecisionResult
from scr.utils.logger import get_logger
from .client import initialize_llm_client
from .retry import process_llm_response_with_retries
from .async_retry import async_process_llm_response_with_retries

# Initialize logger
logger = get_logger(__name__)

def agent_decide_actions(checkpoint: Checkpoint, agent_id: str, retry_tracker=None) -> AgentDecisionResult:
    """
    Process an agent's decision-making turn in the simulation.

    This is now a pure decision function: it returns an AgentDecisionResult
    without mutating the checkpoint. The caller is responsible for applying
    mutations (add_response, update_memory, update_actions).

    Args:
        checkpoint (Checkpoint): The current simulation state (read-only during decision)
        agent_id (str): The ID of the agent to decide for

    Returns:
        AgentDecisionResult: The decision result containing response and messages
    """
    agent = checkpoint.social_environment.get_agent_by_id(agent_id)
    if not agent:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=None,
            messages=Messages(),
            success=False,
            error=f"Agent {agent_id} not found",
        )

    # Initialize AI client and prepare messages
    llm_config = checkpoint.configuration.llm
    client = initialize_llm_client(llm_config)

    # Determine output_type based on whether reasoning_model is available
    if llm_config.provider == "openai":
        output_type = "json"
    elif llm_config.reasoning_model:
        output_type = "markdown"
    else:
        output_type = "json"
    logger.debug(f"Using output_type '{output_type}' for agent {agent_id}")

    messages = prepare_agent_prompts(checkpoint=checkpoint, agent=agent, output_type=output_type)

    # Process AI response with retries
    success, result = process_llm_response_with_retries(
        client=client,
        messages=messages,
        checkpoint=checkpoint,
        llm_config=llm_config,
        agent_id=agent_id,
        max_retries=llm_config.max_retries,
        output_type=output_type,
        retry_tracker=retry_tracker,
    )

    if not success:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=None,
            messages=messages,
            success=False,
            error=str(result),
        )

    return AgentDecisionResult(
        agent_id=agent_id,
        response=result,
        messages=messages,
        success=True,
    )


async def async_agent_decide_actions(
    checkpoint: Checkpoint,
    agent_id: str,
    semaphore: Optional[asyncio.Semaphore] = None,
    on_token: Optional[Callable] = None,
    retry_tracker=None,
    event_bus=None,
) -> AgentDecisionResult:
    """
    Async version of agent_decide_actions for parallel LLM calls.

    Uses asyncio.Semaphore to limit concurrent LLM calls.

    Args:
        checkpoint: The simulation state (read-only during Phase 1).
        agent_id: The agent to decide for.
        semaphore: Optional semaphore to limit concurrency.

    Returns:
        AgentDecisionResult
    """
    agent = checkpoint.social_environment.get_agent_by_id(agent_id)
    if not agent:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=None,
            messages=Messages(),
            success=False,
            error=f"Agent {agent_id} not found",
        )

    llm_config = checkpoint.configuration.llm
    client = initialize_llm_client(llm_config)

    if llm_config.provider == "openai":
        output_type = "json"
    elif llm_config.reasoning_model:
        output_type = "markdown"
    else:
        output_type = "json"
    logger.debug(f"[async] Using output_type '{output_type}' for agent {agent_id}")

    messages = prepare_agent_prompts(checkpoint=checkpoint, agent=agent, output_type=output_type)

    async def _call():
        return await async_process_llm_response_with_retries(
            client=client,
            messages=messages,
            checkpoint=checkpoint,
            llm_config=llm_config,
            agent_id=agent_id,
            max_retries=llm_config.max_retries,
            output_type=output_type,
            on_token=on_token,
            retry_tracker=retry_tracker,
            event_bus=event_bus,
        )

    try:
        if semaphore:
            async with semaphore:
                success, result = await _call()
        else:
            success, result = await _call()
    except Exception as e:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=None,
            messages=messages,
            success=False,
            error=str(e),
        )

    if not success:
        return AgentDecisionResult(
            agent_id=agent_id,
            response=None,
            messages=messages,
            success=False,
            error=str(result),
        )

    return AgentDecisionResult(
        agent_id=agent_id,
        response=result,
        messages=messages,
        success=True,
    )
