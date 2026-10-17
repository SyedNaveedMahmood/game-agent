"""
Async retry module for the Morality-AI Simulation.

Async version of retry.py with exponential backoff and per-call timeout.
"""

import asyncio
import copy
import time
from typing import Callable, Optional, Tuple, TYPE_CHECKING
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages
from scr.utils.logger import get_logger
from scr.utils import sim_logger
from scr.utils.async_utils import exponential_backoff_delay, with_timeout
from scr.models.agent.responses import Response
from .async_response_processor import async_process_llm_response
from .message_saver import save_debug_messages
from .retry_tracker import RetryTracker, RetryRecord, ValidationResult, classify_root_cause, reclassify_error
from scr.simulation.event_bus import EventType

if TYPE_CHECKING:
    from scr.simulation.event_bus import SimulationEventBus

logger = get_logger(__name__)


async def async_process_llm_response_with_retries(
    client,
    messages: Messages,
    checkpoint: Checkpoint,
    llm_config,
    agent_id: str,
    max_retries: int = 3,
    use_chat_model: bool = False,
    output_type: str = "json",
    call_timeout: float = 120.0,
    backoff_base: float = 2.0,
    backoff_max: float = 30.0,
    on_token: Optional[Callable] = None,
    retry_tracker: Optional[RetryTracker] = None,
    event_bus: Optional["SimulationEventBus"] = None,
) -> Tuple[bool, Optional[Response]]:
    """
    Async process LLM response with retries, exponential backoff, and timeout.

    Args:
        client: LLM client.
        messages: Messages to send.
        checkpoint: Current checkpoint (read-only).
        llm_config: LLM configuration.
        agent_id: Agent ID.
        max_retries: Max retry attempts.
        use_chat_model: Whether to use chat model.
        output_type: Output format.
        call_timeout: Per-call timeout in seconds.
        backoff_base: Base for exponential backoff.
        backoff_max: Maximum backoff delay.

    Returns:
        (success, response) without mutating the checkpoint.
    """
    if not llm_config.chat_model:
        raise ValueError("chat_model is required but not provided in llm_config")

    if max_retries is None and hasattr(llm_config, "max_retries"):
        max_retries = llm_config.max_retries

    # Read async config if available
    async_config = getattr(llm_config, "async_config", None)
    if async_config:
        call_timeout = getattr(async_config, "call_timeout_seconds", call_timeout)
        backoff_base = getattr(async_config, "retry_backoff_base", backoff_base)
        backoff_max = getattr(async_config, "retry_backoff_max", backoff_max)

    has_two_models = llm_config.reasoning_model is not None and llm_config.chat_model is not None
    step = checkpoint.metadata.current_time_step

    def _record_failure(attempt: int, error_type: str, error_message: str, duration: float):
        """Record a retry failure to tracker and event bus if available."""
        if not retry_tracker:
            return
        # Extract context from checkpoint and messages
        agent = checkpoint.social_environment.get_agent_by_id(agent_id)
        agent_state = {}
        if agent:
            agent_state = {"hp": getattr(agent.state, "hp", None), "resources": getattr(agent.state, "resources", None)}
        available_targets = [a.id for a in checkpoint.social_environment.agents if a.id != agent_id and a.is_alive()]
        llm_raw_output = messages.get_last_assistant_message() if hasattr(messages, "get_last_assistant_message") else ""
        if not llm_raw_output:
            # Fallback: search messages list for last assistant message
            llm_raw_output = ""
            for msg in reversed(messages.messages if hasattr(messages, "messages") else []):
                role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
                if role == "assistant":
                    llm_raw_output = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                    break
        prompt_tail = ""
        for msg in reversed(messages.messages if hasattr(messages, "messages") else []):
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            if role == "user":
                content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                prompt_tail = content[-500:] if content else ""
                break

        # Safety-net reclassification for misclassified errors
        error_type, error_message = reclassify_error(
            error_type, error_message, duration, call_timeout,
            str(llm_raw_output) if llm_raw_output else "",
        )
        root_cause = classify_root_cause(error_message)

        retry_tracker.record(
            agent_id=agent_id,
            step=step,
            attempt=attempt,
            max_attempts=max_retries + 1,
            timestamp=time.time(),
            error_type=error_type,
            error_message=error_message,
            root_cause_hint=root_cause,
            action_type="",
            model=llm_config.chat_model or "",
            two_stage=bool(has_two_models),
            duration_seconds=duration,
            agent_state=agent_state,
            available_targets=available_targets,
            llm_raw_output=str(llm_raw_output)[:1000] if llm_raw_output else "",
            prompt_tail=prompt_tail,
        )
        sim_logger.emit(
            "retry", type="retry",
            agent_id=agent_id, step=step, attempt=attempt,
            error_type=error_type, error_message=error_message,
            root_cause_hint=root_cause, duration_seconds=duration,
            model=llm_config.chat_model or "",
        )
        if event_bus:
            event_bus.publish(
                EventType.RETRY,
                agent_id=agent_id,
                step=step,
                attempt=attempt,
                error_type=error_type,
                error_message=error_message,
                root_cause_hint=root_cause,
            )

    attempts = 0
    last_error = None
    original_messages = copy.deepcopy(messages.messages)

    while attempts <= max_retries:
        if attempts and getattr(llm_config, "provider", None) == "llama_cpp":
            # Retain every original observation/memory. Superseded drafts and
            # reflection turns are retry artifacts, not new factual evidence.
            # Keep only bounded latest error feedback between two-stage cycles.
            last_chance = next((m.get("content", "") for m in reversed(messages.messages)
                                if isinstance(m, dict) and m.get("role") == "user"
                                and str(m.get("content", "")).startswith("Last chance to generate a valid response.")), "")
            messages.messages[:] = copy.deepcopy(original_messages)
            feedback = "Your previous response was invalid. Please try again. Error: " + str(last_error)[:2000]
            if last_chance:
                feedback += "\n" + last_chance
            messages.append("user", feedback)
        attempts += 1
        attempt_start = time.time()

        try:
            coro = async_process_llm_response(
                client=client,
                messages=messages,
                checkpoint=checkpoint,
                llm_config=llm_config,
                agent_id=agent_id,
                use_chat_model=use_chat_model,
                output_type=output_type,
                on_token=on_token,
            )

            result: ValidationResult = await with_timeout(
                coro,
                timeout_seconds=call_timeout,
                operation_name=f"LLM call for agent {agent_id} (attempt {attempts})",
            )

            if result.success:
                if attempts > 1:
                    logger.info(f"Successfully processed agent {agent_id} response after {attempts} attempts")
                return True, result.response

            # Validation failure — error_type derived from the validation stage
            duration = time.time() - attempt_start
            last_error = result.error_message or "Validation failed"
            _record_failure(attempts, result.error_type, last_error, duration)

        except asyncio.TimeoutError:
            duration = time.time() - attempt_start
            last_error = f"Timeout after {call_timeout}s"
            logger.warning(f"Timeout on attempt {attempts}/{max_retries+1} for agent {agent_id}")
            _record_failure(attempts, "timeout", last_error, duration)

        except Exception as e:
            duration = time.time() - attempt_start
            last_error = str(e) or f"{type(e).__name__}: no details available"
            logger.warning(f"Error on attempt {attempts}/{max_retries+1} for agent {agent_id}: {last_error}")
            _record_failure(attempts, "llm_exception", last_error, duration)

        # Retry logic
        if attempts <= max_retries:
            if has_two_models:
                model_info = f"Reasoning model: {llm_config.reasoning_model}, Chat model: {llm_config.chat_model}"
            else:
                model_info = f"Chat model: {llm_config.chat_model}"

            if attempts == max_retries:
                messages.append(
                    "user",
                    "Last chance to generate a valid response. Check parenthesis, brackets etc "
                    "to not forget any kind of details that make your response invalid. "
                    "If you fail again, you are killed immediately.",
                )

            # Exponential backoff before retry
            delay = exponential_backoff_delay(attempts - 1, base=backoff_base, max_delay=backoff_max)
            logger.info(
                f"Retrying agent {agent_id} response (attempt {attempts+1}/{max_retries+1}) "
                f"with {model_info} after {delay:.1f}s backoff"
            )
            if on_token:
                on_token("stage", f"retry #{attempts}")
            await asyncio.sleep(delay)

        if attempts > 1:
            save_debug_messages(messages, checkpoint, agent_id, retry_count=attempts)

    error_msg = f"Failed to process agent {agent_id} response after {max_retries+1} attempts: {last_error}"
    logger.error(error_msg)
    return False, error_msg
