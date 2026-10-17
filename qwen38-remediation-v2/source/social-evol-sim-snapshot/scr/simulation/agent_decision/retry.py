"""
Retry module for the Morality-AI Simulation.

This module handles retry logic for agent responses.
"""

import time
from typing import Tuple, Optional, Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages
from scr.api.llm_api.completions import get_completions
from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
from scr.simulation.act_manager.validator.validator import validate_llm_response
from scr.utils.logger import get_logger
from scr.models.agent.responses import Response
from .response_processor import process_llm_response, update_agent_memory_from_response
from .message_saver import save_debug_messages
from .retry_tracker import RetryTracker, ValidationResult, classify_root_cause

# Initialize logger
logger = get_logger(__name__)

def process_llm_response_with_retries(
    client: object,
    messages: Messages,
    checkpoint: Checkpoint,
    llm_config: object,
    agent_id: str,
    max_retries: int = 3,
    use_chat_model: bool = False,
    output_type: str = "json",
    retry_tracker: Optional[RetryTracker] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Process an LLM response with automatic retries on failure.
    
    This function:
    1. Attempts to process the LLM response
    2. If it fails, retries up to max_retries times
    3. Returns the success status and either the validated response or an error
    
    Args:
        client (object): The LLM client to use
        messages (Messages): The messages to send to the model
        checkpoint (Checkpoint): The current simulation checkpoint
        llm_config (object): The LLM configuration
        agent_id (str): The ID of the agent
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        use_chat_model (bool, optional): Whether to use the chat model directly. Defaults to False.
        output_type (str, optional): Output format, either "json" or "text". Defaults to "json".
        
    Returns:
        Tuple[bool, Optional[str]]: A tuple containing:
            - A boolean indicating whether the processing was successful
            - Either the validated response or an error message
    """
    # Import locally to avoid circular imports
    from .response_processor import process_llm_response
    
    # Validate required parameters
    if not llm_config.chat_model:
        raise ValueError("chat_model is required but not provided in llm_config")
    
    # Default to max_retries from config if available
    if max_retries is None and hasattr(llm_config, 'max_retries'):
        max_retries = llm_config.max_retries
    
    # Track model information for logging
    has_two_models = llm_config.reasoning_model is not None and llm_config.chat_model is not None
    step = checkpoint.metadata.current_time_step

    def _record_failure(attempt: int, error_type: str, error_message: str, duration: float):
        """Record a retry failure to tracker if available."""
        if not retry_tracker:
            return
        root_cause = classify_root_cause(error_message)
        agent = checkpoint.social_environment.get_agent_by_id(agent_id)
        agent_state = {}
        if agent:
            agent_state = {"hp": getattr(agent.state, "hp", None), "resources": getattr(agent.state, "resources", None)}
        available_targets = [a.id for a in checkpoint.social_environment.agents if a.id != agent_id and a.is_alive()]
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

    # Track retry attempts
    attempts = 0
    last_error = None

    # Retry loop
    while attempts <= max_retries:
        attempts += 1
        attempt_start = time.time()

        try:
            # Process response
            result: ValidationResult = process_llm_response(
                client=client,
                messages=messages,
                checkpoint=checkpoint,
                llm_config=llm_config,
                agent_id=agent_id,
                use_chat_model=use_chat_model,
                output_type=output_type
            )

            # Return successfully if response is valid
            if result.success:
                if attempts > 1:
                    logger.info(f"Successfully processed agent {agent_id} response after {attempts} attempts")

                # Return the response without mutating checkpoint.
                # Caller (step()) is responsible for applying mutations.
                return True, result.response

            # Validation failure — error_type derived from the validation stage
            duration = time.time() - attempt_start
            last_error = result.error_message or "Validation failed"
            _record_failure(attempts, result.error_type, last_error, duration)

        except Exception as e:
            duration = time.time() - attempt_start
            last_error = str(e)
            logger.warning(f"Error on attempt {attempts}/{max_retries+1} for agent {agent_id}: {last_error}")
            _record_failure(attempts, "llm_exception", last_error, duration)

        # Log retry attempt
        if attempts <= max_retries:
            # Format model info for logging
            if has_two_models:
                model_info = f"Reasoning model: {llm_config.reasoning_model}, Chat model: {llm_config.chat_model}"
            else:
                model_info = f"Chat model: {llm_config.chat_model}"
            if attempts == max_retries:
                messages.append("user", "Last chance to generate a valid response. Check parenthesis, brackets etc to not forget any kind of details that make your response invalid. If you fail again, you are killed immediately.")
            logger.info(f"Retrying agent {agent_id} response (attempt {attempts+1}/{max_retries+1}) with {model_info}")
        if attempts > 1:
            save_debug_messages(messages, checkpoint, agent_id, retry_count=attempts)

    # Return failure after max retries
    error_msg = f"Failed to process agent {agent_id} response after {max_retries+1} attempts: {last_error}"
    logger.error(error_msg)
    return False, error_msg