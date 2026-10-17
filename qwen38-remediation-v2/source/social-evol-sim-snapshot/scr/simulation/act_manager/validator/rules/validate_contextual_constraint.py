"""
Main validator class that manages validation rules using action handlers.
"""

import logging
from typing import Dict, Any, List, Tuple
from copy import deepcopy
import random
import numpy as np
from scr.utils.random_utils import shared_random

from scr.models.agent.responses import Response
from scr.models.simulation.checkpoint import Checkpoint
from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
from scr.models.agent.actions import Collect, Allocate, Fight, Rob, Hunt, Reproduce, Communicate, DoNothing


logger = logging.getLogger(__name__)


    
def validate_contextual_constraint(response: Response, checkpoint: Checkpoint, agent_id: str = None) -> Tuple[bool, List[str]]:
    """
    Validate action using the appropriate action handler.

    This function validates an action by:
    1. Checking if the current agent exists
    2. Validating the action's context (e.g., if the target exists)
    3. Attempting to execute the action in a copy of the checkpoint

    Args:
        response: The response to validate
        checkpoint: The current simulation checkpoint
        agent_id: The agent whose action to validate. If None, reads from metadata (legacy).

    Returns:
        Tuple containing:
        - Success flag
        - List of error messages if any
    """
    # Create a deep copy of the checkpoint for validation
    validation_checkpoint = deepcopy(checkpoint)

    # Determine the agent ID: use explicit param, fall back to metadata
    if agent_id is not None:
        current_agent_id = agent_id
        # Sync the metadata index so update_checkpoint_from_actions uses the right agent
        try:
            idx = validation_checkpoint.metadata.execution_queue.index(agent_id)
            validation_checkpoint.metadata.current_agent_index = idx
        except ValueError:
            error_msg = f"Agent {agent_id} not found in execution queue"
            logger.error(error_msg)
            return False, [error_msg]
    else:
        current_agent_id = validation_checkpoint.metadata.get_current_agent_id()
        if not current_agent_id:
            error_msg = "No current agent ID found in checkpoint metadata"
            logger.error(error_msg)
            return False, [error_msg]

    current_agent = next(
        (agent for agent in validation_checkpoint.social_environment.agents
            if agent.id == current_agent_id),
        None
    )

    if not current_agent:
        error_msg = f"Current agent {current_agent_id} not found in checkpoint"
        logger.error(error_msg)
        return False, [error_msg]

    # Validate agent state
    if current_agent.state.hp <= 0:
        error_msg = f"Agent {current_agent_id} is dead and cannot perform actions"
        logger.error(error_msg)
        return False, [error_msg]

    if checkpoint.configuration.isSocialInteractionStep(checkpoint.metadata.current_time_step):
        if not isinstance(response.action.root, (Communicate, Allocate, Fight, Rob, DoNothing)):
            error_msg = f"Agent {current_agent_id} can only choose to communicate, allocate, fight, rob or doNothing on this time step"
            logger.error(error_msg)
            return False, [error_msg]
    else:
        if not isinstance(response.action.root, (Reproduce, Collect, Hunt, DoNothing)):
            error_msg = f"Agent {current_agent_id} can only choose to reproduce, collect, hunt or doNothing on this time step"
            logger.error(error_msg)
            return False, [error_msg]

    current_agent.add_response(response)

    # Executing the copied action can still consume process-global RNGs. A
    # read-only legality check must not alter the later real environment draw.
    python_rng = random.getstate()
    environment_rng = shared_random.getstate()
    numpy_rng = np.random.get_state()
    try:
        # Try to execute the action in the validation checkpoint
        update_checkpoint_from_actions(validation_checkpoint, agent_id=current_agent_id)
        logger.debug(f"Action validated successfully: {response.action}")
        return True, []
    except ValueError as e:
        error_msg = str(e)
        logger.warning(f"Action validation failed: {error_msg}")
        return False, [error_msg]
    except Exception as e:
        error_msg = f"Unexpected error during validation: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]
    finally:
        random.setstate(python_rng)
        shared_random.setstate(environment_rng)
        np.random.set_state(numpy_rng)
