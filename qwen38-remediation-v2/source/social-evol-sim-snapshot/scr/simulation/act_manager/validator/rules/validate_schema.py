"""
Validate response data against the Response schema using Pydantic validation.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
from pydantic import ValidationError

from scr.models.agent.responses import Response
from scr.models.agent.actions import Collect, Allocate, Fight, Rob, Hunt, Reproduce, Communicate, DoNothing

logger = logging.getLogger(__name__)

def validate_schema(response_data: Dict[str, Any]) -> Tuple[bool, Response, List[str]]:
    """
    Validate response data against the Response schema.

    Args:
        response_data: Dictionary containing response data

    Returns:
        Tuple containing:
        - Success flag
        - Validated Response object
        - List of error messages if any
    """
    try:
        # Ensure the required action classes are available in this scope
        # This prevents "cannot access local variable" errors during validation
        _action_classes = {
            "collect": Collect,
            "allocate": Allocate,
            "fight": Fight,
            "rob": Rob, 
            "hunt": Hunt,
            "reproduce": Reproduce,
            "communicate": Communicate,
            "do_nothing": DoNothing
        }
        
        # Validate the entire response data
        try:
            response = Response.model_validate(response_data)
            logger.debug(f"Schema validation successful for response: {response}")
            return True, response, []
        except ValidationError as e:
            error_msg = "Your response failed to match the expected schema. Please try again. "
            for error in e.errors():
                error_msg += f"Field {error['loc']} - {error['msg']} (error type: {error['type']})"
            logger.error(error_msg)
            return False, None, [error_msg]
            
    except Exception as e:
        error_msg = f"Unexpected error during schema validation: {str(e)}"
        logger.error(error_msg)
        return False, None, [error_msg]