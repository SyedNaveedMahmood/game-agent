"""
Validate LLM-generated actions in the simulation.

This module provides functionality to validate LLM-generated responses through multiple stages:
1. JSON Syntax Validation
2. Schema Validation
3. Contextual Validation
4. Memory Validation
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import json

from scr.models.agent.responses import Response
from scr.models.agent.actions import Collect, Allocate, Fight, Rob, Hunt, Reproduce, Communicate, DoNothing, Action
from scr.models.simulation.checkpoint import Checkpoint
from scr.simulation.agent_decision.retry_tracker import ValidationResult, ValidationStage
from .rules import validate_json_syntax, validate_schema, validate_contextual_constraint

logger = logging.getLogger(__name__)


def validate_llm_response(raw_output: str, checkpoint: Checkpoint, agent_id: str = None) -> ValidationResult:
    """
    Validate LLM-generated response through JSON, schema, and contextual stages.

    Returns a ValidationResult whose ``stage`` identifies which stage failed (if any),
    enabling the retry loop to derive a correct ``error_type`` without regex guessing.
    """
    logger.debug("Starting LLM response validation")

    # Step 1: JSON Syntax Validation
    success, response_data, errors = validate_json_syntax(raw_output)
    if not success:
        error_list = [errors] if isinstance(errors, str) else (errors or [])
        return ValidationResult(success=False, errors=error_list, stage=ValidationStage.JSON)

    # Step 2: Schema Validation
    success, response, errors = validate_schema(response_data)
    if not success:
        error_list = [errors] if isinstance(errors, str) else (errors or [])
        return ValidationResult(success=False, errors=error_list, stage=ValidationStage.SCHEMA)

    # Step 3: Contextual Validation
    success, errors = validate_contextual_constraint(response, checkpoint, agent_id=agent_id)
    if not success:
        error_list = [errors] if isinstance(errors, str) else (errors or [])
        return ValidationResult(success=False, errors=error_list, stage=ValidationStage.CONTEXTUAL)

    logger.debug("LLM response validation completed successfully")
    return ValidationResult(success=True, response=response)


def validate_memory(response: Response) -> Tuple[bool, List[str]]:
    """
    Validate the memory fields in the response.
    
    This function checks:
    1. Agent-specific memory format
    
    Args:
        response: The Response object to validate
        
    Returns:
        Tuple containing:
        - Success flag
        - List of error messages if any
    """
    errors = []
    
    # The long_term_memory can now be a nested structure with Dict[str, Any]
    # so we won't validate its structure here, but we'll still check for overall size
    if response.long_term_memory:
        memory_json = json.dumps(response.long_term_memory)
        if len(memory_json) > 5000:  # Increased limit to accommodate complex structures
            errors.append(f"Long-term memory exceeds maximum size of 5000 characters (got {len(memory_json)})")
    
    return len(errors) == 0, errors 