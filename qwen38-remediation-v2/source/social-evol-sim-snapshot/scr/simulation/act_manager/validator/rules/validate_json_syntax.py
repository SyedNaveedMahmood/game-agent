"""
Parse raw LLM output into a dictionary and perform basic JSON validation.
"""

import json
import logging
from typing import Dict, Any, Tuple, Optional

from scr.simulation.act_manager.validator.utils import clean_raw_output

logger = logging.getLogger(__name__)

def validate_json_syntax(raw_output: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse raw LLM output into a dictionary and perform basic JSON validation.

    This function:
    1. Cleans the raw output by removing markdown, comments, and other non-JSON content
    2. Attempts to parse the cleaned JSON
    3. Validates the basic structure

    Args:
        raw_output: Raw string output from the LLM

    Returns:
        Tuple containing:
        - Success flag
        - Parsed data if successful, None otherwise
        - Error message if parsing failed, None otherwise
    """
    try:
        # Clean the raw output
        # cleaned_output = clean_raw_output(raw_output)
        cleaned_output = raw_output
        # First try to parse the JSON
        parsed = json.loads(cleaned_output)
        
        # Basic JSON structure validation
        if not isinstance(parsed, dict):
            error_msg = "JSON must be an object (dictionary)"
            logger.error(error_msg)
            return False, None, error_msg
            
        # Validate required fields if present
        if not parsed:
            error_msg = "JSON object cannot be empty"
            logger.error(error_msg)
            return False, None, error_msg
            
        return True, parsed, None
    except json.JSONDecodeError as e:
        # Try to provide more helpful error messages
        error_msg = f"Invalid JSON syntax: {str(e)}"
        if "Expecting property name" in str(e):
            error_msg += "\n💡 Tip: Make sure all property names are in double quotes"
        elif "Expecting value" in str(e):
            error_msg += "\n💡 Tip: Check for missing values or trailing commas"
        elif "Extra data" in str(e):
            error_msg += "\n💡 Tip: There might be extra content after the JSON object"
        elif "Expecting ',' delimiter" in str(e):
            error_msg += "\n💡 Tip: Don't include any colon within a string"
        logger.error(error_msg)
        return False, None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during JSON parsing: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg