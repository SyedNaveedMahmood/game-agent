"""
JSON parsing and validation utilities.
"""

import logging
import re


logger = logging.getLogger(__name__)


def clean_raw_output(raw_output: str) -> str:
    """
    Clean raw LLM output to make it more suitable for JSON parsing.
    
    Args:
        raw_output: Raw string output from the LLM
        
    Returns:
        Cleaned string ready for JSON parsing
    """
    # Remove leading/trailing whitespace
    cleaned = raw_output.strip()
    
    # Remove any markdown code block markers if present
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    
    # Remove any line comments (both // and # style)
    cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'#.*$', '', cleaned, flags=re.MULTILINE)
    
    # Remove block comments
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Remove empty lines
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    # Remove trailing commas in objects and arrays
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Fix common JSON formatting issues
    cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)  # Add quotes to unquoted keys
    cleaned = re.sub(r':\s*\'([^\']*)\'', r':"\1"', cleaned)  # Convert single quotes to double quotes
    
    # Handle Python literals
    cleaned = re.sub(r':\s*None\b', ':null', cleaned)
    cleaned = re.sub(r':\s*True\b', ':true', cleaned)
    cleaned = re.sub(r':\s*False\b', ':false', cleaned)
    
    # Replace colons in strings with dashes
    def replace_colons_in_strings(match):
        content = match.group(1)
        content = content.replace(':', '-')
        return f'"{content}"'

    cleaned = re.sub(r'"([^"]*)"', replace_colons_in_strings, cleaned)
    
    return cleaned






