"""
Client initialization module for the Morality-AI Simulation.

This module handles the initialization of AI clients for agent decision-making.
"""

from scr.api.llm_api.client import get_client
from scr.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def initialize_llm_client(llm_config: object) -> object:
    """
    Initialize an LLM client based on the provided configuration.

    Args:
        llm_config (object): The LLM configuration object

    Returns:
        object: An initialized LLM client

    Raises:
        RuntimeError: If client initialization fails
    """
    try:
        return get_client(llm_config.provider)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM client: {str(e)}")
