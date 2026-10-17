from abc import ABC, abstractmethod
from typing import Dict, List
from scr.api.llm_api.config import ProviderConfig
from .completion_result import CompletionResult

class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs: any
    ) -> CompletionResult:
        """
        Get a completion from the provider.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            model (str): Model to use for completion
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            CompletionResult: The completion response containing content and optional reasoning
        """
        pass