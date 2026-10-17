from openai import AzureOpenAI
from typing import Dict, List, Any
from .base import BaseProvider
from .completion_result import CompletionResult
from scr.api.llm_api.config import ProviderConfig

class TongyuanProvider(BaseProvider):
    """
    Provider implementation for Tongyuan.
    
    This class handles communication with Tongyuan's API, providing methods
    for both streaming and non-streaming completions.
    
    Attributes:
        client (OpenAI): The OpenAI client instance
    """
    
    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize the Tongyuan provider.
        
        Args:
            config (ProviderConfig): Configuration containing API key and base URL
        """
        super().__init__(config)
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version="2025-03-01-preview",
            azure_endpoint=config.base_url
        )

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs: Any
    ) -> CompletionResult:
        """
        Get a completion from Tongyuan's API.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            model (str): Model to use for completion
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            CompletionResult: The completion response containing content and optional reasoning
            
        Raises:
            ValueError: If the model is not supported
            Exception: For any API-related errors
        """
        try:
            # if stream:
            #     return self._handle_streaming_completion(messages, model, **kwargs)
            return self._handle_regular_completion(messages, model, **kwargs)
        except Exception as e:
            raise Exception(f"Error getting completion from Tongyuan: {str(e)}")

    def _handle_streaming_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any
    ) -> CompletionResult:
        """
        Handle streaming completion requests.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            model (str): Model to use
            **kwargs: Additional arguments
            
        Returns:
            CompletionResult: The complete response containing content and optional reasoning
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            **kwargs
        )

        content = ""
        reasoning = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                    print(delta.content, end="", flush=True)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning += delta.reasoning_content
                    print(delta.reasoning_content, end="", flush=True)
        print("")
        return CompletionResult(content=content, reasoning=reasoning)

    def _handle_regular_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any
    ) -> CompletionResult:
        """
        Handle regular (non-streaming) completion requests.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            model (str): Model to use
            **kwargs: Additional arguments
            
        Returns:
            CompletionResult: The completion response containing content and optional reasoning
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
            **kwargs
        )
        message = response.choices[0].message
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning = message.reasoning_content
        else:
            reasoning = ""
        print(message.content)
        return CompletionResult(content=message.content, reasoning=reasoning) 