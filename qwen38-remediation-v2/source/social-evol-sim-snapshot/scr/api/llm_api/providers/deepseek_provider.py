from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

from .base import BaseProvider
from .completion_result import CompletionResult
from scr.api.llm_api.config import ProviderConfig

class DeepSeekProvider(BaseProvider):
    """
    Provider implementation for DeepSeek's API.
    
    This class handles communication with DeepSeek's API, providing methods
    for both streaming and non-streaming completions.
    
    Attributes:
        client (OpenAI): The OpenAI client configured for DeepSeek's API
    """
    
    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize the DeepSeek provider.
        
        Args:
            config (ProviderConfig): Configuration containing API key and base URL
        """
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        stream: bool,
        **kwargs: any
    ) -> CompletionResult:
        """
        Get a completion from DeepSeek's API.
        
        Args:
            messages (list[dict[str, str]]): List of message dictionaries
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
            if stream:
                return self._handle_streaming_completion(messages, model, **kwargs)
            return self._handle_regular_completion(messages, model, **kwargs)
        except Exception as e:
            raise Exception(f"Error getting completion from DeepSeek: {str(e)}")

    def _handle_streaming_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: any
    ) -> CompletionResult:
        """
        Handle streaming completion requests.
        
        Args:
            messages (list[dict[str, str]]): List of message dictionaries
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
        messages: list[dict[str, str]],
        model: str,
        **kwargs: any
    ) -> CompletionResult:
        """
        Handle regular (non-streaming) completion requests.
        
        Args:
            messages (list[dict[str, str]]): List of message dictionaries
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
        if message.reasoning_content:
            reasoning = message.reasoning_content
        else:
            reasoning = ""
        return CompletionResult(content=message.content, reasoning=reasoning)