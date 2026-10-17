from typing import Dict, List
from openai import OpenAI
import requests
import json

from scr.api.llm_api.config import ProviderConfig
from .base import BaseProvider
from .completion_result import CompletionResult

class AlibabaProvider(BaseProvider):
    """Provider for Alibaba Cloud's Qwen-Plus model."""

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
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs: any
    ) -> CompletionResult:
        """
        Get a completion from Alibaba Cloud's Qwen-Plus model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries
            model (str): Model to use for completion (must be "qwen-plus")
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            CompletionResult: The completion response containing content and optional reasoning
        """
    
        try:
            if stream:
                return self._handle_streaming_completion(messages, model, **kwargs)
            return self._handle_regular_completion(messages, model, **kwargs)
        except Exception as e:
            raise Exception(f"Error getting completion from Alibaba Cloud: {str(e)}")

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
        headers: Dict[str, str],
        data: Dict
    ) -> CompletionResult:
        """
        Handle regular (non-streaming) completion requests.
        
        Args:
            headers (Dict[str, str]): Request headers
            data (Dict): Request data
            
        Returns:
            CompletionResult: The completion response containing content and optional reasoning
        """
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Extract the completion content from the response
        try:
            content = result["choices"][0]["message"]["content"]
        except KeyError:
            # If the expected structure is not found, try the documented structure
            try:
                content = result["output"]["choices"][0]["message"]["content"]
            except KeyError as e:
                raise Exception(f"Unexpected response structure: {result}") from e
        
        # Create and return the completion result
        return CompletionResult(
            content=content,
            reasoning=""  # Qwen-Plus doesn't provide reasoning content, use empty string
        ) 