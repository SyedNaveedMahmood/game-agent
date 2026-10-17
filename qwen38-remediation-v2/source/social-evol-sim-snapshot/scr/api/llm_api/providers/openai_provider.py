from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai import OpenAIError, RateLimitError, APITimeoutError, APIConnectionError

from .base import BaseProvider
from .completion_result import CompletionResult
from scr.api.llm_api.config import ProviderConfig

class OpenAIProvider(BaseProvider):
    """
    Provider implementation for OpenAI's API.
    
    This class handles communication with OpenAI's API, providing methods
    for both streaming and non-streaming completions.
    
    Attributes:
        client (OpenAI): The OpenAI client instance
    """
    
    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize the OpenAI provider.
        
        Args:
            config (ProviderConfig): Configuration containing API key and base URL
        """
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
        )

    def get_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs: any
    ) -> CompletionResult:
        """
        Get a completion from OpenAI's API.
        
        Args:
            messages (list[dict[str, str]]): List of message dictionaries
            model (str): Model to use for completion
            stream (bool, optional): Whether to stream the response. Defaults to False.
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            CompletionResult: The completion response containing content and optional reasoning
            
        Raises:
            ValueError: If the model is not supported
            RateLimitError: If the API rate limit is exceeded
            APITimeoutError: If the API request times out
            APIConnectionError: If there are connection issues
            OpenAIError: For other API-related errors
        """
        try:
            # if stream:
            #     return self._handle_streaming_completion(messages, model, **kwargs)
            return self._handle_regular_completion(messages, model, **kwargs)
        except RateLimitError as e:
            raise RateLimitError(f"OpenAI API rate limit exceeded: {str(e)}")
        except APITimeoutError as e:
            raise APITimeoutError(f"OpenAI API request timed out: {str(e)}")
        except APIConnectionError as e:
            raise APIConnectionError(f"Failed to connect to OpenAI API: {str(e)}")
        except OpenAIError as e:
            raise OpenAIError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in OpenAI provider: {str(e)}")

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
            
        Raises:
            OpenAIError: If there's an error in the streaming response
        """
        try:
            # response = self.client.chat.completions.create(
            #     messages=messages,
            #     model=model,
            #     stream=True,
            #     **kwargs
            # )

            # content = ""
            # for chunk in response:
            #     # Each chunk in the stream is a ChatCompletionChunk object
            #     if chunk.choices:
            #         choice = chunk.choices[0]
            #         if choice.delta and choice.delta.content:
            #             # The actual content part of the chunk
            #             content_piece = choice.delta.content
            #             print(content_piece, end="", flush=True) # Print piece by piece
            #             content += content_piece
            #         elif choice.finish_reason:
            #             print(f"\n--- Stream finished. Reason: {choice.finish_reason} ---")

                    
            # print("")  # New line after streaming completes
            from scr.models.agent.responses import Response

            response = self.client.responses.parse(
                model=model,
                input=messages,
                text_format=Response,
            )

            if not response:
                raise OpenAIError("Received empty response from OpenAI API")

            response = response.output_parsed
    
            text_response = response.model_dump_json(indent=4)

            print(text_response)
                
            return CompletionResult(content=text_response)
            
        except Exception as e:
            raise OpenAIError(f"Error in streaming completion: {str(e)}")

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
            
        Raises:
            OpenAIError: If there's an error in the completion response
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=model,
                stream=False,
                **kwargs
            )
            
            if not response.choices:
                raise OpenAIError("Received empty response from OpenAI API")
                
            message = response.choices[0].message
            
            # Handle content
            content = message.content or ""
            
            # Handle reasoning content if available
            reasoning = ""
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning = message.reasoning_content
                
            # Handle function calls if present
            if hasattr(message, 'function_call') and message.function_call:
                content += f"\nFunction call: {message.function_call.name}\n"
                content += f"Arguments: {message.function_call.arguments}\n"
            print(message.content)    
            return CompletionResult(content=content, reasoning=reasoning)
            
        except Exception as e:
            raise OpenAIError(f"Error in regular completion: {str(e)}")