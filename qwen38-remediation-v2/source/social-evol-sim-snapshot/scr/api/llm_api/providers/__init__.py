# Provider implementations are now handled by litellm.
# This package is kept for the CompletionResult model.
from scr.api.llm_api.providers.completion_result import CompletionResult

__all__ = ['CompletionResult']
