import os
from typing import Dict, List, Optional, Tuple
from scr.utils.logger import get_logger

logger = get_logger(__name__)

# Provider settings configuration (kept for reference and env key mapping)
PROVIDER_SETTINGS = {
    "llama_cpp": {
        "base_url": "http://127.0.0.1:8080/v1",
        "models": ["Qwen3.8-27B-Q4_K_M"]
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": [
            "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06",
            "gpt-4o-2024-05-13", "gpt-4.1-mini-2025-04-14", "gpt-5-mini"
        ]
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder"]
    },
    "tongyuan": {
        "base_url": "https://api.tonggpt.mybigai.ac.cn/proxy/eastus2",
        "models": [
            "gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20",
            "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14",
            "o4-mini-2025-04-16", "o3-mini-2025-01-31", "o1-mini-2024-09-12",
        ]
    },
    "alibaba": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus"]
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "openai/gpt-4o-mini", "openai/gpt-4o",
            "anthropic/claude-sonnet-4", "anthropic/claude-haiku-4",
            "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3-0324",
        ]
    },
    "claude": {
        "base_url": None,
        "models": ["sonnet", "haiku", "opus"]
    },
    "gemini": {
        "base_url": None,
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview"]
    },
    "codex": {
        "base_url": None,
        "models": ["gpt-5.1-codex-mini", "gpt-5.3-codex"]
    }
}

# Environment variable mappings for each provider
PROVIDER_ENV_KEYS = {
    "llama_cpp": None,
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "alibaba": "ALIBABA_CLOUD_API_KEY",
    "tongyuan": "TONGYUAN_OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "claude": None,
    "gemini": None,
    "codex": None
}


def get_litellm_model_string(provider: str, model: str) -> str:
    """Convert (provider, model) to a litellm-compatible model string."""
    if provider == "llama_cpp":
        return f"openai/{model}"
    elif provider == "openai":
        return model
    elif provider == "deepseek":
        return f"deepseek/{model}"
    elif provider == "alibaba":
        return f"openai/{model}"
    elif provider == "tongyuan":
        return f"azure/{model}"
    elif provider == "siliconflow":
        return model
    elif provider == "openrouter":
        return f"openrouter/{model}"
    elif provider == "claude":
        return model
    elif provider == "gemini":
        return model
    elif provider == "codex":
        return model
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_litellm_kwargs(provider: str) -> Dict:
    """Get extra kwargs needed for litellm.completion() for a given provider."""
    kwargs = {}
    if provider == "llama_cpp":
        kwargs["api_base"] = PROVIDER_SETTINGS[provider]["base_url"]
        kwargs["api_key"] = "local"  # SDK placeholder; llama-server needs no key.
        kwargs["timeout"] = 1800.0
    elif provider == "alibaba":
        kwargs["api_base"] = PROVIDER_SETTINGS["alibaba"]["base_url"]
        kwargs["api_key"] = os.getenv(PROVIDER_ENV_KEYS["alibaba"])
    elif provider == "tongyuan":
        kwargs["api_base"] = PROVIDER_SETTINGS["tongyuan"]["base_url"]
        kwargs["api_key"] = os.getenv(PROVIDER_ENV_KEYS["tongyuan"])
        kwargs["api_version"] = "2025-03-01-preview"
    elif provider == "deepseek":
        kwargs["api_key"] = os.getenv(PROVIDER_ENV_KEYS["deepseek"])
    elif provider == "openrouter":
        kwargs["api_key"] = os.getenv(PROVIDER_ENV_KEYS["openrouter"])
    # openai uses OPENAI_API_KEY env var automatically via litellm
    return kwargs


def validate_provider(provider: str) -> None:
    """Validate that a provider is supported."""
    supported = set(PROVIDER_ENV_KEYS.keys()) | {"siliconflow"}
    if provider not in supported:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")


def get_available_models(provider: str) -> List[str]:
    """Get the list of available models for a provider."""
    try:
        return PROVIDER_SETTINGS[provider]["models"]
    except KeyError as e:
        raise KeyError(f"Provider not found: {provider}") from e
