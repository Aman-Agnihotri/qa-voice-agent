"""
Configuration management for QA Voice Agent.
"""

import os
from typing import Dict, Any, Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Fallback for basic functionality
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def Field(default=None, env=None):
            return default
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    GOOGLE = "google"
    MISTRAL = "mistral"
    OLLAMA = "ollama"


class AnalysisMode(str, Enum):
    """Analysis modes for QA evaluation."""
    QUICK = "quick"
    DETAILED = "detailed"
    COMPARATIVE = "comparative"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    mistral_api_key: Optional[str] = Field(None, env="MISTRAL_API_KEY")

    # Model Configuration
    default_provider: ModelProvider = ModelProvider.OPENAI
    default_model: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.1

    # Ollama Configuration (for local models)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Analysis Configuration
    default_analysis_mode: AnalysisMode = AnalysisMode.DETAILED
    enable_batch_processing: bool = True
    concurrent_requests: int = 5

    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    # File Paths
    data_dir: str = "data"
    output_dir: str = "output"
    logs_dir: str = "logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Model-specific configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI Models
    "gpt-4": {
        "provider": ModelProvider.OPENAI,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.03,
    },
    "gpt-4-turbo": {
        "provider": ModelProvider.OPENAI,
        "max_tokens": 4000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.01,
    },
    "gpt-3.5-turbo": {
        "provider": ModelProvider.OPENAI,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.001,
    },

    # Anthropic Models
    "claude-3-sonnet": {
        "provider": ModelProvider.ANTHROPIC,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.015,
    },
    "claude-3-haiku": {
        "provider": ModelProvider.ANTHROPIC,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.0025,
    },

    # Groq Models (fast inference)
    "llama3-70b": {
        "provider": ModelProvider.GROQ,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.0008,
    },
    "mixtral-8x7b": {
        "provider": ModelProvider.GROQ,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.0006,
    },

    # Google Models
    "gemini-pro": {
        "provider": ModelProvider.GOOGLE,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.001,
    },

    # Mistral Models
    "mistral-large": {
        "provider": ModelProvider.MISTRAL,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.008,
    },

    # Ollama Models (local)
    "llama3": {
        "provider": ModelProvider.OLLAMA,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.0,  # Local model, no cost
    },
    "qwen2": {
        "provider": ModelProvider.OLLAMA,
        "max_tokens": 2000,
        "temperature": 0.1,
        "supports_json": True,
        "cost_per_1k_tokens": 0.0,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["gpt-3.5-turbo"])


def get_available_models() -> Dict[ModelProvider, list]:
    """Get available models grouped by provider."""
    models_by_provider = {}
    for model_name, config in MODEL_CONFIGS.items():
        provider = config["provider"]
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model_name)
    return models_by_provider


def validate_api_keys(settings: Settings) -> Dict[ModelProvider, bool]:
    """Validate which API keys are available."""
    key_status = {}

    key_status[ModelProvider.OPENAI] = bool(settings.openai_api_key)
    key_status[ModelProvider.ANTHROPIC] = bool(settings.anthropic_api_key)
    key_status[ModelProvider.GROQ] = bool(settings.groq_api_key)
    key_status[ModelProvider.GOOGLE] = bool(settings.google_api_key)
    key_status[ModelProvider.MISTRAL] = bool(settings.mistral_api_key)
    key_status[ModelProvider.OLLAMA] = True  # Local models don't need API keys

    return key_status


# Global settings instance
settings = Settings()