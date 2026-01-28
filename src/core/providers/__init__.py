"""Provider abstraction layer for AI model providers.

This module provides a unified interface for interacting with various
AI model providers (Anthropic, OpenAI, Google, Azure, etc.).

Example usage:
    ```python
    from src.core.providers import AzureOpenAIProvider, ChatMessage

    # Create Azure OpenAI provider
    provider = AzureOpenAIProvider(
        api_key="your-key",
        endpoint="https://your-resource.openai.azure.com/",
    )

    # Create messages
    messages = [
        ChatMessage.system("You are a helpful assistant."),
        ChatMessage.user("Hello!"),
    ]

    # Get completion using deployment name
    response = await provider.chat(messages, model="gpt-4o-deployment")
    print(response.content)
    ```

Available Providers:
- OpenRouterProvider: OpenRouter aggregator (300+ models, RECOMMENDED)
- DeepSeekProvider: DeepSeek API (V3 chat, R1 reasoning)
- MistralProvider: Mistral AI (Mistral Large, Small, Codestral, Pixtral)
- AzureOpenAIProvider: Azure OpenAI Service (enterprise deployment)
- VertexProvider: Google Cloud Vertex AI (Gemini models)
- BedrockProvider: AWS Bedrock (Claude, Titan, Llama via AWS)
- XAIProvider: xAI's Grok models with real-time data access
- CerebrasProvider: Ultra-fast Llama inference on Wafer-Scale Engine
- (Future) AnthropicProvider: Direct Anthropic API
- (Future) OpenAIProvider: Direct OpenAI API

Configuration:
    OpenRouter (RECOMMENDED):
        - OPENROUTER_API_KEY: Your OpenRouter API key
        - OPENROUTER_REFERER: HTTP-Referer header for tracking (optional)
        - OPENROUTER_APP_NAME: X-Title header for app identification (optional)

    DeepSeek:
        - DEEPSEEK_API_KEY: Your DeepSeek API key

    Mistral AI:
        - MISTRAL_API_KEY: Your Mistral AI API key

    Azure OpenAI:
        - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
        - AZURE_OPENAI_ENDPOINT: Your Azure resource endpoint URL

    Vertex AI (Google Cloud):
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
        - VERTEX_PROJECT_ID: Your GCP project ID
        - VERTEX_LOCATION: GCP region (default: us-central1)

    AWS Bedrock:
        - AWS_BEDROCK_REGION: AWS region (e.g., us-east-1, us-west-2)
        - AWS_ACCESS_KEY_ID: Optional if using IAM roles
        - AWS_SECRET_ACCESS_KEY: Optional if using IAM roles

    xAI:
        - XAI_API_KEY: Your xAI API key

    Cerebras:
        - CEREBRAS_API_KEY: Your Cerebras API key
"""

from .base import (
    # Enums
    ModelTier,
    ProviderCapability,
    # Data classes
    ModelInfo,
    ChatMessage,
    ChatResponse,
    ToolCall,
    ProviderConfig,
    # Base class
    BaseProvider,
    # Exceptions
    ProviderError,
    AuthenticationError,
    RateLimitError,
    QuotaExceededError,
    ModelNotFoundError,
    ContentFilterError,
    ContextLengthError,
)
from .azure_provider import AzureOpenAIProvider
from .bedrock_provider import BedrockProvider, BedrockError, ThrottlingError, ServiceQuotaError
from .cerebras_provider import CerebrasProvider
from .deepseek_provider import DeepSeekProvider, DEEPSEEK_MODELS
from .mistral_provider import MistralProvider
from .openrouter_provider import OpenRouterProvider, get_openrouter_provider
from .vertex_provider import VertexProvider
from .xai_provider import XAIProvider

__all__ = [
    # Enums
    "ModelTier",
    "ProviderCapability",
    # Data classes
    "ModelInfo",
    "ChatMessage",
    "ChatResponse",
    "ToolCall",
    "ProviderConfig",
    # Base class
    "BaseProvider",
    # Provider implementations
    "AzureOpenAIProvider",
    "BedrockProvider",
    "CerebrasProvider",
    "DeepSeekProvider",
    "DEEPSEEK_MODELS",
    "MistralProvider",
    "OpenRouterProvider",
    "VertexProvider",
    "XAIProvider",
    "get_openrouter_provider",
    # Exceptions
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "QuotaExceededError",
    "ModelNotFoundError",
    "ContentFilterError",
    "ContextLengthError",
    # Bedrock-specific exceptions
    "BedrockError",
    "ThrottlingError",
    "ServiceQuotaError",
]
