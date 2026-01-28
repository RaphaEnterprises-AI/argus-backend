"""Base provider abstraction layer for AI model providers.

This module defines the abstract base class and data structures for
integrating with various AI model providers (Anthropic, OpenAI, Google, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelTier(str, Enum):
    """Price/capability tiers for model selection.

    Tiers help with automatic model selection based on task complexity
    and budget constraints.
    """
    FLASH = "flash"      # $0.05-0.15/1M - trivial tasks (classification, simple extraction)
    VALUE = "value"      # $0.14-0.50/1M - code analysis, basic generation
    STANDARD = "standard" # $0.50-2.00/1M - general tasks, moderate complexity
    PREMIUM = "premium"   # $3-15/1M - complex tasks, high-quality output needed
    EXPERT = "expert"     # $15+/1M - expert reasoning, critical decisions


@dataclass
class ModelInfo:
    """Information about an AI model.

    Contains pricing, capabilities, and configuration details
    for a specific model from a provider.
    """
    model_id: str
    provider: str
    display_name: str
    input_price_per_1m: float
    output_price_per_1m: float
    context_window: int
    max_output: int
    supports_vision: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True
    supports_computer_use: bool = False
    tier: ModelTier = ModelTier.STANDARD
    description: str = ""

    # Additional metadata
    aliases: list[str] = field(default_factory=list)
    release_date: str | None = None
    deprecated: bool = False
    deprecation_message: str = ""

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a given number of tokens.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        return input_cost + output_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "input_price_per_1m": self.input_price_per_1m,
            "output_price_per_1m": self.output_price_per_1m,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "supports_computer_use": self.supports_computer_use,
            "tier": self.tier.value,
            "description": self.description,
            "aliases": self.aliases,
            "release_date": self.release_date,
            "deprecated": self.deprecated,
            "deprecation_message": self.deprecation_message,
        }


@dataclass
class ChatMessage:
    """A message in a chat conversation.

    Supports both text-only and multimodal content (images, etc.).
    """
    role: str  # "user", "assistant", "system"
    content: str | list[dict]

    # Optional metadata
    name: str | None = None  # For multi-user conversations
    tool_call_id: str | None = None  # For tool results
    tool_calls: list[dict] | None = None  # For assistant tool calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to provider-agnostic dictionary format."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result

    @classmethod
    def user(cls, content: str | list[dict]) -> "ChatMessage":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | list[dict]) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role="system", content=content)


@dataclass
class ToolCall:
    """Represents a tool call made by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from a chat completion request.

    Contains the generated content, token usage, and metadata.
    """
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # "stop", "length", "tool_calls", "content_filter"
    raw_response: Any = None

    # Optional fields for extended functionality
    tool_calls: list[ToolCall] | None = None
    system_fingerprint: str | None = None

    # Timing information
    latency_ms: float | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in the request/response."""
        return self.input_tokens + self.output_tokens

    def calculate_cost(self, model_info: ModelInfo) -> float:
        """Calculate the cost of this response.

        Args:
            model_info: Model information with pricing

        Returns:
            Cost in USD
        """
        return model_info.calculate_cost(self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "content": self.content,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "total_tokens": self.total_tokens,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        if self.system_fingerprint:
            result["system_fingerprint"] = self.system_fingerprint
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        return result


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class AuthenticationError(ProviderError):
    """Raised when API key is invalid or missing."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(ProviderError):
    """Raised when usage quota is exceeded."""

    def __init__(self, message: str, reset_at: str | None = None):
        super().__init__(message)
        self.reset_at = reset_at


class ModelNotFoundError(ProviderError):
    """Raised when requested model is not available."""
    pass


class ContentFilterError(ProviderError):
    """Raised when content is blocked by safety filters."""
    pass


class ContextLengthError(ProviderError):
    """Raised when input exceeds model's context window."""
    pass


class TemperatureError(ProviderError):
    """Raised when temperature is out of valid range."""
    pass


def validate_temperature(temperature: float, min_temp: float = 0.0, max_temp: float = 2.0) -> float:
    """Validate and clamp temperature to valid range.

    Args:
        temperature: The temperature value to validate
        min_temp: Minimum allowed temperature (default 0.0)
        max_temp: Maximum allowed temperature (default 2.0)

    Returns:
        Clamped temperature value within valid range

    Note:
        This function clamps rather than raises to be permissive,
        but logs a warning when clamping occurs.
    """
    if temperature < min_temp:
        return min_temp
    if temperature > max_temp:
        return max_temp
    return temperature


def mask_api_key(api_key: str | None) -> str:
    """Mask an API key for safe logging/display.

    Args:
        api_key: The API key to mask

    Returns:
        Masked string showing only first 4 and last 4 characters
    """
    if not api_key:
        return "<not set>"
    if len(api_key) <= 12:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


class ProviderCapability(str, Enum):
    """Capabilities that a provider may support."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    TOOLS = "tools"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    COMPUTER_USE = "computer_use"
    REASONING = "reasoning"  # Extended thinking / reasoning mode


@dataclass
class ProviderConfig:
    """Configuration for initializing a provider.

    Contains all settings needed to configure a provider instance,
    including authentication, endpoints, and behavior options.
    """
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    # Optional organization/project identifiers
    organization_id: str | None = None
    project_id: str | None = None
    # Custom headers for requests
    custom_headers: dict[str, str] = field(default_factory=dict)


class BaseProvider(ABC):
    """Abstract base class for all AI providers.

    Subclasses must implement the abstract methods to provide
    provider-specific functionality.

    Example usage:
        ```python
        provider = AnthropicProvider(api_key="sk-ant-...")

        messages = [
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("Hello, how are you?"),
        ]

        response = await provider.chat(
            messages=messages,
            model="claude-sonnet-4-5-20250514",
            temperature=0.7,
        )

        print(response.content)
        ```
    """

    # Provider metadata - subclasses must define these
    provider_id: str
    display_name: str
    website: str
    key_url: str
    description: str

    # Capability flags - subclasses can override
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = True
    supports_computer_use: bool = False
    is_aggregator: bool = False  # True for OpenRouter, Together, etc.

    def __init__(self, api_key: str | None = None):
        """Initialize the provider.

        Args:
            api_key: API key for authentication. If None, will attempt
                     to load from environment variable.
        """
        self.api_key = api_key
        self._models_cache: list[ModelInfo] | None = None

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of chat messages forming the conversation
            model: Model ID to use for completion
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate (None = model default)
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            stop_sequences: Sequences that stop generation
            **kwargs: Provider-specific additional arguments

        Returns:
            ChatResponse with generated content and metadata

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
            ContentFilterError: If content blocked by safety filters
            ContextLengthError: If input too long
        """
        pass

    @abstractmethod
    async def validate_key(self, api_key: str) -> bool:
        """Validate an API key by making a test request.

        Args:
            api_key: API key to validate

        Returns:
            True if key is valid, False otherwise
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List all available models from this provider.

        Returns:
            List of ModelInfo objects for available models
        """
        pass

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            ModelInfo if found, None otherwise
        """
        # Use cached models if available
        if self._models_cache is None:
            self._models_cache = await self.list_models()

        for model in self._models_cache:
            if model.model_id == model_id:
                return model
            # Also check aliases
            if model_id in model.aliases:
                return model

        return None

    def clear_cache(self) -> None:
        """Clear the models cache to force refresh on next list_models call."""
        self._models_cache = None

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the provider.

        Returns:
            Dictionary with health status information
        """
        try:
            is_valid = await self.validate_key(self.api_key or "")
            return {
                "provider": self.provider_id,
                "status": "healthy" if is_valid else "unhealthy",
                "authenticated": is_valid,
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "status": "error",
                "authenticated": False,
                "error": str(e),
            }

    def __repr__(self) -> str:
        """String representation of the provider.

        Note: API key is masked for security - never expose full keys in logs.
        """
        return (
            f"<{self.__class__.__name__}("
            f"provider_id='{self.provider_id}', "
            f"api_key={mask_api_key(self.api_key)})>"
        )
