"""OpenRouter provider implementation.

OpenRouter is an AI model aggregator providing access to 300+ models from
various providers (Anthropic, OpenAI, Google, Meta, Mistral, etc.) through
a single unified API.

Benefits:
- Single API key for all models
- Automatic failover between providers
- No markup on provider pricing (just 5.5% platform fee)
- Gets latest models first
- OpenAI-compatible API format

Usage:
    provider = OpenRouterProvider(api_key="sk-or-...")

    # List all 300+ available models
    models = await provider.list_models()

    # Chat with any model
    response = await provider.chat(
        model="anthropic/claude-sonnet-4-5-20241022",
        messages=[ChatMessage.user("Hello!")],
    )

Model ID format: "provider/model-name" (e.g., "anthropic/claude-sonnet-4-5")
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog

from src.core.providers.base import (
    AuthenticationError,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ContextLengthError,
    ModelInfo,
    ModelNotFoundError,
    ModelTier,
    ProviderError,
    RateLimitError,
    ToolCall,
    mask_api_key,
    validate_temperature,
)

logger = structlog.get_logger()

# OpenRouter API configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_ENDPOINT = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_ENDPOINT = f"{OPENROUTER_BASE_URL}/chat/completions"

# Cache TTL for models list (1 hour)
MODELS_CACHE_TTL = timedelta(hours=1)


def _determine_tier(input_price: float, output_price: float) -> ModelTier:
    """Determine the model tier based on pricing.

    Args:
        input_price: Input price per 1M tokens
        output_price: Output price per 1M tokens

    Returns:
        ModelTier based on average pricing
    """
    avg_price = (input_price + output_price) / 2

    if avg_price < 0.15:
        return ModelTier.FLASH
    elif avg_price < 0.50:
        return ModelTier.VALUE
    elif avg_price < 2.00:
        return ModelTier.STANDARD
    elif avg_price < 15.00:
        return ModelTier.PREMIUM
    else:
        return ModelTier.EXPERT


def _parse_openrouter_model(model_data: dict[str, Any]) -> ModelInfo:
    """Parse OpenRouter model data into ModelInfo.

    OpenRouter model format:
    {
        "id": "anthropic/claude-sonnet-4-5-20241022",
        "name": "Claude Sonnet 4.5",
        "description": "...",
        "pricing": {
            "prompt": "0.000003",  # Price per token
            "completion": "0.000015"
        },
        "context_length": 200000,
        "architecture": {
            "modality": "text+image->text",
            "tokenizer": "Claude",
            "instruct_type": "claude"
        },
        "top_provider": {
            "context_length": 200000,
            "max_completion_tokens": 8192
        },
        "per_request_limits": {...}
    }
    """
    model_id = model_data.get("id", "")
    name = model_data.get("name", model_id)

    # Parse pricing (convert from per-token to per-1M tokens)
    pricing = model_data.get("pricing", {})
    input_price_per_token = float(pricing.get("prompt", "0") or "0")
    output_price_per_token = float(pricing.get("completion", "0") or "0")

    # Convert to per 1M tokens
    input_price_per_1m = input_price_per_token * 1_000_000
    output_price_per_1m = output_price_per_token * 1_000_000

    # Get context and output limits
    context_length = model_data.get("context_length", 4096)
    top_provider = model_data.get("top_provider", {})
    max_output = top_provider.get("max_completion_tokens", 4096)

    # Determine capabilities from architecture
    architecture = model_data.get("architecture", {})
    modality = architecture.get("modality", "text->text")
    supports_vision = "image" in modality.lower()
    supports_tools = True  # Most models on OpenRouter support tools

    # Check for computer use support (currently only Claude models)
    supports_computer_use = "claude" in model_id.lower() and (
        "sonnet" in model_id.lower() or "opus" in model_id.lower()
    )

    # Extract provider from model ID
    provider = model_id.split("/")[0] if "/" in model_id else "unknown"

    # Determine tier based on pricing
    tier = _determine_tier(input_price_per_1m, output_price_per_1m)

    return ModelInfo(
        model_id=model_id,
        provider=provider,
        display_name=name,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        context_window=context_length,
        max_output=max_output,
        supports_vision=supports_vision,
        supports_tools=supports_tools,
        supports_streaming=True,  # All OpenRouter models support streaming
        supports_computer_use=supports_computer_use,
        tier=tier,
        description=model_data.get("description", ""),
    )


class OpenRouterProvider(BaseProvider):
    """Provider implementation for OpenRouter.

    OpenRouter is an aggregator providing 300+ models from various providers
    through a single unified OpenAI-compatible API.

    Features:
    - Single API key for all models
    - Automatic failover
    - Real-time pricing updates
    - 1-hour model caching with automatic refresh

    Environment variables:
    - OPENROUTER_API_KEY: API key for authentication
    - OPENROUTER_REFERER: HTTP-Referer header for tracking (optional)
    - OPENROUTER_APP_NAME: X-Title header for tracking (optional)
    """

    # Provider metadata
    provider_id = "openrouter"
    display_name = "OpenRouter"
    website = "https://openrouter.ai"
    key_url = "https://openrouter.ai/keys"
    description = (
        "AI model aggregator with 300+ models from Anthropic, OpenAI, Google, "
        "Meta, Mistral, and more. Single API key, unified pricing, automatic failover."
    )

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_computer_use = True  # Via Claude models
    is_aggregator = True

    def __init__(
        self,
        api_key: str | None = None,
        referer: str | None = None,
        app_name: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            referer: HTTP-Referer header for tracking. Falls back to OPENROUTER_REFERER.
            app_name: X-Title header for app identification. Falls back to OPENROUTER_APP_NAME.
            timeout: Request timeout in seconds.
        """
        super().__init__(api_key)

        # Load from environment if not provided
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.referer = referer or os.environ.get("OPENROUTER_REFERER", "https://argus.dev")
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "Argus E2E Testing Agent")
        self.timeout = timeout

        # Models cache with TTL
        self._models_cache: list[ModelInfo] | None = None
        self._models_cache_time: datetime | None = None

        # HTTP client (created lazily)
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self, api_key: str | None = None) -> dict[str, str]:
        """Get request headers for OpenRouter API.

        Args:
            api_key: Optional override API key

        Returns:
            Headers dict with authentication and tracking headers
        """
        key = api_key or self.api_key
        headers = {
            "Content-Type": "application/json",
        }

        if key:
            headers["Authorization"] = f"Bearer {key}"

        # OpenRouter tracking headers
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_name:
            headers["X-Title"] = self.app_name

        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._get_headers(),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _is_cache_valid(self) -> bool:
        """Check if the models cache is still valid."""
        if self._models_cache is None or self._models_cache_time is None:
            return False
        return datetime.now() - self._models_cache_time < MODELS_CACHE_TTL

    async def list_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """List all available models from OpenRouter.

        Fetches the complete model list from OpenRouter and caches it
        for 1 hour to reduce API calls.

        Args:
            force_refresh: Force cache refresh even if not expired

        Returns:
            List of ModelInfo objects for all available models (300+)

        Raises:
            AuthenticationError: If API key is invalid
            ProviderError: If request fails
        """
        # Return cached models if valid
        if not force_refresh and self._is_cache_valid():
            return self._models_cache  # type: ignore

        logger.info("Fetching OpenRouter models list")
        start_time = time.time()

        client = await self._get_client()

        try:
            response = await client.get(
                OPENROUTER_MODELS_ENDPOINT,
                headers=self._get_headers(),
            )

            if response.status_code == 401:
                raise AuthenticationError("Invalid OpenRouter API key")

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "OpenRouter rate limit exceeded",
                    retry_after=float(retry_after) if retry_after else None,
                )

            response.raise_for_status()

            data = response.json()
            models_data = data.get("data", [])

            # Parse all models
            models = []
            for model_data in models_data:
                try:
                    model_info = _parse_openrouter_model(model_data)
                    models.append(model_info)
                except Exception as e:
                    logger.warning(
                        "Failed to parse OpenRouter model",
                        model_id=model_data.get("id"),
                        error=str(e),
                    )

            # Update cache
            self._models_cache = models
            self._models_cache_time = datetime.now()

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "Fetched OpenRouter models",
                model_count=len(models),
                elapsed_ms=round(elapsed_ms, 2),
            )

            return models

        except httpx.HTTPStatusError as e:
            logger.error(
                "OpenRouter API error",
                status_code=e.response.status_code,
                response_text=e.response.text[:500],
            )
            raise ProviderError(f"OpenRouter API error: {e.response.status_code}")

        except httpx.RequestError as e:
            logger.error("OpenRouter request failed", error=str(e))
            raise ProviderError(f"OpenRouter request failed: {str(e)}")

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        json_mode: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to OpenRouter.

        Args:
            messages: List of chat messages (ChatMessage or dict)
            model: Model ID (e.g., "anthropic/claude-sonnet-4-5-20241022")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            json_mode: Enable JSON response format
            **kwargs: Additional OpenRouter-specific parameters

        Returns:
            ChatResponse with generated content and metadata

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
            ContextLengthError: If input too long
        """
        if not self.api_key:
            raise AuthenticationError("OpenRouter API key not configured")

        # Validate and clamp temperature to valid range (0.0-2.0)
        validated_temp = validate_temperature(temperature)
        if validated_temp != temperature:
            logger.debug(
                "Temperature clamped to valid range",
                original=temperature,
                clamped=validated_temp,
            )

        # Convert ChatMessage objects to dicts
        message_dicts = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                message_dicts.append(msg.to_dict())
            else:
                message_dicts.append(msg)

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": message_dicts,
            "temperature": validated_temp,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            # Convert to OpenAI tool format if needed
            payload["tools"] = self._format_tools(tools)

        if tool_choice:
            payload["tool_choice"] = tool_choice

        if stop_sequences:
            payload["stop"] = stop_sequences

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # Add any additional kwargs
        payload.update(kwargs)

        logger.debug(
            "OpenRouter chat request",
            model=model,
            message_count=len(message_dicts),
            max_tokens=max_tokens,
        )

        start_time = time.time()
        client = await self._get_client()

        try:
            response = await client.post(
                OPENROUTER_CHAT_ENDPOINT,
                json=payload,
                headers=self._get_headers(),
            )

            latency_ms = (time.time() - start_time) * 1000

            # Handle errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid OpenRouter API key")

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "OpenRouter rate limit exceeded",
                    retry_after=float(retry_after) if retry_after else None,
                )

            if response.status_code == 404:
                raise ModelNotFoundError(f"Model not found: {model}")

            if response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Bad request")
                if "context length" in error_msg.lower() or "too long" in error_msg.lower():
                    raise ContextLengthError(error_msg)
                raise ProviderError(f"OpenRouter error: {error_msg}")

            response.raise_for_status()

            data = response.json()

            # Parse response
            choices = data.get("choices", [])
            if not choices:
                raise ProviderError("No response from OpenRouter")

            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""

            # Parse tool calls if present
            tool_calls_data = message.get("tool_calls", [])
            tool_calls = None
            if tool_calls_data:
                tool_calls = []
                for tc in tool_calls_data:
                    import json
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=json.loads(tc.get("function", {}).get("arguments", "{}")),
                    ))

            # Get usage info
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Determine finish reason
            finish_reason = choice.get("finish_reason", "stop")
            if tool_calls:
                finish_reason = "tool_calls"

            logger.debug(
                "OpenRouter chat response",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
            )

            return ChatResponse(
                content=content,
                model=data.get("model", model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                system_fingerprint=data.get("system_fingerprint"),
                latency_ms=latency_ms,
                raw_response=data,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "OpenRouter chat error",
                status_code=e.response.status_code,
                response_text=e.response.text[:500],
            )
            raise ProviderError(f"OpenRouter API error: {e.response.status_code}")

        except httpx.RequestError as e:
            logger.error("OpenRouter request failed", error=str(e))
            raise ProviderError(f"OpenRouter request failed: {str(e)}")

    def _format_tools(self, tools: list[dict]) -> list[dict]:
        """Format tools for OpenRouter (OpenAI-compatible format).

        Handles conversion from Anthropic tool format if needed.

        Args:
            tools: List of tool definitions

        Returns:
            Tools in OpenAI format
        """
        formatted = []
        for tool in tools:
            # Check if already in OpenAI format
            if "type" in tool and tool["type"] == "function":
                formatted.append(tool)
            else:
                # Convert from Anthropic format
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    }
                })
        return formatted

    async def validate_key(self, api_key: str | None = None) -> bool:
        """Validate an OpenRouter API key.

        Validates by attempting to fetch the models list.

        Args:
            api_key: API key to validate (uses instance key if not provided)

        Returns:
            True if key is valid, False otherwise
        """
        key_to_test = api_key or self.api_key
        if not key_to_test:
            return False

        try:
            client = await self._get_client()
            response = await client.get(
                OPENROUTER_MODELS_ENDPOINT,
                headers=self._get_headers(key_to_test),
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("OpenRouter key validation failed", error=str(e))
            return False

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model.

        Uses cached models list if available.

        Args:
            model_id: Model ID to look up (e.g., "anthropic/claude-sonnet-4-5")

        Returns:
            ModelInfo if found, None otherwise
        """
        # Ensure cache is populated
        if not self._is_cache_valid():
            await self.list_models()

        if self._models_cache is None:
            return None

        # Look up by exact ID or alias
        for model in self._models_cache:
            if model.model_id == model_id:
                return model
            if model_id in model.aliases:
                return model

        return None

    def clear_cache(self) -> None:
        """Clear the models cache to force refresh on next list_models call."""
        self._models_cache = None
        self._models_cache_time = None

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on OpenRouter.

        Returns:
            Dictionary with health status, model count, and cache info
        """
        result = await super().health_check()

        # Add OpenRouter-specific info
        if self._models_cache:
            result["cached_models"] = len(self._models_cache)
            result["cache_age_seconds"] = (
                (datetime.now() - self._models_cache_time).total_seconds()
                if self._models_cache_time else None
            )

        return result

    def __repr__(self) -> str:
        """String representation with masked API key for security."""
        return (
            f"<OpenRouterProvider("
            f"api_key={mask_api_key(self.api_key)}, "
            f"cached_models={len(self._models_cache) if self._models_cache else 0})>"
        )

    async def __aenter__(self) -> "OpenRouterProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Convenience function for creating provider instance
def get_openrouter_provider(
    api_key: str | None = None,
    **kwargs,
) -> OpenRouterProvider:
    """Create an OpenRouter provider instance.

    Args:
        api_key: Optional API key (falls back to environment variable)
        **kwargs: Additional provider configuration

    Returns:
        Configured OpenRouterProvider instance
    """
    return OpenRouterProvider(api_key=api_key, **kwargs)
