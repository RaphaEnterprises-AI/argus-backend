"""Mistral AI provider implementation.

This module provides integration with Mistral AI's API for chat completions,
function calling, and model listing.

Mistral AI offers a range of models optimized for different use cases:
- mistral-large: Flagship model for complex tasks
- mistral-medium: Balanced model for general use
- mistral-small: Fast and cost-effective
- codestral: Specialized for code generation
"""

import json
import logging
import os
import time
from typing import Any

import httpx

from src.core.providers.base import (
    AuthenticationError,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ContentFilterError,
    ContextLengthError,
    ModelInfo,
    ModelNotFoundError,
    ModelTier,
    ProviderError,
    RateLimitError,
    ToolCall,
)

logger = logging.getLogger(__name__)


# Mistral model catalog with pricing and capabilities
MISTRAL_MODELS: dict[str, dict[str, Any]] = {
    "mistral-large-latest": {
        "display_name": "Mistral Large",
        "input_price_per_1m": 2.0,
        "output_price_per_1m": 6.0,
        "context_window": 128000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.PREMIUM,
        "description": "Flagship model for complex reasoning, multilingual tasks, and code generation",
        "aliases": ["mistral-large"],
    },
    "mistral-medium-latest": {
        "display_name": "Mistral Medium",
        "input_price_per_1m": 0.25,
        "output_price_per_1m": 0.25,
        "context_window": 32000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.VALUE,
        "description": "Balanced model for general tasks with good cost efficiency",
        "aliases": ["mistral-medium"],
    },
    "mistral-small-latest": {
        "display_name": "Mistral Small",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.30,
        "context_window": 32000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.FLASH,
        "description": "Fast and cost-effective model for simple tasks and high-volume use",
        "aliases": ["mistral-small"],
    },
    "codestral-latest": {
        "display_name": "Codestral",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 0.90,
        "context_window": 32000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.VALUE,
        "description": "Specialized model for code generation, completion, and analysis",
        "aliases": ["codestral"],
    },
    "open-mistral-nemo": {
        "display_name": "Mistral Nemo",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.15,
        "context_window": 128000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.VALUE,
        "description": "Open-source model with Apache 2.0 license, great for fine-tuning",
        "aliases": ["mistral-nemo", "open-mistral-nemo-2407"],
    },
    "ministral-8b-latest": {
        "display_name": "Ministral 8B",
        "input_price_per_1m": 0.10,
        "output_price_per_1m": 0.10,
        "context_window": 128000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.FLASH,
        "description": "Compact model optimized for edge deployment and fast inference",
        "aliases": ["ministral-8b"],
    },
    "ministral-3b-latest": {
        "display_name": "Ministral 3B",
        "input_price_per_1m": 0.04,
        "output_price_per_1m": 0.04,
        "context_window": 128000,
        "max_output": 8192,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.FLASH,
        "description": "Ultra-compact model for simple tasks and resource-constrained environments",
        "aliases": ["ministral-3b"],
    },
    "pixtral-large-latest": {
        "display_name": "Pixtral Large",
        "input_price_per_1m": 2.0,
        "output_price_per_1m": 6.0,
        "context_window": 128000,
        "max_output": 8192,
        "supports_vision": True,
        "supports_tools": True,
        "tier": ModelTier.PREMIUM,
        "description": "Multimodal model with vision capabilities for image understanding",
        "aliases": ["pixtral-large"],
    },
    "pixtral-12b-2409": {
        "display_name": "Pixtral 12B",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.15,
        "context_window": 128000,
        "max_output": 8192,
        "supports_vision": True,
        "supports_tools": True,
        "tier": ModelTier.VALUE,
        "description": "Compact multimodal model with vision capabilities",
        "aliases": ["pixtral-12b"],
    },
}


class MistralProvider(BaseProvider):
    """Provider for Mistral AI models.

    Mistral AI provides a range of models from compact (Ministral) to large
    (Mistral Large) with specialized variants for code (Codestral) and
    vision (Pixtral).

    Example:
        ```python
        provider = MistralProvider(api_key="your-api-key")

        # List available models
        models = await provider.list_models()

        # Chat completion
        response = await provider.chat(
            messages=[ChatMessage.user("Explain quantum computing")],
            model="mistral-large-latest",
        )
        print(response.content)

        # With function calling
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }]
        response = await provider.chat(
            messages=[ChatMessage.user("What's the weather in Paris?")],
            model="mistral-large-latest",
            tools=tools,
        )
        ```
    """

    # Provider metadata
    provider_id = "mistral"
    display_name = "Mistral AI"
    website = "https://mistral.ai"
    key_url = "https://console.mistral.ai/api-keys"
    description = "European AI lab offering efficient, powerful language models"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True  # Via Pixtral models
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize the Mistral provider.

        Args:
            api_key: Mistral API key. If None, reads from MISTRAL_API_KEY env var.
            base_url: Optional custom base URL for the API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")

        super().__init__(api_key=api_key)

        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries

        # Create HTTP client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses from the API.

        Args:
            response: The HTTP response to check.

        Raises:
            AuthenticationError: If authentication failed.
            RateLimitError: If rate limit exceeded.
            ModelNotFoundError: If model not found.
            ContentFilterError: If content was filtered.
            ContextLengthError: If context length exceeded.
            ProviderError: For other API errors.
        """
        if response.status_code == 200:
            return

        try:
            error_data = response.json()
            error_message = error_data.get("message", "Unknown error")
            error_type = error_data.get("type", "")
        except Exception:
            error_message = response.text or f"HTTP {response.status_code}"
            error_type = ""

        if response.status_code == 401:
            raise AuthenticationError(f"Invalid API key: {error_message}")

        if response.status_code == 403:
            raise AuthenticationError(f"Access denied: {error_message}")

        if response.status_code == 404:
            raise ModelNotFoundError(f"Model not found: {error_message}")

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = float(retry_after) if retry_after else None
            raise RateLimitError(
                f"Rate limit exceeded: {error_message}",
                retry_after=retry_seconds,
            )

        if response.status_code == 400:
            if "context" in error_message.lower() or "token" in error_message.lower():
                raise ContextLengthError(f"Context length exceeded: {error_message}")
            if "content" in error_message.lower() or "filter" in error_message.lower():
                raise ContentFilterError(f"Content filtered: {error_message}")

        raise ProviderError(f"Mistral API error ({response.status_code}): {error_message}")

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to Mistral API format.

        Args:
            messages: List of ChatMessage objects.

        Returns:
            List of message dicts in Mistral API format.
        """
        result = []
        for msg in messages:
            message_dict: dict[str, Any] = {
                "role": msg.role,
            }

            # Handle content (text or multimodal)
            if isinstance(msg.content, str):
                message_dict["content"] = msg.content
            elif isinstance(msg.content, list):
                # Multimodal content with images
                message_dict["content"] = msg.content

            # Handle tool calls in assistant messages
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls

            # Handle tool results
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id

            result.append(message_dict)

        return result

    def _convert_tools(self, tools: list[dict] | None) -> list[dict[str, Any]] | None:
        """Convert tools to Mistral API format.

        Args:
            tools: List of tool definitions.

        Returns:
            List of tools in Mistral API format.
        """
        if not tools:
            return None

        mistral_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                mistral_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"].get("parameters", {}),
                    },
                })
            else:
                # Pass through as-is if already in correct format
                mistral_tools.append(tool)

        return mistral_tools

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to Mistral AI.

        Args:
            messages: List of chat messages forming the conversation.
            model: Model ID to use (e.g., "mistral-large-latest").
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            tools: List of tool definitions for function calling.
            tool_choice: Tool choice strategy ("auto", "none", "any", or specific).
            stop_sequences: Sequences that stop generation.
            **kwargs: Additional Mistral-specific arguments:
                - top_p: Nucleus sampling parameter.
                - random_seed: Seed for reproducibility.
                - safe_prompt: Enable safety guardrails.

        Returns:
            ChatResponse with generated content and metadata.

        Raises:
            AuthenticationError: If API key is invalid.
            RateLimitError: If rate limit exceeded.
            ModelNotFoundError: If model doesn't exist.
            ContentFilterError: If content blocked.
            ContextLengthError: If input too long.
        """
        if not self.api_key:
            raise AuthenticationError("API key is required")

        client = await self._get_client()

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = self._convert_tools(tools)

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Add optional Mistral-specific parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        if "random_seed" in kwargs:
            payload["random_seed"] = kwargs["random_seed"]

        if "safe_prompt" in kwargs:
            payload["safe_prompt"] = kwargs["safe_prompt"]

        # Make request with retry logic
        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post("/chat/completions", json=payload)
                self._handle_error_response(response)
                break
            except RateLimitError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = e.retry_after or (2 ** attempt)
                    logger.warning(
                        f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}"
                    )
                    await self._async_sleep(wait_time)
                else:
                    raise
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Request error: {e}, retrying in {wait_time}s ({attempt + 1}/{self.max_retries})"
                    )
                    await self._async_sleep(wait_time)
                else:
                    raise ProviderError(f"Request failed after {self.max_retries} attempts: {e}")
        else:
            raise last_error or ProviderError("Request failed")

        latency_ms = (time.time() - start_time) * 1000
        data = response.json()

        # Parse response
        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})

        # Extract tool calls if present
        tool_calls: list[ToolCall] | None = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                arguments = tc["function"].get("arguments", "{}")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=arguments,
                    )
                )

        # Determine finish reason
        finish_reason = choice.get("finish_reason", "stop")
        if finish_reason == "tool_calls":
            finish_reason = "tool_calls"
        elif finish_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return ChatResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            raw_response=data,
        )

    async def _async_sleep(self, seconds: float) -> None:
        """Sleep asynchronously.

        Args:
            seconds: Number of seconds to sleep.
        """
        import asyncio
        await asyncio.sleep(seconds)

    async def validate_key(self, api_key: str) -> bool:
        """Validate a Mistral API key.

        Makes a lightweight request to verify the key is valid.

        Args:
            api_key: API key to validate.

        Returns:
            True if key is valid, False otherwise.
        """
        if not api_key:
            return False

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=10.0,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                # Use the models endpoint for validation (lightweight)
                response = await client.get("/models")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Key validation failed: {e}")
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all available Mistral models.

        Returns models from the static catalog enriched with any
        additional models from the API.

        Returns:
            List of ModelInfo objects for available models.
        """
        models: list[ModelInfo] = []

        # Add models from static catalog
        for model_id, config in MISTRAL_MODELS.items():
            models.append(
                ModelInfo(
                    model_id=model_id,
                    provider=self.provider_id,
                    display_name=config["display_name"],
                    input_price_per_1m=config["input_price_per_1m"],
                    output_price_per_1m=config["output_price_per_1m"],
                    context_window=config["context_window"],
                    max_output=config["max_output"],
                    supports_vision=config.get("supports_vision", False),
                    supports_tools=config.get("supports_tools", True),
                    supports_streaming=True,
                    tier=config.get("tier", ModelTier.STANDARD),
                    description=config.get("description", ""),
                    aliases=config.get("aliases", []),
                )
            )

        # Optionally fetch additional models from API
        if self.api_key:
            try:
                client = await self._get_client()
                response = await client.get("/models")
                if response.status_code == 200:
                    data = response.json()
                    api_model_ids = {m["id"] for m in data.get("data", [])}

                    # Add any models from API that aren't in our catalog
                    existing_ids = {m.model_id for m in models}
                    existing_aliases = set()
                    for m in models:
                        existing_aliases.update(m.aliases)

                    for api_model in data.get("data", []):
                        model_id = api_model["id"]
                        if model_id not in existing_ids and model_id not in existing_aliases:
                            # Add unknown model with default pricing
                            models.append(
                                ModelInfo(
                                    model_id=model_id,
                                    provider=self.provider_id,
                                    display_name=model_id,
                                    input_price_per_1m=0.0,  # Unknown pricing
                                    output_price_per_1m=0.0,
                                    context_window=32000,  # Default assumption
                                    max_output=8192,
                                    supports_tools=True,
                                    tier=ModelTier.STANDARD,
                                    description="Model discovered from API",
                                )
                            )
            except Exception as e:
                logger.warning(f"Failed to fetch models from API: {e}")

        # Cache the models
        self._models_cache = models
        return models

    async def __aenter__(self) -> "MistralProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
