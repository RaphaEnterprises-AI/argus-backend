"""xAI Provider for Grok models.

This module provides integration with xAI's Grok models via their
OpenAI-compatible API. xAI models feature real-time data access
and are designed for witty, informative responses.

API Documentation: https://docs.x.ai/
"""

import json
import os
import time
from typing import Any

import httpx

from .base import (
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
    QuotaExceededError,
    RateLimitError,
    ToolCall,
    mask_api_key,
    validate_temperature,
)


class XAIProvider(BaseProvider):
    """Provider for xAI's Grok models.

    xAI offers the Grok family of models with OpenAI-compatible API,
    featuring real-time data access and witty, informative responses.

    Features:
    - OpenAI-compatible API
    - Real-time data access (web search, current events)
    - Vision capabilities (Grok-2 Vision)
    - Tool/function calling support

    Example usage:
        ```python
        provider = XAIProvider(api_key="xai-...")

        messages = [
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("What's happening in the world today?"),
        ]

        response = await provider.chat(
            messages=messages,
            model="grok-2",
            temperature=0.7,
        )
        print(response.content)
        ```
    """

    # Provider metadata
    provider_id = "xai"
    display_name = "xAI"
    website = "https://x.ai"
    key_url = "https://console.x.ai"
    description = "xAI's Grok models with real-time data access"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.x.ai/v1"
    DEFAULT_TIMEOUT = 120.0

    # Model definitions with pricing and capabilities
    MODELS = {
        "grok-2": ModelInfo(
            model_id="grok-2",
            provider="xai",
            display_name="Grok 2",
            input_price_per_1m=2.00,
            output_price_per_1m=10.00,
            context_window=131072,
            max_output=32768,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.PREMIUM,
            description="xAI's flagship model with real-time data access and witty responses",
            aliases=["grok-2-latest", "grok-2-1212"],
            release_date="2024-12",
        ),
        "grok-2-vision": ModelInfo(
            model_id="grok-2-vision",
            provider="xai",
            display_name="Grok 2 Vision",
            input_price_per_1m=2.00,
            output_price_per_1m=10.00,
            context_window=32768,
            max_output=8192,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.PREMIUM,
            description="Grok 2 with vision capabilities for image understanding",
            aliases=["grok-2-vision-latest", "grok-2-vision-1212"],
            release_date="2024-12",
        ),
        "grok-3": ModelInfo(
            model_id="grok-3",
            provider="xai",
            display_name="Grok 3",
            input_price_per_1m=3.00,
            output_price_per_1m=15.00,
            context_window=131072,
            max_output=32768,
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.EXPERT,
            description="xAI's most capable model with enhanced reasoning",
            aliases=["grok-3-latest"],
            release_date="2025-02",
        ),
    }

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize the xAI provider.

        Args:
            api_key: xAI API key. If None, reads from XAI_API_KEY environment variable.
            base_url: Override the default API base URL. Defaults to https://api.x.ai/v1
        """
        super().__init__(api_key=api_key)
        self.api_key = api_key or os.environ.get("XAI_API_KEY")
        self.base_url = base_url or self.BASE_URL
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.DEFAULT_TIMEOUT),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to xAI/OpenAI format.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of message dictionaries in OpenAI format
        """
        result = []
        for msg in messages:
            message_dict: dict[str, Any] = {"role": msg.role}

            # Handle multimodal content (images)
            if isinstance(msg.content, list):
                message_dict["content"] = msg.content
            else:
                message_dict["content"] = msg.content

            # Add optional fields
            if msg.name:
                message_dict["name"] = msg.name
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls

            result.append(message_dict)

        return result

    def _convert_tools(self, tools: list[dict] | None) -> list[dict[str, Any]] | None:
        """Convert tools to xAI/OpenAI format.

        Args:
            tools: List of tool definitions

        Returns:
            Tools in OpenAI function calling format
        """
        if not tools:
            return None

        result = []
        for tool in tools:
            # Already in OpenAI format
            if "type" in tool and tool["type"] == "function":
                result.append(tool)
            # Convert from simplified format
            elif "name" in tool and "parameters" in tool:
                result.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["parameters"],
                    },
                })
            else:
                result.append(tool)

        return result

    def _parse_tool_calls(self, tool_calls_data: list[dict]) -> list[ToolCall]:
        """Parse tool calls from API response.

        Args:
            tool_calls_data: Raw tool calls from API response

        Returns:
            List of ToolCall objects
        """
        tool_calls = []
        for tc in tool_calls_data:
            # Parse arguments from JSON string
            args = tc.get("function", {}).get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=args,
                )
            )
        return tool_calls

    def _handle_error_response(self, response: httpx.Response, response_data: dict) -> None:
        """Handle error responses from the API.

        Args:
            response: HTTP response object
            response_data: Parsed JSON response

        Raises:
            Appropriate ProviderError subclass
        """
        error = response_data.get("error", {})
        error_message = error.get("message", "Unknown error")
        error_type = error.get("type", "")
        error_code = error.get("code", "")

        if response.status_code == 401:
            raise AuthenticationError(f"Invalid xAI API key: {error_message}")

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            retry_seconds = float(retry_after) if retry_after else None
            raise RateLimitError(
                f"xAI rate limit exceeded: {error_message}",
                retry_after=retry_seconds,
            )

        if response.status_code == 402 or error_code == "insufficient_quota":
            raise QuotaExceededError(f"xAI quota exceeded: {error_message}")

        if response.status_code == 404 or error_type == "invalid_request_error":
            if "model" in error_message.lower():
                raise ModelNotFoundError(f"xAI model not found: {error_message}")

        if error_code == "content_filter" or "content" in error_type.lower():
            raise ContentFilterError(f"Content blocked by xAI safety filters: {error_message}")

        if error_code == "context_length_exceeded":
            raise ContextLengthError(f"Input exceeds xAI context window: {error_message}")

        raise ProviderError(f"xAI API error ({response.status_code}): {error_message}")

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
        """Send a chat completion request to xAI.

        Args:
            messages: List of chat messages forming the conversation
            model: Model ID to use (e.g., "grok-2", "grok-2-vision")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy ("auto", "none", or specific tool)
            stop_sequences: Sequences that stop generation
            **kwargs: Additional arguments passed to the API

        Returns:
            ChatResponse with generated content and metadata

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
            ContentFilterError: If content blocked
            ContextLengthError: If input too long
        """
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
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        # Track timing
        start_time = time.time()

        # Make request
        response = await client.post("/chat/completions", json=payload)
        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            raise ProviderError(f"Invalid JSON response from xAI: {response.text}")

        # Handle errors
        if response.status_code != 200:
            self._handle_error_response(response, response_data)

        # Extract response data
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response_data.get("usage", {})

        # Parse tool calls if present
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = self._parse_tool_calls(message["tool_calls"])

        return ChatResponse(
            content=message.get("content", "") or "",
            model=response_data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw_response=response_data,
            tool_calls=tool_calls,
            system_fingerprint=response_data.get("system_fingerprint"),
            latency_ms=latency_ms,
        )

    async def validate_key(self, api_key: str) -> bool:
        """Validate an xAI API key.

        Args:
            api_key: API key to validate

        Returns:
            True if key is valid, False otherwise
        """
        try:
            # Create a temporary client with the provided key
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                # Make a minimal request to validate the key
                response = await client.get("/models")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all available xAI models.

        Returns:
            List of ModelInfo objects for available models
        """
        # Return predefined models - xAI doesn't have a public models endpoint
        # that returns full details, so we use our curated list
        return list(self.MODELS.values())

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            ModelInfo if found, None otherwise
        """
        # Check direct match
        if model_id in self.MODELS:
            return self.MODELS[model_id]

        # Check aliases
        for model_info in self.MODELS.values():
            if model_id in model_info.aliases:
                return model_info

        return None

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the xAI provider.

        Returns:
            Dictionary with health status information
        """
        try:
            is_valid = await self.validate_key(self.api_key or "")
            return {
                "provider": self.provider_id,
                "display_name": self.display_name,
                "status": "healthy" if is_valid else "unhealthy",
                "authenticated": is_valid,
                "base_url": self.base_url,
                "models_available": len(self.MODELS),
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "display_name": self.display_name,
                "status": "error",
                "authenticated": False,
                "error": str(e),
            }

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"<XAIProvider(base_url='{self.base_url}')>"
