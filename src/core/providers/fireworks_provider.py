"""Fireworks AI provider implementation.

Fireworks AI provides fast inference for open-source models like
Llama and Mixtral with OpenAI-compatible APIs.

RAP-212: Implement Fireworks provider
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
    RateLimitError,
    ToolCall,
)


class FireworksProvider(BaseProvider):
    """Provider for Fireworks AI inference platform.

    Fireworks AI offers high-performance inference for open-source models
    with an OpenAI-compatible API. Known for fast inference times and
    competitive pricing.

    Example usage:
        ```python
        provider = FireworksProvider(api_key="fw_...")

        messages = [
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("Explain quantum computing."),
        ]

        response = await provider.chat(
            messages=messages,
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            temperature=0.7,
        )
        print(response.content)
        ```
    """

    # Provider metadata
    provider_id = "fireworks"
    display_name = "Fireworks AI"
    website = "https://fireworks.ai"
    key_url = "https://fireworks.ai/account/api-keys"
    description = "Fast inference for open-source models like Llama and Mixtral"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.fireworks.ai/inference/v1"
    DEFAULT_TIMEOUT = 120.0
    ENV_KEY_NAME = "FIREWORKS_API_KEY"

    # Model definitions with pricing (as of Jan 2025)
    MODELS = {
        "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelInfo(
            model_id="accounts/fireworks/models/llama-v3p3-70b-instruct",
            provider="fireworks",
            display_name="Llama 3.3 70B Instruct",
            input_price_per_1m=0.90,
            output_price_per_1m=0.90,
            context_window=131072,
            max_output=16384,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="Meta's Llama 3.3 70B instruction-tuned model with 128k context",
            aliases=["llama-v3p3-70b-instruct", "llama-3.3-70b"],
        ),
        "accounts/fireworks/models/mixtral-8x7b-instruct": ModelInfo(
            model_id="accounts/fireworks/models/mixtral-8x7b-instruct",
            provider="fireworks",
            display_name="Mixtral 8x7B Instruct",
            input_price_per_1m=0.50,
            output_price_per_1m=0.50,
            context_window=32768,
            max_output=4096,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="Mistral's Mixture of Experts model with 8x7B parameters",
            aliases=["mixtral-8x7b", "mixtral-8x7b-instruct"],
        ),
        "accounts/fireworks/models/llama-v3p1-8b-instruct": ModelInfo(
            model_id="accounts/fireworks/models/llama-v3p1-8b-instruct",
            provider="fireworks",
            display_name="Llama 3.1 8B Instruct",
            input_price_per_1m=0.20,
            output_price_per_1m=0.20,
            context_window=131072,
            max_output=16384,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.FLASH,
            description="Meta's Llama 3.1 8B instruction-tuned model - fast and efficient",
            aliases=["llama-v3p1-8b-instruct", "llama-3.1-8b"],
        ),
        "accounts/fireworks/models/llama-v3p1-405b-instruct": ModelInfo(
            model_id="accounts/fireworks/models/llama-v3p1-405b-instruct",
            provider="fireworks",
            display_name="Llama 3.1 405B Instruct",
            input_price_per_1m=3.00,
            output_price_per_1m=3.00,
            context_window=131072,
            max_output=16384,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.PREMIUM,
            description="Meta's largest Llama 3.1 model - highest capability",
            aliases=["llama-v3p1-405b-instruct", "llama-3.1-405b"],
        ),
        "accounts/fireworks/models/qwen2p5-72b-instruct": ModelInfo(
            model_id="accounts/fireworks/models/qwen2p5-72b-instruct",
            provider="fireworks",
            display_name="Qwen 2.5 72B Instruct",
            input_price_per_1m=0.90,
            output_price_per_1m=0.90,
            context_window=131072,
            max_output=8192,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="Alibaba's Qwen 2.5 72B instruction-tuned model",
            aliases=["qwen2p5-72b-instruct", "qwen-2.5-72b"],
        ),
        "accounts/fireworks/models/deepseek-v3": ModelInfo(
            model_id="accounts/fireworks/models/deepseek-v3",
            provider="fireworks",
            display_name="DeepSeek V3",
            input_price_per_1m=0.90,
            output_price_per_1m=0.90,
            context_window=131072,
            max_output=8192,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="DeepSeek's V3 model with strong reasoning capabilities",
            aliases=["deepseek-v3"],
        ),
    }

    def __init__(self, api_key: str | None = None):
        """Initialize the Fireworks provider.

        Args:
            api_key: Fireworks API key. If None, reads from FIREWORKS_API_KEY
                     environment variable.
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv(self.ENV_KEY_NAME)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(self.DEFAULT_TIMEOUT),
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

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage objects to Fireworks API format.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of message dictionaries in OpenAI format
        """
        result = []
        for msg in messages:
            message: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                message["name"] = msg.name
            if msg.tool_call_id:
                message["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                message["tool_calls"] = msg.tool_calls
            result.append(message)
        return result

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert tools to Fireworks API format (OpenAI-compatible).

        Args:
            tools: List of tool definitions

        Returns:
            Tools in OpenAI function calling format
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            # Already in OpenAI format
            if "type" in tool and tool["type"] == "function":
                converted.append(tool)
            else:
                # Convert from simple format
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                })
        return converted

    def _handle_error(self, status_code: int, response_data: dict) -> None:
        """Handle API error responses.

        Args:
            status_code: HTTP status code
            response_data: Parsed response JSON

        Raises:
            Appropriate ProviderError subclass
        """
        error = response_data.get("error", {})
        message = error.get("message", str(response_data))

        if status_code == 401:
            raise AuthenticationError(f"Invalid API key: {message}")
        elif status_code == 429:
            retry_after = None
            if "retry-after" in error:
                retry_after = float(error["retry-after"])
            raise RateLimitError(f"Rate limit exceeded: {message}", retry_after)
        elif status_code == 404:
            raise ModelNotFoundError(f"Model not found: {message}")
        elif status_code == 400:
            if "context_length" in message.lower() or "token" in message.lower():
                raise ContextLengthError(f"Context length exceeded: {message}")
            elif "content_filter" in message.lower() or "safety" in message.lower():
                raise ContentFilterError(f"Content blocked: {message}")
            raise ProviderError(f"Bad request: {message}")
        else:
            raise ProviderError(f"API error ({status_code}): {message}")

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
        """Send a chat completion request to Fireworks.

        Args:
            messages: List of chat messages
            model: Model ID (e.g., "accounts/fireworks/models/llama-v3p3-70b-instruct")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            **kwargs: Additional Fireworks-specific parameters

        Returns:
            ChatResponse with generated content

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
        """
        if not self.api_key:
            raise AuthenticationError("Fireworks API key not provided")

        client = await self._get_client()
        start_time = time.time()

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

        # Add any additional kwargs (e.g., top_p, presence_penalty)
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        try:
            response = await client.post("/chat/completions", json=payload)
            response_data = response.json()

            if response.status_code != 200:
                self._handle_error(response.status_code, response_data)

            # Parse response
            choice = response_data["choices"][0]
            message = choice["message"]
            usage = response_data.get("usage", {})

            # Extract tool calls if present
            tool_calls = None
            if message.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                    )
                    for tc in message["tool_calls"]
                ]

            latency_ms = (time.time() - start_time) * 1000

            return ChatResponse(
                content=message.get("content", ""),
                model=response_data.get("model", model),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=response_data,
                tool_calls=tool_calls,
                system_fingerprint=response_data.get("system_fingerprint"),
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException as e:
            raise ProviderError(f"Request timeout: {e}")
        except httpx.RequestError as e:
            raise ProviderError(f"Request failed: {e}")

    async def validate_key(self, api_key: str) -> bool:
        """Validate a Fireworks API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            async with httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                # Try a minimal request to validate the key
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                    },
                )
                return response.status_code == 200

        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all available Fireworks models.

        Returns:
            List of ModelInfo objects for available models
        """
        return list(self.MODELS.values())

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            ModelInfo if found, None otherwise
        """
        # Direct lookup
        if model_id in self.MODELS:
            return self.MODELS[model_id]

        # Check aliases
        for model in self.MODELS.values():
            if model_id in model.aliases:
                return model

        return None

    def resolve_model_id(self, model_id: str) -> str:
        """Resolve a model alias to its full model ID.

        Args:
            model_id: Model ID or alias

        Returns:
            Full model ID for Fireworks API
        """
        # Already a full ID
        if model_id.startswith("accounts/fireworks/models/"):
            return model_id

        # Check if it's an alias
        for full_id, model in self.MODELS.items():
            if model_id in model.aliases or model_id == full_id:
                return full_id

        # Assume it's a shorthand, prepend the standard prefix
        return f"accounts/fireworks/models/{model_id}"
