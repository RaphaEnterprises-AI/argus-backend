"""DeepSeek AI provider implementation.

This module implements the DeepSeek provider using their OpenAI-compatible API.
Supports both DeepSeek V3 (chat) and DeepSeek R1 (reasoning) models.

API Documentation: https://platform.deepseek.com/api-docs
"""

import os
import time
from typing import Any

import httpx
import structlog

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
    ProviderCapability,
    ProviderConfig,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ToolCall,
    mask_api_key,
    validate_temperature,
)

logger = structlog.get_logger()


# DeepSeek model definitions with pricing
DEEPSEEK_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="deepseek-chat",
        provider="deepseek",
        display_name="DeepSeek V3",
        input_price_per_1m=0.14,
        output_price_per_1m=0.28,
        context_window=64000,  # 64K context
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="DeepSeek V3 - High-performance chat model with excellent cost efficiency",
        aliases=["deepseek-v3", "deepseek"],
    ),
    ModelInfo(
        model_id="deepseek-reasoner",
        provider="deepseek",
        display_name="DeepSeek R1",
        input_price_per_1m=0.55,
        output_price_per_1m=2.19,
        context_window=64000,  # 64K context
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="DeepSeek R1 - Advanced reasoning model with chain-of-thought capabilities",
        aliases=["deepseek-r1", "deepseek-reasoning"],
    ),
]


class DeepSeekProvider(BaseProvider):
    """DeepSeek AI provider using OpenAI-compatible API.

    DeepSeek offers high-performance models at competitive prices:
    - deepseek-chat (V3): General-purpose chat, $0.14/1M input, $0.28/1M output
    - deepseek-reasoner (R1): Reasoning model, $0.55/1M input, $2.19/1M output

    The R1 model supports extended reasoning/thinking mode for complex problems.

    Example:
        ```python
        provider = DeepSeekProvider(api_key="your-api-key")

        # Standard chat
        response = await provider.chat(
            messages=[ChatMessage.user("Explain quantum computing")],
            model="deepseek-chat",
        )

        # Reasoning mode with R1
        response = await provider.chat(
            messages=[ChatMessage.user("Solve this math problem...")],
            model="deepseek-reasoner",
            reasoning=True,  # Enable extended thinking
        )
        ```
    """

    # Provider metadata
    provider_id = "deepseek"
    display_name = "DeepSeek"
    website = "https://deepseek.com"
    key_url = "https://platform.deepseek.com/api_keys"
    description = "DeepSeek AI - High-performance models with excellent cost efficiency"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = False
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
    ):
        """Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
            config: Optional provider configuration for advanced settings.
        """
        # Get API key from argument, config, or environment
        if api_key:
            self.api_key = api_key
        elif config and config.api_key:
            self.api_key = config.api_key
        else:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")

        super().__init__(api_key=self.api_key)

        # Apply config settings
        self.base_url = (config.base_url if config else None) or self.BASE_URL
        self.timeout = (config.timeout if config else None) or self.DEFAULT_TIMEOUT
        self.max_retries = (config.max_retries if config else None) or self.MAX_RETRIES
        self.custom_headers = (config.custom_headers if config else None) or {}

        # HTTP client (created lazily)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Content-Type": "application/json",
                    **self.custom_headers,
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        if not self.api_key:
            raise AuthenticationError("DeepSeek API key not configured")
        return {"Authorization": f"Bearer {self.api_key}"}

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to DeepSeek API format.

        DeepSeek uses OpenAI-compatible message format.
        """
        result = []
        for msg in messages:
            message_dict: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                message_dict["name"] = msg.name
            if msg.tool_call_id:
                message_dict["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls
            result.append(message_dict)
        return result

    def _parse_tool_calls(
        self, tool_calls_data: list[dict] | None
    ) -> list[ToolCall] | None:
        """Parse tool calls from API response."""
        if not tool_calls_data:
            return None

        tool_calls = []
        for tc in tool_calls_data:
            import json

            # Parse function arguments from JSON string
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
        return tool_calls if tool_calls else None

    def _handle_error_response(
        self, response: httpx.Response, response_data: dict | None = None
    ) -> None:
        """Handle error responses from the API.

        Args:
            response: The HTTP response
            response_data: Parsed JSON response if available

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            QuotaExceededError: If usage quota exceeded
            ModelNotFoundError: If model doesn't exist
            ContentFilterError: If content was filtered
            ContextLengthError: If input too long
            ProviderError: For other errors
        """
        status = response.status_code
        error_msg = "Unknown error"

        if response_data and "error" in response_data:
            error_info = response_data["error"]
            error_msg = error_info.get("message", str(error_info))
            error_type = error_info.get("type", "")
            error_code = error_info.get("code", "")

            # Check for specific error types
            if "context_length" in error_msg.lower() or error_code == "context_length_exceeded":
                raise ContextLengthError(error_msg)
            if "content_filter" in error_msg.lower() or error_type == "content_filter":
                raise ContentFilterError(error_msg)
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ModelNotFoundError(error_msg)

        if status == 401:
            raise AuthenticationError(f"Invalid DeepSeek API key: {error_msg}")
        elif status == 403:
            raise AuthenticationError(f"API key lacks permissions: {error_msg}")
        elif status == 429:
            # Check if it's rate limit or quota
            retry_after = response.headers.get("retry-after")
            retry_seconds = float(retry_after) if retry_after else None

            if "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                raise QuotaExceededError(error_msg)
            raise RateLimitError(error_msg, retry_after=retry_seconds)
        elif status == 400:
            raise ProviderError(f"Bad request: {error_msg}")
        elif status == 404:
            raise ModelNotFoundError(error_msg)
        elif status >= 500:
            raise ProviderError(f"DeepSeek server error ({status}): {error_msg}")
        else:
            raise ProviderError(f"DeepSeek API error ({status}): {error_msg}")

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        reasoning: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to DeepSeek.

        Args:
            messages: List of chat messages forming the conversation
            model: Model ID (deepseek-chat or deepseek-reasoner)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            reasoning: Enable reasoning mode for R1 model (extended thinking)
            **kwargs: Additional provider-specific arguments

        Returns:
            ChatResponse with generated content and metadata

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
            ProviderError: For other errors
        """
        client = await self._get_client()
        start_time = time.time()

        # Validate and clamp temperature to valid range (0.0-2.0)
        validated_temp = validate_temperature(temperature)

        # Build request payload (OpenAI-compatible format)
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": validated_temp,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Enable reasoning mode for R1 model
        # DeepSeek R1 supports extended thinking with special parameters
        if reasoning and "reasoner" in model.lower():
            # R1 reasoning mode - may use specific parameters
            # Note: Exact parameter names may vary based on DeepSeek's API updates
            payload["reasoning_effort"] = kwargs.get("reasoning_effort", "medium")

        # Add any extra kwargs
        for key, value in kwargs.items():
            if key not in payload and key not in ("reasoning_effort",):
                payload[key] = value

        # Make request with retry logic
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    "/chat/completions",
                    headers=self._get_auth_headers(),
                    json=payload,
                )

                # Parse response
                try:
                    response_data = response.json()
                except Exception:
                    response_data = None

                # Handle errors
                if response.status_code != 200:
                    self._handle_error_response(response, response_data)

                # Extract response data
                if not response_data or "choices" not in response_data:
                    raise ProviderError("Invalid response format from DeepSeek")

                choice = response_data["choices"][0]
                message = choice.get("message", {})
                usage = response_data.get("usage", {})

                latency_ms = (time.time() - start_time) * 1000

                # Extract reasoning content if present (R1 model)
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content")
                if reasoning_content:
                    # Prepend reasoning to content for transparency
                    content = f"<reasoning>\n{reasoning_content}\n</reasoning>\n\n{content}"

                return ChatResponse(
                    content=content,
                    model=response_data.get("model", model),
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                    tool_calls=self._parse_tool_calls(message.get("tool_calls")),
                    system_fingerprint=response_data.get("system_fingerprint"),
                    latency_ms=latency_ms,
                    raw_response=response_data,
                )

            except (RateLimitError, QuotaExceededError) as e:
                last_error = e
                # Retry with exponential backoff for rate limits
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) * 1.0
                    if isinstance(e, RateLimitError) and e.retry_after:
                        wait_time = e.retry_after
                    logger.warning(
                        "Rate limited, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    import asyncio
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Request timeout, retrying",
                        attempt=attempt + 1,
                    )
                else:
                    raise ProviderError(f"DeepSeek request timed out after {self.max_retries} attempts")

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Request error, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                else:
                    raise ProviderError(f"DeepSeek request failed: {str(e)}")

        # Should not reach here, but just in case
        raise ProviderError(f"DeepSeek request failed after {self.max_retries} attempts: {last_error}")

    async def validate_key(self, api_key: str) -> bool:
        """Validate a DeepSeek API key.

        Makes a lightweight models list request to verify the key works.

        Args:
            api_key: API key to validate

        Returns:
            True if key is valid, False otherwise
        """
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=10.0,
            ) as client:
                response = await client.get(
                    "/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )

                if response.status_code == 200:
                    return True
                elif response.status_code in (401, 403):
                    return False
                else:
                    # Other errors - might be temporary, log but return False
                    logger.warning(
                        "Unexpected response during key validation",
                        status=response.status_code,
                    )
                    return False

        except httpx.TimeoutException:
            logger.warning("Timeout during DeepSeek key validation")
            return False
        except httpx.RequestError as e:
            logger.warning("Request error during DeepSeek key validation", error=str(e))
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List available DeepSeek models.

        Returns the statically defined models with current pricing.
        DeepSeek's model list is relatively stable.

        Returns:
            List of ModelInfo for available models
        """
        # Return cached static model definitions
        # These are well-defined and don't change frequently
        return DEEPSEEK_MODELS.copy()

    def get_capabilities(self) -> list[ProviderCapability]:
        """Get the capabilities supported by this provider.

        Returns:
            List of ProviderCapability enum values
        """
        capabilities = [
            ProviderCapability.CHAT,
            ProviderCapability.STREAMING,
            ProviderCapability.TOOLS,
            ProviderCapability.JSON_MODE,
        ]

        # R1 model supports reasoning
        capabilities.append(ProviderCapability.REASONING)

        return capabilities

    def is_reasoning_model(self, model: str) -> bool:
        """Check if a model supports reasoning mode.

        Args:
            model: Model ID to check

        Returns:
            True if model supports extended reasoning
        """
        return "reasoner" in model.lower() or "r1" in model.lower()

    def __repr__(self) -> str:
        """String representation with masked API key for security."""
        return (
            f"<DeepSeekProvider("
            f"api_key={mask_api_key(self.api_key)}, "
            f"base_url='{self.base_url}')>"
        )

    async def __aenter__(self) -> "DeepSeekProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
