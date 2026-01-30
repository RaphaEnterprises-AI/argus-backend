"""Ollama provider implementation for local LLM inference.

This module implements the Ollama provider for air-gap deployments
where LLM inference must happen locally without external API calls.

Ollama provides an OpenAI-compatible API for local model inference.

API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md

RAP-299: Air-Gap Foundation - Local LLM Integration
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
    RateLimitError,
    ToolCall,
    validate_temperature,
)

logger = structlog.get_logger()


# Common Ollama models with estimated capabilities
# Note: Actual models depend on what's pulled locally
OLLAMA_COMMON_MODELS: list[ModelInfo] = [
    ModelInfo(
        model_id="llama3.1:8b",
        provider="ollama",
        display_name="Llama 3.1 8B",
        input_price_per_1m=0.0,  # Local inference - no cost
        output_price_per_1m=0.0,
        context_window=128000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="Meta Llama 3.1 8B - Fast, efficient local model (8GB VRAM)",
        aliases=["llama3.1", "llama-3.1-8b"],
    ),
    ModelInfo(
        model_id="llama3.1:70b",
        provider="ollama",
        display_name="Llama 3.1 70B",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=128000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.PREMIUM,
        description="Meta Llama 3.1 70B - High-quality local model (48GB VRAM)",
        aliases=["llama3.1-70b", "llama-3.1-70b"],
    ),
    ModelInfo(
        model_id="codellama:34b",
        provider="ollama",
        display_name="Code Llama 34B",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=16384,
        max_output=4096,
        supports_vision=False,
        supports_tools=False,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="Meta Code Llama 34B - Code-specialized model (24GB VRAM)",
        aliases=["codellama", "code-llama-34b"],
    ),
    ModelInfo(
        model_id="mistral:7b",
        provider="ollama",
        display_name="Mistral 7B",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=32768,
        max_output=4096,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="Mistral 7B - Fast, efficient general-purpose model (8GB VRAM)",
        aliases=["mistral", "mistral-7b"],
    ),
    ModelInfo(
        model_id="mixtral:8x7b",
        provider="ollama",
        display_name="Mixtral 8x7B",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=32768,
        max_output=4096,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="Mixtral 8x7B MoE - High-quality mixture-of-experts (48GB VRAM)",
        aliases=["mixtral", "mixtral-8x7b"],
    ),
    ModelInfo(
        model_id="qwen2.5:72b",
        provider="ollama",
        display_name="Qwen 2.5 72B",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=131072,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.PREMIUM,
        description="Alibaba Qwen 2.5 72B - Excellent multilingual support (48GB VRAM)",
        aliases=["qwen2.5", "qwen-2.5-72b"],
    ),
    ModelInfo(
        model_id="deepseek-r1:70b",
        provider="ollama",
        display_name="DeepSeek R1 70B",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=64000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.PREMIUM,
        description="DeepSeek R1 70B - Advanced reasoning model (48GB VRAM)",
        aliases=["deepseek-r1", "deepseek-r1-70b"],
    ),
    # Embedding model
    ModelInfo(
        model_id="nomic-embed-text",
        provider="ollama",
        display_name="Nomic Embed Text",
        input_price_per_1m=0.0,
        output_price_per_1m=0.0,
        context_window=8192,
        max_output=0,  # Embedding model
        supports_vision=False,
        supports_tools=False,
        supports_streaming=False,
        supports_computer_use=False,
        tier=ModelTier.FLASH,
        description="Nomic AI embedding model - 768-dimensional vectors",
        aliases=["nomic", "nomic-embed"],
    ),
]


class OllamaProvider(BaseProvider):
    """Ollama provider for local LLM inference.

    Ollama enables running large language models locally without
    external API calls, making it ideal for:
    - Air-gap deployments with no internet connectivity
    - Data sovereignty requirements
    - Cost-sensitive workloads (no per-token charges)
    - Low-latency inference on local hardware

    Ollama uses an OpenAI-compatible API format.

    Example:
        ```python
        provider = OllamaProvider(base_url="http://localhost:11434")

        # Chat completion
        response = await provider.chat(
            messages=[ChatMessage.user("Explain quantum computing")],
            model="llama3.1:70b",
        )

        # Generate embeddings
        embeddings = await provider.embed("Hello, world!")
        ```

    GPU Requirements:
        - llama3.1:8b: 8GB VRAM (RTX 4070, A4000)
        - llama3.1:70b: 48GB VRAM (A100 80GB, 2x A10)
        - codellama:34b: 24GB VRAM (A10, A6000)
        - mixtral:8x7b: 48GB VRAM (A100 80GB)
    """

    # Provider metadata
    provider_id = "ollama"
    display_name = "Ollama (Local)"
    website = "https://ollama.ai"
    key_url = ""  # No API key needed for local
    description = "Ollama - Local LLM inference for air-gap deployments"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = False  # Some models support it, but not all
    supports_computer_use = False
    is_aggregator = False

    # API configuration defaults
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # Local inference can be slower
    MAX_RETRIES = 2  # Fewer retries for local

    def __init__(
        self,
        base_url: str | None = None,
        config: ProviderConfig | None = None,
    ):
        """Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL. Defaults to OLLAMA_HOST env var
                     or http://localhost:11434.
            config: Optional provider configuration for advanced settings.
        """
        # No API key needed for Ollama
        super().__init__(api_key=None)

        # Get base URL from argument, config, or environment
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif config and config.base_url:
            self.base_url = config.base_url.rstrip("/")
        else:
            self.base_url = os.environ.get("OLLAMA_HOST", self.DEFAULT_BASE_URL).rstrip("/")

        # Apply config settings
        self.timeout = (config.timeout if config else None) or self.DEFAULT_TIMEOUT
        self.max_retries = (config.max_retries if config else None) or self.MAX_RETRIES
        self.custom_headers = (config.custom_headers if config else None) or {}

        # Default model from environment
        self.default_model = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")

        # HTTP client (created lazily)
        self._client: httpx.AsyncClient | None = None

        logger.info(
            "Ollama provider initialized",
            base_url=self.base_url,
            default_model=self.default_model,
        )

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

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to Ollama API format.

        Ollama uses OpenAI-compatible message format.
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
            ModelNotFoundError: If model doesn't exist
            ContextLengthError: If input too long
            ProviderError: For other errors
        """
        status = response.status_code
        error_msg = "Unknown error"

        if response_data and "error" in response_data:
            error_msg = response_data["error"]
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))

            # Check for specific error types
            if "not found" in error_msg.lower() or "unknown model" in error_msg.lower():
                raise ModelNotFoundError(f"Model not found: {error_msg}. Try 'ollama pull <model>'")
            if "context length" in error_msg.lower() or "too long" in error_msg.lower():
                raise ContextLengthError(error_msg)

        if status == 404:
            raise ModelNotFoundError(f"Model not found: {error_msg}")
        elif status == 400:
            raise ProviderError(f"Bad request: {error_msg}")
        elif status >= 500:
            raise ProviderError(f"Ollama server error ({status}): {error_msg}")
        else:
            raise ProviderError(f"Ollama error ({status}): {error_msg}")

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama.

        Uses the OpenAI-compatible /v1/chat/completions endpoint.

        Args:
            messages: List of chat messages forming the conversation
            model: Model ID (e.g., llama3.1:70b). Defaults to OLLAMA_MODEL env var.
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            **kwargs: Additional provider-specific arguments

        Returns:
            ChatResponse with generated content and metadata

        Raises:
            ModelNotFoundError: If model doesn't exist locally
            ContextLengthError: If input too long
            ProviderError: For other errors
        """
        client = await self._get_client()
        start_time = time.time()

        # Use default model if not specified
        if not model:
            model = self.default_model

        # Validate and clamp temperature
        validated_temp = validate_temperature(temperature)

        # Build request payload (OpenAI-compatible format)
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": validated_temp,
            "stream": False,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Add Ollama-specific options
        if kwargs.get("num_ctx"):
            payload["options"] = payload.get("options", {})
            payload["options"]["num_ctx"] = kwargs["num_ctx"]

        if kwargs.get("num_predict"):
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = kwargs["num_predict"]

        # Make request with retry logic
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                # Use OpenAI-compatible endpoint
                response = await client.post(
                    "/v1/chat/completions",
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
                    raise ProviderError("Invalid response format from Ollama")

                choice = response_data["choices"][0]
                message = choice.get("message", {})
                usage = response_data.get("usage", {})

                latency_ms = (time.time() - start_time) * 1000

                return ChatResponse(
                    content=message.get("content", ""),
                    model=response_data.get("model", model),
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", "stop"),
                    tool_calls=self._parse_tool_calls(message.get("tool_calls")),
                    latency_ms=latency_ms,
                    raw_response=response_data,
                )

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Ollama request timeout, retrying",
                        attempt=attempt + 1,
                        model=model,
                    )
                else:
                    raise ProviderError(
                        f"Ollama request timed out after {self.max_retries} attempts. "
                        "Model may be loading or GPU is busy."
                    )

            except httpx.ConnectError as e:
                raise ProviderError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Ensure Ollama is running: 'ollama serve'"
                ) from e

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Ollama request error, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                else:
                    raise ProviderError(f"Ollama request failed: {str(e)}")

        raise ProviderError(f"Ollama request failed after {self.max_retries} attempts: {last_error}")

    async def embed(
        self,
        text: str | list[str],
        model: str = "nomic-embed-text",
    ) -> list[float] | list[list[float]]:
        """Generate embeddings using Ollama.

        Args:
            text: Text or list of texts to embed
            model: Embedding model to use (default: nomic-embed-text)

        Returns:
            Embedding vector(s) as list of floats

        Raises:
            ModelNotFoundError: If embedding model not found
            ProviderError: For other errors
        """
        client = await self._get_client()

        # Handle both single text and list
        texts = [text] if isinstance(text, str) else text

        try:
            response = await client.post(
                "/api/embed",
                json={
                    "model": model,
                    "input": texts,
                },
            )

            if response.status_code != 200:
                self._handle_error_response(response)

            data = response.json()
            embeddings = data.get("embeddings", [])

            # Return single embedding if input was single text
            if isinstance(text, str) and embeddings:
                return embeddings[0]
            return embeddings

        except httpx.ConnectError as e:
            raise ProviderError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running: 'ollama serve'"
            ) from e

    async def validate_key(self, api_key: str) -> bool:
        """Validate Ollama connectivity.

        Ollama doesn't use API keys, so this checks if the server is reachable.

        Args:
            api_key: Ignored for Ollama

        Returns:
            True if Ollama server is reachable, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List models available in the local Ollama instance.

        Queries the Ollama server for installed models and combines
        with common model metadata.

        Returns:
            List of ModelInfo for available models
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")

            if response.status_code != 200:
                logger.warning("Failed to list Ollama models, returning common models")
                return OLLAMA_COMMON_MODELS.copy()

            data = response.json()
            installed_models = data.get("models", [])

            # Build model list from installed models
            models: list[ModelInfo] = []
            for m in installed_models:
                model_name = m.get("name", "")

                # Check if we have metadata for this model
                known_model = None
                for common in OLLAMA_COMMON_MODELS:
                    if model_name == common.model_id or model_name in common.aliases:
                        known_model = common
                        break

                if known_model:
                    models.append(known_model)
                else:
                    # Create basic ModelInfo for unknown models
                    models.append(
                        ModelInfo(
                            model_id=model_name,
                            provider="ollama",
                            display_name=model_name,
                            input_price_per_1m=0.0,
                            output_price_per_1m=0.0,
                            context_window=4096,  # Conservative default
                            max_output=2048,
                            supports_vision=False,
                            supports_tools=False,
                            supports_streaming=True,
                            tier=ModelTier.VALUE,
                            description=f"Local Ollama model: {model_name}",
                        )
                    )

            return models if models else OLLAMA_COMMON_MODELS.copy()

        except Exception as e:
            logger.warning(f"Error listing Ollama models: {e}")
            return OLLAMA_COMMON_MODELS.copy()

    async def pull_model(self, model: str) -> bool:
        """Pull a model from the Ollama library.

        Args:
            model: Model name to pull (e.g., "llama3.1:70b")

        Returns:
            True if pull succeeded, False otherwise
        """
        try:
            client = await self._get_client()

            # Increase timeout for model download
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(3600.0),  # 1 hour for large models
            ) as pull_client:
                response = await pull_client.post(
                    "/api/pull",
                    json={"name": model, "stream": False},
                )

                if response.status_code == 200:
                    logger.info(f"Successfully pulled model: {model}")
                    return True
                else:
                    logger.error(f"Failed to pull model {model}: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False

    def get_capabilities(self) -> list[ProviderCapability]:
        """Get the capabilities supported by this provider.

        Returns:
            List of ProviderCapability enum values
        """
        return [
            ProviderCapability.CHAT,
            ProviderCapability.STREAMING,
            ProviderCapability.TOOLS,
            ProviderCapability.EMBEDDINGS,
        ]

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on Ollama.

        Returns:
            Dictionary with health status and available models
        """
        try:
            is_reachable = await self.validate_key("")
            models = await self.list_models() if is_reachable else []

            return {
                "provider": self.provider_id,
                "status": "healthy" if is_reachable else "unhealthy",
                "base_url": self.base_url,
                "authenticated": True,  # No auth needed
                "available_models": len(models),
                "models": [m.model_id for m in models],
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "status": "error",
                "base_url": self.base_url,
                "authenticated": False,
                "error": str(e),
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<OllamaProvider("
            f"base_url='{self.base_url}', "
            f"default_model='{self.default_model}')>"
        )

    async def __aenter__(self) -> "OllamaProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


def get_ollama_provider(
    base_url: str | None = None,
    config: ProviderConfig | None = None,
) -> OllamaProvider:
    """Factory function to create an Ollama provider.

    Args:
        base_url: Ollama server URL (defaults to OLLAMA_HOST env var)
        config: Optional provider configuration

    Returns:
        Configured OllamaProvider instance
    """
    return OllamaProvider(base_url=base_url, config=config)
