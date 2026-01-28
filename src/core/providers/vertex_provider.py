"""Google Vertex AI Provider for Gemini models.

This module provides integration with Google Cloud Vertex AI for accessing
Gemini models using service account authentication. It supports the full
range of Gemini capabilities including vision, tool use, and streaming.

Setup:
    1. Create a GCP project and enable Vertex AI API
    2. Create a service account with Vertex AI User role
    3. Download the service account JSON key
    4. Set environment variables:
       - GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
       - VERTEX_PROJECT_ID=your-gcp-project-id
       - VERTEX_LOCATION=us-central1 (or your preferred region)

Example usage:
    ```python
    from src.core.providers.vertex_provider import VertexProvider

    provider = VertexProvider(
        project_id="my-project",
        location="us-central1",
        credentials_path="/path/to/service-account.json",
    )

    # List available models
    models = await provider.list_models()

    # Chat completion
    messages = [ChatMessage.user("Hello, how are you?")]
    response = await provider.chat(
        messages=messages,
        model="gemini-2.0-flash-001",
    )
    print(response.content)
    ```

Supported Models (as of Jan 2026):
    - gemini-2.5-pro (latest, best quality)
    - gemini-2.5-flash (fast, good quality)
    - gemini-2.0-pro-exp (experimental)
    - gemini-2.0-flash-001 (production)
    - gemini-1.5-pro-002 (stable)
    - gemini-1.5-flash-002 (stable)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any

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
    ProviderError,
    RateLimitError,
    ToolCall,
    mask_api_key,
    validate_temperature,
)

logger = structlog.get_logger()

# Gemini model pricing (USD per 1M tokens) - as of Jan 2026
# Prices from: https://cloud.google.com/vertex-ai/generative-ai/pricing
GEMINI_PRICING = {
    # Gemini 2.5 models
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-2.5-pro-preview": {"input": 1.25, "output": 5.00},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-preview": {"input": 0.075, "output": 0.30},
    # Gemini 2.0 models
    "gemini-2.0-pro": {"input": 1.00, "output": 4.00},
    "gemini-2.0-pro-exp": {"input": 0.00, "output": 0.00},  # Free during preview
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Free during preview
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    # Gemini 1.5 models
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro-002": {"input": 1.25, "output": 5.00},
    "gemini-1.5-pro-001": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
    # Default fallback
    "default": {"input": 0.10, "output": 0.40},
}

# Model capabilities
GEMINI_CAPABILITIES = {
    "gemini-2.5-pro": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": True,
        "context_window": 1048576,
        "max_output": 65536,
    },
    "gemini-2.5-flash": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": False,
        "context_window": 1048576,
        "max_output": 65536,
    },
    "gemini-2.0-pro": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": True,
        "context_window": 2097152,
        "max_output": 8192,
    },
    "gemini-2.0-flash": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": False,
        "context_window": 1048576,
        "max_output": 8192,
    },
    "gemini-1.5-pro": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": False,
        "context_window": 2097152,
        "max_output": 8192,
    },
    "gemini-1.5-flash": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": False,
        "context_window": 1048576,
        "max_output": 8192,
    },
    "default": {
        "vision": True,
        "tools": True,
        "streaming": True,
        "computer_use": False,
        "context_window": 128000,
        "max_output": 8192,
    },
}


def _get_model_tier(model_id: str) -> ModelTier:
    """Determine the tier for a Gemini model based on its ID."""
    model_lower = model_id.lower()
    if "pro" in model_lower:
        if "2.5" in model_lower or "2.0" in model_lower:
            return ModelTier.PREMIUM
        return ModelTier.STANDARD
    elif "flash" in model_lower:
        if "lite" in model_lower:
            return ModelTier.FLASH
        return ModelTier.VALUE
    return ModelTier.STANDARD


def _get_pricing(model_id: str) -> tuple[float, float]:
    """Get input and output pricing for a model."""
    # Try exact match first
    if model_id in GEMINI_PRICING:
        p = GEMINI_PRICING[model_id]
        return p["input"], p["output"]

    # Try base model name (strip version suffix)
    base_name = model_id.rsplit("-", 1)[0] if "-" in model_id else model_id
    if base_name in GEMINI_PRICING:
        p = GEMINI_PRICING[base_name]
        return p["input"], p["output"]

    # Check if it's a pro or flash model by name
    model_lower = model_id.lower()
    if "2.5-pro" in model_lower:
        p = GEMINI_PRICING["gemini-2.5-pro"]
    elif "2.5-flash" in model_lower:
        p = GEMINI_PRICING["gemini-2.5-flash"]
    elif "2.0-pro" in model_lower:
        p = GEMINI_PRICING["gemini-2.0-pro"]
    elif "2.0-flash" in model_lower:
        p = GEMINI_PRICING["gemini-2.0-flash"]
    elif "1.5-pro" in model_lower:
        p = GEMINI_PRICING["gemini-1.5-pro"]
    elif "1.5-flash" in model_lower:
        p = GEMINI_PRICING["gemini-1.5-flash"]
    else:
        p = GEMINI_PRICING["default"]

    return p["input"], p["output"]


def _get_capabilities(model_id: str) -> dict[str, Any]:
    """Get capabilities for a model."""
    # Try exact match
    if model_id in GEMINI_CAPABILITIES:
        return GEMINI_CAPABILITIES[model_id]

    # Try base model name
    base_name = model_id.rsplit("-", 1)[0] if "-" in model_id else model_id
    if base_name in GEMINI_CAPABILITIES:
        return GEMINI_CAPABILITIES[base_name]

    # Match by pattern
    model_lower = model_id.lower()
    for key in GEMINI_CAPABILITIES:
        if key in model_lower:
            return GEMINI_CAPABILITIES[key]

    return GEMINI_CAPABILITIES["default"]


class VertexProvider(BaseProvider):
    """Google Vertex AI provider for Gemini models.

    Uses the google-cloud-aiplatform SDK with service account authentication
    for enterprise-grade access to Gemini models.

    Attributes:
        provider_id: "vertex_ai"
        display_name: "Google Vertex AI"
        project_id: GCP project ID
        location: GCP region (e.g., "us-central1")
    """

    # Provider metadata
    provider_id: str = "vertex_ai"
    display_name: str = "Google Vertex AI"
    website: str = "https://cloud.google.com/vertex-ai"
    key_url: str = "https://console.cloud.google.com/apis/credentials"
    description: str = (
        "Access Gemini models through Google Cloud Vertex AI with enterprise "
        "features including VPC-SC, IAM, audit logging, and regional data residency."
    )

    # Capability flags
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = True
    supports_computer_use: bool = True  # Gemini 2.0 Pro+ supports this
    is_aggregator: bool = False

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        credentials_path: str | None = None,
        api_key: str | None = None,  # Not used, but required by base class
    ):
        """Initialize the Vertex AI provider.

        Args:
            project_id: GCP project ID. If None, uses VERTEX_PROJECT_ID env var.
            location: GCP region. If None, uses VERTEX_LOCATION env var or "us-central1".
            credentials_path: Path to service account JSON. If None, uses
                            GOOGLE_APPLICATION_CREDENTIALS env var.
            api_key: Not used for Vertex AI (uses service account instead).
        """
        super().__init__(api_key=api_key)

        # Load configuration from environment if not provided
        self.project_id = project_id or os.environ.get("VERTEX_PROJECT_ID")
        self.location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self.credentials_path = credentials_path or os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        # Validate required configuration
        if not self.project_id:
            raise AuthenticationError(
                "VERTEX_PROJECT_ID environment variable must be set, "
                "or pass project_id to VertexProvider"
            )

        # Initialize client lazily
        self._vertex_client: Any = None
        self._generative_model_class: Any = None

        logger.info(
            "Initialized Vertex AI provider",
            project_id=self.project_id,
            location=self.location,
            credentials_path=self.credentials_path[:20] + "..."
            if self.credentials_path
            else None,
        )

    def _ensure_client(self) -> None:
        """Ensure the Vertex AI client is initialized."""
        if self._vertex_client is not None:
            return

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            # Initialize Vertex AI with project and location
            vertexai.init(
                project=self.project_id,
                location=self.location,
            )

            self._generative_model_class = GenerativeModel
            self._vertex_client = True  # Mark as initialized

            logger.debug(
                "Vertex AI client initialized",
                project_id=self.project_id,
                location=self.location,
            )

        except ImportError as e:
            raise ProviderError(
                "google-cloud-aiplatform is not installed. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e
        except Exception as e:
            logger.error("Failed to initialize Vertex AI", error=str(e))
            raise AuthenticationError(
                f"Failed to initialize Vertex AI: {e}"
            ) from e

    def _get_model(self, model_id: str) -> Any:
        """Get a GenerativeModel instance for the given model ID.

        Args:
            model_id: Model ID (e.g., "gemini-2.0-flash-001")

        Returns:
            GenerativeModel instance
        """
        self._ensure_client()
        return self._generative_model_class(model_id)

    def _convert_messages_to_vertex_format(
        self,
        messages: list[ChatMessage],
        system_instruction: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert ChatMessage list to Vertex AI format.

        Args:
            messages: List of ChatMessage objects
            system_instruction: Optional system instruction

        Returns:
            Tuple of (contents list, system instruction)
        """
        contents = []
        extracted_system = system_instruction

        for msg in messages:
            if msg.role == "system":
                # Extract system instruction
                if isinstance(msg.content, str):
                    extracted_system = msg.content
                else:
                    # Handle list content (take first text part)
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            extracted_system = part.get("text", "")
                            break
                continue

            # Map roles
            role = "user" if msg.role == "user" else "model"

            # Handle content
            if isinstance(msg.content, str):
                parts = [{"text": msg.content}]
            else:
                # Handle multimodal content
                parts = []
                for item in msg.content:
                    if isinstance(item, str):
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image":
                            # Handle base64 image
                            source = item.get("source", {})
                            if source.get("type") == "base64":
                                parts.append({
                                    "inline_data": {
                                        "mime_type": source.get("media_type", "image/png"),
                                        "data": source.get("data", ""),
                                    }
                                })
                        elif item.get("type") == "image_url":
                            # Handle URL-based image
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Parse data URL
                                import base64
                                parts_split = url.split(",", 1)
                                if len(parts_split) == 2:
                                    header, data = parts_split
                                    mime_type = header.split(";")[0].replace("data:", "")
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": data,
                                        }
                                    })

            contents.append({"role": role, "parts": parts})

        return contents, extracted_system

    def _convert_tools_to_vertex_format(
        self,
        tools: list[dict] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert tools to Vertex AI format.

        Args:
            tools: List of tool definitions in standard format

        Returns:
            List of tools in Vertex AI format, or None
        """
        if not tools:
            return None

        vertex_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                vertex_tools.append({
                    "function_declarations": [{
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }]
                })
            elif "name" in tool:
                # Direct tool definition (Anthropic-style)
                vertex_tools.append({
                    "function_declarations": [{
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {})),
                    }]
                })

        return vertex_tools if vertex_tools else None

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
        """Send a chat completion request to Vertex AI.

        Args:
            messages: List of chat messages
            model: Gemini model ID (e.g., "gemini-2.0-flash-001")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            **kwargs: Additional arguments (system_instruction, safety_settings, etc.)

        Returns:
            ChatResponse with generated content

        Raises:
            AuthenticationError: If credentials are invalid
            ModelNotFoundError: If model doesn't exist
            RateLimitError: If rate limit exceeded
            ContentFilterError: If content blocked
            ContextLengthError: If input too long
        """
        self._ensure_client()
        start_time = time.time()

        try:
            # Get model instance
            generative_model = self._get_model(model)

            # Convert messages
            contents, system_instruction = self._convert_messages_to_vertex_format(
                messages,
                system_instruction=kwargs.get("system_instruction"),
            )

            # Build generation config
            generation_config: dict[str, Any] = {}
            if temperature is not None:
                generation_config["temperature"] = temperature
            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            # Handle JSON mode
            if kwargs.get("json_mode"):
                generation_config["response_mime_type"] = "application/json"

            # Convert tools
            vertex_tools = self._convert_tools_to_vertex_format(tools)

            # Create model with system instruction if provided
            if system_instruction:
                generative_model = self._generative_model_class(
                    model_name=model,
                    system_instruction=system_instruction,
                )

            # Make the API call in a thread pool (SDK is synchronous)
            loop = asyncio.get_event_loop()

            def _generate():
                return generative_model.generate_content(
                    contents,
                    generation_config=generation_config if generation_config else None,
                    tools=vertex_tools,
                    safety_settings=kwargs.get("safety_settings"),
                )

            response = await loop.run_in_executor(None, _generate)

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            content = ""
            tool_calls_list: list[ToolCall] | None = None
            finish_reason = "stop"

            # Handle candidates
            if response.candidates:
                candidate = response.candidates[0]

                # Get finish reason
                if hasattr(candidate, "finish_reason"):
                    finish_reason_value = candidate.finish_reason
                    # Convert enum to string if needed
                    if hasattr(finish_reason_value, "name"):
                        finish_reason = finish_reason_value.name.lower()
                    else:
                        finish_reason = str(finish_reason_value).lower()

                # Extract content from parts
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            content += part.text
                        elif hasattr(part, "function_call"):
                            # Handle function calls
                            fc = part.function_call
                            if tool_calls_list is None:
                                tool_calls_list = []
                            tool_calls_list.append(
                                ToolCall(
                                    id=f"call_{len(tool_calls_list)}",
                                    name=fc.name,
                                    arguments=dict(fc.args) if fc.args else {},
                                )
                            )

            # Get token counts
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            return ChatResponse(
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                tool_calls=tool_calls_list,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            error_str = str(e).lower()

            # Map errors to appropriate exceptions
            if "permission" in error_str or "credentials" in error_str:
                raise AuthenticationError(f"Vertex AI authentication failed: {e}") from e
            elif "not found" in error_str or "does not exist" in error_str:
                raise ModelNotFoundError(f"Model not found: {model}") from e
            elif "quota" in error_str or "rate" in error_str or "429" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {e}") from e
            elif "safety" in error_str or "blocked" in error_str:
                raise ContentFilterError(f"Content blocked by safety filters: {e}") from e
            elif "context" in error_str or "too long" in error_str or "maximum" in error_str:
                raise ContextLengthError(f"Input too long: {e}") from e
            else:
                raise ProviderError(f"Vertex AI error: {e}") from e

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a chat completion response.

        Args:
            messages: List of chat messages
            model: Gemini model ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Tool definitions
            **kwargs: Additional arguments

        Yields:
            Text chunks as they are generated
        """
        self._ensure_client()

        try:
            generative_model = self._get_model(model)

            contents, system_instruction = self._convert_messages_to_vertex_format(
                messages,
                system_instruction=kwargs.get("system_instruction"),
            )

            generation_config: dict[str, Any] = {}
            if temperature is not None:
                generation_config["temperature"] = temperature
            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens

            vertex_tools = self._convert_tools_to_vertex_format(tools)

            if system_instruction:
                generative_model = self._generative_model_class(
                    model_name=model,
                    system_instruction=system_instruction,
                )

            # Stream the response
            loop = asyncio.get_event_loop()

            def _stream():
                return generative_model.generate_content(
                    contents,
                    generation_config=generation_config if generation_config else None,
                    tools=vertex_tools,
                    stream=True,
                )

            response_stream = await loop.run_in_executor(None, _stream)

            # Yield chunks
            for chunk in response_stream:
                if chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, "content") and candidate.content:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    yield part.text

        except Exception as e:
            logger.error("Streaming error", error=str(e), model=model)
            raise ProviderError(f"Streaming error: {e}") from e

    async def validate_key(self, api_key: str | None = None) -> bool:
        """Validate Vertex AI credentials by making a test request.

        For Vertex AI, this validates the service account credentials
        rather than an API key.

        Args:
            api_key: Not used (service account credentials are used instead)

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            self._ensure_client()

            # Make a minimal API call to verify credentials
            model = self._get_model("gemini-1.5-flash-002")

            loop = asyncio.get_event_loop()

            def _test():
                return model.generate_content(
                    "Say 'ok'",
                    generation_config={"max_output_tokens": 5},
                )

            response = await loop.run_in_executor(None, _test)

            return bool(response.candidates)

        except Exception as e:
            logger.warning("Credential validation failed", error=str(e))
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List available Gemini models on Vertex AI.

        Returns:
            List of ModelInfo objects for available models
        """
        self._ensure_client()

        # Vertex AI doesn't have a simple model list API like other providers,
        # so we return a curated list of known models with their capabilities.
        # In production, you might want to use the Model Garden API.

        models = [
            # Gemini 2.5 models
            ModelInfo(
                model_id="gemini-2.5-pro-preview-05-06",
                provider=self.provider_id,
                display_name="Gemini 2.5 Pro Preview",
                input_price_per_1m=1.25,
                output_price_per_1m=5.00,
                context_window=1048576,
                max_output=65536,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=True,
                tier=ModelTier.PREMIUM,
                description="Latest Gemini Pro with enhanced reasoning and 1M context.",
            ),
            ModelInfo(
                model_id="gemini-2.5-flash-preview-05-20",
                provider=self.provider_id,
                display_name="Gemini 2.5 Flash Preview",
                input_price_per_1m=0.075,
                output_price_per_1m=0.30,
                context_window=1048576,
                max_output=65536,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=False,
                tier=ModelTier.VALUE,
                description="Fast and efficient Gemini 2.5 model.",
            ),
            # Gemini 2.0 models
            ModelInfo(
                model_id="gemini-2.0-flash-001",
                provider=self.provider_id,
                display_name="Gemini 2.0 Flash",
                input_price_per_1m=0.10,
                output_price_per_1m=0.40,
                context_window=1048576,
                max_output=8192,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=False,
                tier=ModelTier.VALUE,
                description="Production-ready Gemini 2.0 Flash model.",
            ),
            ModelInfo(
                model_id="gemini-2.0-flash-lite-001",
                provider=self.provider_id,
                display_name="Gemini 2.0 Flash Lite",
                input_price_per_1m=0.075,
                output_price_per_1m=0.30,
                context_window=1048576,
                max_output=8192,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=False,
                tier=ModelTier.FLASH,
                description="Lightweight and cost-effective Gemini 2.0 model.",
            ),
            ModelInfo(
                model_id="gemini-2.0-pro-exp-02-05",
                provider=self.provider_id,
                display_name="Gemini 2.0 Pro (Experimental)",
                input_price_per_1m=0.00,  # Free during preview
                output_price_per_1m=0.00,
                context_window=2097152,
                max_output=8192,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=True,
                tier=ModelTier.PREMIUM,
                description="Experimental Gemini 2.0 Pro with Computer Use.",
            ),
            # Gemini 1.5 models (stable)
            ModelInfo(
                model_id="gemini-1.5-pro-002",
                provider=self.provider_id,
                display_name="Gemini 1.5 Pro",
                input_price_per_1m=1.25,
                output_price_per_1m=5.00,
                context_window=2097152,
                max_output=8192,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=False,
                tier=ModelTier.STANDARD,
                description="Stable Gemini 1.5 Pro with 2M context window.",
            ),
            ModelInfo(
                model_id="gemini-1.5-flash-002",
                provider=self.provider_id,
                display_name="Gemini 1.5 Flash",
                input_price_per_1m=0.075,
                output_price_per_1m=0.30,
                context_window=1048576,
                max_output=8192,
                supports_vision=True,
                supports_tools=True,
                supports_streaming=True,
                supports_computer_use=False,
                tier=ModelTier.VALUE,
                description="Stable, fast Gemini 1.5 model.",
            ),
        ]

        # Cache the models
        self._models_cache = models
        return models

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on the Vertex AI provider.

        Returns:
            Dictionary with health status, project info, and authentication status
        """
        try:
            is_valid = await self.validate_key()
            models = await self.list_models()

            return {
                "provider": self.provider_id,
                "status": "healthy" if is_valid else "unhealthy",
                "authenticated": is_valid,
                "project_id": self.project_id,
                "location": self.location,
                "available_models": len(models),
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "status": "error",
                "authenticated": False,
                "project_id": self.project_id,
                "location": self.location,
                "error": str(e),
            }
