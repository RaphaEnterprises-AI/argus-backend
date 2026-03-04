"""Together AI provider implementation.

This module implements the Together AI provider using their OpenAI-compatible API.
Together AI hosts a wide catalog of open-weight models with competitive pricing,
making it an excellent primary inference provider for diverse workloads.

API Documentation: https://docs.together.ai/reference
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


# Together AI model definitions with pricing
# Organized by tier: FLASH → VALUE → STANDARD → PREMIUM → EXPERT
TOGETHER_MODELS: list[ModelInfo] = [
    # ── FLASH tier ────────────────────────────────────────────────────
    ModelInfo(
        model_id="google/gemma-3n-E4B-it",
        provider="together",
        display_name="Gemma 3n E4B Instruct",
        input_price_per_1m=0.02,
        output_price_per_1m=0.04,
        context_window=32000,
        max_output=8192,
        supports_vision=False,
        supports_tools=False,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.FLASH,
        description="Google Gemma 3n E4B - Ultra-cheap edge model for trivial classification and extraction tasks",
        aliases=["gemma-3n", "gemma-3n-e4b"],
    ),
    ModelInfo(
        model_id="openai/gpt-oss-20b",
        provider="together",
        display_name="GPT-OSS 20B",
        input_price_per_1m=0.05,
        output_price_per_1m=0.20,
        context_window=128000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.FLASH,
        description="OpenAI GPT-OSS 20B - Open-source GPT variant with 128K context and tool calling",
        aliases=["gpt-oss", "gpt-oss-20b"],
    ),
    ModelInfo(
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
        provider="together",
        display_name="Mistral Small 24B",
        input_price_per_1m=0.10,
        output_price_per_1m=0.30,
        context_window=32000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.FLASH,
        description="Mistral Small 24B - Efficient instruction-tuned model with strong tool use capabilities",
        aliases=["mistral-small-24b", "mistral-small"],
    ),
    # ── VALUE tier ────────────────────────────────────────────────────
    ModelInfo(
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        provider="together",
        display_name="Qwen3 Next 80B-A3B",
        input_price_per_1m=0.15,
        output_price_per_1m=1.50,
        context_window=131000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="Qwen3 Next 80B MoE (3B active) - Excellent cost-to-quality ratio for code analysis and generation",
        aliases=["qwen3-next-80b", "qwen3-next"],
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        provider="together",
        display_name="Qwen3 235B-A22B (Throughput)",
        input_price_per_1m=0.20,
        output_price_per_1m=0.60,
        context_window=40000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="Qwen3 235B MoE (22B active) FP8 throughput variant - Large-scale MoE at value pricing",
        aliases=["qwen3-235b-tput", "qwen3-235b"],
    ),
    ModelInfo(
        model_id="zai-org/GLM-4.5-Air-FP8",
        provider="together",
        display_name="GLM 4.5 Air FP8",
        input_price_per_1m=0.20,
        output_price_per_1m=1.10,
        context_window=131000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="Zhipu GLM 4.5 Air FP8 - Strong multilingual model with 131K context at budget pricing",
        aliases=["glm-4.5-air", "glm-4.5"],
    ),
    ModelInfo(
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="together",
        display_name="Llama 4 Maverick 17B-128E",
        input_price_per_1m=0.27,
        output_price_per_1m=0.85,
        context_window=1000000,
        max_output=8192,
        supports_vision=True,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="Meta Llama 4 Maverick 17B MoE (128 experts) - 1M context, vision-capable, excellent for multimodal tasks",
        aliases=["llama-4-maverick", "maverick"],
    ),
    ModelInfo(
        model_id="MiniMaxAI/MiniMax-M2.5",
        provider="together",
        display_name="MiniMax M2.5",
        input_price_per_1m=0.30,
        output_price_per_1m=1.20,
        context_window=131000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.VALUE,
        description="MiniMax M2.5 - Strong general-purpose model with 131K context and reliable tool calling",
        aliases=["minimax-m2.5", "minimax"],
    ),
    # ── STANDARD tier ─────────────────────────────────────────────────
    ModelInfo(
        model_id="zai-org/GLM-4.7",
        provider="together",
        display_name="GLM 4.7",
        input_price_per_1m=0.45,
        output_price_per_1m=2.00,
        context_window=131000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="Zhipu GLM 4.7 - Upgraded GLM with strong reasoning and 131K context",
        aliases=["glm-4.7"],
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-Coder-Next-FP8",
        provider="together",
        display_name="Qwen3 Coder Next",
        input_price_per_1m=0.50,
        output_price_per_1m=1.20,
        context_window=262000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="Qwen3 Coder Next - Specialized coding model with 262K context, strong at code generation and analysis",
        aliases=["qwen3-coder-next", "qwen3-coder"],
    ),
    ModelInfo(
        model_id="moonshotai/Kimi-K2.5",
        provider="together",
        display_name="Kimi K2.5",
        input_price_per_1m=0.50,
        output_price_per_1m=2.80,
        context_window=262000,
        max_output=8192,
        supports_vision=True,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="Moonshot Kimi K2.5 - Vision-capable model with 262K context and strong agentic performance",
        aliases=["kimi-k2.5", "kimi"],
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-V3.1",
        provider="together",
        display_name="DeepSeek V3.1",
        input_price_per_1m=0.60,
        output_price_per_1m=1.70,
        context_window=131000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.STANDARD,
        description="DeepSeek V3.1 - Latest DeepSeek chat model, excellent at code and reasoning tasks",
        aliases=["deepseek-v3.1", "deepseek-v3-together"],
    ),
    # ── PREMIUM tier (reasoning models) ───────────────────────────────
    ModelInfo(
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        provider="together",
        display_name="Qwen3 235B Thinking",
        input_price_per_1m=0.65,
        output_price_per_1m=3.00,
        context_window=40000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.PREMIUM,
        description="Qwen3 235B MoE Thinking - Extended reasoning variant for complex multi-step problems",
        aliases=["qwen3-235b-thinking", "qwen3-thinking"],
    ),
    ModelInfo(
        model_id="zai-org/GLM-5",
        provider="together",
        display_name="GLM 5",
        input_price_per_1m=1.00,
        output_price_per_1m=3.20,
        context_window=131000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.PREMIUM,
        description="Zhipu GLM 5 - #2 global agentic ranking, top-tier reasoning and tool use at competitive pricing",
        aliases=["glm-5"],
    ),
    ModelInfo(
        model_id="moonshotai/Kimi-K2-Thinking",
        provider="together",
        display_name="Kimi K2 Thinking",
        input_price_per_1m=1.20,
        output_price_per_1m=4.00,
        context_window=262000,
        max_output=8192,
        supports_vision=True,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.PREMIUM,
        description="Moonshot Kimi K2 Thinking - Vision-capable reasoning model with 262K context and chain-of-thought",
        aliases=["kimi-k2-thinking", "kimi-thinking"],
    ),
    # ── EXPERT tier ───────────────────────────────────────────────────
    ModelInfo(
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        provider="together",
        display_name="Qwen3 Coder 480B",
        input_price_per_1m=2.00,
        output_price_per_1m=2.00,
        context_window=262000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.EXPERT,
        description="Qwen3 Coder 480B MoE (35B active) - Largest open-source coding model, 262K context, state-of-the-art code generation",
        aliases=["qwen3-coder-480b", "qwen3-coder-expert"],
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        provider="together",
        display_name="DeepSeek R1 0528",
        input_price_per_1m=3.00,
        output_price_per_1m=7.00,
        context_window=65000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        supports_streaming=True,
        supports_computer_use=False,
        tier=ModelTier.EXPERT,
        description="DeepSeek R1 (May 2528 release) - Top-tier open reasoning model with chain-of-thought for expert-level analysis",
        aliases=["deepseek-r1-0528", "deepseek-r1-together"],
    ),
]


class TogetherProvider(BaseProvider):
    """Together AI provider using OpenAI-compatible API.

    Together AI hosts a wide catalog of open-weight models across all tiers,
    from ultra-cheap flash models to expert-level reasoning models. Their
    OpenAI-compatible API makes integration straightforward.

    Model tiers and representative pricing:
    - FLASH: Gemma 3n ($0.02/$0.04), GPT-OSS 20B ($0.05/$0.20), Mistral Small ($0.10/$0.30)
    - VALUE: Qwen3 MoE ($0.15-$0.30), Llama 4 Maverick ($0.27/$0.85), MiniMax ($0.30/$1.20)
    - STANDARD: GLM 4.7 ($0.45/$2.00), Qwen3 Coder ($0.50/$1.20), DeepSeek V3.1 ($0.60/$1.70)
    - PREMIUM: Qwen3 Thinking ($0.65/$3.00), GLM 5 ($1.00/$3.20), Kimi K2 Thinking ($1.20/$4.00)
    - EXPERT: Qwen3 Coder 480B ($2.00/$2.00), DeepSeek R1 ($3.00/$7.00)

    Example:
        ```python
        provider = TogetherProvider(api_key="your-api-key")

        # Standard chat with a value-tier model
        response = await provider.chat(
            messages=[ChatMessage.user("Analyze this code for bugs")],
            model="deepseek-ai/DeepSeek-V3.1",
        )

        # Reasoning mode with a thinking model
        response = await provider.chat(
            messages=[ChatMessage.user("Solve this complex problem...")],
            model="Qwen/Qwen3-235B-A22B-Thinking-2507",
            reasoning=True,
        )

        # Ultra-cheap classification with flash tier
        response = await provider.chat(
            messages=[ChatMessage.user("Classify: bug or feature request?")],
            model="google/gemma-3n-E4B-it",
        )
        ```
    """

    # Provider metadata
    provider_id = "together"
    display_name = "Together AI"
    website = "https://together.ai"
    key_url = "https://api.together.xyz/settings/api-keys"
    description = "Together AI - Primary inference provider with 17 curated open-weight models across all tiers"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.together.xyz/v1"
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str | None = None,
        config: ProviderConfig | None = None,
    ):
        """Initialize the Together AI provider.

        Args:
            api_key: Together AI API key. If None, reads from TOGETHER_API_KEY env var.
            config: Optional provider configuration for advanced settings.
        """
        # Get API key from argument, config, or environment
        if api_key:
            self.api_key = api_key
        elif config and config.api_key:
            self.api_key = config.api_key
        else:
            self.api_key = os.environ.get("TOGETHER_API_KEY")

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
            raise AuthenticationError("Together AI API key not configured")
        return {"Authorization": f"Bearer {self.api_key}"}

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        """Convert ChatMessage objects to Together AI API format.

        Together AI uses OpenAI-compatible message format.
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
            if isinstance(error_info, dict):
                error_msg = error_info.get("message", str(error_info))
                error_type = error_info.get("type", "")
                error_code = error_info.get("code", "")
            else:
                error_msg = str(error_info)
                error_type = ""
                error_code = ""

            # Check for specific error types
            if "context_length" in error_msg.lower() or error_code == "context_length_exceeded":
                raise ContextLengthError(error_msg)
            if "content_filter" in error_msg.lower() or error_type == "content_filter":
                raise ContentFilterError(error_msg)
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ModelNotFoundError(error_msg)

        if status == 401:
            raise AuthenticationError(f"Invalid Together AI API key: {error_msg}")
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
            raise ProviderError(f"Together AI server error ({status}): {error_msg}")
        else:
            raise ProviderError(f"Together AI API error ({status}): {error_msg}")

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
        """Send a chat completion request to Together AI.

        Args:
            messages: List of chat messages forming the conversation
            model: Model ID (e.g., "deepseek-ai/DeepSeek-V3.1", "zai-org/GLM-5")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            reasoning: Enable reasoning mode for thinking models (extended thinking)
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

        # Enable reasoning mode for thinking models
        if reasoning and self.is_reasoning_model(model):
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
                    raise ProviderError("Invalid response format from Together AI")

                choice = response_data["choices"][0]
                message = choice.get("message", {})
                usage = response_data.get("usage", {})

                latency_ms = (time.time() - start_time) * 1000

                # Extract reasoning content if present (thinking models)
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
                        provider="together",
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
                        provider="together",
                        attempt=attempt + 1,
                    )
                else:
                    raise ProviderError(f"Together AI request timed out after {self.max_retries} attempts")

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Request error, retrying",
                        provider="together",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                else:
                    raise ProviderError(f"Together AI request failed: {str(e)}")

        # Should not reach here, but just in case
        raise ProviderError(f"Together AI request failed after {self.max_retries} attempts: {last_error}")

    async def validate_key(self, api_key: str) -> bool:
        """Validate a Together AI API key.

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
                        provider="together",
                        status=response.status_code,
                    )
                    return False

        except httpx.TimeoutException:
            logger.warning("Timeout during Together AI key validation")
            return False
        except httpx.RequestError as e:
            logger.warning("Request error during Together AI key validation", error=str(e))
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List available Together AI models.

        Returns the statically defined curated models with current pricing.
        Together AI hosts many models, but we curate a selection across tiers.

        Returns:
            List of ModelInfo for available models
        """
        # Return cached static model definitions
        # These are curated and don't change frequently
        return TOGETHER_MODELS.copy()

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
            ProviderCapability.VISION,
        ]

        # Thinking models support reasoning
        capabilities.append(ProviderCapability.REASONING)

        return capabilities

    def is_reasoning_model(self, model: str) -> bool:
        """Check if a model supports reasoning mode.

        Args:
            model: Model ID to check

        Returns:
            True if model supports extended reasoning
        """
        model_lower = model.lower()
        return (
            "thinking" in model_lower
            or "r1" in model_lower
        )

    def __repr__(self) -> str:
        """String representation with masked API key for security."""
        return (
            f"<TogetherProvider("
            f"api_key={mask_api_key(self.api_key)}, "
            f"base_url='{self.base_url}')>"
        )

    async def __aenter__(self) -> "TogetherProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
