# Multi-Model Orchestration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Together AI the primary inference provider for all non-Computer-Use tasks, upgrade the orchestrator to Claude Opus 4.6, and demote OpenRouter to fallback-only.

**Architecture:** Three providers — Anthropic Direct (orchestrator + Computer Use), Together AI (primary for 15+ models across all tiers), OpenRouter (fallback if Together is down). Together AI uses OpenAI-compatible API format, so the provider is modeled after `DeepSeekProvider`.

**Tech Stack:** Python, httpx, structlog, Together AI OpenAI-compatible API (`https://api.together.xyz/v1`)

---

### Task 1: Create Together AI Provider

**Files:**
- Create: `src/core/providers/together_provider.py`

**Step 1: Write the Together AI provider**

This follows the exact same pattern as `src/core/providers/deepseek_provider.py` — OpenAI-compatible API with `httpx`.

```python
"""Together AI provider implementation.

Together AI hosts 200+ open-source models with an OpenAI-compatible API.
Primary provider for all non-Computer-Use inference in Argus.

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


# Together AI model definitions with pricing (Feb 2026)
TOGETHER_MODELS: list[ModelInfo] = [
    # ── FLASH ($0.02-0.10/1M) ────────────────────────────────────────
    ModelInfo(
        model_id="google/gemma-3n-e4b-it",
        provider="together",
        display_name="Gemma 3n E4B",
        input_price_per_1m=0.02,
        output_price_per_1m=0.04,
        context_window=32768,
        max_output=8192,
        supports_vision=False,
        supports_tools=False,
        tier=ModelTier.FLASH,
        description="Ultra-cheap inference for classification, extraction, JSON parsing",
        aliases=["gemma-3n"],
    ),
    ModelInfo(
        model_id="OpenAI/gpt-oss-20B",
        provider="together",
        display_name="GPT-OSS 20B",
        input_price_per_1m=0.05,
        output_price_per_1m=0.20,
        context_window=128000,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.FLASH,
        description="Cheap model with function calling for simple tool-use tasks",
        aliases=["gpt-oss-20b"],
    ),
    ModelInfo(
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
        provider="together",
        display_name="Mistral Small 3",
        input_price_per_1m=0.10,
        output_price_per_1m=0.30,
        context_window=32768,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.FLASH,
        description="Fast, cheap Mistral model with function calling",
        aliases=["mistral-small-3"],
    ),
    # ── VALUE ($0.15-0.30/1M) ────────────────────────────────────────
    ModelInfo(
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        provider="together",
        display_name="Qwen3-Next 80B",
        input_price_per_1m=0.15,
        output_price_per_1m=1.50,
        context_window=131072,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.VALUE,
        description="MoE 80B (3B active) — fast code analysis and error classification",
        aliases=["qwen3-next-80b"],
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-235B-A22B-fp8-tput",
        provider="together",
        display_name="Qwen3 235B",
        input_price_per_1m=0.20,
        output_price_per_1m=0.60,
        context_window=40960,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.VALUE,
        description="Large MoE for moderate reasoning and general tasks",
        aliases=["qwen3-235b"],
    ),
    ModelInfo(
        model_id="Zai-org/GLM-4.5-Air-FP8",
        provider="together",
        display_name="GLM-4.5 Air",
        input_price_per_1m=0.20,
        output_price_per_1m=1.10,
        context_window=131072,
        max_output=8192,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.VALUE,
        description="Lightweight GLM for general tasks and error classification",
        aliases=["glm-4.5-air"],
    ),
    ModelInfo(
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="together",
        display_name="Llama 4 Maverick",
        input_price_per_1m=0.27,
        output_price_per_1m=0.85,
        context_window=1048576,
        max_output=32768,
        supports_vision=True,
        supports_tools=True,
        tier=ModelTier.VALUE,
        description="128-expert MoE with vision — multilingual image/text understanding",
        aliases=["llama-4-maverick"],
    ),
    ModelInfo(
        model_id="MiniMax/MiniMax-M2.5",
        provider="together",
        display_name="MiniMax M2.5",
        input_price_per_1m=0.30,
        output_price_per_1m=1.20,
        context_window=131072,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.VALUE,
        description="SWE-bench 80.2%, #10 global agentic rank — top code generation",
        aliases=["minimax-m2.5"],
    ),
    # ── STANDARD ($0.45-0.60/1M) ─────────────────────────────────────
    ModelInfo(
        model_id="Zai-org/GLM-4.7",
        provider="together",
        display_name="GLM-4.7",
        input_price_per_1m=0.45,
        output_price_per_1m=2.00,
        context_window=131072,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.STANDARD,
        description="90.6% tool use benchmark — excellent for function calling tasks",
        aliases=["glm-4.7"],
    ),
    ModelInfo(
        model_id="Qwen/Qwen3-Coder-Next",
        provider="together",
        display_name="Qwen3-Coder-Next",
        input_price_per_1m=0.50,
        output_price_per_1m=1.20,
        context_window=262144,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.STANDARD,
        description="SWE-bench 74.2% code agent — 80B params (3B active)",
        aliases=["qwen3-coder-next"],
    ),
    ModelInfo(
        model_id="Moonshotai/Kimi-K2.5",
        provider="together",
        display_name="Kimi K2.5",
        input_price_per_1m=0.50,
        output_price_per_1m=2.80,
        context_window=262144,
        max_output=16384,
        supports_vision=True,
        supports_tools=True,
        tier=ModelTier.STANDARD,
        description="#6 global agentic rank — 1T params, Agent Swarm, 256K context",
        aliases=["kimi-k2.5"],
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-V3.1",
        provider="together",
        display_name="DeepSeek V3.1",
        input_price_per_1m=0.60,
        output_price_per_1m=1.70,
        context_window=131072,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.STANDARD,
        description="671B params (37B active) — hybrid modes, advanced tool calling",
        aliases=["deepseek-v3.1"],
    ),
    # ── REASONING ($0.65-1.20/1M) ────────────────────────────────────
    ModelInfo(
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        provider="together",
        display_name="Qwen3 235B Thinking",
        input_price_per_1m=0.65,
        output_price_per_1m=3.00,
        context_window=40960,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.STANDARD,
        description="Extended reasoning with chain-of-thought for complex analysis",
        aliases=["qwen3-235b-thinking"],
    ),
    ModelInfo(
        model_id="Zai-org/GLM-5",
        provider="together",
        display_name="GLM-5",
        input_price_per_1m=1.00,
        output_price_per_1m=3.20,
        context_window=131072,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.PREMIUM,
        description="#2 global agentic rank (49.64) — complex agent workflows, systems engineering",
        aliases=["glm-5"],
    ),
    ModelInfo(
        model_id="Moonshotai/Kimi-K2-Thinking",
        provider="together",
        display_name="Kimi K2 Thinking",
        input_price_per_1m=1.20,
        output_price_per_1m=4.00,
        context_window=262144,
        max_output=16384,
        supports_vision=True,
        supports_tools=True,
        tier=ModelTier.PREMIUM,
        description="Deep reasoning with vision — extended thinking mode",
        aliases=["kimi-k2-thinking"],
    ),
    # ── EXPERT ($2.00-7.00/1M) ───────────────────────────────────────
    ModelInfo(
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        provider="together",
        display_name="Qwen3-Coder 480B",
        input_price_per_1m=2.00,
        output_price_per_1m=2.00,
        context_window=262144,
        max_output=16384,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.PREMIUM,
        description="480B coding model (35B active) — heavy code generation tasks",
        aliases=["qwen3-coder-480b"],
    ),
    ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        provider="together",
        display_name="DeepSeek R1 0528",
        input_price_per_1m=3.00,
        output_price_per_1m=7.00,
        context_window=65536,
        max_output=65536,
        supports_vision=False,
        supports_tools=True,
        tier=ModelTier.EXPERT,
        description="Expert reasoning with 23K-token thinking — improved function calling",
        aliases=["deepseek-r1-0528"],
    ),
]


class TogetherProvider(BaseProvider):
    """Together AI provider using OpenAI-compatible API.

    Together AI hosts 200+ open-source models including top agentic models:
    - GLM-5 (#2 global agentic rank)
    - Kimi K2.5 (#6 global, Agent Swarm)
    - MiniMax M2.5 (#10 global, SWE-bench 80.2%)

    All 24+ models support function calling via OpenAI-compatible format.

    Example:
        ```python
        provider = TogetherProvider(api_key="tgp_v1_...")

        response = await provider.chat(
            messages=[ChatMessage.user("Analyze this code")],
            model="MiniMax/MiniMax-M2.5",
        )
        ```
    """

    # Provider metadata
    provider_id = "together"
    display_name = "Together AI"
    website = "https://www.together.ai"
    key_url = "https://api.together.ai/settings/api-keys"
    description = "Together AI - Primary inference provider with 200+ open-source models"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True  # Some models (Llama 4, Kimi) support vision
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
        if api_key:
            self.api_key = api_key
        elif config and config.api_key:
            self.api_key = config.api_key
        else:
            self.api_key = os.environ.get("TOGETHER_API_KEY")

        super().__init__(api_key=self.api_key)

        self.base_url = (config.base_url if config else None) or self.BASE_URL
        self.timeout = (config.timeout if config else None) or self.DEFAULT_TIMEOUT
        self.max_retries = (config.max_retries if config else None) or self.MAX_RETRIES
        self.custom_headers = (config.custom_headers if config else None) or {}

        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
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
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AuthenticationError("Together AI API key not configured. Set TOGETHER_API_KEY env var.")
        return {"Authorization": f"Bearer {self.api_key}"}

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
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

    def _parse_tool_calls(self, tool_calls_data: list[dict] | None) -> list[ToolCall] | None:
        if not tool_calls_data:
            return None

        tool_calls = []
        for tc in tool_calls_data:
            import json

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

    def _handle_error_response(self, response: httpx.Response, response_data: dict | None = None) -> None:
        status = response.status_code
        error_msg = "Unknown error"

        if response_data and "error" in response_data:
            error_info = response_data["error"]
            error_msg = error_info.get("message", str(error_info))
            error_code = error_info.get("code", "")

            if "context_length" in error_msg.lower() or error_code == "context_length_exceeded":
                raise ContextLengthError(error_msg)
            if "content_filter" in error_msg.lower():
                raise ContentFilterError(error_msg)
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                raise ModelNotFoundError(error_msg)

        if status == 401:
            raise AuthenticationError(f"Invalid Together AI API key: {error_msg}")
        elif status == 403:
            raise AuthenticationError(f"API key lacks permissions: {error_msg}")
        elif status == 429:
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
        **kwargs,
    ) -> ChatResponse:
        client = await self._get_client()
        start_time = time.time()

        validated_temp = validate_temperature(temperature)

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

        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    "/chat/completions",
                    headers=self._get_auth_headers(),
                    json=payload,
                )

                try:
                    response_data = response.json()
                except Exception:
                    response_data = None

                if response.status_code != 200:
                    self._handle_error_response(response, response_data)

                if not response_data or "choices" not in response_data:
                    raise ProviderError("Invalid response format from Together AI")

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
                    system_fingerprint=response_data.get("system_fingerprint"),
                    latency_ms=latency_ms,
                    raw_response=response_data,
                )

            except (RateLimitError, QuotaExceededError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) * 1.0
                    if isinstance(e, RateLimitError) and e.retry_after:
                        wait_time = e.retry_after
                    logger.warning("Rate limited, retrying", attempt=attempt + 1, wait_time=wait_time)
                    import asyncio
                    await asyncio.sleep(wait_time)
                else:
                    raise

            except httpx.TimeoutException:
                last_error = httpx.TimeoutException("timeout")
                if attempt < self.max_retries - 1:
                    logger.warning("Request timeout, retrying", attempt=attempt + 1)
                else:
                    raise ProviderError(f"Together AI request timed out after {self.max_retries} attempts")

            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning("Request error, retrying", attempt=attempt + 1, error=str(e))
                else:
                    raise ProviderError(f"Together AI request failed: {str(e)}")

        raise ProviderError(f"Together AI request failed after {self.max_retries} attempts: {last_error}")

    async def validate_key(self, api_key: str) -> bool:
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=10.0,
            ) as client:
                response = await client.get(
                    "/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                return response.status_code == 200
        except (httpx.TimeoutException, httpx.RequestError):
            return False

    async def list_models(self) -> list[ModelInfo]:
        return TOGETHER_MODELS.copy()

    def get_capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.CHAT,
            ProviderCapability.STREAMING,
            ProviderCapability.TOOLS,
            ProviderCapability.VISION,
        ]

    def __repr__(self) -> str:
        return f"<TogetherProvider(api_key={mask_api_key(self.api_key)}, base_url='{self.base_url}')>"

    async def __aenter__(self) -> "TogetherProvider":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
```

**Step 2: Commit**

```bash
git add src/core/providers/together_provider.py
git commit -m "feat: add Together AI provider with 17 models across all tiers"
```

---

### Task 2: Export Together Provider from __init__.py

**Files:**
- Modify: `src/core/providers/__init__.py`

**Step 1: Add imports and exports**

Add to imports (after the `from .xai_provider` line):
```python
from .together_provider import TOGETHER_MODELS, TogetherProvider
```

Add to `__all__` list (after `"XAIProvider"`):
```python
    "TogetherProvider",
    "TOGETHER_MODELS",
```

**Step 2: Commit**

```bash
git add src/core/providers/__init__.py
git commit -m "feat: export TogetherProvider from providers package"
```

---

### Task 3: Add Together Models to Model Router MODELS Dict

**Files:**
- Modify: `src/core/model_router.py:186-905` (the `MODELS` dict)

**Step 1: Replace the RECOMMENDED MODELS section header and add Together AI models**

Replace the entire MODELS section header comment (lines 168-184) with:

```python
# =============================================================================
# MODEL REGISTRY - Together AI as Primary Provider (Feb 2026)
# =============================================================================
#
# STRATEGY (Feb 2026):
# - Together AI as PRIMARY for all non-Computer-Use tasks
# - Anthropic Direct for orchestrator (Opus 4.6) + Computer Use (Sonnet)
# - OpenRouter as FALLBACK only (if Together is down)
#
# WHY TOGETHER AI?
# - No markup (direct pricing)
# - Hosts top agentic models: GLM-5 (#2), Kimi K2.5 (#6), MiniMax M2.5 (#10)
# - 24+ models with function calling support
# - OpenAI-compatible API
#
# TOGETHER_API_KEY is the primary key. OPENROUTER_API_KEY for fallback.
# =============================================================================
```

Add the Together AI models BEFORE the existing OpenRouter models. Add these new entries at the top of the MODELS dict, right after the opening `{`:

```python
    # =============================================================================
    # TOGETHER AI MODELS (PRIMARY) - Feb 2026
    # =============================================================================

    # ── FLASH ($0.02-0.10/1M) ────────────────────────────────────────

    "gemma-3n": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="google/gemma-3n-e4b-it",
        input_cost_per_1m=0.02,
        output_cost_per_1m=0.04,
        max_tokens=8192,
        context_window=32768,
        supports_vision=False,
        supports_tools=False,
        supports_json_mode=False,
        latency_ms=50,
    ),

    "gpt-oss-20b": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="OpenAI/gpt-oss-20B",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.20,
        max_tokens=8192,
        context_window=128000,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=100,
    ),

    "mistral-small-3": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.30,
        max_tokens=8192,
        context_window=32768,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=80,
    ),

    # ── VALUE ($0.15-0.30/1M) ────────────────────────────────────────

    "qwen3-next-80b": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
        input_cost_per_1m=0.15,
        output_cost_per_1m=1.50,
        max_tokens=16384,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=150,
    ),

    "qwen3-235b": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Qwen/Qwen3-235B-A22B-fp8-tput",
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.60,
        max_tokens=16384,
        context_window=40960,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=300,
    ),

    "glm-4.5-air": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Zai-org/GLM-4.5-Air-FP8",
        input_cost_per_1m=0.20,
        output_cost_per_1m=1.10,
        max_tokens=8192,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=200,
    ),

    "llama-4-maverick": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        input_cost_per_1m=0.27,
        output_cost_per_1m=0.85,
        max_tokens=32768,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=200,
    ),

    "minimax-m2.5": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="MiniMax/MiniMax-M2.5",
        input_cost_per_1m=0.30,
        output_cost_per_1m=1.20,
        max_tokens=16384,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=250,
    ),

    # ── STANDARD ($0.45-0.60/1M) ─────────────────────────────────────

    "glm-4.7": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Zai-org/GLM-4.7",
        input_cost_per_1m=0.45,
        output_cost_per_1m=2.00,
        max_tokens=16384,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=400,
    ),

    "qwen3-coder-next": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Qwen/Qwen3-Coder-Next",
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.20,
        max_tokens=16384,
        context_window=262144,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=300,
    ),

    "kimi-k2.5": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Moonshotai/Kimi-K2.5",
        input_cost_per_1m=0.50,
        output_cost_per_1m=2.80,
        max_tokens=16384,
        context_window=262144,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=400,
    ),

    "deepseek-v3.1": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="deepseek-ai/DeepSeek-V3.1",
        input_cost_per_1m=0.60,
        output_cost_per_1m=1.70,
        max_tokens=16384,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=300,
    ),

    # ── REASONING ($0.65-1.20/1M) ────────────────────────────────────

    "qwen3-235b-thinking": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        input_cost_per_1m=0.65,
        output_cost_per_1m=3.00,
        max_tokens=16384,
        context_window=40960,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=1500,
    ),

    "glm-5": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Zai-org/GLM-5",
        input_cost_per_1m=1.00,
        output_cost_per_1m=3.20,
        max_tokens=16384,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=800,
    ),

    "kimi-k2-thinking": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Moonshotai/Kimi-K2-Thinking",
        input_cost_per_1m=1.20,
        output_cost_per_1m=4.00,
        max_tokens=16384,
        context_window=262144,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=1500,
    ),

    # ── EXPERT ($2.00-7.00/1M) ───────────────────────────────────────

    "qwen3-coder-480b": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        input_cost_per_1m=2.00,
        output_cost_per_1m=2.00,
        max_tokens=16384,
        context_window=262144,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=600,
    ),

    "deepseek-r1-0528": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="deepseek-ai/DeepSeek-R1-0528",
        input_cost_per_1m=3.00,
        output_cost_per_1m=7.00,
        max_tokens=65536,
        context_window=65536,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=2000,
    ),
```

Keep ALL existing OpenRouter models below as fallbacks (don't delete them).

**Step 2: Commit**

```bash
git add src/core/model_router.py
git commit -m "feat: add 17 Together AI models to MODELS registry"
```

---

### Task 4: Update TASK_MODEL_MAPPING to Use Together Models

**Files:**
- Modify: `src/core/model_router.py:908-969` (the `TASK_MODEL_MAPPING` dict)

**Step 1: Replace the entire TASK_MODEL_MAPPING dict**

Replace lines 908-969 with:

```python
# =============================================================================
# TASK TO MODEL MAPPING - Together AI Primary (Feb 2026)
# =============================================================================
#
# STRATEGY:
# - Together AI models listed FIRST (primary)
# - OpenRouter models as FALLBACK (after Together models)
# - Computer Use stays on Claude/Gemini (Together doesn't host CU models)
#
# Top agentic models on Together:
# - GLM-5: #2 global agentic rank (Quality Index 49.64)
# - Kimi K2.5: #6 global (Agent Swarm, 256K context)
# - MiniMax M2.5: #10 global (SWE-bench 80.2%)
# - GLM-4.7: 90.6% tool use benchmark
# =============================================================================

TASK_MODEL_MAPPING: dict[TaskType, list[str]] = {
    # ─────────────────────────────────────────────────────────────────────────────
    # TRIVIAL TASKS - Together AI cheapest models ($0.02-0.10/1M)
    # ─────────────────────────────────────────────────────────────────────────────
    TaskType.ELEMENT_CLASSIFICATION: ["gemma-3n", "gpt-oss-20b", "mistral-small-3"],
    TaskType.ACTION_EXTRACTION: ["gpt-oss-20b", "gemma-3n", "mistral-small-3"],
    TaskType.SELECTOR_VALIDATION: ["gpt-oss-20b", "mistral-small-3", "gemma-3n"],
    TaskType.TEXT_EXTRACTION: ["gemma-3n", "gpt-oss-20b", "mistral-small-3"],
    TaskType.JSON_PARSING: ["gemma-3n", "gpt-oss-20b", "mistral-small-3"],

    # ─────────────────────────────────────────────────────────────────────────────
    # MODERATE TASKS - MiniMax M2.5 (SWE-bench 80.2%) + Qwen3-Coder ($0.15-0.50/1M)
    # ─────────────────────────────────────────────────────────────────────────────
    TaskType.CODE_ANALYSIS: ["minimax-m2.5", "qwen3-coder-next", "qwen3-235b", "glm-4.5-air"],
    TaskType.TEST_GENERATION: ["minimax-m2.5", "qwen3-coder-next", "llama-4-maverick", "qwen3-235b"],
    TaskType.ASSERTION_GENERATION: ["qwen3-coder-next", "minimax-m2.5", "qwen3-235b"],
    TaskType.ERROR_CLASSIFICATION: ["qwen3-next-80b", "mistral-small-3", "glm-4.5-air"],

    # ─────────────────────────────────────────────────────────────────────────────
    # COMPLEX TASKS - GLM-5 (#2 global), Kimi K2.5 (#6 global) ($0.45-1.20/1M)
    # ─────────────────────────────────────────────────────────────────────────────
    TaskType.VISUAL_COMPARISON: ["llama-4-maverick", "kimi-k2.5", "sonnet"],
    TaskType.SEMANTIC_UNDERSTANDING: ["kimi-k2.5", "glm-5", "qwen3-235b"],
    TaskType.FLOW_DISCOVERY: ["kimi-k2.5", "glm-5", "minimax-m2.5"],
    TaskType.ROOT_CAUSE_ANALYSIS: ["glm-5", "kimi-k2.5", "deepseek-r1-0528"],

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERT TASKS - GLM-5 for agentic, DeepSeek R1 for deep reasoning ($1.00-7.00/1M)
    # ─────────────────────────────────────────────────────────────────────────────
    TaskType.SELF_HEALING: ["glm-5", "kimi-k2.5", "deepseek-r1-0528", "opus"],
    TaskType.FAILURE_PREDICTION: ["glm-5", "qwen3-235b-thinking", "kimi-k2-thinking"],
    TaskType.COGNITIVE_MODELING: ["glm-5", "deepseek-r1-0528", "kimi-k2-thinking"],
    TaskType.COMPLEX_DEBUGGING: ["glm-5", "minimax-m2.5", "deepseek-r1-0528"],

    # ─────────────────────────────────────────────────────────────────────────────
    # COMPUTER USE - Claude/Gemini only (Together doesn't host CU models)
    # ─────────────────────────────────────────────────────────────────────────────
    TaskType.COMPUTER_USE_SIMPLE: ["gemini-computer-use", "claude-computer-use"],
    TaskType.COMPUTER_USE_COMPLEX: ["claude-computer-use", "sonnet", "gemini-computer-use"],
    TaskType.COMPUTER_USE_MOBILE: ["gemini-computer-use", "claude-computer-use"],

    # ─────────────────────────────────────────────────────────────────────────────
    # GENERAL FALLBACK
    # ─────────────────────────────────────────────────────────────────────────────
    TaskType.GENERAL: ["qwen3-235b", "minimax-m2.5", "llama-4-maverick", "glm-4.5-air"],
}
```

**Step 2: Commit**

```bash
git add src/core/model_router.py
git commit -m "feat: update TASK_MODEL_MAPPING to use Together AI models as primary"
```

---

### Task 5: Add Together AI Client to _get_client() Router

**Files:**
- Modify: `src/core/model_router.py:2744-2752` (the `_get_client()` method)

**Step 1: Replace the Together fallback with a real client**

Find the block (around line 2744):
```python
            elif provider in (ModelProvider.CEREBRAS, ModelProvider.DEEPSEEK,
                              ModelProvider.TOGETHER, ModelProvider.VERTEX_AI):
                # These can be accessed via OpenRouter, so fallback if no direct API key
                logger.info(
                    f"Provider {provider.value} requested, using OpenRouter as gateway",
                    provider=provider.value,
                )
                self._clients[provider] = OpenRouterClient()
```

Replace with:
```python
            elif provider == ModelProvider.TOGETHER:
                # Together AI is the primary provider — use direct if key is set
                together_key = os.environ.get("TOGETHER_API_KEY")
                if together_key:
                    logger.info(
                        "Using Together AI direct for primary inference",
                        provider=provider.value,
                    )
                    self._clients[provider] = TogetherClient()
                else:
                    logger.warning(
                        "TOGETHER_API_KEY not set, falling back to OpenRouter",
                        provider=provider.value,
                    )
                    self._clients[provider] = OpenRouterClient()

            elif provider in (ModelProvider.CEREBRAS, ModelProvider.DEEPSEEK,
                              ModelProvider.VERTEX_AI):
                logger.info(
                    f"Provider {provider.value} requested, using OpenRouter as gateway",
                    provider=provider.value,
                )
                self._clients[provider] = OpenRouterClient()
```

**Step 2: Add the TogetherClient class**

Add this class near the other client classes (after `GroqClient` or similar, around line 1100):

```python
class TogetherClient(BaseModelClient):
    """Client for Together AI using OpenAI-compatible API."""

    def __init__(self):
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1"

    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        json_mode: bool = False,
    ) -> dict:
        import httpx

        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_config.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                raise ValueError(f"Together AI error: {error_msg}")

            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return {
                "content": choice["message"].get("content", ""),
                "model": data.get("model", model_config.model_id),
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "tool_calls": choice["message"].get("tool_calls"),
                "finish_reason": choice.get("finish_reason", "stop"),
            }
```

**Step 3: Commit**

```bash
git add src/core/model_router.py
git commit -m "feat: add TogetherClient and route Together models to direct API"
```

---

### Task 6: Add ORCHESTRATOR_MODEL to Model Registry

**Files:**
- Modify: `src/core/model_registry.py:354-378` and add convenience function at ~line 540

**Step 1: Add ORCHESTRATOR_MODEL constant**

After `COMPUTER_USE_MODEL = "claude-sonnet-4-5"` (line 359), add:
```python
    ORCHESTRATOR_MODEL = "claude-opus-4-6"
```

**Step 2: Add env var override**

In `_load_overrides()` (after line 378), add:
```python
        if os.getenv("ARGUS_ORCHESTRATOR_MODEL"):
            self.ORCHESTRATOR_MODEL = os.getenv("ARGUS_ORCHESTRATOR_MODEL")
```

**Step 3: Add getter method**

After `get_powerful_model()` (after line 433), add:
```python
    def get_orchestrator_model(self) -> ModelConfig:
        """Get the orchestrator model for supervisor routing decisions."""
        model = self._models.get(self.ORCHESTRATOR_MODEL)
        if model:
            return model
        # Fall back to powerful model if orchestrator model not in registry
        return self._models[self.POWERFUL_MODEL]
```

**Step 4: Add convenience function**

After `get_fast_api_model_id()` (around line 549), add:
```python
def get_orchestrator_api_model_id() -> str:
    """Get the orchestrator model's full API ID for supervisor routing."""
    registry = get_model_registry()
    return registry.get_api_model_id(registry.ORCHESTRATOR_MODEL)
```

**Step 5: Commit**

```bash
git add src/core/model_registry.py
git commit -m "feat: add ORCHESTRATOR_MODEL to model registry with env var override"
```

---

### Task 7: Upgrade Supervisor to Use Opus 4.6

**Files:**
- Modify: `src/orchestrator/supervisor.py:51,310-313`

**Step 1: Update import**

Change line 51 from:
```python
from src.core.model_registry import get_default_api_model_id
```
to:
```python
from src.core.model_registry import get_orchestrator_api_model_id
```

**Step 2: Update the LLM instantiation**

Change lines 310-313 from:
```python
        llm = ChatAnthropic(
            model=get_default_api_model_id(),
            api_key=api_key,
            max_tokens=1024,
        )
```
to:
```python
        llm = ChatAnthropic(
            model=get_orchestrator_api_model_id(),
            api_key=api_key,
            max_tokens=4096,
        )
```

**Step 3: Commit**

```bash
git add src/orchestrator/supervisor.py
git commit -m "feat: upgrade supervisor orchestrator to Claude Opus 4.6 with 4096 max_tokens"
```

---

### Task 8: Add TOGETHER_API_KEY to .env.example

**Files:**
- Modify: `.env.example`

**Step 1: Add Together AI key**

After the `OPENROUTER_API_KEY` line (line 17), add:
```bash
TOGETHER_API_KEY=tgp_v1_...                    # Together AI (PRIMARY inference provider)
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "feat: add TOGETHER_API_KEY to .env.example"
```

---

### Task 9: Set TOGETHER_API_KEY in Local Environment

**Files:**
- Modify: `.env` (local only, not committed)

**Step 1: Add the key**

```bash
echo 'TOGETHER_API_KEY=tgp_v1_UdXTfyg8TJgf-SfQu-59JwLLztbKu57FP1MrBIqlbE8' >> .env
```

This is the actual API key provided by the user. It goes ONLY in `.env` (gitignored), never in source code.

---

### Task 10: Verify — Run Existing Tests

**Step 1: Run any model_router or provider tests**

```bash
cd /Users/bvk/Downloads/e2e-testing-agent
python -m pytest tests/ -k "model_router or provider or together" -v --no-header 2>&1 | head -50
```

If no tests exist yet, verify imports work:

```bash
python -c "from src.core.providers.together_provider import TogetherProvider, TOGETHER_MODELS; print(f'TogetherProvider loaded with {len(TOGETHER_MODELS)} models')"
```

And verify model registry:
```bash
python -c "from src.core.model_registry import get_orchestrator_api_model_id; print(f'Orchestrator model: {get_orchestrator_api_model_id()}')"
```

**Step 2: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: resolve any import issues from Together AI integration"
```
