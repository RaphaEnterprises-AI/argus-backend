"""
Multi-Model Router - Intelligent Model Selection for Cost Optimization

PROBLEM:
- Claude Sonnet for everything = expensive at scale
- 10,000 tests/day × $0.018/test = $5,400/month just for Claude
- Not all tasks need Claude's reasoning capability

SOLUTION:
- Route tasks to appropriate models based on complexity
- Use cheaper models for simple tasks
- Reserve premium models for complex reasoning
- Estimated 60-80% cost reduction

MODEL TIERS:
- Tier 1 (Flash): Simple classification, extraction → $0.0001/call
- Tier 2 (Standard): Moderate reasoning → $0.001/call
- Tier 3 (Premium): Complex analysis → $0.01/call
- Tier 4 (Expert): Debugging, novel problems → $0.05/call
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal
import os
import time
import httpx
from abc import ABC, abstractmethod
import structlog

from src.services.ai_cost_tracker import (
    get_cost_tracker,
    TaskType as CostTaskType,
    BudgetStatus,
)

logger = structlog.get_logger()


class BudgetExceededError(Exception):
    """Raised when an organization exceeds their AI budget."""

    pass


class ModelProvider(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex_ai"  # Claude via Google Cloud Vertex AI
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"  # Fast inference for open models (100ms latency!)
    TOGETHER = "together"  # Open model hosting
    LOCAL = "local"  # Ollama, vLLM, etc.


class InferenceGateway(str, Enum):
    """Inference gateway/platform options."""
    DIRECT = "direct"  # Direct API calls to providers
    CLOUDFLARE = "cloudflare"  # Cloudflare AI Gateway (recommended)
    AWS_BEDROCK = "aws_bedrock"  # AWS Bedrock
    AZURE = "azure"  # Azure OpenAI


class TaskComplexity(str, Enum):
    """Task complexity levels for routing."""
    TRIVIAL = "trivial"      # Simple extraction, classification
    SIMPLE = "simple"        # Single-step reasoning
    MODERATE = "moderate"    # Multi-step reasoning
    COMPLEX = "complex"      # Deep analysis, novel problems
    EXPERT = "expert"        # Debugging, self-healing


class TaskType(str, Enum):
    """Types of tasks we perform."""
    # Simple tasks - can use cheapest models
    ELEMENT_CLASSIFICATION = "element_classification"
    ACTION_EXTRACTION = "action_extraction"
    SELECTOR_VALIDATION = "selector_validation"
    TEXT_EXTRACTION = "text_extraction"
    JSON_PARSING = "json_parsing"

    # Moderate tasks - need decent reasoning
    CODE_ANALYSIS = "code_analysis"
    TEST_GENERATION = "test_generation"
    ASSERTION_GENERATION = "assertion_generation"
    ERROR_CLASSIFICATION = "error_classification"

    # Complex tasks - need strong reasoning
    VISUAL_COMPARISON = "visual_comparison"
    SEMANTIC_UNDERSTANDING = "semantic_understanding"
    FLOW_DISCOVERY = "flow_discovery"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"

    # Expert tasks - need best models
    SELF_HEALING = "self_healing"
    FAILURE_PREDICTION = "failure_prediction"
    COGNITIVE_MODELING = "cognitive_modeling"
    COMPLEX_DEBUGGING = "complex_debugging"

    # Computer Use tasks - specialized models
    COMPUTER_USE_SIMPLE = "computer_use_simple"  # Forms, navigation
    COMPUTER_USE_COMPLEX = "computer_use_complex"  # Multi-step flows
    COMPUTER_USE_MOBILE = "computer_use_mobile"  # Mobile UI automation

    # General purpose
    GENERAL = "general"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: ModelProvider
    model_id: str
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    max_tokens: int
    supports_vision: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_computer_use: bool = False  # Browser/desktop automation
    supports_thinking: bool = False  # Extended thinking/reasoning
    context_window: int = 128000  # Input token limit
    latency_ms: int = 1000  # Typical latency

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens (assuming 60/40 input/output split)."""
        return (self.input_cost_per_1m * 0.6 + self.output_cost_per_1m * 0.4) / 1000


# Model Registry - All available models with pricing (as of Jan 2026)
MODELS = {
    # ===========================================
    # GEMINI 3.0 SERIES (Preview - Latest)
    # Most advanced Gemini models
    # ===========================================

    "gemini-3-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-3-pro-preview",
        input_cost_per_1m=2.00,  # $4.00 for >200k context
        output_cost_per_1m=12.00,  # $18.00 for >200k context
        max_tokens=65536,
        context_window=1048576,  # 1M tokens
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=1500,
    ),

    "gemini-3-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-3-flash-preview",
        input_cost_per_1m=0.50,  # Free tier available
        output_cost_per_1m=3.00,
        max_tokens=65536,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=400,
    ),

    # ===========================================
    # GEMINI 2.5 SERIES (Stable)
    # Production-ready with thinking support
    # ===========================================

    "gemini-2.5-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-pro",
        input_cost_per_1m=1.25,  # $2.50 for >200k context
        output_cost_per_1m=10.00,  # $15.00 for >200k context
        max_tokens=65536,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=800,
    ),

    "gemini-2.5-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-flash",
        input_cost_per_1m=0.30,  # Free tier available
        output_cost_per_1m=2.50,
        max_tokens=65536,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=300,
    ),

    "gemini-2.5-flash-lite": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-flash-lite",
        input_cost_per_1m=0.10,  # Cheapest Gemini - Free tier available
        output_cost_per_1m=0.40,
        max_tokens=65536,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=200,
    ),

    # ===========================================
    # GEMINI COMPUTER USE (Browser Automation)
    # Specialized for UI automation tasks
    # ===========================================

    "gemini-computer-use": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-computer-use-preview-10-2025",
        input_cost_per_1m=1.25,  # Based on 2.5 Pro pricing
        output_cost_per_1m=5.00,
        max_tokens=64000,
        context_window=128000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_computer_use=True,
        latency_ms=500,
    ),

    # ===========================================
    # GEMINI 2.0 SERIES (Legacy but stable)
    # ===========================================

    "gemini-2.0-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.0-flash",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        max_tokens=8192,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=250,
    ),

    "gemini-2.0-flash-lite": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.0-flash-lite",
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        max_tokens=8192,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=150,
    ),

    # Legacy alias for backward compatibility
    "gemini-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-flash",  # Updated to 2.5
        input_cost_per_1m=0.30,
        output_cost_per_1m=2.50,
        max_tokens=65536,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=300,
    ),

    "gemini-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-pro",  # Updated to 2.5
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.00,
        max_tokens=65536,
        context_window=1048576,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        supports_thinking=True,
        latency_ms=800,
    ),

    # ===========================================
    # TIER 1: FLASH (< $0.50/1M tokens)
    # For: Classification, extraction, simple parsing
    # ===========================================

    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        max_tokens=16384,
        context_window=128000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=400,
    ),

    "haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20250514",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=500,
    ),

    "llama-3.1-8b": ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.1-8b-instant",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.08,
        max_tokens=8192,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=100,  # Groq is very fast
    ),

    # ===========================================
    # TIER 2: STANDARD ($0.50 - $5/1M tokens)
    # For: Code analysis, test generation
    # ===========================================

    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        max_tokens=16384,
        context_window=128000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=800,
    ),

    # ===========================================
    # CLAUDE MODELS (Anthropic)
    # ===========================================

    "claude-computer-use": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250514",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        supports_computer_use=True,
        latency_ms=1000,
    ),

    "sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250514",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        supports_computer_use=True,
        latency_ms=1000,
    ),

    "llama-3.1-70b": ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.1-70b-versatile",
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
        max_tokens=8192,
        context_window=131072,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=200,
    ),

    "deepseek-v3": ModelConfig(
        provider=ModelProvider.TOGETHER,
        model_id="deepseek-ai/DeepSeek-V3",
        input_cost_per_1m=0.27,
        output_cost_per_1m=1.10,
        max_tokens=8192,
        context_window=128000,
        supports_vision=False,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=500,
    ),

    # ===========================================
    # TIER 3: PREMIUM ($5 - $20/1M tokens)
    # For: Complex reasoning, visual analysis
    # ===========================================

    "opus": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-5-20250514",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        supports_computer_use=True,
        supports_thinking=True,
        latency_ms=2000,
    ),

    "o1": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="o1",
        input_cost_per_1m=15.00,
        output_cost_per_1m=60.00,
        max_tokens=100000,
        context_window=200000,
        supports_vision=True,
        supports_tools=False,  # o1 doesn't support tools yet
        supports_json_mode=False,
        supports_thinking=True,
        latency_ms=5000,  # Reasoning takes time
    ),

    # ===========================================
    # VERTEX AI - Claude via Google Cloud
    # Same pricing, unified GCP billing, Computer Use supported
    # ===========================================

    "vertex-sonnet": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-sonnet-4-5-20250514",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        supports_computer_use=True,
        latency_ms=1000,
    ),

    "vertex-opus": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-opus-4-5-20250514",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        supports_computer_use=True,
        latency_ms=2000,
    ),

    "vertex-haiku": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-haiku-4-5-20250514",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=500,
    ),

    "vertex-computer-use": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-sonnet-4-5-20250514",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        context_window=200000,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        supports_computer_use=True,
        latency_ms=1000,
    ),
}


# Task to Model Mapping - Optimized for Cost and Quality (Jan 2026)
TASK_MODEL_MAPPING: dict[TaskType, list[str]] = {
    # ===========================================
    # TRIVIAL TASKS - Use cheapest models
    # Gemini 2.5 Flash-Lite is 75% cheaper than GPT-4o-mini
    # ===========================================
    TaskType.ELEMENT_CLASSIFICATION: ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "llama-3.1-8b", "gpt-4o-mini"],
    TaskType.ACTION_EXTRACTION: ["gemini-2.5-flash-lite", "llama-3.1-8b", "gemini-2.0-flash", "gpt-4o-mini"],
    TaskType.SELECTOR_VALIDATION: ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gpt-4o-mini", "haiku"],
    TaskType.TEXT_EXTRACTION: ["gemini-2.5-flash-lite", "llama-3.1-8b", "gemini-2.0-flash-lite", "gpt-4o-mini"],
    TaskType.JSON_PARSING: ["gemini-2.5-flash-lite", "llama-3.1-8b", "gemini-2.0-flash", "gpt-4o-mini"],

    # ===========================================
    # MODERATE TASKS - Balanced cost/quality
    # Gemini 2.5 Flash has thinking capability at low cost
    # ===========================================
    TaskType.CODE_ANALYSIS: ["gemini-2.5-flash", "deepseek-v3", "gemini-2.5-pro", "gpt-4o", "sonnet"],
    TaskType.TEST_GENERATION: ["gemini-2.5-flash", "deepseek-v3", "gemini-2.5-pro", "sonnet"],
    TaskType.ASSERTION_GENERATION: ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o", "sonnet"],
    TaskType.ERROR_CLASSIFICATION: ["gemini-2.5-flash", "llama-3.1-70b", "gpt-4o", "haiku"],

    # ===========================================
    # COMPLEX TASKS - Need vision or strong reasoning
    # Gemini 2.5 Pro has 1M context and thinking
    # ===========================================
    TaskType.VISUAL_COMPARISON: ["gemini-2.5-pro", "gemini-3-flash", "gpt-4o", "sonnet"],
    TaskType.SEMANTIC_UNDERSTANDING: ["gemini-2.5-pro", "gpt-4o", "sonnet", "gemini-3-pro"],
    TaskType.FLOW_DISCOVERY: ["gemini-2.5-pro", "gpt-4o", "sonnet"],
    TaskType.ROOT_CAUSE_ANALYSIS: ["gemini-3-pro", "sonnet", "gemini-2.5-pro", "opus"],

    # ===========================================
    # EXPERT TASKS - Use best available
    # Gemini 3 Pro or Claude Opus for complex reasoning
    # ===========================================
    TaskType.SELF_HEALING: ["gemini-3-pro", "sonnet", "opus", "o1"],
    TaskType.FAILURE_PREDICTION: ["gemini-2.5-pro", "sonnet", "opus"],
    TaskType.COGNITIVE_MODELING: ["gemini-3-pro", "opus", "o1", "sonnet"],
    TaskType.COMPLEX_DEBUGGING: ["gemini-3-pro", "opus", "o1"],

    # ===========================================
    # COMPUTER USE TASKS - Specialized models
    # Gemini Computer Use is ~60% cheaper than Claude
    # Claude has more mature browser automation
    # ===========================================
    TaskType.COMPUTER_USE_SIMPLE: [
        "gemini-computer-use",  # Cheapest, good for simple forms
        "vertex-computer-use",  # Claude via GCP
        "claude-computer-use",  # Direct Claude
    ],
    TaskType.COMPUTER_USE_COMPLEX: [
        "claude-computer-use",  # More mature for complex flows
        "vertex-computer-use",
        "gemini-computer-use",
    ],
    TaskType.COMPUTER_USE_MOBILE: [
        "gemini-computer-use",  # Gemini excels at mobile UI
    ],

    # ===========================================
    # GENERAL FALLBACK
    # ===========================================
    TaskType.GENERAL: ["gemini-2.5-flash", "sonnet", "gpt-4o", "gemini-2.5-pro"],
}


class BaseModelClient(ABC):
    """Abstract base class for model clients."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        """Generate a completion."""
        pass

    @abstractmethod
    async def complete_with_vision(
        self,
        messages: list[dict],
        images: list[bytes],
        model_config: ModelConfig,
        max_tokens: int = 4096,
    ) -> dict:
        """Generate a completion with images."""
        pass


class AnthropicClient(BaseModelClient):
    """Client for Anthropic models."""

    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic()

    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)

        return {
            "content": response.content[0].text if response.content else "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model_config.model_id,
        }

    async def complete_with_vision(
        self,
        messages: list[dict],
        images: list[bytes],
        model_config: ModelConfig,
        max_tokens: int = 4096,
    ) -> dict:
        import base64

        # Add images to the last user message
        image_content = []
        for img in images:
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode(),
                }
            })

        # Modify messages to include images
        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            if isinstance(enhanced_messages[-1]["content"], str):
                enhanced_messages[-1]["content"] = [
                    {"type": "text", "text": enhanced_messages[-1]["content"]},
                    *image_content
                ]
            else:
                enhanced_messages[-1]["content"].extend(image_content)

        return await self.complete(enhanced_messages, model_config, max_tokens)


class OpenAIClient(BaseModelClient):
    """Client for OpenAI models."""

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode and model_config.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            # Convert Anthropic tool format to OpenAI format
            kwargs["tools"] = self._convert_tools(tools)

        response = await self.client.chat.completions.create(**kwargs)

        return {
            "content": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model_config.model_id,
        }

    def _convert_tools(self, tools: list) -> list:
        """Convert Anthropic tool format to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                }
            }
            for t in tools
        ]

    async def complete_with_vision(
        self,
        messages: list[dict],
        images: list[bytes],
        model_config: ModelConfig,
        max_tokens: int = 4096,
    ) -> dict:
        import base64

        # Add images to messages
        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(img).decode()}"
                }
            }
            for img in images
        ]

        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            content = enhanced_messages[-1]["content"]
            if isinstance(content, str):
                enhanced_messages[-1]["content"] = [
                    {"type": "text", "text": content},
                    *image_content
                ]

        return await self.complete(enhanced_messages, model_config, max_tokens)


class GoogleClient(BaseModelClient):
    """
    Client for Google Gemini models including Computer Use.

    Supports:
    - Gemini 3.0 Pro/Flash (latest)
    - Gemini 2.5 Pro/Flash/Flash-Lite
    - Gemini 2.5 Computer Use (browser automation)
    - Gemini 2.0 Flash/Flash-Lite

    For Computer Use, uses google-genai library with ComputerUse tool.
    """

    # Gemini Computer Use supported actions
    COMPUTER_USE_ACTIONS = [
        "open_web_browser",
        "navigate",
        "go_back",
        "go_forward",
        "search",
        "click_at",
        "hover_at",
        "type_text_at",
        "key_combination",
        "scroll_document",
        "scroll_at",
        "drag_and_drop",
        "wait_5_seconds",
    ]

    def __init__(self, use_new_sdk: bool = True):
        """
        Initialize Google client.

        Args:
            use_new_sdk: Use google-genai (new) vs google.generativeai (legacy)
        """
        self.use_new_sdk = use_new_sdk
        self._genai = None
        self._genai_client = None

    @property
    def genai(self):
        """Lazy load legacy SDK."""
        if self._genai is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self._genai = genai
        return self._genai

    @property
    def genai_client(self):
        """Lazy load new google-genai SDK for Computer Use."""
        if self._genai_client is None:
            try:
                from google import genai
                self._genai_client = genai.Client(
                    api_key=os.environ.get("GOOGLE_API_KEY")
                )
            except ImportError:
                logger.warning("google-genai SDK not installed, Computer Use unavailable")
                self._genai_client = None
        return self._genai_client

    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        model = self.genai.GenerativeModel(model_config.model_id)

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, str):
                gemini_messages.append({"role": role, "parts": [content]})
            else:
                # Handle multimodal content
                gemini_messages.append({"role": role, "parts": content})

        config = {"max_output_tokens": max_tokens}
        if temperature > 0:
            config["temperature"] = temperature
        if json_mode:
            config["response_mime_type"] = "application/json"

        response = await model.generate_content_async(
            gemini_messages,
            generation_config=config,
        )

        return {
            "content": response.text,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "model": model_config.model_id,
        }

    async def complete_with_vision(
        self,
        messages: list[dict],
        images: list[bytes],
        model_config: ModelConfig,
        max_tokens: int = 4096,
    ) -> dict:
        import base64

        model = self.genai.GenerativeModel(model_config.model_id)

        # Build content with images
        parts = []

        # Add text from last message
        text = messages[-1]["content"] if messages else ""
        if text:
            parts.append(text)

        # Add images as inline data
        for img in images:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(img).decode(),
                }
            })

        response = await model.generate_content_async(
            parts,
            generation_config={"max_output_tokens": max_tokens},
        )

        return {
            "content": response.text,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "model": model_config.model_id,
        }

    async def computer_use(
        self,
        task: str,
        screenshot: bytes,
        model_config: ModelConfig,
        previous_actions: Optional[list[dict]] = None,
        excluded_actions: Optional[list[str]] = None,
    ) -> dict:
        """
        Execute a Gemini Computer Use request.

        This uses the google-genai SDK with the ComputerUse tool for
        browser automation tasks.

        Args:
            task: Natural language description of what to do
            screenshot: Current screen state as PNG bytes
            model_config: Model configuration (must be gemini-computer-use)
            previous_actions: History of previous actions taken
            excluded_actions: Actions to exclude from model's repertoire

        Returns:
            dict with:
                - action: The action to perform (e.g., "click_at", "type_text_at")
                - parameters: Action parameters (coordinates use 1000x1000 grid)
                - reasoning: Model's reasoning for the action
                - done: Whether the task is complete
                - input_tokens: Token count
                - output_tokens: Token count
        """
        if not self.genai_client:
            raise RuntimeError(
                "google-genai SDK not available. Install with: pip install google-genai"
            )

        try:
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai SDK required for Computer Use")

        import base64

        # Build the Computer Use tool configuration
        computer_use_config = types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER
        )

        if excluded_actions:
            computer_use_config.excluded_predefined_functions = excluded_actions

        # Build content with screenshot
        content_parts = [
            {"text": f"Task: {task}"},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(screenshot).decode(),
                }
            },
        ]

        # Add action history if provided
        if previous_actions:
            history_text = "\n".join([
                f"- {a['action']}: {a.get('parameters', {})}"
                for a in previous_actions[-5:]  # Last 5 actions
            ])
            content_parts.insert(1, {"text": f"Previous actions:\n{history_text}"})

        # Generate content with Computer Use tool
        config = types.GenerateContentConfig(
            tools=[types.Tool(computer_use=computer_use_config)],
            max_output_tokens=4096,
        )

        response = self.genai_client.models.generate_content(
            model=model_config.model_id,
            contents=content_parts,
            config=config,
        )

        # Parse the response
        result = {
            "action": None,
            "parameters": {},
            "reasoning": "",
            "done": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": model_config.model_id,
        }

        # Extract token counts
        if hasattr(response, "usage_metadata"):
            result["input_tokens"] = response.usage_metadata.prompt_token_count or 0
            result["output_tokens"] = response.usage_metadata.candidates_token_count or 0

        # Parse function calls from response
        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    result["reasoning"] = part.text
                elif hasattr(part, "function_call"):
                    fc = part.function_call
                    result["action"] = fc.name
                    result["parameters"] = dict(fc.args) if fc.args else {}

        # Check if task is complete (model returns no action)
        if not result["action"]:
            result["done"] = True

        return result

    async def computer_use_loop(
        self,
        task: str,
        screenshot_fn,
        action_fn,
        model_config: ModelConfig,
        max_iterations: int = 30,
        excluded_actions: Optional[list[str]] = None,
    ) -> dict:
        """
        Run a complete Computer Use agent loop.

        Args:
            task: Natural language task description
            screenshot_fn: Async function that returns current screenshot bytes
            action_fn: Async function that executes an action and returns success
            model_config: Model configuration
            max_iterations: Maximum number of actions to take
            excluded_actions: Actions to exclude

        Returns:
            dict with:
                - success: Whether task completed
                - actions: List of actions taken
                - total_input_tokens: Total tokens used
                - total_output_tokens: Total tokens used
                - iterations: Number of iterations
        """
        actions_taken = []
        total_input = 0
        total_output = 0

        for i in range(max_iterations):
            # Get current screenshot
            screenshot = await screenshot_fn()

            # Get next action from model
            result = await self.computer_use(
                task=task,
                screenshot=screenshot,
                model_config=model_config,
                previous_actions=actions_taken,
                excluded_actions=excluded_actions,
            )

            total_input += result["input_tokens"]
            total_output += result["output_tokens"]

            if result["done"]:
                return {
                    "success": True,
                    "actions": actions_taken,
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "iterations": i + 1,
                }

            # Execute the action
            action_result = await action_fn(
                result["action"],
                result["parameters"]
            )

            actions_taken.append({
                "action": result["action"],
                "parameters": result["parameters"],
                "reasoning": result["reasoning"],
                "success": action_result,
            })

            if not action_result:
                logger.warning(
                    "Action failed",
                    action=result["action"],
                    parameters=result["parameters"],
                )

        return {
            "success": False,
            "actions": actions_taken,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "iterations": max_iterations,
            "error": "Max iterations reached",
        }


class GroqClient(BaseModelClient):
    """Client for Groq (fast Llama inference)."""

    def __init__(self):
        from groq import AsyncGroq
        self.client = AsyncGroq()

    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        return {
            "content": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": model_config.model_id,
        }

    async def complete_with_vision(
        self,
        messages: list[dict],
        images: list[bytes],
        model_config: ModelConfig,
        max_tokens: int = 4096,
    ) -> dict:
        # Groq doesn't support vision yet, fallback
        raise NotImplementedError("Groq doesn't support vision models")


class VertexAIClient(BaseModelClient):
    """
    Client for Claude models via Google Cloud Vertex AI.

    Benefits:
    - Unified GCP billing (use committed spend)
    - Enterprise features (VPC-SC, IAM, audit logging)
    - Regional data residency (EU compliance)
    - Same Claude capabilities including Computer Use

    Setup:
    1. pip install anthropic[vertex]
    2. gcloud auth application-default login
    3. Set GOOGLE_CLOUD_PROJECT env var

    Pricing: Same as direct Anthropic API
    - Global endpoints: No premium
    - Regional endpoints: 10% premium
    """

    # Vertex AI model ID mapping
    VERTEX_MODEL_IDS = {
        # Sonnet models
        "claude-sonnet-4-5-20250514": "claude-sonnet-4-5@20250929",
        "claude-sonnet-4-20250514": "claude-sonnet-4@20250514",
        # Opus models
        "claude-opus-4-5-20250514": "claude-opus-4-5@20251101",
        "claude-opus-4-1-20250805": "claude-opus-4-1@20250805",
        "claude-opus-4-20250514": "claude-opus-4@20250514",
        # Haiku models
        "claude-3-5-haiku-latest": "claude-haiku-4-5@20251001",
        "claude-3-haiku-20240307": "claude-3-haiku@20240307",
    }

    def __init__(self, project_id: Optional[str] = None, region: str = "global"):
        """
        Initialize Vertex AI client.

        Args:
            project_id: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            region: "global" for dynamic routing, or specific region like "us-east1"
                   Regional endpoints have 10% pricing premium but guarantee data residency.
        """
        from anthropic import AnthropicVertex

        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable must be set, "
                "or pass project_id to VertexAIClient"
            )
        self.region = region
        self.client = AnthropicVertex(project_id=self.project_id, region=self.region)

    def _get_vertex_model_id(self, anthropic_model_id: str) -> str:
        """Convert Anthropic model ID to Vertex AI format."""
        return self.VERTEX_MODEL_IDS.get(anthropic_model_id, anthropic_model_id)

    async def complete(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        vertex_model_id = self._get_vertex_model_id(model_config.model_id)

        kwargs = {
            "model": vertex_model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools

        # Vertex AI uses synchronous client, run in executor for async
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.messages.create(**kwargs)
        )

        return {
            "content": response.content[0].text if response.content else "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": vertex_model_id,
        }

    async def complete_with_vision(
        self,
        messages: list[dict],
        images: list[bytes],
        model_config: ModelConfig,
        max_tokens: int = 4096,
    ) -> dict:
        import base64

        # Add images to the last user message (same as AnthropicClient)
        image_content = []
        for img in images:
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode(),
                }
            })

        enhanced_messages = messages.copy()
        if enhanced_messages and enhanced_messages[-1]["role"] == "user":
            if isinstance(enhanced_messages[-1]["content"], str):
                enhanced_messages[-1]["content"] = [
                    {"type": "text", "text": enhanced_messages[-1]["content"]},
                    *image_content
                ]
            else:
                enhanced_messages[-1]["content"].extend(image_content)

        return await self.complete(enhanced_messages, model_config, max_tokens)

    async def computer_use(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> dict:
        """
        Execute Computer Use task via Vertex AI.

        Computer Use is fully supported on Vertex AI with same capabilities
        as direct Anthropic API.
        """
        return await self.complete(
            messages=messages,
            model_config=model_config,
            max_tokens=max_tokens,
            tools=tools,
        )


class ModelRouter:
    """
    Intelligent model router that selects the best model for each task.

    Optimizes for:
    1. Cost - Use cheapest model that can handle the task
    2. Quality - Ensure model capability matches task complexity
    3. Latency - Consider time constraints
    4. Availability - Fallback if primary model unavailable
    5. Budget - Enforce organization budget limits (NEW)

    Usage with cost tracking:
        router = ModelRouter(organization_id="org-uuid", project_id="proj-uuid")
        result = await router.complete(
            task_type=TaskType.TEST_GENERATION,
            messages=[{"role": "user", "content": "Generate a test"}],
        )
        # Usage is automatically recorded to Supabase
    """

    def __init__(
        self,
        prefer_provider: Optional[ModelProvider] = None,
        cost_limit_per_call: float = 0.10,
        enable_fallback: bool = True,
        organization_id: Optional[str] = None,
        project_id: Optional[str] = None,
        enforce_budget: bool = True,
    ):
        self.prefer_provider = prefer_provider
        self.cost_limit = cost_limit_per_call
        self.enable_fallback = enable_fallback

        # Organization context for cost tracking
        self.organization_id = organization_id
        self.project_id = project_id
        self.enforce_budget = enforce_budget

        # Cost tracker integration
        self._cost_tracker = get_cost_tracker()

        # Initialize clients lazily
        self._clients: dict[ModelProvider, BaseModelClient] = {}

        # Track usage for optimization (local session stats)
        self.usage_stats: dict[str, dict] = {}

        # Task type mapping from router TaskType to cost tracker TaskType
        self._task_type_mapping = {
            TaskType.ELEMENT_CLASSIFICATION: CostTaskType.OTHER,
            TaskType.ACTION_EXTRACTION: CostTaskType.OTHER,
            TaskType.SELECTOR_VALIDATION: CostTaskType.OTHER,
            TaskType.TEXT_EXTRACTION: CostTaskType.OTHER,
            TaskType.JSON_PARSING: CostTaskType.OTHER,
            TaskType.CODE_ANALYSIS: CostTaskType.CODE_REVIEW,
            TaskType.TEST_GENERATION: CostTaskType.TEST_GENERATION,
            TaskType.ASSERTION_GENERATION: CostTaskType.TEST_GENERATION,
            TaskType.ERROR_CLASSIFICATION: CostTaskType.ERROR_ANALYSIS,
            TaskType.VISUAL_COMPARISON: CostTaskType.OTHER,
            TaskType.SEMANTIC_UNDERSTANDING: CostTaskType.CORRELATION,
            TaskType.FLOW_DISCOVERY: CostTaskType.OTHER,
            TaskType.ROOT_CAUSE_ANALYSIS: CostTaskType.ERROR_ANALYSIS,
            TaskType.SELF_HEALING: CostTaskType.SELF_HEALING,
            TaskType.FAILURE_PREDICTION: CostTaskType.RISK_ASSESSMENT,
            TaskType.COGNITIVE_MODELING: CostTaskType.OTHER,
            TaskType.COMPLEX_DEBUGGING: CostTaskType.ERROR_ANALYSIS,
            TaskType.COMPUTER_USE_SIMPLE: CostTaskType.OTHER,
            TaskType.COMPUTER_USE_COMPLEX: CostTaskType.OTHER,
            TaskType.COMPUTER_USE_MOBILE: CostTaskType.OTHER,
            TaskType.GENERAL: CostTaskType.OTHER,
        }

    def _get_client(self, provider: ModelProvider) -> BaseModelClient:
        """Get or create a client for the given provider."""
        if provider not in self._clients:
            if provider == ModelProvider.ANTHROPIC:
                self._clients[provider] = AnthropicClient()
            elif provider == ModelProvider.VERTEX_AI:
                self._clients[provider] = VertexAIClient()
            elif provider == ModelProvider.OPENAI:
                self._clients[provider] = OpenAIClient()
            elif provider == ModelProvider.GOOGLE:
                self._clients[provider] = GoogleClient()
            elif provider == ModelProvider.GROQ:
                self._clients[provider] = GroqClient()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        return self._clients[provider]

    def select_model(
        self,
        task_type: TaskType,
        requires_vision: bool = False,
        requires_tools: bool = False,
        max_latency_ms: Optional[int] = None,
        min_quality: Literal["any", "good", "best"] = "good",
    ) -> tuple[str, ModelConfig]:
        """
        Select the best model for the given task and constraints.

        Returns:
            Tuple of (model_name, model_config)
        """
        candidates = TASK_MODEL_MAPPING.get(task_type, ["sonnet"])

        for model_name in candidates:
            config = MODELS.get(model_name)
            if not config:
                continue

            # Check constraints
            if requires_vision and not config.supports_vision:
                continue
            if requires_tools and not config.supports_tools:
                continue
            if max_latency_ms and config.latency_ms > max_latency_ms:
                continue
            if self.prefer_provider and config.provider != self.prefer_provider:
                # Prefer specific provider but don't require it
                pass

            # Check cost limit
            estimated_cost = config.avg_cost_per_1k * 4  # Assume 4K tokens
            if estimated_cost > self.cost_limit:
                continue

            return model_name, config

        # Fallback to Sonnet if nothing matches
        return "sonnet", MODELS["sonnet"]

    async def check_budget(self) -> BudgetStatus:
        """Check if the organization has remaining AI budget.

        Returns:
            BudgetStatus with budget information

        Raises:
            ValueError: If organization_id is not set
        """
        if not self.organization_id:
            # Return unlimited budget if no org context
            return BudgetStatus(
                has_daily_budget=True,
                has_monthly_budget=True,
                daily_remaining=float("inf"),
                monthly_remaining=float("inf"),
                daily_limit=float("inf"),
                monthly_limit=float("inf"),
                daily_used=0,
                monthly_used=0,
            )

        return await self._cost_tracker.check_budget(self.organization_id)

    async def complete(
        self,
        task_type: TaskType,
        messages: list[dict],
        images: Optional[list[bytes]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
        skip_budget_check: bool = False,
    ) -> dict:
        """
        Route the request to the appropriate model and get a completion.

        Args:
            task_type: Type of task to determine model selection
            messages: Chat messages to send
            images: Optional images for vision tasks
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            json_mode: Enable JSON output mode
            tools: Optional tools for function calling
            skip_budget_check: Skip budget enforcement (for critical tasks)

        Returns:
            dict with content, tokens, cost, and model info

        Raises:
            BudgetExceededError: If organization is over budget
        """
        # Check budget before making the call
        if self.enforce_budget and self.organization_id and not skip_budget_check:
            budget = await self.check_budget()
            if not budget.has_daily_budget:
                logger.warning(
                    "Daily AI budget exceeded",
                    organization_id=self.organization_id,
                    daily_used=float(budget.daily_used),
                    daily_limit=float(budget.daily_limit),
                )
                raise BudgetExceededError(
                    f"Daily AI budget exceeded. Used: ${budget.daily_used:.4f} / ${budget.daily_limit:.2f}"
                )
            if not budget.has_monthly_budget:
                logger.warning(
                    "Monthly AI budget exceeded",
                    organization_id=self.organization_id,
                    monthly_used=float(budget.monthly_used),
                    monthly_limit=float(budget.monthly_limit),
                )
                raise BudgetExceededError(
                    f"Monthly AI budget exceeded. Used: ${budget.monthly_used:.4f} / ${budget.monthly_limit:.2f}"
                )

        requires_vision = images is not None and len(images) > 0
        requires_tools = tools is not None and len(tools) > 0

        model_name, config = self.select_model(
            task_type=task_type,
            requires_vision=requires_vision,
            requires_tools=requires_tools,
        )

        client = self._get_client(config.provider)

        # Track latency
        start_time = time.time()

        try:
            if requires_vision:
                result = await client.complete_with_vision(
                    messages=messages,
                    images=images,
                    model_config=config,
                    max_tokens=max_tokens,
                )
            else:
                result = await client.complete(
                    messages=messages,
                    model_config=config,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                    tools=tools,
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # Calculate cost
            input_cost = (result["input_tokens"] / 1_000_000) * config.input_cost_per_1m
            output_cost = (result["output_tokens"] / 1_000_000) * config.output_cost_per_1m
            result["cost"] = input_cost + output_cost
            result["model_name"] = model_name
            result["latency_ms"] = latency_ms

            # Track usage locally
            self._track_usage(model_name, result)

            # Record usage to cost tracker (persists to Supabase)
            if self.organization_id:
                cost_task_type = self._task_type_mapping.get(task_type, CostTaskType.OTHER)
                await self._cost_tracker.record_usage(
                    organization_id=self.organization_id,
                    model=config.model_id,
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    task_type=cost_task_type,
                    project_id=self.project_id,
                    latency_ms=latency_ms,
                    cached=result.get("cached", False),
                    metadata={
                        "task_type": task_type.value,
                        "model_name": model_name,
                        "requires_vision": requires_vision,
                        "requires_tools": requires_tools,
                    },
                )

            return result

        except BudgetExceededError:
            raise
        except Exception as e:
            if self.enable_fallback:
                # Try fallback to Sonnet
                return await self._fallback_complete(
                    task_type, messages, images, max_tokens, temperature, json_mode, tools, str(e)
                )
            raise

    async def _fallback_complete(
        self,
        task_type: TaskType,
        messages: list[dict],
        images: Optional[list[bytes]],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        tools: Optional[list],
        original_error: str,
    ) -> dict:
        """Fallback to Sonnet if primary model fails."""
        config = MODELS["sonnet"]
        client = self._get_client(ModelProvider.ANTHROPIC)

        start_time = time.time()

        if images:
            result = await client.complete_with_vision(
                messages=messages,
                images=images,
                model_config=config,
                max_tokens=max_tokens,
            )
        else:
            result = await client.complete(
                messages=messages,
                model_config=config,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                tools=tools,
            )

        latency_ms = int((time.time() - start_time) * 1000)

        # Calculate cost
        input_cost = (result["input_tokens"] / 1_000_000) * config.input_cost_per_1m
        output_cost = (result["output_tokens"] / 1_000_000) * config.output_cost_per_1m
        result["cost"] = input_cost + output_cost
        result["model_name"] = "sonnet"
        result["latency_ms"] = latency_ms
        result["fallback"] = True
        result["original_error"] = original_error

        # Track usage locally
        self._track_usage("sonnet", result)

        # Record fallback usage to cost tracker
        if self.organization_id:
            cost_task_type = self._task_type_mapping.get(task_type, CostTaskType.OTHER)
            await self._cost_tracker.record_usage(
                organization_id=self.organization_id,
                model=config.model_id,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                task_type=cost_task_type,
                project_id=self.project_id,
                latency_ms=latency_ms,
                metadata={
                    "task_type": task_type.value,
                    "model_name": "sonnet",
                    "fallback": True,
                    "original_error": original_error[:200],  # Truncate error
                },
            )

        return result

    def _track_usage(self, model_name: str, result: dict):
        """Track usage statistics for optimization."""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = {
                "calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
            }

        stats = self.usage_stats[model_name]
        stats["calls"] += 1
        stats["total_input_tokens"] += result.get("input_tokens", 0)
        stats["total_output_tokens"] += result.get("output_tokens", 0)
        stats["total_cost"] += result.get("cost", 0.0)

    def get_cost_report(self) -> dict:
        """Get a cost report for all models used."""
        total_cost = sum(s["total_cost"] for s in self.usage_stats.values())

        return {
            "total_cost": total_cost,
            "by_model": self.usage_stats,
            "potential_savings": self._calculate_savings(),
        }

    def _calculate_savings(self) -> dict:
        """Calculate how much we saved vs. using Sonnet for everything."""
        sonnet_config = MODELS["sonnet"]

        sonnet_cost_if_used = 0.0
        actual_cost = 0.0

        for model_name, stats in self.usage_stats.items():
            actual_cost += stats["total_cost"]

            # Calculate what it would have cost with Sonnet
            input_cost = (stats["total_input_tokens"] / 1_000_000) * sonnet_config.input_cost_per_1m
            output_cost = (stats["total_output_tokens"] / 1_000_000) * sonnet_config.output_cost_per_1m
            sonnet_cost_if_used += input_cost + output_cost

        savings = sonnet_cost_if_used - actual_cost
        savings_percent = (savings / sonnet_cost_if_used * 100) if sonnet_cost_if_used > 0 else 0

        return {
            "actual_cost": actual_cost,
            "sonnet_equivalent": sonnet_cost_if_used,
            "saved": savings,
            "savings_percent": savings_percent,
        }


# Cost Comparison Helper
def estimate_monthly_costs(
    daily_tests: int = 10000,
    tests_with_vision: float = 0.3,  # 30% have visual comparison
    avg_tokens_per_test: int = 6000,
) -> dict:
    """
    Estimate monthly costs for different strategies.
    """
    monthly_tests = daily_tests * 30
    vision_tests = int(monthly_tests * tests_with_vision)
    text_tests = monthly_tests - vision_tests

    # Tokens breakdown (rough estimate)
    # Text test: 4K input, 2K output
    # Vision test: 6K input (with image), 2K output

    strategies = {
        "all_sonnet": {
            "description": "Claude Sonnet for everything",
            "text_cost": text_tests * (4000 * 3.00 + 2000 * 15.00) / 1_000_000,
            "vision_cost": vision_tests * (6000 * 3.00 + 2000 * 15.00) / 1_000_000,
        },
        "multi_model_aggressive": {
            "description": "Cheapest model for each task",
            "text_cost": text_tests * (4000 * 0.15 + 2000 * 0.60) / 1_000_000,  # GPT-4o-mini
            "vision_cost": vision_tests * (6000 * 1.25 + 2000 * 5.00) / 1_000_000,  # Gemini Pro
        },
        "multi_model_balanced": {
            "description": "Balance cost and quality",
            "text_cost": text_tests * (4000 * 0.59 + 2000 * 0.79) / 1_000_000,  # Llama 70B
            "vision_cost": vision_tests * (6000 * 2.50 + 2000 * 10.00) / 1_000_000,  # GPT-4o
        },
    }

    for name, data in strategies.items():
        data["total"] = data["text_cost"] + data["vision_cost"]

    # Calculate savings
    baseline = strategies["all_sonnet"]["total"]
    for name, data in strategies.items():
        data["savings_vs_sonnet"] = baseline - data["total"]
        data["savings_percent"] = (data["savings_vs_sonnet"] / baseline * 100) if baseline > 0 else 0

    return strategies


# Create default router
def create_router(
    prefer_anthropic: bool = False,
    cost_limit: float = 0.10,
) -> ModelRouter:
    """Create a configured model router."""
    return ModelRouter(
        prefer_provider=ModelProvider.ANTHROPIC if prefer_anthropic else None,
        cost_limit_per_call=cost_limit,
        enable_fallback=True,
    )
