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
import httpx
from abc import ABC, abstractmethod


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
    latency_ms: int = 1000  # Typical latency

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens (assuming 60/40 input/output split)."""
        return (self.input_cost_per_1m * 0.6 + self.output_cost_per_1m * 0.4) / 1000


# Model Registry - All available models with pricing (as of Jan 2025)
MODELS = {
    # ===========================================
    # TIER 1: FLASH (< $0.50/1M tokens)
    # For: Classification, extraction, simple parsing
    # ===========================================

    "gemini-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-flash-latest",
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=300,
    ),

    "gpt-4o-mini": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        max_tokens=16384,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=400,
    ),

    "haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-5-haiku-latest",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        max_tokens=8192,
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
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=800,
    ),

    "gemini-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-pro-latest",
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=600,
    ),

    # ===========================================
    # COMPUTER USE MODELS
    # Specialized for UI automation tasks
    # ===========================================

    "gemini-computer-use": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-computer-use-preview-10-2025",
        input_cost_per_1m=1.25,  # Estimated - Google pricing TBD
        output_cost_per_1m=5.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=True,
        latency_ms=500,
    ),

    "claude-computer-use": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250514",  # Computer use enabled
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=1000,
    ),

    "sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250514",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=1000,
    ),

    "llama-3.1-70b": ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.1-70b-versatile",
        input_cost_per_1m=0.59,
        output_cost_per_1m=0.79,
        max_tokens=8192,
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
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=2000,
    ),

    "o1": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="o1",
        input_cost_per_1m=15.00,
        output_cost_per_1m=60.00,
        max_tokens=100000,
        supports_vision=True,
        supports_tools=False,  # o1 doesn't support tools yet
        supports_json_mode=False,
        latency_ms=5000,  # Reasoning takes time
    ),

    # ===========================================
    # VERTEX AI - Claude via Google Cloud
    # Same pricing, unified GCP billing, Computer Use supported
    # ===========================================

    "vertex-sonnet": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-sonnet-4-5-20250514",  # Will be converted to Vertex format
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=1000,
    ),

    "vertex-opus": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-opus-4-5-20250514",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=2000,
    ),

    "vertex-haiku": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-3-5-haiku-latest",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=500,
    ),

    "vertex-computer-use": ModelConfig(
        provider=ModelProvider.VERTEX_AI,
        model_id="claude-sonnet-4-5-20250514",  # Computer Use enabled
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=8192,
        supports_vision=True,
        supports_tools=True,
        supports_json_mode=False,
        latency_ms=1000,
    ),
}


# Task to Model Mapping
TASK_MODEL_MAPPING: dict[TaskType, list[str]] = {
    # Trivial tasks - use cheapest
    TaskType.ELEMENT_CLASSIFICATION: ["llama-3.1-8b", "gemini-flash", "gpt-4o-mini"],
    TaskType.ACTION_EXTRACTION: ["llama-3.1-8b", "gemini-flash", "gpt-4o-mini"],
    TaskType.SELECTOR_VALIDATION: ["gemini-flash", "gpt-4o-mini", "haiku"],
    TaskType.TEXT_EXTRACTION: ["llama-3.1-8b", "gemini-flash", "gpt-4o-mini"],
    TaskType.JSON_PARSING: ["llama-3.1-8b", "gemini-flash", "gpt-4o-mini"],

    # Moderate tasks - use standard tier
    TaskType.CODE_ANALYSIS: ["deepseek-v3", "gemini-pro", "gpt-4o", "sonnet"],
    TaskType.TEST_GENERATION: ["deepseek-v3", "gpt-4o", "sonnet"],
    TaskType.ASSERTION_GENERATION: ["gemini-pro", "gpt-4o", "sonnet"],
    TaskType.ERROR_CLASSIFICATION: ["llama-3.1-70b", "gpt-4o", "haiku"],

    # Complex tasks - need vision or strong reasoning
    TaskType.VISUAL_COMPARISON: ["gemini-pro", "gpt-4o", "sonnet"],  # Vision required
    TaskType.SEMANTIC_UNDERSTANDING: ["gpt-4o", "sonnet", "gemini-pro"],
    TaskType.FLOW_DISCOVERY: ["gpt-4o", "sonnet"],
    TaskType.ROOT_CAUSE_ANALYSIS: ["sonnet", "gpt-4o", "opus"],

    # Expert tasks - use best available
    TaskType.SELF_HEALING: ["sonnet", "opus", "o1"],
    TaskType.FAILURE_PREDICTION: ["sonnet", "opus"],
    TaskType.COGNITIVE_MODELING: ["opus", "o1", "sonnet"],
    TaskType.COMPLEX_DEBUGGING: ["opus", "o1"],

    # Computer Use tasks - specialized routing
    # Vertex AI provides same capabilities with unified GCP billing
    # Gemini 2.5 Computer Use is ~60% cheaper than Claude for simple tasks
    TaskType.COMPUTER_USE_SIMPLE: ["vertex-computer-use", "gemini-computer-use", "claude-computer-use"],
    TaskType.COMPUTER_USE_COMPLEX: ["vertex-computer-use", "claude-computer-use", "gemini-computer-use"],
    TaskType.COMPUTER_USE_MOBILE: ["gemini-computer-use"],  # Gemini excels at mobile

    # General fallback
    TaskType.GENERAL: ["sonnet", "gpt-4o", "gemini-pro"],
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
    """Client for Google Gemini models."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.genai = genai

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
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

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
        from PIL import Image
        import io

        model = self.genai.GenerativeModel(model_config.model_id)

        # Convert bytes to PIL Images
        pil_images = [Image.open(io.BytesIO(img)) for img in images]

        # Get the text content
        text = messages[-1]["content"] if messages else ""

        response = await model.generate_content_async(
            [text, *pil_images],
            generation_config={"max_output_tokens": max_tokens},
        )

        return {
            "content": response.text,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "model": model_config.model_id,
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
    """

    def __init__(
        self,
        prefer_provider: Optional[ModelProvider] = None,
        cost_limit_per_call: float = 0.10,
        enable_fallback: bool = True,
    ):
        self.prefer_provider = prefer_provider
        self.cost_limit = cost_limit_per_call
        self.enable_fallback = enable_fallback

        # Initialize clients lazily
        self._clients: dict[ModelProvider, BaseModelClient] = {}

        # Track usage for optimization
        self.usage_stats: dict[str, dict] = {}

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

    async def complete(
        self,
        task_type: TaskType,
        messages: list[dict],
        images: Optional[list[bytes]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        json_mode: bool = False,
        tools: Optional[list] = None,
    ) -> dict:
        """
        Route the request to the appropriate model and get a completion.
        """
        requires_vision = images is not None and len(images) > 0
        requires_tools = tools is not None and len(tools) > 0

        model_name, config = self.select_model(
            task_type=task_type,
            requires_vision=requires_vision,
            requires_tools=requires_tools,
        )

        client = self._get_client(config.provider)

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

            # Calculate cost
            input_cost = (result["input_tokens"] / 1_000_000) * config.input_cost_per_1m
            output_cost = (result["output_tokens"] / 1_000_000) * config.output_cost_per_1m
            result["cost"] = input_cost + output_cost
            result["model_name"] = model_name

            # Track usage
            self._track_usage(model_name, result)

            return result

        except Exception as e:
            if self.enable_fallback:
                # Try fallback to Sonnet
                return await self._fallback_complete(
                    messages, images, max_tokens, temperature, json_mode, tools, str(e)
                )
            raise

    async def _fallback_complete(
        self,
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

        result["fallback"] = True
        result["original_error"] = original_error
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
