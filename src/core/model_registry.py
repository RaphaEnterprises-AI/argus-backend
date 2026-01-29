"""
Centralized Model Registry - SINGLE SOURCE OF TRUTH for all model configurations.

This module provides a unified registry for:
- Model IDs across all providers (Anthropic, Google, OpenAI, Groq, etc.)
- Model capabilities (computer use, vision, tool use, etc.)
- Pricing information
- Rate limits and quotas
- Feature flags

IMPORTANT: ALL code should import model IDs from here, never hardcode them.
"""

import os
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    """AI Model providers."""
    # Primary providers
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"

    # Inference providers (fast/cheap)
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    CEREBRAS = "cerebras"

    # Multi-model router (recommended - 400+ models)
    OPENROUTER = "openrouter"

    # Specialized providers
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    PERPLEXITY = "perplexity"
    COHERE = "cohere"
    XAI = "xai"

    # Enterprise providers
    VERTEX_AI = "vertex_ai"      # Claude via Google Cloud
    AZURE_OPENAI = "azure_openai"  # OpenAI via Azure
    AWS_BEDROCK = "aws_bedrock"    # Claude/Llama via AWS


class Capability(str, Enum):
    """Model capabilities."""
    VISION = "vision"
    TOOL_USE = "tool_use"
    COMPUTER_USE = "computer_use"
    JSON_MODE = "json_mode"
    EXTENDED_CONTEXT = "extended_context"  # 200k+ tokens
    CODE_GENERATION = "code_generation"
    FAST_INFERENCE = "fast_inference"


class TaskType(str, Enum):
    """Task types for model routing."""
    # Fast, cheap tasks
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    VALIDATION = "validation"

    # Medium complexity
    CODE_ANALYSIS = "code_analysis"
    TEST_GENERATION = "test_generation"
    SECURITY_SCAN = "security_scan"
    ACCESSIBILITY_CHECK = "accessibility_check"
    PERFORMANCE_ANALYSIS = "performance_analysis"

    # Complex tasks
    DEBUGGING = "debugging"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    SELF_HEALING = "self_healing"

    # Vision tasks
    VISUAL_COMPARISON = "visual_comparison"
    SCREENSHOT_ANALYSIS = "screenshot_analysis"
    COMPUTER_USE = "computer_use"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    provider: Provider
    display_name: str

    # Pricing (per million tokens)
    input_price: float
    output_price: float

    # Capabilities
    capabilities: set[Capability] = field(default_factory=set)

    # Limits
    max_tokens: int = 4096
    context_window: int = 128000

    # Rate limits (requests per minute)
    rpm_limit: int = 1000
    tpm_limit: int = 100000  # Tokens per minute

    # Feature flags
    is_deprecated: bool = False
    supports_streaming: bool = True
    supports_batching: bool = False

    def has_capability(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def cost_per_1k_tokens(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1000


class ModelRegistry:
    """
    Central registry for all AI models.

    Usage:
        registry = get_model_registry()
        model = registry.get_model_for_task(TaskType.TEST_GENERATION)
        model_id = model.model_id
    """

    # =========================================================================
    # MODEL DEFINITIONS - UPDATE THESE WHEN NEW MODELS ARE RELEASED
    # =========================================================================

    MODELS = {
        # Anthropic Claude Models
        "claude-opus-4-5": ModelConfig(
            model_id="claude-opus-4-5",
            provider=Provider.ANTHROPIC,
            display_name="Claude Opus 4.5",
            input_price=15.00,
            output_price=75.00,
            max_tokens=8192,
            context_window=200000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.COMPUTER_USE,
                Capability.JSON_MODE,
                Capability.EXTENDED_CONTEXT,
                Capability.CODE_GENERATION,
            },
        ),
        "claude-sonnet-4-5": ModelConfig(
            model_id="claude-sonnet-4-5",
            provider=Provider.ANTHROPIC,
            display_name="Claude Sonnet 4.5",
            input_price=3.00,
            output_price=15.00,
            max_tokens=8192,
            context_window=200000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.COMPUTER_USE,
                Capability.JSON_MODE,
                Capability.EXTENDED_CONTEXT,
                Capability.CODE_GENERATION,
            },
        ),
        "claude-haiku-4-5": ModelConfig(
            model_id="claude-haiku-4-5",
            provider=Provider.ANTHROPIC,
            display_name="Claude Haiku 4.5",
            input_price=0.80,
            output_price=4.00,
            max_tokens=8192,
            context_window=200000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.JSON_MODE,
                Capability.FAST_INFERENCE,
            },
        ),

        # Google Gemini Models
        "gemini-2.0-flash": ModelConfig(
            model_id="gemini-2.0-flash-exp",
            provider=Provider.GOOGLE,
            display_name="Gemini 2.0 Flash",
            input_price=0.10,
            output_price=0.40,
            max_tokens=8192,
            context_window=1000000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.JSON_MODE,
                Capability.FAST_INFERENCE,
                Capability.EXTENDED_CONTEXT,
            },
        ),
        "gemini-2.0-pro": ModelConfig(
            model_id="gemini-2.0-pro-exp",
            provider=Provider.GOOGLE,
            display_name="Gemini 2.0 Pro",
            input_price=1.25,
            output_price=5.00,
            max_tokens=8192,
            context_window=2000000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.COMPUTER_USE,  # Gemini 2.0 Pro has computer use!
                Capability.JSON_MODE,
                Capability.EXTENDED_CONTEXT,
                Capability.CODE_GENERATION,
            },
        ),
        "gemini-1.5-pro": ModelConfig(
            model_id="gemini-1.5-pro",
            provider=Provider.GOOGLE,
            display_name="Gemini 1.5 Pro",
            input_price=1.25,
            output_price=5.00,
            max_tokens=8192,
            context_window=2000000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.JSON_MODE,
                Capability.EXTENDED_CONTEXT,
            },
        ),

        # OpenAI Models
        "gpt-4o": ModelConfig(
            model_id="gpt-4o",
            provider=Provider.OPENAI,
            display_name="GPT-4o",
            input_price=2.50,
            output_price=10.00,
            max_tokens=16384,
            context_window=128000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.JSON_MODE,
                Capability.CODE_GENERATION,
            },
        ),
        "gpt-4o-mini": ModelConfig(
            model_id="gpt-4o-mini",
            provider=Provider.OPENAI,
            display_name="GPT-4o Mini",
            input_price=0.15,
            output_price=0.60,
            max_tokens=16384,
            context_window=128000,
            capabilities={
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.JSON_MODE,
                Capability.FAST_INFERENCE,
            },
        ),
        "o1": ModelConfig(
            model_id="o1",
            provider=Provider.OPENAI,
            display_name="o1",
            input_price=15.00,
            output_price=60.00,
            max_tokens=32768,
            context_window=200000,
            capabilities={
                Capability.CODE_GENERATION,
                Capability.EXTENDED_CONTEXT,
            },
        ),

        # Groq (Fast Inference)
        "llama-3.3-70b": ModelConfig(
            model_id="llama-3.3-70b-versatile",
            provider=Provider.GROQ,
            display_name="Llama 3.3 70B (Groq)",
            input_price=0.59,
            output_price=0.79,
            max_tokens=8192,
            context_window=128000,
            capabilities={
                Capability.TOOL_USE,
                Capability.FAST_INFERENCE,
                Capability.CODE_GENERATION,
            },
        ),
        "llama-3.1-8b": ModelConfig(
            model_id="llama-3.1-8b-instant",
            provider=Provider.GROQ,
            display_name="Llama 3.1 8B (Groq)",
            input_price=0.05,
            output_price=0.08,
            max_tokens=8192,
            context_window=128000,
            capabilities={
                Capability.FAST_INFERENCE,
            },
        ),

        # Together AI (Open Models)
        "deepseek-v3": ModelConfig(
            model_id="deepseek-ai/DeepSeek-V3",
            provider=Provider.TOGETHER,
            display_name="DeepSeek V3",
            input_price=0.27,
            output_price=1.10,
            max_tokens=8192,
            context_window=64000,
            capabilities={
                Capability.CODE_GENERATION,
                Capability.TOOL_USE,
            },
        ),
    }

    # =========================================================================
    # TASK TO MODEL MAPPING - Defines which models to use for each task
    # =========================================================================

    TASK_MODEL_PRIORITY = {
        # Fast, cheap tasks - use fast/cheap models first
        TaskType.CLASSIFICATION: ["claude-haiku-4-5", "gemini-2.0-flash", "llama-3.1-8b"],
        TaskType.EXTRACTION: ["claude-haiku-4-5", "gemini-2.0-flash", "gpt-4o-mini"],
        TaskType.VALIDATION: ["claude-haiku-4-5", "gemini-2.0-flash", "llama-3.1-8b"],

        # Medium complexity - use Sonnet or equivalent
        TaskType.CODE_ANALYSIS: ["claude-sonnet-4-5", "gemini-2.0-pro", "gpt-4o"],
        TaskType.TEST_GENERATION: ["claude-sonnet-4-5", "gemini-2.0-pro", "gpt-4o"],
        TaskType.SECURITY_SCAN: ["claude-sonnet-4-5", "gemini-2.0-pro", "gpt-4o"],
        TaskType.ACCESSIBILITY_CHECK: ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o-mini"],
        TaskType.PERFORMANCE_ANALYSIS: ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o"],

        # Complex tasks - use most capable models
        TaskType.DEBUGGING: ["claude-opus-4-5", "claude-sonnet-4-5", "o1"],
        TaskType.ROOT_CAUSE_ANALYSIS: ["claude-opus-4-5", "claude-sonnet-4-5", "gemini-2.0-pro"],
        TaskType.SELF_HEALING: ["claude-sonnet-4-5", "gemini-2.0-pro", "gpt-4o"],

        # Vision/Computer Use tasks - MUST have capability
        TaskType.VISUAL_COMPARISON: ["claude-sonnet-4-5", "gemini-2.0-pro", "gpt-4o"],
        TaskType.SCREENSHOT_ANALYSIS: ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o"],
        TaskType.COMPUTER_USE: ["claude-sonnet-4-5", "claude-opus-4-5", "gemini-2.0-pro"],
    }

    # =========================================================================
    # DEFAULT MODELS - Used when no specific task is specified
    # =========================================================================

    DEFAULT_MODEL = "claude-sonnet-4-5"
    FAST_MODEL = "claude-haiku-4-5"
    POWERFUL_MODEL = "claude-opus-4-5"
    VISION_MODEL = "claude-sonnet-4-5"
    COMPUTER_USE_MODEL = "claude-sonnet-4-5"

    def __init__(self):
        self._models = self.MODELS.copy()
        self._load_overrides()

    def _load_overrides(self):
        """Load model overrides from environment variables."""
        # Allow overriding default models via env vars
        if os.getenv("ARGUS_DEFAULT_MODEL"):
            self.DEFAULT_MODEL = os.getenv("ARGUS_DEFAULT_MODEL")
        if os.getenv("ARGUS_FAST_MODEL"):
            self.FAST_MODEL = os.getenv("ARGUS_FAST_MODEL")
        if os.getenv("ARGUS_POWERFUL_MODEL"):
            self.POWERFUL_MODEL = os.getenv("ARGUS_POWERFUL_MODEL")

    def get_model(self, model_key: str) -> ModelConfig | None:
        """Get a model configuration by key."""
        return self._models.get(model_key)

    def get_model_id(self, model_key: str) -> str:
        """Get just the model ID string for API calls."""
        model = self.get_model(model_key)
        return model.model_id if model else model_key

    def get_default_model(self) -> ModelConfig:
        """Get the default model configuration."""
        return self._models[self.DEFAULT_MODEL]

    def get_fast_model(self) -> ModelConfig:
        """Get the fast/cheap model for quick tasks."""
        return self._models[self.FAST_MODEL]

    def get_powerful_model(self) -> ModelConfig:
        """Get the most powerful model for complex tasks."""
        return self._models[self.POWERFUL_MODEL]

    def get_model_for_task(
        self,
        task_type: TaskType,
        required_capability: Capability | None = None,
        prefer_provider: Provider | None = None,
        budget_remaining: float | None = None,
    ) -> ModelConfig:
        """
        Get the best model for a specific task.

        Args:
            task_type: The type of task to perform
            required_capability: Required capability (e.g., COMPUTER_USE)
            prefer_provider: Preferred provider if available
            budget_remaining: Remaining budget in USD

        Returns:
            Best matching ModelConfig
        """
        priority_list = self.TASK_MODEL_PRIORITY.get(task_type, [self.DEFAULT_MODEL])

        for model_key in priority_list:
            model = self._models.get(model_key)
            if not model:
                continue

            # Check required capability
            if required_capability and not model.has_capability(required_capability):
                continue

            # Check provider preference
            if prefer_provider and model.provider != prefer_provider:
                continue

            # Check budget (simple check for one request)
            if budget_remaining is not None:
                estimated_cost = model.cost_per_1k_tokens(1000, 1000)  # ~1k tokens each way
                if estimated_cost > budget_remaining:
                    continue

            return model

        # Fallback to default
        return self._models[self.DEFAULT_MODEL]

    def get_models_with_capability(self, capability: Capability) -> list[ModelConfig]:
        """Get all models that have a specific capability."""
        return [m for m in self._models.values() if m.has_capability(capability)]

    def get_computer_use_models(self) -> list[ModelConfig]:
        """Get all models that support computer use."""
        return self.get_models_with_capability(Capability.COMPUTER_USE)

    def list_all_models(self) -> list[ModelConfig]:
        """List all registered models."""
        return list(self._models.values())


# Singleton instance
_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# Convenience functions for common operations
def get_model_id(model_key: str) -> str:
    """Get a model ID by key. This is the main function to use."""
    return get_model_registry().get_model_id(model_key)


def get_default_model_id() -> str:
    """Get the default model ID."""
    return get_model_registry().get_default_model().model_id


def get_fast_model_id() -> str:
    """Get the fast model ID."""
    return get_model_registry().get_fast_model().model_id


def get_model_for_task(task_type: TaskType) -> str:
    """Get the best model ID for a task."""
    return get_model_registry().get_model_for_task(task_type).model_id
