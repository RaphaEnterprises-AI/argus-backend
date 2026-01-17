"""
Router Agent - Intelligent Model Selection via LLM

This agent decides which model to use for each task based on:
1. Task complexity analysis
2. Required capabilities (vision, tools, etc.)
3. Cost constraints
4. Historical performance data
5. Current context and conversation history

## THE ROUTER'S OWN MODEL SELECTION

The router needs a model to make decisions - but which one?

ANSWER: Use the CHEAPEST, FASTEST model available because:
- Routing is simple classification (not complex reasoning)
- The prompt is small (~500 tokens)
- We want minimal latency overhead

Default hierarchy:
1. Groq Llama 3.1 8B ($0.05/1M tokens, 100ms) - Best choice
2. Gemini Flash ($0.075/1M tokens, 300ms) - If Groq unavailable
3. GPT-4o-mini ($0.15/1M tokens, 400ms) - Fallback
4. Claude Haiku ($0.80/1M tokens, 500ms) - Last resort

Cost per routing decision: ~$0.00003 (negligible)

## CUSTOMIZING THE ROUTER

1. Change router's model: Set ROUTER_MODEL class variable
2. Change routing prompt: Override _get_system_prompt()
3. Add custom rules: Override _is_trivial_task() or add to LOCKED_TASKS
4. Change model registry: Modify MODELS dict in model_router.py
5. A/B test prompts: Use RouterAgentConfig with different prompts

## HOW THE ROUTER KNOWS AVAILABLE MODELS

The router reads from the static MODELS registry in model_router.py.
To add new models:
1. Add to MODELS dict with pricing/capabilities
2. Add to TASK_MODEL_MAPPING for automatic routing
3. The router will automatically consider them

Why an Agent instead of static routing?
- Adapts to novel tasks that don't fit predefined categories
- Learns from success/failure patterns
- Can consider full context, not just task type
- Makes nuanced decisions (e.g., "this looks like simple code but has unusual patterns")
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable
import json
import os
import structlog

from .base import BaseAgent, AgentResult
from .prompts import get_enhanced_prompt
from ..core.model_router import (
    TaskType, TaskComplexity, ModelConfig, ModelProvider,
    MODELS, TASK_MODEL_MAPPING, BaseModelClient
)


logger = structlog.get_logger()


@dataclass
class RouterAgentConfig:
    """Configuration for the Router Agent.

    Use this to customize:
    - Which model the router itself uses
    - Custom system prompts for specialized routing
    - Model priority order
    - Cost/latency constraints
    """

    # Which model the router uses for its own decisions
    # Priority: groq > google > openai > anthropic (by cost)
    router_model: str = "llama-small"
    router_model_fallbacks: list[str] = field(default_factory=lambda: [
        "gemini-flash",
        "gpt-4o-mini",
        "haiku",
    ])

    # Custom system prompt (optional - uses default if None)
    custom_system_prompt: Optional[str] = None

    # Skip LLM routing for tasks under this token count (use heuristics)
    trivial_threshold_tokens: int = 500

    # Maximum cost for routing decision itself
    max_routing_cost: float = 0.001

    # Enable learning from outcomes
    enable_learning: bool = True

    # Custom complexity rules (callable that returns complexity)
    custom_complexity_fn: Optional[Callable[[str], TaskComplexity]] = None


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    model_name: str
    model_config: ModelConfig
    reasoning: str
    confidence: float
    fallback_model: Optional[str] = None
    estimated_cost: float = 0.0


@dataclass
class TaskContext:
    """Context for routing decision."""
    prompt: str
    task_type: Optional[TaskType] = None
    has_images: bool = False
    has_tools: bool = False
    num_tokens_estimate: int = 4000
    max_latency_ms: Optional[int] = None
    max_cost: Optional[float] = None
    conversation_history: list[dict] = field(default_factory=list)
    previous_model_performance: dict = field(default_factory=dict)


class RouterAgent(BaseAgent):
    """
    Intelligent routing agent that selects the best model for each task.

    Uses a cheap, fast model (Llama 8b via Groq) to make routing decisions,
    since the routing task itself is simple classification.

    Key Features:
    - Dynamic complexity analysis
    - Capability matching
    - Cost optimization
    - Learning from historical performance
    - Fallback selection

    Configuration:
        router = RouterAgent(config=RouterAgentConfig(
            router_model="gemini-flash",  # Change router's own model
            custom_system_prompt="...",   # Custom routing instructions
            enable_learning=True,         # Track outcomes
        ))
    """

    # Routing categories
    COMPLEXITY_TIERS = {
        TaskComplexity.TRIVIAL: ["llama-small", "gemini-flash"],
        TaskComplexity.SIMPLE: ["gemini-flash", "gpt-4o-mini", "haiku"],
        TaskComplexity.MODERATE: ["deepseek-v3", "gemini-pro", "gpt-4o"],
        TaskComplexity.COMPLEX: ["sonnet", "gpt-4o", "gemini-pro"],
        TaskComplexity.EXPERT: ["opus", "o1", "sonnet"],
    }

    # Tasks that MUST use specific models (Computer Use - direct API only)
    LOCKED_TASKS = {
        TaskType.COMPUTER_USE_SIMPLE: "claude-computer-use",
        TaskType.COMPUTER_USE_COMPLEX: "claude-computer-use",
        TaskType.COMPUTER_USE_MOBILE: "gemini-computer-use",
    }

    def __init__(self, config: Optional[RouterAgentConfig] = None):
        super().__init__(use_multi_model=False)  # Router doesn't use multi-model for itself
        self.config = config or RouterAgentConfig()
        self._router_client: Optional[BaseModelClient] = None
        self._router_model_name: Optional[str] = None
        self.decision_history: list[dict] = []
        self.performance_tracker: dict[str, dict] = {}

    def _get_router_client(self) -> tuple[BaseModelClient, str]:
        """
        Get the best available client for router's own decisions.

        Tries models in order of cost/speed, falling back if unavailable.

        Returns:
            Tuple of (client, model_name)
        """
        # Try configured model first, then fallbacks
        models_to_try = [self.config.router_model] + self.config.router_model_fallbacks

        for model_name in models_to_try:
            if model_name not in MODELS:
                continue

            model_config = MODELS[model_name]
            provider = model_config.provider

            try:
                if provider == ModelProvider.OPENROUTER:
                    # OpenRouter is the primary provider for all models
                    if os.environ.get("OPENROUTER_API_KEY"):
                        from ..core.model_router import OpenRouterClient
                        return OpenRouterClient(), model_name
                elif provider == ModelProvider.GROQ:
                    # Check if Groq API key is available
                    if os.environ.get("GROQ_API_KEY"):
                        from ..core.model_router import GroqClient
                        return GroqClient(), model_name
                elif provider == ModelProvider.GOOGLE:
                    if os.environ.get("GOOGLE_API_KEY"):
                        from ..core.model_router import GoogleClient
                        return GoogleClient(), model_name
                elif provider == ModelProvider.OPENAI:
                    if os.environ.get("OPENAI_API_KEY"):
                        from ..core.model_router import OpenAIClient
                        return OpenAIClient(), model_name
                elif provider == ModelProvider.ANTHROPIC:
                    # Anthropic is always available (our main provider)
                    from ..core.model_router import AnthropicClient
                    return AnthropicClient(), model_name

            except ImportError as e:
                logger.debug(f"Provider {provider} not available: {e}")
                continue

        # Ultimate fallback - use Anthropic
        logger.warning("No cheap router model available, using Haiku")
        from ..core.model_router import AnthropicClient
        return AnthropicClient(), "haiku"

    @property
    def router_client(self) -> BaseModelClient:
        """Lazy-initialize the router's own client."""
        if self._router_client is None:
            self._router_client, self._router_model_name = self._get_router_client()
            logger.info(f"Router using model: {self._router_model_name}")
        return self._router_client

    @property
    def router_model_name(self) -> str:
        """Get the model name used for routing decisions."""
        if self._router_model_name is None:
            self._router_client, self._router_model_name = self._get_router_client()
        return self._router_model_name

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for routing decisions."""
        # First check for enhanced prompt from prompts.py
        enhanced = get_enhanced_prompt("router_agent")
        if enhanced:
            return enhanced

        # Then use custom prompt if provided in config
        if self.config.custom_system_prompt:
            return self.config.custom_system_prompt

        # Build dynamic model list from registry
        model_info = []
        for name, config in MODELS.items():
            caps = []
            if config.supports_vision:
                caps.append("vision")
            if config.supports_tools:
                caps.append("tools")
            if config.supports_json_mode:
                caps.append("json")
            cap_str = ", ".join(caps) if caps else "text-only"
            model_info.append(
                f"- {name}: ${config.input_cost_per_1m:.2f}/$M in, "
                f"{config.latency_ms}ms, {cap_str}"
            )

        return f"""You are an AI Model Router. Your job is to select the best LLM for a given task.

IMPORTANT CONSTRAINTS:
- Computer Use tasks MUST use: claude-computer-use or gemini-computer-use
- Vision tasks require models with vision support
- Tool-using tasks need models with tool support
- Stay within cost limits
- Prefer cheaper models when they can handle the task

AVAILABLE MODELS:
{chr(10).join(model_info)}

COMPLEXITY TIERS:
- Trivial: Simple extraction, classification, parsing → use cheapest
- Simple: Single-step reasoning → llama, gemini-flash, haiku
- Moderate: Multi-step reasoning → deepseek, gemini-pro, gpt-4o
- Complex: Deep analysis, debugging → sonnet, gpt-4o
- Expert: Novel problems, self-healing → opus, o1

Respond with JSON only:
{{
  "model": "model-name",
  "complexity": "trivial|simple|moderate|complex|expert",
  "reasoning": "brief explanation (1 sentence)",
  "confidence": 0.0-1.0,
  "fallback": "fallback-model-name"
}}"""

    async def execute(self, context: TaskContext) -> AgentResult[RoutingDecision]:
        """Execute routing decision."""

        # Check for locked tasks first (no LLM needed)
        if context.task_type and context.task_type in self.LOCKED_TASKS:
            model_name = self.LOCKED_TASKS[context.task_type]
            return AgentResult(
                success=True,
                data=RoutingDecision(
                    model_name=model_name,
                    model_config=MODELS[model_name],
                    reasoning=f"Task type {context.task_type} requires {model_name}",
                    confidence=1.0,
                ),
            )

        # Quick heuristic for trivial tasks (skip LLM call)
        if self._is_trivial_task(context):
            model_name = "gemini-flash" if context.has_images else "llama-small"
            return AgentResult(
                success=True,
                data=RoutingDecision(
                    model_name=model_name,
                    model_config=MODELS[model_name],
                    reasoning="Trivial task detected via heuristics",
                    confidence=0.9,
                    fallback_model="haiku",
                ),
            )

        # Use LLM for complex routing decisions
        return await self._llm_routing_decision(context)

    def _is_trivial_task(self, context: TaskContext) -> bool:
        """Quick heuristic check for trivial tasks."""
        trivial_indicators = [
            len(context.prompt) < 500,
            "extract" in context.prompt.lower(),
            "classify" in context.prompt.lower(),
            "parse" in context.prompt.lower(),
            context.task_type in [
                TaskType.TEXT_EXTRACTION,
                TaskType.JSON_PARSING,
                TaskType.ELEMENT_CLASSIFICATION,
            ],
        ]
        return sum(trivial_indicators) >= 2

    async def _llm_routing_decision(self, context: TaskContext) -> AgentResult[RoutingDecision]:
        """Use LLM to make routing decision for complex cases."""

        # Build the routing prompt
        prompt = self._build_routing_prompt(context)

        try:
            router_config = MODELS.get(self.router_model_name, MODELS["haiku"])

            result = await self.router_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model_config=router_config,
                max_tokens=500,
                temperature=0.0,
                json_mode=True,
            )

            # Parse the decision
            decision_data = json.loads(result["content"])

            model_name = decision_data.get("model", "sonnet")
            if model_name not in MODELS:
                model_name = "sonnet"

            # Validate model meets requirements
            model_config = MODELS[model_name]
            if context.has_images and not model_config.supports_vision:
                model_name = self._find_vision_model(decision_data.get("complexity", "moderate"))
                model_config = MODELS[model_name]

            if context.has_tools and not model_config.supports_tools:
                model_name = "sonnet"  # Sonnet always supports tools
                model_config = MODELS[model_name]

            decision = RoutingDecision(
                model_name=model_name,
                model_config=model_config,
                reasoning=decision_data.get("reasoning", ""),
                confidence=decision_data.get("confidence", 0.8),
                fallback_model=decision_data.get("fallback", "sonnet"),
                estimated_cost=self._estimate_cost(model_config, context.num_tokens_estimate),
            )

            # Track decision for learning
            self._track_decision(context, decision)

            return AgentResult(success=True, data=decision)

        except Exception as e:
            logger.error("Routing decision failed", error=str(e))
            # Fallback to safe default
            return AgentResult(
                success=True,
                data=RoutingDecision(
                    model_name="sonnet",
                    model_config=MODELS["sonnet"],
                    reasoning=f"Fallback due to routing error: {e}",
                    confidence=0.5,
                    fallback_model="opus",
                ),
            )

    def _build_routing_prompt(self, context: TaskContext) -> str:
        """Build the prompt for routing decision."""

        # Include performance history if available
        history_section = ""
        if context.previous_model_performance:
            history_section = f"""
HISTORICAL PERFORMANCE:
{json.dumps(context.previous_model_performance, indent=2)}
Consider past success rates when selecting models.
"""

        return f"""Select the best model for this task.

TASK DETAILS:
- Task type hint: {context.task_type or 'unknown'}
- Has images: {context.has_images}
- Needs tools: {context.has_tools}
- Token estimate: {context.num_tokens_estimate}
- Max latency: {context.max_latency_ms or 'no limit'} ms
- Max cost: ${context.max_cost or 'no limit'}

PROMPT PREVIEW (first 1000 chars):
{context.prompt[:1000]}

{history_section}

Select the most cost-effective model that can handle this task well.
Respond with JSON only."""

    def _find_vision_model(self, complexity: str) -> str:
        """Find a vision-capable model for the given complexity."""
        vision_models_by_complexity = {
            "trivial": "gemini-flash",
            "simple": "gpt-4o-mini",
            "moderate": "gemini-pro",
            "complex": "gpt-4o",
            "expert": "sonnet",
        }
        return vision_models_by_complexity.get(complexity, "gemini-pro")

    def _estimate_cost(self, config: ModelConfig, tokens: int) -> float:
        """Estimate cost for the given model and token count."""
        # Assume 60/40 input/output split
        input_tokens = tokens * 0.6
        output_tokens = tokens * 0.4
        return (
            (input_tokens / 1_000_000) * config.input_cost_per_1m +
            (output_tokens / 1_000_000) * config.output_cost_per_1m
        )

    def _track_decision(self, context: TaskContext, decision: RoutingDecision) -> None:
        """Track decisions for learning."""
        self.decision_history.append({
            "task_type": context.task_type.value if context.task_type else None,
            "has_images": context.has_images,
            "has_tools": context.has_tools,
            "model_selected": decision.model_name,
            "confidence": decision.confidence,
            "estimated_cost": decision.estimated_cost,
        })

        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def record_outcome(
        self,
        model_name: str,
        task_type: Optional[TaskType],
        success: bool,
        actual_cost: float,
        latency_ms: int,
    ) -> None:
        """Record the outcome of a model call for learning."""
        key = f"{model_name}:{task_type.value if task_type else 'general'}"

        if key not in self.performance_tracker:
            self.performance_tracker[key] = {
                "calls": 0,
                "successes": 0,
                "total_cost": 0.0,
                "total_latency_ms": 0,
            }

        stats = self.performance_tracker[key]
        stats["calls"] += 1
        if success:
            stats["successes"] += 1
        stats["total_cost"] += actual_cost
        stats["total_latency_ms"] += latency_ms

    def get_performance_report(self) -> dict:
        """Get performance statistics for all models."""
        report = {}

        for key, stats in self.performance_tracker.items():
            if stats["calls"] > 0:
                report[key] = {
                    "success_rate": stats["successes"] / stats["calls"],
                    "avg_cost": stats["total_cost"] / stats["calls"],
                    "avg_latency_ms": stats["total_latency_ms"] / stats["calls"],
                    "total_calls": stats["calls"],
                }

        return report


# Integration helper for BaseAgent
async def route_with_agent(
    router: RouterAgent,
    prompt: str,
    task_type: Optional[TaskType] = None,
    images: Optional[list[bytes]] = None,
    tools: Optional[list] = None,
    max_cost: Optional[float] = None,
) -> RoutingDecision:
    """
    Helper function to get routing decision from the Router Agent.

    Usage in BaseAgent:
        decision = await route_with_agent(
            self.router,
            prompt=messages[-1]["content"],
            task_type=self.DEFAULT_TASK_TYPE,
            images=images,
        )
        # Then call decision.model_config.provider with the selected model
    """
    context = TaskContext(
        prompt=prompt,
        task_type=task_type,
        has_images=images is not None and len(images) > 0,
        has_tools=tools is not None and len(tools) > 0,
        max_cost=max_cost,
    )

    result = await router.execute(context)
    return result.data
