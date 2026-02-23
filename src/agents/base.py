"""Base agent class with common functionality.

All specialized agents inherit from BaseAgent which provides:
- Multi-model AI integration (Claude, GPT-4, Gemini, Llama, DeepSeek)
- Intelligent model routing based on task type
- Token tracking and cost estimation
- Structured logging
- JSON response parsing
- Error handling patterns with automatic fallback

RAP-217: Unified _call_ai() abstraction layer
- Routes based on task_type
- Filters by required_capabilities
- Respects preferred_provider
- Enforces cost limits
- Provides automatic fallback

RAP-231: Agent-to-Agent (A2A) Protocol Support
- Capability declarations for inter-agent discovery
- Agent mesh registration for distributed coordination
- Inter-agent querying by capability
"""

import json
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import anthropic
import structlog

from ..config import (
    MODEL_PRICING,
    AgentConfig,
    ModelName,
    MultiModelStrategy,
    get_settings,
)
from ..core.model_router import (
    BudgetExceededError,
    ModelProvider,
    ModelRouter,
    TaskComplexity,
    TaskType,
)
from ..core.providers import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ModelTier,
    OpenRouterProvider,
    ProviderError,
    get_openrouter_provider,
)
from ..core.providers import (
    RateLimitError as ProviderRateLimitError,
)

T = TypeVar("T")


# =============================================================================
# RAP-231: Agent Capability Declarations for A2A Protocol
# =============================================================================


class AgentCapability:
    """Standard agent capabilities for A2A protocol discovery.

    These capability constants allow agents to declare what they can do,
    enabling other agents to discover and query them based on capabilities.

    Usage:
        class MyAgent(BaseAgent):
            CAPABILITIES = [
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.GIT_BLAME,
            ]
    """

    # Code Analysis Capabilities
    CODE_ANALYSIS = "code_analysis"
    GIT_BLAME = "git_blame"
    DEPENDENCY_GRAPH = "dependency_graph"

    # Self-Healing Capabilities
    SELECTOR_FIX = "selector_fix"
    ASSERTION_FIX = "assertion_fix"
    HEALING = "healing"

    # Browser/UI Capabilities
    BROWSER_AUTOMATION = "browser_automation"
    SCREENSHOT = "screenshot"
    DOM_ANALYSIS = "dom_analysis"

    # API Testing Capabilities
    API_TESTING = "api_testing"
    SCHEMA_VALIDATION = "schema_validation"

    # Test Planning Capabilities
    TEST_PLANNING = "test_planning"
    TEST_GENERATION = "test_generation"

    # Advanced Analysis Capabilities
    VISUAL_COMPARISON = "visual_comparison"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    SECURITY_SCAN = "security_scan"
    ACCESSIBILITY_CHECK = "accessibility_check"
    FLAKY_DETECTION = "flaky_detection"
    MR_ANALYSIS = "mr_analysis"  # Merge Request / Code Change Analysis
    QUALITY_SCORING = "quality_scoring"  # Test quality trust scores


class AICapability(str, Enum):
    """Capabilities that can be required for AI calls."""
    VISION = "vision"
    TOOLS = "tools"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    COMPUTER_USE = "computer_use"
    REASONING = "reasoning"  # Extended thinking / reasoning mode
    LONG_CONTEXT = "long_context"  # 100k+ tokens


@dataclass
class AIResponse:
    """Unified response from any AI provider.

    This is the standardized response format returned by _call_ai(),
    regardless of which provider was used.
    """
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    finish_reason: str = "stop"

    # Optional fields
    tool_calls: list[dict] | None = None
    raw_response: Any = None
    fallback_used: bool = False
    fallback_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in the request/response."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "total_tokens": self.total_tokens,
        }
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.fallback_used:
            result["fallback_used"] = self.fallback_used
            result["fallback_reason"] = self.fallback_reason
        return result


@dataclass
class AgentResult[T]:
    """Result from an agent execution."""

    success: bool
    data: T | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    duration_ms: int = 0
    retries: int = 0


@dataclass
class UsageStats:
    """Cumulative usage statistics for an agent."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_calls: int = 0
    total_retries: int = 0

    # Per-provider breakdown
    by_provider: dict[str, dict] = field(default_factory=dict)

    def add_call(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        retries: int = 0,
    ) -> None:
        """Track a call's usage statistics."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.total_calls += 1
        self.total_retries += retries

        # Track per-provider
        if provider not in self.by_provider:
            self.by_provider[provider] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
            }
        self.by_provider[provider]["calls"] += 1
        self.by_provider[provider]["input_tokens"] += input_tokens
        self.by_provider[provider]["output_tokens"] += output_tokens
        self.by_provider[provider]["cost"] += cost


class BaseAgent(ABC):
    """Abstract base class for all testing agents.

    Provides:
    - Multi-model AI client management (Claude, GPT-4, Gemini, Llama)
    - Intelligent model routing based on task type and complexity
    - Token tracking and cost estimation across providers
    - Retry logic with exponential backoff and automatic fallback
    - JSON response parsing
    - Structured logging
    - A2A protocol support for inter-agent communication (RAP-231)

    Subclasses must implement:
    - execute(): Main agent logic
    - _get_system_prompt(): Agent-specific system prompt
    - _get_task_type(): Task type for model routing (optional)

    Subclasses should declare capabilities:
    - CAPABILITIES: List of AgentCapability constants this agent supports
    """

    # Default task type - subclasses should override
    DEFAULT_TASK_TYPE: TaskType = TaskType.GENERAL

    # RAP-231: Agent capabilities for A2A discovery
    # Subclasses should override with their specific capabilities
    CAPABILITIES: list[str] = []

    def __init__(
        self,
        config: AgentConfig | None = None,
        model: ModelName | None = None,
        use_multi_model: bool = True,
        register_with_mesh: bool = False,
    ):
        """Initialize agent with configuration.

        Args:
            config: Optional agent configuration
            model: Override model selection (legacy, single-model mode)
            use_multi_model: Enable multi-model routing for cost optimization
            register_with_mesh: Whether to register with the agent mesh (RAP-231)
        """
        self.settings = get_settings()
        self.config = config or AgentConfig()
        self.model = model or self.settings.default_model
        self.use_multi_model = use_multi_model and self.settings.model_strategy != MultiModelStrategy.ANTHROPIC_ONLY

        self._client: anthropic.Anthropic | None = None
        self._model_router: ModelRouter | None = None
        self._usage = UsageStats()

        # Evaluation tracking
        self._reflexion_scores: list[dict] = []
        self._tool_calls_count: int = 0

        # RAP-231: A2A Protocol support
        # Import here to avoid circular imports
        from ..orchestrator.a2a_protocol import A2AProtocol
        self._a2a_protocol: A2AProtocol | None = None
        self._agent_id: str | None = None

        self.log = structlog.get_logger().bind(
            agent=self.__class__.__name__,
            model=self.model.value,
            multi_model=self.use_multi_model,
        )

        # Auto-register with mesh if requested
        if register_with_mesh:
            self._register_with_mesh()

    @property
    def model_router(self) -> ModelRouter:
        """Lazy-initialize model router."""
        if self._model_router is None:
            from ..core.model_router import ModelProvider
            self._model_router = ModelRouter(
                prefer_provider=ModelProvider(self.settings.prefer_provider.value) if self.settings.prefer_provider else None,
                cost_limit_per_call=self.settings.cost_limit_per_test,
                enable_fallback=self.settings.enable_model_fallback,
            )
        return self._model_router

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy-initialize Anthropic client with Langfuse instrumentation."""
        if self._client is None:
            from ..core.ai_client import get_anthropic_client

            self._client = get_anthropic_client(
                api_key=self.settings.anthropic_api_key.get_secret_value(),
                trace_name=f"agent:{self.__class__.__name__}",
                tags=[f"agent:{self.__class__.__name__.lower()}"],
                metadata={"agent_class": self.__class__.__name__},
            )
        return self._client

    @property
    def usage(self) -> UsageStats:
        """Get cumulative usage statistics."""
        return self._usage

    # =========================================================================
    # RAP-231: Agent-to-Agent (A2A) Protocol Methods
    # =========================================================================

    def _register_with_mesh(self) -> str | None:
        """Register this agent with the agent registry for A2A discovery.

        Registers the agent's capabilities so other agents can discover
        and communicate with it.

        Returns:
            The assigned agent ID, or None if registration failed.

        Example:
            ```python
            agent = CodeAnalyzerAgent(register_with_mesh=True)
            # Agent is now discoverable by other agents
            ```
        """
        try:
            from ..orchestrator.agent_registry import get_agent_registry

            registry = get_agent_registry()
            agent_type = self.__class__.__name__.lower().replace("agent", "")

            # Register with our declared capabilities
            self._agent_id = registry.register(
                agent_type=agent_type,
                capabilities=self.CAPABILITIES,
                metadata={
                    "class": self.__class__.__name__,
                    "model": self.model.value,
                    "multi_model": self.use_multi_model,
                },
            )

            self.log.info(
                "Registered with agent mesh",
                agent_id=self._agent_id,
                capabilities=self.CAPABILITIES,
            )

            return self._agent_id

        except Exception as e:
            self.log.warning(
                "Failed to register with agent mesh",
                error=str(e),
            )
            return None

    async def query_agent(
        self,
        capability: str,
        payload: dict[str, Any],
        timeout_ms: int = 30000,
    ) -> dict[str, Any] | None:
        """Query another agent by capability via A2A protocol.

        Finds an agent that supports the requested capability and sends
        a request to it. This enables inter-agent collaboration.

        Args:
            capability: The capability to query (e.g., AgentCapability.HEALING)
            payload: The request payload to send
            timeout_ms: Request timeout in milliseconds

        Returns:
            The response payload from the target agent, or None if no agent
            was found or the request failed.

        Example:
            ```python
            # From UITesterAgent, request help from SelfHealerAgent
            response = await self.query_agent(
                capability=AgentCapability.SELECTOR_FIX,
                payload={
                    "test_id": "test-123",
                    "failed_selector": "#login-btn",
                    "error": "Element not found",
                },
            )
            if response and response.get("success"):
                new_selector = response["healed_selector"]
            ```
        """
        try:
            from ..orchestrator.agent_registry import get_agent_registry

            # Find an agent with the requested capability
            registry = get_agent_registry()
            agents = registry.discover(capability)

            if not agents:
                self.log.warning(
                    "No agent found with capability",
                    capability=capability,
                )
                return None

            # Select the first healthy agent
            target_agent = agents[0]

            # Initialize A2A protocol if needed
            if self._a2a_protocol is None:
                from ..orchestrator.a2a_protocol import A2AProtocol

                agent_type = self.__class__.__name__.lower().replace("agent", "")
                self._a2a_protocol = A2AProtocol(
                    agent_id=self._agent_id or f"{agent_type}-{id(self)}",
                    agent_type=agent_type,
                )
                await self._a2a_protocol.start()

            # Send the request
            response = await self._a2a_protocol.request(
                to_agent=target_agent.agent_id,
                capability=capability,
                payload=payload,
                timeout_ms=timeout_ms,
            )

            self.log.debug(
                "A2A query completed",
                capability=capability,
                target_agent=target_agent.agent_id,
                success=response.success if response else False,
            )

            return response.payload if response and response.success else None

        except Exception as e:
            self.log.error(
                "A2A query failed",
                capability=capability,
                error=str(e),
            )
            return None

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's main task.

        Must be implemented by subclasses.

        Returns:
            AgentResult with success/failure and data
        """
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent.

        Must be implemented by subclasses.

        Returns:
            System prompt string
        """
        pass

    # =========================================================================
    # RAP-217: Unified _call_ai() Abstraction Layer
    # =========================================================================

    async def _call_ai(
        self,
        messages: list[dict],
        task_type: TaskType | None = None,
        required_capabilities: list[AICapability] | None = None,
        preferred_provider: str | None = None,
        max_cost: float | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        tools: list[dict] | None = None,
        images: list[bytes] | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> AIResponse:
        """Unified AI abstraction layer with intelligent routing.

        This is the recommended method for all AI calls. It provides:
        - Task-based model routing for cost optimization
        - Capability-based model filtering
        - Provider preference with automatic fallback
        - Cost enforcement and budget tracking
        - Automatic retry with exponential backoff
        - Seamless failover between providers

        Args:
            messages: Conversation messages in OpenAI format
            task_type: Type of task for model selection routing.
                Routes to appropriate model tier based on complexity.
            required_capabilities: List of capabilities the model must support.
                E.g., [AICapability.VISION, AICapability.TOOLS]
            preferred_provider: Preferred provider name (e.g., "openrouter", "anthropic").
                Will use this provider if capable, otherwise falls back.
            max_cost: Maximum cost allowed for this call in USD.
                If exceeded, raises BudgetExceededError.
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0.0-2.0)
            system: Optional system prompt override
            tools: Optional tool definitions for function calling
            images: Optional list of image bytes for vision tasks
            json_mode: Enable JSON response format
            timeout: Optional request timeout in seconds

        Returns:
            AIResponse with content, model info, usage stats, and cost

        Raises:
            BudgetExceededError: If max_cost would be exceeded
            ProviderError: If all providers fail

        Example:
            ```python
            response = await self._call_ai(
                messages=[{"role": "user", "content": "Analyze this test failure"}],
                task_type=TaskType.ERROR_CLASSIFICATION,
                required_capabilities=[AICapability.REASONING],
                preferred_provider="openrouter",
                max_cost=0.05,
            )
            print(f"Response: {response.content}")
            print(f"Cost: ${response.cost:.4f}")
            ```
        """
        start_time = time.time()
        task = task_type or self.DEFAULT_TASK_TYPE

        # Determine required capabilities from task and explicit requirements
        capabilities = set(required_capabilities or [])
        if images:
            capabilities.add(AICapability.VISION)
        if tools:
            capabilities.add(AICapability.TOOLS)

        # Format messages with system prompt
        formatted_messages = self._format_messages_for_ai(messages, system)

        # Check cost limits before making the call
        effective_max_cost = max_cost or self.settings.cost_limit_per_test
        if not self._check_cost_limit():
            raise BudgetExceededError(
                f"Agent cost limit exceeded: ${self._usage.total_cost:.4f} / "
                f"${self.settings.cost_limit_per_run:.2f}"
            )

        # Try to use new provider abstraction first (preferred path)
        try:
            response = await self._call_ai_via_provider(
                messages=formatted_messages,
                task_type=task,
                capabilities=capabilities,
                preferred_provider=preferred_provider,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                images=images,
                json_mode=json_mode,
                timeout=timeout,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Calculate cost from model info
            cost = response.get("cost", 0.0)

            # Enforce max_cost
            if effective_max_cost and cost > effective_max_cost:
                self.log.warning(
                    "AI call cost exceeded limit",
                    cost=cost,
                    limit=effective_max_cost,
                    model=response.get("model"),
                )
                # Note: We still return the response but log the warning
                # Actual budget enforcement happens at org level via model_router

            # Build unified response
            ai_response = AIResponse(
                content=response.get("content", ""),
                model=response.get("model", "unknown"),
                provider=response.get("provider", "unknown"),
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                cost=cost,
                latency_ms=latency_ms,
                finish_reason=response.get("finish_reason", "stop"),
                tool_calls=response.get("tool_calls"),
                raw_response=response.get("raw_response"),
                fallback_used=response.get("fallback", False),
                fallback_reason=response.get("original_error"),
            )

            # Track usage
            self._usage.add_call(
                provider=ai_response.provider,
                input_tokens=ai_response.input_tokens,
                output_tokens=ai_response.output_tokens,
                cost=ai_response.cost,
            )

            self.log.debug(
                "_call_ai succeeded",
                model=ai_response.model,
                provider=ai_response.provider,
                input_tokens=ai_response.input_tokens,
                output_tokens=ai_response.output_tokens,
                cost=f"${ai_response.cost:.4f}",
                latency_ms=round(latency_ms, 2),
                fallback_used=ai_response.fallback_used,
            )

            return ai_response

        except BudgetExceededError:
            # Re-raise budget errors
            raise

        except Exception as e:
            # Attempt fallback to legacy _call_model if provider fails
            self.log.warning(
                "_call_ai provider failed, attempting legacy fallback",
                error=str(e),
                task_type=task.value,
            )
            return await self._call_ai_fallback(
                messages=messages,
                task_type=task,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=tools,
                images=images,
                start_time=start_time,
                original_error=str(e),
            )

    async def _call_ai_via_provider(
        self,
        messages: list[dict],
        task_type: TaskType,
        capabilities: set[AICapability],
        preferred_provider: str | None,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        images: list[bytes] | None,
        json_mode: bool,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Make AI call via provider abstraction layer.

        Uses the new provider abstraction from src/core/providers/ when available,
        otherwise falls back to model_router.

        Returns:
            Dict with content, model, provider, tokens, cost, etc.
        """
        # Determine if we should use OpenRouter provider directly
        # Handle both string and enum types for prefer_provider
        settings_provider = self.settings.prefer_provider
        if settings_provider is not None:
            settings_provider_value = settings_provider.value if hasattr(settings_provider, 'value') else str(settings_provider)
        else:
            settings_provider_value = None

        provider_name = preferred_provider or settings_provider_value
        use_openrouter = (
            provider_name in (None, "openrouter")
            and self.settings.openrouter_api_key
        )

        if use_openrouter:
            # Use new provider abstraction
            return await self._call_via_openrouter(
                messages=messages,
                task_type=task_type,
                capabilities=capabilities,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                images=images,
                json_mode=json_mode,
            )
        else:
            # Fall back to model_router for other providers
            return await self._call_via_model_router(
                messages=messages,
                task_type=task_type,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                images=images,
                json_mode=json_mode,
            )

    async def _call_via_openrouter(
        self,
        messages: list[dict],
        task_type: TaskType,
        capabilities: set[AICapability],
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        images: list[bytes] | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        """Make AI call via OpenRouter provider.

        Uses the new OpenRouterProvider from src/core/providers/.
        """
        provider = get_openrouter_provider()

        # Select model based on task type and capabilities
        model_id = self._select_openrouter_model(task_type, capabilities)

        # Convert messages to ChatMessage format
        chat_messages = [ChatMessage(**msg) for msg in messages]

        # Handle vision (images need to be embedded in messages)
        if images:
            chat_messages = self._embed_images_in_messages(chat_messages, images)

        # Make the call
        response: ChatResponse = await provider.chat(
            messages=chat_messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            json_mode=json_mode,
        )

        # Get model info for cost calculation
        model_info = await provider.get_model_info(model_id)
        cost = 0.0
        if model_info:
            cost = model_info.calculate_cost(response.input_tokens, response.output_tokens)

        # Convert tool_calls to dict format
        tool_calls_dict = None
        if response.tool_calls:
            tool_calls_dict = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]

        return {
            "content": response.content,
            "model": response.model,
            "provider": "openrouter",
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": cost,
            "finish_reason": response.finish_reason,
            "tool_calls": tool_calls_dict,
            "raw_response": response.raw_response,
            "latency_ms": response.latency_ms,
        }

    async def _call_via_model_router(
        self,
        messages: list[dict],
        task_type: TaskType,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None,
        images: list[bytes] | None,
        json_mode: bool,
    ) -> dict[str, Any]:
        """Make AI call via legacy model router.

        Falls back to the model_router for providers not yet migrated
        to the new provider abstraction.
        """
        result = await self.model_router.complete(
            task_type=task_type,
            messages=messages,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            tools=tools,
        )

        return {
            "content": result.get("content", ""),
            "model": result.get("model", "unknown"),
            "provider": result.get("model_name", "unknown"),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost": result.get("cost", 0.0),
            "finish_reason": "stop",
            "fallback": result.get("fallback", False),
            "original_error": result.get("original_error"),
            "latency_ms": result.get("latency_ms", 0),
        }

    async def _call_ai_fallback(
        self,
        messages: list[dict],
        task_type: TaskType,
        max_tokens: int,
        temperature: float,
        system: str | None,
        tools: list[dict] | None,
        images: list[bytes] | None,
        start_time: float,
        original_error: str,
    ) -> AIResponse:
        """Fallback to legacy _call_model when provider abstraction fails.

        This ensures backward compatibility during the migration period.
        """
        self.log.info(
            "Using legacy _call_model as fallback",
            task_type=task_type.value,
            original_error=original_error[:100],
        )

        result = await self._call_model(
            messages=messages,
            task_type=task_type,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            images=images,
            tools=tools,
        )

        latency_ms = (time.time() - start_time) * 1000
        usage = result.get("usage", {})

        response = AIResponse(
            content=result.get("content", ""),
            model=result.get("model", "unknown"),
            provider=result.get("provider", "unknown"),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost=result.get("cost", 0.0),
            latency_ms=latency_ms,
            fallback_used=True,
            fallback_reason=original_error,
        )

        # Track usage for fallback
        self._usage.add_call(
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
        )

        return response

    def _format_messages_for_ai(
        self,
        messages: list[dict],
        system: str | None,
    ) -> list[dict]:
        """Format messages with system prompt for AI call.

        Ensures system prompt is properly included in the message list.
        """
        formatted = messages.copy()
        system_prompt = system or self._get_system_prompt()

        if system_prompt:
            # Check if first message is already system
            if formatted and formatted[0].get("role") == "system":
                # Update existing system message
                formatted[0]["content"] = system_prompt
            else:
                # Prepend system message
                formatted.insert(0, {"role": "system", "content": system_prompt})

        return formatted

    def _select_openrouter_model(
        self,
        task_type: TaskType,
        capabilities: set[AICapability],
    ) -> str:
        """Select the best OpenRouter model for the task and capabilities.

        Maps task types to appropriate model tiers and selects models
        that support the required capabilities.

        Returns:
            OpenRouter model ID (e.g., "anthropic/claude-sonnet-4-5")
        """
        # Task type to model tier mapping
        tier_mapping = {
            # Flash tier - cheapest
            TaskType.ELEMENT_CLASSIFICATION: ModelTier.FLASH,
            TaskType.ACTION_EXTRACTION: ModelTier.FLASH,
            TaskType.SELECTOR_VALIDATION: ModelTier.FLASH,
            TaskType.TEXT_EXTRACTION: ModelTier.FLASH,
            TaskType.JSON_PARSING: ModelTier.FLASH,
            # Value tier - good quality/cost ratio
            TaskType.CODE_ANALYSIS: ModelTier.VALUE,
            TaskType.TEST_GENERATION: ModelTier.VALUE,
            TaskType.ASSERTION_GENERATION: ModelTier.VALUE,
            TaskType.ERROR_CLASSIFICATION: ModelTier.VALUE,
            # Standard tier - moderate complexity
            TaskType.VISUAL_COMPARISON: ModelTier.STANDARD,
            TaskType.SEMANTIC_UNDERSTANDING: ModelTier.STANDARD,
            TaskType.FLOW_DISCOVERY: ModelTier.STANDARD,
            TaskType.ROOT_CAUSE_ANALYSIS: ModelTier.STANDARD,
            # Premium tier - complex tasks
            TaskType.SELF_HEALING: ModelTier.PREMIUM,
            TaskType.FAILURE_PREDICTION: ModelTier.PREMIUM,
            TaskType.COGNITIVE_MODELING: ModelTier.PREMIUM,
            TaskType.COMPLEX_DEBUGGING: ModelTier.PREMIUM,
            # Expert tier - computer use
            TaskType.COMPUTER_USE_SIMPLE: ModelTier.PREMIUM,
            TaskType.COMPUTER_USE_COMPLEX: ModelTier.EXPERT,
            TaskType.COMPUTER_USE_MOBILE: ModelTier.PREMIUM,
            # General
            TaskType.GENERAL: ModelTier.VALUE,
        }

        tier = tier_mapping.get(task_type, ModelTier.VALUE)

        # Model selection based on tier and capabilities
        needs_vision = AICapability.VISION in capabilities
        needs_computer_use = AICapability.COMPUTER_USE in capabilities
        needs_reasoning = AICapability.REASONING in capabilities

        # Select model based on requirements
        if needs_computer_use:
            return "anthropic/claude-sonnet-4.5"  # Best for computer use
        elif needs_reasoning:
            if tier == ModelTier.EXPERT:
                return "anthropic/claude-opus-4.5"
            return "deepseek/deepseek-r1"  # Best reasoning for cost
        elif needs_vision:
            if tier in (ModelTier.PREMIUM, ModelTier.EXPERT):
                return "anthropic/claude-sonnet-4.5"
            return "google/gemini-2.5-pro"  # Good vision, lower cost
        else:
            # Text-only models by tier
            if tier == ModelTier.FLASH:
                return "google/gemini-2.5-flash-lite"
            elif tier == ModelTier.VALUE:
                return "deepseek/deepseek-chat-v3-0324"
            elif tier == ModelTier.STANDARD:
                return "google/gemini-2.5-flash"
            elif tier == ModelTier.PREMIUM:
                return "anthropic/claude-sonnet-4.5"
            else:  # EXPERT
                return "anthropic/claude-opus-4.5"

    def _embed_images_in_messages(
        self,
        messages: list[ChatMessage],
        images: list[bytes],
    ) -> list[ChatMessage]:
        """Embed images into the last user message for vision tasks.

        Args:
            messages: List of ChatMessage objects
            images: List of image bytes to embed

        Returns:
            Updated messages with images embedded
        """
        import base64

        if not images or not messages:
            return messages

        # Find last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                # Build multimodal content
                content = messages[i].content
                if isinstance(content, str):
                    multimodal_content: list[dict] = [{"type": "text", "text": content}]
                else:
                    multimodal_content = list(content)

                # Add images
                for img in images:
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(img).decode()}"
                        }
                    })

                # Update message
                messages[i] = ChatMessage(
                    role="user",
                    content=multimodal_content,
                    name=messages[i].name,
                    tool_call_id=messages[i].tool_call_id,
                    tool_calls=messages[i].tool_calls,
                )
                break

        return messages

    # =========================================================================
    # Legacy Methods (Deprecated - Use _call_ai() instead)
    # =========================================================================

    def _call_claude(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> anthropic.types.Message:
        """Make a Claude API call with retry logic.

        .. deprecated:: 1.5.0
            Use :meth:`_call_ai` instead, which provides:
            - Multi-provider support (OpenRouter, Azure, Anthropic, etc.)
            - Automatic model routing based on task type
            - Cost optimization and budget enforcement
            - Automatic fallback on failures

        Args:
            messages: Conversation messages
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            tools: Optional tool definitions
            system: Optional system prompt override

        Returns:
            Claude Message response

        Raises:
            anthropic.APIError: After all retries exhausted
        """
        warnings.warn(
            "_call_claude() is deprecated since v1.5.0. "
            "Use _call_ai() instead for multi-provider support, "
            "automatic model routing, and cost optimization.",
            DeprecationWarning,
            stacklevel=2,
        )

        from ..core.retry import RetryPolicy, is_rate_limit_error, is_server_error

        system_prompt = system or self._get_system_prompt()

        # Create retry policy using centralized configuration (RAP-333)
        def should_retry_anthropic(e: Exception) -> bool:
            """Determine if Anthropic exception should be retried."""
            if is_rate_limit_error(e):
                return True
            if is_server_error(e):
                return True
            if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
                return True
            return False

        retry_policy = RetryPolicy(
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_delay,
            max_delay=60.0,
            retry_on=should_retry_anthropic,
            on_retry=lambda attempt, exc, delay: self.log.warning(
                "Retrying Claude API call",
                attempt=attempt,
                error_type=type(exc).__name__,
                delay_seconds=round(delay, 2),
            ),
        )

        def make_request() -> anthropic.types.Message:
            start_time = time.time()

            kwargs = {
                "model": self.model.value,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

            # Track usage
            self._track_usage(response)

            duration = int((time.time() - start_time) * 1000)
            self.log.debug(
                "Claude API call succeeded",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                duration_ms=duration,
            )

            return response

        return retry_policy.execute_sync(make_request)

    async def _call_model(
        self,
        messages: list[dict],
        task_type: TaskType | None = None,
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        system: str | None = None,
        images: list[bytes] | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        """Make an AI model call with intelligent routing.

        Uses multi-model routing when enabled, falls back to Claude otherwise.
        Automatically selects the best model based on task type and complexity.

        Args:
            messages: Conversation messages
            task_type: Type of task for model selection
            complexity: Task complexity level (used for logging, routing uses task_type)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            system: Optional system prompt override
            images: Optional list of image bytes for vision tasks
            tools: Optional list of tool definitions

        Returns:
            Dict with 'content', 'model', 'provider', 'usage' keys
        """
        task = task_type or self.DEFAULT_TASK_TYPE

        # Add system prompt to messages if provided
        formatted_messages = messages.copy()
        system_prompt = system or self._get_system_prompt()
        if system_prompt and formatted_messages:
            # Prepend system message for providers that don't have dedicated system param
            if formatted_messages[0].get("role") != "system":
                formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        if self.use_multi_model:
            # Use intelligent model routing
            result = await self.model_router.complete(
                task_type=task,
                messages=formatted_messages,
                images=images,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            )
            # Normalize response format
            return {
                "content": result.get("content", ""),
                "model": result.get("model", "unknown"),
                "provider": result.get("model_name", "unknown"),
                "usage": {
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                },
                "cost": result.get("cost", 0.0),
            }
        else:
            # Fall back to Claude-only mode
            response = self._call_claude(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=tools,
            )
            return {
                "content": self._extract_text_response(response),
                "model": self.model.value,
                "provider": "anthropic",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

    def _get_task_type(self) -> TaskType:
        """Get the task type for this agent.

        Subclasses should override to return appropriate task type.

        Returns:
            TaskType for model routing
        """
        return self.DEFAULT_TASK_TYPE

    def _track_usage(self, response: anthropic.types.Message) -> None:
        """Track token usage and costs."""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        pricing = MODEL_PRICING[self.model]
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        self._usage.total_input_tokens += input_tokens
        self._usage.total_output_tokens += output_tokens
        self._usage.total_cost += cost
        self._usage.total_calls += 1

    def _parse_json_response(
        self, content: str, fallback: Any | None = None
    ) -> Any:
        """Parse JSON from Claude response, handling code blocks.

        Args:
            content: Response content string
            fallback: Value to return if parsing fails

        Returns:
            Parsed JSON or fallback value
        """
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]

            return json.loads(content.strip())

        except (json.JSONDecodeError, IndexError) as e:
            self.log.warning(
                "Failed to parse JSON response",
                error=str(e),
                content_preview=content[:200] if content else None,
            )
            return fallback

    def _extract_text_response(self, response: anthropic.types.Message) -> str:
        """Extract text content from Claude response.

        Args:
            response: Claude Message response

        Returns:
            Text content string
        """
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _check_cost_limit(self) -> bool:
        """Check if cost limit has been exceeded.

        Returns:
            True if within budget, False if exceeded
        """
        if self._usage.total_cost >= self.settings.cost_limit_per_run:
            self.log.error(
                "Cost limit exceeded",
                current_cost=self._usage.total_cost,
                limit=self.settings.cost_limit_per_run,
            )
            return False
        return True

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self._usage = UsageStats()

    # =========================================================================
    # Agent Evaluation & Benchmarking
    # =========================================================================

    async def tracked_execute(self, **kwargs) -> AgentResult:
        """Execute with automatic evaluation recording.

        Wraps execute() to capture timing, cost, and success metrics,
        then persists them to the agent_evaluations table.

        Evaluation-specific kwargs are popped before forwarding to execute():
            trigger_source, organization_id, org_id, project_id, eval_metadata

        Returns:
            AgentResult from the underlying execute() call.
        """
        # Pop evaluation-specific kwargs so they don't leak into execute()
        trigger_source = kwargs.pop("trigger_source", "api")
        organization_id = kwargs.pop("organization_id", None) or kwargs.pop("org_id", None)
        project_id = kwargs.pop("project_id", None)
        eval_metadata = kwargs.pop("eval_metadata", None)

        start_time = time.time()
        start_cost = self._usage.total_cost
        self._reflexion_scores = []
        self._tool_calls_count = 0
        error_type = None
        error_message = None

        try:
            result = await self.execute(**kwargs)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)[:500]
            result = AgentResult(success=False, error=str(e))

        latency_ms = int((time.time() - start_time) * 1000)
        cost_usd = self._usage.total_cost - start_cost

        # Compute assurance score from reflexion history
        assurance_score = None
        if self._reflexion_scores:
            assurance_score = self._reflexion_scores[-1].get("quality", 0.0)

        await self._record_evaluation(
            task_completed=result.success,
            efficacy_score=self._reflexion_scores[-1].get("quality") if self._reflexion_scores else None,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            assurance_score=assurance_score,
            input_tokens=self._usage.total_input_tokens,
            output_tokens=self._usage.total_output_tokens,
            tool_calls_count=self._tool_calls_count,
            self_corrections=len(self._reflexion_scores),
            error_type=error_type or (None if result.success else "agent_failure"),
            error_message=error_message or result.error,
            trigger_source=trigger_source,
            organization_id=organization_id,
            project_id=project_id,
            metadata=eval_metadata,
        )

        return result

    async def _record_evaluation(
        self,
        task_completed: bool,
        efficacy_score: float | None = None,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        assurance_score: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls_count: int = 0,
        self_corrections: int = 0,
        human_escalated: bool = False,
        error_type: str | None = None,
        error_message: str | None = None,
        trigger_source: str = "api",
        organization_id: str | None = None,
        project_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Persist CLEAR framework metrics for this agent execution.

        This is fire-and-forget — failures are logged but never propagated.
        Agent execution should never fail because metrics recording broke.
        """
        try:
            from ..services.supabase_client import get_supabase_client

            supabase = get_supabase_client()
            # organization_id column is UUID NOT NULL — use sentinel for benchmarks/anonymous
            ANONYMOUS_ORG_UUID = "00000000-0000-0000-0000-000000000000"
            org_id = organization_id or ANONYMOUS_ORG_UUID

            evaluation = {
                "organization_id": org_id,
                "project_id": project_id,
                "agent_type": self.__class__.__name__,
                "task_type": self.DEFAULT_TASK_TYPE.value if hasattr(self.DEFAULT_TASK_TYPE, "value") else str(self.DEFAULT_TASK_TYPE),
                "task_completed": task_completed,
                "efficacy_score": efficacy_score,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "assurance_score": assurance_score,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tool_calls_count": tool_calls_count,
                "self_corrections": self_corrections,
                "human_escalated": human_escalated,
                "error_type": error_type,
                "error_message": error_message[:500] if error_message else None,
                "trigger_source": trigger_source,
                "metadata": metadata,
            }

            await supabase.insert("agent_evaluations", evaluation)

            self.log.debug(
                "Recorded agent evaluation",
                agent_type=self.__class__.__name__,
                task_completed=task_completed,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
            )
        except Exception as e:
            # Never let evaluation recording break agent execution
            self.log.warning(
                "Failed to record agent evaluation",
                error=str(e),
                agent_type=self.__class__.__name__,
            )

    # =========================================================================
    # RAP-250: Reflexion Middleware - Self-Improving Agent Pattern
    # =========================================================================

    async def _call_ai_with_reflexion(
        self,
        messages: list[dict],
        task_type: TaskType | None = None,
        required_capabilities: list[AICapability] | None = None,
        max_reflexion_rounds: int = 3,
        quality_threshold: float = 0.8,
        critique_prompt: str | None = None,
        refinement_prompt: str | None = None,
        max_cost: float | None = None,
        **kwargs,
    ) -> "ReflexionResult":
        """Execute AI call with Reflexion pattern: Execute → Critique → Refine → Repeat.

        Implements the Reflexion pattern from Shinn et al. (2023) for self-improving
        agent outputs. Each round:
        1. Execute: Generate initial response
        2. Critique: Self-evaluate the response quality
        3. Refine: If below threshold, refine based on critique
        4. Repeat: Until quality threshold met or max rounds reached

        Args:
            messages: Conversation messages in OpenAI format
            task_type: Type of task for model selection routing
            required_capabilities: List of capabilities the model must support
            max_reflexion_rounds: Maximum reflection iterations (default: 3)
            quality_threshold: Quality score (0-1) to accept output (default: 0.8)
            critique_prompt: Custom prompt for self-critique (optional)
            refinement_prompt: Custom prompt for refinement (optional)
            max_cost: Maximum total cost across all rounds
            **kwargs: Additional args passed to _call_ai()

        Returns:
            ReflexionResult with final response, critique history, and metrics

        Example:
            ```python
            result = await self._call_ai_with_reflexion(
                messages=[{"role": "user", "content": "Generate test cases for auth.py"}],
                task_type=TaskType.TEST_GENERATION,
                max_reflexion_rounds=3,
                quality_threshold=0.85,
            )
            print(f"Final output: {result.response.content}")
            print(f"Quality: {result.final_quality_score}")
            print(f"Rounds: {result.rounds_used}")
            ```
        """
        from dataclasses import dataclass, field as dataclass_field

        @dataclass
        class CritiqueResult:
            """Result of a self-critique round."""
            quality_score: float
            strengths: list[str]
            weaknesses: list[str]
            suggestions: list[str]
            should_refine: bool

        @dataclass
        class ReflexionRound:
            """Record of a single reflexion round."""
            round_number: int
            response: AIResponse
            critique: CritiqueResult
            refined: bool
            cost: float

        rounds: list[ReflexionRound] = []
        total_cost = 0.0
        effective_max_cost = max_cost or self.settings.cost_limit_per_test * 3

        # Default critique prompt
        default_critique_prompt = """Evaluate the quality of the following response:

<response>
{response}
</response>

<original_request>
{request}
</original_request>

Provide a structured critique in JSON format:
{{
    "quality_score": <float 0.0-1.0>,
    "strengths": [<list of specific strengths>],
    "weaknesses": [<list of specific weaknesses>],
    "suggestions": [<list of actionable improvements>],
    "should_refine": <boolean - true if quality_score < threshold>
}}

Be rigorous but fair. Consider:
- Completeness: Does it fully address the request?
- Correctness: Are there any errors or inaccuracies?
- Clarity: Is it well-structured and easy to understand?
- Practicality: Is it actionable and implementable?

Quality threshold for acceptance: {threshold}"""

        # Default refinement prompt
        default_refinement_prompt = """Improve the following response based on the critique:

<original_response>
{response}
</original_response>

<critique>
Weaknesses: {weaknesses}
Suggestions: {suggestions}
</critique>

Generate an improved response that:
1. Addresses all identified weaknesses
2. Incorporates the suggestions
3. Maintains the strengths of the original

Provide the improved response directly, without any preamble."""

        # Round 1: Initial execution
        self.log.info(
            "Starting Reflexion execution",
            max_rounds=max_reflexion_rounds,
            threshold=quality_threshold,
        )

        current_response = await self._call_ai(
            messages=messages,
            task_type=task_type,
            required_capabilities=required_capabilities,
            max_cost=effective_max_cost,
            **kwargs,
        )
        total_cost += current_response.cost

        for round_num in range(1, max_reflexion_rounds + 1):
            # Check cost budget
            if total_cost >= effective_max_cost:
                self.log.warning(
                    "Reflexion stopped: cost budget exceeded",
                    total_cost=total_cost,
                    max_cost=effective_max_cost,
                )
                break

            # Self-critique
            critique_text = (critique_prompt or default_critique_prompt).format(
                response=current_response.content,
                request=messages[-1].get("content", "") if messages else "",
                threshold=quality_threshold,
            )

            critique_response = await self._call_ai(
                messages=[{"role": "user", "content": critique_text}],
                task_type=TaskType.ERROR_CLASSIFICATION,  # Use lighter model for critique
                json_mode=True,
                max_cost=(effective_max_cost - total_cost) / 2,
                **{k: v for k, v in kwargs.items() if k not in ["json_mode"]},
            )
            total_cost += critique_response.cost

            # Parse critique
            try:
                critique_data = self._parse_json_response(critique_response.content, {})
                critique = CritiqueResult(
                    quality_score=float(critique_data.get("quality_score", 0.5)),
                    strengths=critique_data.get("strengths", []),
                    weaknesses=critique_data.get("weaknesses", []),
                    suggestions=critique_data.get("suggestions", []),
                    should_refine=critique_data.get("should_refine", True),
                )
            except Exception as e:
                self.log.warning("Failed to parse critique, assuming quality met", error=str(e))
                critique = CritiqueResult(
                    quality_score=quality_threshold,
                    strengths=["Unable to parse critique"],
                    weaknesses=[],
                    suggestions=[],
                    should_refine=False,
                )

            # Persist reflexion score for evaluation tracking
            self._reflexion_scores.append({
                "round": round_num,
                "quality": critique.quality_score,
                "weaknesses_count": len(critique.weaknesses),
            })

            # Record round
            rounds.append(ReflexionRound(
                round_number=round_num,
                response=current_response,
                critique=critique,
                refined=critique.should_refine and critique.quality_score < quality_threshold,
                cost=current_response.cost + critique_response.cost,
            ))

            self.log.debug(
                "Reflexion round complete",
                round=round_num,
                quality_score=critique.quality_score,
                should_refine=critique.should_refine,
            )

            # Check if quality threshold met
            if critique.quality_score >= quality_threshold or not critique.should_refine:
                self.log.info(
                    "Reflexion complete: quality threshold met",
                    rounds_used=round_num,
                    final_quality=critique.quality_score,
                )
                break

            # Refine if needed and not last round
            if round_num < max_reflexion_rounds:
                refinement_text = (refinement_prompt or default_refinement_prompt).format(
                    response=current_response.content,
                    weaknesses="; ".join(critique.weaknesses),
                    suggestions="; ".join(critique.suggestions),
                )

                # Add refinement request to conversation
                refined_messages = messages + [
                    {"role": "assistant", "content": current_response.content},
                    {"role": "user", "content": refinement_text},
                ]

                current_response = await self._call_ai(
                    messages=refined_messages,
                    task_type=task_type,
                    required_capabilities=required_capabilities,
                    max_cost=effective_max_cost - total_cost,
                    **kwargs,
                )
                total_cost += current_response.cost

        # Build final result
        final_round = rounds[-1] if rounds else None

        @dataclass
        class ReflexionResult:
            """Final result of Reflexion execution."""
            response: AIResponse
            rounds: list
            rounds_used: int
            final_quality_score: float
            total_cost: float
            critique_history: list[dict]
            improved: bool

        return ReflexionResult(
            response=current_response,
            rounds=rounds,
            rounds_used=len(rounds),
            final_quality_score=final_round.critique.quality_score if final_round else 0.0,
            total_cost=total_cost,
            critique_history=[
                {
                    "round": r.round_number,
                    "quality": r.critique.quality_score,
                    "weaknesses": r.critique.weaknesses,
                    "refined": r.refined,
                }
                for r in rounds
            ],
            improved=len(rounds) > 1 and rounds[-1].critique.quality_score > rounds[0].critique.quality_score,
        )

    # =========================================================================
    # RAP-251: Agent-as-Judge Pattern for Output Evaluation
    # =========================================================================

    async def _evaluate_output(
        self,
        output: str,
        criteria: list[str] | None = None,
        rubric: dict[str, str] | None = None,
        reference: str | None = None,
    ) -> "EvaluationResult":
        """Evaluate an output using Agent-as-Judge pattern.

        Uses a separate evaluation pass to assess output quality against
        specified criteria. This is useful for:
        - Validating generated code
        - Assessing test coverage
        - Checking compliance with requirements

        Args:
            output: The output to evaluate
            criteria: List of criteria to evaluate (default: accuracy, completeness, clarity)
            rubric: Optional detailed rubric with scoring guidelines
            reference: Optional reference/gold standard to compare against

        Returns:
            EvaluationResult with scores and feedback

        Example:
            ```python
            result = await self._evaluate_output(
                output=generated_code,
                criteria=["correctness", "efficiency", "readability"],
                rubric={
                    "correctness": "Code should pass all test cases",
                    "efficiency": "O(n) or better time complexity",
                },
            )
            if result.overall_score < 0.7:
                # Trigger refinement
                pass
            ```
        """
        from dataclasses import dataclass

        @dataclass
        class CriterionScore:
            """Score for a single criterion."""
            criterion: str
            score: float  # 0.0-1.0
            feedback: str
            evidence: str

        @dataclass
        class EvaluationResult:
            """Result of Agent-as-Judge evaluation."""
            overall_score: float
            scores: dict[str, CriterionScore]
            summary: str
            pass_fail: bool
            recommendations: list[str]

        default_criteria = criteria or ["accuracy", "completeness", "clarity", "practicality"]

        # Build evaluation prompt
        rubric_text = ""
        if rubric:
            rubric_text = "\n".join([f"- {k}: {v}" for k, v in rubric.items()])

        reference_text = ""
        if reference:
            reference_text = f"\n\n<reference>\n{reference}\n</reference>"

        eval_prompt = f"""You are an expert evaluator. Assess the following output against the specified criteria.

<output>
{output}
</output>{reference_text}

Criteria to evaluate: {', '.join(default_criteria)}

{f"Rubric:{chr(10)}{rubric_text}" if rubric_text else ""}

Provide your evaluation in JSON format:
{{
    "overall_score": <float 0.0-1.0>,
    "criteria_scores": {{
        "<criterion>": {{
            "score": <float 0.0-1.0>,
            "feedback": "<specific feedback>",
            "evidence": "<quote or reference from output>"
        }}
    }},
    "summary": "<2-3 sentence summary>",
    "pass_fail": <boolean - true if overall_score >= 0.7>,
    "recommendations": [<list of specific improvements>]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": eval_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=2000,
        )

        # Parse evaluation
        eval_data = self._parse_json_response(response.content, {})

        scores = {}
        for criterion in default_criteria:
            criterion_data = eval_data.get("criteria_scores", {}).get(criterion, {})
            scores[criterion] = CriterionScore(
                criterion=criterion,
                score=float(criterion_data.get("score", 0.5)),
                feedback=criterion_data.get("feedback", "No feedback provided"),
                evidence=criterion_data.get("evidence", ""),
            )

        return EvaluationResult(
            overall_score=float(eval_data.get("overall_score", 0.5)),
            scores=scores,
            summary=eval_data.get("summary", "Evaluation complete"),
            pass_fail=eval_data.get("pass_fail", False),
            recommendations=eval_data.get("recommendations", []),
        )

    # =========================================================================
    # RAP-330: Hallucination Detection via SelfCheckGPT
    # =========================================================================

    async def _call_ai_with_hallucination_check(
        self,
        messages: list[dict],
        query: str,
        context: str | None = None,
        hallucination_threshold: float = 0.6,
        num_samples: int = 3,
        auto_retry: bool = False,
        max_retries: int = 2,
        **kwargs,
    ) -> tuple[AIResponse, "HallucinationResult | None"]:
        """Execute AI call with automatic hallucination detection.

        Uses the SelfCheckGPT pattern to detect hallucinated content by
        generating multiple samples and checking consistency.

        Args:
            messages: Conversation messages
            query: The query being answered (for regeneration)
            context: Optional grounding context to check against
            hallucination_threshold: Minimum score to accept (0.0-1.0)
            num_samples: Number of samples for consistency check
            auto_retry: Whether to retry if hallucination detected
            max_retries: Maximum retry attempts if auto_retry=True
            **kwargs: Additional args passed to _call_ai()

        Returns:
            Tuple of (AIResponse, HallucinationResult or None if check disabled)

        Example:
            ```python
            response, halluc_result = await self._call_ai_with_hallucination_check(
                messages=[{"role": "user", "content": "Explain the auth flow"}],
                query="Explain the auth flow",
                context=api_docs,
                hallucination_threshold=0.7,
            )

            if halluc_result and not halluc_result.is_reliable:
                self.log.warning(
                    "Response may contain hallucinations",
                    hallucinated=halluc_result.hallucinated_sentences,
                )
            ```
        """
        from .hallucination_detector import HallucinationDetectorAgent

        # Generate initial response
        response = await self._call_ai(messages=messages, **kwargs)

        # Create detector
        detector = HallucinationDetectorAgent(
            num_samples=num_samples,
            consistency_threshold=hallucination_threshold,
            enable_grounding=context is not None,
        )

        current_response = response
        current_messages = messages.copy()

        for attempt in range(max_retries + 1 if auto_retry else 1):
            # Check for hallucinations
            result = await detector.detect(
                response=current_response.content,
                query=query,
                context=context,
                messages=current_messages,
            )

            if result.overall_score >= hallucination_threshold:
                self.log.debug(
                    "Response passed hallucination check",
                    score=result.overall_score,
                    attempt=attempt + 1,
                )
                return current_response, result

            if not auto_retry or attempt >= max_retries:
                self.log.warning(
                    "Response failed hallucination check",
                    score=result.overall_score,
                    severity=result.severity.value,
                    hallucinated_count=len(result.hallucinated_sentences),
                )
                return current_response, result

            # Retry with feedback
            self.log.info(
                "Retrying due to hallucination detection",
                attempt=attempt + 1,
                hallucinated=result.hallucinated_sentences[:2],
            )

            current_messages.append({"role": "assistant", "content": current_response.content})
            current_messages.append({
                "role": "user",
                "content": f"""Your response may contain inaccuracies. The following statements could not be verified:
{chr(10).join(f'- {s}' for s in result.hallucinated_sentences[:3])}

Please provide a more accurate response, focusing only on verifiable information."""
            })

            current_response = await self._call_ai(messages=current_messages, **kwargs)

        return current_response, result
