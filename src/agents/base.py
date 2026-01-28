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
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key.get_secret_value()
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
            return "anthropic/claude-sonnet-4"  # Best for computer use
        elif needs_reasoning:
            if tier == ModelTier.EXPERT:
                return "anthropic/claude-opus-4"
            return "deepseek/deepseek-r1"  # Best reasoning for cost
        elif needs_vision:
            if tier in (ModelTier.PREMIUM, ModelTier.EXPERT):
                return "anthropic/claude-sonnet-4"
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
                return "anthropic/claude-sonnet-4"
            else:  # EXPERT
                return "anthropic/claude-opus-4"

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
        system_prompt = system or self._get_system_prompt()
        retries = 0
        last_error = None

        while retries <= self.config.max_retries:
            try:
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

            except anthropic.RateLimitError as e:
                retries += 1
                last_error = e
                wait_time = self.config.retry_delay * (2 ** (retries - 1))
                self.log.warning(
                    "Rate limited, retrying",
                    retry=retries,
                    wait_seconds=wait_time,
                )
                time.sleep(wait_time)

            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    retries += 1
                    last_error = e
                    wait_time = self.config.retry_delay * (2 ** (retries - 1))
                    self.log.warning(
                        "Server error, retrying",
                        status_code=e.status_code,
                        retry=retries,
                    )
                    time.sleep(wait_time)
                else:
                    raise

        self._usage.total_retries += retries
        raise last_error or Exception("Max retries exceeded")

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
