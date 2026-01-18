"""Base agent class with common functionality.

All specialized agents inherit from BaseAgent which provides:
- Multi-model AI integration (Claude, GPT-4, Gemini, Llama, DeepSeek)
- Intelligent model routing based on task type
- Token tracking and cost estimation
- Structured logging
- JSON response parsing
- Error handling patterns with automatic fallback
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from ..core.model_router import ModelRouter, TaskComplexity, TaskType

T = TypeVar("T")


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


class BaseAgent(ABC):
    """Abstract base class for all testing agents.

    Provides:
    - Multi-model AI client management (Claude, GPT-4, Gemini, Llama)
    - Intelligent model routing based on task type and complexity
    - Token tracking and cost estimation across providers
    - Retry logic with exponential backoff and automatic fallback
    - JSON response parsing
    - Structured logging

    Subclasses must implement:
    - execute(): Main agent logic
    - _get_system_prompt(): Agent-specific system prompt
    - _get_task_type(): Task type for model routing (optional)
    """

    # Default task type - subclasses should override
    DEFAULT_TASK_TYPE: TaskType = TaskType.GENERAL

    def __init__(
        self,
        config: AgentConfig | None = None,
        model: ModelName | None = None,
        use_multi_model: bool = True,
    ):
        """Initialize agent with configuration.

        Args:
            config: Optional agent configuration
            model: Override model selection (legacy, single-model mode)
            use_multi_model: Enable multi-model routing for cost optimization
        """
        self.settings = get_settings()
        self.config = config or AgentConfig()
        self.model = model or self.settings.default_model
        self.use_multi_model = use_multi_model and self.settings.model_strategy != MultiModelStrategy.ANTHROPIC_ONLY

        self._client: anthropic.Anthropic | None = None
        self._model_router: ModelRouter | None = None
        self._usage = UsageStats()
        self.log = structlog.get_logger().bind(
            agent=self.__class__.__name__,
            model=self.model.value,
            multi_model=self.use_multi_model,
        )

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

    def _call_claude(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> anthropic.types.Message:
        """Make a Claude API call with retry logic.

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
