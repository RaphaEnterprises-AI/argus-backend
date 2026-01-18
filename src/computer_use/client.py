"""Claude Computer Use API client wrapper with cost tracking."""

import asyncio
import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import structlog

from ..config import MODEL_PRICING, ModelName, Settings, estimate_screenshot_tokens

logger = structlog.get_logger()


@dataclass
class UsageStats:
    """Track API usage and costs."""
    input_tokens: int = 0
    output_tokens: int = 0
    screenshots_taken: int = 0
    actions_executed: int = 0
    api_calls: int = 0

    @property
    def total_cost(self) -> float:
        """Calculate total cost based on Sonnet pricing."""
        pricing = MODEL_PRICING[ModelName.SONNET]
        return (
            self.input_tokens * pricing["input"] / 1_000_000 +
            self.output_tokens * pricing["output"] / 1_000_000
        )


@dataclass
class ActionResult:
    """Result of a computer use action."""
    action: str
    success: bool
    screenshot: bytes | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a complete computer use task."""
    success: bool
    final_response: str
    usage: UsageStats
    actions_taken: list[ActionResult]
    screenshots: list[bytes]
    error: str | None = None
    iterations: int = 0


class ComputerUseClient:
    """
    Wrapper for Claude Computer Use API with full agent loop implementation.

    Usage:
        client = ComputerUseClient(settings)
        result = await client.execute_task(
            task="Login to example.com with test credentials",
            screenshot_fn=capture_screen,
            action_fn=execute_action,
        )
    """

    BETA_HEADER = "computer-use-2025-01-24"
    TOOL_VERSION = "20250124"
    TEXT_EDITOR_VERSION = "20250728"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self.model = settings.default_model.value

    def _get_tools(self) -> list[dict]:
        """Get Computer Use tool definitions."""
        return [
            {
                "type": f"computer_{self.TOOL_VERSION}",
                "name": "computer",
                "display_width_px": self.settings.screenshot_width,
                "display_height_px": self.settings.screenshot_height,
                "display_number": 1,
            },
            {
                "type": f"bash_{self.TOOL_VERSION}",
                "name": "bash",
            },
            {
                "type": f"text_editor_{self.TEXT_EDITOR_VERSION}",
                "name": "str_replace_based_edit_tool",
            },
        ]

    async def execute_task(
        self,
        task: str,
        screenshot_fn: Callable[[], bytes],
        action_fn: Callable[[dict], Any],
        max_iterations: int | None = None,
        system_prompt: str | None = None,
    ) -> TaskResult:
        """
        Execute a computer use task with full agent loop.

        Args:
            task: Natural language description of what to do
            screenshot_fn: Function that captures and returns screenshot bytes
            action_fn: Function that executes actions (click, type, etc.)
            max_iterations: Override default max iterations
            system_prompt: Custom system prompt

        Returns:
            TaskResult with success status, response, and usage stats
        """
        max_iters = max_iterations or self.settings.max_iterations
        usage = UsageStats()
        actions_taken = []
        screenshots = []

        # Default system prompt for testing
        default_system = """You are an autonomous testing agent executing UI tests.

CRITICAL INSTRUCTIONS:
1. After each action, take a screenshot and verify the result before proceeding
2. If an element is not visible, wait and retry up to 3 times
3. If something unexpected happens, document it clearly
4. Be precise with click coordinates - aim for center of elements
5. Always take a final screenshot showing the end state

Execute the task step by step, verifying each action succeeded."""

        messages = [{
            "role": "user",
            "content": task
        }]

        log = logger.bind(task=task[:100])
        log.info("Starting computer use task")

        for iteration in range(max_iters):
            try:
                # Call Claude API
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt or default_system,
                    tools=self._get_tools(),
                    messages=messages,
                    betas=[self.BETA_HEADER],
                )

                # Track usage
                usage.api_calls += 1
                usage.input_tokens += response.usage.input_tokens
                usage.output_tokens += response.usage.output_tokens

                log.debug(
                    "API call completed",
                    iteration=iteration,
                    stop_reason=response.stop_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

                # Check if task is complete
                if response.stop_reason == "end_turn":
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text

                    log.info(
                        "Task completed",
                        iterations=iteration + 1,
                        total_cost=usage.total_cost,
                    )

                    return TaskResult(
                        success=True,
                        final_response=final_text,
                        usage=usage,
                        actions_taken=actions_taken,
                        screenshots=screenshots,
                        iterations=iteration + 1,
                    )

                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await self._handle_tool_call(
                            block,
                            screenshot_fn,
                            action_fn,
                            usage,
                            actions_taken,
                            screenshots,
                        )
                        tool_results.append(result)

                # Continue conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                # Check cost limit
                if usage.total_cost > self.settings.cost_limit_per_test:
                    log.warning("Cost limit exceeded", cost=usage.total_cost)
                    return TaskResult(
                        success=False,
                        final_response="",
                        usage=usage,
                        actions_taken=actions_taken,
                        screenshots=screenshots,
                        error=f"Cost limit exceeded: ${usage.total_cost:.2f}",
                        iterations=iteration + 1,
                    )

            except anthropic.APIError as e:
                log.error("API error", error=str(e))
                return TaskResult(
                    success=False,
                    final_response="",
                    usage=usage,
                    actions_taken=actions_taken,
                    screenshots=screenshots,
                    error=f"API error: {str(e)}",
                    iterations=iteration + 1,
                )

        # Max iterations reached
        log.warning("Max iterations reached", max_iterations=max_iters)
        return TaskResult(
            success=False,
            final_response="",
            usage=usage,
            actions_taken=actions_taken,
            screenshots=screenshots,
            error=f"Max iterations ({max_iters}) reached without completion",
            iterations=max_iters,
        )

    async def _handle_tool_call(
        self,
        block: Any,
        screenshot_fn: Callable[[], bytes],
        action_fn: Callable[[dict], Any],
        usage: UsageStats,
        actions_taken: list[ActionResult],
        screenshots: list[bytes],
    ) -> dict:
        """Handle a single tool call from Claude."""
        tool_name = block.name
        tool_input = block.input

        log = logger.bind(tool=tool_name)

        if tool_name == "computer":
            action = tool_input.get("action")
            log.debug("Executing computer action", action=action, input=tool_input)

            if action == "screenshot":
                # Just take screenshot
                screenshot = screenshot_fn()
                usage.screenshots_taken += 1
                screenshots.append(screenshot)

                # Estimate tokens for this screenshot
                usage.input_tokens += estimate_screenshot_tokens(
                    self.settings.screenshot_width,
                    self.settings.screenshot_height,
                )

                actions_taken.append(ActionResult(
                    action="screenshot",
                    success=True,
                    screenshot=screenshot,
                ))

                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": [{
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(screenshot).decode(),
                        }
                    }]
                }
            else:
                # Execute action then take screenshot
                try:
                    action_fn(tool_input)
                    usage.actions_executed += 1

                    # Small delay for UI to update
                    await asyncio.sleep(0.5)

                    # Capture result
                    screenshot = screenshot_fn()
                    usage.screenshots_taken += 1
                    screenshots.append(screenshot)
                    usage.input_tokens += estimate_screenshot_tokens(
                        self.settings.screenshot_width,
                        self.settings.screenshot_height,
                    )

                    actions_taken.append(ActionResult(
                        action=action,
                        success=True,
                        screenshot=screenshot,
                        metadata=tool_input,
                    ))

                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [{
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(screenshot).decode(),
                            }
                        }]
                    }

                except Exception as e:
                    log.error("Action failed", error=str(e))
                    actions_taken.append(ActionResult(
                        action=action,
                        success=False,
                        error=str(e),
                        metadata=tool_input,
                    ))

                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Action failed: {str(e)}",
                        "is_error": True,
                    }

        elif tool_name == "bash":
            command = tool_input.get("command", "")
            log.debug("Executing bash command", command=command[:100])

            import subprocess
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = result.stdout + result.stderr

                actions_taken.append(ActionResult(
                    action="bash",
                    success=result.returncode == 0,
                    metadata={"command": command, "output": output[:500]},
                ))

                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output or "(no output)",
                }
            except subprocess.TimeoutExpired:
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Command timed out after 30 seconds",
                    "is_error": True,
                }

        elif tool_name == "str_replace_based_edit_tool":
            # Handle text editor tool
            log.debug("Text editor operation", input=tool_input)

            # Implementation depends on how you want to handle file editing
            # For testing, this might not be commonly used
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Text editor operation completed",
            }

        else:
            log.warning("Unknown tool", tool=tool_name)
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Unknown tool: {tool_name}",
                "is_error": True,
            }
