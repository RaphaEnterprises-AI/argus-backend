"""Action execution layer for Computer Use and Playwright."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()


class ActionType(str, Enum):
    """Supported action types."""
    # Navigation
    GOTO = "goto"
    BACK = "back"
    FORWARD = "forward"
    RELOAD = "reload"

    # Mouse actions
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    DRAG = "drag"

    # Keyboard actions
    TYPE = "type"
    FILL = "fill"
    PRESS = "press"
    KEY = "key"

    # Wait actions
    WAIT = "wait"
    WAIT_FOR_SELECTOR = "wait_for_selector"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"
    WAIT_FOR_LOAD_STATE = "wait_for_load_state"

    # Form actions
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"

    # Scroll actions
    SCROLL = "scroll"
    SCROLL_TO = "scroll_to"

    # Screenshot
    SCREENSHOT = "screenshot"

    # Assertions (for verification)
    ASSERT_VISIBLE = "assert_visible"
    ASSERT_TEXT = "assert_text"
    ASSERT_VALUE = "assert_value"
    ASSERT_URL = "assert_url"


@dataclass
class Action:
    """Represents a single action to execute."""
    type: ActionType
    target: str | None = None  # Selector or URL
    value: str | None = None   # Value for fill/type/select
    options: dict = field(default_factory=dict)  # Additional options

    # For coordinate-based actions (Computer Use)
    coordinate: tuple[int, int] | None = None

    # Timing
    timeout_ms: int = 5000
    delay_before_ms: int = 0
    delay_after_ms: int = 100

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "target": self.target,
            "value": self.value,
            "options": self.options,
            "coordinate": self.coordinate,
            "timeout_ms": self.timeout_ms,
        }


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    action: Action
    duration_ms: float = 0.0
    error: str | None = None
    screenshot: bytes | None = None
    return_value: Any = None
    metadata: dict = field(default_factory=dict)


class ActionExecutor(Protocol):
    """Protocol for action executors."""

    async def execute(self, action: Action) -> ActionResult:
        """Execute a single action."""
        ...

    async def screenshot(self) -> bytes:
        """Capture current screen state."""
        ...


class PlaywrightActionExecutor:
    """
    Execute actions using Playwright.

    This is the primary executor for speed and reliability.
    """

    def __init__(self, page):
        """
        Initialize with a Playwright page.

        Args:
            page: Playwright page object
        """
        self.page = page
        self.log = logger.bind(executor="playwright")

    async def execute(self, action: Action) -> ActionResult:
        """Execute an action and return result."""
        import time

        start = time.time()

        # Pre-action delay
        if action.delay_before_ms > 0:
            await asyncio.sleep(action.delay_before_ms / 1000)

        try:
            return_value = await self._execute_action(action)

            # Post-action delay
            if action.delay_after_ms > 0:
                await asyncio.sleep(action.delay_after_ms / 1000)

            duration = (time.time() - start) * 1000

            self.log.debug(
                "Action executed",
                action=action.type.value,
                target=action.target,
                duration_ms=duration,
            )

            return ActionResult(
                success=True,
                action=action,
                duration_ms=duration,
                return_value=return_value,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000

            self.log.error(
                "Action failed",
                action=action.type.value,
                target=action.target,
                error=str(e),
            )

            return ActionResult(
                success=False,
                action=action,
                duration_ms=duration,
                error=str(e),
            )

    async def _execute_action(self, action: Action) -> Any:
        """Execute the specific action type."""
        timeout = action.timeout_ms

        match action.type:
            # Navigation
            case ActionType.GOTO:
                await self.page.goto(action.target, timeout=timeout)
                return self.page.url

            case ActionType.BACK:
                await self.page.go_back(timeout=timeout)
                return self.page.url

            case ActionType.FORWARD:
                await self.page.go_forward(timeout=timeout)
                return self.page.url

            case ActionType.RELOAD:
                await self.page.reload(timeout=timeout)
                return self.page.url

            # Mouse actions
            case ActionType.CLICK:
                if action.coordinate:
                    await self.page.mouse.click(
                        action.coordinate[0],
                        action.coordinate[1],
                    )
                else:
                    await self.page.click(action.target, timeout=timeout)

            case ActionType.DOUBLE_CLICK:
                if action.coordinate:
                    await self.page.mouse.dblclick(
                        action.coordinate[0],
                        action.coordinate[1],
                    )
                else:
                    await self.page.dblclick(action.target, timeout=timeout)

            case ActionType.RIGHT_CLICK:
                if action.coordinate:
                    await self.page.mouse.click(
                        action.coordinate[0],
                        action.coordinate[1],
                        button="right",
                    )
                else:
                    await self.page.click(
                        action.target,
                        button="right",
                        timeout=timeout,
                    )

            case ActionType.HOVER:
                if action.coordinate:
                    await self.page.mouse.move(
                        action.coordinate[0],
                        action.coordinate[1],
                    )
                else:
                    await self.page.hover(action.target, timeout=timeout)

            case ActionType.DRAG:
                # Drag from target to coordinate
                source = await self.page.query_selector(action.target)
                if source and action.coordinate:
                    box = await source.bounding_box()
                    if box:
                        await self.page.mouse.move(
                            box["x"] + box["width"] / 2,
                            box["y"] + box["height"] / 2,
                        )
                        await self.page.mouse.down()
                        await self.page.mouse.move(
                            action.coordinate[0],
                            action.coordinate[1],
                        )
                        await self.page.mouse.up()

            # Keyboard actions
            case ActionType.TYPE:
                if action.target:
                    await self.page.type(
                        action.target,
                        action.value or "",
                        timeout=timeout,
                    )
                else:
                    await self.page.keyboard.type(action.value or "")

            case ActionType.FILL:
                await self.page.fill(
                    action.target,
                    action.value or "",
                    timeout=timeout,
                )

            case ActionType.PRESS:
                await self.page.press(
                    action.target or "body",
                    action.value or "Enter",
                    timeout=timeout,
                )

            case ActionType.KEY:
                await self.page.keyboard.press(action.value or "Enter")

            # Wait actions
            case ActionType.WAIT:
                await asyncio.sleep(int(action.value or 1000) / 1000)

            case ActionType.WAIT_FOR_SELECTOR:
                await self.page.wait_for_selector(
                    action.target,
                    timeout=timeout,
                    state=action.options.get("state", "visible"),
                )

            case ActionType.WAIT_FOR_NAVIGATION:
                await self.page.wait_for_load_state("networkidle", timeout=timeout)

            case ActionType.WAIT_FOR_LOAD_STATE:
                state = action.value or "load"
                await self.page.wait_for_load_state(state, timeout=timeout)

            # Form actions
            case ActionType.SELECT:
                await self.page.select_option(
                    action.target,
                    action.value,
                    timeout=timeout,
                )

            case ActionType.CHECK:
                await self.page.check(action.target, timeout=timeout)

            case ActionType.UNCHECK:
                await self.page.uncheck(action.target, timeout=timeout)

            # Scroll actions
            case ActionType.SCROLL:
                delta = int(action.value or 300)
                direction = action.options.get("direction", "down")
                if direction == "up":
                    delta = -delta
                await self.page.mouse.wheel(0, delta)

            case ActionType.SCROLL_TO:
                element = await self.page.query_selector(action.target)
                if element:
                    await element.scroll_into_view_if_needed()

            # Screenshot
            case ActionType.SCREENSHOT:
                return await self.page.screenshot(type="png")

            # Assertions
            case ActionType.ASSERT_VISIBLE:
                element = await self.page.query_selector(action.target)
                if not element:
                    raise AssertionError(f"Element not found: {action.target}")
                is_visible = await element.is_visible()
                if not is_visible:
                    raise AssertionError(f"Element not visible: {action.target}")
                return True

            case ActionType.ASSERT_TEXT:
                element = await self.page.query_selector(action.target)
                if not element:
                    raise AssertionError(f"Element not found: {action.target}")
                text = await element.text_content()
                expected = action.value or ""
                if expected not in (text or ""):
                    raise AssertionError(
                        f"Text mismatch: expected '{expected}' in '{text}'"
                    )
                return text

            case ActionType.ASSERT_VALUE:
                value = await self.page.input_value(action.target, timeout=timeout)
                expected = action.value or ""
                if value != expected:
                    raise AssertionError(
                        f"Value mismatch: expected '{expected}', got '{value}'"
                    )
                return value

            case ActionType.ASSERT_URL:
                current_url = self.page.url
                expected = action.value or ""
                if expected not in current_url:
                    raise AssertionError(
                        f"URL mismatch: expected '{expected}' in '{current_url}'"
                    )
                return current_url

            case _:
                raise ValueError(f"Unknown action type: {action.type}")

    async def screenshot(self) -> bytes:
        """Capture current page screenshot."""
        return await self.page.screenshot(type="png")

    async def get_element_coordinates(self, selector: str) -> tuple[int, int] | None:
        """
        Get center coordinates of an element.

        Useful for converting selector-based actions to coordinate-based
        for Computer Use.
        """
        element = await self.page.query_selector(selector)
        if not element:
            return None

        box = await element.bounding_box()
        if not box:
            return None

        return (
            int(box["x"] + box["width"] / 2),
            int(box["y"] + box["height"] / 2),
        )


class ComputerUseActionExecutor:
    """
    Execute actions using Claude Computer Use API.

    This executor translates actions to Computer Use tool calls.
    """

    def __init__(self, computer_use_client):
        """
        Initialize with Computer Use client.

        Args:
            computer_use_client: ComputerUseClient instance
        """
        self.client = computer_use_client
        self.log = logger.bind(executor="computer_use")

    def action_to_tool_input(self, action: Action) -> dict:
        """Convert Action to Computer Use tool input."""
        match action.type:
            case ActionType.CLICK:
                if action.coordinate:
                    return {
                        "action": "left_click",
                        "coordinate": list(action.coordinate),
                    }
                else:
                    # Need to find element first
                    raise ValueError(
                        "Computer Use requires coordinates. "
                        "Use get_element_coordinates first."
                    )

            case ActionType.DOUBLE_CLICK:
                return {
                    "action": "double_click",
                    "coordinate": list(action.coordinate),
                }

            case ActionType.RIGHT_CLICK:
                return {
                    "action": "right_click",
                    "coordinate": list(action.coordinate),
                }

            case ActionType.TYPE | ActionType.FILL:
                return {
                    "action": "type",
                    "text": action.value or "",
                }

            case ActionType.KEY | ActionType.PRESS:
                return {
                    "action": "key",
                    "text": action.value or "Return",
                }

            case ActionType.HOVER:
                return {
                    "action": "mouse_move",
                    "coordinate": list(action.coordinate),
                }

            case ActionType.SCROLL:
                direction = action.options.get("direction", "down")
                return {
                    "action": "scroll",
                    "coordinate": list(action.coordinate) if action.coordinate else [960, 540],
                    "direction": direction,
                    "amount": int(action.value or 3),
                }

            case ActionType.WAIT:
                return {
                    "action": "wait",
                    "duration": int(action.value or 1000),
                }

            case ActionType.SCREENSHOT:
                return {
                    "action": "screenshot",
                }

            case _:
                raise ValueError(
                    f"Action type {action.type} not supported for Computer Use"
                )


class HybridActionExecutor:
    """
    Execute actions using Playwright with Computer Use fallback.

    Strategy:
    1. Try Playwright first (faster, more reliable)
    2. If Playwright fails, fall back to Computer Use (more visual)
    """

    def __init__(
        self,
        playwright_executor: PlaywrightActionExecutor,
        computer_use_executor: ComputerUseActionExecutor | None = None,
    ):
        self.playwright = playwright_executor
        self.computer_use = computer_use_executor
        self.log = logger.bind(executor="hybrid")

    async def execute(self, action: Action) -> ActionResult:
        """Execute with fallback strategy."""
        # Try Playwright first
        result = await self.playwright.execute(action)

        if result.success:
            return result

        # If failed and we have Computer Use, try that
        if self.computer_use and action.coordinate:
            self.log.info(
                "Falling back to Computer Use",
                action=action.type.value,
                playwright_error=result.error,
            )

            try:
                self.computer_use.action_to_tool_input(action)
                # Execute via Computer Use client
                # This would need to be implemented in the ComputerUseClient
                # For now, return the original failure
                return result
            except Exception as e:
                self.log.error("Computer Use fallback failed", error=str(e))

        return result

    async def execute_sequence(
        self,
        actions: list[Action],
        stop_on_failure: bool = True,
        screenshot_after_each: bool = False,
    ) -> list[ActionResult]:
        """
        Execute a sequence of actions.

        Args:
            actions: List of actions to execute
            stop_on_failure: Stop if any action fails
            screenshot_after_each: Capture screenshot after each action

        Returns:
            List of action results
        """
        results = []

        for i, action in enumerate(actions):
            self.log.info(
                "Executing action",
                step=i + 1,
                total=len(actions),
                action=action.type.value,
            )

            result = await self.execute(action)

            if screenshot_after_each:
                result.screenshot = await self.playwright.screenshot()

            results.append(result)

            if not result.success and stop_on_failure:
                self.log.warning(
                    "Stopping sequence due to failure",
                    step=i + 1,
                    error=result.error,
                )
                break

        return results


def parse_test_step(step: dict) -> Action:
    """
    Parse a test step from test specification into an Action.

    Args:
        step: Test step dictionary from test spec
              e.g., {"action": "click", "target": "#button", "value": None}

    Returns:
        Action object
    """
    action_type = ActionType(step.get("action", "click"))

    return Action(
        type=action_type,
        target=step.get("target"),
        value=step.get("value"),
        options=step.get("options", {}),
        coordinate=tuple(step["coordinate"]) if step.get("coordinate") else None,
        timeout_ms=step.get("timeout", 5000),
        delay_before_ms=step.get("delay_before", 0),
        delay_after_ms=step.get("delay_after", 100),
    )
