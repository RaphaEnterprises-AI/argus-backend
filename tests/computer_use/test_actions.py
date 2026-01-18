"""Tests for the computer_use actions module."""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_types_exist(self):
        """Test that all action types are defined."""
        from src.computer_use.actions import ActionType

        # Navigation
        assert ActionType.GOTO.value == "goto"
        assert ActionType.BACK.value == "back"
        assert ActionType.FORWARD.value == "forward"
        assert ActionType.RELOAD.value == "reload"

        # Mouse
        assert ActionType.CLICK.value == "click"
        assert ActionType.DOUBLE_CLICK.value == "double_click"
        assert ActionType.RIGHT_CLICK.value == "right_click"
        assert ActionType.HOVER.value == "hover"

        # Keyboard
        assert ActionType.TYPE.value == "type"
        assert ActionType.FILL.value == "fill"
        assert ActionType.PRESS.value == "press"

        # Wait
        assert ActionType.WAIT.value == "wait"
        assert ActionType.WAIT_FOR_SELECTOR.value == "wait_for_selector"

        # Screenshot
        assert ActionType.SCREENSHOT.value == "screenshot"


class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation_basic(self):
        """Test basic Action creation."""
        from src.computer_use.actions import Action, ActionType

        action = Action(
            type=ActionType.CLICK,
            target="#button",
        )

        assert action.type == ActionType.CLICK
        assert action.target == "#button"
        assert action.value is None
        assert action.timeout_ms == 5000

    def test_action_with_all_fields(self):
        """Test Action with all fields."""
        from src.computer_use.actions import Action, ActionType

        action = Action(
            type=ActionType.FILL,
            target="#email",
            value="test@example.com",
            options={"force": True},
            coordinate=(100, 200),
            timeout_ms=10000,
            delay_before_ms=500,
            delay_after_ms=200,
        )

        assert action.value == "test@example.com"
        assert action.options["force"] is True
        assert action.coordinate == (100, 200)
        assert action.delay_before_ms == 500

    def test_action_to_dict(self):
        """Test Action to_dict method."""
        from src.computer_use.actions import Action, ActionType

        action = Action(
            type=ActionType.GOTO,
            target="https://example.com",
        )

        result = action.to_dict()

        assert result["type"] == "goto"
        assert result["target"] == "https://example.com"
        assert result["timeout_ms"] == 5000


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_result_creation(self):
        """Test ActionResult creation."""
        from src.computer_use.actions import Action, ActionResult, ActionType

        action = Action(type=ActionType.CLICK, target="#btn")
        result = ActionResult(
            success=True,
            action=action,
            duration_ms=150.5,
        )

        assert result.success is True
        assert result.duration_ms == 150.5
        assert result.error is None

    def test_result_with_error(self):
        """Test ActionResult with error."""
        from src.computer_use.actions import Action, ActionResult, ActionType

        action = Action(type=ActionType.CLICK, target="#missing")
        result = ActionResult(
            success=False,
            action=action,
            error="Element not found",
        )

        assert result.success is False
        assert "Element not found" in result.error

    def test_result_with_screenshot(self):
        """Test ActionResult with screenshot."""
        from src.computer_use.actions import Action, ActionResult, ActionType

        action = Action(type=ActionType.SCREENSHOT)
        result = ActionResult(
            success=True,
            action=action,
            screenshot=b"PNG_DATA",
            return_value=b"PNG_DATA",
        )

        assert result.screenshot == b"PNG_DATA"


class TestPlaywrightActionExecutor:
    """Tests for PlaywrightActionExecutor class."""

    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page."""
        page = MagicMock()
        page.goto = AsyncMock()
        page.go_back = AsyncMock()
        page.go_forward = AsyncMock()
        page.reload = AsyncMock()
        page.click = AsyncMock()
        page.dblclick = AsyncMock()
        page.hover = AsyncMock()
        page.type = AsyncMock()
        page.fill = AsyncMock()
        page.press = AsyncMock()
        page.select_option = AsyncMock()
        page.check = AsyncMock()
        page.uncheck = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"PNG_DATA")
        page.query_selector = AsyncMock()
        page.input_value = AsyncMock(return_value="test")
        page.url = "https://example.com"
        page.keyboard = MagicMock()
        page.keyboard.type = AsyncMock()
        page.keyboard.press = AsyncMock()
        page.mouse = MagicMock()
        page.mouse.click = AsyncMock()
        page.mouse.dblclick = AsyncMock()
        page.mouse.move = AsyncMock()
        page.mouse.down = AsyncMock()
        page.mouse.up = AsyncMock()
        page.mouse.wheel = AsyncMock()
        return page

    def test_executor_creation(self, mock_page):
        """Test PlaywrightActionExecutor creation."""
        from src.computer_use.actions import PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)

        assert executor.page == mock_page

    @pytest.mark.asyncio
    async def test_execute_goto(self, mock_page):
        """Test GOTO action execution."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.GOTO, target="https://example.com")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_click(self, mock_page):
        """Test CLICK action execution."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.CLICK, target="#button")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_click_with_coordinate(self, mock_page):
        """Test CLICK action with coordinates."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.CLICK, coordinate=(100, 200))

        result = await executor.execute(action)

        assert result.success is True
        mock_page.mouse.click.assert_called_with(100, 200)

    @pytest.mark.asyncio
    async def test_execute_fill(self, mock_page):
        """Test FILL action execution."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.FILL, target="#email", value="test@test.com")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_type(self, mock_page):
        """Test TYPE action execution."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.TYPE, target="#input", value="hello")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.type.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_type_no_target(self, mock_page):
        """Test TYPE action without target uses keyboard."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.TYPE, value="hello")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.keyboard.type.assert_called_once_with("hello")

    @pytest.mark.asyncio
    async def test_execute_wait(self, mock_page):
        """Test WAIT action execution."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.WAIT, value="100")

        result = await executor.execute(action)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_wait_for_selector(self, mock_page):
        """Test WAIT_FOR_SELECTOR action."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.WAIT_FOR_SELECTOR, target="#element")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.wait_for_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_screenshot(self, mock_page):
        """Test SCREENSHOT action execution."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.SCREENSHOT)

        result = await executor.execute(action)

        assert result.success is True
        assert result.return_value == b"PNG_DATA"

    @pytest.mark.asyncio
    async def test_execute_assert_visible_success(self, mock_page):
        """Test successful ASSERT_VISIBLE action."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        mock_element = MagicMock()
        mock_element.is_visible = AsyncMock(return_value=True)
        mock_page.query_selector.return_value = mock_element

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.ASSERT_VISIBLE, target="#element")

        result = await executor.execute(action)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_assert_visible_failure(self, mock_page):
        """Test failed ASSERT_VISIBLE action."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        mock_page.query_selector.return_value = None

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.ASSERT_VISIBLE, target="#missing")

        result = await executor.execute(action)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_delay(self, mock_page):
        """Test action execution with delays."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(
            type=ActionType.CLICK,
            target="#button",
            delay_before_ms=50,
            delay_after_ms=50,
        )

        result = await executor.execute(action)

        assert result.success is True
        assert result.duration_ms >= 100  # At least the delays

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, mock_page):
        """Test action execution error handling."""
        from src.computer_use.actions import Action, ActionType, PlaywrightActionExecutor

        mock_page.click.side_effect = Exception("Click failed")

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.CLICK, target="#button")

        result = await executor.execute(action)

        assert result.success is False
        assert "Click failed" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_method(self, mock_page):
        """Test screenshot method."""
        from src.computer_use.actions import PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)

        screenshot = await executor.screenshot()

        assert screenshot == b"PNG_DATA"

    @pytest.mark.asyncio
    async def test_get_element_coordinates(self, mock_page):
        """Test getting element coordinates."""
        from src.computer_use.actions import PlaywrightActionExecutor

        mock_element = MagicMock()
        mock_element.bounding_box = AsyncMock(return_value={
            "x": 100, "y": 200, "width": 50, "height": 30
        })
        mock_page.query_selector.return_value = mock_element

        executor = PlaywrightActionExecutor(mock_page)

        coords = await executor.get_element_coordinates("#button")

        assert coords == (125, 215)  # Center of element

    @pytest.mark.asyncio
    async def test_get_element_coordinates_not_found(self, mock_page):
        """Test getting coordinates for missing element."""
        from src.computer_use.actions import PlaywrightActionExecutor

        mock_page.query_selector.return_value = None

        executor = PlaywrightActionExecutor(mock_page)

        coords = await executor.get_element_coordinates("#missing")

        assert coords is None


class TestComputerUseActionExecutor:
    """Tests for ComputerUseActionExecutor class."""

    def test_executor_creation(self):
        """Test ComputerUseActionExecutor creation."""
        from src.computer_use.actions import ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        assert executor.client == mock_client

    def test_action_to_tool_input_click(self):
        """Test converting click action to tool input."""
        from src.computer_use.actions import Action, ActionType, ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        action = Action(type=ActionType.CLICK, coordinate=(100, 200))
        tool_input = executor.action_to_tool_input(action)

        assert tool_input["action"] == "left_click"
        assert tool_input["coordinate"] == [100, 200]

    def test_action_to_tool_input_type(self):
        """Test converting type action to tool input."""
        from src.computer_use.actions import Action, ActionType, ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        action = Action(type=ActionType.TYPE, value="hello world")
        tool_input = executor.action_to_tool_input(action)

        assert tool_input["action"] == "type"
        assert tool_input["text"] == "hello world"

    def test_action_to_tool_input_key(self):
        """Test converting key action to tool input."""
        from src.computer_use.actions import Action, ActionType, ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        action = Action(type=ActionType.KEY, value="Enter")
        tool_input = executor.action_to_tool_input(action)

        assert tool_input["action"] == "key"
        assert tool_input["text"] == "Enter"

    def test_action_to_tool_input_scroll(self):
        """Test converting scroll action to tool input."""
        from src.computer_use.actions import Action, ActionType, ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        action = Action(
            type=ActionType.SCROLL,
            value="5",
            options={"direction": "down"},
            coordinate=(500, 500),
        )
        tool_input = executor.action_to_tool_input(action)

        assert tool_input["action"] == "scroll"
        assert tool_input["amount"] == 5
        assert tool_input["direction"] == "down"

    def test_action_to_tool_input_click_no_coordinate(self):
        """Test click action without coordinate raises error."""
        from src.computer_use.actions import Action, ActionType, ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        action = Action(type=ActionType.CLICK, target="#button")

        with pytest.raises(ValueError, match="requires coordinates"):
            executor.action_to_tool_input(action)

    def test_action_to_tool_input_unsupported(self):
        """Test unsupported action type raises error."""
        from src.computer_use.actions import Action, ActionType, ComputerUseActionExecutor

        mock_client = MagicMock()
        executor = ComputerUseActionExecutor(mock_client)

        action = Action(type=ActionType.SELECT, target="#dropdown")

        with pytest.raises(ValueError, match="not supported"):
            executor.action_to_tool_input(action)


class TestHybridActionExecutor:
    """Tests for HybridActionExecutor class."""

    @pytest.fixture
    def mock_playwright_executor(self):
        """Create mock Playwright executor."""
        from src.computer_use.actions import Action, ActionResult, ActionType

        executor = MagicMock()
        executor.execute = AsyncMock(return_value=ActionResult(
            success=True,
            action=Action(type=ActionType.CLICK, target="#btn"),
        ))
        executor.screenshot = AsyncMock(return_value=b"PNG")
        return executor

    @pytest.fixture
    def mock_computer_use_executor(self):
        """Create mock Computer Use executor."""
        executor = MagicMock()
        executor.action_to_tool_input = MagicMock(return_value={"action": "left_click"})
        return executor

    def test_hybrid_executor_creation(self, mock_playwright_executor):
        """Test HybridActionExecutor creation."""
        from src.computer_use.actions import HybridActionExecutor

        executor = HybridActionExecutor(mock_playwright_executor)

        assert executor.playwright == mock_playwright_executor
        assert executor.computer_use is None

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_playwright_executor):
        """Test successful hybrid execution."""
        from src.computer_use.actions import Action, ActionType, HybridActionExecutor

        executor = HybridActionExecutor(mock_playwright_executor)
        action = Action(type=ActionType.CLICK, target="#button")

        result = await executor.execute(action)

        assert result.success is True
        mock_playwright_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_sequence(self, mock_playwright_executor):
        """Test executing action sequence."""
        from src.computer_use.actions import Action, ActionType, HybridActionExecutor

        executor = HybridActionExecutor(mock_playwright_executor)
        actions = [
            Action(type=ActionType.GOTO, target="https://example.com"),
            Action(type=ActionType.CLICK, target="#button"),
            Action(type=ActionType.FILL, target="#email", value="test@test.com"),
        ]

        results = await executor.execute_sequence(actions)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_sequence_stop_on_failure(self, mock_playwright_executor):
        """Test sequence stops on failure."""
        from src.computer_use.actions import Action, ActionResult, ActionType, HybridActionExecutor

        # Make second action fail
        mock_playwright_executor.execute = AsyncMock(side_effect=[
            ActionResult(success=True, action=Action(type=ActionType.CLICK, target="#1")),
            ActionResult(success=False, action=Action(type=ActionType.CLICK, target="#2"), error="Failed"),
            ActionResult(success=True, action=Action(type=ActionType.CLICK, target="#3")),
        ])

        executor = HybridActionExecutor(mock_playwright_executor)
        actions = [
            Action(type=ActionType.CLICK, target="#1"),
            Action(type=ActionType.CLICK, target="#2"),
            Action(type=ActionType.CLICK, target="#3"),
        ]

        results = await executor.execute_sequence(actions, stop_on_failure=True)

        assert len(results) == 2  # Stopped after failure

    @pytest.mark.asyncio
    async def test_execute_sequence_with_screenshots(self, mock_playwright_executor):
        """Test sequence with screenshots after each action."""
        from src.computer_use.actions import Action, ActionType, HybridActionExecutor

        executor = HybridActionExecutor(mock_playwright_executor)
        actions = [
            Action(type=ActionType.CLICK, target="#button"),
        ]

        results = await executor.execute_sequence(actions, screenshot_after_each=True)

        assert len(results) == 1
        assert results[0].screenshot == b"PNG"


class TestParseTestStep:
    """Tests for parse_test_step function."""

    def test_parse_click_step(self):
        """Test parsing click step."""
        from src.computer_use.actions import ActionType, parse_test_step

        step = {"action": "click", "target": "#button"}
        action = parse_test_step(step)

        assert action.type == ActionType.CLICK
        assert action.target == "#button"

    def test_parse_fill_step(self):
        """Test parsing fill step."""
        from src.computer_use.actions import ActionType, parse_test_step

        step = {
            "action": "fill",
            "target": "#email",
            "value": "test@example.com",
        }
        action = parse_test_step(step)

        assert action.type == ActionType.FILL
        assert action.value == "test@example.com"

    def test_parse_step_with_coordinate(self):
        """Test parsing step with coordinates."""
        from src.computer_use.actions import parse_test_step

        step = {
            "action": "click",
            "coordinate": [100, 200],
        }
        action = parse_test_step(step)

        assert action.coordinate == (100, 200)

    def test_parse_step_with_options(self):
        """Test parsing step with options."""
        from src.computer_use.actions import parse_test_step

        step = {
            "action": "wait_for_selector",
            "target": "#element",
            "options": {"state": "hidden"},
        }
        action = parse_test_step(step)

        assert action.options["state"] == "hidden"

    def test_parse_step_with_timing(self):
        """Test parsing step with timing options."""
        from src.computer_use.actions import parse_test_step

        step = {
            "action": "click",
            "target": "#button",
            "timeout": 10000,
            "delay_before": 500,
            "delay_after": 200,
        }
        action = parse_test_step(step)

        assert action.timeout_ms == 10000
        assert action.delay_before_ms == 500
        assert action.delay_after_ms == 200
