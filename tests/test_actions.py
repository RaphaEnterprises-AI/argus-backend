"""Tests for action execution layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation(self):
        """Test creating an action."""
        from src.computer_use.actions import Action, ActionType

        action = Action(
            type=ActionType.CLICK,
            target="#button",
        )

        assert action.type == ActionType.CLICK
        assert action.target == "#button"
        assert action.value is None
        assert action.timeout_ms == 5000

    def test_action_to_dict(self):
        """Test converting action to dict."""
        from src.computer_use.actions import Action, ActionType

        action = Action(
            type=ActionType.FILL,
            target="#input",
            value="test value",
            timeout_ms=10000,
        )

        result = action.to_dict()

        assert result["type"] == "fill"
        assert result["target"] == "#input"
        assert result["value"] == "test value"
        assert result["timeout_ms"] == 10000

    def test_action_with_coordinate(self):
        """Test action with coordinate."""
        from src.computer_use.actions import Action, ActionType

        action = Action(
            type=ActionType.CLICK,
            coordinate=(100, 200),
        )

        assert action.coordinate == (100, 200)
        result = action.to_dict()
        assert result["coordinate"] == (100, 200)


class TestParseTestStep:
    """Tests for parse_test_step function."""

    def test_parse_click_step(self):
        """Test parsing a click step."""
        from src.computer_use.actions import parse_test_step, ActionType

        step = {
            "action": "click",
            "target": "#submit-button",
        }

        action = parse_test_step(step)

        assert action.type == ActionType.CLICK
        assert action.target == "#submit-button"

    def test_parse_fill_step(self):
        """Test parsing a fill step."""
        from src.computer_use.actions import parse_test_step, ActionType

        step = {
            "action": "fill",
            "target": "#email",
            "value": "test@example.com",
        }

        action = parse_test_step(step)

        assert action.type == ActionType.FILL
        assert action.target == "#email"
        assert action.value == "test@example.com"

    def test_parse_step_with_timeout(self):
        """Test parsing step with custom timeout."""
        from src.computer_use.actions import parse_test_step

        step = {
            "action": "wait_for_selector",
            "target": "#loading",
            "timeout": 10000,
        }

        action = parse_test_step(step)

        assert action.timeout_ms == 10000

    def test_parse_step_with_coordinate(self):
        """Test parsing step with coordinate."""
        from src.computer_use.actions import parse_test_step

        step = {
            "action": "click",
            "coordinate": [500, 300],
        }

        action = parse_test_step(step)

        assert action.coordinate == (500, 300)


class TestPlaywrightActionExecutor:
    """Tests for PlaywrightActionExecutor."""

    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page."""
        page = AsyncMock()
        page.url = "https://example.com"
        page.goto = AsyncMock()
        page.click = AsyncMock()
        page.fill = AsyncMock()
        page.type = AsyncMock()
        page.screenshot = AsyncMock(return_value=b"screenshot")
        page.mouse = MagicMock()
        page.mouse.click = AsyncMock()
        page.mouse.wheel = AsyncMock()
        page.keyboard = MagicMock()
        page.keyboard.press = AsyncMock()
        return page

    @pytest.mark.asyncio
    async def test_execute_goto(self, mock_page):
        """Test executing goto action."""
        from src.computer_use.actions import PlaywrightActionExecutor, Action, ActionType

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.GOTO, target="https://example.com/page")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.goto.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_click(self, mock_page):
        """Test executing click action."""
        from src.computer_use.actions import PlaywrightActionExecutor, Action, ActionType

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.CLICK, target="#button")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_fill(self, mock_page):
        """Test executing fill action."""
        from src.computer_use.actions import PlaywrightActionExecutor, Action, ActionType

        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.FILL, target="#input", value="test")

        result = await executor.execute(action)

        assert result.success is True
        mock_page.fill.assert_called_once_with("#input", "test", timeout=5000)

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_page):
        """Test handling action failure."""
        from src.computer_use.actions import PlaywrightActionExecutor, Action, ActionType

        mock_page.click.side_effect = Exception("Element not found")
        executor = PlaywrightActionExecutor(mock_page)
        action = Action(type=ActionType.CLICK, target="#nonexistent")

        result = await executor.execute(action)

        assert result.success is False
        assert "Element not found" in result.error

    @pytest.mark.asyncio
    async def test_screenshot(self, mock_page):
        """Test screenshot capture."""
        from src.computer_use.actions import PlaywrightActionExecutor

        executor = PlaywrightActionExecutor(mock_page)
        result = await executor.screenshot()

        assert result == b"screenshot"


class TestHybridActionExecutor:
    """Tests for HybridActionExecutor."""

    @pytest.fixture
    def mock_playwright_executor(self):
        """Create mock Playwright executor."""
        from src.computer_use.actions import ActionResult, Action, ActionType

        executor = AsyncMock()
        executor.execute = AsyncMock(
            return_value=ActionResult(
                success=True,
                action=Action(type=ActionType.CLICK, target="#button"),
            )
        )
        executor.screenshot = AsyncMock(return_value=b"screenshot")
        return executor

    @pytest.mark.asyncio
    async def test_execute_uses_playwright_first(self, mock_playwright_executor):
        """Test that hybrid executor uses Playwright first."""
        from src.computer_use.actions import HybridActionExecutor, Action, ActionType

        executor = HybridActionExecutor(mock_playwright_executor)
        action = Action(type=ActionType.CLICK, target="#button")

        result = await executor.execute(action)

        assert result.success is True
        mock_playwright_executor.execute.assert_called_once_with(action)

    @pytest.mark.asyncio
    async def test_execute_sequence(self, mock_playwright_executor):
        """Test executing a sequence of actions."""
        from src.computer_use.actions import HybridActionExecutor, Action, ActionType

        executor = HybridActionExecutor(mock_playwright_executor)
        actions = [
            Action(type=ActionType.GOTO, target="https://example.com"),
            Action(type=ActionType.CLICK, target="#button"),
            Action(type=ActionType.FILL, target="#input", value="test"),
        ]

        results = await executor.execute_sequence(actions)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_sequence_stops_on_failure(self, mock_playwright_executor):
        """Test that sequence stops on failure when stop_on_failure=True."""
        from src.computer_use.actions import (
            HybridActionExecutor,
            Action,
            ActionType,
            ActionResult,
        )

        # Make second action fail
        mock_playwright_executor.execute.side_effect = [
            ActionResult(success=True, action=Action(type=ActionType.GOTO, target="/")),
            ActionResult(
                success=False,
                action=Action(type=ActionType.CLICK, target="#button"),
                error="Element not found",
            ),
            ActionResult(success=True, action=Action(type=ActionType.FILL, target="#input")),
        ]

        executor = HybridActionExecutor(mock_playwright_executor)
        actions = [
            Action(type=ActionType.GOTO, target="/"),
            Action(type=ActionType.CLICK, target="#button"),
            Action(type=ActionType.FILL, target="#input", value="test"),
        ]

        results = await executor.execute_sequence(actions, stop_on_failure=True)

        # Should stop after second action fails
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
