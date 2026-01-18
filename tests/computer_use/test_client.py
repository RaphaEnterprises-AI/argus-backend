"""Tests for the Computer Use client module."""

from unittest.mock import MagicMock, patch

import pytest

# Patch targets
ANTHROPIC_PATCH = "anthropic.Anthropic"


class TestUsageStats:
    """Tests for UsageStats dataclass."""

    def test_usage_stats_creation(self, mock_env_vars):
        """Test UsageStats creation."""
        from src.computer_use.client import UsageStats

        stats = UsageStats()

        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.screenshots_taken == 0
        assert stats.actions_executed == 0
        assert stats.api_calls == 0

    def test_usage_stats_with_values(self, mock_env_vars):
        """Test UsageStats with values."""
        from src.computer_use.client import UsageStats

        stats = UsageStats(
            input_tokens=1000,
            output_tokens=500,
            screenshots_taken=5,
            actions_executed=10,
            api_calls=3,
        )

        assert stats.input_tokens == 1000
        assert stats.output_tokens == 500

    def test_usage_stats_total_cost(self, mock_env_vars):
        """Test UsageStats total_cost property."""
        from src.computer_use.client import UsageStats

        stats = UsageStats(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

        # Based on Sonnet pricing
        cost = stats.total_cost

        assert cost > 0


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_creation(self, mock_env_vars):
        """Test ActionResult creation."""
        from src.computer_use.client import ActionResult

        result = ActionResult(
            action="click",
            success=True,
        )

        assert result.action == "click"
        assert result.success is True
        assert result.screenshot is None
        assert result.error is None
        assert result.metadata == {}

    def test_action_result_with_error(self, mock_env_vars):
        """Test ActionResult with error."""
        from src.computer_use.client import ActionResult

        result = ActionResult(
            action="click",
            success=False,
            error="Element not found",
        )

        assert result.success is False
        assert result.error == "Element not found"

    def test_action_result_with_screenshot(self, mock_env_vars):
        """Test ActionResult with screenshot."""
        from src.computer_use.client import ActionResult

        screenshot_data = b"png data"

        result = ActionResult(
            action="screenshot",
            success=True,
            screenshot=screenshot_data,
        )

        assert result.screenshot == screenshot_data


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_creation(self, mock_env_vars):
        """Test TaskResult creation."""
        from src.computer_use.client import TaskResult, UsageStats

        usage = UsageStats()

        result = TaskResult(
            success=True,
            final_response="Task completed",
            usage=usage,
            actions_taken=[],
            screenshots=[],
        )

        assert result.success is True
        assert result.final_response == "Task completed"
        assert result.error is None
        assert result.iterations == 0

    def test_task_result_with_error(self, mock_env_vars):
        """Test TaskResult with error."""
        from src.computer_use.client import TaskResult, UsageStats

        usage = UsageStats()

        result = TaskResult(
            success=False,
            final_response="",
            usage=usage,
            actions_taken=[],
            screenshots=[],
            error="Max iterations reached",
            iterations=50,
        )

        assert result.success is False
        assert result.error == "Max iterations reached"
        assert result.iterations == 50


class TestComputerUseClient:
    """Tests for ComputerUseClient class."""

    def test_client_creation(self, mock_env_vars):
        """Test ComputerUseClient creation."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            assert client.settings == settings
            assert client.BETA_HEADER == "computer-use-2025-01-24"
            assert client.TOOL_VERSION == "20250124"

    def test_get_tools(self, mock_env_vars):
        """Test getting tool definitions."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            tools = client._get_tools()

            assert len(tools) == 3

            # Check computer tool
            computer_tool = tools[0]
            assert computer_tool["name"] == "computer"
            assert "display_width_px" in computer_tool
            assert "display_height_px" in computer_tool

            # Check bash tool
            bash_tool = tools[1]
            assert bash_tool["name"] == "bash"

            # Check text editor tool
            editor_tool = tools[2]
            assert editor_tool["name"] == "str_replace_based_edit_tool"

    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_env_vars):
        """Test successful task execution."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            # Setup mock response - end_turn on first call
            mock_response = MagicMock()
            mock_response.stop_reason = "end_turn"
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.content = [MagicMock(text="Task completed successfully")]

            mock_anthropic.return_value.beta.messages.create.return_value = (
                mock_response
            )

            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            result = await client.execute_task(
                task="Click the login button",
                screenshot_fn=lambda: b"screenshot",
                action_fn=lambda x: None,
            )

            assert result.success is True
            assert "Task completed" in result.final_response
            assert result.usage.api_calls == 1
            assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_execute_task_with_tool_calls(self, mock_env_vars):
        """Test task execution with tool calls."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            # First call - returns tool use
            tool_response = MagicMock()
            tool_response.stop_reason = "tool_use"
            tool_response.usage.input_tokens = 100
            tool_response.usage.output_tokens = 50

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "computer"
            tool_block.id = "tool-1"
            tool_block.input = {"action": "screenshot"}
            tool_response.content = [tool_block]

            # Second call - end turn
            end_response = MagicMock()
            end_response.stop_reason = "end_turn"
            end_response.usage.input_tokens = 200
            end_response.usage.output_tokens = 100
            end_response.content = [MagicMock(text="Done")]

            mock_anthropic.return_value.beta.messages.create.side_effect = [
                tool_response,
                end_response,
            ]

            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            result = await client.execute_task(
                task="Take a screenshot",
                screenshot_fn=lambda: b"screenshot_data",
                action_fn=lambda x: None,
            )

            assert result.success is True
            assert result.usage.screenshots_taken == 1

    @pytest.mark.asyncio
    async def test_execute_task_max_iterations(self, mock_env_vars):
        """Test task execution hitting max iterations."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            # Always return tool use
            mock_response = MagicMock()
            mock_response.stop_reason = "tool_use"
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "computer"
            tool_block.id = "tool-1"
            tool_block.input = {"action": "screenshot"}
            mock_response.content = [tool_block]

            mock_anthropic.return_value.beta.messages.create.return_value = (
                mock_response
            )

            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            result = await client.execute_task(
                task="Do something",
                screenshot_fn=lambda: b"screenshot",
                action_fn=lambda x: None,
                max_iterations=2,
            )

            assert result.success is False
            assert "Max iterations" in result.error
            assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_execute_task_cost_limit(self, mock_env_vars):
        """Test task execution hitting cost limit."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            # Return tool use with high token usage
            mock_response = MagicMock()
            mock_response.stop_reason = "tool_use"
            mock_response.usage.input_tokens = 10_000_000  # Very high
            mock_response.usage.output_tokens = 5_000_000

            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.name = "computer"
            tool_block.id = "tool-1"
            tool_block.input = {"action": "screenshot"}
            mock_response.content = [tool_block]

            mock_anthropic.return_value.beta.messages.create.return_value = (
                mock_response
            )

            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            settings.cost_limit_per_test = 0.01  # Very low limit
            client = ComputerUseClient(settings)

            result = await client.execute_task(
                task="Do something",
                screenshot_fn=lambda: b"screenshot",
                action_fn=lambda x: None,
            )

            assert result.success is False
            assert "Cost limit" in result.error

    @pytest.mark.asyncio
    async def test_execute_task_api_error(self, mock_env_vars):
        """Test task execution with API error."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            import anthropic

            mock_anthropic.return_value.beta.messages.create.side_effect = (
                anthropic.APIError(
                    message="API Error",
                    request=MagicMock(),
                    body=None,
                )
            )

            from src.computer_use.client import ComputerUseClient
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            result = await client.execute_task(
                task="Do something",
                screenshot_fn=lambda: b"screenshot",
                action_fn=lambda x: None,
            )

            assert result.success is False
            assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_handle_tool_call_screenshot(self, mock_env_vars):
        """Test handling screenshot tool call."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "computer"
            block.input = {"action": "screenshot"}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            result = await client._handle_tool_call(
                block=block,
                screenshot_fn=lambda: b"screenshot_data",
                action_fn=lambda x: None,
                usage=usage,
                actions_taken=actions_taken,
                screenshots=screenshots,
            )

            assert result["type"] == "tool_result"
            assert result["tool_use_id"] == "tool-1"
            assert usage.screenshots_taken == 1
            assert len(screenshots) == 1

    @pytest.mark.asyncio
    async def test_handle_tool_call_click(self, mock_env_vars):
        """Test handling click tool call."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "computer"
            block.input = {"action": "click", "coordinate": [100, 200]}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            action_fn = MagicMock()

            result = await client._handle_tool_call(
                block=block,
                screenshot_fn=lambda: b"screenshot_data",
                action_fn=action_fn,
                usage=usage,
                actions_taken=actions_taken,
                screenshots=screenshots,
            )

            assert result["type"] == "tool_result"
            assert usage.actions_executed == 1
            action_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_tool_call_action_error(self, mock_env_vars):
        """Test handling action error."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "computer"
            block.input = {"action": "click", "coordinate": [100, 200]}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            def failing_action(x):
                raise Exception("Click failed")

            result = await client._handle_tool_call(
                block=block,
                screenshot_fn=lambda: b"screenshot_data",
                action_fn=failing_action,
                usage=usage,
                actions_taken=actions_taken,
                screenshots=screenshots,
            )

            assert result["is_error"] is True
            assert "Click failed" in result["content"]

    @pytest.mark.asyncio
    async def test_handle_tool_call_bash(self, mock_env_vars):
        """Test handling bash tool call."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "bash"
            block.input = {"command": "echo 'hello'"}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="hello\n", stderr=""
                )

                result = await client._handle_tool_call(
                    block=block,
                    screenshot_fn=lambda: b"screenshot",
                    action_fn=lambda x: None,
                    usage=usage,
                    actions_taken=actions_taken,
                    screenshots=screenshots,
                )

            assert result["type"] == "tool_result"
            assert "hello" in result["content"]

    @pytest.mark.asyncio
    async def test_handle_tool_call_bash_timeout(self, mock_env_vars):
        """Test handling bash timeout."""
        with patch(ANTHROPIC_PATCH):
            import subprocess

            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "bash"
            block.input = {"command": "sleep 100"}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep", timeout=30)

                result = await client._handle_tool_call(
                    block=block,
                    screenshot_fn=lambda: b"screenshot",
                    action_fn=lambda x: None,
                    usage=usage,
                    actions_taken=actions_taken,
                    screenshots=screenshots,
                )

            assert result["is_error"] is True
            assert "timed out" in result["content"]

    @pytest.mark.asyncio
    async def test_handle_tool_call_text_editor(self, mock_env_vars):
        """Test handling text editor tool call."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "str_replace_based_edit_tool"
            block.input = {"command": "view", "path": "/test.txt"}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            result = await client._handle_tool_call(
                block=block,
                screenshot_fn=lambda: b"screenshot",
                action_fn=lambda x: None,
                usage=usage,
                actions_taken=actions_taken,
                screenshots=screenshots,
            )

            assert result["type"] == "tool_result"
            assert "completed" in result["content"]

    @pytest.mark.asyncio
    async def test_handle_tool_call_unknown_tool(self, mock_env_vars):
        """Test handling unknown tool call."""
        with patch(ANTHROPIC_PATCH):
            from src.computer_use.client import ComputerUseClient, UsageStats
            from src.config import Settings

            settings = Settings()
            client = ComputerUseClient(settings)

            block = MagicMock()
            block.id = "tool-1"
            block.name = "unknown_tool"
            block.input = {}

            usage = UsageStats()
            actions_taken = []
            screenshots = []

            result = await client._handle_tool_call(
                block=block,
                screenshot_fn=lambda: b"screenshot",
                action_fn=lambda x: None,
                usage=usage,
                actions_taken=actions_taken,
                screenshots=screenshots,
            )

            assert result["is_error"] is True
            assert "Unknown tool" in result["content"]
