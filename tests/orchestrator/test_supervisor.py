"""Tests for the supervisor orchestrator module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage


class TestSupervisorState:
    """Tests for the SupervisorState TypedDict."""

    def test_supervisor_state_structure(self, mock_env_vars):
        """Test SupervisorState has correct structure."""
        from src.orchestrator.supervisor import SupervisorState

        annotations = SupervisorState.__annotations__

        # Core state fields
        assert "messages" in annotations
        assert "next_agent" in annotations
        assert "task_complete" in annotations
        assert "results" in annotations
        assert "current_phase" in annotations
        assert "iteration" in annotations

        # Testing context
        assert "codebase_path" in annotations
        assert "app_url" in annotations
        assert "pr_number" in annotations
        assert "changed_files" in annotations

        # Test state
        assert "codebase_summary" in annotations
        assert "testable_surfaces" in annotations
        assert "test_plan" in annotations
        assert "test_results" in annotations
        assert "failures" in annotations
        assert "healing_queue" in annotations

        # Metrics
        assert "passed_count" in annotations
        assert "failed_count" in annotations
        assert "skipped_count" in annotations
        assert "total_cost" in annotations
        assert "total_input_tokens" in annotations
        assert "total_output_tokens" in annotations

        # Error handling
        assert "error" in annotations


class TestAgentDefinitions:
    """Tests for agent and phase definitions."""

    def test_agents_defined(self, mock_env_vars):
        """Test all agents are defined."""
        from src.orchestrator.supervisor import AGENTS

        expected_agents = [
            "code_analyzer",
            "test_planner",
            "ui_tester",
            "api_tester",
            "self_healer",
            "reporter",
        ]

        for agent in expected_agents:
            assert agent in AGENTS

    def test_agent_descriptions_exist(self, mock_env_vars):
        """Test all agents have descriptions."""
        from src.orchestrator.supervisor import AGENT_DESCRIPTIONS, AGENTS

        for agent in AGENTS:
            assert agent in AGENT_DESCRIPTIONS
            assert len(AGENT_DESCRIPTIONS[agent]) > 0

    def test_phase_descriptions_exist(self, mock_env_vars):
        """Test all phases have descriptions."""
        from src.orchestrator.supervisor import PHASE_DESCRIPTIONS

        expected_phases = ["analysis", "planning", "execution", "healing", "reporting"]

        for phase in expected_phases:
            assert phase in PHASE_DESCRIPTIONS
            assert len(PHASE_DESCRIPTIONS[phase]) > 0


class TestCreateSupervisorPrompt:
    """Tests for the create_supervisor_prompt function."""

    def test_create_supervisor_prompt_contains_agents(self, mock_env_vars):
        """Test supervisor prompt lists all agents."""
        from src.orchestrator.supervisor import AGENTS, create_supervisor_prompt

        prompt = create_supervisor_prompt()

        for agent in AGENTS:
            assert agent in prompt

    def test_create_supervisor_prompt_contains_phases(self, mock_env_vars):
        """Test supervisor prompt lists all phases."""
        from src.orchestrator.supervisor import create_supervisor_prompt

        prompt = create_supervisor_prompt()

        phases = ["analysis", "planning", "execution", "healing", "reporting"]
        for phase in phases:
            assert phase in prompt

    def test_create_supervisor_prompt_mentions_workflow(self, mock_env_vars):
        """Test supervisor prompt mentions workflow guidelines."""
        from src.orchestrator.supervisor import create_supervisor_prompt

        prompt = create_supervisor_prompt()

        assert "Workflow Guidelines" in prompt or "workflow" in prompt.lower()
        assert "FINISH" in prompt

    def test_create_supervisor_prompt_mentions_argus(self, mock_env_vars):
        """Test supervisor prompt mentions Argus."""
        from src.orchestrator.supervisor import create_supervisor_prompt

        prompt = create_supervisor_prompt()

        assert "Argus" in prompt


class TestCreateInitialSupervisorState:
    """Tests for the create_initial_supervisor_state function."""

    def test_create_initial_state_basic(self, mock_env_vars):
        """Test creating initial state with basic params."""
        from src.orchestrator.supervisor import create_initial_supervisor_state

        state = create_initial_supervisor_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        assert state["codebase_path"] == "/path/to/app"
        assert state["app_url"] == "http://localhost:3000"
        assert state["current_phase"] == "analysis"
        assert state["iteration"] == 0
        assert state["task_complete"] is False
        assert state["next_agent"] is None
        assert len(state["messages"]) == 1

    def test_create_initial_state_with_pr(self, mock_env_vars):
        """Test creating initial state with PR info."""
        from src.orchestrator.supervisor import create_initial_supervisor_state

        state = create_initial_supervisor_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            pr_number=123,
            changed_files=["src/login.py", "src/auth.py"],
        )

        assert state["pr_number"] == 123
        assert state["changed_files"] == ["src/login.py", "src/auth.py"]

    def test_create_initial_state_default_message(self, mock_env_vars):
        """Test default initial message is generated."""
        from src.orchestrator.supervisor import create_initial_supervisor_state

        state = create_initial_supervisor_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
        )

        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert "http://localhost:3000" in state["messages"][0].content

    def test_create_initial_state_custom_message(self, mock_env_vars):
        """Test custom initial message is used."""
        from src.orchestrator.supervisor import create_initial_supervisor_state

        state = create_initial_supervisor_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            initial_message="Run only login tests",
        )

        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Run only login tests"

    def test_create_initial_state_with_changed_files_in_message(self, mock_env_vars):
        """Test changed files are mentioned in initial message."""
        from src.orchestrator.supervisor import create_initial_supervisor_state

        state = create_initial_supervisor_state(
            codebase_path="/path/to/app",
            app_url="http://localhost:3000",
            changed_files=["auth.py", "login.py"],
        )

        assert "auth.py" in state["messages"][0].content or "changes" in state["messages"][0].content.lower()

    def test_create_initial_state_default_values(self, mock_env_vars):
        """Test initial state has correct default values."""
        from src.orchestrator.supervisor import create_initial_supervisor_state

        state = create_initial_supervisor_state(
            codebase_path="/app",
            app_url="http://localhost",
        )

        assert state["results"] == {}
        assert state["codebase_summary"] is None
        assert state["testable_surfaces"] is None
        assert state["test_plan"] is None
        assert state["test_results"] == []
        assert state["failures"] == []
        assert state["healing_queue"] == []
        assert state["passed_count"] == 0
        assert state["failed_count"] == 0
        assert state["skipped_count"] == 0
        assert state["total_cost"] == 0.0
        assert state["total_input_tokens"] == 0
        assert state["total_output_tokens"] == 0
        assert state["error"] is None


class TestSupervisorNode:
    """Tests for the supervisor_node function."""

    @pytest.mark.asyncio
    async def test_supervisor_node_max_iterations(self, mock_env_vars):
        """Test supervisor_node handles max iterations."""
        from src.orchestrator.supervisor import supervisor_node

        state = {
            "iteration": 50,
            "messages": [],
            "current_phase": "execution",
            "codebase_path": "/app",
            "app_url": "http://localhost",
        }

        result = await supervisor_node(state, {})

        assert result["next_agent"] == "reporter"
        assert result["current_phase"] == "reporting"
        assert result["iteration"] == 51

    @pytest.mark.asyncio
    async def test_supervisor_node_error_detected(self, mock_env_vars):
        """Test supervisor_node handles errors."""
        from src.orchestrator.supervisor import supervisor_node

        state = {
            "iteration": 5,
            "messages": [],
            "current_phase": "execution",
            "codebase_path": "/app",
            "app_url": "http://localhost",
            "error": "Something went wrong",
        }

        result = await supervisor_node(state, {})

        assert result["next_agent"] == "reporter"
        assert result["current_phase"] == "reporting"
        assert "error" in result["messages"][0].content.lower()

    @pytest.mark.asyncio
    async def test_supervisor_node_routes_code_analyzer(self, mock_env_vars):
        """Test supervisor_node routes to code_analyzer."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "The codebase needs analysis. I will use CODE_ANALYZER."

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 0,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "analysis",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "testable_surfaces": None,
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "code_analyzer"
        assert result["current_phase"] == "analysis"

    @pytest.mark.asyncio
    async def test_supervisor_node_routes_test_planner(self, mock_env_vars):
        """Test supervisor_node routes to test_planner."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Analysis complete. Now I will use TEST_PLANNER to create a plan."

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 1,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "planning",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "codebase_summary": "A web application with login and checkout features",
                    "testable_surfaces": [{"name": "login", "type": "ui"}],
                    "test_plan": None,
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "test_planner"
        assert result["current_phase"] == "planning"

    @pytest.mark.asyncio
    async def test_supervisor_node_routes_ui_tester(self, mock_env_vars):
        """Test supervisor_node routes to ui_tester."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Plan created. Now executing UI tests with UI_TESTER."

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 2,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "execution",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "test_plan": [{"id": "test-1", "type": "ui"}],
                    "test_results": [],
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "ui_tester"
        assert result["current_phase"] == "execution"

    @pytest.mark.asyncio
    async def test_supervisor_node_routes_self_healer(self, mock_env_vars):
        """Test supervisor_node routes to self_healer."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Tests failed. Using SELF_HEALER to fix the issues."

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 3,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "execution",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "failures": [{"test_id": "test-1", "error": "Element not found"}],
                    "healing_queue": ["test-1"],
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "self_healer"
        assert result["current_phase"] == "healing"

    @pytest.mark.asyncio
    async def test_supervisor_node_routes_reporter(self, mock_env_vars):
        """Test supervisor_node routes to reporter."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "All tests complete. Using REPORTER to generate final report."

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 5,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "execution",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "test_results": [{"id": "test-1", "status": "passed"}],
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "reporter"
        assert result["current_phase"] == "reporting"

    @pytest.mark.asyncio
    async def test_supervisor_node_finish(self, mock_env_vars):
        """Test supervisor_node handles FINISH decision."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "All testing is complete. FINISH"

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 10,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "reporting",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] is None
        assert result["task_complete"] is True
        assert result["current_phase"] == "complete"

    @pytest.mark.asyncio
    async def test_supervisor_node_fallback_logic(self, mock_env_vars):
        """Test supervisor_node uses fallback logic when LLM response is unclear."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "I'm not sure what to do next."  # No agent mentioned

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 0,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "analysis",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "testable_surfaces": None,
                }

                result = await supervisor_node(state, {})

        # Fallback should route to code_analyzer for analysis phase
        assert result["next_agent"] == "code_analyzer"

    @pytest.mark.asyncio
    async def test_supervisor_node_handles_exception(self, mock_env_vars):
        """Test supervisor_node handles exceptions gracefully."""
        from src.orchestrator.supervisor import supervisor_node

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 0,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "analysis",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "reporter"
        assert result["current_phase"] == "reporting"
        assert result["error"] == "LLM error"

    @pytest.mark.asyncio
    async def test_supervisor_node_builds_context(self, mock_env_vars):
        """Test supervisor_node builds correct context for LLM."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Using REPORTER to finish."

        captured_messages = None

        async def capture_messages(messages):
            nonlocal captured_messages
            captured_messages = messages
            return mock_response

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = capture_messages
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 5,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "execution",
                    "codebase_path": "/test/app",
                    "app_url": "http://localhost:3000",
                    "codebase_summary": "A test application",
                    "testable_surfaces": [{"name": "login"}],
                    "test_plan": [{"id": "t1"}],
                    "test_results": [{"id": "t1", "status": "passed"}],
                    "passed_count": 1,
                    "failed_count": 0,
                    "total_cost": 0.05,
                }

                await supervisor_node(state, {})

        assert captured_messages is not None
        # Context message should be in the messages
        context_found = False
        for msg in captured_messages:
            if hasattr(msg, 'content') and "Current State" in str(msg.content):
                context_found = True
                assert "Iteration: 5" in msg.content
                assert "/test/app" in msg.content
                break
        assert context_found


class TestRouteToAgent:
    """Tests for the route_to_agent function."""

    def test_route_to_agent_task_complete(self, mock_env_vars):
        """Test routing when task is complete."""
        from src.orchestrator.supervisor import route_to_agent

        state = {
            "task_complete": True,
            "next_agent": "reporter",
        }

        result = route_to_agent(state)
        assert result == "end"

    def test_route_to_agent_next_agent_none(self, mock_env_vars):
        """Test routing when next_agent is None."""
        from src.orchestrator.supervisor import route_to_agent

        state = {
            "task_complete": False,
            "next_agent": None,
        }

        result = route_to_agent(state)
        assert result == "end"

    def test_route_to_agent_code_analyzer(self, mock_env_vars):
        """Test routing to code_analyzer."""
        from src.orchestrator.supervisor import route_to_agent

        state = {
            "task_complete": False,
            "next_agent": "code_analyzer",
        }

        result = route_to_agent(state)
        assert result == "code_analyzer"

    def test_route_to_agent_ui_tester(self, mock_env_vars):
        """Test routing to ui_tester."""
        from src.orchestrator.supervisor import route_to_agent

        state = {
            "task_complete": False,
            "next_agent": "ui_tester",
        }

        result = route_to_agent(state)
        assert result == "ui_tester"

    def test_route_to_agent_self_healer(self, mock_env_vars):
        """Test routing to self_healer."""
        from src.orchestrator.supervisor import route_to_agent

        state = {
            "task_complete": False,
            "next_agent": "self_healer",
        }

        result = route_to_agent(state)
        assert result == "self_healer"


class TestSupervisorWrapperNodes:
    """Tests for the supervisor wrapper nodes."""

    @pytest.mark.asyncio
    async def test_supervisor_code_analyzer_node(self, mock_env_vars):
        """Test supervisor_code_analyzer_node wrapper."""
        from src.orchestrator.supervisor import supervisor_code_analyzer_node

        state = {
            "codebase_path": "/app",
            "app_url": "http://localhost",
            "pr_number": None,
            "changed_files": [],
            "results": {},
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

        mock_result = {
            "codebase_summary": "A web application",
            "testable_surfaces": [{"name": "login", "type": "ui"}],
            "total_cost": 0.01,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
        }

        with patch("src.orchestrator.nodes.analyze_code_node", AsyncMock(return_value=mock_result)):
            with patch("src.orchestrator.state.create_initial_state") as mock_create_state:
                mock_create_state.return_value = {}

                result = await supervisor_code_analyzer_node(state)

        assert "codebase_summary" in result
        assert "testable_surfaces" in result
        assert result["current_phase"] == "planning"

    @pytest.mark.asyncio
    async def test_supervisor_test_planner_node(self, mock_env_vars):
        """Test supervisor_test_planner_node wrapper."""
        from src.orchestrator.supervisor import supervisor_test_planner_node

        state = {
            "codebase_path": "/app",
            "app_url": "http://localhost",
            "codebase_summary": "A web app",
            "testable_surfaces": [{"name": "login"}],
            "changed_files": [],
            "results": {},
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

        mock_result = {
            "test_plan": [
                {"id": "test-1", "name": "Login Test", "type": "ui"},
                {"id": "test-2", "name": "API Test", "type": "api"},
            ],
            "total_cost": 0.02,
            "total_input_tokens": 200,
            "total_output_tokens": 100,
        }

        with patch("src.orchestrator.nodes.plan_tests_node", AsyncMock(return_value=mock_result)):
            with patch("src.orchestrator.state.create_initial_state") as mock_create_state:
                mock_create_state.return_value = {}

                result = await supervisor_test_planner_node(state)

        assert "test_plan" in result
        assert len(result["test_plan"]) == 2
        assert result["current_phase"] == "execution"

    @pytest.mark.asyncio
    async def test_supervisor_execute_test_node(self, mock_env_vars):
        """Test supervisor_execute_test_node wrapper."""
        from src.orchestrator.supervisor import supervisor_execute_test_node

        state = {
            "codebase_path": "/app",
            "app_url": "http://localhost",
            "test_plan": [
                {"id": "test-1", "name": "Test 1"},
                {"id": "test-2", "name": "Test 2"},
            ],
            "test_results": [],
            "failures": [],
            "healing_queue": [],
            "passed_count": 0,
            "failed_count": 0,
            "results": {},
            "total_cost": 0.0,
        }

        mock_result = {
            "test_results": [{"test_id": "test-1", "status": "passed"}],
            "failures": [],
            "healing_queue": [],
            "total_cost": 0.01,
        }

        with patch("src.orchestrator.nodes.execute_test_node", AsyncMock(return_value=mock_result)):
            with patch("src.orchestrator.state.create_initial_state") as mock_create_state:
                mock_create_state.return_value = {}

                result = await supervisor_execute_test_node(state)

        assert "test_results" in result
        assert "passed_count" in result

    @pytest.mark.asyncio
    async def test_supervisor_self_healer_node(self, mock_env_vars):
        """Test supervisor_self_healer_node wrapper."""
        from src.orchestrator.supervisor import supervisor_self_healer_node

        state = {
            "codebase_path": "/app",
            "app_url": "http://localhost",
            "test_plan": [{"id": "test-1"}],
            "test_results": [{"test_id": "test-1", "status": "failed"}],
            "failures": [{"test_id": "test-1", "error": "Element not found"}],
            "healing_queue": ["test-1"],
            "results": {},
            "total_cost": 0.0,
        }

        mock_result = {
            "test_plan": [{"id": "test-1", "healed": True}],
            "failures": [],
            "healing_queue": [],
            "total_cost": 0.01,
        }

        with patch("src.orchestrator.nodes.self_heal_node", AsyncMock(return_value=mock_result)):
            with patch("src.orchestrator.state.create_initial_state") as mock_create_state:
                mock_create_state.return_value = {}

                result = await supervisor_self_healer_node(state)

        assert "healing_queue" in result
        assert len(result["healing_queue"]) == 0  # All healed

    @pytest.mark.asyncio
    async def test_supervisor_reporter_node(self, mock_env_vars):
        """Test supervisor_reporter_node wrapper."""
        from src.orchestrator.supervisor import supervisor_reporter_node

        state = {
            "codebase_path": "/app",
            "app_url": "http://localhost",
            "pr_number": 123,
            "test_results": [{"test_id": "test-1", "status": "passed"}],
            "failures": [],
            "passed_count": 1,
            "failed_count": 0,
            "skipped_count": 0,
            "total_cost": 0.05,
            "iteration": 5,
            "results": {},
        }

        mock_result = {
            "executive_summary": "All tests passed!",
            "report_paths": {"html": "/reports/test.html"},
            "total_cost": 0.01,
        }

        with patch("src.orchestrator.nodes.report_node", AsyncMock(return_value=mock_result)):
            with patch("src.orchestrator.state.create_initial_state") as mock_create_state:
                mock_create_state.return_value = {}

                result = await supervisor_reporter_node(state)

        assert result["task_complete"] is True
        assert result["current_phase"] == "complete"
        assert "results" in result


class TestCreateSupervisorGraph:
    """Tests for the create_supervisor_graph function."""

    def test_create_supervisor_graph_returns_state_graph(self, mock_env_vars):
        """Test create_supervisor_graph returns a StateGraph."""
        from langgraph.graph import StateGraph

        from src.orchestrator.supervisor import create_supervisor_graph

        graph = create_supervisor_graph()

        assert isinstance(graph, StateGraph)

    def test_create_supervisor_graph_has_all_nodes(self, mock_env_vars):
        """Test create_supervisor_graph has all expected nodes."""
        from src.orchestrator.supervisor import AGENTS, create_supervisor_graph

        graph = create_supervisor_graph()

        # Check supervisor node
        assert "supervisor" in graph.nodes

        # Check all agent nodes
        for agent in AGENTS:
            assert agent in graph.nodes

    def test_create_supervisor_graph_compiles(self, mock_env_vars):
        """Test create_supervisor_graph compiles successfully."""
        from langgraph.checkpoint.memory import MemorySaver

        from src.orchestrator.supervisor import create_supervisor_graph

        graph = create_supervisor_graph()
        checkpointer = MemorySaver()

        compiled = graph.compile(checkpointer=checkpointer)

        assert compiled is not None


class TestSupervisorOrchestrator:
    """Tests for the SupervisorOrchestrator class."""

    def test_orchestrator_init(self, mock_env_vars):
        """Test SupervisorOrchestrator initialization."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            assert orchestrator.codebase_path == "/app"
            assert orchestrator.app_url == "http://localhost:3000"
            assert orchestrator.pr_number is None
            assert orchestrator.changed_files == []

    def test_orchestrator_init_with_pr(self, mock_env_vars):
        """Test SupervisorOrchestrator initialization with PR."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
                pr_number=123,
                changed_files=["file1.py", "file2.py"],
            )

            assert orchestrator.pr_number == 123
            assert orchestrator.changed_files == ["file1.py", "file2.py"]

    @pytest.mark.asyncio
    async def test_orchestrator_run(self, mock_env_vars):
        """Test SupervisorOrchestrator.run method."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            mock_final_state = {
                "task_complete": True,
                "passed_count": 5,
                "failed_count": 1,
                "iteration": 10,
                "total_cost": 0.5,
            }

            orchestrator.app.ainvoke = AsyncMock(return_value=mock_final_state)

            result = await orchestrator.run()

            assert result["task_complete"] is True
            assert result["passed_count"] == 5

    @pytest.mark.asyncio
    async def test_orchestrator_run_with_thread_id(self, mock_env_vars):
        """Test SupervisorOrchestrator.run with custom thread_id."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            mock_final_state = {"task_complete": True}
            orchestrator.app.ainvoke = AsyncMock(return_value=mock_final_state)

            await orchestrator.run(thread_id="custom-thread-123")

            # Verify ainvoke was called with correct config
            call_args = orchestrator.app.ainvoke.call_args
            config = call_args[0][1]
            assert config["configurable"]["thread_id"] == "custom-thread-123"

    @pytest.mark.asyncio
    async def test_orchestrator_get_state(self, mock_env_vars):
        """Test SupervisorOrchestrator.get_state method."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            mock_state_snapshot = MagicMock()
            mock_state_snapshot.values = {
                "task_complete": False,
                "current_phase": "execution",
                "iteration": 5,
            }
            mock_state_snapshot.next = ["execute_test"]

            orchestrator.app.aget_state = AsyncMock(return_value=mock_state_snapshot)

            result = await orchestrator.get_state("thread-123")

            assert result is not None
            assert result["is_complete"] is False
            assert result["current_phase"] == "execution"
            assert result["iteration"] == 5

    @pytest.mark.asyncio
    async def test_orchestrator_get_state_not_found(self, mock_env_vars):
        """Test SupervisorOrchestrator.get_state when state not found."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            orchestrator.app.aget_state = AsyncMock(return_value=None)

            result = await orchestrator.get_state("nonexistent-thread")

            assert result is None

    @pytest.mark.asyncio
    async def test_orchestrator_resume(self, mock_env_vars):
        """Test SupervisorOrchestrator.resume method."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            mock_final_state = {"task_complete": True, "passed_count": 10}
            orchestrator.app.ainvoke = AsyncMock(return_value=mock_final_state)

            result = await orchestrator.resume("thread-123")

            assert result["task_complete"] is True
            # Verify ainvoke was called with None as first arg (resume from checkpoint)
            call_args = orchestrator.app.ainvoke.call_args
            assert call_args[0][0] is None

    def test_orchestrator_get_summary(self, mock_env_vars):
        """Test SupervisorOrchestrator.get_summary method."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            state = {
                "thread_id": "test-thread",
                "current_phase": "complete",
                "task_complete": True,
                "iteration": 15,
                "test_plan": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}],
                "test_results": [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}],
                "passed_count": 2,
                "failed_count": 1,
                "total_cost": 0.75,
                "results": {"code_analysis": {"summary": "done"}},
                "error": None,
            }

            summary = orchestrator.get_summary(state)

            assert summary["current_phase"] == "complete"
            assert summary["task_complete"] is True
            assert summary["iterations"] == 15
            assert summary["tests"]["total"] == 3
            assert summary["tests"]["executed"] == 3
            assert summary["tests"]["passed"] == 2
            assert summary["tests"]["failed"] == 1
            assert summary["cost"] == 0.75
            assert summary["error"] is None

    def test_orchestrator_get_summary_with_error(self, mock_env_vars):
        """Test SupervisorOrchestrator.get_summary with error state."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            state = {
                "current_phase": "reporting",
                "task_complete": False,
                "iteration": 5,
                "test_plan": [],
                "test_results": [],
                "passed_count": 0,
                "failed_count": 0,
                "total_cost": 0.1,
                "results": {},
                "error": "Connection timeout",
            }

            summary = orchestrator.get_summary(state)

            assert summary["error"] == "Connection timeout"
            assert summary["task_complete"] is False


class TestSupervisorNodeFallbackLogicEdgeCases:
    """Additional edge case tests for supervisor_node fallback logic."""

    @pytest.mark.asyncio
    async def test_supervisor_node_fallback_api_tester(self, mock_env_vars):
        """Test fallback logic routes to api_tester when test plan has API tests."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "I'll continue with testing."  # No agent mentioned

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 2,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "execution",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "test_plan": [
                        {"id": "test-1", "type": "api"},
                        {"id": "test-2", "type": "ui"},
                    ],
                    "test_results": [],
                }

                result = await supervisor_node(state, {})

        # Should route to api_tester since there's an API test
        assert result["next_agent"] == "api_tester"
        assert result["current_phase"] == "execution"

    @pytest.mark.asyncio
    async def test_supervisor_node_fallback_ui_tester_no_api(self, mock_env_vars):
        """Test fallback logic routes to ui_tester when no API tests."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Running tests now."  # No agent mentioned

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 2,
                    "messages": [HumanMessage(content="Run tests")],
                    "current_phase": "execution",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "test_plan": [
                        {"id": "test-1", "type": "ui"},
                        {"id": "test-2", "type": "ui"},
                    ],
                    "test_results": [],
                }

                result = await supervisor_node(state, {})

        # Should route to ui_tester
        assert result["next_agent"] == "ui_tester"

    @pytest.mark.asyncio
    async def test_supervisor_node_fallback_planning_no_test_plan(self, mock_env_vars):
        """Test fallback logic routes to test_planner when planning phase with no plan."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Proceeding to next step."  # No agent mentioned

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 1,
                    "messages": [HumanMessage(content="Create test plan")],
                    "current_phase": "planning",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "testable_surfaces": [{"name": "login"}],
                    "test_plan": None,  # No test plan yet
                }

                result = await supervisor_node(state, {})

        # Should route to test_planner
        assert result["next_agent"] == "test_planner"
        assert result["current_phase"] == "planning"

    @pytest.mark.asyncio
    async def test_supervisor_node_fallback_healing(self, mock_env_vars):
        """Test fallback logic routes to self_healer when failures and healing queue exist."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Let me handle this."  # No agent mentioned

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 5,
                    "messages": [HumanMessage(content="Fix tests")],
                    "current_phase": "execution",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "test_plan": [],  # No more tests in plan
                    "failures": [{"test_id": "test-1", "error": "Failed"}],
                    "healing_queue": ["test-1"],
                }

                result = await supervisor_node(state, {})

        # Should route to self_healer
        assert result["next_agent"] == "self_healer"
        assert result["current_phase"] == "healing"

    @pytest.mark.asyncio
    async def test_supervisor_node_fallback_reporter_default(self, mock_env_vars):
        """Test fallback logic routes to reporter by default."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "What's next?"  # No agent mentioned

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            mock_api_key = MagicMock()
            mock_api_key.get_secret_value.return_value = "sk-ant-test-key"
            mock_settings.return_value.anthropic_api_key = mock_api_key

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 10,
                    "messages": [HumanMessage(content="Done")],
                    "current_phase": "complete",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                    "testable_surfaces": [{"name": "login"}],
                    "test_plan": [],
                    "failures": [],
                    "healing_queue": [],
                }

                result = await supervisor_node(state, {})

        # Should route to reporter as default fallback
        assert result["next_agent"] == "reporter"
        assert result["current_phase"] == "reporting"

    @pytest.mark.asyncio
    async def test_supervisor_node_api_key_without_secret_value(self, mock_env_vars):
        """Test supervisor_node handles plain string API key."""
        from src.orchestrator.supervisor import supervisor_node

        mock_response = MagicMock()
        mock_response.content = "Using REPORTER now."

        with patch("src.orchestrator.supervisor.get_settings") as mock_settings:
            # API key is a plain string, not SecretStr
            mock_settings.return_value.anthropic_api_key = "sk-ant-plain-key"

            with patch("langchain_anthropic.ChatAnthropic") as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm_class.return_value = mock_llm

                state = {
                    "iteration": 0,
                    "messages": [HumanMessage(content="Test")],
                    "current_phase": "reporting",
                    "codebase_path": "/app",
                    "app_url": "http://localhost",
                }

                result = await supervisor_node(state, {})

        assert result["next_agent"] == "reporter"


class TestSupervisorOrchestratorExceptionHandling:
    """Tests for SupervisorOrchestrator exception handling."""

    @pytest.mark.asyncio
    async def test_orchestrator_run_exception(self, mock_env_vars):
        """Test SupervisorOrchestrator.run handles exceptions."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            orchestrator.app.ainvoke = AsyncMock(side_effect=Exception("Test error"))

            with pytest.raises(Exception, match="Test error"):
                await orchestrator.run()

    @pytest.mark.asyncio
    async def test_orchestrator_get_state_exception(self, mock_env_vars):
        """Test SupervisorOrchestrator.get_state handles exceptions."""
        with patch("src.orchestrator.checkpointer.get_checkpointer") as mock_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            mock_checkpointer.return_value = MemorySaver()

            from src.orchestrator.supervisor import SupervisorOrchestrator

            orchestrator = SupervisorOrchestrator(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            orchestrator.app.aget_state = AsyncMock(side_effect=Exception("State error"))

            result = await orchestrator.get_state("thread-123")

            assert result is None
