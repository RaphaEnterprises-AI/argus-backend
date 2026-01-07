"""Tests for the test planner module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# Patch target for Anthropic client in base agent
ANTHROPIC_PATCH = 'anthropic.Anthropic'


class TestTestStep:
    """Tests for TestStep dataclass."""

    def test_step_creation(self, mock_env_vars):
        """Test TestStep creation."""
        from src.agents.test_planner import TestStep

        step = TestStep(
            action="click",
            target="#submit-btn",
        )

        assert step.action == "click"
        assert step.target == "#submit-btn"
        assert step.timeout == 5000  # default

    def test_step_with_all_fields(self, mock_env_vars):
        """Test TestStep with all fields."""
        from src.agents.test_planner import TestStep

        step = TestStep(
            action="fill",
            target="#email",
            value="test@example.com",
            timeout=10000,
            description="Enter email address",
        )

        assert step.value == "test@example.com"
        assert step.timeout == 10000
        assert step.description is not None

    def test_step_to_dict(self, mock_env_vars):
        """Test TestStep to_dict method."""
        from src.agents.test_planner import TestStep

        step = TestStep(
            action="goto",
            target="/login",
        )

        result = step.to_dict()

        assert result["action"] == "goto"
        assert result["target"] == "/login"
        assert result["timeout"] == 5000


class TestTestAssertion:
    """Tests for TestAssertion dataclass."""

    def test_assertion_creation(self, mock_env_vars):
        """Test TestAssertion creation."""
        from src.agents.test_planner import TestAssertion

        assertion = TestAssertion(
            type="url_matches",
            expected="/dashboard",
        )

        assert assertion.type == "url_matches"
        assert assertion.expected == "/dashboard"

    def test_assertion_with_target(self, mock_env_vars):
        """Test TestAssertion with target."""
        from src.agents.test_planner import TestAssertion

        assertion = TestAssertion(
            type="element_visible",
            target=".welcome-message",
            description="Welcome message should be visible",
        )

        assert assertion.target == ".welcome-message"

    def test_assertion_to_dict(self, mock_env_vars):
        """Test TestAssertion to_dict method."""
        from src.agents.test_planner import TestAssertion

        assertion = TestAssertion(
            type="text_contains",
            target="#heading",
            expected="Welcome",
        )

        result = assertion.to_dict()

        assert result["type"] == "text_contains"
        assert result["target"] == "#heading"
        assert result["expected"] == "Welcome"


class TestTestSpec:
    """Tests for TestSpec dataclass."""

    def test_spec_creation(self, mock_env_vars):
        """Test TestSpec creation."""
        from src.agents.test_planner import TestSpec

        spec = TestSpec(
            id="test-001",
            name="Login Test",
            type="ui",
            priority="critical",
            description="Test user login flow",
        )

        assert spec.id == "test-001"
        assert spec.priority == "critical"
        assert spec.steps == []
        assert spec.assertions == []

    def test_spec_with_steps(self, mock_env_vars):
        """Test TestSpec with steps."""
        from src.agents.test_planner import TestSpec, TestStep, TestAssertion

        steps = [
            TestStep(action="goto", target="/login"),
            TestStep(action="fill", target="#email", value="test@test.com"),
            TestStep(action="click", target="#submit"),
        ]

        assertions = [
            TestAssertion(type="url_matches", expected="/dashboard"),
        ]

        spec = TestSpec(
            id="test-002",
            name="Login Flow",
            type="ui",
            priority="high",
            description="Complete login flow",
            preconditions=["User must be logged out"],
            steps=steps,
            assertions=assertions,
            cleanup=["Logout user"],
            tags=["auth", "smoke"],
            estimated_duration_ms=15000,
        )

        assert len(spec.steps) == 3
        assert len(spec.assertions) == 1
        assert len(spec.preconditions) == 1
        assert spec.estimated_duration_ms == 15000

    def test_spec_to_dict(self, mock_env_vars):
        """Test TestSpec to_dict method."""
        from src.agents.test_planner import TestSpec, TestStep

        spec = TestSpec(
            id="test-003",
            name="API Test",
            type="api",
            priority="medium",
            description="API test",
            steps=[TestStep(action="request", target="/api/users")],
            metadata={"method": "GET"},
        )

        result = spec.to_dict()

        assert result["id"] == "test-003"
        assert result["type"] == "api"
        assert len(result["steps"]) == 1
        assert result["metadata"]["method"] == "GET"


class TestTestPlan:
    """Tests for TestPlan dataclass."""

    def test_plan_creation(self, mock_env_vars):
        """Test TestPlan creation."""
        from src.agents.test_planner import TestPlan, TestSpec

        tests = [
            TestSpec(
                id="t1",
                name="Test 1",
                type="ui",
                priority="high",
                description="Desc",
                estimated_duration_ms=10000,
            ),
            TestSpec(
                id="t2",
                name="Test 2",
                type="ui",
                priority="medium",
                description="Desc",
                estimated_duration_ms=15000,
            ),
        ]

        plan = TestPlan(tests=tests)

        assert len(plan.tests) == 2
        assert plan.total_estimated_duration_ms == 25000

    def test_plan_with_explicit_duration(self, mock_env_vars):
        """Test TestPlan with explicit duration."""
        from src.agents.test_planner import TestPlan, TestSpec

        tests = [
            TestSpec(id="t1", name="Test", type="ui", priority="high", description="D"),
        ]

        plan = TestPlan(
            tests=tests,
            total_estimated_duration_ms=60000,  # Override
        )

        assert plan.total_estimated_duration_ms == 60000


class TestTestPlannerAgent:
    """Tests for TestPlannerAgent class."""

    def test_agent_creation(self, mock_env_vars):
        """Test TestPlannerAgent creation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            assert agent is not None

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()
            prompt = agent._get_system_prompt()

            # Enhanced prompt uses different terminology
            assert "test" in prompt.lower()
            assert "planning" in prompt.lower() or "strategy" in prompt.lower()
            assert "JSON" in prompt or "json" in prompt

    def test_build_planning_prompt(self, mock_env_vars):
        """Test planning prompt building."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            surfaces = [
                {"type": "ui", "name": "Login", "path": "/login"},
            ]

            prompt = agent._build_planning_prompt(
                surfaces=surfaces,
                app_url="http://localhost:3000",
                codebase_summary="",
                changed_files=None,
                max_tests_per_surface=3,
            )

            assert "http://localhost:3000" in prompt
            assert "/login" in prompt
            assert "3 tests per surface" in prompt

    def test_build_planning_prompt_with_summary(self, mock_env_vars):
        """Test planning prompt with codebase summary."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            prompt = agent._build_planning_prompt(
                surfaces=[],
                app_url="http://localhost",
                codebase_summary="React application with Next.js",
                changed_files=None,
                max_tests_per_surface=2,
            )

            assert "CODEBASE SUMMARY" in prompt
            assert "React application" in prompt

    def test_build_planning_prompt_with_changed_files(self, mock_env_vars):
        """Test planning prompt with changed files."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            prompt = agent._build_planning_prompt(
                surfaces=[],
                app_url="http://localhost",
                codebase_summary="",
                changed_files=["src/login.tsx", "src/auth.ts"],
                max_tests_per_surface=2,
            )

            assert "CHANGED FILES" in prompt
            assert "src/login.tsx" in prompt

    def test_parse_test_spec(self, mock_env_vars):
        """Test parsing test spec from JSON data."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            data = {
                "id": "test-001",
                "name": "Login Test",
                "type": "ui",
                "priority": "critical",
                "description": "Test login",
                "preconditions": ["User logged out"],
                "steps": [
                    {"action": "goto", "target": "/login"},
                    {"action": "fill", "target": "#email", "value": "test@test.com"},
                ],
                "assertions": [
                    {"type": "url_matches", "expected": "/dashboard"},
                ],
                "cleanup": ["Logout"],
                "tags": ["auth"],
                "estimated_duration_ms": 20000,
            }

            spec = agent._parse_test_spec(data)

            assert spec.id == "test-001"
            assert spec.name == "Login Test"
            assert len(spec.steps) == 2
            assert len(spec.assertions) == 1
            assert spec.estimated_duration_ms == 20000

    def test_parse_test_spec_defaults(self, mock_env_vars):
        """Test parsing test spec with defaults."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            data = {}  # Empty data

            spec = agent._parse_test_spec(data)

            assert "test-" in spec.id
            assert spec.name == "Unnamed Test"
            assert spec.type == "ui"
            assert spec.priority == "medium"

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars):
        """Test successful test plan execution."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 500
            mock_response.content = [MagicMock(text='''
            {
                "tests": [
                    {
                        "id": "test-001",
                        "name": "Login Test",
                        "type": "ui",
                        "priority": "critical",
                        "description": "Test login",
                        "steps": [
                            {"action": "goto", "target": "/login"}
                        ],
                        "assertions": [
                            {"type": "url_matches", "expected": "/dashboard"}
                        ]
                    }
                ],
                "coverage_summary": {
                    "total_surfaces": 1,
                    "covered_surfaces": 1
                }
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            surfaces = [
                {"type": "ui", "name": "Login", "path": "/login", "priority": "critical"},
            ]

            result = await agent.execute(
                testable_surfaces=surfaces,
                app_url="http://localhost:3000",
            )

            assert result.success is True
            assert len(result.data.tests) == 1
            assert result.data.tests[0].name == "Login Test"

    @pytest.mark.asyncio
    async def test_execute_with_testable_surface_objects(self, mock_env_vars):
        """Test execution with TestableSurface objects."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [MagicMock(text='{"tests": []}')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_planner import TestPlannerAgent
            from src.agents.code_analyzer import TestableSurface

            agent = TestPlannerAgent()

            surfaces = [
                TestableSurface(
                    type="ui",
                    name="Dashboard",
                    path="/dashboard",
                    priority="high",
                    description="Main dashboard",
                    test_scenarios=["Load data"],
                ),
            ]

            result = await agent.execute(
                testable_surfaces=surfaces,
                app_url="http://localhost",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_cost_limit_exceeded(self, mock_env_vars):
        """Test execution when cost limit exceeded."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()
            agent._usage.total_cost = 100.0  # Exceed default limit (10.0)

            result = await agent.execute(
                testable_surfaces=[],
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Cost limit" in result.error

    @pytest.mark.asyncio
    async def test_execute_parse_failure(self, mock_env_vars):
        """Test execution with parse failure."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.content = [MagicMock(text="Invalid JSON")]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            result = await agent.execute(
                testable_surfaces=[],
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Failed to parse" in result.error

    @pytest.mark.asyncio
    async def test_execute_exception(self, mock_env_vars):
        """Test execution with exception."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            result = await agent.execute(
                testable_surfaces=[],
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Planning failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_list_response(self, mock_env_vars):
        """Test execution when response is a list instead of object."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            # Response is a list of tests directly
            mock_response.content = [MagicMock(text='''
            [
                {"id": "t1", "name": "Test 1", "type": "ui", "priority": "high", "description": "D"}
            ]
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            result = await agent.execute(
                testable_surfaces=[],
                app_url="http://localhost",
            )

            assert result.success is True
            assert len(result.data.tests) == 1

    @pytest.mark.asyncio
    async def test_execute_sorts_by_priority(self, mock_env_vars):
        """Test that results are sorted by priority."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [MagicMock(text='''
            {
                "tests": [
                    {"id": "t1", "name": "Low Priority", "priority": "low"},
                    {"id": "t2", "name": "Critical", "priority": "critical"},
                    {"id": "t3", "name": "Medium", "priority": "medium"},
                    {"id": "t4", "name": "High", "priority": "high"}
                ]
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            result = await agent.execute(
                testable_surfaces=[],
                app_url="http://localhost",
            )

            assert result.success is True
            assert result.data.tests[0].priority == "critical"
            assert result.data.tests[1].priority == "high"
            assert result.data.tests[2].priority == "medium"
            assert result.data.tests[3].priority == "low"

    @pytest.mark.asyncio
    async def test_generate_api_tests(self, mock_env_vars):
        """Test API test generation."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 300
            mock_response.content = [MagicMock(text='''
            {
                "tests": [
                    {
                        "id": "api-001",
                        "name": "Create User API",
                        "type": "api",
                        "priority": "high",
                        "description": "Test user creation",
                        "steps": [
                            {"action": "request", "target": "/api/users", "value": "POST"}
                        ]
                    }
                ]
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_planner import TestPlannerAgent

            agent = TestPlannerAgent()

            endpoints = [
                {
                    "method": "POST",
                    "path": "/api/users",
                    "description": "Create a new user",
                    "priority": "high",
                },
            ]

            result = await agent.generate_api_tests(
                endpoints=endpoints,
                app_url="http://localhost:3000",
            )

            assert result.success is True
            assert len(result.data.tests) == 1
            assert result.data.tests[0].type == "api"
