"""Tests for the NLP Test Creator module.

Tests cover:
- TestStep, TestAssertion, GeneratedTest dataclasses
- GeneratedTest.to_spec() and to_dict() methods
- NLPTestCreator initialization and async methods
- ConversationalTestBuilder class
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# TestStep Dataclass Tests
# ============================================================================


class TestTestStep:
    """Tests for TestStep dataclass."""

    def test_step_creation_minimal(self, mock_env_vars):
        """Test creating a TestStep with minimal required fields."""
        from src.agents.nlp_test_creator import TestStep

        step = TestStep(action="click")

        assert step.action == "click"
        assert step.target is None
        assert step.value is None
        assert step.description == ""

    def test_step_creation_full(self, mock_env_vars):
        """Test creating a TestStep with all fields."""
        from src.agents.nlp_test_creator import TestStep

        step = TestStep(
            action="fill",
            target="placeholder=Email",
            value="test@example.com",
            description="Enter email address",
        )

        assert step.action == "fill"
        assert step.target == "placeholder=Email"
        assert step.value == "test@example.com"
        assert step.description == "Enter email address"

    @pytest.mark.parametrize(
        "action,target,value",
        [
            ("goto", "/login", None),
            ("click", "text=Submit", None),
            ("fill", "#username", "testuser"),
            ("type", "[data-testid=search]", "search query"),
            ("wait", None, "2000"),
            ("press", "placeholder=Search", "Enter"),
            ("hover", ".dropdown-menu", None),
            ("select", "#country", "US"),
            ("screenshot", None, None),
        ],
    )
    def test_step_action_types(self, mock_env_vars, action, target, value):
        """Test various action types for TestStep."""
        from src.agents.nlp_test_creator import TestStep

        step = TestStep(action=action, target=target, value=value)

        assert step.action == action
        assert step.target == target
        assert step.value == value


# ============================================================================
# TestAssertion Dataclass Tests
# ============================================================================


class TestTestAssertion:
    """Tests for TestAssertion dataclass."""

    def test_assertion_creation_minimal(self, mock_env_vars):
        """Test creating a TestAssertion with minimal fields."""
        from src.agents.nlp_test_creator import TestAssertion

        assertion = TestAssertion(
            type="element_visible",
            target=".success-message",
            expected="true",
        )

        assert assertion.type == "element_visible"
        assert assertion.target == ".success-message"
        assert assertion.expected == "true"
        assert assertion.description == ""

    def test_assertion_creation_full(self, mock_env_vars):
        """Test creating a TestAssertion with all fields."""
        from src.agents.nlp_test_creator import TestAssertion

        assertion = TestAssertion(
            type="text_contains",
            target=".welcome-message",
            expected="Welcome back",
            description="Verify welcome message is displayed",
        )

        assert assertion.type == "text_contains"
        assert assertion.target == ".welcome-message"
        assert assertion.expected == "Welcome back"
        assert assertion.description == "Verify welcome message is displayed"

    @pytest.mark.parametrize(
        "assertion_type,target,expected",
        [
            ("element_visible", "#submit-btn", "true"),
            ("text_contains", ".message", "Success"),
            ("url_matches", None, "/dashboard"),
            ("value_equals", "#email", "test@example.com"),
            ("element_count", ".list-item", "5"),
        ],
    )
    def test_assertion_types(self, mock_env_vars, assertion_type, target, expected):
        """Test various assertion types."""
        from src.agents.nlp_test_creator import TestAssertion

        assertion = TestAssertion(
            type=assertion_type,
            target=target,
            expected=expected,
        )

        assert assertion.type == assertion_type
        assert assertion.target == target
        assert assertion.expected == expected


# ============================================================================
# GeneratedTest Dataclass Tests
# ============================================================================


class TestGeneratedTest:
    """Tests for GeneratedTest dataclass."""

    def test_generated_test_creation_minimal(self, mock_env_vars):
        """Test creating a GeneratedTest with minimal fields."""
        from src.agents.nlp_test_creator import GeneratedTest

        test = GeneratedTest(
            id="test-001",
            name="Login Test",
            description="Test login functionality",
            original_prompt="Test that login works",
        )

        assert test.id == "test-001"
        assert test.name == "Login Test"
        assert test.description == "Test login functionality"
        assert test.original_prompt == "Test that login works"
        assert test.steps == []
        assert test.assertions == []
        assert test.preconditions == []
        assert test.tags == []
        assert test.priority == "medium"
        assert test.estimated_duration_seconds == 30

    def test_generated_test_creation_full(self, mock_env_vars):
        """Test creating a GeneratedTest with all fields."""
        from src.agents.nlp_test_creator import GeneratedTest, TestAssertion, TestStep

        steps = [
            TestStep(action="goto", target="/login"),
            TestStep(action="fill", target="#email", value="test@example.com"),
            TestStep(action="click", target="text=Login"),
        ]
        assertions = [
            TestAssertion(type="url_matches", target=None, expected="/dashboard"),
            TestAssertion(type="element_visible", target=".welcome", expected="true"),
        ]

        test = GeneratedTest(
            id="test-002",
            name="Full Login Test",
            description="Complete login test with validation",
            original_prompt="Login and verify dashboard",
            steps=steps,
            assertions=assertions,
            preconditions=["User must exist", "App must be running"],
            tags=["login", "smoke", "critical"],
            priority="critical",
            estimated_duration_seconds=60,
        )

        assert test.id == "test-002"
        assert len(test.steps) == 3
        assert len(test.assertions) == 2
        assert len(test.preconditions) == 2
        assert len(test.tags) == 3
        assert test.priority == "critical"
        assert test.estimated_duration_seconds == 60


class TestGeneratedTestToSpec:
    """Tests for GeneratedTest.to_spec() method."""

    def test_to_spec_empty_test(self, mock_env_vars):
        """Test to_spec with no steps or assertions."""
        from src.agents.nlp_test_creator import GeneratedTest

        test = GeneratedTest(
            id="test-empty",
            name="Empty Test",
            description="No steps",
            original_prompt="Empty",
        )

        spec = test.to_spec()

        assert spec["id"] == "test-empty"
        assert spec["name"] == "Empty Test"
        assert spec["type"] == "ui"
        assert spec["priority"] == "medium"
        assert spec["preconditions"] == []
        assert spec["steps"] == []
        assert spec["assertions"] == []
        assert spec["tags"] == []

    def test_to_spec_with_steps_and_assertions(self, mock_env_vars):
        """Test to_spec with steps and assertions."""
        from src.agents.nlp_test_creator import GeneratedTest, TestAssertion, TestStep

        test = GeneratedTest(
            id="test-full",
            name="Full Test",
            description="Complete test",
            original_prompt="Complete test prompt",
            steps=[
                TestStep(action="goto", target="/page", value=None, description="Go to page"),
                TestStep(action="click", target="#btn", value=None, description="Click button"),
            ],
            assertions=[
                TestAssertion(
                    type="element_visible",
                    target=".result",
                    expected="true",
                    description="Check result",
                ),
            ],
            preconditions=["Logged in"],
            tags=["smoke"],
            priority="high",
        )

        spec = test.to_spec()

        assert spec["id"] == "test-full"
        assert spec["type"] == "ui"
        assert spec["priority"] == "high"
        assert len(spec["steps"]) == 2
        assert spec["steps"][0] == {"action": "goto", "target": "/page", "value": None}
        assert spec["steps"][1] == {"action": "click", "target": "#btn", "value": None}
        assert len(spec["assertions"]) == 1
        assert spec["assertions"][0] == {
            "type": "element_visible",
            "target": ".result",
            "expected": "true",
        }
        assert spec["preconditions"] == ["Logged in"]
        assert spec["tags"] == ["smoke"]

    def test_to_spec_does_not_include_description_in_steps(self, mock_env_vars):
        """Verify to_spec excludes description from steps (spec format)."""
        from src.agents.nlp_test_creator import GeneratedTest, TestStep

        test = GeneratedTest(
            id="test-x",
            name="Test",
            description="Desc",
            original_prompt="Prompt",
            steps=[
                TestStep(
                    action="click",
                    target="#btn",
                    value=None,
                    description="This should not be in spec",
                )
            ],
        )

        spec = test.to_spec()

        # The step in spec should only have action, target, value - no description
        assert "description" not in spec["steps"][0]


class TestGeneratedTestToDict:
    """Tests for GeneratedTest.to_dict() method."""

    def test_to_dict_empty_test(self, mock_env_vars):
        """Test to_dict with minimal fields."""
        from src.agents.nlp_test_creator import GeneratedTest

        test = GeneratedTest(
            id="test-dict",
            name="Dict Test",
            description="Testing to_dict",
            original_prompt="Original prompt here",
        )

        result = test.to_dict()

        assert result["id"] == "test-dict"
        assert result["name"] == "Dict Test"
        assert result["description"] == "Testing to_dict"
        assert result["original_prompt"] == "Original prompt here"
        assert result["steps"] == []
        assert result["assertions"] == []
        assert result["preconditions"] == []
        assert result["tags"] == []
        assert result["priority"] == "medium"
        assert result["estimated_duration_seconds"] == 30

    def test_to_dict_includes_descriptions(self, mock_env_vars):
        """Verify to_dict includes description fields for steps and assertions."""
        from src.agents.nlp_test_creator import GeneratedTest, TestAssertion, TestStep

        test = GeneratedTest(
            id="test-with-desc",
            name="Test",
            description="Desc",
            original_prompt="Prompt",
            steps=[
                TestStep(
                    action="click",
                    target="#btn",
                    value=None,
                    description="Click the button",
                )
            ],
            assertions=[
                TestAssertion(
                    type="visible",
                    target=".msg",
                    expected="true",
                    description="Check message",
                )
            ],
        )

        result = test.to_dict()

        assert result["steps"][0]["description"] == "Click the button"
        assert result["assertions"][0]["description"] == "Check message"

    def test_to_dict_full_structure(self, mock_env_vars):
        """Test to_dict returns complete structure."""
        from src.agents.nlp_test_creator import GeneratedTest, TestAssertion, TestStep

        test = GeneratedTest(
            id="full-test",
            name="Complete Test",
            description="Full description",
            original_prompt="Full original prompt",
            steps=[
                TestStep(
                    action="goto",
                    target="/home",
                    value=None,
                    description="Navigate home",
                ),
                TestStep(
                    action="fill",
                    target="#search",
                    value="query",
                    description="Enter search",
                ),
            ],
            assertions=[
                TestAssertion(
                    type="text_contains",
                    target=".results",
                    expected="Found",
                    description="Results found",
                ),
            ],
            preconditions=["App running", "User logged in"],
            tags=["search", "e2e"],
            priority="high",
            estimated_duration_seconds=45,
        )

        result = test.to_dict()

        assert result["id"] == "full-test"
        assert result["name"] == "Complete Test"
        assert result["description"] == "Full description"
        assert result["original_prompt"] == "Full original prompt"
        assert len(result["steps"]) == 2
        assert result["steps"][0] == {
            "action": "goto",
            "target": "/home",
            "value": None,
            "description": "Navigate home",
        }
        assert len(result["assertions"]) == 1
        assert result["assertions"][0]["description"] == "Results found"
        assert result["preconditions"] == ["App running", "User logged in"]
        assert result["tags"] == ["search", "e2e"]
        assert result["priority"] == "high"
        assert result["estimated_duration_seconds"] == 45


# ============================================================================
# NLPTestCreator Initialization Tests
# ============================================================================


class TestNLPTestCreatorInit:
    """Tests for NLPTestCreator initialization."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for NLPTestCreator."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock:
            mock.return_value.anthropic_api_key = MagicMock()
            mock.return_value.anthropic_api_key.get_secret_value.return_value = "test-api-key"
            yield mock

    def test_init_basic(self, mock_env_vars, mock_settings):
        """Test basic initialization."""
        with patch("src.agents.nlp_test_creator.anthropic.Anthropic") as mock_client:
            from src.agents.nlp_test_creator import NLPTestCreator

            creator = NLPTestCreator(app_url="https://example.com")

            assert creator.app_url == "https://example.com"
            mock_client.assert_called_once_with(api_key="test-api-key")

    def test_init_with_custom_model(self, mock_env_vars, mock_settings):
        """Test initialization with custom model."""
        with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
            from src.agents.nlp_test_creator import NLPTestCreator

            creator = NLPTestCreator(
                app_url="https://example.com",
                model="claude-opus-4-20250514",
            )

            assert creator.model == "claude-opus-4-20250514"

    def test_init_with_secret_value_api_key(self, mock_env_vars):
        """Test initialization when API key has get_secret_value method."""
        mock_api_key = MagicMock()
        mock_api_key.get_secret_value.return_value = "secret-key-value"

        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = mock_api_key
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic") as mock_client:
                from src.agents.nlp_test_creator import NLPTestCreator

                NLPTestCreator(app_url="https://example.com")

                mock_client.assert_called_once_with(api_key="secret-key-value")

    def test_init_with_plain_api_key(self, mock_env_vars):
        """Test initialization when API key is a plain string."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            # Simulate plain string API key (no get_secret_value)
            mock_settings.return_value.anthropic_api_key = "plain-api-key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic") as mock_client:
                from src.agents.nlp_test_creator import NLPTestCreator

                NLPTestCreator(app_url="https://example.com")

                mock_client.assert_called_once_with(api_key="plain-api-key")


# ============================================================================
# NLPTestCreator.create() Tests
# ============================================================================


class TestNLPTestCreatorCreate:
    """Tests for NLPTestCreator.create() async method."""

    @pytest.fixture
    def mock_creator(self, mock_env_vars):
        """Create a mock NLPTestCreator instance."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
                from src.agents.nlp_test_creator import NLPTestCreator

                return NLPTestCreator(app_url="https://example.com")

    @pytest.mark.asyncio
    async def test_create_basic(self, mock_creator):
        """Test basic test creation from description."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "id": "test-login",
                        "name": "User Login Test",
                        "description": "Verify user can log in",
                        "priority": "high",
                        "preconditions": ["App running"],
                        "steps": [
                            {
                                "action": "goto",
                                "target": "/login",
                                "value": None,
                                "description": "Go to login page",
                            },
                            {
                                "action": "fill",
                                "target": "#email",
                                "value": "test@example.com",
                                "description": "Enter email",
                            },
                            {
                                "action": "click",
                                "target": "text=Login",
                                "value": None,
                                "description": "Click login",
                            },
                        ],
                        "assertions": [
                            {
                                "type": "url_matches",
                                "target": None,
                                "expected": "/dashboard",
                                "description": "Verify redirect",
                            }
                        ],
                        "tags": ["login", "smoke"],
                        "estimated_duration_seconds": 30,
                    }
                )
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await mock_creator.create("Login with email test@example.com and verify dashboard")

        assert test.id == "test-login"
        assert test.name == "User Login Test"
        assert test.priority == "high"
        assert len(test.steps) == 3
        assert len(test.assertions) == 1
        assert test.tags == ["login", "smoke"]

    @pytest.mark.asyncio
    async def test_create_with_context(self, mock_creator):
        """Test create with app context."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "Test", "description": "Desc", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        await mock_creator.create(
            description="Test login",
            context="This is a React app with routes: /login, /dashboard, /profile",
        )

        # Verify context was included in the prompt
        call_args = mock_creator.client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "APP CONTEXT" in prompt
        assert "React app" in prompt

    @pytest.mark.asyncio
    async def test_create_with_test_data(self, mock_creator):
        """Test create with test data."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "Test", "description": "Desc", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        test_data = {"email": "admin@test.com", "password": "secret123"}
        await mock_creator.create(
            description="Test login",
            test_data=test_data,
        )

        # Verify test data was included in the prompt
        call_args = mock_creator.client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "TEST DATA AVAILABLE" in prompt
        assert "admin@test.com" in prompt

    @pytest.mark.asyncio
    async def test_create_parses_json_code_block(self, mock_creator):
        """Test parsing JSON from markdown code block."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='Here is the test:\n```json\n{"id": "t1", "name": "Test", "description": "Desc", "steps": [], "assertions": [], "tags": []}\n```'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await mock_creator.create("Create a test")

        assert test.id == "t1"
        assert test.name == "Test"

    @pytest.mark.asyncio
    async def test_create_parses_generic_code_block(self, mock_creator):
        """Test parsing JSON from generic code block."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='Test:\n```\n{"id": "t2", "name": "Test2", "description": "Desc", "steps": [], "assertions": [], "tags": []}\n```'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await mock_creator.create("Create a test")

        assert test.id == "t2"
        assert test.name == "Test2"

    @pytest.mark.asyncio
    async def test_create_generates_uuid_if_no_id(self, mock_creator):
        """Test that UUID is generated if no id in response."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"name": "Test No ID", "description": "Desc", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await mock_creator.create("Create test")

        assert test.id is not None
        assert len(test.id) > 0
        # UUID format check
        assert len(test.id) == 36  # UUID string length

    @pytest.mark.asyncio
    async def test_create_handles_missing_optional_fields(self, mock_creator):
        """Test handling of missing optional fields in response."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"id": "t1", "name": "Minimal", "steps": [], "assertions": []}')
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await mock_creator.create("Minimal test")

        assert test.id == "t1"
        assert test.name == "Minimal"
        assert test.priority == "medium"  # Default
        assert test.tags == []
        assert test.preconditions == []
        assert test.estimated_duration_seconds == 30  # Default

    @pytest.mark.asyncio
    async def test_create_raises_on_invalid_json(self, mock_creator):
        """Test that create raises exception on invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not valid JSON at all")]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        with pytest.raises(Exception):
            await mock_creator.create("Create test")

    @pytest.mark.asyncio
    async def test_create_raises_on_api_error(self, mock_creator):
        """Test that create raises exception on API error."""
        mock_creator.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception) as exc_info:
            await mock_creator.create("Create test")

        assert "API Error" in str(exc_info.value)


# ============================================================================
# NLPTestCreator.create_multiple() Tests
# ============================================================================


class TestNLPTestCreatorCreateMultiple:
    """Tests for NLPTestCreator.create_multiple() async method."""

    @pytest.fixture
    def mock_creator(self, mock_env_vars):
        """Create a mock NLPTestCreator instance."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
                from src.agents.nlp_test_creator import NLPTestCreator

                return NLPTestCreator(app_url="https://example.com")

    @pytest.mark.asyncio
    async def test_create_multiple_empty_list(self, mock_creator):
        """Test create_multiple with empty list."""
        tests = await mock_creator.create_multiple([])

        assert tests == []

    @pytest.mark.asyncio
    async def test_create_multiple_single_description(self, mock_creator):
        """Test create_multiple with single description."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "Test 1", "description": "Desc", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        tests = await mock_creator.create_multiple(["Test login"])

        assert len(tests) == 1
        assert tests[0].id == "t1"

    @pytest.mark.asyncio
    async def test_create_multiple_descriptions(self, mock_creator):
        """Test create_multiple with multiple descriptions."""
        responses = [
            MagicMock(
                content=[
                    MagicMock(
                        text='{"id": "t1", "name": "Login Test", "description": "D1", "steps": [], "assertions": [], "tags": []}'
                    )
                ]
            ),
            MagicMock(
                content=[
                    MagicMock(
                        text='{"id": "t2", "name": "Signup Test", "description": "D2", "steps": [], "assertions": [], "tags": []}'
                    )
                ]
            ),
            MagicMock(
                content=[
                    MagicMock(
                        text='{"id": "t3", "name": "Checkout Test", "description": "D3", "steps": [], "assertions": [], "tags": []}'
                    )
                ]
            ),
        ]
        mock_creator.client.messages.create = MagicMock(side_effect=responses)

        tests = await mock_creator.create_multiple(
            ["Test login", "Test signup", "Test checkout"]
        )

        assert len(tests) == 3
        assert tests[0].name == "Login Test"
        assert tests[1].name == "Signup Test"
        assert tests[2].name == "Checkout Test"

    @pytest.mark.asyncio
    async def test_create_multiple_with_context(self, mock_creator):
        """Test create_multiple passes context to each create call."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "Test", "description": "D", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        await mock_creator.create_multiple(
            descriptions=["Test 1", "Test 2"],
            context="React app with TypeScript",
        )

        # Both calls should have the context
        assert mock_creator.client.messages.create.call_count == 2
        for call in mock_creator.client.messages.create.call_args_list:
            prompt = call[1]["messages"][0]["content"]
            assert "APP CONTEXT" in prompt
            assert "React app" in prompt


# ============================================================================
# NLPTestCreator.create_from_story() Tests
# ============================================================================


class TestNLPTestCreatorCreateFromStory:
    """Tests for NLPTestCreator.create_from_story() async method."""

    @pytest.fixture
    def mock_creator(self, mock_env_vars):
        """Create a mock NLPTestCreator instance."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
                from src.agents.nlp_test_creator import NLPTestCreator

                return NLPTestCreator(app_url="https://example.com")

    @pytest.mark.asyncio
    async def test_create_from_story_basic(self, mock_creator):
        """Test creating tests from user story."""
        # First call returns scenarios, subsequent calls create tests
        scenario_response = MagicMock()
        scenario_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "scenarios": [
                            {
                                "name": "Valid Login",
                                "description": "Login with valid credentials",
                                "priority": "critical",
                                "tags": ["login", "auth"],
                            },
                            {
                                "name": "Invalid Password",
                                "description": "Login with invalid password shows error",
                                "priority": "high",
                                "tags": ["login", "error"],
                            },
                        ]
                    }
                )
            )
        ]

        test_response = MagicMock()
        test_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "Generated", "description": "D", "steps": [], "assertions": [], "tags": [], "priority": "medium"}'
            )
        ]

        mock_creator.client.messages.create = MagicMock(
            side_effect=[scenario_response, test_response, test_response]
        )

        user_story = """
        As a user, I want to log in so that I can access my account.
        - Valid credentials should redirect to dashboard
        - Invalid password should show error message
        """

        tests = await mock_creator.create_from_story(user_story)

        assert len(tests) == 2
        # Verify scenario names/priorities are applied
        assert tests[0].name == "Valid Login"
        assert tests[0].priority == "critical"
        assert tests[0].tags == ["login", "auth"]
        assert tests[1].name == "Invalid Password"
        assert tests[1].priority == "high"

    @pytest.mark.asyncio
    async def test_create_from_story_empty_scenarios(self, mock_creator):
        """Test handling of empty scenarios."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"scenarios": []}')]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        tests = await mock_creator.create_from_story("Simple user story")

        assert tests == []

    @pytest.mark.asyncio
    async def test_create_from_story_with_context(self, mock_creator):
        """Test create_from_story passes context."""
        scenario_response = MagicMock()
        scenario_response.content = [
            MagicMock(
                text='{"scenarios": [{"name": "Test", "description": "Desc", "priority": "medium", "tags": []}]}'
            )
        ]
        test_response = MagicMock()
        test_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "Test", "description": "D", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(
            side_effect=[scenario_response, test_response]
        )

        await mock_creator.create_from_story(
            user_story="As a user...",
            context="Django app with REST API",
        )

        # Second call (create test) should have context
        second_call = mock_creator.client.messages.create.call_args_list[1]
        prompt = second_call[1]["messages"][0]["content"]
        assert "APP CONTEXT" in prompt

    @pytest.mark.asyncio
    async def test_create_from_story_parses_json_code_block(self, mock_creator):
        """Test parsing scenarios from JSON code block."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='```json\n{"scenarios": [{"name": "Test", "description": "D", "priority": "low", "tags": ["e2e"]}]}\n```'
            )
        ]
        test_response = MagicMock()
        test_response.content = [
            MagicMock(
                text='{"id": "t1", "name": "X", "description": "D", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(
            side_effect=[mock_response, test_response]
        )

        tests = await mock_creator.create_from_story("Story")

        assert len(tests) == 1
        assert tests[0].priority == "low"
        assert tests[0].tags == ["e2e"]

    @pytest.mark.asyncio
    async def test_create_from_story_raises_on_error(self, mock_creator):
        """Test that create_from_story raises exception on API error."""
        mock_creator.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception) as exc_info:
            await mock_creator.create_from_story("User story")

        assert "API Error" in str(exc_info.value)


# ============================================================================
# NLPTestCreator.enhance_test() Tests
# ============================================================================


class TestNLPTestCreatorEnhanceTest:
    """Tests for NLPTestCreator.enhance_test() async method."""

    @pytest.fixture
    def mock_creator(self, mock_env_vars):
        """Create a mock NLPTestCreator instance."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
                from src.agents.nlp_test_creator import NLPTestCreator

                return NLPTestCreator(app_url="https://example.com")

    @pytest.fixture
    def sample_test(self, mock_env_vars):
        """Create a sample test for enhancement."""
        from src.agents.nlp_test_creator import GeneratedTest, TestAssertion, TestStep

        return GeneratedTest(
            id="test-original",
            name="Original Test",
            description="Original description",
            original_prompt="Original prompt",
            steps=[
                TestStep(action="goto", target="/login"),
                TestStep(action="click", target="#submit"),
            ],
            assertions=[
                TestAssertion(type="url_matches", target=None, expected="/dashboard")
            ],
            preconditions=["User exists"],
            tags=["login"],
            priority="medium",
        )

    @pytest.mark.asyncio
    async def test_enhance_test_basic(self, mock_creator, sample_test):
        """Test basic test enhancement."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "id": "test-original",
                        "name": "Enhanced Test",
                        "description": "Enhanced description",
                        "priority": "high",
                        "preconditions": ["User exists", "App running"],
                        "steps": [
                            {"action": "goto", "target": "/login"},
                            {"action": "fill", "target": "#email", "value": "test@example.com"},
                            {"action": "click", "target": "#submit"},
                        ],
                        "assertions": [
                            {"type": "url_matches", "target": None, "expected": "/dashboard"},
                            {"type": "element_visible", "target": ".welcome", "expected": "true"},
                        ],
                        "tags": ["login", "critical"],
                    }
                )
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        enhanced = await mock_creator.enhance_test(
            test=sample_test,
            feedback="Add email input step and verify welcome message",
        )

        assert enhanced.id == "test-original"  # ID preserved
        assert enhanced.name == "Enhanced Test"
        assert enhanced.priority == "high"
        assert len(enhanced.steps) == 3
        assert len(enhanced.assertions) == 2
        assert enhanced.original_prompt == "Original prompt"  # Preserved

    @pytest.mark.asyncio
    async def test_enhance_test_preserves_original_prompt(self, mock_creator, sample_test):
        """Test that original_prompt is preserved after enhancement."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"id": "test-original", "name": "Enhanced", "description": "D", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        enhanced = await mock_creator.enhance_test(sample_test, "Make it better")

        assert enhanced.original_prompt == sample_test.original_prompt

    @pytest.mark.asyncio
    async def test_enhance_test_preserves_id(self, mock_creator, sample_test):
        """Test that test ID is preserved after enhancement."""
        mock_response = MagicMock()
        # Response might have different ID, but we should use original
        mock_response.content = [
            MagicMock(
                text='{"id": "new-id", "name": "Test", "description": "D", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        enhanced = await mock_creator.enhance_test(sample_test, "Feedback")

        # Should preserve original ID
        assert enhanced.id == "test-original"

    @pytest.mark.asyncio
    async def test_enhance_test_uses_original_values_as_fallback(
        self, mock_creator, sample_test
    ):
        """Test that original values are used as fallback."""
        mock_response = MagicMock()
        # Response missing some fields
        mock_response.content = [
            MagicMock(text='{"name": "New Name", "steps": [], "assertions": []}')
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        enhanced = await mock_creator.enhance_test(sample_test, "Feedback")

        # Original values used as fallback
        assert enhanced.description == sample_test.description
        assert enhanced.preconditions == sample_test.preconditions
        assert enhanced.tags == sample_test.tags
        assert enhanced.priority == sample_test.priority

    @pytest.mark.asyncio
    async def test_enhance_test_includes_current_spec_in_prompt(
        self, mock_creator, sample_test
    ):
        """Test that current spec is included in the prompt."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text='{"id": "x", "name": "X", "description": "D", "steps": [], "assertions": [], "tags": []}'
            )
        ]
        mock_creator.client.messages.create = MagicMock(return_value=mock_response)

        await mock_creator.enhance_test(sample_test, "Add more assertions")

        call_args = mock_creator.client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "CURRENT TEST" in prompt
        assert "test-original" in prompt
        assert "Add more assertions" in prompt

    @pytest.mark.asyncio
    async def test_enhance_test_raises_on_error(self, mock_creator, sample_test):
        """Test that enhance_test raises exception on API error."""
        mock_creator.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        with pytest.raises(Exception):
            await mock_creator.enhance_test(sample_test, "Feedback")


# ============================================================================
# ConversationalTestBuilder Tests
# ============================================================================


class TestConversationalTestBuilder:
    """Tests for ConversationalTestBuilder class."""

    @pytest.fixture
    def mock_builder(self, mock_env_vars):
        """Create a mock ConversationalTestBuilder instance."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
                from src.agents.nlp_test_creator import ConversationalTestBuilder

                return ConversationalTestBuilder(app_url="https://example.com")

    def test_init(self, mock_builder):
        """Test ConversationalTestBuilder initialization."""
        assert mock_builder.app_url == "https://example.com"
        assert mock_builder.conversation_history == []
        assert mock_builder.current_tests == []
        assert mock_builder.creator is not None

    @pytest.mark.asyncio
    async def test_chat_basic(self, mock_builder):
        """Test basic chat functionality."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I can help you create tests for that.")]
        mock_builder.client.messages.create = MagicMock(return_value=mock_response)

        reply = await mock_builder.chat("What tests should I create?")

        assert "help" in reply.lower()
        assert len(mock_builder.conversation_history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_chat_create_test_action(self, mock_builder):
        """Test chat with CREATE_TEST action."""
        # First response from chat
        chat_response = MagicMock()
        chat_response.content = [
            MagicMock(text="ACTION: CREATE_TEST\nDESCRIPTION: Test login with valid credentials")
        ]
        mock_builder.client.messages.create = MagicMock(return_value=chat_response)

        # Mock the creator's create method
        from src.agents.nlp_test_creator import GeneratedTest

        mock_test = GeneratedTest(
            id="t1",
            name="Login Test",
            description="D",
            original_prompt="P",
            steps=[MagicMock()] * 3,  # 3 steps
        )
        mock_builder.creator.create = AsyncMock(return_value=mock_test)

        reply = await mock_builder.chat("Create a login test")

        assert "Created test" in reply
        assert "Login Test" in reply
        assert len(mock_builder.current_tests) == 1

    @pytest.mark.asyncio
    async def test_chat_records_history(self, mock_builder):
        """Test that chat records conversation history."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response 1")]
        mock_builder.client.messages.create = MagicMock(return_value=mock_response)

        await mock_builder.chat("Message 1")
        await mock_builder.chat("Message 2")

        assert len(mock_builder.conversation_history) == 4
        assert mock_builder.conversation_history[0]["role"] == "user"
        assert mock_builder.conversation_history[0]["content"] == "Message 1"
        assert mock_builder.conversation_history[1]["role"] == "assistant"
        assert mock_builder.conversation_history[2]["role"] == "user"
        assert mock_builder.conversation_history[2]["content"] == "Message 2"

    @pytest.mark.asyncio
    async def test_chat_multiline_description(self, mock_builder):
        """Test CREATE_TEST with multiline description."""
        chat_response = MagicMock()
        chat_response.content = [
            MagicMock(
                text="ACTION: CREATE_TEST\nDESCRIPTION: Login test\nwith multiple lines\nof description\n\nSome other text"
            )
        ]
        mock_builder.client.messages.create = MagicMock(return_value=chat_response)

        from src.agents.nlp_test_creator import GeneratedTest

        mock_test = GeneratedTest(
            id="t1",
            name="Test",
            description="D",
            original_prompt="P",
            steps=[],
        )
        mock_builder.creator.create = AsyncMock(return_value=mock_test)

        await mock_builder.chat("Create test")

        # Verify description was combined
        create_call = mock_builder.creator.create.call_args
        description = create_call[0][0]
        assert "Login test" in description
        assert "multiple lines" in description

    def test_get_all_tests(self, mock_builder):
        """Test get_all_tests returns current tests."""
        from src.agents.nlp_test_creator import GeneratedTest

        test1 = GeneratedTest(
            id="t1", name="Test 1", description="D1", original_prompt="P1"
        )
        test2 = GeneratedTest(
            id="t2", name="Test 2", description="D2", original_prompt="P2"
        )
        mock_builder.current_tests = [test1, test2]

        tests = mock_builder.get_all_tests()

        assert len(tests) == 2
        assert tests[0].id == "t1"
        assert tests[1].id == "t2"

    def test_export_tests(self, mock_builder):
        """Test export_tests converts to specs."""
        from src.agents.nlp_test_creator import GeneratedTest, TestStep

        test = GeneratedTest(
            id="t1",
            name="Test 1",
            description="Description",
            original_prompt="Prompt",
            steps=[TestStep(action="goto", target="/page")],
        )
        mock_builder.current_tests = [test]

        specs = mock_builder.export_tests()

        assert len(specs) == 1
        assert specs[0]["id"] == "t1"
        assert specs[0]["type"] == "ui"
        assert len(specs[0]["steps"]) == 1

    def test_export_tests_empty(self, mock_builder):
        """Test export_tests with no tests."""
        specs = mock_builder.export_tests()

        assert specs == []


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateNLPTestCreator:
    """Tests for create_nlp_test_creator factory function."""

    def test_factory_function(self, mock_env_vars):
        """Test factory function creates NLPTestCreator."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"
            with patch("src.agents.nlp_test_creator.anthropic.Anthropic"):
                from src.agents.nlp_test_creator import create_nlp_test_creator

                creator = create_nlp_test_creator(app_url="https://test.com")

                assert creator.app_url == "https://test.com"
