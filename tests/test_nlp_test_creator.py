"""Tests for NLP Test Creator module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGeneratedTest:
    """Tests for GeneratedTest dataclass."""

    def test_generated_test_creation(self):
        """Test creating a generated test."""
        from src.agents.nlp_test_creator import GeneratedTest, TestStep, TestAssertion

        test = GeneratedTest(
            id="login-test-1",
            name="User Login Test",
            description="Test user login with valid credentials",
            original_prompt="Login as test@example.com and verify dashboard loads",
            steps=[
                TestStep(action="goto", target="/login"),
                TestStep(action="fill", target="#email", value="test@example.com"),
                TestStep(action="fill", target="#password", value="password123"),
                TestStep(action="click", target="#submit"),
            ],
            assertions=[
                TestAssertion(type="url_contains", target="url", expected="/dashboard"),
                TestAssertion(type="element_visible", target="#welcome-message", expected="visible"),
            ],
            tags=["smoke", "auth"],
        )

        assert test.id == "login-test-1"
        assert len(test.steps) == 4
        assert len(test.assertions) == 2

    def test_to_spec(self):
        """Test converting to test spec format."""
        from src.agents.nlp_test_creator import GeneratedTest, TestStep, TestAssertion

        test = GeneratedTest(
            id="test-1",
            name="Simple Test",
            description="A simple test",
            original_prompt="test",
            steps=[TestStep(action="goto", target="/")],
            assertions=[TestAssertion(type="title_contains", target="title", expected="Home")],
        )

        spec = test.to_spec()

        assert spec["id"] == "test-1"
        assert spec["name"] == "Simple Test"
        assert spec["type"] == "ui"
        assert "steps" in spec
        assert "assertions" in spec


class TestTestStep:
    """Tests for TestStep dataclass."""

    def test_step_creation(self):
        """Test creating a test step."""
        from src.agents.nlp_test_creator import TestStep

        step = TestStep(
            action="click",
            target="#submit-btn",
            description="Click the submit button",
        )

        assert step.action == "click"
        assert step.target == "#submit-btn"
        assert step.value is None


class TestTestAssertion:
    """Tests for TestAssertion dataclass."""

    def test_assertion_creation(self):
        """Test creating a test assertion."""
        from src.agents.nlp_test_creator import TestAssertion

        assertion = TestAssertion(
            type="element_visible",
            target="#success-message",
            expected="visible",
            description="Check success message shows",
        )

        assert assertion.type == "element_visible"
        assert assertion.target == "#success-message"


class TestNLPTestCreator:
    """Tests for NLPTestCreator class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock:
            settings = MagicMock()
            settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def creator(self, mock_settings):
        """Create NLPTestCreator with mocked client."""
        with patch("src.agents.nlp_test_creator.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            from src.agents.nlp_test_creator import NLPTestCreator
            c = NLPTestCreator(app_url="http://localhost:3000")
            c.client = mock_client
            return c

    @pytest.mark.asyncio
    async def test_create_simple_test(self, creator):
        """Test creating a simple test from description."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "id": "login-test",
    "name": "User Login Test",
    "description": "Verify user can login successfully",
    "steps": [
        {"action": "goto", "target": "/login", "description": "Navigate to login page"},
        {"action": "fill", "target": "#email", "value": "admin@example.com", "description": "Enter email"},
        {"action": "fill", "target": "#password", "value": "admin123", "description": "Enter password"},
        {"action": "click", "target": "button[type=submit]", "description": "Click submit"}
    ],
    "assertions": [
        {"type": "url_contains", "target": "url", "expected": "/dashboard", "description": "On dashboard"},
        {"type": "element_visible", "target": ".welcome-banner", "expected": "visible", "description": "Welcome shown"}
    ],
    "tags": ["smoke", "auth", "login"],
    "priority": "high"
}
```""")]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=200)
        creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await creator.create("Login as admin@example.com and verify dashboard loads")

        assert test.id == "login-test"
        assert len(test.steps) >= 1
        assert test.steps[0].action == "goto"

    @pytest.mark.asyncio
    async def test_create_test_with_context(self, creator):
        """Test creating a test with additional context."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "id": "checkout-test",
    "name": "Checkout Flow Test",
    "description": "Test checkout process",
    "steps": [
        {"action": "goto", "target": "/cart", "description": "Go to cart"},
        {"action": "click", "target": "#checkout-btn", "description": "Click checkout"}
    ],
    "assertions": [
        {"type": "url_contains", "target": "url", "expected": "/checkout", "description": "On checkout"}
    ],
    "tags": ["checkout", "e2e"],
    "priority": "high"
}
```""")]
        mock_response.usage = MagicMock(input_tokens=600, output_tokens=250)
        creator.client.messages.create = MagicMock(return_value=mock_response)

        test = await creator.create(
            "Complete the checkout process",
            context="E-commerce app with cart at /cart"
        )

        assert test.id == "checkout-test"
        assert "checkout" in test.tags

    @pytest.mark.asyncio
    async def test_create_from_story(self, creator):
        """Test creating tests from user story."""
        # First response: parse scenarios
        scenario_response = MagicMock()
        scenario_response.content = [MagicMock(text="""```json
{
    "scenarios": [
        {
            "name": "Password Reset Request",
            "description": "User requests password reset via forgot password page",
            "priority": "high",
            "tags": ["auth", "password-reset"]
        }
    ]
}
```""")]

        # Second response: create test from scenario
        test_response = MagicMock()
        test_response.content = [MagicMock(text="""```json
{
    "id": "password-reset-request",
    "name": "Password Reset Request",
    "description": "User requests password reset",
    "steps": [
        {"action": "goto", "target": "/forgot-password", "description": "Go to forgot password"},
        {"action": "fill", "target": "#email", "value": "user@example.com", "description": "Enter email"},
        {"action": "click", "target": "#submit", "description": "Submit"}
    ],
    "assertions": [
        {"type": "element_visible", "target": ".success-message", "expected": "visible", "description": "Success shown"}
    ],
    "tags": ["auth", "password-reset"],
    "priority": "high"
}
```""")]

        creator.client.messages.create = MagicMock(side_effect=[scenario_response, test_response])

        tests = await creator.create_from_story("""
            As a user, I want to reset my password
            So that I can regain access to my account
        """)

        assert len(tests) >= 1
        assert all(hasattr(t, 'id') for t in tests)

    @pytest.mark.asyncio
    async def test_create_multiple(self, creator):
        """Test creating multiple tests from descriptions."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "tests": [
        {
            "id": "test-1",
            "name": "Login Test",
            "description": "First test",
            "steps": [],
            "assertions": [],
            "tags": [],
            "priority": "medium"
        },
        {
            "id": "test-2",
            "name": "Signup Test",
            "description": "Second test",
            "steps": [],
            "assertions": [],
            "tags": [],
            "priority": "medium"
        }
    ]
}
```""")]
        mock_response.usage = MagicMock(input_tokens=400, output_tokens=200)
        creator.client.messages.create = MagicMock(return_value=mock_response)

        tests = await creator.create_multiple([
            "Test the login page",
            "Test the signup page",
        ])

        assert len(tests) == 2

    @pytest.mark.asyncio
    async def test_enhance_test(self, creator):
        """Test enhancing an existing test with feedback."""
        from src.agents.nlp_test_creator import GeneratedTest

        original = GeneratedTest(
            id="test-1",
            name="Login Test",
            description="Basic login",
            original_prompt="Login test",
            steps=[],
            assertions=[],
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "id": "test-1",
    "name": "Login Test - Enhanced",
    "description": "Login with validation",
    "steps": [
        {"action": "goto", "target": "/login", "description": "Go to login"},
        {"action": "fill", "target": "#email", "value": "test@example.com", "description": "Enter email"},
        {"action": "fill", "target": "#password", "value": "password", "description": "Enter password"},
        {"action": "click", "target": "#submit", "description": "Submit"}
    ],
    "assertions": [
        {"type": "url_contains", "target": "url", "expected": "/dashboard", "description": "On dashboard"}
    ],
    "tags": ["login", "enhanced"],
    "priority": "high"
}
```""")]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=250)
        creator.client.messages.create = MagicMock(return_value=mock_response)

        enhanced = await creator.enhance_test(original, "Add actual login steps and assertions")

        assert len(enhanced.steps) > len(original.steps)


class TestConversationalTestBuilder:
    """Tests for ConversationalTestBuilder class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.nlp_test_creator.get_settings") as mock:
            settings = MagicMock()
            settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def builder(self, mock_settings):
        """Create ConversationalTestBuilder with mocked client."""
        with patch("src.agents.nlp_test_creator.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            from src.agents.nlp_test_creator import ConversationalTestBuilder
            b = ConversationalTestBuilder(app_url="http://localhost:3000")
            b.client = mock_client
            return b

    @pytest.mark.asyncio
    async def test_chat_initial_message(self, builder):
        """Test initial chat message."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I'll help you create a test. What would you like to test?")]
        builder.client.messages.create = MagicMock(return_value=mock_response)

        response = await builder.chat("I want to test the login page")

        assert len(response) > 0
        assert len(builder.conversation_history) >= 1

    @pytest.mark.asyncio
    async def test_chat_builds_conversation(self, builder):
        """Test that chat builds conversation history."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        builder.client.messages.create = MagicMock(return_value=mock_response)

        await builder.chat("First message")
        await builder.chat("Second message")

        assert len(builder.conversation_history) >= 2

    def test_get_all_tests(self, builder):
        """Test getting all tests."""
        from src.agents.nlp_test_creator import GeneratedTest
        test = GeneratedTest(
            id="test-1",
            name="Test",
            description="Test desc",
            original_prompt="prompt",
            steps=[],
            assertions=[],
        )
        builder.current_tests = [test]

        tests = builder.get_all_tests()

        assert len(tests) == 1
        assert tests[0].id == "test-1"

    def test_export_tests(self, builder):
        """Test exporting tests as specs."""
        from src.agents.nlp_test_creator import GeneratedTest, TestStep
        test = GeneratedTest(
            id="test-1",
            name="Test",
            description="Test desc",
            original_prompt="prompt",
            steps=[TestStep(action="goto", target="/")],
            assertions=[],
        )
        builder.current_tests = [test]

        specs = builder.export_tests()

        assert len(specs) == 1
        assert specs[0]["id"] == "test-1"
