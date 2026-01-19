"""Shared fixtures for E2E Testing Agent tests."""

import os
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if tree-sitter is available
try:
    import tree_sitter
    import tree_sitter_languages
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test (fast, critical path - included in CI)"
    )
    config.addinivalue_line(
        "markers", "requires_tree_sitter: mark test as requiring tree-sitter"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real API keys"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require tree-sitter when it's not installed."""
    if TREE_SITTER_AVAILABLE:
        return

    skip_tree_sitter = pytest.mark.skip(
        reason="tree-sitter not installed. Install with: pip install tree-sitter tree-sitter-languages"
    )

    for item in items:
        if "requires_tree_sitter" in item.keywords:
            item.add_marker(skip_tree_sitter)


@pytest.fixture
def tree_sitter_available():
    """Fixture that returns whether tree-sitter is available."""
    return TREE_SITTER_AVAILABLE


# Set test environment variables before importing modules
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-openai-key")
os.environ.setdefault("STAGEHAND_WORKER_URL", "https://test-worker.workers.dev")


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("STAGEHAND_WORKER_URL", "https://test-worker.workers.dev")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account-id")
    # Clear DATABASE_URL to ensure tests use MemorySaver instead of PostgresSaver
    # (PostgresSaver.from_conn_string returns a context manager in newer langgraph)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Reset the checkpointer singleton to pick up the env change
    from src.orchestrator.checkpointer import reset_checkpointer
    reset_checkpointer()


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"result": "success"}')]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_async_anthropic_client():
    """Create a mock AsyncAnthropic client."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"result": "success"}')]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}
    mock_response.text = '{"success": true}'
    mock_response.content = b"test content"
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.delete = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def sample_test_spec():
    """Sample test specification for testing."""
    return {
        "id": "test-001",
        "name": "Login Flow Test",
        "type": "ui",
        "priority": "high",
        "description": "Test the login flow",
        "steps": [
            {"action": "goto", "target": "/login", "value": None, "timeout": 5000},
            {"action": "fill", "target": "#email", "value": "test@example.com", "timeout": 5000},
            {"action": "fill", "target": "#password", "value": "password123", "timeout": 5000},
            {"action": "click", "target": "#submit-btn", "value": None, "timeout": 5000},
        ],
        "assertions": [
            {"type": "url_contains", "target": None, "expected": "/dashboard"},
            {"type": "element_visible", "target": ".welcome-message", "expected": None},
        ],
    }


@pytest.fixture
def sample_api_test_spec():
    """Sample API test specification."""
    return {
        "id": "api-test-001",
        "name": "User API Test",
        "type": "api",
        "priority": "high",
        "description": "Test user API endpoints",
        "steps": [
            {
                "action": "get",
                "target": "/api/users",
                "headers": {"Content-Type": "application/json"},
            },
        ],
        "assertions": [
            {"type": "status_code", "expected": "200"},
            {"type": "response_contains", "expected": "Test User"},
        ],
    }


@pytest.fixture
def mock_playwright_tools():
    """Create mock PlaywrightTools."""
    mock_pw = AsyncMock()
    mock_pw.goto = AsyncMock()
    mock_pw.click = AsyncMock()
    mock_pw.fill = AsyncMock()
    mock_pw.type_text = AsyncMock()
    mock_pw.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
    mock_pw.is_visible = AsyncMock(return_value=True)
    mock_pw.get_text = AsyncMock(return_value="Welcome, Test User!")
    mock_pw.get_current_url = AsyncMock(return_value="https://example.com/dashboard")
    mock_pw.wait_for_selector = AsyncMock()
    mock_pw.wait = AsyncMock()
    mock_pw.hover = AsyncMock()
    mock_pw.select_option = AsyncMock()
    mock_pw.press_key = AsyncMock()
    mock_pw.double_click = AsyncMock()
    mock_pw.get_input_value = AsyncMock(return_value="test value")
    mock_pw._page = MagicMock()
    mock_pw._page.evaluate = AsyncMock()
    return mock_pw


@pytest.fixture
def sample_application_model():
    """Sample ApplicationModel for cognitive engine tests."""
    from datetime import datetime
    return {
        "app_id": "test-app-001",
        "name": "https://example.com",
        "purpose": "E-commerce platform",
        "user_personas": [
            {"name": "Shopper", "goals": ["Browse products", "Make purchases"]}
        ],
        "core_user_journeys": [
            {"name": "Checkout", "steps": ["Add to cart", "Proceed to checkout", "Pay"]}
        ],
        "business_rules": [
            {"rule": "Users must be logged in to checkout", "severity": "high"}
        ],
        "invariants": ["Cart total must equal sum of item prices"],
        "risk_areas": [{"area": "Payment processing", "risk": "high"}],
        "version": "1.0",
        "last_learned": datetime.utcnow(),
        "confidence_score": 0.85,
    }


@pytest.fixture
def mock_model_response():
    """Create a standard mock model response."""
    return {
        "content": "Test response content",
        "model": "claude-sonnet-4-5-20250514",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.001,
        "model_name": "sonnet",
    }


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Any cleanup code here


# ==============================================================================
# Session-scoped fixtures for test environment setup
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(request):
    """Setup test environment variables for the entire test session.

    Note: Integration tests (marked with @pytest.mark.integration) are excluded
    from the environment patching so they can use real API keys.
    """
    # Check if we're running integration tests (they need real API keys)
    # Integration tests are detected by checking if any collected item has the marker
    has_integration_tests = any(
        item.get_closest_marker("integration") is not None
        for item in request.session.items
    )

    if has_integration_tests:
        # Don't patch environment for integration tests - they use real keys
        yield
    else:
        # Patch environment for unit tests
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-test-key-12345",
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-service-key",
        }):
            yield


# ==============================================================================
# Mock Anthropic/LangChain fixtures
# ==============================================================================

@pytest.fixture
def mock_chat_anthropic():
    """Mock ChatAnthropic for LangChain."""
    with patch("langchain_anthropic.ChatAnthropic") as mock:
        mock_instance = mock.return_value
        mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
            content="Test response",
            tool_calls=[],
        ))
        mock_instance.invoke = MagicMock(return_value=MagicMock(
            content="Test response",
            tool_calls=[],
        ))
        yield mock_instance


@pytest.fixture
def mock_langchain_messages():
    """Provide mock LangChain message classes."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    return {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "SystemMessage": SystemMessage,
        "ToolMessage": ToolMessage,
    }


# ==============================================================================
# LangGraph fixtures
# ==============================================================================

@pytest.fixture
def mock_checkpointer():
    """Create a MemorySaver checkpointer for testing."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


@pytest.fixture
def sample_test_state():
    """Sample test state for testing orchestrator."""
    from datetime import datetime
    return {
        "run_id": "test-run-123",
        "codebase_path": "/test/app",
        "app_url": "http://localhost:3000",
        "messages": [],
        "codebase_summary": "",
        "testable_surfaces": [],
        "changed_files": [],
        "test_plan": [],
        "test_priorities": {},
        "current_test_index": 0,
        "current_test": None,
        "test_results": [],
        "passed_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "failures": [],
        "healing_queue": [],
        "screenshots": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "iteration": 0,
        "max_iterations": 100,
        "next_agent": "analyze_code",
        "should_continue": True,
        "error": None,
        "started_at": datetime.now(UTC).isoformat(),
        "pr_number": None,
        "user_id": "test-user",
        "session_id": "test-session",
        "security_summary": None,
    }


@pytest.fixture
def sample_test_plan():
    """Sample test plan for testing."""
    return [
        {
            "id": "test-001",
            "name": "Login Flow Test",
            "type": "ui",
            "priority": "high",
            "steps": [
                {"action": "goto", "target": "/login"},
                {"action": "fill", "target": "#email", "value": "test@example.com"},
                {"action": "fill", "target": "#password", "value": "password123"},
                {"action": "click", "target": "#submit"},
            ],
            "assertions": [
                {"type": "url_contains", "expected": "/dashboard"},
            ],
        },
        {
            "id": "test-002",
            "name": "Logout Flow Test",
            "type": "ui",
            "priority": "medium",
            "steps": [
                {"action": "click", "target": "#logout-btn"},
            ],
            "assertions": [
                {"type": "url_contains", "expected": "/login"},
            ],
        },
    ]


@pytest.fixture
def sample_test_results():
    """Sample test results for testing."""
    return [
        {
            "test_id": "test-001",
            "status": "passed",
            "duration_seconds": 5.5,
            "assertions_passed": 2,
            "assertions_failed": 0,
            "error_message": None,
        },
        {
            "test_id": "test-002",
            "status": "failed",
            "duration_seconds": 3.2,
            "assertions_passed": 0,
            "assertions_failed": 1,
            "error_message": "Element not found: #logout-btn",
        },
    ]


# ==============================================================================
# Streaming/SSE fixtures
# ==============================================================================

@pytest.fixture
def mock_sse_response():
    """Mock SSE EventSourceResponse for testing."""
    from sse_starlette.sse import EventSourceResponse

    async def mock_generator():
        yield {"event": "start", "data": '{"thread_id": "test-123"}'}
        yield {"event": "complete", "data": '{"success": true}'}

    return EventSourceResponse(mock_generator())


class AsyncIteratorMock:
    """Mock async iterator for testing streaming."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def async_iterator_mock():
    """Factory for creating async iterator mocks."""
    return AsyncIteratorMock


# ==============================================================================
# Graph state fixtures for time travel testing
# ==============================================================================

@pytest.fixture
def mock_state_history():
    """Create mock state history for time travel tests."""
    states = []
    for i in range(5):
        mock_state = MagicMock()
        mock_state.config = {"configurable": {"checkpoint_id": f"cp-{i}"}}
        mock_state.parent_config = {"configurable": {"checkpoint_id": f"cp-{i-1}"}} if i > 0 else None
        mock_state.next = ["execute_test"] if i < 4 else None
        mock_state.values = {
            "iteration": i,
            "passed_count": i,
            "failed_count": 0,
            "current_test_index": i,
            "error": None,
            "should_continue": True,
        }
        states.append(mock_state)
    return states


@pytest.fixture
def mock_langgraph_app():
    """Create a mock LangGraph compiled app."""
    mock_app = AsyncMock()
    mock_app.aget_state = AsyncMock(return_value=None)
    mock_app.aupdate_state = AsyncMock()
    mock_app.ainvoke = AsyncMock(return_value={})

    async def mock_empty_history(*args, **kwargs):
        return
        yield

    mock_app.aget_state_history = mock_empty_history
    mock_app.astream = MagicMock(return_value=AsyncIteratorMock([]))
    return mock_app


# ==============================================================================
# Settings fixtures
# ==============================================================================

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.anthropic_api_key = MagicMock()
    settings.anthropic_api_key.get_secret_value.return_value = "sk-ant-test-key"
    settings.cost_limit_per_run = 10.0
    settings.cost_limit_per_test = 1.0
    settings.self_heal_enabled = True
    settings.self_heal_confidence_threshold = 0.8
    settings.require_healing_approval = False
    settings.require_human_approval_for_healing = False
    settings.require_test_plan_approval = False
    settings.browser_worker_url = "https://test-worker.example.com"
    return settings
