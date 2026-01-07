"""Shared fixtures for E2E Testing Agent tests."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Generator, AsyncGenerator

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
        "markers", "requires_tree_sitter: mark test as requiring tree-sitter"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring real API keys"
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
