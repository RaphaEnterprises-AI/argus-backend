"""Tests for Auto-Discovery module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_discovery_result_creation(self):
        """Test creating a discovery result."""
        from src.agents.auto_discovery import DiscoveredFlow, DiscoveredPage, DiscoveryResult

        result = DiscoveryResult(
            app_url="http://localhost:3000",
            pages_discovered=[
                DiscoveredPage(
                    url="/login",
                    title="Login Page",
                    description="User login page",
                    elements=[],
                )
            ],
            flows_discovered=[
                DiscoveredFlow(
                    id="login-flow",
                    name="Login Flow",
                    description="User login flow",
                    start_url="/login",
                    steps=[],
                )
            ],
            suggested_tests=[
                {
                    "id": "test-login",
                    "name": "Login Test",
                    "steps": [],
                }
            ],
        )

        assert len(result.pages_discovered) == 1
        assert len(result.flows_discovered) == 1
        assert result.app_url == "http://localhost:3000"


class TestDiscoveredPage:
    """Tests for DiscoveredPage dataclass."""

    def test_page_creation(self):
        """Test creating a discovered page."""
        from src.agents.auto_discovery import DiscoveredElement, DiscoveredPage

        page = DiscoveredPage(
            url="/products",
            title="Products Page",
            description="Product listing page",
            elements=[
                DiscoveredElement(
                    type="button",
                    text="Add to Cart",
                    selector="#add-to-cart",
                    action="click",
                    purpose="Add item to cart",
                )
            ],
            forms=[{"id": "search-form", "fields": ["query"]}],
            links=["/product/1", "/product/2"],
        )

        assert page.url == "/products"
        assert len(page.elements) == 1
        assert page.elements[0].type == "button"


class TestDiscoveredElement:
    """Tests for DiscoveredElement dataclass."""

    def test_element_creation(self):
        """Test creating a discovered element."""
        from src.agents.auto_discovery import DiscoveredElement

        element = DiscoveredElement(
            type="button",
            text="Submit",
            selector="button.submit",
            action="click",
            purpose="Submit the form",
        )

        assert element.type == "button"
        assert element.selector == "button.submit"


class TestDiscoveredFlow:
    """Tests for DiscoveredFlow dataclass."""

    def test_flow_creation(self):
        """Test creating a discovered flow."""
        from src.agents.auto_discovery import DiscoveredFlow

        flow = DiscoveredFlow(
            id="checkout-flow",
            name="Checkout Flow",
            description="Complete purchase flow",
            start_url="/cart",
            steps=[
                {"action": "click", "target": "#checkout"},
                {"action": "fill", "target": "#address", "value": "123 Main St"},
            ],
            priority="high",
            category="user_journey",
        )

        assert flow.id == "checkout-flow"
        assert flow.name == "Checkout Flow"
        assert len(flow.steps) == 2
        assert flow.priority == "high"


class TestAutoDiscovery:
    """Tests for AutoDiscovery class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.auto_discovery.get_settings") as mock:
            settings = MagicMock()
            settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def discovery(self, mock_settings):
        """Create AutoDiscovery with mocked dependencies."""
        with patch("src.agents.auto_discovery.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            from src.agents.auto_discovery import AutoDiscovery
            d = AutoDiscovery(app_url="http://localhost:3000")
            d.client = mock_client
            return d

    def test_init(self, discovery):
        """Test initialization."""
        assert discovery.app_url == "http://localhost:3000"
        assert discovery.max_pages > 0
        assert discovery.max_depth > 0

    @pytest.mark.asyncio
    async def test_discover_basic(self, discovery):
        """Test basic discovery with simulated mode (no Playwright)."""
        # Mock Claude response for simulated discovery
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "pages": [
        {
            "url": "/",
            "title": "Home",
            "description": "Homepage",
            "elements": ["button: Login"],
            "links": ["/login"]
        },
        {
            "url": "/login",
            "title": "Login",
            "description": "Login page",
            "elements": ["input: Email", "input: Password", "button: Submit"],
            "links": ["/signup"]
        }
    ]
}
```""")]
        mock_response.usage = MagicMock(input_tokens=1000, output_tokens=200)
        discovery.client.messages.create = MagicMock(return_value=mock_response)

        # Mock the import to trigger simulation mode
        with patch.dict('sys.modules', {'src.tools.playwright_tools': None}):
            with patch.object(discovery, '_simulate_discovery', new_callable=AsyncMock):
                with patch.object(discovery, '_analyze_flows', new_callable=AsyncMock) as mock_flows:
                    with patch.object(discovery, '_generate_test_suggestions', new_callable=AsyncMock) as mock_tests:
                        mock_flows.return_value = []
                        mock_tests.return_value = [{"id": "test-1", "name": "Login Test"}]

                        result = await discovery.discover(start_paths=["/"])

        assert result is not None
        assert hasattr(result, 'pages_discovered')
        assert hasattr(result, 'suggested_tests')


class TestQuickDiscover:
    """Tests for QuickDiscover class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.auto_discovery.get_settings") as mock:
            settings = MagicMock()
            settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def quick_discover(self, mock_settings):
        """Create QuickDiscover with mocked dependencies."""
        with patch("src.agents.auto_discovery.anthropic.Anthropic"):
            from src.agents.auto_discovery import QuickDiscover
            return QuickDiscover(app_url="http://localhost:3000")

    def test_init(self, quick_discover):
        """Test initialization."""
        assert quick_discover.app_url == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_discover_login_flow(self, quick_discover):
        """Test discovering login flow."""
        # Mock the internal discovery
        from src.agents.auto_discovery import DiscoveryResult

        mock_result = DiscoveryResult(
            app_url="http://localhost:3000",
            suggested_tests=[
                {"id": "login-1", "name": "Login Test", "steps": []}
            ]
        )

        with patch.object(quick_discover.discovery, 'discover', new_callable=AsyncMock) as mock:
            mock.return_value = mock_result
            result = await quick_discover.discover_login_flow()

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_discover_critical_flows(self, quick_discover):
        """Test discovering critical flows."""
        from src.agents.auto_discovery import DiscoveryResult

        mock_result = DiscoveryResult(
            app_url="http://localhost:3000",
            suggested_tests=[
                {"id": "critical-1", "name": "Critical Test", "priority": "high"},
                {"id": "critical-2", "name": "Another Critical", "priority": "high"},
            ]
        )

        with patch.object(quick_discover.discovery, 'discover', new_callable=AsyncMock) as mock:
            mock.return_value = mock_result
            result = await quick_discover.discover_critical_flows()

        assert isinstance(result, list)
