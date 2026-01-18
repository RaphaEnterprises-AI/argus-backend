"""Tests for auto discovery module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.auto_discovery import (
    AutoDiscovery,
    DiscoveredElement,
    DiscoveredFlow,
    DiscoveredPage,
    DiscoveryResult,
    QuickDiscover,
    create_auto_discovery,
)


class TestDiscoveredElement:
    """Tests for DiscoveredElement dataclass."""

    def test_create_element(self):
        """Test creating a discovered element."""
        element = DiscoveredElement(
            type="button",
            text="Submit",
            selector="#submit-btn",
            action="click",
            purpose="Submit the form",
        )
        assert element.type == "button"
        assert element.text == "Submit"
        assert element.selector == "#submit-btn"
        assert element.action == "click"
        assert element.purpose == "Submit the form"


class TestDiscoveredPage:
    """Tests for DiscoveredPage dataclass."""

    def test_create_page_minimal(self):
        """Test creating a page with minimal data."""
        page = DiscoveredPage(
            url="https://example.com/login",
            title="Login Page",
            description="User login page",
        )
        assert page.url == "https://example.com/login"
        assert page.title == "Login Page"
        assert page.description == "User login page"
        assert page.elements == []
        assert page.screenshot is None
        assert page.forms == []
        assert page.links == []
        assert page.user_flows == []

    def test_create_page_full(self):
        """Test creating a page with all data."""
        element = DiscoveredElement(
            type="button",
            text="Login",
            selector="#login",
            action="click",
            purpose="Submit login",
        )
        page = DiscoveredPage(
            url="https://example.com/login",
            title="Login Page",
            description="User login page",
            elements=[element],
            screenshot="base64data",
            forms=[{"action": "/login", "inputs": []}],
            links=["/signup", "/forgot-password"],
            user_flows=["Authentication Flow"],
        )
        assert len(page.elements) == 1
        assert page.screenshot == "base64data"
        assert len(page.forms) == 1
        assert len(page.links) == 2
        assert len(page.user_flows) == 1


class TestDiscoveredFlow:
    """Tests for DiscoveredFlow dataclass."""

    def test_create_flow_minimal(self):
        """Test creating a flow with minimal data."""
        flow = DiscoveredFlow(
            id="flow-1",
            name="Login Flow",
            description="User authentication",
            start_url="/login",
        )
        assert flow.id == "flow-1"
        assert flow.name == "Login Flow"
        assert flow.description == "User authentication"
        assert flow.start_url == "/login"
        assert flow.steps == []
        assert flow.priority == "medium"
        assert flow.category == "user_journey"

    def test_create_flow_full(self):
        """Test creating a flow with all data."""
        flow = DiscoveredFlow(
            id="flow-2",
            name="Checkout Flow",
            description="Complete purchase",
            start_url="/cart",
            steps=[
                {"page": "/cart", "action": "view cart"},
                {"page": "/checkout", "action": "enter details"},
                {"page": "/confirmation", "action": "verify purchase"},
            ],
            priority="critical",
            category="checkout",
        )
        assert len(flow.steps) == 3
        assert flow.priority == "critical"
        assert flow.category == "checkout"


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_create_result_minimal(self):
        """Test creating a result with minimal data."""
        result = DiscoveryResult(app_url="https://example.com")

        assert result.app_url == "https://example.com"
        assert result.pages_discovered == []
        assert result.flows_discovered == []
        assert result.suggested_tests == []
        assert result.coverage_summary == {}
        assert result.timestamp is not None

    def test_create_result_full(self):
        """Test creating a result with all data."""
        page = DiscoveredPage(
            url="https://example.com",
            title="Home",
            description="Home page",
        )
        flow = DiscoveredFlow(
            id="flow-1",
            name="Flow",
            description="Desc",
            start_url="/",
        )
        result = DiscoveryResult(
            app_url="https://example.com",
            pages_discovered=[page],
            flows_discovered=[flow],
            suggested_tests=[{"name": "Test 1"}],
            coverage_summary={"pages": 1, "flows": 1},
        )
        assert len(result.pages_discovered) == 1
        assert len(result.flows_discovered) == 1
        assert len(result.suggested_tests) == 1


class TestAutoDiscovery:
    """Tests for AutoDiscovery class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.auto_discovery.get_settings") as mock:
            mock.return_value.anthropic_api_key = MagicMock()
            mock.return_value.anthropic_api_key.get_secret_value.return_value = "test_key"
            yield mock

    @pytest.fixture
    def discovery(self, mock_settings):
        """Create an AutoDiscovery instance."""
        with patch("src.agents.auto_discovery.anthropic.Anthropic"):
            return AutoDiscovery(
                app_url="https://example.com",
                max_pages=10,
                max_depth=2,
            )

    def test_init(self, mock_settings):
        """Test AutoDiscovery initialization."""
        with patch("src.agents.auto_discovery.anthropic.Anthropic"):
            discovery = AutoDiscovery(
                app_url="https://example.com",
                max_pages=15,
                max_depth=3,
                model="claude-sonnet-4-5-20250514",
            )

            assert discovery.app_url == "https://example.com"
            assert discovery.base_domain == "example.com"
            assert discovery.max_pages == 15
            assert discovery.max_depth == 3
            assert discovery.model == "claude-sonnet-4-5-20250514"
            assert discovery.visited_urls == set()
            assert discovery.discovered_pages == []
            assert discovery.discovered_flows == []

    @pytest.mark.asyncio
    async def test_discover_calls_analysis_methods(self, discovery):
        """Test that discover calls analysis methods after crawling."""
        # Mock internal methods to verify they're called
        crawl_calls = []

        async def mock_crawl(page, url, depth):
            crawl_calls.append((url, depth))

        discovery._crawl_page = mock_crawl
        discovery._analyze_flows = AsyncMock(return_value=[])
        discovery._generate_test_suggestions = AsyncMock(return_value=[])

        # Mock the browser manager to just return a mock page
        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.page = mock_page
        mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
        mock_browser.__aexit__ = AsyncMock(return_value=None)

        with patch("src.tools.playwright_tools.BrowserManager", return_value=mock_browser):
            with patch("src.tools.playwright_tools.BrowserConfig"):
                result = await discovery.discover()

        discovery._analyze_flows.assert_called_once()
        discovery._generate_test_suggestions.assert_called_once()
        assert isinstance(result, DiscoveryResult)

    @pytest.mark.asyncio
    async def test_discover_with_focus_areas_calls_methods(self, discovery):
        """Test discovery with focus areas."""
        discovery._crawl_page = AsyncMock()
        discovery._analyze_flows = AsyncMock(return_value=[])
        discovery._generate_test_suggestions = AsyncMock(return_value=[])

        # Mock the browser manager
        mock_page = MagicMock()
        mock_browser = MagicMock()
        mock_browser.page = mock_page
        mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
        mock_browser.__aexit__ = AsyncMock(return_value=None)

        with patch("src.tools.playwright_tools.BrowserManager", return_value=mock_browser):
            with patch("src.tools.playwright_tools.BrowserConfig"):
                result = await discovery.discover(
                    start_paths=["/login"],
                    focus_areas=["authentication"],
                )

        discovery._analyze_flows.assert_called_once_with(["authentication"])
        assert isinstance(result, DiscoveryResult)

    def test_calculate_coverage(self, discovery):
        """Test coverage calculation."""
        discovery.discovered_pages = [
            DiscoveredPage(
                url="https://example.com",
                title="Home",
                description="Home page",
                forms=[{"action": "/submit"}],
                elements=[
                    DiscoveredElement(type="button", text="Click", selector="#btn", action="click", purpose=""),
                ],
            ),
        ]
        discovery.discovered_flows = [
            DiscoveredFlow(id="1", name="Flow", description="", start_url="/", priority="critical"),
            DiscoveredFlow(id="2", name="Flow2", description="", start_url="/", priority="high"),
        ]

        coverage = discovery._calculate_coverage()

        assert coverage["pages_discovered"] == 1
        assert coverage["flows_identified"] == 2
        assert coverage["forms_found"] == 1
        assert coverage["interactive_elements"] == 1
        assert coverage["critical_flows"] == 1
        assert coverage["coverage_score"] == 20  # 2 flows * 10

    def test_to_test_specs(self, discovery):
        """Test converting result to test specs."""
        result = DiscoveryResult(
            app_url="https://example.com",
            suggested_tests=[
                {"id": "test-1", "name": "Login Test"},
                {"id": "test-2", "name": "Signup Test"},
            ],
        )

        specs = discovery.to_test_specs(result)

        assert len(specs) == 2
        assert specs[0]["id"] == "test-1"

    @pytest.mark.asyncio
    async def test_analyze_flows_empty(self, discovery):
        """Test analyzing flows with no pages."""
        result = await discovery._analyze_flows()
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_flows_with_pages(self, discovery):
        """Test analyzing flows with pages."""
        discovery.discovered_pages = [
            DiscoveredPage(
                url="https://example.com/login",
                title="Login",
                description="Login page",
                forms=[{"action": "/login"}],
                links=["/signup"],
                user_flows=["Authentication"],
            ),
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"flows": [{"id": "flow-1", "name": "Login Flow", "description": "User login", "start_url": "/login", "steps": [], "priority": "critical", "category": "authentication"}]}')]

        discovery.client.messages.create = MagicMock(return_value=mock_response)

        flows = await discovery._analyze_flows(focus_areas=["authentication"])

        assert len(flows) == 1
        assert flows[0].name == "Login Flow"
        assert flows[0].priority == "critical"

    @pytest.mark.asyncio
    async def test_analyze_flows_error(self, discovery):
        """Test analyzing flows with API error."""
        discovery.discovered_pages = [
            DiscoveredPage(url="https://example.com", title="Home", description="Home"),
        ]

        discovery.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        flows = await discovery._analyze_flows()

        assert flows == []

    @pytest.mark.asyncio
    async def test_generate_test_suggestions_empty(self, discovery):
        """Test generating suggestions with no flows."""
        result = await discovery._generate_test_suggestions()
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_test_suggestions_with_flows(self, discovery):
        """Test generating suggestions with flows."""
        discovery.discovered_flows = [
            DiscoveredFlow(
                id="flow-1",
                name="Login Flow",
                description="User login",
                start_url="/login",
                steps=[{"page": "/login", "action": "fill credentials"}],
                priority="critical",
            ),
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"tests": [{"id": "test-1", "name": "Login Test", "description": "Test login", "priority": "critical", "steps": []}]}')]

        discovery.client.messages.create = MagicMock(return_value=mock_response)

        tests = await discovery._generate_test_suggestions()

        assert len(tests) == 1
        assert tests[0]["name"] == "Login Test"

    @pytest.mark.asyncio
    async def test_generate_test_suggestions_error(self, discovery):
        """Test generating suggestions with API error."""
        discovery.discovered_flows = [
            DiscoveredFlow(id="1", name="Flow", description="", start_url="/"),
        ]

        discovery.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        tests = await discovery._generate_test_suggestions()

        assert tests == []

    @pytest.mark.asyncio
    async def test_simulate_discovery(self, discovery):
        """Test simulated discovery."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"pages": [{"url": "/login", "title": "Login", "description": "Login page", "links": ["/signup"]}]}')]

        discovery.client.messages.create = MagicMock(return_value=mock_response)

        await discovery._simulate_discovery(["/"])

        assert len(discovery.discovered_pages) == 1
        assert "login" in discovery.discovered_pages[0].url

    @pytest.mark.asyncio
    async def test_analyze_page_with_vision_success(self, discovery):
        """Test analyzing page with vision."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"description": "Login page", "page_type": "login", "key_actions": ["login"], "possible_flows": ["Authentication"], "element_purposes": {"Login": "Submit login"}, "test_priority": "critical"}')]

        discovery.client.messages.create = MagicMock(return_value=mock_response)

        result = await discovery._analyze_page_with_vision(
            screenshot_b64="base64data",
            title="Login",
            url="https://example.com/login",
            elements=[],
        )

        assert result["description"] == "Login page"
        assert result["page_type"] == "login"

    @pytest.mark.asyncio
    async def test_analyze_page_with_vision_json_in_code_block(self, discovery):
        """Test analyzing page with JSON in code block."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"description": "Home page", "possible_flows": []}\n```')]

        discovery.client.messages.create = MagicMock(return_value=mock_response)

        result = await discovery._analyze_page_with_vision(
            screenshot_b64="base64data",
            title="Home",
            url="https://example.com",
            elements=[],
        )

        assert result["description"] == "Home page"

    @pytest.mark.asyncio
    async def test_analyze_page_with_vision_error(self, discovery):
        """Test analyzing page with vision error."""
        discovery.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        result = await discovery._analyze_page_with_vision(
            screenshot_b64="base64data",
            title="Home",
            url="https://example.com",
            elements=[],
        )

        assert result["description"] == "Home"
        assert result["possible_flows"] == []


class TestQuickDiscover:
    """Tests for QuickDiscover class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.auto_discovery.get_settings") as mock:
            mock.return_value.anthropic_api_key = MagicMock()
            mock.return_value.anthropic_api_key.get_secret_value.return_value = "test_key"
            yield mock

    @pytest.fixture
    def quick_discover(self, mock_settings):
        """Create a QuickDiscover instance."""
        with patch("src.agents.auto_discovery.anthropic.Anthropic"):
            return QuickDiscover("https://example.com")

    def test_init(self, mock_settings):
        """Test QuickDiscover initialization."""
        with patch("src.agents.auto_discovery.anthropic.Anthropic"):
            qd = QuickDiscover("https://example.com")

            assert qd.app_url == "https://example.com"
            assert qd.discovery.max_pages == 5
            assert qd.discovery.max_depth == 1

    @pytest.mark.asyncio
    async def test_discover_login_flow(self, quick_discover):
        """Test discovering login flow."""
        mock_result = DiscoveryResult(
            app_url="https://example.com",
            suggested_tests=[
                {"name": "User Login Test"},
                {"name": "Signup Test"},
            ],
        )
        quick_discover.discovery.discover = AsyncMock(return_value=mock_result)

        tests = await quick_discover.discover_login_flow()

        assert len(tests) == 1
        assert "login" in tests[0]["name"].lower()

    @pytest.mark.asyncio
    async def test_discover_signup_flow(self, quick_discover):
        """Test discovering signup flow."""
        mock_result = DiscoveryResult(
            app_url="https://example.com",
            suggested_tests=[
                {"name": "User Signup Test"},
                {"name": "Login Test"},
                {"name": "Register New User"},
            ],
        )
        quick_discover.discovery.discover = AsyncMock(return_value=mock_result)

        tests = await quick_discover.discover_signup_flow()

        assert len(tests) == 2  # signup and register

    @pytest.mark.asyncio
    async def test_discover_critical_flows(self, quick_discover):
        """Test discovering critical flows."""
        mock_result = DiscoveryResult(
            app_url="https://example.com",
            suggested_tests=[
                {"name": "Login Test", "priority": "critical"},
                {"name": "Checkout Test", "priority": "critical"},
                {"name": "Settings Test", "priority": "low"},
            ],
        )
        quick_discover.discovery.discover = AsyncMock(return_value=mock_result)

        tests = await quick_discover.discover_critical_flows()

        assert len(tests) == 2


class TestCreateAutoDiscovery:
    """Tests for create_auto_discovery factory function."""

    def test_create_auto_discovery(self):
        """Test factory function creates AutoDiscovery."""
        with patch("src.agents.auto_discovery.get_settings") as mock_settings, \
             patch("src.agents.auto_discovery.anthropic.Anthropic"):
            mock_settings.return_value.anthropic_api_key = MagicMock()
            mock_settings.return_value.anthropic_api_key.get_secret_value.return_value = "key"

            discovery = create_auto_discovery(
                app_url="https://example.com",
                max_pages=30,
            )

            assert discovery.app_url == "https://example.com"
            assert discovery.max_pages == 30
