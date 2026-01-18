"""Tests for the Feature Mesh Integration module.

This module tests the FeatureMeshIntegration class which connects
Discovery to Visual AI and Self-Healing systems.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.discovery.feature_mesh import (
    FeatureMeshConfig,
    FeatureMeshIntegration,
    get_feature_mesh,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = AsyncMock()
    client.select = AsyncMock(return_value={"data": []})
    client.insert = AsyncMock(return_value={"data": [{"id": "test-id"}]})
    client.update = AsyncMock(return_value={"data": [{}]})
    return client


@pytest.fixture
def feature_mesh_config():
    """Create a FeatureMeshConfig for testing."""
    return FeatureMeshConfig(
        auto_create_baselines=True,
        share_selectors=True,
        min_selector_stability=0.5,
        max_alternatives_per_element=5,
    )


@pytest.fixture
def sample_pages():
    """Create sample discovered pages."""
    return [
        {
            "url": "https://example.com/login",
            "title": "Login Page",
            "page_title": "Login Page",
            "category": "auth_login",
            "actions": [{"selector": "#username"}, {"selector": "#password"}],
        },
        {
            "url": "https://example.com/dashboard",
            "title": "Dashboard",
            "category": "dashboard",
            "actions": [{"selector": ".widget"}],
        },
        {
            "url": "https://example.com/settings",
            "category": "settings",
            "actions": [],
        },
    ]


@pytest.fixture
def sample_elements():
    """Create sample discovered elements."""
    return [
        {
            "selector": "#login-button",
            "xpath": "//button[@id='login-button']",
            "category": "authentication",
            "label": "Login",
            "tag_name": "button",
            "page_url": "https://example.com/login",
            "aria_label": "Sign in to your account",
            "role": "button",
            "html_attributes": {
                "data-testid": "login-btn",
            },
        },
        {
            "selector": "#username",
            "alternative_selectors": ["[name='username']", "[data-field='username']"],
            "category": "form",
            "label": "Username",
            "tag_name": "input",
            "page_url": "https://example.com/login",
        },
        {
            "selector": ".nav-home",
            "category": "navigation",
            "label": "Home",
            "tag_name": "a",
            "page_url": "https://example.com/",
        },
    ]


# ==============================================================================
# Test FeatureMeshConfig
# ==============================================================================


class TestFeatureMeshConfig:
    """Tests for FeatureMeshConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeatureMeshConfig()
        assert config.auto_create_baselines is True
        assert config.share_selectors is True
        assert config.min_selector_stability == 0.5
        assert config.max_alternatives_per_element == 5
        assert config.async_processing is True
        assert config.baseline_browser == "chromium"

    def test_default_viewports(self):
        """Test default viewport configurations."""
        config = FeatureMeshConfig()
        assert len(config.baseline_viewports) == 2
        assert config.baseline_viewports[0]["name"] == "mobile"
        assert config.baseline_viewports[1]["name"] == "desktop"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FeatureMeshConfig(
            auto_create_baselines=False,
            share_selectors=False,
            min_selector_stability=0.8,
            max_alternatives_per_element=10,
        )
        assert config.auto_create_baselines is False
        assert config.share_selectors is False
        assert config.min_selector_stability == 0.8
        assert config.max_alternatives_per_element == 10


# ==============================================================================
# Test FeatureMeshIntegration Initialization
# ==============================================================================


class TestFeatureMeshIntegrationInit:
    """Tests for FeatureMeshIntegration initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration()
            assert integration.config is not None
            assert isinstance(integration.config, FeatureMeshConfig)

    def test_init_custom_config(self, feature_mesh_config):
        """Test initialization with custom config."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration(config=feature_mesh_config)
            assert integration.config == feature_mesh_config


# ==============================================================================
# Test Baseline Name Generation
# ==============================================================================


class TestBaselineNameGeneration:
    """Tests for baseline name generation."""

    def test_generate_baseline_name_with_title(self):
        """Test name generation with title."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration()

            name = integration._generate_baseline_name(
                "https://example.com/login",
                "Login Page",
                "auth_login",
            )
            assert "[Login]" in name
            assert "Login Page" in name

    def test_generate_baseline_name_without_title(self):
        """Test name generation without title uses URL path."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration()

            name = integration._generate_baseline_name(
                "https://example.com/users/profile",
                None,
                "other",
            )
            assert "users" in name or "profile" in name

    def test_generate_baseline_name_root_url(self):
        """Test name generation for root URL without title."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration()

            name = integration._generate_baseline_name(
                "https://example.com/",
                None,
                "landing",
            )
            assert "example.com" in name

    def test_generate_baseline_name_truncates_long_title(self):
        """Test that long titles are truncated."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration()

            long_title = "A" * 100
            name = integration._generate_baseline_name(
                "https://example.com/page",
                long_title,
                "other",
            )
            # Title should be truncated to 50 characters
            assert len(name.replace("[", "").replace("]", "").split(" ")[0]) <= 51

    def test_generate_baseline_name_categories(self):
        """Test category prefixes in baseline names."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()
            integration = FeatureMeshIntegration()

            test_cases = [
                ("auth_login", "[Login]"),
                ("auth_signup", "[Signup]"),
                ("dashboard", "[Dashboard]"),
                ("settings", "[Settings]"),
                ("profile", "[Profile]"),
                ("checkout", "[Checkout]"),
                ("landing", "[Landing]"),
            ]

            for category, expected_prefix in test_cases:
                name = integration._generate_baseline_name(
                    "https://example.com/page",
                    "Test Page",
                    category,
                )
                assert expected_prefix in name, f"Failed for category {category}"


# ==============================================================================
# Test Create Baselines from Discovery
# ==============================================================================


class TestCreateBaselinesFromDiscovery:
    """Tests for creating baselines from discovery."""

    @pytest.mark.asyncio
    async def test_create_baselines_success(self, mock_supabase, sample_pages):
        """Test successful baseline creation."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.create_baselines_from_discovery(
                session_id="session-123",
                project_id="project-456",
                pages=sample_pages,
            )

            assert result["session_id"] == "session-123"
            assert result["project_id"] == "project-456"
            assert result["total_pages"] == 3
            assert result["baselines_created"] == 3

    @pytest.mark.asyncio
    async def test_create_baselines_skips_existing(self, mock_supabase, sample_pages):
        """Test that existing baselines are skipped."""
        # Mock that first page already has a baseline
        async def mock_select(table, filters=None):
            if filters and "eq.https://example.com/login" in str(filters):
                return {"data": [{"id": "existing-baseline"}]}
            return {"data": []}

        mock_supabase.select = mock_select

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.create_baselines_from_discovery(
                session_id="session-123",
                project_id="project-456",
                pages=sample_pages,
            )

            assert result["baselines_skipped"] == 1
            assert result["baselines_created"] == 2

    @pytest.mark.asyncio
    async def test_create_baselines_handles_errors(self, mock_supabase, sample_pages):
        """Test error handling during baseline creation."""
        async def mock_insert(table, record):
            if "login" in record.get("url", ""):
                return {"error": "Database error"}
            return {"data": [{"id": "new-baseline"}]}

        mock_supabase.insert = mock_insert

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.create_baselines_from_discovery(
                session_id="session-123",
                project_id="project-456",
                pages=sample_pages,
            )

            assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_create_baselines_skips_pages_without_url(self, mock_supabase):
        """Test that pages without URL are skipped."""
        pages = [{"title": "No URL Page"}]

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.create_baselines_from_discovery(
                session_id="session-123",
                project_id="project-456",
                pages=pages,
            )

            assert result["baselines_created"] == 0


# ==============================================================================
# Test Extract Alternative Selectors
# ==============================================================================


class TestExtractAlternativeSelectors:
    """Tests for extracting alternative selectors."""

    def test_extract_explicit_alternatives(self):
        """Test extraction of explicit alternative selectors."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#main",
                "alternative_selectors": [".main-class", "[data-id='main']"],
            }

            alternatives = integration._extract_alternative_selectors(element)
            assert len(alternatives) >= 2
            selectors = [a["selector"] for a in alternatives]
            assert ".main-class" in selectors
            assert "[data-id='main']" in selectors

    def test_extract_xpath_alternative(self):
        """Test extraction of XPath selector."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#btn",
                "xpath": "//button[@id='btn']",
            }

            alternatives = integration._extract_alternative_selectors(element)
            xpath_alts = [a for a in alternatives if a["strategy"] == "xpath"]
            assert len(xpath_alts) == 1
            assert xpath_alts[0]["selector"] == "//button[@id='btn']"

    def test_extract_text_content_alternative(self):
        """Test extraction of text content selector."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#btn",
                "label": "Click Me",
                "tag_name": "button",
            }

            alternatives = integration._extract_alternative_selectors(element)
            text_alts = [a for a in alternatives if a["strategy"] == "text_content"]
            assert len(text_alts) == 1
            assert "Click Me" in text_alts[0]["selector"]

    def test_extract_aria_label_alternative(self):
        """Test extraction of aria-label selector."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#btn",
                "aria_label": "Submit form",
            }

            alternatives = integration._extract_alternative_selectors(element)
            aria_alts = [a for a in alternatives if a["strategy"] == "aria_label"]
            assert len(aria_alts) == 1
            assert "Submit form" in aria_alts[0]["selector"]

    def test_extract_role_label_alternative(self):
        """Test extraction of role + label selector."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#btn",
                "role": "button",
                "label": "Submit",
            }

            alternatives = integration._extract_alternative_selectors(element)
            role_alts = [a for a in alternatives if a["strategy"] == "role_label"]
            assert len(role_alts) == 1
            assert "button" in role_alts[0]["selector"]
            assert "Submit" in role_alts[0]["selector"]

    def test_extract_data_attribute_alternative(self):
        """Test extraction of data attribute selector."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#btn",
                "html_attributes": {
                    "data-testid": "submit-button",
                },
            }

            alternatives = integration._extract_alternative_selectors(element)
            data_alts = [a for a in alternatives if a["strategy"] == "data_attribute"]
            assert len(data_alts) == 1
            assert "data-testid" in data_alts[0]["selector"]

    def test_extract_limits_alternatives(self):
        """Test that alternatives are limited to max_alternatives_per_element."""
        config = FeatureMeshConfig(max_alternatives_per_element=2)

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration(config=config)
            element = {
                "selector": "#btn",
                "alternative_selectors": ["a", "b", "c", "d", "e"],
                "xpath": "//xpath",
                "label": "Label",
                "aria_label": "Aria",
            }

            alternatives = integration._extract_alternative_selectors(element)
            assert len(alternatives) <= 2

    def test_extract_sorts_by_confidence(self):
        """Test that alternatives are sorted by confidence."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            element = {
                "selector": "#btn",
                "aria_label": "High confidence",  # Should have high confidence
                "xpath": "//low",  # Should have lower confidence
                "label": "Medium",
                "tag_name": "button",
            }

            alternatives = integration._extract_alternative_selectors(element)
            if len(alternatives) >= 2:
                # First should have higher confidence than last
                assert alternatives[0]["confidence"] >= alternatives[-1]["confidence"]


# ==============================================================================
# Test Selector Fingerprint Generation
# ==============================================================================


class TestSelectorFingerprint:
    """Tests for selector fingerprint generation."""

    def test_generate_fingerprint(self):
        """Test fingerprint generation."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            fingerprint = integration._generate_selector_fingerprint(
                "#primary",
                [{"selector": "#alt1"}, {"selector": "#alt2"}],
            )

            assert len(fingerprint) == 16
            # Should be deterministic
            fingerprint2 = integration._generate_selector_fingerprint(
                "#primary",
                [{"selector": "#alt1"}, {"selector": "#alt2"}],
            )
            assert fingerprint == fingerprint2

    def test_fingerprint_different_for_different_input(self):
        """Test that different inputs produce different fingerprints."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            integration = FeatureMeshIntegration()
            fp1 = integration._generate_selector_fingerprint(
                "#primary1",
                [{"selector": "#alt1"}],
            )
            fp2 = integration._generate_selector_fingerprint(
                "#primary2",
                [{"selector": "#alt2"}],
            )
            assert fp1 != fp2


# ==============================================================================
# Test Share Selectors with Self-Healer
# ==============================================================================


class TestShareSelectorsWithSelfHealer:
    """Tests for sharing selectors with self-healer."""

    @pytest.mark.asyncio
    async def test_share_selectors_success(self, mock_supabase, sample_elements):
        """Test successful selector sharing."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.share_selectors_with_self_healer(
                session_id="session-123",
                project_id="project-456",
                elements=sample_elements,
            )

            assert result["session_id"] == "session-123"
            assert result["total_elements"] == 3
            assert result["selectors_shared"] > 0

    @pytest.mark.asyncio
    async def test_share_selectors_skips_without_selector(self, mock_supabase):
        """Test that elements without selector are skipped."""
        elements = [{"label": "No selector element"}]

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.share_selectors_with_self_healer(
                session_id="session-123",
                project_id="project-456",
                elements=elements,
            )

            assert result["selectors_shared"] == 0


# ==============================================================================
# Test Record Healing Feedback
# ==============================================================================


class TestRecordHealingFeedback:
    """Tests for recording healing feedback."""

    @pytest.mark.asyncio
    async def test_record_feedback_success(self, mock_supabase):
        """Test successful feedback recording."""
        existing_record = {
            "id": "record-123",
            "usage_count": 5,
            "success_count": 4,
            "alternatives": [
                {"selector": "#alt", "confidence": 0.7, "usage_count": 2}
            ],
        }

        async def mock_select(table, filters=None):
            return {"data": [existing_record]}

        mock_supabase.select = mock_select

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.record_healing_feedback(
                primary_selector="#primary",
                used_alternative="#alt",
                success=True,
                project_id="project-456",
            )

            assert result is True
            mock_supabase.update.assert_called()

    @pytest.mark.asyncio
    async def test_record_feedback_not_found(self, mock_supabase):
        """Test feedback recording when selector not found."""
        async def mock_select(table, filters=None):
            return {"data": []}

        mock_supabase.select = mock_select

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.record_healing_feedback(
                primary_selector="#unknown",
                used_alternative="#alt",
                success=True,
                project_id="project-456",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_record_feedback_adjusts_confidence(self, mock_supabase):
        """Test that confidence is adjusted based on success."""
        existing_record = {
            "id": "record-123",
            "usage_count": 5,
            "success_count": 4,
            "alternatives": [
                {"selector": "#alt", "confidence": 0.7}
            ],
        }

        async def mock_select(table, filters=None):
            return {"data": [existing_record]}

        mock_supabase.select = mock_select

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            # Test success increases confidence
            await integration.record_healing_feedback(
                primary_selector="#primary",
                used_alternative="#alt",
                success=True,
                project_id="project-456",
            )

            # Verify update was called with increased confidence
            call_args = mock_supabase.update.call_args
            update_data = call_args[0][2]
            alt = update_data["alternatives"][0]
            assert alt["confidence"] > 0.7


# ==============================================================================
# Test Process Discovery Completion
# ==============================================================================


class TestProcessDiscoveryCompletion:
    """Tests for processing discovery completion."""

    @pytest.mark.asyncio
    async def test_process_completion_runs_integrations(
        self, mock_supabase, sample_pages, sample_elements
    ):
        """Test that completion processing runs all integrations."""
        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.process_discovery_completion(
                session_id="session-123",
                project_id="project-456",
                pages=sample_pages,
                elements=sample_elements,
            )

            assert result["session_id"] == "session-123"
            assert "visual_ai" in result["integrations"]
            assert "self_healing" in result["integrations"]
            assert "processed_at" in result

    @pytest.mark.asyncio
    async def test_process_completion_disabled_baselines(
        self, mock_supabase, sample_pages, sample_elements
    ):
        """Test completion with baselines disabled."""
        config = FeatureMeshConfig(auto_create_baselines=False)

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration(config=config)

            result = await integration.process_discovery_completion(
                session_id="session-123",
                project_id="project-456",
                pages=sample_pages,
                elements=sample_elements,
            )

            assert "visual_ai" not in result["integrations"]

    @pytest.mark.asyncio
    async def test_process_completion_disabled_selectors(
        self, mock_supabase, sample_pages, sample_elements
    ):
        """Test completion with selector sharing disabled."""
        config = FeatureMeshConfig(share_selectors=False)

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration(config=config)

            result = await integration.process_discovery_completion(
                session_id="session-123",
                project_id="project-456",
                pages=sample_pages,
                elements=sample_elements,
            )

            assert "self_healing" not in result["integrations"]

    @pytest.mark.asyncio
    async def test_process_completion_handles_errors(self, mock_supabase):
        """Test that errors in integrations are handled."""
        async def mock_select_error(table, filters=None):
            raise Exception("Database error")

        mock_supabase.select = mock_select_error

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = mock_supabase

            integration = FeatureMeshIntegration()

            result = await integration.process_discovery_completion(
                session_id="session-123",
                project_id="project-456",
                pages=[{"url": "https://example.com"}],
                elements=[{"selector": "#test"}],
            )

            # Should have error in the result
            assert "error" in result["integrations"].get("visual_ai", {})


# ==============================================================================
# Test Singleton Instance
# ==============================================================================


class TestSingletonInstance:
    """Tests for the singleton get_feature_mesh function."""

    def test_get_feature_mesh_creates_instance(self):
        """Test that get_feature_mesh creates an instance."""
        # Reset the singleton
        import src.discovery.feature_mesh as module
        module._feature_mesh = None

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            instance = get_feature_mesh()
            assert instance is not None
            assert isinstance(instance, FeatureMeshIntegration)

    def test_get_feature_mesh_returns_same_instance(self):
        """Test that get_feature_mesh returns the same instance."""
        import src.discovery.feature_mesh as module
        module._feature_mesh = None

        with patch(
            "src.discovery.feature_mesh.get_supabase_client"
        ) as mock_get_client:
            mock_get_client.return_value = MagicMock()

            instance1 = get_feature_mesh()
            instance2 = get_feature_mesh()
            assert instance1 is instance2
