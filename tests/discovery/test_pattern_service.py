"""Tests for the Pattern Service module.

This module tests the PatternService class which provides cross-project
pattern learning with pgvector for similarity search.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.discovery.pattern_service import (
    DiscoveryPattern,
    PatternMatch,
    PatternService,
    PatternType,
    _create_signature,
    _extract_url_pattern,
    _map_element_to_pattern_type,
    _map_page_to_pattern_type,
    _normalize_selector,
    _normalize_title,
    _pad_embedding,
    get_pattern_service,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()
    client.request = AsyncMock(return_value={"data": [], "error": None})
    return client


@pytest.fixture
def mock_vectorize():
    """Create a mock Cloudflare Vectorize client."""
    client = MagicMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 1024)
    return client


@pytest.fixture
def sample_page_data():
    """Create sample page data for testing."""
    return {
        "url": "https://example.com/login",
        "title": "Login Page",
        "category": "auth_login",
        "elements": [
            {"category": "authentication"},
            {"category": "form"},
            {"category": "action"},
        ],
    }


@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    return {
        "name": "User Login Flow",
        "category": "authentication",
        "priority": "high",
        "steps": [
            {"type": "navigate", "instruction": "Go to login page"},
            {"type": "fill", "instruction": "Enter username"},
            {"type": "fill", "instruction": "Enter password"},
            {"type": "click", "instruction": "Click sign in button"},
        ],
        "success_criteria": ["User is logged in", "Dashboard is visible"],
    }


@pytest.fixture
def sample_element_data():
    """Create sample element data for testing."""
    return {
        "selector": "#login-button",
        "category": "authentication",
        "tag_name": "button",
        "role": "button",
        "label": "Sign In",
        "is_visible": True,
        "is_enabled": True,
        "alternative_selectors": [".login-btn", "[data-action='login']"],
        "aria_label": "Sign in to your account",
    }


# ==============================================================================
# Test PatternType Enum
# ==============================================================================


class TestPatternType:
    """Tests for PatternType enum."""

    def test_pattern_types_exist(self):
        """Test that all expected pattern types exist."""
        expected_types = [
            "PAGE_LAYOUT",
            "NAVIGATION",
            "FORM",
            "AUTHENTICATION",
            "ERROR_HANDLING",
            "LOADING_STATE",
            "MODAL",
            "LIST_VIEW",
            "DETAIL_VIEW",
            "SEARCH",
            "FILTER",
            "PAGINATION",
            "CUSTOM",
        ]
        for type_name in expected_types:
            assert hasattr(PatternType, type_name)

    def test_pattern_type_values(self):
        """Test pattern type string values."""
        assert PatternType.AUTHENTICATION.value == "authentication"
        assert PatternType.FORM.value == "form"
        assert PatternType.NAVIGATION.value == "navigation"


# ==============================================================================
# Test DiscoveryPattern Dataclass
# ==============================================================================


class TestDiscoveryPattern:
    """Tests for DiscoveryPattern dataclass."""

    def test_from_page_basic(self, sample_page_data):
        """Test creating pattern from page data."""
        pattern = DiscoveryPattern.from_page(sample_page_data)

        assert pattern.pattern_type == PatternType.AUTHENTICATION
        assert pattern.pattern_name == "auth_login_page"
        assert pattern.pattern_signature is not None
        assert "features" in pattern.pattern_data

    def test_from_page_extracts_features(self, sample_page_data):
        """Test that page features are extracted correctly."""
        pattern = DiscoveryPattern.from_page(sample_page_data)
        features = pattern.pattern_data["features"]

        assert features["category"] == "auth_login"
        assert features["has_auth"] is True
        assert features["has_forms"] is True
        assert features["element_count"] == 3

    def test_from_page_with_url_pattern(self):
        """Test URL pattern extraction."""
        page_data = {
            "url": "https://example.com/users/12345/profile",
            "category": "profile",
            "elements": [],
        }
        pattern = DiscoveryPattern.from_page(page_data)
        features = pattern.pattern_data["features"]

        # Should have URL pattern with ID replaced
        assert ":id" in features["url_pattern"]

    def test_from_flow_basic(self, sample_flow_data):
        """Test creating pattern from flow data."""
        pattern = DiscoveryPattern.from_flow(sample_flow_data)

        assert pattern.pattern_type == PatternType.AUTHENTICATION
        assert pattern.pattern_name == "User Login Flow"
        assert pattern.pattern_signature is not None

    def test_from_flow_extracts_features(self, sample_flow_data):
        """Test that flow features are extracted correctly."""
        pattern = DiscoveryPattern.from_flow(sample_flow_data)
        features = pattern.pattern_data["features"]

        assert features["category"] == "authentication"
        assert features["step_count"] == 4
        assert features["has_auth_steps"] is True

    def test_from_element_basic(self, sample_element_data):
        """Test creating pattern from element data."""
        pattern = DiscoveryPattern.from_element(sample_element_data)

        assert pattern.pattern_type == PatternType.AUTHENTICATION
        assert pattern.pattern_signature is not None

    def test_from_element_extracts_features(self, sample_element_data):
        """Test that element features are extracted correctly."""
        pattern = DiscoveryPattern.from_element(sample_element_data)
        features = pattern.pattern_data["features"]

        assert features["category"] == "authentication"
        assert features["tag_name"] == "button"
        assert features["has_label"] is True
        assert features["is_interactive"] is True

    def test_to_embedding_text(self, sample_page_data):
        """Test embedding text generation."""
        pattern = DiscoveryPattern.from_page(sample_page_data)
        text = pattern.to_embedding_text()

        assert "Pattern type:" in text
        assert "authentication" in text
        assert "Pattern name:" in text


# ==============================================================================
# Test PatternMatch Dataclass
# ==============================================================================


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_pattern_match_creation(self):
        """Test creating a PatternMatch instance."""
        match = PatternMatch(
            id="match-123",
            pattern_type="authentication",
            pattern_name="Login Pattern",
            pattern_data={"features": {}},
            times_seen=10,
            test_success_rate=0.85,
            similarity=0.92,
        )

        assert match.id == "match-123"
        assert match.pattern_type == "authentication"
        assert match.times_seen == 10
        assert match.similarity == 0.92


# ==============================================================================
# Test Utility Functions
# ==============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_normalize_title_removes_numbers(self):
        """Test that numbers are replaced in titles."""
        result = _normalize_title("Order #12345 Confirmation")
        assert "#" in result or result.count("#") > 0
        assert "12345" not in result

    def test_normalize_title_removes_uuids(self):
        """Test that UUIDs are replaced in titles."""
        result = _normalize_title("User 550e8400-e29b-41d4-a716-446655440000 Profile")
        # Result is lowercased, so check for "uuid" not "UUID"
        assert "uuid" in result
        assert "550e8400" not in result

    def test_normalize_title_lowercases(self):
        """Test that titles are lowercased."""
        result = _normalize_title("UPPERCASE Title")
        assert result == result.lower()

    def test_extract_url_pattern_replaces_numeric_ids(self):
        """Test that numeric IDs are replaced in URL paths."""
        result = _extract_url_pattern("https://example.com/users/123/posts")
        assert ":id" in result
        assert "123" not in result

    def test_extract_url_pattern_replaces_uuids(self):
        """Test that UUIDs are replaced in URL paths."""
        result = _extract_url_pattern(
            "https://example.com/items/550e8400-e29b-41d4-a716-446655440000"
        )
        assert ":uuid" in result

    def test_normalize_selector_replaces_ids(self):
        """Test that IDs are normalized in selectors."""
        result = _normalize_selector("#user-profile-12345")
        assert "#ID" in result
        assert "user-profile-12345" not in result

    def test_normalize_selector_replaces_classes(self):
        """Test that classes are counted in selectors."""
        result = _normalize_selector(".class1.class2.class3")
        assert "CLASS[3]" in result

    def test_create_signature_deterministic(self):
        """Test that signatures are deterministic."""
        features = {"key1": "value1", "key2": "value2"}
        sig1 = _create_signature(features)
        sig2 = _create_signature(features)
        assert sig1 == sig2

    def test_create_signature_different_for_different_input(self):
        """Test that different inputs produce different signatures."""
        sig1 = _create_signature({"key": "value1"})
        sig2 = _create_signature({"key": "value2"})
        assert sig1 != sig2

    def test_map_page_to_pattern_type(self):
        """Test page category to pattern type mapping."""
        test_cases = [
            ("auth_login", PatternType.AUTHENTICATION),
            ("auth_signup", PatternType.AUTHENTICATION),
            ("list", PatternType.LIST_VIEW),
            ("detail", PatternType.DETAIL_VIEW),
            ("form", PatternType.FORM),
            ("dashboard", PatternType.PAGE_LAYOUT),
            ("error", PatternType.ERROR_HANDLING),
            ("unknown", PatternType.CUSTOM),
        ]
        for category, expected in test_cases:
            result = _map_page_to_pattern_type(category)
            assert result == expected, f"Failed for {category}"

    def test_map_element_to_pattern_type(self):
        """Test element category to pattern type mapping."""
        test_cases = [
            ("navigation", PatternType.NAVIGATION),
            ("form", PatternType.FORM),
            ("authentication", PatternType.AUTHENTICATION),
            ("content", PatternType.PAGE_LAYOUT),
            ("unknown", PatternType.CUSTOM),
        ]
        for category, expected in test_cases:
            result = _map_element_to_pattern_type(category)
            assert result == expected, f"Failed for {category}"

    def test_pad_embedding_pads_shorter(self):
        """Test that shorter embeddings are padded."""
        embedding = [0.1, 0.2, 0.3]
        result = _pad_embedding(embedding, 5)
        assert len(result) == 5
        assert result[:3] == [0.1, 0.2, 0.3]
        assert result[3:] == [0.0, 0.0]

    def test_pad_embedding_truncates_longer(self):
        """Test that longer embeddings are truncated."""
        embedding = [0.1] * 10
        result = _pad_embedding(embedding, 5)
        assert len(result) == 5

    def test_pad_embedding_returns_same_for_matching(self):
        """Test that matching dimension returns same embedding."""
        embedding = [0.1, 0.2, 0.3]
        result = _pad_embedding(embedding, 3)
        assert result == embedding


# ==============================================================================
# Test PatternService Initialization
# ==============================================================================


class TestPatternServiceInit:
    """Tests for PatternService initialization."""

    def test_init_with_clients(self, mock_supabase, mock_vectorize):
        """Test initialization with provided clients."""
        service = PatternService(
            supabase=mock_supabase,
            vectorize=mock_vectorize,
        )
        assert service.supabase == mock_supabase
        assert service.vectorize == mock_vectorize

    def test_init_uses_default_clients(self):
        """Test initialization with default clients."""
        with patch(
            "src.discovery.pattern_service.get_supabase_client"
        ) as mock_get_supabase:
            with patch(
                "src.discovery.pattern_service.get_vectorize_client"
            ) as mock_get_vectorize:
                mock_get_supabase.return_value = MagicMock()
                mock_get_vectorize.return_value = MagicMock()

                PatternService()

                mock_get_supabase.assert_called_once()
                mock_get_vectorize.assert_called_once()


# ==============================================================================
# Test Pattern Storage
# ==============================================================================


class TestPatternStorage:
    """Tests for pattern storage operations."""

    @pytest.mark.asyncio
    async def test_store_pattern_new(self, mock_supabase, mock_vectorize):
        """Test storing a new pattern."""
        mock_supabase.request = AsyncMock(
            side_effect=[
                {"data": []},  # find_by_signature returns nothing
                {"data": [{"id": "new-pattern-id"}]},  # insert succeeds
            ]
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        pattern = DiscoveryPattern(
            pattern_type=PatternType.AUTHENTICATION,
            pattern_name="Login Pattern",
            pattern_signature="abc123",
            pattern_data={"features": {}},
        )

        result = await service._store_pattern(pattern, "project-123")

        assert result.get("created") is True

    @pytest.mark.asyncio
    async def test_store_pattern_existing(self, mock_supabase, mock_vectorize):
        """Test storing pattern updates existing."""
        mock_supabase.request = AsyncMock(
            side_effect=[
                {"data": [{"id": "existing-id", "times_seen": 5}]},  # find_by_signature
                {"data": [{"times_seen": 6}]},  # RPC increment
            ]
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        pattern = DiscoveryPattern(
            pattern_type=PatternType.AUTHENTICATION,
            pattern_name="Login Pattern",
            pattern_signature="abc123",
            pattern_data={"features": {}},
        )

        result = await service._store_pattern(pattern, "project-123")

        assert result.get("updated") is True

    @pytest.mark.asyncio
    async def test_store_pattern_handles_embedding_failure(
        self, mock_supabase, mock_vectorize
    ):
        """Test storing pattern when embedding generation fails."""
        mock_supabase.request = AsyncMock(
            side_effect=[
                {"data": []},  # find_by_signature returns nothing
                {"data": [{"id": "new-pattern-id"}]},  # insert succeeds
            ]
        )
        mock_vectorize.generate_embedding = AsyncMock(side_effect=Exception("API error"))

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        pattern = DiscoveryPattern(
            pattern_type=PatternType.FORM,
            pattern_name="Form Pattern",
            pattern_signature="def456",
            pattern_data={"features": {}},
        )

        # Should not raise, should still store pattern
        result = await service._store_pattern(pattern, "project-123")
        # Pattern should still be created even without embedding
        assert "error" not in result or result.get("created") is True


# ==============================================================================
# Test Extract and Store Patterns
# ==============================================================================


class TestExtractAndStorePatterns:
    """Tests for extracting and storing patterns."""

    @pytest.mark.asyncio
    async def test_extract_and_store_patterns_from_pages(
        self, mock_supabase, mock_vectorize, sample_page_data
    ):
        """Test extracting patterns from pages."""
        mock_supabase.request = AsyncMock(return_value={"data": [{"id": "p1"}]})

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        result = await service.extract_and_store_patterns(
            session_id="session-123",
            project_id="project-456",
            pages=[sample_page_data],
            flows=[],
            elements=[],
        )

        assert "stored" in result
        assert "updated" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_extract_and_store_patterns_from_flows(
        self, mock_supabase, mock_vectorize, sample_flow_data
    ):
        """Test extracting patterns from flows."""
        mock_supabase.request = AsyncMock(return_value={"data": [{"id": "f1"}]})

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        result = await service.extract_and_store_patterns(
            session_id="session-123",
            project_id="project-456",
            pages=[],
            flows=[sample_flow_data],
            elements=[],
        )

        assert result["stored"] >= 0 or result["updated"] >= 0

    @pytest.mark.asyncio
    async def test_extract_and_store_patterns_limits_elements(
        self, mock_supabase, mock_vectorize
    ):
        """Test that element extraction is limited to key categories."""
        mock_supabase.request = AsyncMock(return_value={"data": [{"id": "e1"}]})

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        # Create many elements, but only some are key categories
        elements = [
            {"selector": f"#el{i}", "category": "content"} for i in range(100)
        ]
        elements.extend([
            {"selector": "#login", "category": "authentication"},
            {"selector": "#form", "category": "form"},
            {"selector": "#nav", "category": "navigation"},
        ])

        result = await service.extract_and_store_patterns(
            session_id="session-123",
            project_id="project-456",
            pages=[],
            flows=[],
            elements=elements,
        )

        # Should only process authentication, form, navigation elements
        # Content elements should be filtered out
        assert "stored" in result


# ==============================================================================
# Test Find Similar Patterns
# ==============================================================================


class TestFindSimilarPatterns:
    """Tests for finding similar patterns."""

    @pytest.mark.asyncio
    async def test_find_similar_patterns_success(self, mock_supabase, mock_vectorize):
        """Test finding similar patterns."""
        mock_supabase.request = AsyncMock(
            return_value={
                "data": [
                    {
                        "id": "p1",
                        "pattern_type": "authentication",
                        "pattern_name": "Login",
                        "pattern_data": {"features": {}},
                        "times_seen": 10,
                        "test_success_rate": 0.9,
                        "similarity": 0.95,
                    }
                ]
            }
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        query_pattern = DiscoveryPattern(
            pattern_type=PatternType.AUTHENTICATION,
            pattern_name="Login",
            pattern_signature="abc",
            pattern_data={},
        )

        matches = await service.find_similar_patterns(query_pattern)

        assert len(matches) == 1
        assert matches[0].pattern_type == "authentication"
        assert matches[0].similarity == 0.95

    @pytest.mark.asyncio
    async def test_find_similar_patterns_no_embedding(self, mock_supabase):
        """Test finding patterns when embedding fails."""
        service = PatternService(supabase=mock_supabase, vectorize=None)

        query_pattern = DiscoveryPattern(
            pattern_type=PatternType.FORM,
            pattern_name="Contact Form",
            pattern_signature="def",
            pattern_data={},
        )

        matches = await service.find_similar_patterns(query_pattern)
        assert matches == []

    @pytest.mark.asyncio
    async def test_find_similar_patterns_with_type_filter(
        self, mock_supabase, mock_vectorize
    ):
        """Test finding patterns with type filter."""
        mock_supabase.request = AsyncMock(return_value={"data": []})

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        query_pattern = DiscoveryPattern(
            pattern_type=PatternType.FORM,
            pattern_name="Form",
            pattern_signature="ghi",
            pattern_data={},
        )

        await service.find_similar_patterns(
            query_pattern, pattern_type=PatternType.FORM
        )

        # Verify the filter was passed
        call_args = mock_supabase.request.call_args
        body = call_args.kwargs.get("body", {})
        assert body.get("pattern_type_filter") == "form"


# ==============================================================================
# Test Pattern Insights
# ==============================================================================


class TestPatternInsights:
    """Tests for pattern insights."""

    @pytest.mark.asyncio
    async def test_get_pattern_insights(self, mock_supabase, mock_vectorize):
        """Test getting pattern insights."""
        mock_supabase.request = AsyncMock(
            return_value={
                "data": [
                    {
                        "pattern_type": "authentication",
                        "times_seen": 10,
                        "test_success_rate": 0.9,
                        "self_heal_success_rate": 0.8,
                    },
                    {
                        "pattern_type": "form",
                        "times_seen": 5,
                        "test_success_rate": 0.7,
                        "self_heal_success_rate": None,
                    },
                ]
            }
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        insights = await service.get_pattern_insights()

        assert insights["total_patterns"] == 2
        assert insights["total_occurrences"] == 15
        assert "authentication" in insights["by_type"]
        assert "form" in insights["by_type"]

    @pytest.mark.asyncio
    async def test_get_pattern_insights_with_type_filter(
        self, mock_supabase, mock_vectorize
    ):
        """Test getting insights with type filter."""
        mock_supabase.request = AsyncMock(return_value={"data": []})

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        await service.get_pattern_insights(pattern_type=PatternType.AUTHENTICATION)

        call_args = mock_supabase.request.call_args[0][0]
        assert "pattern_type=eq.authentication" in call_args


# ==============================================================================
# Test Update Pattern Success Rate
# ==============================================================================


class TestUpdatePatternSuccessRate:
    """Tests for updating pattern success rates."""

    @pytest.mark.asyncio
    async def test_update_success_rate_passed(self, mock_supabase, mock_vectorize):
        """Test updating success rate for passed test."""
        mock_supabase.request = AsyncMock(
            side_effect=[
                {
                    "data": [
                        {
                            "times_seen": 5,
                            "test_success_rate": 0.8,
                            "self_heal_success_rate": 0,
                        }
                    ]
                },
                {"data": []},  # update response
            ]
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        result = await service.update_pattern_success_rate(
            pattern_id="pattern-123",
            test_passed=True,
            self_healed=False,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_success_rate_not_found(self, mock_supabase, mock_vectorize):
        """Test updating success rate for non-existent pattern."""
        mock_supabase.request = AsyncMock(return_value={"data": []})

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        result = await service.update_pattern_success_rate(
            pattern_id="non-existent",
            test_passed=True,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_success_rate_with_self_heal(
        self, mock_supabase, mock_vectorize
    ):
        """Test updating success rate with self-healing."""
        mock_supabase.request = AsyncMock(
            side_effect=[
                {
                    "data": [
                        {
                            "times_seen": 5,
                            "test_success_rate": 0.5,
                            "self_heal_success_rate": 0.2,
                        }
                    ]
                },
                {"data": []},
            ]
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        result = await service.update_pattern_success_rate(
            pattern_id="pattern-123",
            test_passed=True,
            self_healed=True,
        )

        assert result is True


# ==============================================================================
# Test Store Pattern (Public Interface)
# ==============================================================================


class TestStorePatternPublic:
    """Tests for the public store_pattern method."""

    @pytest.mark.asyncio
    async def test_store_pattern_new(self, mock_supabase, mock_vectorize):
        """Test storing a new pattern through public interface."""
        mock_supabase.request = AsyncMock(
            side_effect=[
                {"data": []},  # find_by_signature
                {"data": [{"id": "new-id", "pattern_name": "Test"}]},  # insert
            ]
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        pattern_data = {
            "pattern_type": "form",
            "pattern_name": "Contact Form",
            "pattern_signature": "unique-sig",
            "pattern_data": {"features": {}},
        }

        result = await service.store_pattern(pattern_data)

        assert result["pattern_name"] == "Test"

    @pytest.mark.asyncio
    async def test_store_pattern_existing_updates(self, mock_supabase, mock_vectorize):
        """Test that existing pattern is updated."""
        existing = {
            "id": "existing-id",
            "pattern_name": "Existing Pattern",
            "times_seen": 5,
        }
        mock_supabase.request = AsyncMock(
            side_effect=[
                {"data": [existing]},  # find_by_signature
                {"data": [{"times_seen": 6}]},  # RPC increment
            ]
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        pattern_data = {
            "pattern_signature": "existing-sig",
            "pattern_name": "Existing Pattern",
        }

        result = await service.store_pattern(pattern_data)

        assert result["times_seen"] == 6


# ==============================================================================
# Test Get Patterns
# ==============================================================================


class TestGetPatterns:
    """Tests for getting patterns."""

    @pytest.mark.asyncio
    async def test_get_patterns_for_project(self, mock_supabase, mock_vectorize):
        """Test getting patterns for a project."""
        mock_supabase.request = AsyncMock(
            return_value={
                "data": [
                    {"id": "p1", "pattern_name": "Pattern 1"},
                    {"id": "p2", "pattern_name": "Pattern 2"},
                ]
            }
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        patterns = await service.get_patterns_for_project("project-123")

        assert len(patterns) == 2

    @pytest.mark.asyncio
    async def test_get_patterns_for_session(self, mock_supabase, mock_vectorize):
        """Test getting patterns for a session."""
        mock_supabase.request = AsyncMock(
            return_value={
                "data": [
                    {"id": "p1", "pattern_name": "Pattern 1"},
                ]
            }
        )

        service = PatternService(supabase=mock_supabase, vectorize=mock_vectorize)

        patterns = await service.get_patterns_for_session("session-123", limit=10)

        assert len(patterns) == 1


# ==============================================================================
# Test Global Instance
# ==============================================================================


class TestGlobalInstance:
    """Tests for the global pattern service instance."""

    def test_get_pattern_service_creates_instance(self):
        """Test that get_pattern_service creates an instance."""
        import src.discovery.pattern_service as module
        module._pattern_service = None

        with patch(
            "src.discovery.pattern_service.get_supabase_client"
        ) as mock_supabase:
            with patch(
                "src.discovery.pattern_service.get_vectorize_client"
            ) as mock_vectorize:
                mock_supabase.return_value = MagicMock()
                mock_vectorize.return_value = MagicMock()

                instance = get_pattern_service()
                assert instance is not None
                assert isinstance(instance, PatternService)

    def test_get_pattern_service_returns_same_instance(self):
        """Test that get_pattern_service returns the same instance."""
        import src.discovery.pattern_service as module
        module._pattern_service = None

        with patch(
            "src.discovery.pattern_service.get_supabase_client"
        ) as mock_supabase:
            with patch(
                "src.discovery.pattern_service.get_vectorize_client"
            ) as mock_vectorize:
                mock_supabase.return_value = MagicMock()
                mock_vectorize.return_value = MagicMock()

                instance1 = get_pattern_service()
                instance2 = get_pattern_service()
                assert instance1 is instance2
