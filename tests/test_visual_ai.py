"""Tests for Visual AI module."""

import base64
from unittest.mock import MagicMock, patch

import pytest


class TestVisualComparisonResult:
    """Tests for VisualComparisonResult dataclass."""

    def test_result_creation_match(self):
        """Test creating a matching result."""
        from src.agents.visual_ai import VisualComparisonResult

        result = VisualComparisonResult(
            baseline_path="/path/baseline.png",
            current_path="/path/current.png",
            match=True,
            match_percentage=98.0,
            differences=[],
            summary="Screenshots match perfectly",
        )

        assert result.match is True
        assert result.match_percentage == 98.0
        assert len(result.differences) == 0

    def test_result_creation_mismatch(self):
        """Test creating a mismatch result."""
        from src.agents.visual_ai import (
            DifferenceType,
            Severity,
            VisualComparisonResult,
            VisualDifference,
        )

        result = VisualComparisonResult(
            baseline_path="/path/baseline.png",
            current_path="/path/current.png",
            match=False,
            match_percentage=75.0,
            differences=[
                VisualDifference(
                    type=DifferenceType.LAYOUT,
                    severity=Severity.MAJOR,
                    description="Button moved 50px right",
                    location="top-left",
                    is_regression=True,
                )
            ],
            summary="Layout differences detected",
        )

        assert result.match is False
        assert result.match_percentage == 75.0
        assert len(result.differences) == 1
        assert result.differences[0].type == DifferenceType.LAYOUT

    def test_to_dict(self):
        """Test converting result to dict."""
        from src.agents.visual_ai import VisualComparisonResult

        result = VisualComparisonResult(
            baseline_path="/path/baseline.png",
            current_path="/path/current.png",
            match=True,
            match_percentage=95.0,
            differences=[],
            summary="Match",
        )

        d = result.to_dict()
        assert d["match"] is True
        assert d["match_percentage"] == 95.0

    def test_has_regressions(self):
        """Test has_regressions method."""
        from src.agents.visual_ai import (
            DifferenceType,
            Severity,
            VisualComparisonResult,
            VisualDifference,
        )

        # No regressions
        result1 = VisualComparisonResult(
            baseline_path="/path/baseline.png",
            current_path="/path/current.png",
            match=True,
            match_percentage=100.0,
            differences=[],
            summary="Match",
        )
        assert result1.has_regressions() is False

        # With regression
        result2 = VisualComparisonResult(
            baseline_path="/path/baseline.png",
            current_path="/path/current.png",
            match=False,
            match_percentage=80.0,
            differences=[
                VisualDifference(
                    type=DifferenceType.LAYOUT,
                    severity=Severity.MAJOR,
                    description="Layout changed",
                    location="center",
                    is_regression=True,
                )
            ],
            summary="Issues found",
        )
        assert result2.has_regressions() is True


class TestVisualAI:
    """Tests for VisualAI class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.visual_ai.get_settings") as mock:
            settings = MagicMock()
            settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def visual_ai(self, mock_settings):
        """Create VisualAI instance with mocked client."""
        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            from src.agents.visual_ai import VisualAI
            ai = VisualAI()
            ai.client = mock_client
            return ai

    @pytest.fixture
    def temp_screenshots(self, tmp_path):
        """Create temporary screenshot files."""
        # Create minimal PNG files
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        baseline = tmp_path / "baseline.png"
        current = tmp_path / "current.png"
        baseline.write_bytes(png_bytes)
        current.write_bytes(png_bytes)
        return baseline, current

    @pytest.mark.asyncio
    async def test_compare_matching_screenshots(self, visual_ai, temp_screenshots):
        """Test comparing matching screenshots."""
        baseline, current = temp_screenshots

        # Mock Claude response for matching screenshots
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "match": true,
    "match_percentage": 98,
    "differences": [],
    "summary": "Screenshots are visually identical"
}
```""")]
        mock_response.usage = MagicMock(input_tokens=1000, output_tokens=100)
        visual_ai.client.messages.create = MagicMock(return_value=mock_response)

        result = await visual_ai.compare(baseline, current)

        assert result.match is True
        assert result.match_percentage >= 95

    @pytest.mark.asyncio
    async def test_compare_different_screenshots(self, visual_ai, temp_screenshots):
        """Test comparing different screenshots."""
        baseline, current = temp_screenshots

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "match": false,
    "match_percentage": 65,
    "differences": [
        {
            "type": "layout",
            "severity": "major",
            "description": "Header layout changed significantly",
            "location": "top",
            "element": "header",
            "is_regression": true
        }
    ],
    "summary": "Significant layout changes detected"
}
```""")]
        mock_response.usage = MagicMock(input_tokens=1000, output_tokens=200)
        visual_ai.client.messages.create = MagicMock(return_value=mock_response)

        result = await visual_ai.compare(baseline, current)

        assert result.match is False
        assert len(result.differences) == 1

    @pytest.mark.asyncio
    async def test_analyze_single_screenshot(self, visual_ai, temp_screenshots):
        """Test analyzing a single screenshot."""
        baseline, _ = temp_screenshots

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "page_loaded": true,
    "title": "Test Page",
    "main_elements": ["login button", "username field"],
    "issues": [],
    "expected_elements_found": {"login button": true},
    "overall_health": "healthy"
}
```""")]
        visual_ai.client.messages.create = MagicMock(return_value=mock_response)

        result = await visual_ai.analyze_single(
            baseline,
            expected_elements=["login button"]
        )

        assert result["page_loaded"] is True
        assert len(result.get("issues", [])) == 0


class TestVisualRegressionManager:
    """Tests for VisualRegressionManager class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        with patch("src.agents.visual_ai.get_settings") as mock:
            settings = MagicMock()
            settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
            mock.return_value = settings
            yield mock

    @pytest.fixture
    def manager(self, tmp_path, mock_settings):
        """Create VisualRegressionManager with temp directories."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            from src.agents.visual_ai import VisualRegressionManager
            return VisualRegressionManager(
                baseline_dir=str(tmp_path / "baselines"),
                results_dir=str(tmp_path / "results"),
            )

    def test_manager_init(self, manager):
        """Test manager initialization."""
        assert manager.baseline_dir.exists()
        assert manager.results_dir.exists()

    def test_has_baseline_false(self, manager):
        """Test has_baseline when no baseline exists."""
        assert manager.has_baseline("new-test") is False

    def test_save_baseline(self, manager):
        """Test saving baseline."""
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        path = manager.save_baseline("test-id", png_bytes, step=0)

        assert path.exists()
        assert manager.has_baseline("test-id", step=0)

    @pytest.mark.asyncio
    async def test_check_regression_no_baseline(self, manager):
        """Test checking regression when no baseline exists."""
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        result = await manager.check_visual_regression("new-test", png_bytes, step=0)

        # Should create baseline and return match
        assert result.match is True
        assert "baseline created" in result.summary.lower()

    def test_get_baseline_path(self, manager):
        """Test getting baseline path."""
        path = manager.get_baseline_path("test-id", step=2)
        assert "test-id_step2.png" in str(path)


class TestDifferenceTypes:
    """Tests for difference type enums."""

    def test_difference_types(self):
        """Test all difference types exist."""
        from src.agents.visual_ai import DifferenceType

        assert DifferenceType.LAYOUT
        assert DifferenceType.CONTENT
        assert DifferenceType.STYLE
        assert DifferenceType.MISSING
        assert DifferenceType.NEW
        assert DifferenceType.DYNAMIC

    def test_severity_levels(self):
        """Test all severity levels exist."""
        from src.agents.visual_ai import Severity

        assert Severity.CRITICAL
        assert Severity.MAJOR
        assert Severity.MINOR
        assert Severity.INFO


class TestVisualDifference:
    """Tests for VisualDifference dataclass."""

    def test_difference_creation(self):
        """Test creating a visual difference."""
        from src.agents.visual_ai import DifferenceType, Severity, VisualDifference

        diff = VisualDifference(
            type=DifferenceType.CONTENT,
            severity=Severity.MINOR,
            description="Text changed",
            location="center",
            element="paragraph",
            expected="Hello",
            actual="Hi",
            is_regression=True,
        )

        assert diff.type == DifferenceType.CONTENT
        assert diff.severity == Severity.MINOR
        assert diff.is_regression is True
