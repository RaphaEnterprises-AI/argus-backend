"""Tests for Visual AI agent - comprehensive enterprise-grade test suite.

Following industry best practices:
- Parametrization for edge cases
- Proper mocking for external dependencies
- Async testing with pytest-asyncio
- Clear test naming with docstrings
"""

import json
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from src.agents.visual_ai import (
    DifferenceType,
    Severity,
    VisualDifference,
    VisualComparisonResult,
    VisualAI,
    VisualRegressionManager,
    create_visual_ai,
)


class TestDifferenceTypeEnum:
    """Tests for DifferenceType enum."""

    @pytest.mark.parametrize("diff_type,expected_value", [
        (DifferenceType.LAYOUT, "layout"),
        (DifferenceType.CONTENT, "content"),
        (DifferenceType.STYLE, "style"),
        (DifferenceType.MISSING, "missing"),
        (DifferenceType.NEW, "new"),
        (DifferenceType.DYNAMIC, "dynamic"),
        (DifferenceType.NONE, "none"),
    ])
    def test_difference_type_values(self, mock_env_vars, diff_type, expected_value):
        """Test DifferenceType enum values."""
        assert diff_type.value == expected_value


class TestSeverityEnum:
    """Tests for Severity enum."""

    @pytest.mark.parametrize("severity,expected_value", [
        (Severity.CRITICAL, "critical"),
        (Severity.MAJOR, "major"),
        (Severity.MINOR, "minor"),
        (Severity.INFO, "info"),
    ])
    def test_severity_values(self, mock_env_vars, severity, expected_value):
        """Test Severity enum values."""
        assert severity.value == expected_value


class TestVisualDifference:
    """Tests for VisualDifference dataclass."""

    def test_visual_difference_creation(self, mock_env_vars):
        """Test VisualDifference creation with all fields."""
        diff = VisualDifference(
            type=DifferenceType.LAYOUT,
            severity=Severity.MAJOR,
            description="Button moved from left to right",
            location="top-left",
            element="submit button",
            expected="Button at coordinates (10, 20)",
            actual="Button at coordinates (200, 20)",
            is_regression=True,
        )

        assert diff.type == DifferenceType.LAYOUT
        assert diff.severity == Severity.MAJOR
        assert diff.is_regression is True

    def test_visual_difference_defaults(self, mock_env_vars):
        """Test VisualDifference default values."""
        diff = VisualDifference(
            type=DifferenceType.STYLE,
            severity=Severity.MINOR,
            description="Color changed",
            location="header",
        )

        assert diff.element is None
        assert diff.expected is None
        assert diff.actual is None
        assert diff.is_regression is True


class TestVisualComparisonResult:
    """Tests for VisualComparisonResult dataclass."""

    def test_comparison_result_creation(self, mock_env_vars):
        """Test VisualComparisonResult creation."""
        result = VisualComparisonResult(
            baseline_path="/path/to/baseline.png",
            current_path="/path/to/current.png",
            match=True,
            match_percentage=95.5,
            summary="Screenshots match with minor dynamic content differences",
        )

        assert result.match is True
        assert result.match_percentage == 95.5
        assert result.baseline_path == "/path/to/baseline.png"

    def test_has_regressions_with_critical_regression(self, mock_env_vars):
        """Test has_regressions returns True for critical regression."""
        result = VisualComparisonResult(
            baseline_path="/path/to/baseline.png",
            current_path="/path/to/current.png",
            match=False,
            match_percentage=50.0,
            differences=[
                VisualDifference(
                    type=DifferenceType.MISSING,
                    severity=Severity.CRITICAL,
                    description="Submit button missing",
                    location="form",
                    is_regression=True,
                )
            ],
        )

        assert result.has_regressions() is True

    def test_has_regressions_with_info_only(self, mock_env_vars):
        """Test has_regressions returns False for info-only differences."""
        result = VisualComparisonResult(
            baseline_path="/path/to/baseline.png",
            current_path="/path/to/current.png",
            match=False,
            match_percentage=98.0,
            differences=[
                VisualDifference(
                    type=DifferenceType.DYNAMIC,
                    severity=Severity.INFO,
                    description="Timestamp changed",
                    location="footer",
                    is_regression=False,
                )
            ],
        )

        assert result.has_regressions() is False

    def test_has_regressions_with_non_regression_major(self, mock_env_vars):
        """Test has_regressions when is_regression=False on major issue."""
        result = VisualComparisonResult(
            baseline_path="/path/to/baseline.png",
            current_path="/path/to/current.png",
            match=False,
            match_percentage=80.0,
            differences=[
                VisualDifference(
                    type=DifferenceType.CONTENT,
                    severity=Severity.MAJOR,
                    description="Content updated intentionally",
                    location="main",
                    is_regression=False,  # Intentional change
                )
            ],
        )

        assert result.has_regressions() is False

    def test_to_dict(self, mock_env_vars):
        """Test VisualComparisonResult serialization."""
        # Use MAJOR severity so has_regressions returns True
        diff = VisualDifference(
            type=DifferenceType.STYLE,
            severity=Severity.MAJOR,  # MINOR doesn't count as regression
            description="Font size changed",
            location="header",
            is_regression=True,
        )
        result = VisualComparisonResult(
            baseline_path="/baseline.png",
            current_path="/current.png",
            match=False,
            match_percentage=90.0,
            differences=[diff],
            summary="Major style differences",
        )

        data = result.to_dict()

        assert data["baseline_path"] == "/baseline.png"
        assert data["match"] is False
        assert data["match_percentage"] == 90.0
        assert len(data["differences"]) == 1
        assert data["differences"][0]["type"] == "style"
        assert data["has_regressions"] is True


class TestVisualAI:
    """Tests for VisualAI class."""

    def test_visual_ai_initialization(self, mock_env_vars):
        """Test VisualAI initialization with defaults."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            visual = VisualAI()

            assert visual.model is not None
            assert visual.ignore_dynamic is True
            assert visual.sensitivity == "medium"

    @pytest.mark.parametrize("sensitivity", ["low", "medium", "high"])
    def test_visual_ai_sensitivity_levels(self, mock_env_vars, sensitivity):
        """Test VisualAI with different sensitivity levels."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            visual = VisualAI(sensitivity=sensitivity)
            assert visual.sensitivity == sensitivity

    def test_load_image_png(self, mock_env_vars, tmp_path):
        """Test loading PNG image."""
        # Create a fake PNG file
        png_file = tmp_path / "test.png"
        png_content = b"fake png content"
        png_file.write_bytes(png_content)

        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            visual = VisualAI()
            data, media_type = visual._load_image(png_file)

            assert media_type == "image/png"
            assert base64.b64decode(data) == png_content

    @pytest.mark.parametrize("extension,expected_media_type", [
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".gif", "image/gif"),
        (".webp", "image/webp"),
    ])
    def test_load_image_media_types(self, mock_env_vars, tmp_path, extension, expected_media_type):
        """Test loading different image formats."""
        image_file = tmp_path / f"test{extension}"
        image_file.write_bytes(b"fake image content")

        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            visual = VisualAI()
            _, media_type = visual._load_image(image_file)
            assert media_type == expected_media_type

    @pytest.mark.asyncio
    async def test_compare_success(self, mock_env_vars, tmp_path):
        """Test successful screenshot comparison."""
        # Create test images
        baseline = tmp_path / "baseline.png"
        current = tmp_path / "current.png"
        baseline.write_bytes(b"baseline image")
        current.write_bytes(b"current image")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "match": True,
            "match_percentage": 98.5,
            "summary": "Screenshots match with minor differences",
            "differences": [
                {
                    "type": "dynamic",
                    "severity": "info",
                    "description": "Timestamp changed",
                    "location": "footer",
                    "is_regression": False,
                }
            ],
        }))]
        mock_response.usage = MagicMock(input_tokens=1000, output_tokens=200)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            result = await visual.compare(baseline, current, context="Login page")

            assert result.match is True
            assert result.match_percentage == 98.5
            assert len(result.differences) == 1
            assert result.differences[0].type == DifferenceType.DYNAMIC

    @pytest.mark.asyncio
    async def test_compare_with_json_code_block(self, mock_env_vars, tmp_path):
        """Test comparison with JSON wrapped in code blocks."""
        baseline = tmp_path / "baseline.png"
        current = tmp_path / "current.png"
        baseline.write_bytes(b"baseline")
        current.write_bytes(b"current")

        # Response with ```json wrapper
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="""```json
{
    "match": false,
    "match_percentage": 75.0,
    "summary": "Layout changed",
    "differences": []
}
```""")]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            result = await visual.compare(baseline, current)

            assert result.match is False
            assert result.match_percentage == 75.0

    @pytest.mark.asyncio
    async def test_compare_failure(self, mock_env_vars, tmp_path):
        """Test comparison when API call fails."""
        baseline = tmp_path / "baseline.png"
        current = tmp_path / "current.png"
        baseline.write_bytes(b"baseline")
        current.write_bytes(b"current")

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("API error")
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            result = await visual.compare(baseline, current)

            assert result.match is False
            assert result.match_percentage == 0
            assert "failed" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_compare_with_ignore_regions(self, mock_env_vars, tmp_path):
        """Test comparison with ignored regions."""
        baseline = tmp_path / "baseline.png"
        current = tmp_path / "current.png"
        baseline.write_bytes(b"baseline")
        current.write_bytes(b"current")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"match": true, "match_percentage": 100, "summary": "Match", "differences": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=50)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            result = await visual.compare(
                baseline,
                current,
                ignore_regions=["header timestamp", "ad banner"],
            )

            # Verify ignore regions were passed in the prompt
            call_args = mock_client.messages.create.call_args
            messages = call_args[1]["messages"]
            prompt = messages[0]["content"][-1]["text"]
            assert "header timestamp" in prompt
            assert "ad banner" in prompt

    @pytest.mark.asyncio
    async def test_compare_batch(self, mock_env_vars, tmp_path):
        """Test batch comparison of multiple screenshot pairs."""
        comparisons = []
        for i in range(3):
            baseline = tmp_path / f"baseline_{i}.png"
            current = tmp_path / f"current_{i}.png"
            baseline.write_bytes(f"baseline{i}".encode())
            current.write_bytes(f"current{i}".encode())
            comparisons.append((str(baseline), str(current)))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"match": true, "match_percentage": 95, "summary": "Match", "differences": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=50)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            results = await visual.compare_batch(comparisons)

            assert len(results) == 3
            assert all(r.match for r in results)

    @pytest.mark.asyncio
    async def test_analyze_single(self, mock_env_vars, tmp_path):
        """Test single screenshot analysis."""
        screenshot = tmp_path / "page.png"
        screenshot.write_bytes(b"page screenshot")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "page_loaded": True,
            "title": "Welcome Page",
            "main_elements": ["header", "login form", "footer"],
            "issues": [],
            "expected_elements_found": {"login button": True},
            "overall_health": "healthy",
        }))]
        mock_response.usage = MagicMock(input_tokens=800, output_tokens=150)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            result = await visual.analyze_single(
                screenshot,
                expected_elements=["login button"],
                context="Home page",
            )

            assert result["page_loaded"] is True
            assert result["overall_health"] == "healthy"
            assert "login button" in result["expected_elements_found"]

    @pytest.mark.asyncio
    async def test_analyze_single_failure(self, mock_env_vars, tmp_path):
        """Test single screenshot analysis when it fails."""
        screenshot = tmp_path / "broken.png"
        screenshot.write_bytes(b"broken page")

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("Analysis failed")
            mock_anthropic.return_value = mock_client

            visual = VisualAI()
            result = await visual.analyze_single(screenshot)

            assert result["page_loaded"] is False
            assert result["overall_health"] == "broken"
            assert len(result["issues"]) > 0


class TestVisualRegressionManager:
    """Tests for VisualRegressionManager class."""

    def test_manager_initialization(self, mock_env_vars, tmp_path):
        """Test VisualRegressionManager initialization."""
        baseline_dir = tmp_path / "baselines"
        results_dir = tmp_path / "results"

        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            manager = VisualRegressionManager(
                baseline_dir=str(baseline_dir),
                results_dir=str(results_dir),
            )

            assert baseline_dir.exists()
            assert results_dir.exists()

    def test_get_baseline_path(self, mock_env_vars, tmp_path):
        """Test baseline path generation."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            manager = VisualRegressionManager(
                baseline_dir=str(tmp_path / "baselines"),
                results_dir=str(tmp_path / "results"),
            )

            path = manager.get_baseline_path("login_test", step=2)
            assert path.name == "login_test_step2.png"

    def test_has_baseline(self, mock_env_vars, tmp_path):
        """Test baseline existence check."""
        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()

        # Create a baseline file
        (baseline_dir / "test_id_step0.png").write_bytes(b"baseline")

        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            manager = VisualRegressionManager(
                baseline_dir=str(baseline_dir),
                results_dir=str(tmp_path / "results"),
            )

            assert manager.has_baseline("test_id", step=0) is True
            assert manager.has_baseline("test_id", step=1) is False
            assert manager.has_baseline("nonexistent", step=0) is False

    def test_save_baseline(self, mock_env_vars, tmp_path):
        """Test saving a new baseline."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            manager = VisualRegressionManager(
                baseline_dir=str(tmp_path / "baselines"),
                results_dir=str(tmp_path / "results"),
            )

            screenshot_data = b"test screenshot"
            path = manager.save_baseline("my_test", screenshot_data, step=1)

            assert path.exists()
            assert path.read_bytes() == screenshot_data

    @pytest.mark.asyncio
    async def test_check_visual_regression_first_run(self, mock_env_vars, tmp_path):
        """Test visual regression check on first run (creates baseline)."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            manager = VisualRegressionManager(
                baseline_dir=str(tmp_path / "baselines"),
                results_dir=str(tmp_path / "results"),
            )

            result = await manager.check_visual_regression(
                test_id="new_test",
                current_screenshot=b"first run screenshot",
                step=0,
            )

            assert result.match is True
            assert result.match_percentage == 100.0
            assert "baseline created" in result.summary.lower()
            assert manager.has_baseline("new_test", step=0)

    @pytest.mark.asyncio
    async def test_check_visual_regression_comparison(self, mock_env_vars, tmp_path):
        """Test visual regression check against existing baseline."""
        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        (baseline_dir / "existing_test_step0.png").write_bytes(b"baseline")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"match": false, "match_percentage": 85, "summary": "Changes detected", "differences": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            manager = VisualRegressionManager(
                baseline_dir=str(baseline_dir),
                results_dir=str(tmp_path / "results"),
            )

            result = await manager.check_visual_regression(
                test_id="existing_test",
                current_screenshot=b"new screenshot",
                step=0,
            )

            assert result.match is False
            assert result.match_percentage == 85

    @pytest.mark.asyncio
    async def test_check_visual_regression_auto_update(self, mock_env_vars, tmp_path):
        """Test auto-update baseline when no regressions."""
        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        baseline_file = baseline_dir / "auto_update_test_step0.png"
        baseline_file.write_bytes(b"old baseline")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"match": true, "match_percentage": 99, "summary": "Match", "differences": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=50)

        with patch("src.agents.visual_ai.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            manager = VisualRegressionManager(
                baseline_dir=str(baseline_dir),
                results_dir=str(tmp_path / "results"),
            )

            new_screenshot = b"new baseline content"
            await manager.check_visual_regression(
                test_id="auto_update_test",
                current_screenshot=new_screenshot,
                step=0,
                auto_update_baseline=True,
            )

            # Baseline should be updated
            assert baseline_file.read_bytes() == new_screenshot

    def test_generate_report(self, mock_env_vars, tmp_path):
        """Test report generation from multiple results."""
        results = [
            VisualComparisonResult(
                baseline_path="/base1.png",
                current_path="/curr1.png",
                match=True,
                match_percentage=100,
                analysis_cost_usd=0.01,
            ),
            VisualComparisonResult(
                baseline_path="/base2.png",
                current_path="/curr2.png",
                match=False,
                match_percentage=80,
                differences=[
                    VisualDifference(
                        type=DifferenceType.LAYOUT,
                        severity=Severity.MAJOR,
                        description="Layout changed",
                        location="main",
                        is_regression=True,
                    )
                ],
                analysis_cost_usd=0.02,
            ),
            VisualComparisonResult(
                baseline_path="/base3.png",
                current_path="/curr3.png",
                match=False,
                match_percentage=95,
                differences=[
                    VisualDifference(
                        type=DifferenceType.DYNAMIC,
                        severity=Severity.INFO,
                        description="Timestamp",
                        location="footer",
                        is_regression=False,
                    )
                ],
                analysis_cost_usd=0.01,
            ),
        ]

        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            manager = VisualRegressionManager(
                baseline_dir=str(tmp_path / "baselines"),
                results_dir=str(tmp_path / "results"),
            )

            report = manager.generate_report(results)

            assert report["summary"]["total"] == 3
            assert report["summary"]["passed"] == 2  # 1 match + 1 non-regression
            assert report["summary"]["failed"] == 1
            assert report["total_cost_usd"] == 0.04
            assert len(report["regressions"]) == 1


class TestCreateVisualAI:
    """Tests for create_visual_ai factory function."""

    def test_create_visual_ai_default(self, mock_env_vars):
        """Test creating VisualAI with default settings."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            visual = create_visual_ai()
            assert visual.sensitivity == "medium"

    @pytest.mark.parametrize("sensitivity", ["low", "medium", "high"])
    def test_create_visual_ai_with_sensitivity(self, mock_env_vars, sensitivity):
        """Test creating VisualAI with specific sensitivity."""
        with patch("src.agents.visual_ai.anthropic.Anthropic"):
            visual = create_visual_ai(sensitivity=sensitivity)
            assert visual.sensitivity == sensitivity
