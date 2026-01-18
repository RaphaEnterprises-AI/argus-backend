"""Comprehensive tests for visual_ai/semantic_analyzer.py.

Tests semantic analysis including AI-powered change interpretation,
context understanding, and intelligent change categorization.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.visual_ai.models import (
    ChangeCategory,
    ChangeIntent,
    Severity,
    VisualChange,
)
from src.visual_ai.semantic_analyzer import (
    SemanticAnalysis,
    SemanticAnalyzer,
    create_semantic_analyzer,
)
from src.visual_ai.structural_analyzer import StructuralDiff


class TestSemanticAnalysis:
    """Tests for SemanticAnalysis dataclass."""

    def test_semantic_analysis_creation(self):
        """Test creating a SemanticAnalysis instance."""
        analysis = SemanticAnalysis(
            changes=[],
            overall_assessment="Minor layout adjustments detected",
            summary="Minor changes",
            auto_approval_recommended=True,
            approval_confidence=0.85,
            blocking_issues=[],
            suggestions=["Review the header changes", "Test on mobile"],
            user_impact={"overall_impact": "low"},
            model_used="claude-sonnet-4-5",
            analysis_duration_ms=1500,
            token_usage={"input": 1000, "output": 500},
            cost_usd=0.05,
        )
        assert analysis.overall_assessment == "Minor layout adjustments detected"
        assert analysis.auto_approval_recommended is True
        assert analysis.approval_confidence == 0.85
        assert len(analysis.suggestions) == 2

    def test_semantic_analysis_with_changes(self):
        """Test SemanticAnalysis with actual changes."""
        change = VisualChange(
            id="change_001",
            category=ChangeCategory.LAYOUT,
            intent=ChangeIntent.REGRESSION,
            severity=Severity.MAJOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name="width",
            baseline_value="100px",
            current_value="200px",
            description="Element width doubled",
            root_cause=None,
            impact_assessment="May affect layout",
            recommendation="Review width change",
            confidence=0.9,
            related_commit=None,
        )
        analysis = SemanticAnalysis(
            changes=[change],
            overall_assessment="Layout regression detected",
            summary="Regression",
            auto_approval_recommended=False,
            approval_confidence=0.3,
            blocking_issues=["Major layout regression in change_001"],
            suggestions=["Fix layout regression"],
            model_used="claude-sonnet-4-5",
        )
        assert len(analysis.changes) == 1
        assert len(analysis.blocking_issues) == 1
        assert analysis.auto_approval_recommended is False

    def test_semantic_analysis_defaults(self):
        """Test SemanticAnalysis default values."""
        analysis = SemanticAnalysis()
        assert analysis.changes == []
        assert analysis.overall_assessment == ""
        assert analysis.summary == ""
        assert analysis.auto_approval_recommended is False
        assert analysis.approval_confidence == 0.0
        assert analysis.blocking_issues == []
        assert analysis.suggestions == []
        assert analysis.user_impact is None
        assert analysis.model_used == ""
        assert analysis.analysis_duration_ms == 0

    def test_to_dict(self):
        """Test to_dict serialization."""
        analysis = SemanticAnalysis(
            changes=[],
            overall_assessment="Test assessment",
            summary="Test summary",
            auto_approval_recommended=True,
            approval_confidence=0.75,
            blocking_issues=[],
            suggestions=["Test suggestion"],
            model_used="claude-sonnet-4-5",
            analysis_duration_ms=1000,
            token_usage={"input": 100, "output": 50},
            cost_usd=0.01,
        )
        result = analysis.to_dict()
        assert result["overall_assessment"] == "Test assessment"
        assert result["summary"] == "Test summary"
        assert result["auto_approval_recommended"] is True
        assert result["approval_confidence"] == 0.75
        assert result["suggestions"] == ["Test suggestion"]
        assert "timestamp" in result

    def test_has_regressions_true(self):
        """Test has_regressions returns True when regression exists."""
        change = VisualChange(
            id="regression_change",
            category=ChangeCategory.LAYOUT,
            intent=ChangeIntent.REGRESSION,
            severity=Severity.MAJOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name=None,
            baseline_value=None,
            current_value=None,
            description="Layout broke",
            root_cause=None,
            impact_assessment="Affects layout",
            recommendation="Fix layout",
            confidence=0.9,
            related_commit=None,
        )
        analysis = SemanticAnalysis(changes=[change])
        assert analysis.has_regressions() is True

    def test_has_regressions_false(self):
        """Test has_regressions returns False when no regression."""
        change = VisualChange(
            id="intentional_change",
            category=ChangeCategory.STYLE,
            intent=ChangeIntent.INTENTIONAL,
            severity=Severity.MINOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name=None,
            baseline_value=None,
            current_value=None,
            description="Style updated",
            root_cause=None,
            impact_assessment="Minor style change",
            recommendation="Review",
            confidence=0.9,
            related_commit=None,
        )
        analysis = SemanticAnalysis(changes=[change])
        assert analysis.has_regressions() is False

    def test_has_blocking_issues_true(self):
        """Test has_blocking_issues returns True."""
        analysis = SemanticAnalysis(
            blocking_issues=["Critical issue found"]
        )
        assert analysis.has_blocking_issues() is True

    def test_has_blocking_issues_false(self):
        """Test has_blocking_issues returns False when empty."""
        analysis = SemanticAnalysis(blocking_issues=[])
        assert analysis.has_blocking_issues() is False

    def test_has_blocking_issues_from_changes(self):
        """Test has_blocking_issues from critical changes."""
        change = VisualChange(
            id="critical_change",
            category=ChangeCategory.LAYOUT,
            intent=ChangeIntent.REGRESSION,
            severity=Severity.CRITICAL,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name=None,
            baseline_value=None,
            current_value=None,
            description="Critical layout issue",
            root_cause=None,
            impact_assessment="Critical impact",
            recommendation="Fix immediately",
            confidence=0.95,
            related_commit=None,
        )
        analysis = SemanticAnalysis(changes=[change], blocking_issues=[])
        assert analysis.has_blocking_issues() is True

    def test_get_highest_severity_empty(self):
        """Test get_highest_severity with no changes."""
        analysis = SemanticAnalysis(changes=[])
        assert analysis.get_highest_severity() == Severity.SAFE

    def test_get_highest_severity_with_changes(self):
        """Test get_highest_severity with multiple changes."""
        changes = [
            VisualChange(
                id="minor",
                category=ChangeCategory.STYLE,
                intent=ChangeIntent.INTENTIONAL,
                severity=Severity.MINOR,
                element=None,
                bounds_baseline=None,
                bounds_current=None,
                property_name=None,
                baseline_value=None,
                current_value=None,
                description="Minor",
                root_cause=None,
                impact_assessment="Minor",
                recommendation="Review",
                confidence=0.8,
                related_commit=None,
            ),
            VisualChange(
                id="major",
                category=ChangeCategory.LAYOUT,
                intent=ChangeIntent.REGRESSION,
                severity=Severity.MAJOR,
                element=None,
                bounds_baseline=None,
                bounds_current=None,
                property_name=None,
                baseline_value=None,
                current_value=None,
                description="Major",
                root_cause=None,
                impact_assessment="Major",
                recommendation="Fix",
                confidence=0.9,
                related_commit=None,
            ),
            VisualChange(
                id="info",
                category=ChangeCategory.CONTENT,
                intent=ChangeIntent.DYNAMIC,
                severity=Severity.INFO,
                element=None,
                bounds_baseline=None,
                bounds_current=None,
                property_name=None,
                baseline_value=None,
                current_value=None,
                description="Info",
                root_cause=None,
                impact_assessment="Info",
                recommendation="Note",
                confidence=0.7,
                related_commit=None,
            ),
        ]
        analysis = SemanticAnalysis(changes=changes)
        assert analysis.get_highest_severity() == Severity.MAJOR


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.anthropic_api_key = MagicMock()
        settings.anthropic_api_key.get_secret_value.return_value = "test-api-key"
        return settings

    @pytest.fixture
    def analyzer(self, mock_settings):
        """Create a SemanticAnalyzer instance with mocked dependencies."""
        with patch("src.visual_ai.semantic_analyzer.get_settings", return_value=mock_settings):
            with patch("src.visual_ai.semantic_analyzer.anthropic.Anthropic"):
                with patch("src.visual_ai.semantic_analyzer.get_model_id", return_value="claude-sonnet-4-5"):
                    return SemanticAnalyzer()

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create a mocked Anthropic API response."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text=json.dumps({
                "changes": [],
                "overall_assessment": "No significant changes detected",
                "auto_approval_recommended": True,
                "approval_confidence": 0.95,
                "blocking_issues": [],
                "suggestions": []
            }))
        ]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 200
        return mock_response

    @pytest.fixture
    def sample_structural_diff(self):
        """Create a sample StructuralDiff."""
        return StructuralDiff(
            baseline_id="base",
            current_id="current",
            timestamp="2024-01-01T12:00:00Z",
            total_elements_baseline=50,
            total_elements_current=52,
            elements_added=5,
            elements_removed=3,
            elements_modified=10,
            elements_unchanged=35,
            changes=[],
            baseline_layout_regions=[],
            current_layout_regions=[],
            layout_shift_score=0.05,
            pixel_diff_percentage=3.5,
            pixel_diff_regions=[],
            baseline_layout_hash="hash1",
            current_layout_hash="hash2",
            baseline_content_hash="content1",
            current_content_hash="content2",
        )

    def test_init(self, analyzer):
        """Test SemanticAnalyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, "model")
        assert hasattr(analyzer, "client")

    def test_init_with_custom_model(self, mock_settings):
        """Test initialization with custom model."""
        with patch("src.visual_ai.semantic_analyzer.get_settings", return_value=mock_settings):
            with patch("src.visual_ai.semantic_analyzer.anthropic.Anthropic"):
                with patch("src.visual_ai.semantic_analyzer.get_model_id", return_value="claude-haiku-3-5"):
                    analyzer = SemanticAnalyzer(model="claude-haiku-3-5")
                    assert analyzer.model == "claude-haiku-3-5"

    def test_encode_image_png(self, analyzer):
        """Test image encoding for PNG."""
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\ntest_image_data"
        data, media_type = analyzer._encode_image(png_data)
        assert media_type == "image/png"
        assert len(data) > 0

    def test_encode_image_jpeg(self, analyzer):
        """Test image encoding for JPEG."""
        # JPEG magic bytes
        jpeg_data = b"\xff\xd8test_image_data"
        data, media_type = analyzer._encode_image(jpeg_data)
        assert media_type == "image/jpeg"

    def test_encode_image_gif(self, analyzer):
        """Test image encoding for GIF."""
        # GIF magic bytes
        gif_data = b"GIF89atest_image_data"
        data, media_type = analyzer._encode_image(gif_data)
        assert media_type == "image/gif"

    def test_encode_image_webp(self, analyzer):
        """Test image encoding for WebP."""
        # WebP magic bytes
        webp_data = b"RIFF\x00\x00\x00\x00WEBPtest"
        data, media_type = analyzer._encode_image(webp_data)
        assert media_type == "image/webp"

    def test_encode_image_unknown(self, analyzer):
        """Test image encoding for unknown format."""
        unknown_data = b"unknown_format_data"
        data, media_type = analyzer._encode_image(unknown_data)
        assert media_type == "image/png"  # Default fallback

    def test_build_analysis_prompt(self, analyzer, sample_structural_diff):
        """Test prompt building for API call."""
        prompt = analyzer._build_analysis_prompt(
            structural_diff=sample_structural_diff,
            context="Test page",
            git_diff="+ added line\n- removed line",
            pr_description="Fix button styling",
        )
        assert isinstance(prompt, str)
        assert "STRUCTURAL ANALYSIS" in prompt
        assert "CODE CHANGES" in prompt
        assert "PR DESCRIPTION" in prompt

    def test_build_analysis_prompt_minimal(self, analyzer):
        """Test prompt building with minimal input."""
        prompt = analyzer._build_analysis_prompt()
        assert isinstance(prompt, str)
        assert "IDENTIFY CHANGES" in prompt

    def test_parse_json_response(self, analyzer):
        """Test parsing JSON response."""
        response_text = json.dumps({
            "changes": [],
            "overall_assessment": "Parsed analysis",
            "auto_approval_recommended": True,
            "approval_confidence": 0.85,
            "blocking_issues": [],
            "suggestions": ["Test on mobile"]
        })
        result = analyzer._parse_json_response(response_text)
        assert isinstance(result, dict)
        assert result["overall_assessment"] == "Parsed analysis"

    def test_parse_json_response_with_code_block(self, analyzer):
        """Test parsing JSON wrapped in code block."""
        response_text = """```json
{
    "changes": [],
    "overall_assessment": "Code block parsed",
    "auto_approval_recommended": true
}
```"""
        result = analyzer._parse_json_response(response_text)
        assert result["overall_assessment"] == "Code block parsed"

    def test_parse_visual_change(self, analyzer):
        """Test parsing a visual change from API response."""
        change_data = {
            "id": "change-1",
            "category": "layout",
            "description": "Element moved 10px",
            "severity": 2,
            "intent": "intentional",
            "bounds_baseline": {"x": 0, "y": 0, "width": 100, "height": 50},
            "bounds_current": {"x": 10, "y": 0, "width": 100, "height": 50},
            "baseline_value": "x: 0",
            "current_value": "x: 10",
            "root_cause": "CSS margin change",
            "impact_assessment": "Minor visual shift",
            "recommendation": "Verify intentional",
            "confidence": 0.9,
            "related_files": ["styles.css"],
        }
        change = analyzer._parse_visual_change(change_data)
        assert isinstance(change, VisualChange)
        assert change.category == ChangeCategory.LAYOUT
        assert change.intent == ChangeIntent.INTENTIONAL
        assert change.severity == Severity.MINOR
        assert change.confidence == 0.9

    def test_parse_visual_change_unknown_category(self, analyzer):
        """Test parsing change with unknown category."""
        change_data = {
            "id": "change-1",
            "category": "unknown_category",
            "description": "Some change",
            "severity": 1,
        }
        change = analyzer._parse_visual_change(change_data)
        assert change.category == ChangeCategory.CONTENT  # Default

    def test_parse_visual_change_unknown_intent(self, analyzer):
        """Test parsing change with unknown intent."""
        change_data = {
            "id": "change-1",
            "category": "style",
            "description": "Color change",
            "severity": 1,
            "intent": "something_else",
        }
        change = analyzer._parse_visual_change(change_data)
        assert change.intent == ChangeIntent.UNKNOWN  # Default

    @pytest.mark.asyncio
    async def test_analyze_changes(self, analyzer, mock_anthropic_response):
        """Test analyze_changes method."""
        analyzer.client.messages.create = MagicMock(return_value=mock_anthropic_response)

        baseline = b"\x89PNG\r\n\x1a\nbaseline"
        current = b"\x89PNG\r\n\x1a\ncurrent"

        result = await analyzer.analyze_changes(
            baseline_screenshot=baseline,
            current_screenshot=current,
        )

        assert isinstance(result, SemanticAnalysis)
        assert result.auto_approval_recommended is True

    @pytest.mark.asyncio
    async def test_analyze_changes_with_context(self, analyzer, mock_anthropic_response, sample_structural_diff):
        """Test analyze_changes with full context."""
        analyzer.client.messages.create = MagicMock(return_value=mock_anthropic_response)

        baseline = b"\x89PNG\r\n\x1a\nbaseline"
        current = b"\x89PNG\r\n\x1a\ncurrent"

        result = await analyzer.analyze_changes(
            baseline_screenshot=baseline,
            current_screenshot=current,
            structural_diff=sample_structural_diff,
            context="Homepage test",
            git_diff="+ new code",
            pr_description="Update styling",
        )

        assert isinstance(result, SemanticAnalysis)

    @pytest.mark.asyncio
    async def test_analyze_changes_api_error(self, analyzer):
        """Test handling API errors gracefully."""
        analyzer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        baseline = b"\x89PNG\r\n\x1a\nbaseline"
        current = b"\x89PNG\r\n\x1a\ncurrent"

        result = await analyzer.analyze_changes(
            baseline_screenshot=baseline,
            current_screenshot=current,
        )

        assert isinstance(result, SemanticAnalysis)
        assert "failed" in result.overall_assessment.lower()
        assert result.approval_confidence == 0.0

    @pytest.mark.asyncio
    async def test_classify_change_intent(self, analyzer):
        """Test change intent classification."""
        change = VisualChange(
            id="test",
            category=ChangeCategory.STYLE,
            intent=ChangeIntent.UNKNOWN,
            severity=Severity.MINOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name="color",
            baseline_value="blue",
            current_value="green",
            description="Button color changed",
            root_cause=None,
            impact_assessment="Minor visual change",
            recommendation="Verify intentional",
            confidence=0.5,
            related_commit=None,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"intent": "intentional", "confidence": 0.9, "reasoning": "Matches PR"}')]
        analyzer.client.messages.create = MagicMock(return_value=mock_response)

        result = await analyzer.classify_change_intent(
            change,
            git_diff="- color: blue\n+ color: green",
            pr_description="Update button colors",
        )

        assert result == ChangeIntent.INTENTIONAL

    @pytest.mark.asyncio
    async def test_classify_change_intent_no_context(self, analyzer):
        """Test change intent classification without context."""
        change = VisualChange(
            id="test",
            category=ChangeCategory.LAYOUT,
            intent=ChangeIntent.REGRESSION,
            severity=Severity.MAJOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name=None,
            baseline_value=None,
            current_value=None,
            description="Layout broke",
            root_cause=None,
            impact_assessment="Major impact",
            recommendation="Fix",
            confidence=0.9,
            related_commit=None,
        )

        # Without git_diff and pr_description, should return existing intent
        result = await analyzer.classify_change_intent(change)
        assert result == ChangeIntent.REGRESSION

    @pytest.mark.asyncio
    async def test_generate_change_description(self, analyzer):
        """Test change description generation."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="The header color changed from blue to green.")]
        analyzer.client.messages.create = MagicMock(return_value=mock_response)

        baseline = b"\x89PNG\r\n\x1a\nbaseline"
        current = b"\x89PNG\r\n\x1a\ncurrent"

        result = await analyzer.generate_change_description(baseline, current)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_change_description_with_region(self, analyzer):
        """Test change description generation with specific region."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Button styling changed in the specified region.")]
        analyzer.client.messages.create = MagicMock(return_value=mock_response)

        baseline = b"\x89PNG\r\n\x1a\nbaseline"
        current = b"\x89PNG\r\n\x1a\ncurrent"
        region = {"x": 100, "y": 200, "width": 50, "height": 30}

        result = await analyzer.generate_change_description(baseline, current, region)

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_predict_user_impact_no_changes(self, analyzer):
        """Test user impact prediction with no changes."""
        result = await analyzer.predict_user_impact([])

        assert result["overall_impact"] == "none"
        assert result["risk_level"] == 0

    @pytest.mark.asyncio
    async def test_predict_user_impact_with_changes(self, analyzer):
        """Test user impact prediction with changes."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "overall_impact": "medium",
            "usability_impact": "Button harder to find",
            "accessibility_impact": "None",
            "conversion_impact": "May reduce clicks",
            "affected_user_flows": ["checkout"],
            "risk_level": 2,
            "recommendations": ["Test with users"]
        }))]
        analyzer.client.messages.create = MagicMock(return_value=mock_response)

        changes = [
            VisualChange(
                id="change1",
                category=ChangeCategory.LAYOUT,
                intent=ChangeIntent.INTENTIONAL,
                severity=Severity.MINOR,
                element=None,
                bounds_baseline={"x": 100, "y": 200, "width": 80, "height": 40},
                bounds_current={"x": 120, "y": 200, "width": 80, "height": 40},
                property_name="position",
                baseline_value="x: 100",
                current_value="x: 120",
                description="Button moved",
                root_cause=None,
                impact_assessment="Minor layout shift",
                recommendation="Verify intentional",
                confidence=0.8,
                related_commit=None,
            )
        ]

        result = await analyzer.predict_user_impact(changes)

        assert result["overall_impact"] == "medium"
        assert "checkout" in result["affected_user_flows"]


class TestCreateSemanticAnalyzer:
    """Tests for the factory function."""

    def test_create_semantic_analyzer_default(self):
        """Test creating analyzer with defaults."""
        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = MagicMock()
        mock_settings.anthropic_api_key.get_secret_value.return_value = "test-key"

        with patch("src.visual_ai.semantic_analyzer.get_settings", return_value=mock_settings):
            with patch("src.visual_ai.semantic_analyzer.anthropic.Anthropic"):
                with patch("src.visual_ai.semantic_analyzer.get_model_id", return_value="claude-sonnet-4-5"):
                    analyzer = create_semantic_analyzer()
                    assert isinstance(analyzer, SemanticAnalyzer)

    def test_create_semantic_analyzer_with_model(self):
        """Test creating analyzer with specific model."""
        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = MagicMock()
        mock_settings.anthropic_api_key.get_secret_value.return_value = "test-key"

        with patch("src.visual_ai.semantic_analyzer.get_settings", return_value=mock_settings):
            with patch("src.visual_ai.semantic_analyzer.anthropic.Anthropic"):
                with patch("src.visual_ai.semantic_analyzer.get_model_id", return_value="custom-model"):
                    analyzer = create_semantic_analyzer(model="custom-model")
                    assert analyzer.model == "custom-model"


class TestSemanticAnalyzerErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.anthropic_api_key = MagicMock()
        settings.anthropic_api_key.get_secret_value.return_value = "test-key"
        return settings

    @pytest.fixture
    def analyzer(self, mock_settings):
        with patch("src.visual_ai.semantic_analyzer.get_settings", return_value=mock_settings):
            with patch("src.visual_ai.semantic_analyzer.anthropic.Anthropic"):
                with patch("src.visual_ai.semantic_analyzer.get_model_id", return_value="claude-sonnet-4-5"):
                    return SemanticAnalyzer()

    def test_parse_json_response_invalid(self, analyzer):
        """Test handling invalid JSON response."""
        with pytest.raises(json.JSONDecodeError):
            analyzer._parse_json_response("This is not valid JSON")

    @pytest.mark.asyncio
    async def test_classify_intent_api_error(self, analyzer):
        """Test classify_change_intent handles API errors."""
        analyzer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        change = VisualChange(
            id="test",
            category=ChangeCategory.LAYOUT,
            intent=ChangeIntent.UNKNOWN,
            severity=Severity.MINOR,
            element=None,
            bounds_baseline=None,
            bounds_current=None,
            property_name=None,
            baseline_value=None,
            current_value=None,
            description="Test",
            root_cause=None,
            impact_assessment="Test",
            recommendation="Test",
            confidence=0.5,
            related_commit=None,
        )

        result = await analyzer.classify_change_intent(
            change,
            git_diff="some diff",
            pr_description="some description",
        )

        # Should return original intent on error
        assert result == ChangeIntent.UNKNOWN

    @pytest.mark.asyncio
    async def test_generate_description_api_error(self, analyzer):
        """Test generate_change_description handles API errors."""
        analyzer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        result = await analyzer.generate_change_description(
            b"\x89PNG\r\n\x1a\nbaseline",
            b"\x89PNG\r\n\x1a\ncurrent",
        )

        assert result == "Unable to generate description"

    @pytest.mark.asyncio
    async def test_predict_impact_api_error(self, analyzer):
        """Test predict_user_impact handles API errors."""
        analyzer.client.messages.create = MagicMock(side_effect=Exception("API Error"))

        changes = [
            VisualChange(
                id="test",
                category=ChangeCategory.STYLE,
                intent=ChangeIntent.INTENTIONAL,
                severity=Severity.INFO,
                element=None,
                bounds_baseline=None,
                bounds_current=None,
                property_name="color",
                baseline_value="#000",
                current_value="#333",
                description="Color change",
                root_cause=None,
                impact_assessment="Minor",
                recommendation="Review",
                confidence=0.8,
                related_commit=None,
            )
        ]

        result = await analyzer.predict_user_impact(changes)

        assert result["overall_impact"] == "unknown"
        assert "error" in result


class TestSemanticAnalyzerIntegration:
    """Integration-style tests for SemanticAnalyzer."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.anthropic_api_key = MagicMock()
        settings.anthropic_api_key.get_secret_value.return_value = "test-key"
        return settings

    @pytest.fixture
    def analyzer(self, mock_settings):
        with patch("src.visual_ai.semantic_analyzer.get_settings", return_value=mock_settings):
            with patch("src.visual_ai.semantic_analyzer.anthropic.Anthropic"):
                with patch("src.visual_ai.semantic_analyzer.get_model_id", return_value="claude-sonnet-4-5"):
                    return SemanticAnalyzer()

    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, analyzer):
        """Test a complete analysis workflow."""
        # Mock API response with changes
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "changes": [
                {
                    "id": "change-1",
                    "category": "style",
                    "description": "Button color changed from blue to green",
                    "severity": 1,
                    "intent": "intentional",
                    "element": "button.primary",
                    "bounds_baseline": {"x": 100, "y": 200, "width": 80, "height": 40},
                    "bounds_current": {"x": 100, "y": 200, "width": 80, "height": 40},
                    "baseline_value": "#0066cc",
                    "current_value": "#00cc66",
                    "root_cause": "CSS variable update",
                    "impact_assessment": "Visual only, no functional impact",
                    "recommendation": "Accept change",
                    "confidence": 0.95,
                    "related_files": ["theme.css"]
                }
            ],
            "overall_assessment": "Minor styling update matching PR description",
            "auto_approval_recommended": True,
            "approval_confidence": 0.92,
            "blocking_issues": [],
            "suggestions": ["Verify color accessibility"]
        }))]
        mock_response.usage.input_tokens = 1500
        mock_response.usage.output_tokens = 300
        analyzer.client.messages.create = MagicMock(return_value=mock_response)

        result = await analyzer.analyze_changes(
            baseline_screenshot=b"\x89PNG\r\n\x1a\nbaseline",
            current_screenshot=b"\x89PNG\r\n\x1a\ncurrent",
            pr_description="Update primary button color to green",
        )

        assert isinstance(result, SemanticAnalysis)
        assert len(result.changes) == 1
        assert result.changes[0].category == ChangeCategory.STYLE
        assert result.auto_approval_recommended is True
        assert result.approval_confidence > 0.9
