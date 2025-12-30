"""Tests for the root cause analyzer module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_failure_categories(self, mock_env_vars):
        """Test FailureCategory enum values."""
        from src.agents.root_cause_analyzer import FailureCategory

        assert FailureCategory.UI_CHANGE.value == "ui_change"
        assert FailureCategory.NETWORK_ERROR.value == "network_error"
        assert FailureCategory.TIMING_ISSUE.value == "timing_issue"
        assert FailureCategory.DATA_MISMATCH.value == "data_mismatch"
        assert FailureCategory.REAL_BUG.value == "real_bug"
        assert FailureCategory.ENVIRONMENT.value == "environment"
        assert FailureCategory.TEST_DEFECT.value == "test_defect"
        assert FailureCategory.UNKNOWN.value == "unknown"


class TestRootCauseResult:
    """Tests for RootCauseResult dataclass."""

    def test_result_creation(self, mock_env_vars):
        """Test RootCauseResult creation."""
        from src.agents.root_cause_analyzer import RootCauseResult, FailureCategory

        result = RootCauseResult(
            category=FailureCategory.TIMING_ISSUE,
            confidence=0.85,
            summary="Element not found due to timing",
            detailed_analysis="The element was not ready in time",
            suggested_fix="Add explicit wait",
        )

        assert result.category == FailureCategory.TIMING_ISSUE
        assert result.confidence == 0.85
        assert result.is_flaky is False
        assert result.auto_healable is False

    def test_result_with_all_fields(self, mock_env_vars):
        """Test RootCauseResult with all fields."""
        from src.agents.root_cause_analyzer import RootCauseResult, FailureCategory

        result = RootCauseResult(
            category=FailureCategory.UI_CHANGE,
            confidence=0.9,
            summary="Button moved",
            detailed_analysis="The submit button was relocated",
            suggested_fix="Update selector",
            related_failures=["test-001", "test-002"],
            code_location="src/login.js:45",
            is_flaky=False,
            flaky_confidence=0.0,
            historical_occurrences=3,
            auto_healable=True,
            healing_suggestion={"type": "update_locator"},
        )

        assert len(result.related_failures) == 2
        assert result.code_location == "src/login.js:45"
        assert result.auto_healable is True


class TestFailureContext:
    """Tests for FailureContext dataclass."""

    def test_context_creation(self, mock_env_vars):
        """Test FailureContext creation."""
        from src.agents.root_cause_analyzer import FailureContext

        context = FailureContext(
            test_id="test-001",
            test_name="Login Test",
            error_message="Element not found: #submit-btn",
        )

        assert context.test_id == "test-001"
        assert context.test_name == "Login Test"
        assert context.stack_trace is None
        assert context.screenshot_base64 is None

    def test_context_with_all_fields(self, mock_env_vars):
        """Test FailureContext with all fields."""
        from src.agents.root_cause_analyzer import FailureContext

        context = FailureContext(
            test_id="test-001",
            test_name="Login Test",
            error_message="Element not found",
            stack_trace="at line 45\nat line 30",
            screenshot_base64="base64data",
            html_snapshot="<html>...</html>",
            network_logs=[{"url": "/api", "status": 500}],
            console_logs=["Error: API failed"],
            step_history=[{"action": "click", "target": "#btn"}],
            expected_vs_actual={"expected": "Dashboard", "actual": "Login"},
            previous_runs=[{"passed": True}, {"passed": False}],
            code_diff="+added line\n-removed line",
            environment={"browser": "chrome", "os": "linux"},
        )

        assert len(context.network_logs) == 1
        assert len(context.console_logs) == 1


class TestRootCauseAnalyzer:
    """Tests for RootCauseAnalyzer class."""

    def test_analyzer_creation(self, mock_env_vars):
        """Test RootCauseAnalyzer creation."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer

            analyzer = RootCauseAnalyzer()

            assert analyzer.client is not None
            assert analyzer.failure_history == {}

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer

            analyzer = RootCauseAnalyzer()
            prompt = analyzer._get_system_prompt()

            assert "root cause" in prompt.lower()
            assert "JSON" in prompt

    def test_build_analysis_prompt(self, mock_env_vars):
        """Test building analysis prompt."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()
            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            messages = analyzer._build_analysis_prompt(context)

            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_build_analysis_prompt_with_stack_trace(self, mock_env_vars):
        """Test building prompt with stack trace."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()
            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
                stack_trace="at click() line 45\nat test() line 30",
            )

            messages = analyzer._build_analysis_prompt(context)
            content = messages[0]["content"]

            # Should include stack trace section
            assert any("Stack Trace" in str(part) for part in content)

    def test_build_analysis_prompt_with_screenshot(self, mock_env_vars):
        """Test building prompt with screenshot."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()
            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
                screenshot_base64="base64_image_data",
            )

            messages = analyzer._build_analysis_prompt(context)
            content = messages[0]["content"]

            # Should include image
            assert any(part.get("type") == "image" for part in content if isinstance(part, dict))

    def test_build_analysis_prompt_with_network_logs(self, mock_env_vars):
        """Test building prompt with network logs."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()
            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="API error",
                network_logs=[
                    {"url": "/api/login", "status": 500},
                    {"url": "/api/users", "status": 200},
                ],
            )

            messages = analyzer._build_analysis_prompt(context)
            content = messages[0]["content"]

            # Should include failed network requests
            assert any("Failed Network" in str(part) for part in content)

    def test_parse_analysis_success(self, mock_env_vars):
        """Test parsing successful analysis."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext, FailureCategory

            analyzer = RootCauseAnalyzer()
            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            response_text = '''
            Here's my analysis:
            {
                "category": "timing_issue",
                "confidence": 0.85,
                "summary": "Element not ready in time",
                "detailed_analysis": "The page was still loading",
                "suggested_fix": "Add explicit wait",
                "code_location": null,
                "is_flaky": true,
                "flaky_confidence": 0.7
            }
            '''

            result = analyzer._parse_analysis(response_text, context)

            assert result.category == FailureCategory.TIMING_ISSUE
            assert result.confidence == 0.85
            assert result.is_flaky is True

    def test_parse_analysis_no_json(self, mock_env_vars):
        """Test parsing when no JSON is found."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext, FailureCategory

            analyzer = RootCauseAnalyzer()
            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            response_text = "I couldn't determine the cause."

            result = analyzer._parse_analysis(response_text, context)

            assert result.category == FailureCategory.UNKNOWN
            assert result.confidence == 0.3

    def test_create_failure_fingerprint(self, mock_env_vars):
        """Test failure fingerprint creation."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()

            context1 = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            context2 = FailureContext(
                test_id="test-002",  # Different ID
                test_name="Login Test",  # Same name
                error_message="Element not found",  # Same error
            )

            fp1 = analyzer._create_failure_fingerprint(context1)
            fp2 = analyzer._create_failure_fingerprint(context2)

            # Same test name and error should produce same fingerprint
            assert fp1 == fp2

    def test_record_failure(self, mock_env_vars):
        """Test recording a failure."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, FailureContext, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            result = RootCauseResult(
                category=FailureCategory.TIMING_ISSUE,
                confidence=0.85,
                summary="Timing issue",
                detailed_analysis="Details",
                suggested_fix="Add wait",
            )

            analyzer._record_failure(context, result)
            fingerprint = analyzer._create_failure_fingerprint(context)

            assert fingerprint in analyzer.failure_history
            assert len(analyzer.failure_history[fingerprint]) == 1

    def test_record_failure_limits_history(self, mock_env_vars):
        """Test that failure history is limited to 50 entries."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, FailureContext, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            result = RootCauseResult(
                category=FailureCategory.TIMING_ISSUE,
                confidence=0.85,
                summary="Timing issue",
                detailed_analysis="Details",
                suggested_fix="Add wait",
            )

            # Record 55 failures
            for _ in range(55):
                analyzer._record_failure(context, result)

            fingerprint = analyzer._create_failure_fingerprint(context)

            # Should be limited to 50
            assert len(analyzer.failure_history[fingerprint]) == 50

    def test_enhance_with_history(self, mock_env_vars):
        """Test enhancing result with history."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, FailureContext, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            result = RootCauseResult(
                category=FailureCategory.TIMING_ISSUE,
                confidence=0.85,
                summary="Timing issue",
                detailed_analysis="Details",
                suggested_fix="Add wait",
            )

            # Add some history with recovery
            fingerprint = analyzer._create_failure_fingerprint(context)
            analyzer.failure_history[fingerprint] = [
                {"test_id": "old-1", "recovered": True},
                {"test_id": "old-2", "recovered": True},
                {"test_id": "old-3", "recovered": False},
            ]

            enhanced = analyzer._enhance_with_history(result, context)

            assert enhanced.historical_occurrences == 3
            assert len(enhanced.related_failures) <= 5

    def test_check_auto_healability_ui_change(self, mock_env_vars):
        """Test auto-healability check for UI change."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, FailureContext, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            result = RootCauseResult(
                category=FailureCategory.UI_CHANGE,
                confidence=0.9,
                summary="Button moved",
                detailed_analysis="Details",
                suggested_fix="Update selector",
            )

            enhanced = analyzer._check_auto_healability(result, context)

            assert enhanced.auto_healable is True
            assert enhanced.healing_suggestion["type"] == "update_locator"

    def test_check_auto_healability_timing(self, mock_env_vars):
        """Test auto-healability check for timing issue."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, FailureContext, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            result = RootCauseResult(
                category=FailureCategory.TIMING_ISSUE,
                confidence=0.9,
                summary="Element not ready",
                detailed_analysis="Details",
                suggested_fix="Add wait",
            )

            enhanced = analyzer._check_auto_healability(result, context)

            assert enhanced.auto_healable is True
            assert enhanced.healing_suggestion["type"] == "add_wait"

    def test_check_auto_healability_real_bug(self, mock_env_vars):
        """Test auto-healability check for real bug (not healable)."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, FailureContext, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Login failed",
            )

            result = RootCauseResult(
                category=FailureCategory.REAL_BUG,
                confidence=0.9,
                summary="Login logic broken",
                detailed_analysis="Details",
                suggested_fix="Fix the bug",
            )

            enhanced = analyzer._check_auto_healability(result, context)

            assert enhanced.auto_healable is False

    @pytest.mark.asyncio
    async def test_analyze(self, mock_env_vars):
        """Test full analysis."""
        with patch('src.agents.root_cause_analyzer.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='''
            {
                "category": "timing_issue",
                "confidence": 0.85,
                "summary": "Element not ready",
                "detailed_analysis": "Page was still loading",
                "suggested_fix": "Add wait",
                "code_location": null,
                "is_flaky": false,
                "flaky_confidence": 0.0
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()

            context = FailureContext(
                test_id="test-001",
                test_name="Login Test",
                error_message="Element not found",
            )

            result = await analyzer.analyze(context)

            assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_analyze_batch(self, mock_env_vars):
        """Test batch analysis."""
        with patch('src.agents.root_cause_analyzer.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='''
            {
                "category": "timing_issue",
                "confidence": 0.85,
                "summary": "Timing issue",
                "detailed_analysis": "Details",
                "suggested_fix": "Add wait",
                "code_location": null,
                "is_flaky": false,
                "flaky_confidence": 0.0
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.root_cause_analyzer import RootCauseAnalyzer, FailureContext

            analyzer = RootCauseAnalyzer()

            failures = [
                FailureContext(
                    test_id=f"test-{i}",
                    test_name=f"Test {i}",
                    error_message="Element not found",
                )
                for i in range(3)
            ]

            results, insights = await analyzer.analyze_batch(failures)

            assert len(results) == 3
            assert insights["total_failures"] == 3
            assert "category_distribution" in insights

    def test_generate_batch_recommendation_timing(self, mock_env_vars):
        """Test batch recommendation for timing issues."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            # Majority timing issues
            results = [
                RootCauseResult(
                    category=FailureCategory.TIMING_ISSUE,
                    confidence=0.8,
                    summary="Timing",
                    detailed_analysis="Details",
                    suggested_fix="Add wait",
                )
                for _ in range(6)
            ] + [
                RootCauseResult(
                    category=FailureCategory.UI_CHANGE,
                    confidence=0.8,
                    summary="UI",
                    detailed_analysis="Details",
                    suggested_fix="Update",
                )
                for _ in range(4)
            ]

            rec = analyzer._generate_batch_recommendation(results)

            assert "flakiness" in rec.lower() or "waits" in rec.lower()

    def test_generate_batch_recommendation_ui_change(self, mock_env_vars):
        """Test batch recommendation for UI changes."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            # Many UI changes
            results = [
                RootCauseResult(
                    category=FailureCategory.UI_CHANGE,
                    confidence=0.8,
                    summary="UI",
                    detailed_analysis="Details",
                    suggested_fix="Update",
                )
                for _ in range(4)
            ] + [
                RootCauseResult(
                    category=FailureCategory.TIMING_ISSUE,
                    confidence=0.8,
                    summary="Timing",
                    detailed_analysis="Details",
                    suggested_fix="Wait",
                )
                for _ in range(6)
            ]

            rec = analyzer._generate_batch_recommendation(results)

            # Should mention UI changes or locators
            assert "ui" in rec.lower() or "locator" in rec.lower() or "flaki" in rec.lower()

    def test_generate_batch_recommendation_real_bug(self, mock_env_vars):
        """Test batch recommendation when real bugs found."""
        with patch('src.agents.root_cause_analyzer.Anthropic'):
            from src.agents.root_cause_analyzer import (
                RootCauseAnalyzer, RootCauseResult, FailureCategory
            )

            analyzer = RootCauseAnalyzer()

            results = [
                RootCauseResult(
                    category=FailureCategory.REAL_BUG,
                    confidence=0.9,
                    summary="Bug",
                    detailed_analysis="Details",
                    suggested_fix="Fix",
                ),
                RootCauseResult(
                    category=FailureCategory.TIMING_ISSUE,
                    confidence=0.8,
                    summary="Timing",
                    detailed_analysis="Details",
                    suggested_fix="Wait",
                ),
            ]

            rec = analyzer._generate_batch_recommendation(results)

            assert "bug" in rec.lower()
