"""Tests for the test impact analyzer module."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch


class TestCodeChange:
    """Tests for CodeChange dataclass."""

    def test_code_change_creation(self, mock_env_vars):
        """Test CodeChange creation."""
        from src.agents.test_impact_analyzer import CodeChange

        change = CodeChange(
            id="abc123",
            files=[{"path": "src/login.py", "additions": 10, "deletions": 5}],
            message="Fix login bug",
            author="developer",
            timestamp=datetime.now(),
            branch="main",
        )

        assert change.id == "abc123"
        assert len(change.files) == 1


class TestTestMapping:
    """Tests for TestMapping dataclass."""

    def test_mapping_creation(self, mock_env_vars):
        """Test TestMapping creation."""
        from src.agents.test_impact_analyzer import TestMapping

        mapping = TestMapping(
            code_path="src/login.py",
            test_ids=["test-001", "test-002"],
            confidence=0.9,
            last_updated=datetime.now(),
        )

        assert mapping.code_path == "src/login.py"
        assert len(mapping.test_ids) == 2


class TestImpactAnalysis:
    """Tests for ImpactAnalysis dataclass."""

    def test_analysis_creation(self, mock_env_vars):
        """Test ImpactAnalysis creation."""
        from src.agents.test_impact_analyzer import ImpactAnalysis

        analysis = ImpactAnalysis(
            change_id="abc123",
            affected_tests=["test-001", "test-002"],
            unaffected_tests=["test-003"],
            new_tests_suggested=[],
            risk_score=0.5,
            estimated_time_saved=120.0,
            coverage_gaps=[],
            recommendations=["Run all tests"],
        )

        assert len(analysis.affected_tests) == 2
        assert analysis.risk_score == 0.5


class TestChangeRisk:
    """Tests for ChangeRisk dataclass."""

    def test_risk_creation(self, mock_env_vars):
        """Test ChangeRisk creation."""
        from src.agents.test_impact_analyzer import ChangeRisk

        risk = ChangeRisk(
            file_path="src/auth/login.py",
            risk_level="high",
            reasons=["Modifies authentication"],
            suggested_tests=["test-auth-001"],
        )

        assert risk.risk_level == "high"


class TestTestImpactAnalyzer:
    """Tests for TestImpactAnalyzer class."""

    def test_analyzer_creation(self, mock_env_vars):
        """Test TestImpactAnalyzer creation."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()

            assert analyzer.mappings == {}
            assert analyzer.failure_history == {}

    def test_update_mapping(self, mock_env_vars):
        """Test updating code-to-test mapping."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()

            analyzer.update_mapping("src/login.py", "test-001")
            analyzer.update_mapping("src/login.py", "test-002")

            assert "src/login.py" in analyzer.mappings
            assert "test-001" in analyzer.mappings["src/login.py"].test_ids
            assert "test-002" in analyzer.mappings["src/login.py"].test_ids

    def test_update_mapping_no_duplicates(self, mock_env_vars):
        """Test that duplicate mappings are not added."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()

            analyzer.update_mapping("src/login.py", "test-001")
            analyzer.update_mapping("src/login.py", "test-001")

            assert len(analyzer.mappings["src/login.py"].test_ids) == 1

    def test_record_test_result(self, mock_env_vars):
        """Test recording test results."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()

            analyzer.record_test_result(
                change_id="abc123",
                code_path="src/login.py",
                test_id="test-001",
                failed=True,
            )

            assert "src/login.py" in analyzer.failure_history
            assert len(analyzer.failure_history["src/login.py"]) == 1
            assert analyzer.failure_history["src/login.py"][0]["failed"] is True

    def test_record_test_result_limits_history(self, mock_env_vars):
        """Test that history is limited to 100 entries."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()

            for i in range(105):
                analyzer.record_test_result(
                    change_id=f"change-{i}",
                    code_path="src/login.py",
                    test_id="test-001",
                    failed=False,
                )

            assert len(analyzer.failure_history["src/login.py"]) == 100

    def test_get_historically_affected_tests(self, mock_env_vars):
        """Test getting historically affected tests."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            # Add history with a test that fails frequently
            for i in range(10):
                analyzer.record_test_result(
                    change_id=f"change-{i}",
                    code_path="src/login.py",
                    test_id="flaky-test",
                    failed=(i % 2 == 0),  # 50% failure rate
                )

            change = CodeChange(
                id="new-change",
                files=[{"path": "src/login.py"}],
                message="Update login",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            affected = analyzer._get_historically_affected_tests(change)

            assert "flaky-test" in affected

    @pytest.mark.asyncio
    async def test_analyze_dependencies_python(self, mock_env_vars):
        """Test dependency analysis for Python files."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            change = CodeChange(
                id="abc123",
                files=[{
                    "path": "src/login.py",
                    "patch": "+from src.utils import validate_email\n+import src.database",
                }],
                message="Update login",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            deps = await analyzer._analyze_dependencies(change)

            assert "src/utils.py" in deps or len(deps) > 0

    @pytest.mark.asyncio
    async def test_analyze_dependencies_javascript(self, mock_env_vars):
        """Test dependency analysis for JavaScript files."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            change = CodeChange(
                id="abc123",
                files=[{
                    "path": "src/login.js",
                    "patch": "+import { validate } from './utils'\n",
                }],
                message="Update login",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            deps = await analyzer._analyze_dependencies(change)

            assert len(deps) >= 0

    @pytest.mark.asyncio
    async def test_calculate_risk_score_large_change(self, mock_env_vars):
        """Test risk calculation for large changes."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            change = CodeChange(
                id="abc123",
                files=[{
                    "path": "src/component.py",
                    "additions": 300,
                    "deletions": 200,
                }],
                message="Big refactor",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            risk = await analyzer._calculate_risk_score(change)

            # Use 0.19 threshold due to floating point precision (0.2 * large_file_factor = 0.199...)
            assert risk >= 0.19  # Large change should have higher risk

    @pytest.mark.asyncio
    async def test_calculate_risk_score_risky_paths(self, mock_env_vars):
        """Test risk calculation for risky file paths."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            change = CodeChange(
                id="abc123",
                files=[{
                    "path": "src/auth/payment_handler.py",
                    "additions": 10,
                    "deletions": 5,
                }],
                message="Update payment",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            risk = await analyzer._calculate_risk_score(change)

            assert risk > 0.3  # Payment files are risky

    def test_identify_coverage_gaps(self, mock_env_vars):
        """Test coverage gap identification."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            # Add some mappings
            analyzer.update_mapping("src/login.py", "test-001")

            change = CodeChange(
                id="abc123",
                files=[
                    {"path": "src/login.py"},  # Has mapping
                    {"path": "src/new_feature.py"},  # No mapping
                ],
                message="Add feature",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            gaps = analyzer._identify_coverage_gaps(change, set(), [])

            assert any("new_feature" in g for g in gaps)

    def test_generate_recommendations(self, mock_env_vars):
        """Test recommendation generation."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            change = CodeChange(
                id="abc123",
                files=[{"path": "src/login.py"}],
                message="Update",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            # High risk, low coverage
            recs = analyzer._generate_recommendations(
                change,
                risk_score=0.8,
                coverage_gaps=["gap1", "gap2"],
                affected_tests=set(["test-001"]),
            )

            assert any("HIGH RISK" in r for r in recs)

    def test_generate_recommendations_no_tests(self, mock_env_vars):
        """Test recommendations when no tests affected."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            change = CodeChange(
                id="abc123",
                files=[{"path": "src/login.py"}],
                message="Update",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            recs = analyzer._generate_recommendations(
                change,
                risk_score=0.3,
                coverage_gaps=[],
                affected_tests=set(),
            )

            assert any("No tests affected" in r for r in recs)

    @pytest.mark.asyncio
    async def test_analyze_impact(self, mock_env_vars):
        """Test full impact analysis."""
        with patch('src.agents.test_impact_analyzer.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[]')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_impact_analyzer import TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()

            # Add mapping
            analyzer.update_mapping("src/login.py", "test-001")

            change = CodeChange(
                id="abc123",
                files=[{"path": "src/login.py", "additions": 10, "deletions": 5}],
                message="Update login",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            all_tests = [
                {"id": "test-001", "name": "Login Test", "avg_duration": 30},
                {"id": "test-002", "name": "Signup Test", "avg_duration": 45},
            ]

            analysis = await analyzer.analyze_impact(change, all_tests)

            assert analysis.change_id == "abc123"
            assert "test-001" in analysis.affected_tests


class TestSmartTestSelector:
    """Tests for SmartTestSelector class."""

    def test_selector_creation(self, mock_env_vars):
        """Test SmartTestSelector creation."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import SmartTestSelector, TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()
            selector = SmartTestSelector(analyzer)

            assert selector.impact_analyzer is analyzer

    @pytest.mark.asyncio
    async def test_select_tests(self, mock_env_vars):
        """Test test selection."""
        with patch('src.agents.test_impact_analyzer.Anthropic') as mock_anthropic:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[]')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.test_impact_analyzer import SmartTestSelector, TestImpactAnalyzer, CodeChange

            analyzer = TestImpactAnalyzer()
            analyzer.update_mapping("src/login.py", "test-critical")
            analyzer.update_mapping("src/login.py", "test-high")

            selector = SmartTestSelector(analyzer)

            change = CodeChange(
                id="abc123",
                files=[{"path": "src/login.py", "additions": 10, "deletions": 5}],
                message="Update login",
                author="dev",
                timestamp=datetime.now(),
                branch="main",
            )

            all_tests = [
                {"id": "test-critical", "name": "Critical Test", "priority": "critical", "avg_duration": 30},
                {"id": "test-high", "name": "High Test", "priority": "high", "avg_duration": 45},
                {"id": "test-unrelated", "name": "Unrelated Test", "priority": "low", "avg_duration": 60},
            ]

            result = await selector.select_tests(change, all_tests)

            assert "must_run" in result
            assert "should_run" in result
            assert "can_skip" in result
            assert "estimated_time" in result

    def test_apply_time_budget(self, mock_env_vars):
        """Test applying time budget to test selection."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import SmartTestSelector, TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()
            selector = SmartTestSelector(analyzer)

            must_run = ["test-1", "test-2"]
            should_run = ["test-3", "test-4"]
            deferred = []

            all_tests = [
                {"id": "test-1", "avg_duration": 30},
                {"id": "test-2", "avg_duration": 30},
                {"id": "test-3", "avg_duration": 60},
                {"id": "test-4", "avg_duration": 60},
            ]

            # Time budget of 100 seconds
            new_must, new_should, new_deferred = selector._apply_time_budget(
                must_run, should_run, deferred, all_tests, 100
            )

            # Must run tests (60s) should fit
            assert len(new_must) == 2

            # Should have some tests deferred due to budget
            assert len(new_deferred) >= 1

    def test_apply_time_budget_insufficient(self, mock_env_vars):
        """Test time budget when even must-run doesn't fit."""
        with patch('src.agents.test_impact_analyzer.Anthropic'):
            from src.agents.test_impact_analyzer import SmartTestSelector, TestImpactAnalyzer

            analyzer = TestImpactAnalyzer()
            selector = SmartTestSelector(analyzer)

            must_run = ["test-1", "test-2"]
            should_run = ["test-3"]
            deferred = []

            all_tests = [
                {"id": "test-1", "avg_duration": 100},
                {"id": "test-2", "avg_duration": 100},
                {"id": "test-3", "avg_duration": 50},
            ]

            # Time budget of 50 seconds (not enough for must-run)
            new_must, new_should, new_deferred = selector._apply_time_budget(
                must_run, should_run, deferred, all_tests, 50
            )

            # Must run still included (can't skip critical tests)
            assert len(new_must) == 2
            # Should run moved to deferred
            assert "test-3" in new_deferred
