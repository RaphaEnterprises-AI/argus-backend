"""Tests for the MR/PR analyzer agent module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch target for Anthropic client in base agent
ANTHROPIC_PATCH = "anthropic.Anthropic"


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_type_values(self, mock_env_vars):
        """Test ChangeType enum values."""
        from src.agents.mr_analyzer import ChangeType

        assert ChangeType.FEATURE.value == "feature"
        assert ChangeType.BUGFIX.value == "bugfix"
        assert ChangeType.REFACTOR.value == "refactor"
        assert ChangeType.TEST.value == "test"
        assert ChangeType.DOCS.value == "docs"
        assert ChangeType.CONFIG.value == "config"
        assert ChangeType.DEPENDENCY.value == "dependency"
        assert ChangeType.SECURITY.value == "security"
        assert ChangeType.PERFORMANCE.value == "performance"
        assert ChangeType.UNKNOWN.value == "unknown"

    def test_change_type_is_string_enum(self, mock_env_vars):
        """Test that ChangeType is a string enum."""
        from src.agents.mr_analyzer import ChangeType

        # str enum allows direct string comparison
        assert ChangeType.FEATURE == "feature"
        # Value comparison
        assert ChangeType.BUGFIX.value == "bugfix"


class TestTestPriority:
    """Tests for TestPriority enum."""

    def test_priority_values(self, mock_env_vars):
        """Test TestPriority enum values."""
        from src.agents.mr_analyzer import TestPriority

        assert TestPriority.CRITICAL.value == "critical"
        assert TestPriority.HIGH.value == "high"
        assert TestPriority.MEDIUM.value == "medium"
        assert TestPriority.LOW.value == "low"

    @pytest.mark.parametrize(
        "priority_str,expected",
        [
            ("critical", "critical"),
            ("high", "high"),
            ("medium", "medium"),
            ("low", "low"),
        ],
    )
    def test_priority_from_string(self, mock_env_vars, priority_str, expected):
        """Test creating TestPriority from string."""
        from src.agents.mr_analyzer import TestPriority

        priority = TestPriority(priority_str)
        assert priority.value == expected


class TestTestType:
    """Tests for TestType enum."""

    def test_test_type_values(self, mock_env_vars):
        """Test TestType enum values."""
        from src.agents.mr_analyzer import TestType

        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.E2E.value == "e2e"
        assert TestType.API.value == "api"
        assert TestType.VISUAL.value == "visual"
        assert TestType.PERFORMANCE.value == "performance"
        assert TestType.SECURITY.value == "security"

    def test_test_type_is_string_enum(self, mock_env_vars):
        """Test that TestType is a string enum."""
        from src.agents.mr_analyzer import TestType

        assert TestType.UNIT == "unit"
        # Value comparison
        assert TestType.E2E.value == "e2e"


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_file_change_creation_minimal(self, mock_env_vars):
        """Test FileChange creation with minimal fields."""
        from src.agents.mr_analyzer import FileChange

        change = FileChange(
            path="src/auth.py",
            status="modified",
        )

        assert change.path == "src/auth.py"
        assert change.status == "modified"
        assert change.additions == 0
        assert change.deletions == 0
        assert change.patch is None
        assert change.previous_path is None

    def test_file_change_creation_full(self, mock_env_vars):
        """Test FileChange creation with all fields."""
        from src.agents.mr_analyzer import FileChange

        change = FileChange(
            path="src/new_auth.py",
            status="renamed",
            additions=50,
            deletions=20,
            patch="+ def new_function():\n+     pass",
            previous_path="src/auth.py",
        )

        assert change.path == "src/new_auth.py"
        assert change.status == "renamed"
        assert change.additions == 50
        assert change.deletions == 20
        assert change.patch is not None
        assert change.previous_path == "src/auth.py"

    @pytest.mark.parametrize(
        "status",
        ["added", "modified", "removed", "renamed"],
    )
    def test_file_change_status_values(self, mock_env_vars, status):
        """Test FileChange with different status values."""
        from src.agents.mr_analyzer import FileChange

        change = FileChange(path="test.py", status=status)
        assert change.status == status


class TestChangeAnalysis:
    """Tests for ChangeAnalysis dataclass."""

    def test_change_analysis_creation_minimal(self, mock_env_vars):
        """Test ChangeAnalysis creation with minimal fields."""
        from src.agents.mr_analyzer import ChangeAnalysis, ChangeType

        analysis = ChangeAnalysis(
            file_path="src/login.py",
            change_type=ChangeType.FEATURE,
            risk_level="medium",
            affected_areas=["authentication"],
            semantic_summary="Added new login feature",
        )

        assert analysis.file_path == "src/login.py"
        assert analysis.change_type == ChangeType.FEATURE
        assert analysis.risk_level == "medium"
        assert analysis.affected_areas == ["authentication"]
        assert analysis.semantic_summary == "Added new login feature"
        assert analysis.functions_changed == []
        assert analysis.imports_changed == []
        assert analysis.potential_impacts == []

    def test_change_analysis_creation_full(self, mock_env_vars):
        """Test ChangeAnalysis creation with all fields."""
        from src.agents.mr_analyzer import ChangeAnalysis, ChangeType

        analysis = ChangeAnalysis(
            file_path="src/payment.py",
            change_type=ChangeType.SECURITY,
            risk_level="critical",
            affected_areas=["payments", "billing"],
            semantic_summary="Updated payment validation",
            functions_changed=["validate_card", "process_payment"],
            imports_changed=["stripe", "cryptography"],
            potential_impacts=["May affect checkout flow", "Requires PCI compliance review"],
        )

        assert analysis.file_path == "src/payment.py"
        assert analysis.change_type == ChangeType.SECURITY
        assert analysis.risk_level == "critical"
        assert len(analysis.functions_changed) == 2
        assert len(analysis.imports_changed) == 2
        assert len(analysis.potential_impacts) == 2


class TestMRAnalysis:
    """Tests for MRAnalysis dataclass."""

    def test_mr_analysis_creation(self, mock_env_vars):
        """Test MRAnalysis creation."""
        from src.agents.mr_analyzer import ChangeAnalysis, ChangeType, MRAnalysis

        file_analysis = ChangeAnalysis(
            file_path="src/app.py",
            change_type=ChangeType.FEATURE,
            risk_level="medium",
            affected_areas=["core"],
            semantic_summary="New feature added",
        )

        analysis = MRAnalysis(
            file_analyses=[file_analysis],
            overall_change_type=ChangeType.FEATURE,
            overall_risk_score=0.5,
            affected_components=["core", "api"],
            affected_routes=["/api/v1/feature"],
            coverage_gaps=["Missing unit tests"],
            summary="Added new feature",
            recommendations=["Add unit tests", "Review API changes"],
        )

        assert len(analysis.file_analyses) == 1
        assert analysis.overall_change_type == ChangeType.FEATURE
        assert analysis.overall_risk_score == 0.5
        assert len(analysis.affected_components) == 2
        assert len(analysis.affected_routes) == 1
        assert len(analysis.coverage_gaps) == 1
        assert len(analysis.recommendations) == 2
        assert analysis.analyzed_at is not None

    def test_mr_analysis_to_dict(self, mock_env_vars):
        """Test MRAnalysis to_dict method."""
        from src.agents.mr_analyzer import ChangeAnalysis, ChangeType, MRAnalysis

        file_analysis = ChangeAnalysis(
            file_path="src/test.py",
            change_type=ChangeType.BUGFIX,
            risk_level="low",
            affected_areas=["testing"],
            semantic_summary="Fixed bug",
            functions_changed=["fix_issue"],
            imports_changed=[],
            potential_impacts=["Fixes issue #123"],
        )

        analysis = MRAnalysis(
            file_analyses=[file_analysis],
            overall_change_type=ChangeType.BUGFIX,
            overall_risk_score=0.3,
            affected_components=["testing"],
            affected_routes=[],
            coverage_gaps=[],
            summary="Bug fix",
            recommendations=["Verify fix works"],
        )

        result = analysis.to_dict()

        assert isinstance(result, dict)
        assert len(result["file_analyses"]) == 1
        assert result["file_analyses"][0]["file_path"] == "src/test.py"
        assert result["file_analyses"][0]["change_type"] == "bugfix"
        assert result["overall_change_type"] == "bugfix"
        assert result["overall_risk_score"] == 0.3
        assert "analyzed_at" in result

    def test_mr_analysis_to_dict_preserves_all_fields(self, mock_env_vars):
        """Test MRAnalysis to_dict preserves all file analysis fields."""
        from src.agents.mr_analyzer import ChangeAnalysis, ChangeType, MRAnalysis

        file_analysis = ChangeAnalysis(
            file_path="src/api.py",
            change_type=ChangeType.REFACTOR,
            risk_level="high",
            affected_areas=["api", "backend"],
            semantic_summary="Major refactor",
            functions_changed=["func1", "func2"],
            imports_changed=["module1"],
            potential_impacts=["Breaking changes possible"],
        )

        analysis = MRAnalysis(
            file_analyses=[file_analysis],
            overall_change_type=ChangeType.REFACTOR,
            overall_risk_score=0.7,
            affected_components=["api"],
            affected_routes=["/api/v1/resource"],
            coverage_gaps=["No integration tests"],
            summary="API refactor",
            recommendations=["Update API docs"],
        )

        result = analysis.to_dict()
        fa_dict = result["file_analyses"][0]

        assert fa_dict["functions_changed"] == ["func1", "func2"]
        assert fa_dict["imports_changed"] == ["module1"]
        assert fa_dict["potential_impacts"] == ["Breaking changes possible"]
        assert fa_dict["risk_level"] == "high"
        assert fa_dict["affected_areas"] == ["api", "backend"]


class TestTestSuggestion:
    """Tests for TestSuggestion dataclass."""

    def test_test_suggestion_creation_minimal(self, mock_env_vars):
        """Test TestSuggestion creation with minimal fields."""
        from src.agents.mr_analyzer import TestPriority, TestSuggestion, TestType

        suggestion = TestSuggestion(
            name="test_login_flow",
            description="Test the login authentication flow",
            test_type=TestType.INTEGRATION,
            priority=TestPriority.HIGH,
            target_files=["src/auth/login.py"],
            coverage_target="Login functionality",
            estimated_effort="medium",
        )

        assert suggestion.name == "test_login_flow"
        assert suggestion.description == "Test the login authentication flow"
        assert suggestion.test_type == TestType.INTEGRATION
        assert suggestion.priority == TestPriority.HIGH
        assert suggestion.target_files == ["src/auth/login.py"]
        assert suggestion.coverage_target == "Login functionality"
        assert suggestion.estimated_effort == "medium"
        assert suggestion.example_code is None
        assert suggestion.rationale == ""

    def test_test_suggestion_creation_full(self, mock_env_vars):
        """Test TestSuggestion creation with all fields."""
        from src.agents.mr_analyzer import TestPriority, TestSuggestion, TestType

        suggestion = TestSuggestion(
            name="test_payment_validation",
            description="Test payment card validation",
            test_type=TestType.UNIT,
            priority=TestPriority.CRITICAL,
            target_files=["src/payments/validate.py"],
            coverage_target="Card validation logic",
            estimated_effort="small",
            example_code="def test_valid_card():\n    assert validate('4111111111111111')",
            rationale="Critical payment functionality needs thorough testing",
        )

        assert suggestion.name == "test_payment_validation"
        assert suggestion.example_code is not None
        assert "def test_valid_card" in suggestion.example_code
        assert suggestion.rationale != ""

    def test_test_suggestion_to_dict_without_code(self, mock_env_vars):
        """Test TestSuggestion to_dict without example code."""
        from src.agents.mr_analyzer import TestPriority, TestSuggestion, TestType

        suggestion = TestSuggestion(
            name="test_basic",
            description="Basic test",
            test_type=TestType.UNIT,
            priority=TestPriority.LOW,
            target_files=["test.py"],
            coverage_target="Basic coverage",
            estimated_effort="trivial",
            rationale="Basic testing",
        )

        result = suggestion.to_dict()

        assert result["name"] == "test_basic"
        assert result["test_type"] == "unit"
        assert result["priority"] == "low"
        assert "example_code" not in result
        assert result["rationale"] == "Basic testing"

    def test_test_suggestion_to_dict_with_code(self, mock_env_vars):
        """Test TestSuggestion to_dict with example code."""
        from src.agents.mr_analyzer import TestPriority, TestSuggestion, TestType

        suggestion = TestSuggestion(
            name="test_with_code",
            description="Test with example",
            test_type=TestType.API,
            priority=TestPriority.MEDIUM,
            target_files=["api.py"],
            coverage_target="API endpoint",
            estimated_effort="medium",
            example_code="def test_api():\n    response = client.get('/api')",
            rationale="API needs testing",
        )

        result = suggestion.to_dict()

        assert "example_code" in result
        assert "def test_api" in result["example_code"]

    @pytest.mark.parametrize(
        "effort",
        ["trivial", "small", "medium", "large"],
    )
    def test_test_suggestion_effort_levels(self, mock_env_vars, effort):
        """Test TestSuggestion with different effort levels."""
        from src.agents.mr_analyzer import TestPriority, TestSuggestion, TestType

        suggestion = TestSuggestion(
            name="test_effort",
            description="Test effort",
            test_type=TestType.UNIT,
            priority=TestPriority.MEDIUM,
            target_files=["test.py"],
            coverage_target="Coverage",
            estimated_effort=effort,
        )

        assert suggestion.estimated_effort == effort
        assert suggestion.to_dict()["estimated_effort"] == effort


class TestMRAnalyzerAgent:
    """Tests for MRAnalyzerAgent class."""

    def test_agent_creation(self, mock_env_vars):
        """Test MRAnalyzerAgent creation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            assert agent is not None

    def test_agent_capabilities(self, mock_env_vars):
        """Test MRAnalyzerAgent capabilities."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.base import AgentCapability
            from src.agents.mr_analyzer import MRAnalyzerAgent

            assert AgentCapability.MR_ANALYSIS in MRAnalyzerAgent.CAPABILITIES
            assert AgentCapability.CODE_ANALYSIS in MRAnalyzerAgent.CAPABILITIES
            assert AgentCapability.TEST_GENERATION in MRAnalyzerAgent.CAPABILITIES

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            prompt = agent._get_system_prompt()

            assert "code reviewer" in prompt.lower() or "test engineer" in prompt.lower()
            assert "test" in prompt.lower()
            assert "coverage" in prompt.lower() or "risk" in prompt.lower()


class TestMRAnalyzerAgentAnalyze:
    """Tests for MRAnalyzerAgent.analyze() method."""

    @pytest.fixture
    def mock_ai_response(self):
        """Create a mock AI response."""
        return MagicMock(
            content="""
            {
                "file_analyses": [
                    {
                        "file_path": "src/auth.py",
                        "change_type": "feature",
                        "risk_level": "high",
                        "affected_areas": ["authentication"],
                        "semantic_summary": "Added OAuth support",
                        "functions_changed": ["authenticate", "get_token"],
                        "imports_changed": ["oauth2"],
                        "potential_impacts": ["Login flow affected"]
                    }
                ],
                "overall_change_type": "feature",
                "overall_risk_score": 0.7,
                "affected_components": ["auth", "api"],
                "affected_routes": ["/login", "/oauth/callback"],
                "coverage_gaps": ["No tests for OAuth flow"],
                "summary": "Added OAuth authentication support",
                "recommendations": ["Add OAuth integration tests"]
            }
            """
        )

    @pytest.mark.asyncio
    async def test_analyze_with_file_change_objects(self, mock_env_vars, mock_ai_response):
        """Test analyze with FileChange objects."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            agent._call_ai = AsyncMock(return_value=mock_ai_response)

            changes = [
                FileChange(
                    path="src/auth.py",
                    status="modified",
                    additions=100,
                    deletions=20,
                    patch="+def authenticate():\n+    pass",
                ),
            ]

            result = await agent.analyze(changes, "project-123")

            assert result.overall_change_type == ChangeType.FEATURE
            assert result.overall_risk_score == 0.7
            assert len(result.file_analyses) == 1
            assert result.file_analyses[0].file_path == "src/auth.py"

    @pytest.mark.asyncio
    async def test_analyze_with_dict_changes(self, mock_env_vars, mock_ai_response):
        """Test analyze with dictionary changes."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            agent._call_ai = AsyncMock(return_value=mock_ai_response)

            changes = [
                {
                    "path": "src/auth.py",
                    "status": "modified",
                    "additions": 100,
                    "deletions": 20,
                },
            ]

            result = await agent.analyze(changes, "project-123")

            assert result is not None
            assert result.summary == "Added OAuth authentication support"

    @pytest.mark.asyncio
    async def test_analyze_handles_invalid_json_gracefully(self, mock_env_vars):
        """Test analyze handles invalid JSON response gracefully."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            # Return invalid JSON - _parse_json_response returns {} default
            mock_response = MagicMock(content="not valid json")
            agent._call_ai = AsyncMock(return_value=mock_response)

            changes = [
                FileChange(
                    path="src/test_auth.py",
                    status="added",
                    additions=50,
                ),
            ]

            result = await agent.analyze(changes, "project-123")

            # Should get empty analysis from _build_analysis with empty data
            # (because _parse_json_response catches JSON error and returns {})
            assert result is not None
            # Empty file_analyses when parse returns {}
            assert len(result.file_analyses) == 0
            assert result.summary == ""

    @pytest.mark.asyncio
    async def test_analyze_builds_default_on_exception(self, mock_env_vars):
        """Test analyze builds default analysis on exception during parsing."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            # Mock _parse_json_response to raise an exception
            mock_response = MagicMock(content='{"valid": "json"}')
            agent._call_ai = AsyncMock(return_value=mock_response)
            agent._parse_json_response = MagicMock(side_effect=Exception("Parse error"))

            changes = [
                FileChange(
                    path="src/test_auth.py",
                    status="added",
                    additions=50,
                ),
            ]

            result = await agent.analyze(changes, "project-123")

            # Should get default analysis since exception was raised
            assert result is not None
            assert len(result.file_analyses) == 1
            # Test files should be identified as TEST type
            assert result.file_analyses[0].change_type == ChangeType.TEST

    @pytest.mark.asyncio
    async def test_analyze_multiple_files(self, mock_env_vars):
        """Test analyze with multiple files."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(
                content="""
                {
                    "file_analyses": [
                        {"file_path": "src/a.py", "change_type": "feature", "risk_level": "medium", "affected_areas": [], "semantic_summary": "File A"},
                        {"file_path": "src/b.py", "change_type": "bugfix", "risk_level": "low", "affected_areas": [], "semantic_summary": "File B"}
                    ],
                    "overall_change_type": "feature",
                    "overall_risk_score": 0.5,
                    "affected_components": [],
                    "affected_routes": [],
                    "coverage_gaps": [],
                    "summary": "Multiple changes",
                    "recommendations": []
                }
                """
            )
            agent._call_ai = AsyncMock(return_value=mock_response)

            changes = [
                FileChange(path="src/a.py", status="modified"),
                FileChange(path="src/b.py", status="modified"),
            ]

            result = await agent.analyze(changes, "project-123")

            assert len(result.file_analyses) == 2


class TestMRAnalyzerAgentSuggestTests:
    """Tests for MRAnalyzerAgent.suggest_tests() method."""

    @pytest.fixture
    def sample_analysis(self, mock_env_vars):
        """Create a sample MRAnalysis."""
        from src.agents.mr_analyzer import ChangeAnalysis, ChangeType, MRAnalysis

        return MRAnalysis(
            file_analyses=[
                ChangeAnalysis(
                    file_path="src/auth.py",
                    change_type=ChangeType.FEATURE,
                    risk_level="high",
                    affected_areas=["authentication"],
                    semantic_summary="OAuth support added",
                    functions_changed=["authenticate"],
                    potential_impacts=["Login flow affected"],
                ),
            ],
            overall_change_type=ChangeType.FEATURE,
            overall_risk_score=0.7,
            affected_components=["auth"],
            affected_routes=["/login"],
            coverage_gaps=["No OAuth tests"],
            summary="OAuth feature",
            recommendations=["Add tests"],
        )

    @pytest.mark.asyncio
    async def test_suggest_tests_returns_suggestions(self, mock_env_vars, sample_analysis):
        """Test suggest_tests returns test suggestions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent, TestPriority, TestType

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(
                content="""
                {
                    "suggestions": [
                        {
                            "name": "test_oauth_login",
                            "description": "Test OAuth login flow",
                            "test_type": "integration",
                            "priority": "critical",
                            "target_files": ["src/auth.py"],
                            "coverage_target": "OAuth authentication",
                            "estimated_effort": "medium",
                            "rationale": "Critical auth change",
                            "example_code": "def test_oauth():\\n    pass"
                        }
                    ]
                }
                """
            )
            agent._call_ai = AsyncMock(return_value=mock_response)

            suggestions = await agent.suggest_tests(sample_analysis)

            assert len(suggestions) == 1
            assert suggestions[0].name == "test_oauth_login"
            assert suggestions[0].test_type == TestType.INTEGRATION
            assert suggestions[0].priority == TestPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_suggest_tests_handles_invalid_test_type(self, mock_env_vars, sample_analysis):
        """Test suggest_tests handles invalid test type gracefully."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent, TestType

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(
                content="""
                {
                    "suggestions": [
                        {
                            "name": "test_invalid",
                            "description": "Test with invalid type",
                            "test_type": "invalid_type",
                            "priority": "medium",
                            "target_files": [],
                            "coverage_target": "",
                            "estimated_effort": "small"
                        }
                    ]
                }
                """
            )
            agent._call_ai = AsyncMock(return_value=mock_response)

            suggestions = await agent.suggest_tests(sample_analysis)

            assert len(suggestions) == 1
            assert suggestions[0].test_type == TestType.UNIT  # Default

    @pytest.mark.asyncio
    async def test_suggest_tests_handles_invalid_priority(self, mock_env_vars, sample_analysis):
        """Test suggest_tests handles invalid priority gracefully."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent, TestPriority

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(
                content="""
                {
                    "suggestions": [
                        {
                            "name": "test_invalid_priority",
                            "description": "Test",
                            "test_type": "unit",
                            "priority": "invalid_priority",
                            "target_files": [],
                            "coverage_target": "",
                            "estimated_effort": "small"
                        }
                    ]
                }
                """
            )
            agent._call_ai = AsyncMock(return_value=mock_response)

            suggestions = await agent.suggest_tests(sample_analysis)

            assert len(suggestions) == 1
            assert suggestions[0].priority == TestPriority.MEDIUM  # Default

    @pytest.mark.asyncio
    async def test_suggest_tests_handles_invalid_json_gracefully(
        self, mock_env_vars, sample_analysis
    ):
        """Test suggest_tests handles invalid JSON response gracefully."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(content="not valid json")
            agent._call_ai = AsyncMock(return_value=mock_response)

            suggestions = await agent.suggest_tests(sample_analysis)

            # When _parse_json_response returns {} (default), _build_suggestions
            # is called with empty list, returning empty suggestions
            assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_suggest_tests_builds_default_on_exception(
        self, mock_env_vars, sample_analysis
    ):
        """Test suggest_tests builds default suggestions on exception."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(content='{"suggestions": []}')
            agent._call_ai = AsyncMock(return_value=mock_response)
            # Force exception during JSON processing
            agent._parse_json_response = MagicMock(side_effect=Exception("Parse error"))

            suggestions = await agent.suggest_tests(sample_analysis)

            # Should get default suggestions based on high-risk files
            # sample_analysis has one file with risk_level "high"
            assert len(suggestions) >= 1

    @pytest.mark.asyncio
    async def test_suggest_tests_multiple_suggestions(self, mock_env_vars, sample_analysis):
        """Test suggest_tests returns multiple suggestions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            mock_response = MagicMock(
                content="""
                {
                    "suggestions": [
                        {"name": "test_1", "description": "Test 1", "test_type": "unit", "priority": "high", "target_files": [], "coverage_target": "", "estimated_effort": "small"},
                        {"name": "test_2", "description": "Test 2", "test_type": "integration", "priority": "medium", "target_files": [], "coverage_target": "", "estimated_effort": "medium"},
                        {"name": "test_3", "description": "Test 3", "test_type": "e2e", "priority": "low", "target_files": [], "coverage_target": "", "estimated_effort": "large"}
                    ]
                }
                """
            )
            agent._call_ai = AsyncMock(return_value=mock_response)

            suggestions = await agent.suggest_tests(sample_analysis)

            assert len(suggestions) == 3


class TestMRAnalyzerAgentGenerateComment:
    """Tests for MRAnalyzerAgent.generate_comment() method."""

    @pytest.fixture
    def sample_suggestions(self, mock_env_vars):
        """Create sample test suggestions."""
        from src.agents.mr_analyzer import TestPriority, TestSuggestion, TestType

        return [
            TestSuggestion(
                name="test_critical_auth",
                description="Critical auth test",
                test_type=TestType.INTEGRATION,
                priority=TestPriority.CRITICAL,
                target_files=["src/auth.py"],
                coverage_target="Auth",
                estimated_effort="medium",
                rationale="Critical security",
                example_code="def test_auth():\n    pass",
            ),
            TestSuggestion(
                name="test_high_priority",
                description="High priority test",
                test_type=TestType.UNIT,
                priority=TestPriority.HIGH,
                target_files=["src/api.py"],
                coverage_target="API",
                estimated_effort="small",
                rationale="Important API",
            ),
            TestSuggestion(
                name="test_medium_priority",
                description="Medium priority test",
                test_type=TestType.API,
                priority=TestPriority.MEDIUM,
                target_files=["src/service.py"],
                coverage_target="Service",
                estimated_effort="medium",
            ),
            TestSuggestion(
                name="test_low_priority",
                description="Low priority test",
                test_type=TestType.VISUAL,
                priority=TestPriority.LOW,
                target_files=["src/ui.py"],
                coverage_target="UI",
                estimated_effort="trivial",
            ),
        ]

    @pytest.mark.asyncio
    async def test_generate_comment_empty_suggestions(self, mock_env_vars):
        """Test generate_comment with empty suggestions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment([])

            assert comment == ""

    @pytest.mark.asyncio
    async def test_generate_comment_includes_header(self, mock_env_vars, sample_suggestions):
        """Test generate_comment includes proper header."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(sample_suggestions)

            assert "## Test Suggestions" in comment
            assert "Based on the changes" in comment

    @pytest.mark.asyncio
    async def test_generate_comment_sorts_by_priority(self, mock_env_vars, sample_suggestions):
        """Test generate_comment sorts suggestions by priority."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(sample_suggestions)

            # Critical should appear before High
            critical_pos = comment.find("### Critical Priority")
            high_pos = comment.find("### High Priority")

            assert critical_pos < high_pos

    @pytest.mark.asyncio
    async def test_generate_comment_includes_code_snippets(self, mock_env_vars, sample_suggestions):
        """Test generate_comment includes code snippets when requested."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(
                sample_suggestions, include_code_snippets=True
            )

            assert "Example code" in comment
            assert "def test_auth" in comment

    @pytest.mark.asyncio
    async def test_generate_comment_excludes_code_snippets(self, mock_env_vars, sample_suggestions):
        """Test generate_comment excludes code snippets when requested."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(
                sample_suggestions, include_code_snippets=False
            )

            assert "def test_auth" not in comment

    @pytest.mark.asyncio
    async def test_generate_comment_respects_max_suggestions(
        self, mock_env_vars, sample_suggestions
    ):
        """Test generate_comment respects max_suggestions limit."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(sample_suggestions, max_suggestions=2)

            # Should only include top 2 (critical and high)
            assert "test_critical_auth" in comment
            assert "test_high_priority" in comment
            # Medium and low should be excluded
            assert "test_medium_priority" not in comment
            assert "test_low_priority" not in comment

    @pytest.mark.asyncio
    async def test_generate_comment_low_priority_in_details(
        self, mock_env_vars, sample_suggestions
    ):
        """Test generate_comment puts low priority in collapsible details."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(sample_suggestions)

            # Low priority should be in details tag
            assert "<details>" in comment
            assert "Low Priority Suggestions" in comment

    @pytest.mark.asyncio
    async def test_generate_comment_includes_footer(self, mock_env_vars, sample_suggestions):
        """Test generate_comment includes Argus footer."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            comment = await agent.generate_comment(sample_suggestions)

            assert "Argus E2E Testing Agent" in comment


class TestMRAnalyzerAgentInferChangeType:
    """Tests for MRAnalyzerAgent._infer_change_type() static method."""

    @pytest.mark.parametrize(
        "file_path,expected_type",
        [
            # Test files
            ("tests/test_auth.py", "test"),
            ("test_login.py", "test"),
            ("spec/auth_spec.js", "test"),
            # Docs files (txt, md, rst)
            ("README.md", "docs"),
            ("CHANGELOG.rst", "docs"),
            ("notes.txt", "docs"),
            # Config files (json, yaml, yml, toml, env)
            ("config.json", "config"),
            ("settings.yaml", "config"),
            ("app.yml", "config"),
            # Note: pyproject.toml matches .toml -> CONFIG before DEPENDENCY check
            ("pyproject.toml", "config"),
            # Note: requirements.txt ends with .txt -> DOCS before DEPENDENCY check
            ("requirements.txt", "docs"),
            # Note: package.json matches .json -> CONFIG before DEPENDENCY check
            ("package.json", "config"),
            # Security files
            ("auth/security.py", "security"),
            ("password_handler.py", "security"),
            ("token_manager.py", "security"),
            # Unknown files
            ("src/app.py", "unknown"),
            ("utils/helpers.py", "unknown"),
        ],
    )
    def test_infer_change_type(self, mock_env_vars, file_path, expected_type):
        """Test _infer_change_type for various file paths."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, MRAnalyzerAgent

            result = MRAnalyzerAgent._infer_change_type(file_path)

            assert result == ChangeType(expected_type)

    @pytest.mark.parametrize(
        "file_path,expected_type",
        [
            # These files WOULD be dependency if the logic order was different
            # but they match config/docs first due to extension matching
            ("requirements/prod.txt", "docs"),
            ("pyproject.toml", "config"),
            ("package.json", "config"),
        ],
    )
    def test_infer_change_type_extension_priority(self, mock_env_vars, file_path, expected_type):
        """Test that extension matching takes priority over keyword matching."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, MRAnalyzerAgent

            result = MRAnalyzerAgent._infer_change_type(file_path)

            assert result == ChangeType(expected_type)

    def test_infer_change_type_case_insensitive(self, mock_env_vars):
        """Test _infer_change_type is case insensitive."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, MRAnalyzerAgent

            assert MRAnalyzerAgent._infer_change_type("TEST_auth.py") == ChangeType.TEST
            assert MRAnalyzerAgent._infer_change_type("SECURITY/auth.py") == ChangeType.SECURITY
            assert MRAnalyzerAgent._infer_change_type("README.MD") == ChangeType.DOCS


class TestMRAnalyzerAgentInferRiskLevel:
    """Tests for MRAnalyzerAgent._infer_risk_level() static method."""

    @pytest.mark.parametrize(
        "file_path,expected_risk",
        [
            ("src/auth/login.py", "high"),
            ("payment_handler.py", "high"),
            ("billing/invoice.py", "high"),
            ("security/validator.py", "high"),
            ("password_reset.py", "high"),
            ("token_manager.py", "high"),
            ("secret_key.py", "high"),
            ("api/endpoints.py", "medium"),
            ("database/models.py", "medium"),
            ("migrations/001.sql", "medium"),
            ("config/settings.py", "medium"),
            ("user/profile.py", "medium"),
            ("account/settings.py", "medium"),
            ("utils/helpers.py", "low"),
            ("tests/test_utils.py", "low"),
            ("docs/readme.md", "low"),
        ],
    )
    def test_infer_risk_level(self, mock_env_vars, file_path, expected_risk):
        """Test _infer_risk_level for various file paths."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            result = MRAnalyzerAgent._infer_risk_level(file_path)

            assert result == expected_risk

    def test_infer_risk_level_case_insensitive(self, mock_env_vars):
        """Test _infer_risk_level is case insensitive."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            assert MRAnalyzerAgent._infer_risk_level("AUTH/login.py") == "high"
            assert MRAnalyzerAgent._infer_risk_level("PAYMENT.py") == "high"
            assert MRAnalyzerAgent._infer_risk_level("API/endpoints.py") == "medium"


class TestMRAnalyzerAgentExecute:
    """Tests for MRAnalyzerAgent.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_no_changes(self, mock_env_vars):
        """Test execute with no changes returns error."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            result = await agent.execute(changes=[], project_id="proj-123")

            assert result.success is False
            assert "No changes provided" in result.error

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars):
        """Test execute with valid changes."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            # Mock analyze and suggest_tests
            mock_analysis = MagicMock()
            mock_analysis.to_dict.return_value = {"summary": "Test analysis"}

            mock_suggestion = MagicMock()
            mock_suggestion.to_dict.return_value = {"name": "test_suggestion"}

            agent.analyze = AsyncMock(return_value=mock_analysis)
            agent.suggest_tests = AsyncMock(return_value=[mock_suggestion])

            changes = [{"path": "src/auth.py", "status": "modified"}]
            result = await agent.execute(changes=changes, project_id="proj-123")

            assert result.success is True
            assert "analysis" in result.data
            assert "suggestions" in result.data
            assert len(result.data["suggestions"]) == 1

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self, mock_env_vars):
        """Test execute handles exceptions gracefully."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            agent.log = MagicMock()
            agent.analyze = AsyncMock(side_effect=Exception("Analysis failed"))

            changes = [{"path": "src/auth.py", "status": "modified"}]
            result = await agent.execute(changes=changes, project_id="proj-123")

            assert result.success is False
            assert "Analysis failed" in result.error


class TestMRAnalyzerAgentHelperMethods:
    """Tests for MRAnalyzerAgent helper methods."""

    def test_prepare_changes_summary(self, mock_env_vars):
        """Test _prepare_changes_summary method."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            changes = [
                FileChange(
                    path="src/auth.py",
                    status="modified",
                    additions=10,
                    deletions=5,
                    patch="+def login():\n+    pass",
                ),
            ]

            summary = agent._prepare_changes_summary(changes)

            assert "### File: src/auth.py" in summary
            assert "Status: modified" in summary
            assert "Additions: 10" in summary
            assert "Deletions: 5" in summary
            assert "```diff" in summary
            assert "+def login()" in summary

    def test_prepare_changes_summary_truncates_long_patch(self, mock_env_vars):
        """Test _prepare_changes_summary truncates long patches."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            long_patch = "+" + "x" * 4000  # Longer than 3000 char limit

            changes = [
                FileChange(
                    path="src/big_file.py",
                    status="modified",
                    patch=long_patch,
                ),
            ]

            summary = agent._prepare_changes_summary(changes)

            assert "... (truncated)" in summary

    def test_prepare_changes_summary_with_renamed(self, mock_env_vars):
        """Test _prepare_changes_summary with renamed file."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            changes = [
                FileChange(
                    path="src/new_name.py",
                    status="renamed",
                    previous_path="src/old_name.py",
                ),
            ]

            summary = agent._prepare_changes_summary(changes)

            assert "Renamed from: src/old_name.py" in summary

    def test_build_analysis_with_invalid_change_type(self, mock_env_vars):
        """Test _build_analysis handles invalid change type."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            data = {
                "file_analyses": [
                    {
                        "file_path": "test.py",
                        "change_type": "invalid_type",
                        "risk_level": "low",
                        "affected_areas": [],
                        "semantic_summary": "Test",
                    }
                ],
                "overall_change_type": "also_invalid",
                "overall_risk_score": 0.5,
                "affected_components": [],
                "affected_routes": [],
                "coverage_gaps": [],
                "summary": "Test",
                "recommendations": [],
            }

            changes = [FileChange(path="test.py", status="modified")]

            result = agent._build_analysis(data, changes)

            assert result.file_analyses[0].change_type == ChangeType.UNKNOWN
            assert result.overall_change_type == ChangeType.UNKNOWN

    def test_build_default_analysis(self, mock_env_vars):
        """Test _build_default_analysis method."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import ChangeType, FileChange, MRAnalyzerAgent

            agent = MRAnalyzerAgent()

            changes = [
                FileChange(path="src/auth/login.py", status="modified"),
                FileChange(path="tests/test_auth.py", status="added"),
            ]

            result = agent._build_default_analysis(changes)

            assert len(result.file_analyses) == 2
            # Auth file should be high risk
            assert result.file_analyses[0].risk_level == "high"
            # Test file should be identified as TEST type
            assert result.file_analyses[1].change_type == ChangeType.TEST
            assert "manual review recommended" in result.coverage_gaps[0].lower()

    def test_build_default_suggestions(self, mock_env_vars):
        """Test _build_default_suggestions method."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import (
                ChangeAnalysis,
                ChangeType,
                MRAnalysis,
                MRAnalyzerAgent,
                TestPriority,
            )

            agent = MRAnalyzerAgent()

            analysis = MRAnalysis(
                file_analyses=[
                    ChangeAnalysis(
                        file_path="src/auth.py",
                        change_type=ChangeType.FEATURE,
                        risk_level="critical",
                        affected_areas=[],
                        semantic_summary="Auth change",
                    ),
                    ChangeAnalysis(
                        file_path="src/utils.py",
                        change_type=ChangeType.REFACTOR,
                        risk_level="low",
                        affected_areas=[],
                        semantic_summary="Utils change",
                    ),
                ],
                overall_change_type=ChangeType.FEATURE,
                overall_risk_score=0.7,
                affected_components=[],
                affected_routes=[],
                coverage_gaps=[],
                summary="Test",
                recommendations=[],
            )

            suggestions = agent._build_default_suggestions(analysis)

            # Should only create suggestions for high-risk files (critical)
            assert len(suggestions) == 1
            assert suggestions[0].priority == TestPriority.HIGH

    def test_format_suggestion_with_code(self, mock_env_vars):
        """Test _format_suggestion with example code."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import (
                MRAnalyzerAgent,
                TestPriority,
                TestSuggestion,
                TestType,
            )

            agent = MRAnalyzerAgent()

            suggestion = TestSuggestion(
                name="test_example",
                description="Example test",
                test_type=TestType.UNIT,
                priority=TestPriority.HIGH,
                target_files=["src/auth.py"],
                coverage_target="Auth",
                estimated_effort="small",
                rationale="Important test",
                example_code="def test_example():\n    assert True",
            )

            lines = agent._format_suggestion(suggestion, include_code=True)

            output = "\n".join(lines)
            assert "**test_example**" in output
            assert "unit" in output
            assert "Example test" in output
            assert "Important test" in output
            assert "`src/auth.py`" in output
            assert "Effort:* small" in output
            assert "<details>" in output
            assert "def test_example" in output

    def test_format_suggestion_without_code(self, mock_env_vars):
        """Test _format_suggestion without example code."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import (
                MRAnalyzerAgent,
                TestPriority,
                TestSuggestion,
                TestType,
            )

            agent = MRAnalyzerAgent()

            suggestion = TestSuggestion(
                name="test_no_code",
                description="No code test",
                test_type=TestType.INTEGRATION,
                priority=TestPriority.MEDIUM,
                target_files=[],
                coverage_target="Coverage",
                estimated_effort="medium",
            )

            lines = agent._format_suggestion(suggestion, include_code=False)

            output = "\n".join(lines)
            assert "<details>" not in output


class TestCreateMRAnalyzer:
    """Tests for create_mr_analyzer factory function."""

    def test_create_mr_analyzer(self, mock_env_vars):
        """Test create_mr_analyzer factory function."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.mr_analyzer import MRAnalyzerAgent, create_mr_analyzer

            agent = create_mr_analyzer()

            assert isinstance(agent, MRAnalyzerAgent)
