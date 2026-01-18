"""Tests for the code analyzer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Patch target for Anthropic client in base agent
ANTHROPIC_PATCH = 'anthropic.Anthropic'


class TestTestableSurface:
    """Tests for TestableSurface dataclass."""

    def test_surface_creation(self, mock_env_vars):
        """Test TestableSurface creation."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type="ui",
            name="Login Page",
            path="/login",
            priority="critical",
            description="User login page",
            test_scenarios=["Valid login", "Invalid password"],
        )

        assert surface.type == "ui"
        assert surface.name == "Login Page"
        assert surface.priority == "critical"
        assert len(surface.test_scenarios) == 2

    def test_surface_with_metadata(self, mock_env_vars):
        """Test TestableSurface with metadata."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type="api",
            name="User API",
            path="/api/users",
            priority="high",
            description="User management API",
            test_scenarios=["Create user"],
            metadata={"method": "POST", "auth": True},
        )

        assert surface.metadata["method"] == "POST"
        assert surface.metadata["auth"] is True

    def test_surface_to_dict(self, mock_env_vars):
        """Test TestableSurface to_dict method."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type="ui",
            name="Dashboard",
            path="/dashboard",
            priority="high",
            description="Main dashboard",
            test_scenarios=["Load data"],
        )

        result = surface.to_dict()

        assert result["type"] == "ui"
        assert result["name"] == "Dashboard"
        assert result["metadata"] == {}

    def test_surface_default_metadata(self, mock_env_vars):
        """Test TestableSurface default metadata is None."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type="db",
            name="Users Table",
            path="users",
            priority="medium",
            description="Users table",
            test_scenarios=["Check constraints"],
        )

        assert surface.metadata is None
        assert surface.to_dict()["metadata"] == {}


class TestCodeAnalysisResult:
    """Tests for CodeAnalysisResult dataclass."""

    def test_result_creation(self, mock_env_vars):
        """Test CodeAnalysisResult creation."""
        from src.agents.code_analyzer import CodeAnalysisResult, TestableSurface

        surface = TestableSurface(
            type="ui",
            name="Test",
            path="/test",
            priority="low",
            description="Test",
            test_scenarios=[],
        )

        result = CodeAnalysisResult(
            summary="Test application",
            testable_surfaces=[surface],
            framework_detected="React",
            language="TypeScript",
            recommendations=["Add more tests"],
        )

        assert result.summary == "Test application"
        assert len(result.testable_surfaces) == 1
        assert result.framework_detected == "React"

    def test_result_defaults(self, mock_env_vars):
        """Test CodeAnalysisResult default values."""
        from src.agents.code_analyzer import CodeAnalysisResult

        result = CodeAnalysisResult(
            summary="Test",
            testable_surfaces=[],
        )

        assert result.framework_detected is None
        assert result.language is None
        assert result.recommendations is None


class TestCodeAnalyzerAgent:
    """Tests for CodeAnalyzerAgent class."""

    def test_agent_creation(self, mock_env_vars):
        """Test CodeAnalyzerAgent creation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()

            assert agent is not None

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._get_system_prompt()

            # Enhanced prompt uses different terminology
            assert "testing architect" in prompt.lower() or "code analyzer" in prompt.lower()
            assert "test surface" in prompt.lower() or "testable surfaces" in prompt.lower()
            assert "JSON" in prompt or "json" in prompt

    def test_build_analysis_prompt_basic(self, mock_env_vars):
        """Test analysis prompt building."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._build_analysis_prompt(
                codebase_path="/path/to/app",
                app_url="http://localhost:3000",
                changed_files=None,
                file_contents=None,
            )

            assert "/path/to/app" in prompt
            assert "http://localhost:3000" in prompt

    def test_build_analysis_prompt_with_changed_files(self, mock_env_vars):
        """Test analysis prompt with changed files."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._build_analysis_prompt(
                codebase_path="/app",
                app_url="http://localhost",
                changed_files=["src/login.py", "src/signup.py"],
                file_contents=None,
            )

            assert "CHANGED FILES" in prompt
            assert "src/login.py" in prompt
            assert "src/signup.py" in prompt

    def test_build_analysis_prompt_with_file_contents(self, mock_env_vars):
        """Test analysis prompt with file contents."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._build_analysis_prompt(
                codebase_path="/app",
                app_url="http://localhost",
                changed_files=None,
                file_contents={
                    "src/app.py": "def hello(): pass",
                    "src/routes.py": "routes = []",
                },
            )

            assert "FILE CONTENTS" in prompt
            assert "src/app.py" in prompt
            assert "def hello" in prompt

    def test_build_analysis_prompt_truncates_large_files(self, mock_env_vars):
        """Test that large file contents are truncated."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            large_content = "x" * 5000  # Larger than 2000 char limit

            prompt = agent._build_analysis_prompt(
                codebase_path="/app",
                app_url="http://localhost",
                changed_files=None,
                file_contents={"large.py": large_content},
            )

            assert "..." in prompt  # Truncation marker

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars):
        """Test successful code analysis."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [MagicMock(text='''
            {
                "summary": "A web application",
                "framework": "Next.js",
                "language": "TypeScript",
                "testable_surfaces": [
                    {
                        "type": "ui",
                        "name": "Login Page",
                        "path": "/login",
                        "priority": "critical",
                        "description": "User login",
                        "test_scenarios": ["Valid login"]
                    }
                ],
                "recommendations": ["Test auth flows"]
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            result = await agent.execute(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            assert result.success is True
            assert result.data.summary == "A web application"
            assert len(result.data.testable_surfaces) == 1
            assert result.data.framework_detected == "Next.js"

    @pytest.mark.asyncio
    async def test_execute_cost_limit_exceeded(self, mock_env_vars):
        """Test execution when cost limit exceeded."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            # Set a very high cost to exceed the limit (default is 10.0)
            agent._usage.total_cost = 100.0  # Exceed default limit

            result = await agent.execute(
                codebase_path="/app",
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Cost limit" in result.error

    @pytest.mark.asyncio
    async def test_execute_parse_failure(self, mock_env_vars):
        """Test execution with parse failure."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.content = [MagicMock(text="Not valid JSON")]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            result = await agent.execute(
                codebase_path="/app",
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Failed to parse" in result.error

    @pytest.mark.asyncio
    async def test_execute_exception(self, mock_env_vars):
        """Test execution with exception."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception("API Error")

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            result = await agent.execute(
                codebase_path="/app",
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Analysis failed" in result.error

    def test_should_skip_directories(self, mock_env_vars):
        """Test file skip logic for directories."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()

            # Should skip
            assert agent._should_skip(Path("app/node_modules/pkg/index.js")) is True
            assert agent._should_skip(Path("app/.git/config")) is True
            assert agent._should_skip(Path("app/__pycache__/mod.pyc")) is True
            assert agent._should_skip(Path("app/.venv/lib/pkg.py")) is True

    def test_should_skip_extensions(self, mock_env_vars):
        """Test file skip logic for extensions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()

            # Should skip
            assert agent._should_skip(Path("app/bundle.min.js")) is True
            assert agent._should_skip(Path("app/file.map")) is True
            assert agent._should_skip(Path("app/logo.png")) is True
            assert agent._should_skip(Path("app/package-lock.json.lock")) is True

    def test_should_not_skip_valid_files(self, mock_env_vars):
        """Test file skip logic for valid files."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()

            # Should not skip
            assert agent._should_skip(Path("src/app.py")) is False
            assert agent._should_skip(Path("src/components/Login.tsx")) is False
            assert agent._should_skip(Path("routes/api.js")) is False

    @pytest.mark.asyncio
    async def test_analyze_with_file_access_no_path(self, mock_env_vars):
        """Test analyze_with_file_access with non-existent path."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            result = await agent.analyze_with_file_access(
                codebase_path="/nonexistent/path",
                app_url="http://localhost",
            )

            assert result.success is False
            assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_analyze_with_file_access_real_path(self, mock_env_vars):
        """Test analyze_with_file_access with real path."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [MagicMock(text='''
            {
                "summary": "Test app",
                "testable_surfaces": []
            }
            ''')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()

            # Create a temp directory with a test file
            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "app.py"
                test_file.write_text("def main(): pass")

                result = await agent.analyze_with_file_access(
                    codebase_path=tmpdir,
                    app_url="http://localhost",
                    patterns=["**/*.py"],
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_analyze_with_file_access_custom_patterns(self, mock_env_vars):
        """Test analyze_with_file_access with custom patterns."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [MagicMock(text='{"summary": "Test", "testable_surfaces": []}')]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create test files
                (Path(tmpdir) / "test.ts").write_text("export default {}")
                (Path(tmpdir) / "test.py").write_text("def test(): pass")

                result = await agent.analyze_with_file_access(
                    codebase_path=tmpdir,
                    app_url="http://localhost",
                    patterns=["**/*.ts"],  # Only TypeScript
                )

                assert result.success is True
