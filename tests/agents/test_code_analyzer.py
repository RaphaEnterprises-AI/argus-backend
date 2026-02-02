"""Tests for the code analyzer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Patch target for Anthropic client in base agent
ANTHROPIC_PATCH = 'anthropic.Anthropic'


class TestTestableSurface:
    """Tests for TestableSurface dataclass."""

    @pytest.mark.parametrize(
        "surface_type,name,path,priority,description,scenarios,expected_scenario_count",
        [
            ("ui", "Login Page", "/login", "critical", "User login page", ["Valid login", "Invalid password"], 2),
            ("api", "User API", "/api/users", "high", "User management API", ["Create user", "Get user", "Delete user"], 3),
            ("db", "Users Table", "users", "medium", "Users table", ["Check constraints"], 1),
            ("ui", "Dashboard", "/dashboard", "low", "Main dashboard", [], 0),
        ],
        ids=["ui-critical", "api-high", "db-medium", "ui-low-empty-scenarios"],
    )
    def test_surface_creation(
        self,
        mock_env_vars,
        surface_type,
        name,
        path,
        priority,
        description,
        scenarios,
        expected_scenario_count,
    ):
        """Test TestableSurface creation with various configurations."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type=surface_type,
            name=name,
            path=path,
            priority=priority,
            description=description,
            test_scenarios=scenarios,
        )

        assert surface.type == surface_type
        assert surface.name == name
        assert surface.path == path
        assert surface.priority == priority
        assert surface.description == description
        assert len(surface.test_scenarios) == expected_scenario_count

    @pytest.mark.parametrize(
        "metadata,expected_keys",
        [
            ({"method": "POST", "auth": True}, ["method", "auth"]),
            ({"method": "GET"}, ["method"]),
            ({"auth": False, "rate_limit": 100, "version": "v1"}, ["auth", "rate_limit", "version"]),
        ],
        ids=["post-with-auth", "get-only", "multiple-fields"],
    )
    def test_surface_with_metadata(self, mock_env_vars, metadata, expected_keys):
        """Test TestableSurface with various metadata configurations."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type="api",
            name="User API",
            path="/api/users",
            priority="high",
            description="User management API",
            test_scenarios=["Create user"],
            metadata=metadata,
        )

        for key in expected_keys:
            assert key in surface.metadata
            assert surface.metadata[key] == metadata[key]

    @pytest.mark.parametrize(
        "surface_type,name,path,priority,expected_metadata",
        [
            ("ui", "Dashboard", "/dashboard", "high", {}),
            ("api", "Health", "/health", "low", {}),
            ("db", "Sessions", "sessions", "medium", {}),
        ],
        ids=["ui-dashboard", "api-health", "db-sessions"],
    )
    def test_surface_to_dict(self, mock_env_vars, surface_type, name, path, priority, expected_metadata):
        """Test TestableSurface to_dict method with various configurations."""
        from src.agents.code_analyzer import TestableSurface

        surface = TestableSurface(
            type=surface_type,
            name=name,
            path=path,
            priority=priority,
            description="Test description",
            test_scenarios=["Test scenario"],
        )

        result = surface.to_dict()

        assert result["type"] == surface_type
        assert result["name"] == name
        assert result["path"] == path
        assert result["priority"] == priority
        assert result["metadata"] == expected_metadata

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

    @pytest.mark.parametrize(
        "summary,framework,language,recommendations,surface_count",
        [
            ("Test application", "React", "TypeScript", ["Add more tests"], 1),
            ("Next.js app", "Next.js", "TypeScript", ["Test SSR", "Test API routes"], 2),
            ("Django backend", "Django", "Python", None, 3),
            ("Simple app", None, None, None, 0),
        ],
        ids=["react-ts", "nextjs-ts", "django-py", "minimal"],
    )
    def test_result_creation(
        self,
        mock_env_vars,
        summary,
        framework,
        language,
        recommendations,
        surface_count,
    ):
        """Test CodeAnalysisResult creation with various configurations."""
        from src.agents.code_analyzer import CodeAnalysisResult, TestableSurface

        surfaces = [
            TestableSurface(
                type="ui",
                name=f"Test {i}",
                path=f"/test{i}",
                priority="low",
                description="Test",
                test_scenarios=[],
            )
            for i in range(surface_count)
        ]

        result = CodeAnalysisResult(
            summary=summary,
            testable_surfaces=surfaces,
            framework_detected=framework,
            language=language,
            recommendations=recommendations,
        )

        assert result.summary == summary
        assert len(result.testable_surfaces) == surface_count
        assert result.framework_detected == framework
        assert result.language == language
        assert result.recommendations == recommendations

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

    @pytest.mark.parametrize(
        "codebase_path,app_url,expected_in_prompt",
        [
            ("/path/to/app", "http://localhost:3000", ["/path/to/app", "http://localhost:3000"]),
            ("/home/user/project", "http://127.0.0.1:8080", ["/home/user/project", "http://127.0.0.1:8080"]),
            ("./relative/path", "https://staging.example.com", ["./relative/path", "https://staging.example.com"]),
        ],
        ids=["standard-localhost", "ip-based-url", "https-staging"],
    )
    def test_build_analysis_prompt_basic(self, mock_env_vars, codebase_path, app_url, expected_in_prompt):
        """Test analysis prompt building with various path and URL combinations."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._build_analysis_prompt(
                codebase_path=codebase_path,
                app_url=app_url,
                changed_files=None,
                file_contents=None,
            )

            for expected in expected_in_prompt:
                assert expected in prompt

    @pytest.mark.parametrize(
        "changed_files",
        [
            ["src/login.py", "src/signup.py"],
            ["src/auth/handler.ts", "src/auth/middleware.ts", "tests/auth.test.ts"],
            ["single_file.py"],
            ["a.py", "b.py", "c.py", "d.py", "e.py"],
        ],
        ids=["two-files", "three-nested-files", "single-file", "five-files"],
    )
    def test_build_analysis_prompt_with_changed_files(self, mock_env_vars, changed_files):
        """Test analysis prompt includes all changed files."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._build_analysis_prompt(
                codebase_path="/app",
                app_url="http://localhost",
                changed_files=changed_files,
                file_contents=None,
            )

            assert "CHANGED FILES" in prompt
            for file in changed_files:
                assert file in prompt

    @pytest.mark.parametrize(
        "file_contents,expected_snippets",
        [
            (
                {"src/app.py": "def hello(): pass"},
                ["FILE CONTENTS", "src/app.py", "def hello"],
            ),
            (
                {"src/app.py": "def hello(): pass", "src/routes.py": "routes = []"},
                ["FILE CONTENTS", "src/app.py", "def hello", "src/routes.py", "routes = []"],
            ),
            (
                {"config.json": '{"key": "value"}'},
                ["FILE CONTENTS", "config.json", '"key"'],
            ),
        ],
        ids=["single-py-file", "multiple-py-files", "json-config"],
    )
    def test_build_analysis_prompt_with_file_contents(self, mock_env_vars, file_contents, expected_snippets):
        """Test analysis prompt includes file contents."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            prompt = agent._build_analysis_prompt(
                codebase_path="/app",
                app_url="http://localhost",
                changed_files=None,
                file_contents=file_contents,
            )

            for snippet in expected_snippets:
                assert snippet in prompt

    @pytest.mark.parametrize(
        "content_size",
        [3000, 5000, 10000],
        ids=["3k-chars", "5k-chars", "10k-chars"],
    )
    def test_build_analysis_prompt_truncates_large_files(self, mock_env_vars, content_size):
        """Test that large file contents are truncated."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            large_content = "x" * content_size

            prompt = agent._build_analysis_prompt(
                codebase_path="/app",
                app_url="http://localhost",
                changed_files=None,
                file_contents={"large.py": large_content},
            )

            assert "..." in prompt  # Truncation marker

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_json,expected_summary,expected_framework,expected_surface_count",
        [
            (
                '''
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
                ''',
                "A web application",
                "Next.js",
                1,
            ),
            (
                '''
                {
                    "summary": "Django REST API",
                    "framework": "Django",
                    "language": "Python",
                    "testable_surfaces": [
                        {
                            "type": "api",
                            "name": "User API",
                            "path": "/api/users",
                            "priority": "high",
                            "description": "User CRUD",
                            "test_scenarios": ["Create", "Read", "Update", "Delete"]
                        },
                        {
                            "type": "api",
                            "name": "Auth API",
                            "path": "/api/auth",
                            "priority": "critical",
                            "description": "Authentication",
                            "test_scenarios": ["Login", "Logout"]
                        }
                    ],
                    "recommendations": []
                }
                ''',
                "Django REST API",
                "Django",
                2,
            ),
            (
                '''
                {
                    "summary": "Simple static site",
                    "testable_surfaces": []
                }
                ''',
                "Simple static site",
                None,
                0,
            ),
        ],
        ids=["nextjs-single-surface", "django-multiple-surfaces", "minimal-no-surfaces"],
    )
    async def test_execute_success(
        self,
        mock_env_vars,
        response_json,
        expected_summary,
        expected_framework,
        expected_surface_count,
    ):
        """Test successful code analysis with various response formats."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 200
            mock_response.content = [MagicMock(text=response_json)]
            mock_anthropic.return_value.messages.create.return_value = mock_response

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            result = await agent.execute(
                codebase_path="/app",
                app_url="http://localhost:3000",
            )

            assert result.success is True
            assert result.data.summary == expected_summary
            assert len(result.data.testable_surfaces) == expected_surface_count
            assert result.data.framework_detected == expected_framework

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
    @pytest.mark.parametrize(
        "invalid_response",
        [
            "Not valid JSON",
            "{ invalid json }",
            "```json\n{}\n```",  # Markdown code block without proper content
            "",
            "null",
            "[]",
        ],
        ids=["plain-text", "malformed-json", "markdown-block", "empty-string", "null-literal", "empty-array"],
    )
    async def test_execute_parse_failure(self, mock_env_vars, invalid_response):
        """Test execution with various parse failures."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_response = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.content = [MagicMock(text=invalid_response)]
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
    @pytest.mark.parametrize(
        "exception_type,exception_message",
        [
            (Exception, "API Error"),
            (ConnectionError, "Connection refused"),
            (TimeoutError, "Request timed out"),
            (ValueError, "Invalid parameter"),
        ],
        ids=["generic-exception", "connection-error", "timeout-error", "value-error"],
    )
    async def test_execute_exception(self, mock_env_vars, exception_type, exception_message):
        """Test execution with various exception types."""
        with patch(ANTHROPIC_PATCH) as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = exception_type(exception_message)

            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            result = await agent.execute(
                codebase_path="/app",
                app_url="http://localhost",
            )

            assert result.success is False
            assert "Analysis failed" in result.error

    @pytest.mark.parametrize(
        "file_path",
        [
            # Directory-based skips (skip_dirs: node_modules, .git, __pycache__, .venv, venv, dist, build, .next)
            "app/node_modules/pkg/index.js",
            "app/node_modules/react/lib/react.js",
            "app/.git/config",
            "app/.git/objects/pack/pack-123.idx",
            "app/__pycache__/mod.pyc",
            "app/__pycache__/module.cpython-39.pyc",
            "app/.venv/lib/pkg.py",
            "app/.venv/bin/python",
            "project/venv/bin/activate",
            "project/venv/lib/python3.9/site-packages/pip.py",
            "app/dist/bundle.js",
            "app/dist/main.chunk.js",
            "app/build/output.js",
            "app/build/static/css/main.css",
            "app/.next/server/pages/index.js",
            "app/.next/cache/webpack/client-development.pack",
            # Extension-based skips (.min.js, .map, .lock, .svg, .png, .jpg)
            "app/bundle.min.js",
            "app/vendor.min.js",
            "app/file.map",
            "app/bundle.js.map",
            "app/logo.png",
            "app/icon.png",
            "app/image.jpg",
            "app/photo.jpg",
            "app/icon.svg",
            "app/logo.svg",
            "app/package-lock.json.lock",
            "app/yarn.lock",
        ],
        ids=[
            "node_modules-pkg",
            "node_modules-nested",
            "git-config",
            "git-objects",
            "pycache-mod",
            "pycache-cpython",
            "venv-dot-lib",
            "venv-dot-bin",
            "venv-bin",
            "venv-site-packages",
            "dist-bundle",
            "dist-chunk",
            "build-output",
            "build-static",
            "next-server",
            "next-cache",
            "minified-js-bundle",
            "minified-js-vendor",
            "map-file",
            "map-js",
            "png-logo",
            "png-icon",
            "jpg-image",
            "jpg-photo",
            "svg-icon",
            "svg-logo",
            "lock-package",
            "lock-yarn",
        ],
    )
    def test_should_skip_files(self, mock_env_vars, file_path):
        """Test file skip logic for files that should be skipped."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            assert agent._should_skip(Path(file_path)) is True

    @pytest.mark.parametrize(
        "file_path",
        [
            "src/app.py",
            "src/components/Login.tsx",
            "routes/api.js",
            "lib/utils.ts",
            "app/models.py",
            "tests/test_main.py",
            "src/index.html",
            "config/settings.json",
            "src/styles.css",
            "README.md",
            "Makefile",
            "Dockerfile",
            "docker-compose.yml",
            "src/main.go",
            "src/lib.rs",
            "src/Main.java",
        ],
        ids=[
            "python-file",
            "tsx-component",
            "js-routes",
            "ts-utils",
            "python-models",
            "python-tests",
            "html-index",
            "json-config",
            "css-styles",
            "markdown-readme",
            "makefile",
            "dockerfile",
            "docker-compose",
            "go-file",
            "rust-file",
            "java-file",
        ],
    )
    def test_should_not_skip_valid_files(self, mock_env_vars, file_path):
        """Test file skip logic for valid files that should not be skipped."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.code_analyzer import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            assert agent._should_skip(Path(file_path)) is False

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
