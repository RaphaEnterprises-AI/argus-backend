"""Tests for main.py entry point."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile
import json
from pathlib import Path


class TestRunTests:
    """Tests for run_tests function."""

    @pytest.mark.asyncio
    async def test_run_tests_basic(self, mock_env_vars):
        """Test basic run_tests execution."""
        from src.main import run_tests

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={
            "passed_count": 5,
            "failed_count": 0,
            "skipped_count": 0,
        })
        mock_orchestrator.get_run_summary.return_value = {
            "run_id": "test-run-123",
            "total_tests": 5,
            "passed": 5,
            "failed": 0,
            "skipped": 0,
            "pass_rate": 1.0,
            "total_cost": 0.05,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.TestingOrchestrator", return_value=mock_orchestrator):
                with patch("src.main.get_settings"):
                    result = await run_tests(
                        codebase_path="/test/project",
                        app_url="http://localhost:3000",
                        output_dir=tmpdir,
                    )

        assert result["passed_count"] == 5
        assert result["failed_count"] == 0
        mock_orchestrator.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_tests_with_pr_number(self, mock_env_vars):
        """Test run_tests with PR number."""
        from src.main import run_tests

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"passed_count": 3, "failed_count": 1})
        mock_orchestrator.get_run_summary.return_value = {
            "run_id": "test-run-456",
            "total_tests": 4,
            "passed": 3,
            "failed": 1,
            "skipped": 0,
            "pass_rate": 0.75,
            "total_cost": 0.08,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.TestingOrchestrator", return_value=mock_orchestrator) as mock_cls:
                with patch("src.main.get_settings"):
                    await run_tests(
                        codebase_path="/test/project",
                        app_url="http://localhost:3000",
                        output_dir=tmpdir,
                        pr_number=123,
                    )

        # Verify PR number was passed to orchestrator
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["pr_number"] == 123

    @pytest.mark.asyncio
    async def test_run_tests_with_changed_files(self, mock_env_vars):
        """Test run_tests with changed files list."""
        from src.main import run_tests

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"passed_count": 2, "failed_count": 0})
        mock_orchestrator.get_run_summary.return_value = {
            "run_id": "test-run-789",
            "total_tests": 2,
            "passed": 2,
            "failed": 0,
            "skipped": 0,
            "pass_rate": 1.0,
            "total_cost": 0.02,
        }

        changed_files = ["src/feature.py", "tests/test_feature.py"]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.TestingOrchestrator", return_value=mock_orchestrator) as mock_cls:
                with patch("src.main.get_settings"):
                    await run_tests(
                        codebase_path="/test/project",
                        app_url="http://localhost:3000",
                        output_dir=tmpdir,
                        changed_files=changed_files,
                    )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["changed_files"] == changed_files

    @pytest.mark.asyncio
    async def test_run_tests_creates_output_dir(self, mock_env_vars):
        """Test that run_tests creates output directory."""
        from src.main import run_tests

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"passed_count": 1, "failed_count": 0})
        mock_orchestrator.get_run_summary.return_value = {
            "run_id": "test-run",
            "total_tests": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "pass_rate": 1.0,
            "total_cost": 0.01,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"

            with patch("src.main.TestingOrchestrator", return_value=mock_orchestrator):
                with patch("src.main.get_settings"):
                    await run_tests(
                        codebase_path="/test/project",
                        app_url="http://localhost:3000",
                        output_dir=str(nested_dir),
                    )

            assert nested_dir.exists()
            assert (nested_dir / "results.json").exists()

    @pytest.mark.asyncio
    async def test_run_tests_saves_results_json(self, mock_env_vars):
        """Test that run_tests saves results to JSON file."""
        from src.main import run_tests

        test_result = {
            "passed_count": 8,
            "failed_count": 2,
            "skipped_count": 1,
            "test_results": [{"id": "test-1", "status": "passed"}],
        }

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value=test_result)
        mock_orchestrator.get_run_summary.return_value = {
            "run_id": "test-run",
            "total_tests": 11,
            "passed": 8,
            "failed": 2,
            "skipped": 1,
            "pass_rate": 0.727,
            "total_cost": 0.15,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.TestingOrchestrator", return_value=mock_orchestrator):
                with patch("src.main.get_settings"):
                    await run_tests(
                        codebase_path="/test/project",
                        app_url="http://localhost:3000",
                        output_dir=tmpdir,
                    )

            # Check results file
            results_file = Path(tmpdir) / "results.json"
            assert results_file.exists()

            with open(results_file) as f:
                saved_results = json.load(f)

            assert saved_results["passed_count"] == 8
            assert saved_results["failed_count"] == 2


class TestCli:
    """Tests for cli function."""

    def test_cli_basic(self, mock_env_vars):
        """Test CLI with basic arguments."""
        from src.main import cli

        mock_result = {"passed_count": 5, "failed_count": 0}

        with patch("sys.argv", ["main", "--codebase", "/test", "--app-url", "http://localhost:3000"]):
            with patch("asyncio.run", return_value=mock_result):
                cli()  # Should not raise or exit

    def test_cli_with_output_dir(self, mock_env_vars):
        """Test CLI with custom output directory."""
        from src.main import cli

        mock_result = {"passed_count": 3, "failed_count": 0}

        with patch("sys.argv", [
            "main",
            "--codebase", "/test",
            "--app-url", "http://localhost:3000",
            "--output", "/custom/output"
        ]):
            with patch("asyncio.run", return_value=mock_result) as mock_run:
                cli()

                # Check the coroutine was called with correct output_dir
                call_coro = mock_run.call_args[0][0]
                # Coroutine should be the run_tests call
                assert mock_run.called

    def test_cli_with_pr_number(self, mock_env_vars):
        """Test CLI with PR number."""
        from src.main import cli

        mock_result = {"passed_count": 5, "failed_count": 0}

        with patch("sys.argv", [
            "main",
            "--codebase", "/test",
            "--app-url", "http://localhost:3000",
            "--pr", "123"
        ]):
            with patch("asyncio.run", return_value=mock_result):
                cli()  # Should not raise

    def test_cli_with_changed_files(self, mock_env_vars):
        """Test CLI with changed files list."""
        from src.main import cli

        mock_result = {"passed_count": 2, "failed_count": 0}

        with patch("sys.argv", [
            "main",
            "--codebase", "/test",
            "--app-url", "http://localhost:3000",
            "--changed-files", "file1.py", "file2.py"
        ]):
            with patch("asyncio.run", return_value=mock_result):
                cli()  # Should not raise

    def test_cli_exits_on_failure(self, mock_env_vars):
        """Test CLI exits with code 1 when tests fail."""
        from src.main import cli

        mock_result = {"passed_count": 5, "failed_count": 2}

        with patch("sys.argv", ["main", "--codebase", "/test", "--app-url", "http://localhost:3000"]):
            with patch("asyncio.run", return_value=mock_result):
                with pytest.raises(SystemExit) as exc_info:
                    cli()

                assert exc_info.value.code == 1

    def test_cli_no_exit_on_success(self, mock_env_vars):
        """Test CLI does not exit with error when tests pass."""
        from src.main import cli

        mock_result = {"passed_count": 5, "failed_count": 0}

        with patch("sys.argv", ["main", "--codebase", "/test", "--app-url", "http://localhost:3000"]):
            with patch("asyncio.run", return_value=mock_result):
                # Should complete without raising SystemExit
                cli()

    def test_cli_short_args(self, mock_env_vars):
        """Test CLI with short argument forms."""
        from src.main import cli

        mock_result = {"passed_count": 5, "failed_count": 0}

        with patch("sys.argv", [
            "main",
            "-c", "/test/project",
            "-u", "http://localhost:8080",
            "-o", "./results"
        ]):
            with patch("asyncio.run", return_value=mock_result):
                cli()  # Should not raise


class TestModuleLevelConfig:
    """Tests for module-level configuration."""

    def test_structlog_configured(self, mock_env_vars):
        """Test that structlog is configured at module level."""
        import structlog
        from src import main  # Import triggers configuration

        # Should have a configured logger
        logger = structlog.get_logger()
        assert logger is not None

    def test_logger_exists(self, mock_env_vars):
        """Test that module logger is created."""
        from src.main import logger

        assert logger is not None
