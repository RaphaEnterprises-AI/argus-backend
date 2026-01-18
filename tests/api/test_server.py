"""Tests for FastAPI server module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRequestModels:
    """Tests for request model validation."""

    def test_test_run_request(self, mock_env_vars):
        """Test TestRunRequest model."""
        from src.api.server import TestRunRequest

        request = TestRunRequest(
            codebase_path="/path/to/code",
            app_url="http://localhost:3000",
            pr_number=123,
            changed_files=["src/app.py"],
            max_tests=10,
            focus_areas=["authentication"],
        )

        assert request.codebase_path == "/path/to/code"
        assert request.app_url == "http://localhost:3000"
        assert request.pr_number == 123
        assert request.changed_files == ["src/app.py"]
        assert request.max_tests == 10
        assert request.focus_areas == ["authentication"]

    def test_test_run_request_minimal(self, mock_env_vars):
        """Test TestRunRequest with only required fields."""
        from src.api.server import TestRunRequest

        request = TestRunRequest(
            codebase_path="/path/to/code",
            app_url="http://localhost:3000",
        )

        assert request.pr_number is None
        assert request.changed_files is None
        assert request.max_tests is None
        assert request.focus_areas is None

    def test_nlp_test_request(self, mock_env_vars):
        """Test NLPTestRequest model."""
        from src.api.server import NLPTestRequest

        request = NLPTestRequest(
            description="Login as admin and verify dashboard",
            context="E-commerce application",
            project_id="test-project-id",
        )

        assert request.description == "Login as admin and verify dashboard"
        assert request.context == "E-commerce application"
        assert request.project_id == "test-project-id"

    def test_visual_compare_request(self, mock_env_vars):
        """Test VisualCompareRequest model."""
        from src.api.server import VisualCompareRequest

        request = VisualCompareRequest(
            baseline_b64="base64data1",
            current_b64="base64data2",
            context="Homepage comparison",
        )

        assert request.baseline_b64 == "base64data1"
        assert request.current_b64 == "base64data2"
        assert request.context == "Homepage comparison"

    def test_webhook_payload(self, mock_env_vars):
        """Test WebhookPayload model."""
        from src.api.server import WebhookPayload

        payload = WebhookPayload(
            action="opened",
            repository={"full_name": "user/repo"},
            pull_request={"number": 123},
        )

        assert payload.action == "opened"
        assert payload.repository == {"full_name": "user/repo"}
        assert payload.pull_request == {"number": 123}


class TestResponseModels:
    """Tests for response model creation."""

    def test_test_run_response(self, mock_env_vars):
        """Test TestRunResponse model."""
        from src.api.server import TestRunResponse

        response = TestRunResponse(
            job_id="test-123",
            status="pending",
            message="Test run started",
            created_at="2024-01-01T00:00:00",
        )

        assert response.job_id == "test-123"
        assert response.status == "pending"

    def test_job_status_response(self, mock_env_vars):
        """Test JobStatusResponse model."""
        from src.api.server import JobStatusResponse

        response = JobStatusResponse(
            job_id="test-123",
            status="completed",
            progress={"phase": "done"},
            result={"summary": "All tests passed"},
            created_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:10:00",
        )

        assert response.job_id == "test-123"
        assert response.status == "completed"
        assert response.progress == {"phase": "done"}
        assert response.result == {"summary": "All tests passed"}

    def test_health_response(self, mock_env_vars):
        """Test HealthResponse model."""
        from src.api.server import HealthResponse

        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp="2024-01-01T00:00:00",
        )

        assert response.status == "healthy"
        assert response.version == "0.1.0"


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, mock_env_vars):
        """Test /health endpoint."""
        from src.api.server import health_check

        response = await health_check()

        from src.api.server import API_VERSION
        assert response.status == "healthy"
        assert response.version == API_VERSION
        assert response.timestamp is not None

    @pytest.mark.asyncio
    async def test_readiness_check_ready(self, mock_env_vars):
        """Test /ready endpoint when ready."""
        from src.api.server import readiness_check

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.output_dir = "/tmp"

        with patch("src.api.server.get_settings", return_value=mock_settings):
            with patch("os.path.exists", return_value=True):
                response = await readiness_check()

                assert response.status_code == 200
                body = response.body.decode()
                assert '"ready":true' in body

    @pytest.mark.asyncio
    async def test_readiness_check_not_ready(self, mock_env_vars):
        """Test /ready endpoint when not ready."""
        from src.api.server import readiness_check

        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = None  # Not configured
        mock_settings.output_dir = "/nonexistent"

        with patch("src.api.server.get_settings", return_value=mock_settings):
            with patch("os.path.exists", return_value=False):
                response = await readiness_check()

                assert response.status_code == 503
                body = response.body.decode()
                assert '"ready":false' in body


class TestTestRunEndpoints:
    """Tests for test run endpoints."""

    @pytest.mark.asyncio
    async def test_start_test_run(self, mock_env_vars):
        """Test /api/v1/tests/run endpoint."""
        from src.api.server import TestRunRequest, jobs, start_test_run

        # Clear any existing jobs
        jobs.clear()

        mock_background = MagicMock()
        mock_background.add_task = MagicMock()

        request = TestRunRequest(
            codebase_path="/path/to/code",
            app_url="http://localhost:3000",
        )

        response = await start_test_run(request, mock_background)

        assert response.status == "pending"
        assert response.job_id is not None
        assert response.job_id in jobs
        mock_background.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_job_status_found(self, mock_env_vars):
        """Test /api/v1/jobs/{job_id} when job exists."""
        from src.api.server import get_job_status, jobs

        job_id = "test-job-123"
        jobs[job_id] = {
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
            "progress": {"phase": "testing"},
        }

        response = await get_job_status(job_id)

        assert response.job_id == job_id
        assert response.status == "running"
        assert response.progress == {"phase": "testing"}

    @pytest.mark.asyncio
    async def test_get_job_status_not_found(self, mock_env_vars):
        """Test /api/v1/jobs/{job_id} when job doesn't exist."""
        from fastapi import HTTPException

        from src.api.server import get_job_status, jobs

        jobs.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_job_status("nonexistent-job")

        assert exc_info.value.status_code == 404
        assert "Job not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_list_jobs(self, mock_env_vars):
        """Test /api/v1/jobs endpoint."""
        from src.api.server import jobs, list_jobs

        jobs.clear()
        jobs["job-1"] = {"status": "completed", "created_at": "2024-01-01T00:00:00"}
        jobs["job-2"] = {"status": "running", "created_at": "2024-01-01T00:05:00"}
        jobs["job-3"] = {"status": "completed", "created_at": "2024-01-01T00:10:00"}

        # All jobs
        response = await list_jobs()
        assert response["total"] == 3
        assert len(response["jobs"]) == 3

        # Filter by status
        response = await list_jobs(status="completed")
        assert response["total"] == 2
        assert all(j["status"] == "completed" for j in response["jobs"])

        # Limit
        response = await list_jobs(limit=1)
        assert len(response["jobs"]) == 1


class TestRunTestsBackground:
    """Tests for run_tests_background function."""

    @pytest.mark.asyncio
    async def test_run_tests_background_success(self, mock_env_vars):
        """Test successful background test run."""
        from src.api.server import jobs, run_tests_background

        job_id = "test-job"
        jobs[job_id] = {
            "status": "pending",
            "progress": {"phase": "initializing"},
        }

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = AsyncMock(return_value={"test_results": []})
        mock_orchestrator.get_run_summary = MagicMock(return_value={"total": 5})

        mock_reporter = MagicMock()
        mock_reporter.generate_all = MagicMock(return_value={"json": "/path/to/report.json"})

        mock_settings = MagicMock()
        mock_settings.output_dir = "/tmp"

        with patch(
            "src.api.server.TestingOrchestrator",
            return_value=mock_orchestrator,
        ):
            with patch("src.api.server.create_initial_state", return_value={}):
                with patch("src.api.server.get_settings", return_value=mock_settings):
                    with patch(
                        "src.api.server.create_reporter",
                        return_value=mock_reporter,
                    ):
                        with patch(
                            "src.api.server.create_report_from_state",
                            return_value={},
                        ):
                            await run_tests_background(
                                job_id,
                                "/path/to/code",
                                "http://localhost:3000",
                                None,
                                None,
                                None,
                            )

        assert jobs[job_id]["status"] == "completed"
        assert "result" in jobs[job_id]
        assert "completed_at" in jobs[job_id]

    @pytest.mark.asyncio
    async def test_run_tests_background_failure(self, mock_env_vars):
        """Test failed background test run."""
        from src.api.server import jobs, run_tests_background

        job_id = "test-job-fail"
        jobs[job_id] = {
            "status": "pending",
            "progress": {"phase": "initializing"},
        }

        with patch(
            "src.api.server.TestingOrchestrator",
            side_effect=Exception("Orchestrator failed"),
        ):
            await run_tests_background(
                job_id,
                "/path/to/code",
                "http://localhost:3000",
                None,
                None,
                None,
            )

        assert jobs[job_id]["status"] == "failed"
        assert "Orchestrator failed" in jobs[job_id]["error"]


class TestNLPEndpoints:
    """Tests for NLP test creation endpoints."""

    @pytest.mark.asyncio
    async def test_create_test_from_nlp_success(self, mock_env_vars):
        """Test /api/v1/tests/create success."""
        from src.api.server import NLPTestRequest, create_test_from_nlp

        mock_test = MagicMock()
        mock_test.to_dict = MagicMock(return_value={"name": "Login Test", "steps": [], "assertions": []})
        mock_test.to_spec = MagicMock(return_value={"steps": []})

        mock_creator = MagicMock()
        mock_creator.create = AsyncMock(return_value=mock_test)

        nlp_request = NLPTestRequest(description="Login as admin", project_id="test-project")
        mock_http_request = MagicMock()

        # Mock user authentication
        mock_user = {"user_id": "test-user", "email": "test@example.com"}

        # Mock Supabase client with async insert
        mock_supabase = MagicMock()
        mock_supabase.insert = AsyncMock(return_value={
            "data": [{
                "id": "test-id",
                "name": "Login Test",
                "project_id": "test-project",
                "description": "Login as admin",
                "steps": [],
                "tags": [],
                "priority": "medium",
            }],
            "error": None
        })

        with patch("src.agents.nlp_test_creator.NLPTestCreator", return_value=mock_creator), \
             patch("src.api.teams.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_project_access", AsyncMock()), \
             patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.teams.log_audit", AsyncMock()), \
             patch("src.api.tests.get_project_org_id", AsyncMock(return_value="test-org-id")):
            response = await create_test_from_nlp(nlp_request, mock_http_request)

            assert response["success"] is True
            assert "test" in response

    @pytest.mark.asyncio
    async def test_create_test_from_nlp_failure(self, mock_env_vars):
        """Test /api/v1/tests/create failure."""
        from fastapi import HTTPException

        from src.api.server import NLPTestRequest, create_test_from_nlp

        nlp_request = NLPTestRequest(description="Test something", project_id="test-project")
        mock_http_request = MagicMock()

        # Mock user authentication
        mock_user = {"user_id": "test-user", "email": "test@example.com"}

        with patch("src.api.teams.get_current_user", AsyncMock(return_value=mock_user)), \
             patch("src.api.projects.verify_project_access", AsyncMock()), \
             patch("src.agents.nlp_test_creator.NLPTestCreator", side_effect=Exception("NLP parsing failed")):
            with pytest.raises(HTTPException) as exc_info:
                await create_test_from_nlp(nlp_request, mock_http_request)

            assert exc_info.value.status_code == 500


class TestVisualAIEndpoints:
    """Tests for Visual AI endpoints."""

    @pytest.mark.asyncio
    async def test_compare_screenshots_success(self, mock_env_vars):
        """Test /api/v1/visual/compare success."""
        from src.api.server import VisualCompareRequest, compare_screenshots

        mock_result = MagicMock()
        mock_result.to_dict = MagicMock(return_value={
            "similarity": 0.95,
            "differences": [],
        })

        mock_visual_ai = MagicMock()
        mock_visual_ai.compare = AsyncMock(return_value=mock_result)

        request = VisualCompareRequest(
            baseline_b64="base64-baseline",
            current_b64="base64-current",
        )

        with patch(
            "src.agents.visual_ai.VisualAI",
            return_value=mock_visual_ai,
        ):
            response = await compare_screenshots(request)

            assert response["success"] is True
            assert response["result"]["similarity"] == 0.95

    @pytest.mark.asyncio
    async def test_compare_screenshots_failure(self, mock_env_vars):
        """Test /api/v1/visual/compare failure."""
        from fastapi import HTTPException

        from src.api.server import VisualCompareRequest, compare_screenshots

        request = VisualCompareRequest(
            baseline_b64="invalid",
            current_b64="invalid",
        )

        with patch(
            "src.agents.visual_ai.VisualAI",
            side_effect=Exception("Comparison failed"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await compare_screenshots(request)

            assert exc_info.value.status_code == 500


class TestAutoDiscoveryEndpoints:
    """Tests for auto-discovery endpoints."""

    @pytest.mark.asyncio
    async def test_discover_tests_success(self, mock_env_vars):
        """Test /api/v1/discover success."""
        from src.api.server import discover_tests

        mock_result = MagicMock()
        mock_result.to_dict = MagicMock(return_value={
            "pages_discovered": 10,
            "tests_generated": 5,
        })

        mock_discovery = MagicMock()
        mock_discovery.discover = AsyncMock(return_value=mock_result)

        with patch(
            "src.agents.auto_discovery.AutoDiscovery",
            return_value=mock_discovery,
        ):
            response = await discover_tests(
                app_url="http://localhost:3000",
                focus_areas=["login"],
            )

            assert response["success"] is True
            assert response["result"]["pages_discovered"] == 10

    @pytest.mark.asyncio
    async def test_discover_tests_failure(self, mock_env_vars):
        """Test /api/v1/discover failure."""
        from fastapi import HTTPException

        from src.api.server import discover_tests

        with patch(
            "src.agents.auto_discovery.AutoDiscovery",
            side_effect=Exception("Discovery failed"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await discover_tests(app_url="http://localhost:3000")

            assert exc_info.value.status_code == 500


class TestWebhookEndpoints:
    """Tests for webhook endpoints."""

    @pytest.mark.asyncio
    async def test_github_webhook_pr_opened(self, mock_env_vars):
        """Test /api/v1/webhooks/github for PR opened."""
        from src.api.server import github_webhook, jobs

        jobs.clear()

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "opened",
            "pull_request": {"number": 123},
            "repository": {"full_name": "user/repo"},
        })

        mock_background = MagicMock()
        mock_background.add_task = MagicMock()

        response = await github_webhook(mock_request, mock_background)

        assert response["status"] == "accepted"
        assert "job_id" in response
        mock_background.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_github_webhook_pr_synchronize(self, mock_env_vars):
        """Test /api/v1/webhooks/github for PR synchronize."""
        from src.api.server import github_webhook, jobs

        jobs.clear()

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "synchronize",
            "pull_request": {"number": 123},
            "repository": {"full_name": "user/repo"},
        })

        mock_background = MagicMock()
        mock_background.add_task = MagicMock()

        response = await github_webhook(mock_request, mock_background)

        assert response["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_github_webhook_ignored_action(self, mock_env_vars):
        """Test /api/v1/webhooks/github ignores other actions."""
        from src.api.server import github_webhook

        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={
            "action": "closed",
            "pull_request": {"number": 123},
        })

        mock_background = MagicMock()

        response = await github_webhook(mock_request, mock_background)

        assert response["status"] == "ignored"
        assert response["action"] == "closed"


class TestReportEndpoints:
    """Tests for report endpoints."""

    @pytest.mark.asyncio
    async def test_get_report_success(self, mock_env_vars):
        """Test /api/v1/reports/{job_id} success."""
        import os
        import tempfile

        from src.api.server import get_report, jobs

        # Create a temporary report file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"test": "data"}')
            report_path = f.name

        try:
            job_id = "report-job"
            jobs[job_id] = {
                "status": "completed",
                "result": {"report_paths": {"json": report_path}},
            }

            response = await get_report(job_id, format="json")

            assert response.status_code == 200
        finally:
            os.unlink(report_path)

    @pytest.mark.asyncio
    async def test_get_report_job_not_found(self, mock_env_vars):
        """Test /api/v1/reports/{job_id} job not found."""
        from fastapi import HTTPException

        from src.api.server import get_report, jobs

        jobs.clear()

        with pytest.raises(HTTPException) as exc_info:
            await get_report("nonexistent")

        assert exc_info.value.status_code == 404
        assert "Job not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_report_not_completed(self, mock_env_vars):
        """Test /api/v1/reports/{job_id} when job not completed."""
        from fastapi import HTTPException

        from src.api.server import get_report, jobs

        job_id = "running-job"
        jobs[job_id] = {"status": "running"}

        with pytest.raises(HTTPException) as exc_info:
            await get_report(job_id)

        assert exc_info.value.status_code == 400
        assert "not yet completed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_report_format_not_available(self, mock_env_vars):
        """Test /api/v1/reports/{job_id} format not available."""
        from fastapi import HTTPException

        from src.api.server import get_report, jobs

        job_id = "completed-job"
        jobs[job_id] = {
            "status": "completed",
            "result": {"report_paths": {"json": "/path/to/report.json"}},
        }

        with pytest.raises(HTTPException) as exc_info:
            await get_report(job_id, format="html")

        assert exc_info.value.status_code == 404
        assert "not available" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_report_file_not_found(self, mock_env_vars):
        """Test /api/v1/reports/{job_id} when file doesn't exist."""
        from fastapi import HTTPException

        from src.api.server import get_report, jobs

        job_id = "completed-job"
        jobs[job_id] = {
            "status": "completed",
            "result": {"report_paths": {"json": "/nonexistent/report.json"}},
        }

        with pytest.raises(HTTPException) as exc_info:
            await get_report(job_id, format="json")

        assert exc_info.value.status_code == 404
        assert "file not found" in str(exc_info.value.detail)


class TestExceptionHandler:
    """Tests for global exception handler."""

    @pytest.mark.asyncio
    async def test_global_exception_handler(self, mock_env_vars):
        """Test global exception handler."""
        from src.api.server import global_exception_handler

        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        exc = ValueError("Test error")

        with patch.dict("os.environ", {}, clear=False):
            response = await global_exception_handler(mock_request, exc)

            assert response.status_code == 500
            body = response.body.decode()
            assert "Internal server error" in body

    @pytest.mark.asyncio
    async def test_global_exception_handler_debug_mode(self, mock_env_vars):
        """Test global exception handler in debug mode."""
        from src.api.server import global_exception_handler

        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        exc = ValueError("Detailed error message")

        with patch.dict("os.environ", {"DEBUG": "true"}):
            response = await global_exception_handler(mock_request, exc)

            body = response.body.decode()
            assert "Detailed error message" in body


class TestStartupShutdown:
    """Tests for startup and shutdown events."""

    @pytest.mark.asyncio
    async def test_startup(self, mock_env_vars):
        """Test startup event."""
        from src.api.server import startup

        mock_settings = MagicMock()
        mock_settings.output_dir = "./test-results"

        with patch("src.api.server.get_settings", return_value=mock_settings):
            with patch("os.makedirs") as mock_makedirs:
                await startup()

                mock_makedirs.assert_called_once_with(
                    "./test-results", exist_ok=True
                )

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_env_vars):
        """Test shutdown event."""
        from src.api.server import shutdown

        # Should complete without error
        await shutdown()


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_metadata(self, mock_env_vars):
        """Test FastAPI app metadata."""
        from src.api.server import API_VERSION, app

        assert app.title == "Argus E2E Testing Agent API"
        assert app.version == API_VERSION
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_cors_middleware(self, mock_env_vars):
        """Test CORS middleware is configured."""
        from src.api.server import app

        # Check that CORSMiddleware is in the middleware stack
        middleware_classes = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_classes
