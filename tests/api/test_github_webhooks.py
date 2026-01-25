"""Tests for GitHub Webhooks API endpoints."""

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


class TestGitHubSignatureVerification:
    """Tests for GitHub webhook signature verification."""

    def test_verify_valid_signature(self, mock_env_vars):
        """Test verification with valid signature."""
        from src.api.github_webhooks import verify_github_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        # Calculate expected signature
        mac = hmac.new(secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
        signature = f"sha256={mac.hexdigest()}"

        result = verify_github_signature(payload, signature, secret)
        assert result is True

    def test_verify_invalid_signature(self, mock_env_vars):
        """Test verification with invalid signature."""
        from src.api.github_webhooks import verify_github_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        result = verify_github_signature(payload, "sha256=invalid", secret)
        assert result is False

    def test_verify_missing_signature(self, mock_env_vars):
        """Test verification with missing signature."""
        from src.api.github_webhooks import verify_github_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        result = verify_github_signature(payload, None, secret)
        assert result is False

    def test_verify_invalid_format(self, mock_env_vars):
        """Test verification with invalid format."""
        from src.api.github_webhooks import verify_github_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        # Missing sha256= prefix
        result = verify_github_signature(payload, "invalid-format", secret)
        assert result is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_commit_files(self, mock_env_vars):
        """Test parsing files from commit data."""
        from src.api.github_webhooks import parse_commit_files

        commit_data = {
            "files": [
                {"filename": "src/api/handler.py", "status": "modified"},
                {"filename": "tests/test_handler.py", "status": "added"},
                {"filename": "", "status": "removed"},  # Empty filename should be filtered
            ]
        }

        files = parse_commit_files(commit_data)
        assert len(files) == 2
        assert "src/api/handler.py" in files
        assert "tests/test_handler.py" in files

    def test_parse_commit_files_empty(self, mock_env_vars):
        """Test parsing files from empty commit data."""
        from src.api.github_webhooks import parse_commit_files

        assert parse_commit_files({}) == []
        assert parse_commit_files({"files": []}) == []

    def test_parse_commit_stats(self, mock_env_vars):
        """Test parsing commit statistics."""
        from src.api.github_webhooks import parse_commit_stats

        commit_data = {
            "stats": {
                "additions": 50,
                "deletions": 20,
            },
            "files": [{"filename": "a.py"}, {"filename": "b.py"}, {"filename": "c.py"}],
        }

        stats = parse_commit_stats(commit_data)
        assert stats["lines_added"] == 50
        assert stats["lines_deleted"] == 20
        assert stats["files_changed"] == 3

    def test_parse_commit_stats_empty(self, mock_env_vars):
        """Test parsing stats from empty commit data."""
        from src.api.github_webhooks import parse_commit_stats

        stats = parse_commit_stats({})
        assert stats["lines_added"] == 0
        assert stats["lines_deleted"] == 0
        assert stats["files_changed"] == 0

    def test_extract_components(self, mock_env_vars):
        """Test extracting components from file paths."""
        from src.api.github_webhooks import extract_components

        files = [
            "src/api/handler.py",
            "src/api/routes.py",
            "src/services/auth.py",
            "tests/test_handler.py",
            "config.py",  # Top-level file
        ]

        components = extract_components(files)
        assert "api" in components
        assert "services" in components
        # Top-level directories like src should not be included
        assert "src" not in components
        # tests should not be included
        assert "tests" not in components

    def test_extract_components_empty(self, mock_env_vars):
        """Test extracting components from empty list."""
        from src.api.github_webhooks import extract_components

        assert extract_components([]) == []

    def test_calculate_risk_score_low_risk(self, mock_env_vars):
        """Test risk calculation for low-risk commit."""
        from src.api.github_webhooks import calculate_risk_score

        stats = {"lines_added": 10, "lines_deleted": 5, "files_changed": 2}
        predictions = []
        security_issues = []
        affected_tests = []

        risk, factors = calculate_risk_score(stats, predictions, security_issues, affected_tests)
        assert risk == 0.0
        assert len(factors) == 0

    def test_calculate_risk_score_large_change(self, mock_env_vars):
        """Test risk calculation for large commit."""
        from src.api.github_webhooks import calculate_risk_score

        stats = {"lines_added": 600, "lines_deleted": 100, "files_changed": 25}
        predictions = []
        security_issues = []
        affected_tests = []

        risk, factors = calculate_risk_score(stats, predictions, security_issues, affected_tests)
        assert risk > 0.0
        assert any(f["factor"] == "large_change" for f in factors)
        assert any(f["factor"] == "many_files" for f in factors)

    def test_calculate_risk_score_with_predictions(self, mock_env_vars):
        """Test risk calculation with failure predictions."""
        from src.api.github_webhooks import calculate_risk_score

        stats = {"lines_added": 50, "lines_deleted": 10, "files_changed": 3}
        predictions = [
            {"failure_probability": 0.8, "reason": "Pattern match"},
            {"failure_probability": 0.6, "reason": "Another pattern"},
        ]
        security_issues = []
        affected_tests = []

        risk, factors = calculate_risk_score(stats, predictions, security_issues, affected_tests)
        assert risk > 0.0
        assert any(f["factor"] == "predicted_failures" for f in factors)

    def test_calculate_risk_score_with_security(self, mock_env_vars):
        """Test risk calculation with security issues."""
        from src.api.github_webhooks import calculate_risk_score

        stats = {"lines_added": 50, "lines_deleted": 10, "files_changed": 3}
        predictions = []
        security_issues = [
            {"severity": "critical", "type": "sql_injection"},
            {"severity": "high", "type": "xss"},
        ]
        affected_tests = []

        risk, factors = calculate_risk_score(stats, predictions, security_issues, affected_tests)
        assert risk > 0.0
        assert any(f["factor"] == "critical_security" for f in factors)

    def test_determine_deployment_strategy_safe(self, mock_env_vars):
        """Test deployment strategy for safe commit."""
        from src.api.github_webhooks import determine_deployment_strategy

        strategy, notes = determine_deployment_strategy(0.1, [])
        assert strategy == "safe_to_deploy"

    def test_determine_deployment_strategy_medium_risk(self, mock_env_vars):
        """Test deployment strategy for medium risk commit."""
        from src.api.github_webhooks import determine_deployment_strategy

        strategy, notes = determine_deployment_strategy(0.5, [])
        assert strategy == "deploy_with_monitoring"

    def test_determine_deployment_strategy_high_risk(self, mock_env_vars):
        """Test deployment strategy for high risk commit."""
        from src.api.github_webhooks import determine_deployment_strategy

        strategy, notes = determine_deployment_strategy(0.8, [])
        assert strategy == "staged_rollout"

    def test_determine_deployment_strategy_blocked(self, mock_env_vars):
        """Test deployment strategy with critical security issues."""
        from src.api.github_webhooks import determine_deployment_strategy

        security_issues = [{"severity": "critical", "type": "sql_injection"}]
        strategy, notes = determine_deployment_strategy(0.3, security_issues)
        assert strategy == "blocked"

    def test_determine_deployment_strategy_manual_review(self, mock_env_vars):
        """Test deployment strategy with high severity security issues."""
        from src.api.github_webhooks import determine_deployment_strategy

        security_issues = [{"severity": "high", "type": "xss"}]
        strategy, notes = determine_deployment_strategy(0.2, security_issues)
        assert strategy == "manual_review"


class TestWebhookEventHandlers:
    """Tests for webhook event handlers."""

    @pytest.mark.asyncio
    async def test_handle_push_event(self, mock_env_vars):
        """Test handling push event."""
        from src.api.github_webhooks import handle_push_event

        payload = {
            "ref": "refs/heads/main",
            "repository": {
                "full_name": "owner/repo",
                "html_url": "https://github.com/owner/repo",
            },
            "commits": [
                {"id": "abc123", "message": "Test commit"},
            ],
            "head_commit": {
                "id": "abc123",
                "message": "Test commit",
                "timestamp": "2026-01-25T10:00:00Z",
            },
            "pusher": {"name": "testuser"},
        }

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "event-123"}]})
            # Return full analysis record from request
            mock_client.request = AsyncMock(return_value={
                "data": [{
                    "id": "analysis-123",
                    "commit_sha": "abc123",
                    "risk_score": 0.1,
                    "deployment_strategy": "safe_to_deploy",
                }],
                "error": None,
            })
            mock_client.rpc = AsyncMock(return_value={"data": [], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.github_webhooks.fetch_github_diff", AsyncMock(return_value={})):
                background_tasks = MagicMock()
                result = await handle_push_event(
                    project_id="project-123",
                    payload=payload,
                    delivery_id="delivery-123",
                    background_tasks=background_tasks,
                )

                assert "sdlc_event_id" in result
                # Analysis should be triggered for head commit
                assert "commit_analysis_id" in result

    @pytest.mark.asyncio
    async def test_handle_pull_request_event_opened(self, mock_env_vars):
        """Test handling pull_request opened event."""
        from src.api.github_webhooks import handle_pull_request_event

        payload = {
            "action": "opened",
            "pull_request": {
                "number": 42,
                "title": "Add new feature",
                "body": "This PR adds a cool feature",
                "state": "open",
                "draft": False,
                "head": {
                    "sha": "abc123",
                    "ref": "feature/new-feature",
                },
                "base": {
                    "ref": "main",
                },
                "user": {"login": "testuser"},
                "html_url": "https://github.com/owner/repo/pull/42",
                "updated_at": "2026-01-25T10:00:00Z",
            },
            "repository": {
                "full_name": "owner/repo",
            },
        }

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "event-123"}]})
            # Return full analysis record from request
            mock_client.request = AsyncMock(return_value={
                "data": [{
                    "id": "analysis-123",
                    "commit_sha": "abc123",
                    "risk_score": 0.1,
                    "deployment_strategy": "safe_to_deploy",
                }],
                "error": None,
            })
            mock_client.rpc = AsyncMock(return_value={"data": [], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.github_webhooks.fetch_github_diff", AsyncMock(return_value={})):
                background_tasks = MagicMock()
                result = await handle_pull_request_event(
                    project_id="project-123",
                    payload=payload,
                    delivery_id="delivery-123",
                    background_tasks=background_tasks,
                )

                assert "sdlc_event_id" in result
                # Analysis should be triggered for opened PRs
                assert "commit_analysis_id" in result

    @pytest.mark.asyncio
    async def test_handle_pull_request_event_closed(self, mock_env_vars):
        """Test handling pull_request closed event (no analysis)."""
        from src.api.github_webhooks import handle_pull_request_event

        payload = {
            "action": "closed",
            "pull_request": {
                "number": 42,
                "title": "Add new feature",
                "state": "closed",
                "head": {"sha": "abc123", "ref": "feature"},
                "base": {"ref": "main"},
                "user": {"login": "testuser"},
                "updated_at": "2026-01-25T10:00:00Z",
            },
            "repository": {"full_name": "owner/repo"},
        }

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "event-123"}]})
            mock_supabase.return_value = mock_client

            background_tasks = MagicMock()
            result = await handle_pull_request_event(
                project_id="project-123",
                payload=payload,
                delivery_id="delivery-123",
                background_tasks=background_tasks,
            )

            assert "sdlc_event_id" in result
            # No analysis for closed PRs
            assert result.get("commit_analysis_id") is None

    @pytest.mark.asyncio
    async def test_handle_check_run_event(self, mock_env_vars):
        """Test handling check_run event."""
        from src.api.github_webhooks import handle_check_run_event

        payload = {
            "action": "completed",
            "check_run": {
                "id": 123456,
                "name": "CI",
                "status": "completed",
                "conclusion": "success",
                "head_sha": "abc123",
                "html_url": "https://github.com/owner/repo/runs/123456",
                "completed_at": "2026-01-25T10:00:00Z",
            },
            "repository": {"full_name": "owner/repo"},
        }

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "event-123"}]})
            mock_supabase.return_value = mock_client

            background_tasks = MagicMock()
            result = await handle_check_run_event(
                project_id="project-123",
                payload=payload,
                delivery_id="delivery-123",
                background_tasks=background_tasks,
            )

            assert "sdlc_event_id" in result

    @pytest.mark.asyncio
    async def test_handle_deployment_event(self, mock_env_vars):
        """Test handling deployment event."""
        from src.api.github_webhooks import handle_deployment_event

        payload = {
            "deployment": {
                "id": 789,
                "environment": "production",
                "description": "Deploy to prod",
                "sha": "abc123",
                "ref": "main",
                "task": "deploy",
                "creator": {"login": "testuser"},
                "created_at": "2026-01-25T10:00:00Z",
            },
            "repository": {"full_name": "owner/repo"},
        }

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "event-123"}]})
            mock_supabase.return_value = mock_client

            background_tasks = MagicMock()
            result = await handle_deployment_event(
                project_id="project-123",
                payload=payload,
                delivery_id="delivery-123",
                background_tasks=background_tasks,
            )

            assert "sdlc_event_id" in result

    @pytest.mark.asyncio
    async def test_handle_deployment_status_event(self, mock_env_vars):
        """Test handling deployment_status event."""
        from src.api.github_webhooks import handle_deployment_status_event

        payload = {
            "deployment_status": {
                "id": 456,
                "state": "success",
                "description": "Deployment successful",
                "environment": "production",
                "target_url": "https://app.example.com",
                "created_at": "2026-01-25T10:00:00Z",
            },
            "deployment": {
                "id": 789,
                "sha": "abc123",
            },
            "repository": {"full_name": "owner/repo"},
        }

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={"data": [{"id": "event-123"}]})
            mock_supabase.return_value = mock_client

            background_tasks = MagicMock()
            result = await handle_deployment_status_event(
                project_id="project-123",
                payload=payload,
                delivery_id="delivery-123",
                background_tasks=background_tasks,
            )

            assert "sdlc_event_id" in result


class TestCommitAnalysis:
    """Tests for commit analysis."""

    @pytest.mark.asyncio
    async def test_analyze_commit(self, mock_env_vars):
        """Test full commit analysis."""
        from src.api.github_webhooks import analyze_commit

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            # Return full analysis record
            mock_client.request = AsyncMock(return_value={
                "data": [{
                    "id": "analysis-123",
                    "commit_sha": "abc123",
                    "risk_score": 0.1,
                    "deployment_strategy": "safe_to_deploy",
                }]
            })
            mock_client.rpc = AsyncMock(return_value={"data": [], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.github_webhooks.fetch_github_diff") as mock_fetch:
                mock_fetch.return_value = {
                    "sha": "abc123",
                    "commit": {
                        "author": {
                            "email": "test@example.com",
                            "name": "Test User",
                        },
                    },
                    "files": [
                        {"filename": "src/api/handler.py"},
                        {"filename": "tests/test_handler.py"},
                    ],
                    "stats": {
                        "additions": 50,
                        "deletions": 10,
                    },
                }

                result = await analyze_commit(
                    project_id="project-123",
                    commit_sha="abc123",
                    pr_number=42,
                    branch_name="feature/test",
                    repository="owner/repo",
                )

                assert "commit_sha" in result
                assert "risk_score" in result
                assert "deployment_strategy" in result

    @pytest.mark.asyncio
    async def test_analyze_commit_no_github_data(self, mock_env_vars):
        """Test commit analysis without GitHub data."""
        from src.api.github_webhooks import analyze_commit

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            # Return full analysis record
            mock_client.request = AsyncMock(return_value={
                "data": [{
                    "id": "analysis-123",
                    "commit_sha": "abc123",
                    "files_changed": 0,
                    "risk_score": 0.0,
                    "deployment_strategy": "safe_to_deploy",
                }]
            })
            mock_client.rpc = AsyncMock(return_value={"data": [], "error": None})
            mock_supabase.return_value = mock_client

            with patch("src.api.github_webhooks.fetch_github_diff", AsyncMock(return_value={})):
                result = await analyze_commit(
                    project_id="project-123",
                    commit_sha="abc123",
                )

                # Should still return analysis with default values
                assert "commit_sha" in result
                assert result.get("files_changed", 0) == 0


class TestWebhookEndpoint:
    """Tests for the main webhook endpoint."""

    @pytest.mark.asyncio
    async def test_receive_webhook_missing_event_header(self, mock_env_vars):
        """Test webhook without X-GitHub-Event header."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/webhooks/github?project_id=test-project",
            json={"test": "data"},
            headers={"X-GitHub-Delivery": "delivery-123"},
        )

        assert response.status_code == 400
        assert "Missing X-GitHub-Event" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_receive_webhook_missing_delivery_header(self, mock_env_vars):
        """Test webhook without X-GitHub-Delivery header."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        client = TestClient(app)

        response = client.post(
            "/api/v1/webhooks/github?project_id=test-project",
            json={"test": "data"},
            headers={"X-GitHub-Event": "push"},
        )

        assert response.status_code == 400
        assert "Missing X-GitHub-Delivery" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_receive_webhook_unsupported_event(self, mock_env_vars):
        """Test webhook with unsupported event type."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.github_webhooks.store_webhook_event", AsyncMock(return_value={})):
            with patch("src.api.github_webhooks.update_webhook_status", AsyncMock()):
                client = TestClient(app)

                response = client.post(
                    "/api/v1/webhooks/github?project_id=test-project",
                    json={"test": "data"},
                    headers={
                        "X-GitHub-Event": "star",  # Unsupported event
                        "X-GitHub-Delivery": "delivery-123",
                    },
                )

                # Unsupported events should return success but skip processing
                assert response.status_code == 200
                assert response.json()["success"] is True
                assert "not processed" in response.json()["message"]


class TestAnalyzeEndpoint:
    """Tests for the analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_endpoint(self, mock_env_vars):
        """Test manual commit analysis endpoint."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.github_webhooks.analyze_commit") as mock_analyze:
            mock_analyze.return_value = {
                "id": "analysis-123",
                "commit_sha": "abc123",
                "pr_number": 42,
                "risk_score": 0.3,
                "deployment_strategy": "safe_to_deploy",
                "tests_to_run_suggested": [{"test_name": "test_handler"}],
                "predicted_test_failures": [],
                "security_vulnerabilities": [],
                "recommendations": [],
            }

            client = TestClient(app)

            response = client.post(
                "/api/v1/webhooks/github/analyze?project_id=test-project",
                json={
                    "commit_sha": "abc123",
                    "pr_number": 42,
                    "branch_name": "feature/test",
                    "repository": "owner/repo",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["commit_sha"] == "abc123"
            assert data["risk_score"] == 0.3
            assert data["deployment_strategy"] == "safe_to_deploy"
            assert data["tests_to_run_count"] == 1


class TestGetAnalysis:
    """Tests for getting commit analysis."""

    @pytest.mark.asyncio
    async def test_get_analysis_found(self, mock_env_vars):
        """Test getting existing commit analysis."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={
                "data": [{
                    "id": "analysis-123",
                    "commit_sha": "abc123",
                    "risk_score": 0.3,
                    "deployment_strategy": "safe_to_deploy",
                }]
            })
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/webhooks/github/analysis/abc123?project_id=test-project"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["commit_sha"] == "abc123"

    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self, mock_env_vars):
        """Test getting non-existent commit analysis."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={"data": []})
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/webhooks/github/analysis/nonexistent?project_id=test-project"
            )

            assert response.status_code == 404


class TestListEvents:
    """Tests for listing webhook events."""

    @pytest.mark.asyncio
    async def test_list_events(self, mock_env_vars):
        """Test listing webhook events."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={
                "data": [
                    {"delivery_id": "d1", "event_type": "push", "status": "completed"},
                    {"delivery_id": "d2", "event_type": "pull_request", "status": "completed"},
                ]
            })
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/webhooks/github/events?project_id=test-project"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["events"]) == 2

    @pytest.mark.asyncio
    async def test_list_events_with_filters(self, mock_env_vars):
        """Test listing webhook events with filters."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.github_webhooks.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={
                "data": [
                    {"delivery_id": "d1", "event_type": "push", "status": "failed"},
                ]
            })
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/webhooks/github/events"
                "?project_id=test-project"
                "&event_type=push"
                "&status=failed"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
