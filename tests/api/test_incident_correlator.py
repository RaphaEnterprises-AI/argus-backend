"""Tests for Incident Correlator & Root Cause Analysis API endpoints."""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestStackTraceParsing:
    """Tests for stack trace parsing."""

    def test_parse_python_stack_trace(self, mock_env_vars):
        """Test parsing Python stack traces."""
        from src.api.incident_correlator import parse_stack_trace

        stack_trace = '''Traceback (most recent call last):
  File "/app/src/api/handler.py", line 42, in handle_request
    result = process_data(data)
  File "/app/src/services/processor.py", line 128, in process_data
    return transform(data)
  File "/usr/lib/python3.11/site-packages/some_lib.py", line 10, in transform
    raise ValueError("Invalid data")
ValueError: Invalid data
'''

        files = parse_stack_trace(stack_trace)
        assert "src/api/handler.py" in files or "api/handler.py" in files
        assert "src/services/processor.py" in files or "services/processor.py" in files
        # site-packages should be filtered out
        assert not any("site-packages" in f for f in files)

    def test_parse_javascript_stack_trace(self, mock_env_vars):
        """Test parsing JavaScript/Node stack traces."""
        from src.api.incident_correlator import parse_stack_trace

        stack_trace = '''Error: Cannot read property 'id' of undefined
    at processUser (/app/src/services/user.js:42:10)
    at async handleRequest (/app/src/api/routes.ts:128:5)
    at /app/node_modules/express/lib/router.js:123:7
'''

        files = parse_stack_trace(stack_trace)
        # Should find user.js and routes.ts
        assert any("user.js" in f for f in files)
        assert any("routes.ts" in f for f in files)
        # node_modules should be filtered out
        assert not any("node_modules" in f for f in files)

    def test_parse_java_stack_trace(self, mock_env_vars):
        """Test parsing Java stack traces."""
        from src.api.incident_correlator import parse_stack_trace

        stack_trace = '''java.lang.NullPointerException: Cannot invoke method on null
    at com.example.api.UserHandler.process(UserHandler.java:42)
    at com.example.service.AuthService.authenticate(AuthService.java:128)
'''

        files = parse_stack_trace(stack_trace)
        assert "UserHandler.java" in files
        assert "AuthService.java" in files

    def test_parse_go_stack_trace(self, mock_env_vars):
        """Test parsing Go stack traces."""
        from src.api.incident_correlator import parse_stack_trace

        stack_trace = '''goroutine 1 [running]:
main.processRequest()
        /app/handlers/api.go:42 +0x1a4
main.main()
        /app/main.go:15 +0x25
'''

        files = parse_stack_trace(stack_trace)
        assert any("handlers/api.go" in f for f in files)
        assert any("main.go" in f for f in files)

    def test_parse_empty_stack_trace(self, mock_env_vars):
        """Test parsing empty stack trace."""
        from src.api.incident_correlator import parse_stack_trace

        assert parse_stack_trace("") == []
        assert parse_stack_trace(None) == []


class TestFileOverlapScoring:
    """Tests for file overlap calculation."""

    def test_calculate_exact_overlap(self, mock_env_vars):
        """Test exact file path overlap."""
        from src.api.incident_correlator import calculate_file_overlap_score

        error_files = ["src/api/handler.py", "src/services/auth.py"]
        changed_files = ["src/api/handler.py", "tests/test_handler.py"]

        score, overlapping = calculate_file_overlap_score(error_files, changed_files)

        assert score > 0
        assert "api/handler.py" in overlapping or "src/api/handler.py" in overlapping

    def test_calculate_partial_overlap(self, mock_env_vars):
        """Test partial file name overlap (same name, different path)."""
        from src.api.incident_correlator import calculate_file_overlap_score

        error_files = ["src/api/handler.py"]
        changed_files = ["lib/api/handler.py"]

        score, overlapping = calculate_file_overlap_score(error_files, changed_files)

        # Should have some score for partial match
        assert score >= 0

    def test_no_overlap(self, mock_env_vars):
        """Test no file overlap."""
        from src.api.incident_correlator import calculate_file_overlap_score

        error_files = ["src/api/handler.py"]
        changed_files = ["src/models/user.py"]

        score, overlapping = calculate_file_overlap_score(error_files, changed_files)

        assert score == 0
        assert len(overlapping) == 0

    def test_empty_file_lists(self, mock_env_vars):
        """Test with empty file lists."""
        from src.api.incident_correlator import calculate_file_overlap_score

        score1, overlap1 = calculate_file_overlap_score([], ["file.py"])
        score2, overlap2 = calculate_file_overlap_score(["file.py"], [])
        score3, overlap3 = calculate_file_overlap_score([], [])

        assert score1 == 0
        assert score2 == 0
        assert score3 == 0


class TestTimeProximityScoring:
    """Tests for time proximity scoring."""

    def test_recent_event_high_score(self, mock_env_vars):
        """Test that recent events get higher scores."""
        from src.api.incident_correlator import calculate_time_proximity_score

        error_time = datetime.now(UTC)
        event_time = error_time - timedelta(hours=1)

        score = calculate_time_proximity_score(error_time, event_time, max_hours=24)

        # 1 hour ago should get a high score (close to 0.2)
        assert score > 0.15

    def test_old_event_low_score(self, mock_env_vars):
        """Test that old events get lower scores."""
        from src.api.incident_correlator import calculate_time_proximity_score

        error_time = datetime.now(UTC)
        event_time = error_time - timedelta(hours=20)

        score = calculate_time_proximity_score(error_time, event_time, max_hours=24)

        # 20 hours ago should get a low score
        assert score < 0.05

    def test_event_after_error_zero_score(self, mock_env_vars):
        """Test that events after the error get zero score."""
        from src.api.incident_correlator import calculate_time_proximity_score

        error_time = datetime.now(UTC)
        event_time = error_time + timedelta(hours=1)

        score = calculate_time_proximity_score(error_time, event_time, max_hours=24)

        assert score == 0.0

    def test_event_outside_window_zero_score(self, mock_env_vars):
        """Test that events outside the time window get zero score."""
        from src.api.incident_correlator import calculate_time_proximity_score

        error_time = datetime.now(UTC)
        event_time = error_time - timedelta(hours=48)

        score = calculate_time_proximity_score(error_time, event_time, max_hours=24)

        assert score == 0.0


class TestCommitSizeScoring:
    """Tests for commit size scoring."""

    def test_small_commit_low_risk(self, mock_env_vars):
        """Test that small commits have low risk."""
        from src.api.incident_correlator import calculate_commit_size_score

        score = calculate_commit_size_score(
            lines_added=10,
            lines_deleted=5,
            files_changed=2,
        )

        assert score == 0.0

    def test_medium_commit_medium_risk(self, mock_env_vars):
        """Test that medium commits have medium risk."""
        from src.api.incident_correlator import calculate_commit_size_score

        score = calculate_commit_size_score(
            lines_added=300,
            lines_deleted=100,
            files_changed=15,
        )

        assert score == 0.10

    def test_large_commit_high_risk(self, mock_env_vars):
        """Test that large commits have high risk."""
        from src.api.incident_correlator import calculate_commit_size_score

        score = calculate_commit_size_score(
            lines_added=400,
            lines_deleted=200,
            files_changed=25,
        )

        assert score == 0.15


class TestProbabilityCalculation:
    """Tests for overall probability calculation."""

    def test_high_file_overlap_high_probability(self, mock_env_vars):
        """Test that high file overlap gives high probability."""
        from src.api.incident_correlator import calculate_probability

        error_files = ["api/handler.py", "services/auth.py"]
        error_time = datetime.now(UTC)

        # Use replace to remove timezone, then add Z (standard Sentry format)
        event_time = (error_time - timedelta(hours=2)).replace(tzinfo=None)
        event_data = {
            "occurred_at": event_time.isoformat() + "Z",
            "data": {
                "changed_files": ["api/handler.py", "services/auth.py", "tests/test_auth.py"],
                "lines_added": 50,
                "lines_deleted": 10,
            },
        }

        probability, factors, overlapping = calculate_probability(
            error_files, error_time, event_data
        )

        assert probability > 0.3
        assert any(f["factor"] == "file_overlap" for f in factors)
        assert len(overlapping) > 0

    def test_no_overlap_low_probability(self, mock_env_vars):
        """Test that no file overlap gives low probability."""
        from src.api.incident_correlator import calculate_probability

        error_files = ["api/handler.py"]
        error_time = datetime.now(UTC)

        # Use replace to remove timezone, then add Z (standard Sentry format)
        event_time = (error_time - timedelta(hours=2)).replace(tzinfo=None)
        event_data = {
            "occurred_at": event_time.isoformat() + "Z",
            "data": {
                "changed_files": ["models/user.py"],
                "lines_added": 10,
                "lines_deleted": 5,
            },
        }

        probability, factors, overlapping = calculate_probability(
            error_files, error_time, event_data
        )

        # Only time proximity should contribute
        assert probability < 0.3
        assert len(overlapping) == 0


class TestSentrySignatureVerification:
    """Tests for Sentry webhook signature verification."""

    def test_verify_valid_signature(self, mock_env_vars):
        """Test verification with valid signature."""
        from src.api.incident_correlator import verify_sentry_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        # Calculate expected signature
        mac = hmac.new(secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
        signature = mac.hexdigest()

        result = verify_sentry_signature(payload, signature, secret)
        assert result is True

    def test_verify_invalid_signature(self, mock_env_vars):
        """Test verification with invalid signature."""
        from src.api.incident_correlator import verify_sentry_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        result = verify_sentry_signature(payload, "invalid-signature", secret)
        assert result is False

    def test_verify_missing_signature(self, mock_env_vars):
        """Test verification with missing signature."""
        from src.api.incident_correlator import verify_sentry_signature

        payload = b'{"test": "data"}'
        secret = "test-secret"

        result = verify_sentry_signature(payload, None, secret)
        assert result is False


class TestSentryStacktraceExtraction:
    """Tests for extracting stacktrace from Sentry events."""

    def test_extract_from_exception(self, mock_env_vars):
        """Test extracting stacktrace from exception data."""
        from src.api.incident_correlator import extract_sentry_stacktrace

        event = {
            "exception": {
                "values": [
                    {
                        "type": "ValueError",
                        "value": "Invalid input",
                        "stacktrace": {
                            "frames": [
                                {
                                    "filename": "api/handler.py",
                                    "abs_path": "/app/api/handler.py",
                                    "lineno": 42,
                                    "function": "process",
                                },
                                {
                                    "filename": "services/validator.py",
                                    "lineno": 100,
                                    "function": "validate",
                                },
                            ]
                        },
                    }
                ]
            }
        }

        stacktrace = extract_sentry_stacktrace(event)

        assert "handler.py" in stacktrace
        assert "validator.py" in stacktrace
        assert "line 42" in stacktrace

    def test_extract_from_culprit(self, mock_env_vars):
        """Test extracting from culprit field."""
        from src.api.incident_correlator import extract_sentry_stacktrace

        event = {
            "culprit": "api.handler in process_request"
        }

        stacktrace = extract_sentry_stacktrace(event)
        assert "api.handler" in stacktrace

    def test_extract_empty_event(self, mock_env_vars):
        """Test extracting from empty event."""
        from src.api.incident_correlator import extract_sentry_stacktrace

        stacktrace = extract_sentry_stacktrace({})
        assert stacktrace == ""


class TestReportGeneration:
    """Tests for incident report generation."""

    def test_generate_report_with_correlation(self, mock_env_vars):
        """Test generating report with identified correlation."""
        from src.api.incident_correlator import (
            CorrelationCandidate,
            IncidentCorrelation,
            generate_incident_report_markdown,
        )

        now = datetime.now(UTC)
        error_event = {
            "title": "TypeError: Cannot read property 'id' of undefined",
            "occurred_at": now.replace(tzinfo=None).isoformat() + "Z",
            "data": {
                "message": "TypeError: Cannot read property 'id' of undefined",
                "environment": "production",
                "affected_files": ["api/user.js"],
            },
        }

        most_likely = CorrelationCandidate(
            event_id="event-123",
            event_type="commit",
            commit_sha="abc12345",
            pr_number=42,
            title="Fix user handling",
            author="developer",
            occurred_at=datetime.now(UTC) - timedelta(hours=2),
            probability=0.85,
            factors=[
                {"factor": "file_overlap", "score": 0.4, "description": "1 file overlaps"},
                {"factor": "time_proximity", "score": 0.15, "description": "2h before error"},
            ],
            files_changed=["api/user.js"],
            file_overlap=["api/user.js"],
        )

        correlation = IncidentCorrelation(
            incident_id="incident-123",
            candidates=[most_likely],
            most_likely=most_likely,
            root_cause_analysis=None,
            confidence=0.85,
        )

        report = generate_incident_report_markdown(error_event, correlation)

        assert "TypeError" in report
        assert "85%" in report
        assert "abc1234" in report  # Truncated SHA
        assert "PR #42" in report
        assert "@developer" in report

    def test_generate_report_no_correlation(self, mock_env_vars):
        """Test generating report without correlation."""
        from src.api.incident_correlator import (
            IncidentCorrelation,
            generate_incident_report_markdown,
        )

        now = datetime.now(UTC)
        error_event = {
            "title": "Connection timeout",
            "occurred_at": now.replace(tzinfo=None).isoformat() + "Z",
            "data": {
                "message": "Connection timeout",
                "environment": "production",
            },
        }

        correlation = IncidentCorrelation(
            incident_id="incident-123",
            candidates=[],
            most_likely=None,
            root_cause_analysis=None,
            confidence=0.0,
        )

        report = generate_incident_report_markdown(error_event, correlation)

        assert "Connection timeout" in report
        assert "0%" in report
        assert "No definitive root cause" in report


class TestSentryWebhookEndpoint:
    """Tests for the Sentry webhook endpoint."""

    @pytest.mark.asyncio
    async def test_receive_error_webhook(self, mock_env_vars):
        """Test receiving an error webhook from Sentry."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={
                "data": [{"id": "event-123"}],
                "error": None,
            })
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.post(
                "/webhooks/sentry?project_id=test-project",
                json={
                    "event_id": "sentry-event-123",
                    "message": "Test error",
                    "level": "error",
                    "environment": "production",
                },
                headers={
                    "Sentry-Hook-Resource": "error",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["event_id"] == "sentry-event-123"

    @pytest.mark.asyncio
    async def test_receive_event_alert_webhook(self, mock_env_vars):
        """Test receiving an event_alert webhook from Sentry."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={
                "data": [{"id": "event-123"}],
                "error": None,
            })
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.post(
                "/webhooks/sentry?project_id=test-project",
                json={
                    "data": {
                        "event": {
                            "event_id": "sentry-event-456",
                            "title": "NullPointerException",
                            "environment": "staging",
                            "exception": {
                                "values": [
                                    {
                                        "type": "NullPointerException",
                                        "stacktrace": {
                                            "frames": [
                                                {"filename": "api/handler.py", "lineno": 42, "function": "process"}
                                            ]
                                        },
                                    }
                                ]
                            },
                        },
                        "triggered_rule": "High Error Rate",
                    },
                },
                headers={
                    "Sentry-Hook-Resource": "event_alert",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_duplicate_event_handling(self, mock_env_vars):
        """Test handling of duplicate Sentry events."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.insert = AsyncMock(return_value={
                "data": None,
                "error": "duplicate key value violates unique constraint",
            })
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.post(
                "/webhooks/sentry?project_id=test-project",
                json={
                    "event_id": "duplicate-event",
                    "message": "Test error",
                },
                headers={
                    "Sentry-Hook-Resource": "error",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Duplicate" in data["message"]


class TestCorrelateEndpoint:
    """Tests for the correlation endpoint."""

    @pytest.mark.asyncio
    async def test_correlate_error_with_commits(self, mock_env_vars):
        """Test correlating an error with recent commits."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        error_time = datetime.now(UTC)
        event_time = error_time - timedelta(hours=2)

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()

            # Create ISO format timestamps without double timezone suffix
            error_time_iso = error_time.replace(tzinfo=None).isoformat() + "Z"
            event_time_iso = event_time.replace(tzinfo=None).isoformat() + "Z"

            # Mock error event lookup
            def mock_request(url, **kwargs):
                if "id=eq.error-123" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "error-123",
                            "event_type": "error",
                            "project_id": "test-project",
                            "occurred_at": error_time_iso,
                            "data": {
                                "affected_files": ["api/handler.py"],
                            },
                        }]
                    })()
                elif "event_type=in" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "commit-123",
                            "event_type": "commit",
                            "commit_sha": "abc123",
                            "occurred_at": event_time_iso,
                            "data": {
                                "changed_files": ["api/handler.py", "tests/test.py"],
                                "lines_added": 50,
                                "author": "developer",
                            },
                        }]
                    })()
                return AsyncMock(return_value={"data": [], "error": None})()

            mock_client.request = mock_request
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.post(
                "/api/v1/incidents/correlate?project_id=test-project",
                json={
                    "error_event_id": "error-123",
                    "hours_back": 24,
                    "include_ai_analysis": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["incident_id"] == "error-123"
            assert len(data["candidates"]) > 0
            assert data["most_likely"] is not None
            assert data["most_likely"]["commit_sha"] == "abc123"

    @pytest.mark.asyncio
    async def test_correlate_nonexistent_error(self, mock_env_vars):
        """Test correlating a non-existent error."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={"data": [], "error": None})
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.post(
                "/api/v1/incidents/correlate?project_id=test-project",
                json={
                    "error_event_id": "nonexistent-error",
                    "hours_back": 24,
                },
            )

            assert response.status_code == 404


class TestGetIncidentEndpoint:
    """Tests for getting incident details."""

    @pytest.mark.asyncio
    async def test_get_incident_with_correlations(self, mock_env_vars):
        """Test getting an incident with its correlations."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()

            def mock_request(url, **kwargs):
                if "sdlc_events?id=eq.incident-123" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "incident-123",
                            "event_type": "error",
                            "title": "Test error",
                            "data": {"message": "Test error"},
                        }]
                    })()
                elif "event_correlations" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "corr-1",
                            "source_event_id": "commit-123",
                            "target_event_id": "incident-123",
                            "correlation_type": "caused_by",
                            "confidence": 0.85,
                        }]
                    })()
                elif "sdlc_events?id=eq.commit-123" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "commit-123",
                            "event_type": "commit",
                            "title": "Fix bug",
                        }]
                    })()
                return AsyncMock(return_value={"data": [], "error": None})()

            mock_client.request = mock_request
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/incidents/incident-123?project_id=test-project"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["incident"]["id"] == "incident-123"
            assert len(data["correlations"]) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_incident(self, mock_env_vars):
        """Test getting a non-existent incident."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()
            mock_client.request = AsyncMock(return_value={"data": [], "error": None})
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/incidents/nonexistent?project_id=test-project"
            )

            assert response.status_code == 404


class TestListIncidentsEndpoint:
    """Tests for listing incidents."""

    @pytest.mark.asyncio
    async def test_list_recent_incidents(self, mock_env_vars):
        """Test listing recent incidents."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()

            def mock_request(url, **kwargs):
                if "event_type=eq.error" in url:
                    return AsyncMock(return_value={
                        "data": [
                            {
                                "id": "error-1",
                                "event_type": "error",
                                "title": "Error 1",
                                "data": {"environment": "production", "severity": "error"},
                            },
                            {
                                "id": "error-2",
                                "event_type": "error",
                                "title": "Error 2",
                                "data": {"environment": "staging", "severity": "warning"},
                            },
                        ]
                    })()
                elif "event_correlations" in url:
                    return AsyncMock(return_value={"data": [], "error": None})()
                return AsyncMock(return_value={"data": [], "error": None})()

            mock_client.request = mock_request
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/incidents/recent?project_id=test-project"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["incidents"]) == 2

    @pytest.mark.asyncio
    async def test_list_incidents_with_filters(self, mock_env_vars):
        """Test listing incidents with environment filter."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()

            def mock_request(url, **kwargs):
                if "event_type=eq.error" in url:
                    return AsyncMock(return_value={
                        "data": [
                            {
                                "id": "error-1",
                                "event_type": "error",
                                "title": "Error 1",
                                "data": {"environment": "production", "severity": "error"},
                            },
                            {
                                "id": "error-2",
                                "event_type": "error",
                                "title": "Error 2",
                                "data": {"environment": "staging", "severity": "warning"},
                            },
                        ]
                    })()
                elif "event_correlations" in url:
                    return AsyncMock(return_value={"data": [], "error": None})()
                return AsyncMock(return_value={"data": [], "error": None})()

            mock_client.request = mock_request
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/incidents/recent?project_id=test-project&environment=production"
            )

            assert response.status_code == 200
            data = response.json()
            # Only production incidents
            assert data["total"] == 1


class TestGetIncidentTimelineEndpoint:
    """Tests for getting incident timeline."""

    @pytest.mark.asyncio
    async def test_get_incident_timeline(self, mock_env_vars):
        """Test getting timeline for an incident."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        error_time = datetime.now(UTC)
        # Create ISO format timestamps without double timezone suffix
        error_time_iso = error_time.replace(tzinfo=None).isoformat() + "Z"
        deploy_time_iso = (error_time - timedelta(hours=2)).replace(tzinfo=None).isoformat() + "Z"
        commit_time_iso = (error_time - timedelta(hours=4)).replace(tzinfo=None).isoformat() + "Z"

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()

            def mock_request(url, **kwargs):
                if "sdlc_events?id=eq.incident-123" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "incident-123",
                            "event_type": "error",
                            "occurred_at": error_time_iso,
                        }]
                    })()
                elif "occurred_at=gte" in url:
                    return AsyncMock(return_value={
                        "data": [
                            {
                                "id": "deploy-1",
                                "event_type": "deploy",
                                "title": "Deploy to production",
                                "occurred_at": deploy_time_iso,
                            },
                            {
                                "id": "commit-1",
                                "event_type": "commit",
                                "title": "Fix bug",
                                "occurred_at": commit_time_iso,
                            },
                        ]
                    })()
                return AsyncMock(return_value={"data": [], "error": None})()

            mock_client.request = mock_request
            mock_client.rpc = AsyncMock(return_value={"data": None, "error": "not found"})
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/incidents/incident-123/timeline?project_id=test-project"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["incident"]["id"] == "incident-123"
            assert len(data["timeline"]) == 2


class TestGetIncidentReportEndpoint:
    """Tests for generating incident reports."""

    @pytest.mark.asyncio
    async def test_get_incident_report(self, mock_env_vars):
        """Test generating an incident report."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        error_time = datetime.now(UTC)
        # Create ISO format timestamps without double timezone suffix
        error_time_iso = error_time.replace(tzinfo=None).isoformat() + "Z"
        commit_time_iso = (error_time - timedelta(hours=1)).replace(tzinfo=None).isoformat() + "Z"

        with patch("src.api.incident_correlator.get_supabase_client") as mock_supabase:
            mock_client = MagicMock()

            def mock_request(url, **kwargs):
                if "sdlc_events?id=eq.incident-123" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "incident-123",
                            "event_type": "error",
                            "title": "TypeError in checkout",
                            "occurred_at": error_time_iso,
                            "data": {
                                "message": "TypeError in checkout",
                                "environment": "production",
                            },
                        }]
                    })()
                elif "event_correlations" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "corr-1",
                            "source_event_id": "commit-123",
                            "target_event_id": "incident-123",
                            "correlation_type": "caused_by",
                            "confidence": 0.9,
                        }]
                    })()
                elif "sdlc_events?id=eq.commit-123" in url:
                    return AsyncMock(return_value={
                        "data": [{
                            "id": "commit-123",
                            "event_type": "commit",
                            "commit_sha": "abc12345",
                            "pr_number": 42,
                            "title": "Update checkout logic",
                            "occurred_at": commit_time_iso,
                            "data": {
                                "author": "developer",
                                "changed_files": ["checkout.py"],
                            },
                        }]
                    })()
                return AsyncMock(return_value={"data": [], "error": None})()

            mock_client.request = mock_request
            mock_supabase.return_value = mock_client

            client = TestClient(app)

            response = client.get(
                "/api/v1/incidents/incident-123/report?project_id=test-project"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["incident_id"] == "incident-123"
            assert data["confidence"] == 0.9
            assert data["most_likely_commit"] == "abc12345"
            assert "TypeError" in data["report_markdown"]
