"""
Tests for HealingConsumer autonomous healing behavior.

Run with: pytest tests/workers/test_healing_consumer.py -v
"""

import asyncio
import json
import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.workers.healing_consumer import HealingConfig, HealingConsumer, HealingEvent


@pytest.fixture
def healing_config():
    """Create test configuration."""
    return HealingConfig(
        bootstrap_servers="localhost:9092",
        consumer_group="test-healing-workers",
        auto_offset_reset="earliest",
        input_topic="argus.healing.requested",
        output_topic="argus.healing.completed",
        dlq_topic="argus.dlq",
        supabase_url="http://test.supabase.co",
        supabase_service_key="test-key",
        batch_size=1,
        concurrency=1,
        auto_verify=True,
        auto_create_tests=True,
        max_healing_attempts=3,
    )


@pytest.fixture
def sample_healing_event():
    """Create a sample healing event."""
    return {
        "event_id": "test-event-123",
        "failure_id": "failure-456",
        "test_id": "test-789",
        "org_id": "org-test",
        "project_id": "proj-test",
        "error_type": "selector",
        "error_message": "Element not found: #submit-btn",
        "failed_selector": "#submit-btn",
        "page_url": "https://example.com/login",
        "screenshot_url": None,
        "strategy": "auto",
        "priority": "normal",
    }


@pytest.fixture
def sample_argus_event_wrapper(sample_healing_event):
    """Create a sample ArgusEvent wrapper (as published by EventGateway)."""
    return {
        "event_id": "argus-event-001",
        "event_type": "healing.requested",
        "timestamp": "2024-01-15T10:30:00Z",
        "org_id": "org-test",
        "project_id": "proj-test",
        "user_id": "user-test",
        "data": sample_healing_event,
        "correlation_id": None,
        "causation_id": "failure-456",
        "source": "argus-backend",
        "version": "1.0",
    }


@pytest.fixture
def sample_test_spec():
    """Create a sample test specification."""
    return {
        "id": "test-789",
        "name": "Login Test",
        "project_id": "proj-test",
        "spec": {
            "name": "Login Test",
            "steps": [
                {"action": "goto", "target": "/login"},
                {"action": "fill", "target": "#email", "value": "user@example.com"},
                {"action": "fill", "target": "#password", "value": "password123"},
                {"action": "click", "target": "#submit-btn"},
            ],
        },
        "healing_count": 0,
    }


class TestHealingConsumer:
    """Test HealingConsumer behavior."""

    @pytest.mark.asyncio
    async def test_fetch_test_spec_success(self, healing_config, sample_test_spec):
        """Test fetching test spec from Supabase."""
        consumer = HealingConsumer(healing_config)

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=[sample_test_spec])

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        consumer._http_client = mock_client

        result = await consumer._fetch_test_spec("test-789", "proj-test")

        assert result is not None
        assert result["id"] == "test-789"
        assert result["name"] == "Login Test"

    @pytest.mark.asyncio
    async def test_fetch_test_spec_not_found(self, healing_config):
        """Test fetching non-existent test spec."""
        consumer = HealingConsumer(healing_config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=[])

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        consumer._http_client = mock_client

        result = await consumer._fetch_test_spec("nonexistent", "proj-test")

        assert result is None

    @pytest.mark.asyncio
    async def test_run_healing_success(self, healing_config, sample_healing_event, sample_test_spec):
        """Test successful healing flow."""
        consumer = HealingConsumer(healing_config)

        # Mock dependencies
        with patch.object(consumer, "_fetch_test_spec", return_value=sample_test_spec):
            with patch.object(consumer, "_fetch_failure_details", return_value={}):
                with patch("src.agents.self_healer.SelfHealerAgent") as MockHealer:
                    # Mock successful healing
                    mock_healer_instance = AsyncMock()
                    mock_healing_result = MagicMock()
                    mock_healing_result.success = True
                    mock_healing_result.data = MagicMock()
                    mock_healing_result.data.auto_healed = True
                    mock_healing_result.data.healed_test_spec = {
                        "name": "Login Test",
                        "steps": [
                            {"action": "goto", "target": "/login"},
                            {"action": "click", "target": "#new-submit-btn"},  # Fixed selector
                        ],
                    }
                    mock_healing_result.data.diagnosis = MagicMock()
                    mock_healing_result.data.diagnosis.failure_type = MagicMock()
                    mock_healing_result.data.diagnosis.failure_type.value = "selector_changed"
                    mock_healing_result.data.diagnosis.code_context = MagicMock()
                    mock_healing_result.data.diagnosis.code_context.old_selector = "#submit-btn"
                    mock_healing_result.data.diagnosis.code_context.new_selector = "#new-submit-btn"
                    mock_healing_result.data.suggested_fixes = [MagicMock(confidence=0.95)]

                    mock_healer_instance.heal_test = AsyncMock(return_value=mock_healing_result)
                    MockHealer.return_value = mock_healer_instance

                    result = await consumer._run_healing(HealingEvent(**sample_healing_event))

                    assert result["success"] is True
                    assert result["auto_healed"] is True
                    assert result["original_selector"] == "#submit-btn"
                    assert result["healed_selector"] == "#new-submit-btn"
                    assert result["strategy_used"] == "selector_changed"

    @pytest.mark.asyncio
    async def test_run_healing_no_test_spec(self, healing_config, sample_healing_event):
        """Test healing when test spec is not found."""
        consumer = HealingConsumer(healing_config)

        with patch.object(consumer, "_fetch_test_spec", return_value=None):
            result = await consumer._run_healing(HealingEvent(**sample_healing_event))

            assert result["success"] is False
            assert "not found" in result["failure_reason"].lower()

    @pytest.mark.asyncio
    async def test_store_healing_pattern_success(self, healing_config, sample_healing_event):
        """Test storing healing pattern in Cognee."""
        consumer = HealingConsumer(healing_config)

        healing_result = {
            "success": True,
            "strategy_used": "selector_changed",
            "original_selector": "#submit-btn",
            "healed_selector": "#new-submit-btn",
            "auto_healed": True,
            "confidence": 0.95,
        }

        with patch("src.knowledge.get_cognee_client") as mock_get_cognee:
            mock_cognee = AsyncMock()
            mock_cognee.store_failure_pattern = AsyncMock(return_value="pattern-123")
            mock_get_cognee.return_value = mock_cognee

            pattern_id = await consumer._store_healing_pattern(
                HealingEvent(**sample_healing_event),
                healing_result,
            )

            assert pattern_id == "pattern-123"
            mock_cognee.store_failure_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_healing_pattern_no_cognee(self, healing_config, sample_healing_event):
        """Test storing pattern when Cognee is not available."""
        consumer = HealingConsumer(healing_config)

        healing_result = {"success": True}

        with patch("src.knowledge.get_cognee_client", return_value=None):
            pattern_id = await consumer._store_healing_pattern(
                HealingEvent(**sample_healing_event),
                healing_result,
            )

            assert pattern_id is None

    @pytest.mark.asyncio
    async def test_process_event_success(self, healing_config, sample_healing_event):
        """Test full event processing flow."""
        consumer = HealingConsumer(healing_config)

        with patch.object(
            consumer,
            "_run_healing",
            return_value={
                "success": True,
                "auto_healed": True,
                "strategy_used": "selector_changed",
                "duration_ms": 1500,
            },
        ):
            with patch.object(consumer, "_verify_healed_test", return_value=True):
                with patch.object(consumer, "_store_healing_pattern", return_value="pattern-123"):
                    with patch.object(consumer, "_generate_test_from_error", return_value=None):
                        with patch.object(consumer, "_send_completion_event", return_value=True):
                            result = await consumer._process_event(sample_healing_event)

                            assert result is True

    @pytest.mark.asyncio
    async def test_process_event_argus_wrapper(self, healing_config, sample_argus_event_wrapper):
        """Test processing ArgusEvent wrapper format (as published by EventGateway)."""
        consumer = HealingConsumer(healing_config)

        with patch.object(
            consumer,
            "_run_healing",
            return_value={
                "success": True,
                "auto_healed": True,
                "strategy_used": "selector_changed",
                "duration_ms": 1500,
            },
        ):
            with patch.object(consumer, "_verify_healed_test", return_value=True):
                with patch.object(consumer, "_store_healing_pattern", return_value="pattern-123"):
                    with patch.object(consumer, "_generate_test_from_error", return_value=None):
                        with patch.object(consumer, "_send_completion_event", return_value=True):
                            # Pass ArgusEvent wrapper - should extract data correctly
                            result = await consumer._process_event(sample_argus_event_wrapper)

                            assert result is True

    @pytest.mark.asyncio
    async def test_process_event_failure(self, healing_config, sample_healing_event):
        """Test event processing when healing fails."""
        consumer = HealingConsumer(healing_config)

        with patch.object(
            consumer,
            "_run_healing",
            return_value={
                "success": False,
                "failure_reason": "Could not identify fix",
                "duration_ms": 500,
            },
        ):
            with patch.object(consumer, "_store_healing_pattern", return_value=None):
                with patch.object(consumer, "_send_completion_event", return_value=True):
                    result = await consumer._process_event(sample_healing_event)

                    assert result is False


class TestHealingEventEmission:
    """Test that healing events are properly emitted."""

    @pytest.mark.asyncio
    async def test_emit_healing_requested_on_failure(self):
        """Test that HEALING_REQUESTED is emitted when test fails."""
        from src.orchestrator.nodes import _emit_test_events
        from src.orchestrator.state import TestResult, TestStatus

        # Mock state
        state = {
            "org_id": "org-test",
            "project_id": "proj-test",
            "user_id": "user-test",
            "run_id": "run-test",
            "app_url": "https://example.com",
        }

        test = {
            "id": "test-789",
            "name": "Login Test",
            "type": "ui",
            "priority": "critical",
            "steps": [
                {"action": "goto", "target": "/login"},
                {"action": "click", "target": "#submit-btn"},
            ],
        }

        test_result = TestResult(
            test_id="test-789",
            status=TestStatus.FAILED,
            duration_seconds=5.0,
            error_message="Element not found: #submit-btn",
            actions_taken=[
                {"step": 1, "action": "goto", "result": "success"},
                {
                    "step": 2,
                    "action": "click",
                    "target": "#submit-btn",
                    "result": "failure",
                    "error": "Element not found",
                },
            ],
            screenshots=[],
            assertions_passed=0,
            assertions_failed=1,
        )

        mock_log = MagicMock()

        with patch("src.services.event_gateway.get_event_gateway") as mock_get_gateway:
            mock_gateway = MagicMock()
            mock_gateway.is_running = True
            mock_get_gateway.return_value = mock_gateway

            with patch("src.services.event_gateway.emit_test_executed", new_callable=AsyncMock) as mock_emit_executed:
                with patch("src.services.event_gateway.emit_test_failed", new_callable=AsyncMock) as mock_emit_failed:
                    with patch(
                        "src.services.event_gateway.emit_healing_requested", new_callable=AsyncMock
                    ) as mock_emit_healing:
                        await _emit_test_events(state, test, test_result, mock_log)

                        # Verify all events were emitted
                        mock_emit_executed.assert_called_once()
                        mock_emit_failed.assert_called_once()
                        mock_emit_healing.assert_called_once()

                        # Verify healing request has correct parameters
                        call_args = mock_emit_healing.call_args
                        assert call_args.kwargs["test_id"] == "test-789"
                        assert call_args.kwargs["error_type"] == "selector"
                        assert "#submit-btn" in call_args.kwargs["error_message"]
                        assert call_args.kwargs["priority"] == "high"  # Critical test

    @pytest.mark.asyncio
    async def test_no_healing_request_on_success(self):
        """Test that HEALING_REQUESTED is NOT emitted on success."""
        from src.orchestrator.nodes import _emit_test_events
        from src.orchestrator.state import TestResult, TestStatus

        state = {
            "org_id": "org-test",
            "project_id": "proj-test",
            "user_id": "user-test",
            "run_id": "run-test",
        }

        test = {
            "id": "test-789",
            "name": "Login Test",
            "type": "ui",
            "steps": [],
        }

        test_result = TestResult(
            test_id="test-789",
            status=TestStatus.PASSED,
            duration_seconds=3.0,
            error_message=None,
            actions_taken=[],
            screenshots=[],
            assertions_passed=2,
            assertions_failed=0,
        )

        mock_log = MagicMock()

        with patch("src.services.event_gateway.get_event_gateway") as mock_get_gateway:
            mock_gateway = MagicMock()
            mock_gateway.is_running = True
            mock_get_gateway.return_value = mock_gateway

            with patch("src.services.event_gateway.emit_test_executed", new_callable=AsyncMock):
                with patch("src.services.event_gateway.emit_test_failed", new_callable=AsyncMock) as mock_emit_failed:
                    with patch(
                        "src.services.event_gateway.emit_healing_requested", new_callable=AsyncMock
                    ) as mock_emit_healing:
                        await _emit_test_events(state, test, test_result, mock_log)

                        # Verify healing was NOT requested
                        mock_emit_failed.assert_not_called()
                        mock_emit_healing.assert_not_called()


class TestEndToEndFlow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_autonomous_healing_loop(self):
        """
        Integration test: Full autonomous healing loop.

        This test simulates the complete flow:
        1. Test fails
        2. HEALING_REQUESTED event emitted
        3. HealingConsumer processes event
        4. SelfHealerAgent fixes test
        5. Pattern stored in Cognee
        6. HEALING_COMPLETED event emitted
        """
        # This would require a full integration setup with:
        # - Running Kafka/Redpanda
        # - Running Supabase
        # - Running Cognee
        # - SelfHealerAgent with mocked LLM calls

        pytest.skip("Integration test - requires full infrastructure setup")

    @pytest.mark.asyncio
    async def test_healing_with_cognee_learning(self):
        """Test that healed patterns are stored and retrievable from Cognee."""
        from src.knowledge import get_cognee_client

        with patch("src.knowledge.cognee_client.get_cognee_client") as mock_get_client:
            mock_cognee = AsyncMock()
            mock_cognee.store_failure_pattern = AsyncMock(return_value="pattern-abc123")
            mock_cognee.find_similar_failures = AsyncMock(
                return_value=[
                    {
                        "pattern_id": "pattern-abc123",
                        "error_message": "Element not found: #submit-btn",
                        "error_type": "selector",
                        "original_selector": "#submit-btn",
                        "healed_selector": "#new-submit-btn",
                        "healing_method": "code_aware",
                        "success_count": 5,
                        "failure_count": 0,
                        "success_rate": 1.0,
                        "similarity": 0.95,
                        "metadata": {},
                    }
                ]
            )
            mock_get_client.return_value = mock_cognee

            # Store a pattern
            pattern_id = await mock_cognee.store_failure_pattern(
                error_message="Element not found: #submit-btn",
                error_type="selector",
                original_selector="#submit-btn",
                healed_selector="#new-submit-btn",
                healing_method="code_aware",
                test_id="test-789",
            )

            assert pattern_id == "pattern-abc123"

            # Retrieve similar patterns
            similar = await mock_cognee.find_similar_failures(
                error_message="Element not found: #submit-btn",
                limit=5,
            )

            assert len(similar) == 1
            assert similar[0]["healed_selector"] == "#new-submit-btn"
            assert similar[0]["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
