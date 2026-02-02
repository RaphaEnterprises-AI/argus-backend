"""Tests for the SRE Agent module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSREAgentDataClasses:
    """Tests for SRE Agent data classes."""

    def test_correlated_incident_creation(self, mock_env_vars):
        """Test CorrelatedIncident dataclass."""
        from src.agents.sre_agent import CorrelatedIncident, IncidentSeverity, RemediationStatus

        incident = CorrelatedIncident(
            id="INC-001",
            title="Database Connection Pool Exhausted",
            severity=IncidentSeverity.SEV2,
            detected_at=datetime.now(),
            primary_signal="metric_alert",
            related_signals=["error_spike"],
            root_cause_hypothesis="Connection leak in auth service",
            root_cause_confidence=0.85,
            affected_services=["auth-service"],
            blast_radius="medium",
            recent_changes=[],
            similar_past_incidents=[],
            suggested_actions=["restart_service"],
            remediation_status=RemediationStatus.PENDING,
        )

        assert incident.id == "INC-001"
        assert incident.severity == IncidentSeverity.SEV2
        assert incident.root_cause_confidence == 0.85

    def test_runbook_creation(self, mock_env_vars):
        """Test Runbook dataclass."""
        from src.agents.sre_agent import Runbook, IncidentSeverity

        runbook = Runbook(
            id="RB-001",
            name="Restart Selenium Node",
            description="Restarts a stuck Selenium Grid node",
            trigger_conditions=["node_unresponsive"],
            steps=["Identify stuck node", "Drain connections", "Restart pod"],
            requires_approval=True,
            max_auto_execute_severity=IncidentSeverity.SEV3,
        )

        assert runbook.id == "RB-001"
        assert runbook.requires_approval is True
        assert len(runbook.steps) == 3

    def test_sre_metrics(self, mock_env_vars):
        """Test SREMetrics dataclass."""
        from src.agents.sre_agent import SREMetrics

        metrics = SREMetrics(
            mttr_minutes=15.5,
            mttr_trend=-0.1,
            incidents_last_24h=10,
            incidents_last_7d=50,
            auto_resolved_percentage=60.0,
            uptime_percentage=99.9,
            error_budget_remaining=0.5,
        )

        assert metrics.mttr_minutes == 15.5
        assert metrics.incidents_last_24h == 10
        assert metrics.auto_resolved_percentage == 60.0


class TestSREAgentEnums:
    """Tests for SRE Agent enums."""

    def test_incident_severity_ordering(self, mock_env_vars):
        """Test IncidentSeverity enum values."""
        from src.agents.sre_agent import IncidentSeverity

        assert IncidentSeverity.SEV1.value == "sev1"
        assert IncidentSeverity.SEV2.value == "sev2"
        assert IncidentSeverity.SEV3.value == "sev3"
        assert IncidentSeverity.SEV4.value == "sev4"
        assert IncidentSeverity.SEV5.value == "sev5"

    def test_remediation_status(self, mock_env_vars):
        """Test RemediationStatus enum values."""
        from src.agents.sre_agent import RemediationStatus

        assert RemediationStatus.PENDING.value == "pending"
        assert RemediationStatus.IN_PROGRESS.value == "in_progress"
        assert RemediationStatus.COMPLETED.value == "completed"
        assert RemediationStatus.FAILED.value == "failed"
        assert RemediationStatus.ROLLED_BACK.value == "rolled_back"


class TestSREAgent:
    """Tests for SREAgent class."""

    def test_sre_agent_initialization(self, mock_env_vars):
        """Test SREAgent initialization."""
        from src.agents.sre_agent import SREAgent

        agent = SREAgent()

        assert agent is not None
        assert hasattr(agent, 'correlate_signals')
        assert hasattr(agent, 'execute_remediation')
        assert hasattr(agent, 'get_metrics')

    def test_sre_agent_capabilities(self, mock_env_vars):
        """Test SREAgent declares correct capabilities."""
        from src.agents.sre_agent import SREAgent

        agent = SREAgent()

        # SRE agent should have CAPABILITIES defined
        assert hasattr(agent, 'CAPABILITIES')
        assert isinstance(agent.CAPABILITIES, list)

    @pytest.mark.asyncio
    async def test_correlate_signals_empty(self, mock_env_vars):
        """Test correlate_signals with empty signals."""
        from src.agents.sre_agent import SREAgent

        agent = SREAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"incidents": [], "confidence": 0.0}'
            )

            result = await agent.correlate_signals(
                signals=[],
                time_window_minutes=30,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_execute_runbook_with_approval(self, mock_env_vars):
        """Test execute_remediation with approval."""
        from src.agents.sre_agent import SREAgent

        agent = SREAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"success": true, "steps_executed": []}'
            )

            result = await agent.execute_remediation(
                incident_id="INC-001",
                runbook_id="test_runbook",
                approved_by="admin@example.com",
            )

            assert result is not None


class TestSREAgentIntegration:
    """Integration tests for SRE Agent (require mocking)."""

    @pytest.mark.asyncio
    async def test_full_incident_workflow(self, mock_env_vars):
        """Test full incident correlation -> remediation workflow."""
        from src.agents.sre_agent import SREAgent

        agent = SREAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            # Mock correlation response
            mock_ai.return_value = MagicMock(
                content='''{
                    "incidents": [{
                        "id": "INC-001",
                        "title": "Test Incident",
                        "severity": "sev2",
                        "root_cause_confidence": 0.9,
                        "primary_signal": "alert1",
                        "related_signals": ["alert2"],
                        "root_cause_hypothesis": "Service overload",
                        "suggested_actions": ["scale_service"]
                    }],
                    "confidence": 0.9
                }'''
            )

            result = await agent.correlate_signals(
                signals=[
                    {"type": "metric", "name": "cpu_high"},
                    {"type": "alert", "name": "service_timeout"},
                ],
                time_window_minutes=15,
            )

            assert result is not None
            mock_ai.assert_called()
