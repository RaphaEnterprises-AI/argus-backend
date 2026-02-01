"""Tests for the SRE Agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSREAgentDataClasses:
    """Tests for SRE Agent data classes."""

    def test_correlated_incident_creation(self, mock_env_vars):
        """Test CorrelatedIncident dataclass."""
        from src.agents.sre_agent import CorrelatedIncident, IncidentSeverity

        incident = CorrelatedIncident(
            incident_id="INC-001",
            title="Database Connection Pool Exhausted",
            severity=IncidentSeverity.HIGH,
            signals=["metric_alert", "error_spike"],
            correlation_confidence=0.85,
            root_cause_hypothesis="Connection leak in auth service",
            suggested_runbook="db_connection_reset",
        )

        assert incident.incident_id == "INC-001"
        assert incident.severity == IncidentSeverity.HIGH
        assert incident.correlation_confidence == 0.85
        assert len(incident.signals) == 2

    def test_runbook_creation(self, mock_env_vars):
        """Test Runbook dataclass."""
        from src.agents.sre_agent import Runbook

        runbook = Runbook(
            runbook_id="RB-001",
            name="Restart Selenium Node",
            description="Restarts a stuck Selenium Grid node",
            steps=["Identify stuck node", "Drain connections", "Restart pod"],
            requires_approval=True,
            estimated_duration_minutes=5,
        )

        assert runbook.runbook_id == "RB-001"
        assert runbook.requires_approval is True
        assert len(runbook.steps) == 3

    def test_sre_metrics(self, mock_env_vars):
        """Test SREMetrics dataclass."""
        from src.agents.sre_agent import SREMetrics

        metrics = SREMetrics(
            mttr_minutes=15.5,
            incidents_correlated=10,
            runbooks_executed=5,
            auto_remediated=3,
            escalated=2,
        )

        assert metrics.mttr_minutes == 15.5
        assert metrics.incidents_correlated == 10
        assert metrics.auto_remediation_rate == 0.6  # 3/5


class TestSREAgentEnums:
    """Tests for SRE Agent enums."""

    def test_incident_severity_ordering(self, mock_env_vars):
        """Test IncidentSeverity enum values."""
        from src.agents.sre_agent import IncidentSeverity

        assert IncidentSeverity.CRITICAL.value == "critical"
        assert IncidentSeverity.HIGH.value == "high"
        assert IncidentSeverity.MEDIUM.value == "medium"
        assert IncidentSeverity.LOW.value == "low"

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
        assert hasattr(agent, 'execute_runbook')
        assert hasattr(agent, 'get_mttr_metrics')

    def test_sre_agent_capabilities(self, mock_env_vars):
        """Test SREAgent declares correct capabilities."""
        from src.agents.sre_agent import SREAgent, SRECapability

        agent = SREAgent()

        # SRE agent should have A2A capabilities
        assert hasattr(SRECapability, 'INCIDENT_CORRELATION')
        assert hasattr(SRECapability, 'RUNBOOK_EXECUTION')
        assert hasattr(SRECapability, 'MTTR_TRACKING')

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
    async def test_execute_runbook_dry_run(self, mock_env_vars):
        """Test execute_runbook in dry run mode."""
        from src.agents.sre_agent import SREAgent

        agent = SREAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"success": true, "steps_executed": [], "dry_run": true}'
            )

            result = await agent.execute_runbook(
                runbook_id="test_runbook",
                dry_run=True,
            )

            assert result is not None


class TestSREAgentIntegration:
    """Integration tests for SRE Agent (require mocking)."""

    @pytest.mark.asyncio
    async def test_full_incident_workflow(self, mock_env_vars):
        """Test full incident correlation -> remediation workflow."""
        from src.agents.sre_agent import SREAgent, IncidentSeverity

        agent = SREAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            # Mock correlation response
            mock_ai.return_value = MagicMock(
                content='''{
                    "incidents": [{
                        "incident_id": "INC-001",
                        "title": "Test Incident",
                        "severity": "high",
                        "correlation_confidence": 0.9,
                        "signals": ["alert1", "alert2"],
                        "root_cause_hypothesis": "Service overload",
                        "suggested_runbook": "scale_service"
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
