"""Tests for Agent Promotion Pipeline."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAgentVersionDataClass:
    """Tests for AgentVersion dataclass."""

    def test_agent_version_creation(self, mock_env_vars):
        """Test AgentVersion creation with valid data."""
        from src.orchestrator.agent_promotion import AgentVersion

        version = AgentVersion(
            agent_type="CodeAnalyzerAgent",
            version="1.2.3",
            config_hash="abc123def456",
            changelog="Added new feature",
            breaking_changes=False,
        )

        assert version.agent_type == "CodeAnalyzerAgent"
        assert version.version == "1.2.3"
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_agent_version_invalid_version(self, mock_env_vars):
        """Test AgentVersion with invalid version format."""
        from src.orchestrator.agent_promotion import AgentVersion

        with pytest.raises(ValueError, match="Invalid semantic version"):
            AgentVersion(
                agent_type="TestAgent",
                version="1.2",  # Missing patch version
                config_hash="abc123",
            )

    def test_agent_version_to_dict(self, mock_env_vars):
        """Test AgentVersion serialization."""
        from src.orchestrator.agent_promotion import AgentVersion

        version = AgentVersion(
            agent_type="TestAgent",
            version="2.0.0",
            config_hash="hash123",
            breaking_changes=True,
        )

        data = version.to_dict()

        assert data["agent_type"] == "TestAgent"
        assert data["version"] == "2.0.0"
        assert data["breaking_changes"] is True


class TestEnvironmentEnum:
    """Tests for Environment enum."""

    def test_environment_order(self, mock_env_vars):
        """Test environment ordering."""
        from src.orchestrator.agent_promotion import Environment

        assert Environment.DEVELOPMENT.order == 0
        assert Environment.STAGING.order == 1
        assert Environment.PRODUCTION.order == 2

    def test_can_promote_to(self, mock_env_vars):
        """Test promotion path validation."""
        from src.orchestrator.agent_promotion import Environment

        # Valid promotions
        assert Environment.DEVELOPMENT.can_promote_to(Environment.STAGING) is True
        assert Environment.STAGING.can_promote_to(Environment.PRODUCTION) is True

        # Invalid promotions
        assert Environment.DEVELOPMENT.can_promote_to(Environment.PRODUCTION) is False
        assert Environment.PRODUCTION.can_promote_to(Environment.STAGING) is False


class TestPromotionStatusEnum:
    """Tests for PromotionStatus enum."""

    def test_promotion_status_values(self, mock_env_vars):
        """Test PromotionStatus enum values."""
        from src.orchestrator.agent_promotion import PromotionStatus

        assert PromotionStatus.PENDING.value == "pending"
        assert PromotionStatus.APPROVED.value == "approved"
        assert PromotionStatus.IN_PROGRESS.value == "in_progress"
        assert PromotionStatus.COMPLETED.value == "completed"
        assert PromotionStatus.FAILED.value == "failed"
        assert PromotionStatus.ROLLED_BACK.value == "rolled_back"


class TestAgentPromotionPipeline:
    """Tests for AgentPromotionPipeline class."""

    def test_pipeline_initialization(self, mock_env_vars):
        """Test pipeline initialization."""
        from src.orchestrator.agent_promotion import AgentPromotionPipeline

        pipeline = AgentPromotionPipeline()

        assert pipeline is not None
        assert hasattr(pipeline, 'register_version')
        assert hasattr(pipeline, 'create_promotion_request')
        assert hasattr(pipeline, 'execute_promotion')
        assert hasattr(pipeline, 'rollback')

    def test_register_version(self, mock_env_vars):
        """Test registering a new agent version."""
        from src.orchestrator.agent_promotion import AgentPromotionPipeline

        pipeline = AgentPromotionPipeline()

        version = pipeline.register_version(
            agent_type="TestAgent",
            version="1.0.0",
            config={"model": "claude-sonnet-4-5"},
            changelog="Initial release",
        )

        assert version.agent_type == "TestAgent"
        assert version.version == "1.0.0"
        assert version.config_hash is not None

    def test_get_all_versions(self, mock_env_vars):
        """Test getting all versions for an agent."""
        from src.orchestrator.agent_promotion import AgentPromotionPipeline

        pipeline = AgentPromotionPipeline()

        pipeline.register_version("TestAgent", "1.0.0", {})
        pipeline.register_version("TestAgent", "1.1.0", {})
        pipeline.register_version("TestAgent", "2.0.0", {})

        versions = pipeline.get_all_versions("TestAgent")

        assert len(versions) == 3
        assert versions[0].version == "1.0.0"
        assert versions[2].version == "2.0.0"

    def test_create_promotion_request(self, mock_env_vars):
        """Test creating a promotion request."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            Environment,
            PromotionStatus,
        )

        pipeline = AgentPromotionPipeline()

        # Register version first
        pipeline.register_version("TestAgent", "1.0.0", {"key": "value"})

        # Create promotion request
        request = pipeline.create_promotion_request(
            agent_type="TestAgent",
            version="1.0.0",
            target_environment=Environment.STAGING,
            requested_by="test_user",
            reason="Testing promotion",
        )

        assert request.agent_type == "TestAgent"
        assert request.target_environment == Environment.STAGING
        assert request.status == PromotionStatus.PENDING
        assert "promo_" in request.request_id

    @pytest.mark.asyncio
    async def test_approve_promotion(self, mock_env_vars):
        """Test approving a promotion request."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            Environment,
            PromotionStatus,
        )

        pipeline = AgentPromotionPipeline()
        pipeline.register_version("TestAgent", "1.0.0", {})

        request = pipeline.create_promotion_request(
            agent_type="TestAgent",
            version="1.0.0",
            target_environment=Environment.STAGING,
            requested_by="requester",
        )

        approved = await pipeline.approve_promotion(
            request.request_id,
            approved_by="approver",
        )

        assert approved.status == PromotionStatus.APPROVED
        assert approved.approved_by == "approver"
        assert approved.approved_at is not None

    @pytest.mark.asyncio
    async def test_execute_promotion(self, mock_env_vars):
        """Test executing an approved promotion."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            Environment,
            PromotionStatus,
        )

        pipeline = AgentPromotionPipeline()
        pipeline.register_version("TestAgent", "1.0.0", {})

        request = pipeline.create_promotion_request(
            agent_type="TestAgent",
            version="1.0.0",
            target_environment=Environment.STAGING,
            requested_by="user",
        )

        await pipeline.approve_promotion(request.request_id, "approver")
        result = await pipeline.execute_promotion(request.request_id)

        assert result.status == PromotionStatus.COMPLETED
        assert result.completed_at is not None

        # Verify deployment
        deployment = pipeline.get_deployed_version("TestAgent", Environment.STAGING)
        assert deployment is not None
        assert deployment.version.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_rollback(self, mock_env_vars):
        """Test rolling back a deployment."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            Environment,
        )

        pipeline = AgentPromotionPipeline()

        # Deploy v1.0.0
        pipeline.register_version("TestAgent", "1.0.0", {})
        request1 = pipeline.create_promotion_request(
            "TestAgent", "1.0.0", Environment.STAGING, "user"
        )
        await pipeline.approve_promotion(request1.request_id, "approver")
        await pipeline.execute_promotion(request1.request_id)

        # Deploy v1.1.0
        pipeline.register_version("TestAgent", "1.1.0", {})
        request2 = pipeline.create_promotion_request(
            "TestAgent", "1.1.0", Environment.STAGING, "user"
        )
        await pipeline.approve_promotion(request2.request_id, "approver")
        await pipeline.execute_promotion(request2.request_id)

        # Rollback
        rollback_deployment = await pipeline.rollback(
            "TestAgent",
            Environment.STAGING,
            reason="Bug found",
            rolled_back_by="admin",
        )

        assert rollback_deployment is not None
        assert rollback_deployment.version.version == "1.0.0"


class TestPromotionGates:
    """Tests for promotion gates."""

    @pytest.mark.asyncio
    async def test_require_staging_deployment_gate(self, mock_env_vars):
        """Test staging deployment requirement gate."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            AgentVersion,
            Environment,
            PromotionRequest,
            require_staging_deployment_gate,
        )

        # Create a production promotion request without staging deployment
        version = AgentVersion(
            agent_type="TestAgent",
            version="1.0.0",
            config_hash="hash",
        )

        request = PromotionRequest(
            request_id="test_request",
            agent_type="TestAgent",
            version=version,
            source_environment=Environment.STAGING,
            target_environment=Environment.PRODUCTION,
            requested_by="user",
        )

        # Should fail - no staging deployment
        passed, message = await require_staging_deployment_gate(request)

        assert passed is False
        assert "staging" in message.lower()

    @pytest.mark.asyncio
    async def test_no_breaking_changes_gate(self, mock_env_vars):
        """Test breaking changes gate."""
        from src.orchestrator.agent_promotion import (
            AgentVersion,
            Environment,
            PromotionRequest,
            no_breaking_changes_without_approval_gate,
        )

        version = AgentVersion(
            agent_type="TestAgent",
            version="2.0.0",
            config_hash="hash",
            breaking_changes=True,
        )

        request = PromotionRequest(
            request_id="test_request",
            agent_type="TestAgent",
            version=version,
            source_environment=Environment.STAGING,
            target_environment=Environment.PRODUCTION,
            requested_by="user",
        )

        passed, message = await no_breaking_changes_without_approval_gate(request)

        assert passed is False
        assert "breaking" in message.lower()


class TestRolloutStrategies:
    """Tests for rollout strategies."""

    def test_should_use_new_version_immediate(self, mock_env_vars):
        """Test immediate rollout strategy."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            AgentVersion,
            Environment,
            EnvironmentDeployment,
            RolloutStrategy,
        )

        pipeline = AgentPromotionPipeline()

        version = AgentVersion("TestAgent", "1.0.0", "hash")
        deployment = EnvironmentDeployment(
            agent_type="TestAgent",
            environment=Environment.STAGING,
            version=version,
            rollout_strategy=RolloutStrategy.IMMEDIATE,
        )

        pipeline._deployments[("TestAgent", Environment.STAGING)] = deployment

        # Should always use new version
        assert pipeline.should_use_new_version("TestAgent", Environment.STAGING) is True

    def test_should_use_new_version_canary(self, mock_env_vars):
        """Test canary rollout strategy."""
        from src.orchestrator.agent_promotion import (
            AgentPromotionPipeline,
            AgentVersion,
            Environment,
            EnvironmentDeployment,
            RolloutStrategy,
        )

        pipeline = AgentPromotionPipeline()

        version = AgentVersion("TestAgent", "1.0.0", "hash")
        deployment = EnvironmentDeployment(
            agent_type="TestAgent",
            environment=Environment.STAGING,
            version=version,
            rollout_strategy=RolloutStrategy.CANARY,
            rollout_percentage=50.0,
        )

        pipeline._deployments[("TestAgent", Environment.STAGING)] = deployment

        # With request_id, should be deterministic
        result1 = pipeline.should_use_new_version("TestAgent", Environment.STAGING, "req_123")
        result2 = pipeline.should_use_new_version("TestAgent", Environment.STAGING, "req_123")

        # Same request_id should give same result
        assert result1 == result2


class TestGetPromotionPipelineSingleton:
    """Tests for get_promotion_pipeline singleton."""

    def test_returns_singleton(self, mock_env_vars):
        """Test that get_promotion_pipeline returns singleton."""
        from src.orchestrator.agent_promotion import get_promotion_pipeline

        # Reset singleton
        import src.orchestrator.agent_promotion as module
        module._promotion_pipeline = None

        pipeline1 = get_promotion_pipeline()
        pipeline2 = get_promotion_pipeline()

        assert pipeline1 is pipeline2
