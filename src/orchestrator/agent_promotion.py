"""Agent Promotion Pipeline for Dev → Staging → Production workflows.

This module implements environment-aware agent lifecycle management with:
- Version tracking and semantic versioning
- Promotion workflows with approval gates
- Automatic rollback on failure
- A/B testing support for gradual rollouts

References:
- GitOps promotion patterns
- Kubernetes deployment strategies
- Feature flag systems (LaunchDarkly pattern)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
import hashlib
import json
import logging

from src.config import get_settings

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def order(self) -> int:
        """Get environment order for promotion validation."""
        return {
            Environment.DEVELOPMENT: 0,
            Environment.STAGING: 1,
            Environment.PRODUCTION: 2,
        }[self]

    def can_promote_to(self, target: "Environment") -> bool:
        """Check if promotion to target environment is valid."""
        return target.order == self.order + 1


class PromotionStatus(Enum):
    """Status of a promotion request."""

    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class RolloutStrategy(Enum):
    """Deployment rollout strategies."""

    IMMEDIATE = "immediate"  # All traffic immediately
    CANARY = "canary"  # Gradual percentage increase
    BLUE_GREEN = "blue_green"  # Switch all at once with instant rollback
    A_B_TEST = "a_b_test"  # Split traffic for comparison


@dataclass
class AgentVersion:
    """Represents a specific version of an agent."""

    agent_type: str
    version: str  # Semantic version (e.g., "1.2.3")
    config_hash: str  # Hash of agent configuration
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"

    # Version metadata
    changelog: str = ""
    breaking_changes: bool = False

    # Performance baseline
    baseline_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate version format."""
        parts = self.version.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid semantic version: {self.version}")

    @property
    def major(self) -> int:
        return int(self.version.split(".")[0])

    @property
    def minor(self) -> int:
        return int(self.version.split(".")[1])

    @property
    def patch(self) -> int:
        return int(self.version.split(".")[2])

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "agent_type": self.agent_type,
            "version": self.version,
            "config_hash": self.config_hash,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "changelog": self.changelog,
            "breaking_changes": self.breaking_changes,
            "baseline_metrics": self.baseline_metrics,
        }


@dataclass
class EnvironmentDeployment:
    """Tracks which agent version is deployed in which environment."""

    agent_type: str
    environment: Environment
    version: AgentVersion
    deployed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_by: str = "system"

    # Rollout configuration
    rollout_strategy: RolloutStrategy = RolloutStrategy.IMMEDIATE
    rollout_percentage: float = 100.0  # For canary deployments

    # Health status
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    error_rate: float = 0.0

    # Previous version for rollback
    previous_version: Optional[AgentVersion] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "agent_type": self.agent_type,
            "environment": self.environment.value,
            "version": self.version.to_dict(),
            "deployed_at": self.deployed_at.isoformat(),
            "deployed_by": self.deployed_by,
            "rollout_strategy": self.rollout_strategy.value,
            "rollout_percentage": self.rollout_percentage,
            "is_healthy": self.is_healthy,
            "error_rate": self.error_rate,
            "previous_version": self.previous_version.to_dict() if self.previous_version else None,
        }


@dataclass
class PromotionRequest:
    """Request to promote an agent to a new environment."""

    request_id: str
    agent_type: str
    version: AgentVersion
    source_environment: Environment
    target_environment: Environment

    # Request metadata
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requested_by: str = "system"
    reason: str = ""

    # Approval workflow
    status: PromotionStatus = PromotionStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Execution details
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Rollout configuration
    rollout_strategy: RolloutStrategy = RolloutStrategy.IMMEDIATE

    # Gate results
    gate_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "request_id": self.request_id,
            "agent_type": self.agent_type,
            "version": self.version.to_dict(),
            "source_environment": self.source_environment.value,
            "target_environment": self.target_environment.value,
            "requested_at": self.requested_at.isoformat(),
            "requested_by": self.requested_by,
            "reason": self.reason,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "rollout_strategy": self.rollout_strategy.value,
            "gate_results": self.gate_results,
        }


class AgentPromotionPipeline:
    """Manages agent promotion across environments.

    Features:
    - Environment-aware agent registry
    - Promotion workflow with approval gates
    - Automatic rollback on failure
    - Canary and A/B testing support
    """

    def __init__(self):
        self.settings = get_settings()

        # In-memory storage (replace with database in production)
        self._versions: dict[str, list[AgentVersion]] = {}
        self._deployments: dict[tuple[str, Environment], EnvironmentDeployment] = {}
        self._promotion_requests: dict[str, PromotionRequest] = {}

        # Promotion gates (callbacks that must pass before promotion)
        self._promotion_gates: list[Callable] = []

        # Event handlers
        self._on_promotion_complete: list[Callable] = []
        self._on_rollback: list[Callable] = []

    def register_version(
        self,
        agent_type: str,
        version: str,
        config: dict,
        changelog: str = "",
        breaking_changes: bool = False,
        created_by: str = "system",
    ) -> AgentVersion:
        """Register a new agent version.

        Args:
            agent_type: Type of agent (e.g., "CodeAnalyzerAgent")
            version: Semantic version string
            config: Agent configuration dictionary
            changelog: Description of changes
            breaking_changes: Whether this version has breaking changes
            created_by: User or system that created this version

        Returns:
            AgentVersion object
        """
        # Generate config hash
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]

        agent_version = AgentVersion(
            agent_type=agent_type,
            version=version,
            config_hash=config_hash,
            changelog=changelog,
            breaking_changes=breaking_changes,
            created_by=created_by,
        )

        # Store version
        if agent_type not in self._versions:
            self._versions[agent_type] = []
        self._versions[agent_type].append(agent_version)

        logger.info(
            f"Registered agent version: {agent_type} v{version} (hash: {config_hash})"
        )

        return agent_version

    def get_deployed_version(
        self,
        agent_type: str,
        environment: Environment,
    ) -> Optional[EnvironmentDeployment]:
        """Get the currently deployed version for an agent in an environment."""
        return self._deployments.get((agent_type, environment))

    def get_all_versions(self, agent_type: str) -> list[AgentVersion]:
        """Get all registered versions for an agent type."""
        return self._versions.get(agent_type, [])

    def create_promotion_request(
        self,
        agent_type: str,
        version: str,
        target_environment: Environment,
        requested_by: str,
        reason: str = "",
        rollout_strategy: RolloutStrategy = RolloutStrategy.IMMEDIATE,
    ) -> PromotionRequest:
        """Create a promotion request.

        Args:
            agent_type: Type of agent to promote
            version: Version string to promote
            target_environment: Target environment
            requested_by: User requesting promotion
            reason: Reason for promotion
            rollout_strategy: How to roll out the new version

        Returns:
            PromotionRequest object
        """
        # Find the version
        versions = self._versions.get(agent_type, [])
        agent_version = next(
            (v for v in versions if v.version == version),
            None
        )
        if not agent_version:
            raise ValueError(f"Version {version} not found for agent {agent_type}")

        # Determine source environment
        source_env = Environment(
            list(Environment)[target_environment.order - 1].value
            if target_environment.order > 0
            else Environment.DEVELOPMENT.value
        )

        # Validate promotion path
        if not source_env.can_promote_to(target_environment):
            raise ValueError(
                f"Cannot promote from {source_env.value} to {target_environment.value}"
            )

        # Generate request ID
        request_id = f"promo_{agent_type}_{version}_{target_environment.value}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        request = PromotionRequest(
            request_id=request_id,
            agent_type=agent_type,
            version=agent_version,
            source_environment=source_env,
            target_environment=target_environment,
            requested_by=requested_by,
            reason=reason,
            rollout_strategy=rollout_strategy,
        )

        self._promotion_requests[request_id] = request

        logger.info(
            f"Created promotion request: {request_id} "
            f"({agent_type} v{version} → {target_environment.value})"
        )

        return request

    def add_promotion_gate(self, gate: Callable) -> None:
        """Add a promotion gate that must pass before promotion.

        Gate function signature:
            async def gate(request: PromotionRequest) -> tuple[bool, str]
            Returns (passed, message)
        """
        self._promotion_gates.append(gate)

    async def run_promotion_gates(
        self,
        request: PromotionRequest,
    ) -> tuple[bool, dict]:
        """Run all promotion gates.

        Returns:
            Tuple of (all_passed, gate_results)
        """
        results = {}
        all_passed = True

        for i, gate in enumerate(self._promotion_gates):
            gate_name = getattr(gate, "__name__", f"gate_{i}")
            try:
                passed, message = await gate(request)
                results[gate_name] = {
                    "passed": passed,
                    "message": message,
                }
                if not passed:
                    all_passed = False
                    logger.warning(f"Promotion gate failed: {gate_name} - {message}")
            except Exception as e:
                results[gate_name] = {
                    "passed": False,
                    "message": f"Gate error: {str(e)}",
                }
                all_passed = False
                logger.error(f"Promotion gate error: {gate_name} - {e}")

        request.gate_results = results
        return all_passed, results

    async def approve_promotion(
        self,
        request_id: str,
        approved_by: str,
    ) -> PromotionRequest:
        """Approve a promotion request.

        Args:
            request_id: ID of the promotion request
            approved_by: User approving the promotion

        Returns:
            Updated PromotionRequest
        """
        request = self._promotion_requests.get(request_id)
        if not request:
            raise ValueError(f"Promotion request not found: {request_id}")

        if request.status != PromotionStatus.PENDING:
            raise ValueError(f"Cannot approve request in status: {request.status.value}")

        request.status = PromotionStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now(timezone.utc)

        logger.info(f"Promotion approved: {request_id} by {approved_by}")

        return request

    async def execute_promotion(
        self,
        request_id: str,
    ) -> PromotionRequest:
        """Execute an approved promotion.

        Args:
            request_id: ID of the promotion request

        Returns:
            Updated PromotionRequest
        """
        request = self._promotion_requests.get(request_id)
        if not request:
            raise ValueError(f"Promotion request not found: {request_id}")

        if request.status != PromotionStatus.APPROVED:
            raise ValueError(f"Cannot execute request in status: {request.status.value}")

        request.status = PromotionStatus.IN_PROGRESS
        request.started_at = datetime.now(timezone.utc)

        try:
            # Run promotion gates
            gates_passed, gate_results = await self.run_promotion_gates(request)

            if not gates_passed:
                request.status = PromotionStatus.FAILED
                request.error_message = f"Promotion gates failed: {gate_results}"
                logger.error(f"Promotion failed gates: {request_id}")
                return request

            # Get previous deployment for rollback
            previous = self._deployments.get(
                (request.agent_type, request.target_environment)
            )
            previous_version = previous.version if previous else None

            # Create new deployment
            deployment = EnvironmentDeployment(
                agent_type=request.agent_type,
                environment=request.target_environment,
                version=request.version,
                deployed_by=request.approved_by or "system",
                rollout_strategy=request.rollout_strategy,
                previous_version=previous_version,
            )

            # Update deployment registry
            self._deployments[(request.agent_type, request.target_environment)] = deployment

            request.status = PromotionStatus.COMPLETED
            request.completed_at = datetime.now(timezone.utc)

            logger.info(
                f"Promotion completed: {request_id} "
                f"({request.agent_type} v{request.version.version} → {request.target_environment.value})"
            )

            # Notify handlers
            for handler in self._on_promotion_complete:
                try:
                    await handler(request, deployment)
                except Exception as e:
                    logger.error(f"Promotion handler error: {e}")

        except Exception as e:
            request.status = PromotionStatus.FAILED
            request.error_message = str(e)
            request.completed_at = datetime.now(timezone.utc)
            logger.error(f"Promotion failed: {request_id} - {e}")

        return request

    async def rollback(
        self,
        agent_type: str,
        environment: Environment,
        reason: str,
        rolled_back_by: str,
    ) -> Optional[EnvironmentDeployment]:
        """Rollback an agent to its previous version.

        Args:
            agent_type: Type of agent to rollback
            environment: Environment to rollback in
            reason: Reason for rollback
            rolled_back_by: User initiating rollback

        Returns:
            New deployment with previous version, or None if no rollback available
        """
        current = self._deployments.get((agent_type, environment))
        if not current or not current.previous_version:
            logger.warning(f"No rollback available for {agent_type} in {environment.value}")
            return None

        # Create rollback deployment
        deployment = EnvironmentDeployment(
            agent_type=agent_type,
            environment=environment,
            version=current.previous_version,
            deployed_by=rolled_back_by,
            rollout_strategy=RolloutStrategy.IMMEDIATE,
            previous_version=current.version,
        )

        self._deployments[(agent_type, environment)] = deployment

        logger.warning(
            f"Rolled back {agent_type} in {environment.value}: "
            f"v{current.version.version} → v{current.previous_version.version} "
            f"(reason: {reason})"
        )

        # Notify handlers
        for handler in self._on_rollback:
            try:
                await handler(agent_type, environment, current, deployment, reason)
            except Exception as e:
                logger.error(f"Rollback handler error: {e}")

        return deployment

    async def update_rollout_percentage(
        self,
        agent_type: str,
        environment: Environment,
        percentage: float,
    ) -> EnvironmentDeployment:
        """Update canary rollout percentage.

        Args:
            agent_type: Type of agent
            environment: Environment
            percentage: New rollout percentage (0-100)

        Returns:
            Updated deployment
        """
        deployment = self._deployments.get((agent_type, environment))
        if not deployment:
            raise ValueError(f"No deployment found for {agent_type} in {environment.value}")

        if deployment.rollout_strategy != RolloutStrategy.CANARY:
            raise ValueError("Can only update percentage for canary deployments")

        deployment.rollout_percentage = min(max(percentage, 0), 100)
        deployment.last_health_check = datetime.now(timezone.utc)

        logger.info(
            f"Updated rollout percentage: {agent_type} in {environment.value} → {percentage}%"
        )

        return deployment

    def should_use_new_version(
        self,
        agent_type: str,
        environment: Environment,
        request_id: Optional[str] = None,
    ) -> bool:
        """Determine if a request should use the new version (for canary/A-B).

        Args:
            agent_type: Type of agent
            environment: Environment
            request_id: Optional request ID for consistent routing

        Returns:
            True if new version should be used
        """
        deployment = self._deployments.get((agent_type, environment))
        if not deployment:
            return True  # No deployment, use default

        if deployment.rollout_strategy == RolloutStrategy.IMMEDIATE:
            return True

        if deployment.rollout_strategy in (RolloutStrategy.CANARY, RolloutStrategy.A_B_TEST):
            # Use request_id for consistent routing, or random if not provided
            if request_id:
                # Hash-based routing for consistency
                hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                return (hash_val % 100) < deployment.rollout_percentage
            else:
                import random
                return random.random() * 100 < deployment.rollout_percentage

        return True

    def get_promotion_history(
        self,
        agent_type: Optional[str] = None,
        environment: Optional[Environment] = None,
        limit: int = 50,
    ) -> list[PromotionRequest]:
        """Get promotion history.

        Args:
            agent_type: Filter by agent type
            environment: Filter by target environment
            limit: Maximum number of results

        Returns:
            List of promotion requests
        """
        requests = list(self._promotion_requests.values())

        if agent_type:
            requests = [r for r in requests if r.agent_type == agent_type]

        if environment:
            requests = [r for r in requests if r.target_environment == environment]

        # Sort by requested_at descending
        requests.sort(key=lambda r: r.requested_at, reverse=True)

        return requests[:limit]


# Singleton instance
_promotion_pipeline: Optional[AgentPromotionPipeline] = None


def get_promotion_pipeline() -> AgentPromotionPipeline:
    """Get the singleton promotion pipeline instance."""
    global _promotion_pipeline
    if _promotion_pipeline is None:
        _promotion_pipeline = AgentPromotionPipeline()
    return _promotion_pipeline


# Default promotion gates
async def require_staging_deployment_gate(request: PromotionRequest) -> tuple[bool, str]:
    """Gate: Production promotions require successful staging deployment."""
    if request.target_environment != Environment.PRODUCTION:
        return True, "Not a production promotion"

    pipeline = get_promotion_pipeline()
    staging_deployment = pipeline.get_deployed_version(
        request.agent_type,
        Environment.STAGING,
    )

    if not staging_deployment:
        return False, "Agent must be deployed to staging before production"

    if staging_deployment.version.version != request.version.version:
        return False, f"Staging has v{staging_deployment.version.version}, promoting v{request.version.version}"

    if not staging_deployment.is_healthy:
        return False, "Staging deployment is not healthy"

    return True, "Staging deployment verified"


async def require_minimum_staging_time_gate(request: PromotionRequest) -> tuple[bool, str]:
    """Gate: Require minimum time in staging before production."""
    if request.target_environment != Environment.PRODUCTION:
        return True, "Not a production promotion"

    pipeline = get_promotion_pipeline()
    staging_deployment = pipeline.get_deployed_version(
        request.agent_type,
        Environment.STAGING,
    )

    if not staging_deployment:
        return False, "No staging deployment found"

    # Require 1 hour in staging (configurable)
    min_staging_hours = 1
    staging_duration = datetime.now(timezone.utc) - staging_deployment.deployed_at

    if staging_duration.total_seconds() < min_staging_hours * 3600:
        remaining = min_staging_hours * 3600 - staging_duration.total_seconds()
        return False, f"Minimum staging time not met. {remaining/60:.0f} minutes remaining."

    return True, f"Staging duration: {staging_duration.total_seconds()/3600:.1f} hours"


async def no_breaking_changes_without_approval_gate(request: PromotionRequest) -> tuple[bool, str]:
    """Gate: Breaking changes require explicit approval."""
    if request.target_environment == Environment.DEVELOPMENT:
        return True, "Development allows breaking changes"

    if request.version.breaking_changes:
        if request.target_environment == Environment.PRODUCTION:
            return False, "Breaking changes require manual review for production"
        return True, "Breaking changes allowed in staging with caution"

    return True, "No breaking changes"
