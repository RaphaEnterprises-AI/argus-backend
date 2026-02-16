"""Agent Registry for Agent-to-Agent (A2A) communication.

This module provides a centralized registry for discovering and managing agents
in the Argus testing system. It enables:
- Agent registration with capabilities
- Discovery of agents by capability
- Health monitoring via heartbeats
- Automatic cleanup of unhealthy agents
- Persistent storage in Supabase for server restarts

RAP-228: Create Agent Registry for A2A communication.

Example usage:
    ```python
    from src.orchestrator.agent_registry import get_agent_registry, Capability

    # Register an agent
    registry = get_agent_registry()
    agent_id = registry.register(
        agent_type="code_analyzer",
        capabilities=[Capability.CODE_ANALYSIS, Capability.GIT_BLAME],
        metadata={"version": "1.0.0"}
    )

    # Discover agents by capability
    agents = registry.discover(Capability.BROWSER_AUTOMATION)
    for agent in agents:
        print(f"Found agent: {agent.agent_id} with status {agent.status}")

    # Update heartbeat to keep agent healthy
    registry.update_heartbeat(agent_id)
    ```
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from src.services.supabase_client import SupabaseClient

logger = structlog.get_logger(__name__)


class Capability(str, Enum):
    """Agent capabilities for discovery and routing.

    These capabilities define what an agent can do, allowing other agents
    and the orchestrator to discover and delegate tasks appropriately.
    """

    # Code Analysis Capabilities
    CODE_ANALYSIS = "code_analysis"
    GIT_BLAME = "git_blame"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    SECURITY_SCAN = "security_scan"
    TEST_IMPACT_ANALYSIS = "test_impact_analysis"
    CODE_COVERAGE = "code_coverage"

    # UI Testing Capabilities
    BROWSER_AUTOMATION = "browser_automation"
    COMPUTER_USE = "computer_use"
    VISUAL_TESTING = "visual_testing"
    ACCESSIBILITY_CHECK = "accessibility_check"
    SCREENSHOT_CAPTURE = "screenshot_capture"
    VIDEO_RECORDING = "video_recording"

    # API Testing Capabilities
    API_TESTING = "api_testing"
    SCHEMA_VALIDATION = "schema_validation"
    GRAPHQL_TESTING = "graphql_testing"
    WEBSOCKET_TESTING = "websocket_testing"

    # Self-Healing Capabilities
    SELECTOR_FIX = "selector_fix"
    ASSERTION_FIX = "assertion_fix"
    FLAKY_DETECTION = "flaky_detection"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    AUTO_HEALING = "auto_healing"

    # Test Planning Capabilities
    TEST_PLANNING = "test_planning"
    TEST_PRIORITIZATION = "test_prioritization"
    RISK_ASSESSMENT = "risk_assessment"
    TEST_GENERATION = "test_generation"

    # Reporting Capabilities
    REPORT_GENERATION = "report_generation"
    SLACK_INTEGRATION = "slack_integration"
    GITHUB_INTEGRATION = "github_integration"
    JIRA_INTEGRATION = "jira_integration"

    # Database Capabilities
    DB_TESTING = "db_testing"
    DATA_VALIDATION = "data_validation"
    MIGRATION_TESTING = "migration_testing"

    # Performance Capabilities
    PERFORMANCE_ANALYSIS = "performance_analysis"
    LOAD_TESTING = "load_testing"
    LATENCY_MONITORING = "latency_monitoring"

    # Discovery Capabilities
    AUTO_DISCOVERY = "auto_discovery"
    FLOW_DISCOVERY = "flow_discovery"
    CRAWLING = "crawling"

    # AI Capabilities
    NLP_UNDERSTANDING = "nlp_understanding"
    VISUAL_AI = "visual_ai"
    COGNITIVE_MODELING = "cognitive_modeling"

    # Penetration Testing Capabilities
    RECON = "recon"
    NETWORK_SCAN = "network_scan"
    VULN_SCAN = "vuln_scan"
    INJECTION_TEST = "injection_test"
    EXPLOITATION = "exploitation"


@dataclass
class AgentInfo:
    """Information about a registered agent.

    Attributes:
        agent_id: Unique identifier for the agent instance
        agent_type: Type/class of the agent (e.g., "code_analyzer", "ui_tester")
        capabilities: List of capabilities this agent provides
        status: Current health status of the agent
        last_heartbeat: Timestamp of the last heartbeat received
        metadata: Additional metadata about the agent
        registered_at: Timestamp when the agent was registered
    """

    agent_id: str
    agent_type: str
    capabilities: list[Capability]
    status: Literal["healthy", "unhealthy", "unknown"]
    last_heartbeat: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "registered_at": self.registered_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentInfo":
        """Create from dictionary representation."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            capabilities=[Capability(cap) for cap in data["capabilities"]],
            status=data["status"],
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            registered_at=datetime.fromisoformat(data.get("registered_at", data["last_heartbeat"])),
            metadata=data.get("metadata", {}),
        )


class AgentRegistry:
    """Centralized registry for agent discovery and health monitoring.

    This singleton class manages:
    - Agent registration and unregistration
    - Capability-based agent discovery
    - Health monitoring via heartbeats
    - Automatic cleanup of unhealthy agents
    - Persistent storage in Supabase (with in-memory cache for performance)

    The registry is thread-safe and supports both sync and async operations.

    Configuration:
        - HEARTBEAT_TIMEOUT_SECONDS: Time before an agent is marked unhealthy (default: 60)
        - CLEANUP_INTERVAL_SECONDS: Interval for background cleanup task (default: 30)
        - AUTO_REMOVE_UNHEALTHY_AFTER: Remove agents after this many seconds unhealthy (default: 300)
        - DB_SYNC_INTERVAL_SECONDS: Interval for syncing in-memory cache with DB (default: 60)
    """

    # Configuration
    HEARTBEAT_TIMEOUT_SECONDS: int = 60
    CLEANUP_INTERVAL_SECONDS: int = 30
    AUTO_REMOVE_UNHEALTHY_AFTER: int = 300  # 5 minutes
    DB_SYNC_INTERVAL_SECONDS: int = 60  # Sync with DB every minute

    def __init__(self) -> None:
        """Initialize the agent registry."""
        self._agents: dict[str, AgentInfo] = {}
        self._lock = threading.RLock()
        self._cleanup_task: asyncio.Task | None = None
        self._db_sync_task: asyncio.Task | None = None
        self._supabase: SupabaseClient | None = None
        self._db_loaded: bool = False
        self._log = logger.bind(component="agent_registry")

    def _get_supabase(self) -> SupabaseClient | None:
        """Get Supabase client lazily."""
        if self._supabase is None:
            try:
                from src.services.supabase_client import get_supabase_client
                self._supabase = get_supabase_client()
                if not self._supabase.is_configured:
                    self._log.warning("Supabase not configured, running in memory-only mode")
                    return None
            except Exception as e:
                self._log.warning("Failed to get Supabase client", error=str(e))
                return None
        return self._supabase if self._supabase.is_configured else None

    async def _persist_agent(self, agent: AgentInfo) -> bool:
        """Persist agent registration to database.

        Args:
            agent: The agent info to persist

        Returns:
            True if persisted successfully, False otherwise
        """
        supabase = self._get_supabase()
        if not supabase:
            return False

        try:
            # Upsert the agent (insert or update on conflict)
            result = await supabase.request(
                "/agent_registrations",
                method="POST",
                body={
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "status": agent.status,
                    "last_heartbeat": agent.last_heartbeat.isoformat(),
                    "metadata": agent.metadata,
                    "created_at": agent.registered_at.isoformat(),
                },
                headers={"Prefer": "resolution=merge-duplicates"},
            )

            if result.get("error"):
                self._log.error(
                    "Failed to persist agent to database",
                    agent_id=agent.agent_id,
                    error=result["error"],
                )
                return False

            self._log.debug("Agent persisted to database", agent_id=agent.agent_id)
            return True

        except Exception as e:
            self._log.error(
                "Exception persisting agent to database",
                agent_id=agent.agent_id,
                error=str(e),
            )
            return False

    async def _delete_agent_from_db(self, agent_id: str) -> bool:
        """Delete agent registration from database.

        Args:
            agent_id: The agent ID to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        supabase = self._get_supabase()
        if not supabase:
            return False

        try:
            result = await supabase.request(
                f"/agent_registrations?agent_id=eq.{agent_id}",
                method="DELETE",
            )

            if result.get("error"):
                self._log.error(
                    "Failed to delete agent from database",
                    agent_id=agent_id,
                    error=result["error"],
                )
                return False

            self._log.debug("Agent deleted from database", agent_id=agent_id)
            return True

        except Exception as e:
            self._log.error(
                "Exception deleting agent from database",
                agent_id=agent_id,
                error=str(e),
            )
            return False

    async def _update_heartbeat_in_db(self, agent_id: str, heartbeat: datetime, status: str) -> bool:
        """Update agent heartbeat in database.

        Args:
            agent_id: The agent ID to update
            heartbeat: The new heartbeat timestamp
            status: The new status

        Returns:
            True if updated successfully, False otherwise
        """
        supabase = self._get_supabase()
        if not supabase:
            return False

        try:
            result = await supabase.request(
                f"/agent_registrations?agent_id=eq.{agent_id}",
                method="PATCH",
                body={
                    "last_heartbeat": heartbeat.isoformat(),
                    "status": status,
                },
            )

            if result.get("error"):
                self._log.error(
                    "Failed to update heartbeat in database",
                    agent_id=agent_id,
                    error=result["error"],
                )
                return False

            return True

        except Exception as e:
            self._log.error(
                "Exception updating heartbeat in database",
                agent_id=agent_id,
                error=str(e),
            )
            return False

    async def load_from_database(self) -> int:
        """Load agent registrations from database into memory cache.

        This should be called during application startup to restore
        agent state from the previous session.

        Returns:
            Number of agents loaded from database
        """
        supabase = self._get_supabase()
        if not supabase:
            self._log.info("Supabase not available, skipping database load")
            self._db_loaded = True
            return 0

        try:
            result = await supabase.request(
                "/agent_registrations?select=*",
                method="GET",
            )

            if result.get("error"):
                self._log.error(
                    "Failed to load agents from database",
                    error=result["error"],
                )
                self._db_loaded = True
                return 0

            agents_data = result.get("data", [])
            if not agents_data:
                self._log.info("No agents found in database")
                self._db_loaded = True
                return 0

            loaded_count = 0
            with self._lock:
                for agent_data in agents_data:
                    try:
                        # Parse capabilities from JSON array
                        capabilities = []
                        for cap_str in agent_data.get("capabilities", []):
                            try:
                                capabilities.append(Capability(cap_str))
                            except ValueError:
                                self._log.warning(
                                    "Unknown capability in database",
                                    capability=cap_str,
                                    agent_id=agent_data.get("agent_id"),
                                )

                        # Parse timestamps
                        last_heartbeat_str = agent_data.get("last_heartbeat")
                        last_heartbeat = (
                            datetime.fromisoformat(last_heartbeat_str.replace("Z", "+00:00"))
                            if last_heartbeat_str
                            else datetime.now(UTC)
                        )

                        created_at_str = agent_data.get("created_at")
                        registered_at = (
                            datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                            if created_at_str
                            else datetime.now(UTC)
                        )

                        agent_info = AgentInfo(
                            agent_id=agent_data["agent_id"],
                            agent_type=agent_data["agent_type"],
                            capabilities=capabilities,
                            status=agent_data.get("status", "unknown"),
                            last_heartbeat=last_heartbeat,
                            metadata=agent_data.get("metadata", {}),
                            registered_at=registered_at,
                        )

                        self._agents[agent_info.agent_id] = agent_info
                        loaded_count += 1

                    except Exception as e:
                        self._log.error(
                            "Failed to parse agent from database",
                            agent_data=agent_data,
                            error=str(e),
                        )

            self._log.info(
                "Agents loaded from database",
                count=loaded_count,
                total_in_db=len(agents_data),
            )
            self._db_loaded = True
            return loaded_count

        except Exception as e:
            self._log.error("Exception loading agents from database", error=str(e))
            self._db_loaded = True
            return 0

    def register(
        self,
        agent_type: str,
        capabilities: list[Capability] | list[str],
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Register a new agent with the registry.

        This method updates the in-memory cache synchronously and schedules
        an async task to persist to the database. For async persistence,
        use `register_async()` instead.

        Args:
            agent_type: Type/class of the agent (e.g., "code_analyzer")
            capabilities: List of capabilities this agent provides
            metadata: Optional metadata about the agent
            agent_id: Optional specific agent ID (auto-generated if not provided)

        Returns:
            The agent ID (generated or provided)

        Example:
            ```python
            agent_id = registry.register(
                agent_type="ui_tester",
                capabilities=[Capability.BROWSER_AUTOMATION, Capability.VISUAL_TESTING],
                metadata={"browser": "chromium", "version": "2.0.0"}
            )
            ```
        """
        with self._lock:
            # Generate agent ID if not provided
            if agent_id is None:
                agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"

            # Convert string capabilities to Capability enum
            normalized_capabilities = []
            for cap in capabilities:
                if isinstance(cap, str):
                    try:
                        normalized_capabilities.append(Capability(cap))
                    except ValueError:
                        self._log.warning(
                            "Unknown capability, skipping",
                            capability=cap,
                            agent_type=agent_type,
                        )
                else:
                    normalized_capabilities.append(cap)

            now = datetime.now(UTC)
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=normalized_capabilities,
                status="healthy",
                last_heartbeat=now,
                registered_at=now,
                metadata=metadata or {},
            )

            self._agents[agent_id] = agent_info

            self._log.info(
                "Agent registered",
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=[cap.value for cap in normalized_capabilities],
            )

            # Schedule async persistence (fire and forget)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_agent(agent_info))
            except RuntimeError:
                # No event loop running, persistence will happen on next sync
                self._log.debug(
                    "No event loop, agent will be persisted on next sync",
                    agent_id=agent_id,
                )

            return agent_id

    async def register_async(
        self,
        agent_type: str,
        capabilities: list[Capability] | list[str],
        metadata: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> str:
        """Register a new agent with the registry (async version with guaranteed persistence).

        This method updates the in-memory cache and persists to the database
        before returning. Use this when you need to ensure the agent is
        persisted before continuing.

        Args:
            agent_type: Type/class of the agent (e.g., "code_analyzer")
            capabilities: List of capabilities this agent provides
            metadata: Optional metadata about the agent
            agent_id: Optional specific agent ID (auto-generated if not provided)

        Returns:
            The agent ID (generated or provided)
        """
        # First register in memory
        registered_id = self.register(agent_type, capabilities, metadata, agent_id)

        # Then persist to database
        agent_info = self._agents.get(registered_id)
        if agent_info:
            await self._persist_agent(agent_info)

        return registered_id

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry.

        This method removes from the in-memory cache synchronously and schedules
        an async task to delete from the database. For async deletion,
        use `unregister_async()` instead.

        Args:
            agent_id: The ID of the agent to remove

        Returns:
            True if the agent was removed, False if not found
        """
        with self._lock:
            if agent_id in self._agents:
                agent_info = self._agents.pop(agent_id)
                self._log.info(
                    "Agent unregistered",
                    agent_id=agent_id,
                    agent_type=agent_info.agent_type,
                )

                # Schedule async deletion (fire and forget)
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._delete_agent_from_db(agent_id))
                except RuntimeError:
                    # No event loop running, deletion will happen on next sync
                    self._log.debug(
                        "No event loop, agent will be deleted from DB on next sync",
                        agent_id=agent_id,
                    )

                return True
            else:
                self._log.warning(
                    "Attempted to unregister unknown agent",
                    agent_id=agent_id,
                )
                return False

    async def unregister_async(self, agent_id: str) -> bool:
        """Remove an agent from the registry (async version with guaranteed DB deletion).

        This method removes from the in-memory cache and deletes from the database
        before returning.

        Args:
            agent_id: The ID of the agent to remove

        Returns:
            True if the agent was removed, False if not found
        """
        with self._lock:
            if agent_id in self._agents:
                agent_info = self._agents.pop(agent_id)
                self._log.info(
                    "Agent unregistered",
                    agent_id=agent_id,
                    agent_type=agent_info.agent_type,
                )
            else:
                self._log.warning(
                    "Attempted to unregister unknown agent",
                    agent_id=agent_id,
                )
                return False

        # Delete from database
        await self._delete_agent_from_db(agent_id)
        return True

    def discover(self, capability: Capability | str) -> list[AgentInfo]:
        """Find agents that have a specific capability.

        Args:
            capability: The capability to search for

        Returns:
            List of AgentInfo for agents with the capability

        Example:
            ```python
            browser_agents = registry.discover(Capability.BROWSER_AUTOMATION)
            healthy_agents = [a for a in browser_agents if a.status == "healthy"]
            ```
        """
        # Normalize capability to enum
        if isinstance(capability, str):
            try:
                capability = Capability(capability)
            except ValueError:
                self._log.warning("Unknown capability requested", capability=capability)
                return []

        with self._lock:
            matching = [
                agent
                for agent in self._agents.values()
                if capability in agent.capabilities
            ]

            self._log.debug(
                "Discovered agents",
                capability=capability.value,
                count=len(matching),
            )

            return matching

    def discover_all(self, capabilities: list[Capability] | list[str]) -> list[AgentInfo]:
        """Find agents that have ALL of the specified capabilities.

        Args:
            capabilities: List of capabilities the agent must have

        Returns:
            List of AgentInfo for agents with all capabilities
        """
        # Normalize capabilities to enum
        normalized_caps = []
        for cap in capabilities:
            if isinstance(cap, str):
                try:
                    normalized_caps.append(Capability(cap))
                except ValueError:
                    self._log.warning("Unknown capability requested", capability=cap)
                    return []
            else:
                normalized_caps.append(cap)

        with self._lock:
            matching = [
                agent
                for agent in self._agents.values()
                if all(cap in agent.capabilities for cap in normalized_caps)
            ]

            return matching

    def discover_any(self, capabilities: list[Capability] | list[str]) -> list[AgentInfo]:
        """Find agents that have ANY of the specified capabilities.

        Args:
            capabilities: List of capabilities to search for

        Returns:
            List of AgentInfo for agents with at least one capability
        """
        # Normalize capabilities to enum
        normalized_caps = set()
        for cap in capabilities:
            if isinstance(cap, str):
                try:
                    normalized_caps.add(Capability(cap))
                except ValueError:
                    continue
            else:
                normalized_caps.add(cap)

        with self._lock:
            matching = [
                agent
                for agent in self._agents.values()
                if any(cap in agent.capabilities for cap in normalized_caps)
            ]

            return matching

    def get(self, agent_id: str) -> AgentInfo | None:
        """Get information about a specific agent.

        Args:
            agent_id: The ID of the agent to look up

        Returns:
            AgentInfo if found, None otherwise
        """
        with self._lock:
            return self._agents.get(agent_id)

    def update_heartbeat(self, agent_id: str) -> bool:
        """Update the heartbeat timestamp for an agent.

        This should be called periodically by agents to indicate they are
        still alive and functioning. Agents that don't send heartbeats
        will be marked as unhealthy after HEARTBEAT_TIMEOUT_SECONDS.

        This method updates the in-memory cache synchronously and schedules
        an async task to update the database.

        Args:
            agent_id: The ID of the agent

        Returns:
            True if the heartbeat was recorded, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                self._log.warning(
                    "Heartbeat from unknown agent",
                    agent_id=agent_id,
                )
                return False

            agent = self._agents[agent_id]
            agent.last_heartbeat = datetime.now(UTC)

            # Mark as healthy if it was unhealthy
            old_status = agent.status
            if agent.status == "unhealthy":
                agent.status = "healthy"
                self._log.info(
                    "Agent recovered",
                    agent_id=agent_id,
                    agent_type=agent.agent_type,
                )

            # Schedule async DB update (fire and forget)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._update_heartbeat_in_db(agent_id, agent.last_heartbeat, agent.status)
                )
            except RuntimeError:
                # No event loop running, update will happen on next sync
                pass

            return True

    async def update_heartbeat_async(self, agent_id: str) -> bool:
        """Update the heartbeat timestamp for an agent (async version with guaranteed DB update).

        Args:
            agent_id: The ID of the agent

        Returns:
            True if the heartbeat was recorded, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                self._log.warning(
                    "Heartbeat from unknown agent",
                    agent_id=agent_id,
                )
                return False

            agent = self._agents[agent_id]
            agent.last_heartbeat = datetime.now(UTC)

            # Mark as healthy if it was unhealthy
            if agent.status == "unhealthy":
                agent.status = "healthy"
                self._log.info(
                    "Agent recovered",
                    agent_id=agent_id,
                    agent_type=agent.agent_type,
                )

            heartbeat = agent.last_heartbeat
            status = agent.status

        # Update in database (outside lock)
        await self._update_heartbeat_in_db(agent_id, heartbeat, status)
        return True

    def update_status(
        self,
        agent_id: str,
        status: Literal["healthy", "unhealthy", "unknown"],
    ) -> bool:
        """Manually update the status of an agent.

        Args:
            agent_id: The ID of the agent
            status: The new status

        Returns:
            True if status was updated, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False

            old_status = self._agents[agent_id].status
            self._agents[agent_id].status = status

            if old_status != status:
                self._log.info(
                    "Agent status changed",
                    agent_id=agent_id,
                    old_status=old_status,
                    new_status=status,
                )

            return True

    def update_metadata(
        self,
        agent_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> bool:
        """Update the metadata for an agent.

        Args:
            agent_id: The ID of the agent
            metadata: New metadata to set or merge
            merge: If True, merge with existing metadata; if False, replace

        Returns:
            True if metadata was updated, False if agent not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False

            if merge:
                self._agents[agent_id].metadata.update(metadata)
            else:
                self._agents[agent_id].metadata = metadata

            return True

    def get_healthy_agents(self) -> list[AgentInfo]:
        """Get all agents with healthy status.

        Returns:
            List of healthy AgentInfo objects
        """
        with self._lock:
            return [
                agent
                for agent in self._agents.values()
                if agent.status == "healthy"
            ]

    def get_all_agents(self) -> list[AgentInfo]:
        """Get all registered agents regardless of status.

        Returns:
            List of all AgentInfo objects
        """
        with self._lock:
            return list(self._agents.values())

    def get_agents_by_type(self, agent_type: str) -> list[AgentInfo]:
        """Get all agents of a specific type.

        Args:
            agent_type: The type of agent to find

        Returns:
            List of matching AgentInfo objects
        """
        with self._lock:
            return [
                agent
                for agent in self._agents.values()
                if agent.agent_type == agent_type
            ]

    def check_health(self) -> dict[str, Any]:
        """Check and update health status of all agents.

        This method:
        1. Marks agents as unhealthy if their heartbeat has timed out
        2. Optionally removes agents that have been unhealthy too long
        3. Schedules async DB updates for status changes

        Returns:
            Dict with health check results
        """
        now = datetime.now(UTC)
        timeout_threshold = now - timedelta(seconds=self.HEARTBEAT_TIMEOUT_SECONDS)
        removal_threshold = now - timedelta(seconds=self.AUTO_REMOVE_UNHEALTHY_AFTER)

        marked_unhealthy = []
        removed = []
        status_updates: list[tuple[str, datetime, str]] = []

        with self._lock:
            for agent_id, agent in list(self._agents.items()):
                # Check for heartbeat timeout
                if agent.last_heartbeat < timeout_threshold:
                    if agent.status == "healthy":
                        agent.status = "unhealthy"
                        marked_unhealthy.append(agent_id)
                        status_updates.append((agent_id, agent.last_heartbeat, "unhealthy"))
                        self._log.warning(
                            "Agent marked unhealthy (heartbeat timeout)",
                            agent_id=agent_id,
                            agent_type=agent.agent_type,
                            last_heartbeat=agent.last_heartbeat.isoformat(),
                        )

                    # Check for removal (unhealthy too long)
                    if agent.last_heartbeat < removal_threshold:
                        del self._agents[agent_id]
                        removed.append(agent_id)
                        self._log.info(
                            "Agent removed (unhealthy too long)",
                            agent_id=agent_id,
                            agent_type=agent.agent_type,
                        )

        # Schedule async DB updates (fire and forget)
        try:
            loop = asyncio.get_running_loop()
            for agent_id, heartbeat, status in status_updates:
                loop.create_task(self._update_heartbeat_in_db(agent_id, heartbeat, status))
            for agent_id in removed:
                loop.create_task(self._delete_agent_from_db(agent_id))
        except RuntimeError:
            # No event loop running, updates will happen on next sync
            pass

        return {
            "total_agents": len(self._agents),
            "healthy_count": len(self.get_healthy_agents()),
            "marked_unhealthy": marked_unhealthy,
            "removed": removed,
            "timestamp": now.isoformat(),
        }

    async def start_health_monitor(self) -> None:
        """Start the background health monitoring task.

        This task periodically checks agent health and removes stale agents.
        Should be called during application startup.

        Also loads agents from the database if not already loaded.
        """
        # Load from database if not already loaded
        if not self._db_loaded:
            await self.load_from_database()

        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._log.warning("Health monitor already running")
            return

        self._log.info(
            "Starting health monitor",
            interval_seconds=self.CLEANUP_INTERVAL_SECONDS,
        )

        self._cleanup_task = asyncio.create_task(self._health_monitor_loop())

    async def stop_health_monitor(self) -> None:
        """Stop the background health monitoring task.

        Should be called during application shutdown.
        """
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            self._log.info("Health monitor stopped")

    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)
                result = self.check_health()

                if result["marked_unhealthy"] or result["removed"]:
                    self._log.info(
                        "Health check completed",
                        total=result["total_agents"],
                        healthy=result["healthy_count"],
                        marked_unhealthy=len(result["marked_unhealthy"]),
                        removed=len(result["removed"]),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error("Health monitor error", error=str(e))

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the registry.

        Returns:
            Dict with registry statistics
        """
        with self._lock:
            agents = list(self._agents.values())
            capabilities_count: dict[str, int] = {}
            types_count: dict[str, int] = {}
            status_count = {"healthy": 0, "unhealthy": 0, "unknown": 0}

            for agent in agents:
                # Count by type
                types_count[agent.agent_type] = types_count.get(agent.agent_type, 0) + 1

                # Count by status
                status_count[agent.status] += 1

                # Count capabilities
                for cap in agent.capabilities:
                    capabilities_count[cap.value] = capabilities_count.get(cap.value, 0) + 1

            return {
                "total_agents": len(agents),
                "by_status": status_count,
                "by_type": types_count,
                "capabilities": capabilities_count,
                "health_monitor_running": self._cleanup_task is not None and not self._cleanup_task.done(),
            }

    def clear(self) -> int:
        """Remove all agents from the in-memory registry.

        Note: This only clears the in-memory cache. Use `clear_async()` to
        also clear the database.

        Returns:
            Number of agents removed
        """
        with self._lock:
            count = len(self._agents)
            self._agents.clear()
            self._log.info("Registry cleared (in-memory only)", removed_count=count)
            return count

    async def clear_async(self, clear_database: bool = True) -> int:
        """Remove all agents from the registry and optionally the database.

        Args:
            clear_database: If True, also delete all agents from the database

        Returns:
            Number of agents removed from memory
        """
        with self._lock:
            count = len(self._agents)
            agent_ids = list(self._agents.keys())
            self._agents.clear()

        if clear_database:
            supabase = self._get_supabase()
            if supabase:
                try:
                    # Delete all agent registrations
                    result = await supabase.request(
                        "/agent_registrations",
                        method="DELETE",
                        headers={"Prefer": "return=minimal"},
                    )
                    if result.get("error"):
                        self._log.error(
                            "Failed to clear agents from database",
                            error=result["error"],
                        )
                    else:
                        self._log.info(
                            "Registry cleared (including database)",
                            removed_count=count,
                        )
                except Exception as e:
                    self._log.error(
                        "Exception clearing agents from database",
                        error=str(e),
                    )
            else:
                self._log.info("Registry cleared (in-memory only, no DB)", removed_count=count)
        else:
            self._log.info("Registry cleared (in-memory only)", removed_count=count)

        return count


# =========================================================================
# Global Instance Management (Singleton Pattern)
# =========================================================================

_registry: AgentRegistry | None = None
_registry_lock = threading.Lock()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance.

    This function returns a singleton instance of the AgentRegistry,
    creating it on first access.

    Returns:
        The global AgentRegistry instance

    Example:
        ```python
        from src.orchestrator.agent_registry import get_agent_registry, Capability

        registry = get_agent_registry()
        agents = registry.discover(Capability.CODE_ANALYSIS)
        ```
    """
    global _registry

    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = AgentRegistry()
                logger.info("Agent registry initialized")

    return _registry


def reset_agent_registry() -> None:
    """Reset the global agent registry instance.

    This is primarily useful for testing or when reconfiguring the system.
    Note: This does not stop the health monitor if it's running.
    """
    global _registry

    with _registry_lock:
        if _registry is not None:
            _registry.clear()
        _registry = None
        logger.info("Agent registry reset")


async def init_agent_registry(
    start_health_monitor: bool = True,
    load_from_db: bool = True,
) -> AgentRegistry:
    """Initialize the agent registry with optional health monitoring.

    This should be called during application startup to ensure the registry
    is ready, agents are loaded from the database, and the health monitor
    is running.

    Args:
        start_health_monitor: Whether to start the background health monitor
        load_from_db: Whether to load agents from the database on startup

    Returns:
        The initialized AgentRegistry instance

    Example:
        ```python
        # In your FastAPI startup
        @app.on_event("startup")
        async def startup():
            await init_agent_registry(start_health_monitor=True)
        ```
    """
    registry = get_agent_registry()

    # Load agents from database (restores state from previous session)
    if load_from_db:
        loaded_count = await registry.load_from_database()
        logger.info(
            "Agents loaded from database",
            count=loaded_count,
        )

    if start_health_monitor:
        await registry.start_health_monitor()

    logger.info(
        "Agent registry initialized",
        health_monitor=start_health_monitor,
        db_loaded=load_from_db,
        agent_count=len(registry.get_all_agents()),
    )

    return registry


async def shutdown_agent_registry() -> None:
    """Shutdown the agent registry gracefully.

    This should be called during application shutdown to properly
    stop the health monitor.

    Example:
        ```python
        # In your FastAPI shutdown
        @app.on_event("shutdown")
        async def shutdown():
            await shutdown_agent_registry()
        ```
    """
    global _registry

    if _registry is not None:
        await _registry.stop_health_monitor()
        logger.info("Agent registry shutdown complete")


# =========================================================================
# Convenience Functions
# =========================================================================


def register_agent(
    agent_type: str,
    capabilities: list[Capability] | list[str],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Convenience function to register an agent with the global registry.

    Args:
        agent_type: Type/class of the agent
        capabilities: List of capabilities
        metadata: Optional metadata

    Returns:
        The agent ID
    """
    return get_agent_registry().register(agent_type, capabilities, metadata)


def unregister_agent(agent_id: str) -> bool:
    """Convenience function to unregister an agent from the global registry.

    Args:
        agent_id: The ID of the agent to remove

    Returns:
        True if removed, False if not found
    """
    return get_agent_registry().unregister(agent_id)


def discover_agents(capability: Capability | str) -> list[AgentInfo]:
    """Convenience function to discover agents by capability.

    Args:
        capability: The capability to search for

    Returns:
        List of matching agents
    """
    return get_agent_registry().discover(capability)


def heartbeat(agent_id: str) -> bool:
    """Convenience function to send a heartbeat.

    Args:
        agent_id: The ID of the agent

    Returns:
        True if recorded, False if agent not found
    """
    return get_agent_registry().update_heartbeat(agent_id)


async def heartbeat_async(agent_id: str) -> bool:
    """Convenience function to send a heartbeat with guaranteed DB persistence.

    Args:
        agent_id: The ID of the agent

    Returns:
        True if recorded, False if agent not found
    """
    return await get_agent_registry().update_heartbeat_async(agent_id)


# =========================================================================
# Agent Type to Capability Mapping (Default Capabilities)
# =========================================================================

DEFAULT_AGENT_CAPABILITIES: dict[str, list[Capability]] = {
    "code_analyzer": [
        Capability.CODE_ANALYSIS,
        Capability.GIT_BLAME,
        Capability.DEPENDENCY_ANALYSIS,
        Capability.TEST_IMPACT_ANALYSIS,
    ],
    "ui_tester": [
        Capability.BROWSER_AUTOMATION,
        Capability.COMPUTER_USE,
        Capability.VISUAL_TESTING,
        Capability.SCREENSHOT_CAPTURE,
    ],
    "api_tester": [
        Capability.API_TESTING,
        Capability.SCHEMA_VALIDATION,
        Capability.GRAPHQL_TESTING,
    ],
    "self_healer": [
        Capability.SELECTOR_FIX,
        Capability.ASSERTION_FIX,
        Capability.FLAKY_DETECTION,
        Capability.ROOT_CAUSE_ANALYSIS,
        Capability.AUTO_HEALING,
    ],
    "test_planner": [
        Capability.TEST_PLANNING,
        Capability.TEST_PRIORITIZATION,
        Capability.RISK_ASSESSMENT,
        Capability.TEST_GENERATION,
    ],
    "reporter": [
        Capability.REPORT_GENERATION,
        Capability.SLACK_INTEGRATION,
        Capability.GITHUB_INTEGRATION,
        Capability.JIRA_INTEGRATION,
    ],
    "db_tester": [
        Capability.DB_TESTING,
        Capability.DATA_VALIDATION,
        Capability.MIGRATION_TESTING,
    ],
    "performance_analyzer": [
        Capability.PERFORMANCE_ANALYSIS,
        Capability.LOAD_TESTING,
        Capability.LATENCY_MONITORING,
    ],
    "security_scanner": [
        Capability.SECURITY_SCAN,
        Capability.CODE_ANALYSIS,
    ],
    "accessibility_checker": [
        Capability.ACCESSIBILITY_CHECK,
        Capability.VISUAL_TESTING,
    ],
    "auto_discovery": [
        Capability.AUTO_DISCOVERY,
        Capability.FLOW_DISCOVERY,
        Capability.CRAWLING,
    ],
    "visual_ai": [
        Capability.VISUAL_AI,
        Capability.VISUAL_TESTING,
        Capability.COGNITIVE_MODELING,
    ],
    "nlp_test_creator": [
        Capability.NLP_UNDERSTANDING,
        Capability.TEST_GENERATION,
    ],
}


def get_default_capabilities(agent_type: str) -> list[Capability]:
    """Get the default capabilities for an agent type.

    Args:
        agent_type: The type of agent

    Returns:
        List of default capabilities, empty list if unknown type
    """
    return DEFAULT_AGENT_CAPABILITIES.get(agent_type, [])
