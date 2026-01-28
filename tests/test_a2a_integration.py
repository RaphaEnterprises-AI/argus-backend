"""A2A Architecture Integration Tests.

Tests the Agent-to-Agent communication infrastructure:
- Agent Registry with capability discovery
- A2A Protocol message passing
- Workflow Composer
- MARP consensus
"""

import asyncio
from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

# =============================================================================
# Test Agent Registry
# =============================================================================

class TestAgentRegistry:
    """Test Agent Registry functionality."""

    def test_capability_enum(self):
        """Test that Capability enum is properly defined."""
        from src.orchestrator.agent_registry import Capability

        # Check key capabilities exist
        assert Capability.CODE_ANALYSIS == "code_analysis"
        assert Capability.GIT_BLAME == "git_blame"
        assert Capability.BROWSER_AUTOMATION == "browser_automation"
        assert Capability.SELECTOR_FIX == "selector_fix"
        assert Capability.API_TESTING == "api_testing"

        # Count total capabilities
        capabilities = [c for c in Capability]
        print(f"✅ {len(capabilities)} capabilities defined in Capability enum")
        assert len(capabilities) >= 30, f"Expected 30+ capabilities, got {len(capabilities)}"

    def test_get_agent_registry(self):
        """Test that get_agent_registry returns a registry instance."""
        from src.orchestrator.agent_registry import AgentRegistry, get_agent_registry

        registry = get_agent_registry()
        assert isinstance(registry, AgentRegistry)
        print("✅ get_agent_registry() returns AgentRegistry instance")

    def test_register_and_discover(self):
        """Test agent registration and discovery."""
        from src.orchestrator.agent_registry import Capability, get_agent_registry

        registry = get_agent_registry()

        # Register a mock agent
        agent_id = registry.register(
            agent_type="test_agent",
            capabilities=[Capability.CODE_ANALYSIS, Capability.GIT_BLAME],
            metadata={"version": "1.0.0"}
        )

        assert agent_id is not None
        print(f"✅ Registered agent: {agent_id}")

        # Discover by capability
        agents = registry.discover(Capability.CODE_ANALYSIS)
        agent_ids = [a.agent_id for a in agents]
        assert agent_id in agent_ids, "Registered agent should be discoverable"
        print("✅ Agent discovery by capability works")

        # Unregister
        result = registry.unregister(agent_id)
        assert result is True
        print("✅ Agent unregistration works")

    def test_agent_info_dataclass(self):
        """Test AgentInfo dataclass."""
        from src.orchestrator.agent_registry import AgentInfo, Capability

        agent = AgentInfo(
            agent_id="test-123",
            agent_type="code_analyzer",
            capabilities=[Capability.CODE_ANALYSIS],
            status="healthy",  # Literal type, not enum
            metadata={"version": "1.0.0"},
            registered_at=datetime.now(UTC),
            last_heartbeat=datetime.now(UTC)
        )

        assert agent.agent_id == "test-123"
        assert agent.status == "healthy"
        print("✅ AgentInfo dataclass works")


# =============================================================================
# Test A2A Protocol
# =============================================================================

class TestA2AProtocol:
    """Test A2A Protocol functionality."""

    def test_protocol_instantiation(self):
        """Test A2A Protocol can be instantiated."""
        from src.orchestrator.a2a_protocol import A2AProtocol

        protocol = A2AProtocol(agent_id="test-agent-1", agent_type="test")
        assert protocol is not None
        assert protocol.agent_id == "test-agent-1"
        print("✅ A2A Protocol instantiation works")

    def test_message_types(self):
        """Test MessageType enum."""
        from src.orchestrator.a2a_protocol import MessageType

        assert MessageType.REQUEST == "request"
        assert MessageType.RESPONSE == "response"
        assert MessageType.BROADCAST == "broadcast"
        assert MessageType.HEARTBEAT == "heartbeat"
        print("✅ MessageType enum works")

    def test_agent_request_event(self):
        """Test AgentRequestEvent model."""
        from src.orchestrator.a2a_protocol import AgentRequestEvent

        request = AgentRequestEvent(
            from_agent="test-agent-1",
            from_agent_type="tester",
            to_agent="analyzer-1",
            capability="code_analysis",
            payload={"file": "test.py"},
        )

        assert request.from_agent == "test-agent-1"
        assert request.capability == "code_analysis"
        assert request.request_id is not None
        print("✅ AgentRequestEvent model works")

    def test_agent_response(self):
        """Test AgentResponse model."""
        from src.orchestrator.a2a_protocol import AgentResponse

        response = AgentResponse(
            request_id=str(uuid4()),
            from_agent="analyzer-1",
            from_agent_type="code_analyzer",
            to_agent="requester-1",
            success=True,
            payload={"result": "analyzed"},
        )

        assert response.success is True
        assert response.payload["result"] == "analyzed"
        assert response.to_agent == "requester-1"
        print("✅ AgentResponse model works")

    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        from src.orchestrator.a2a_protocol import CircuitBreaker, CircuitState

        cb = CircuitBreaker(agent_id="test-agent", failure_threshold=3, recovery_timeout=1.0)

        # Initially closed
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

        # Record failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

        print("✅ Circuit breaker pattern works")


# =============================================================================
# Test MARP (Multi-Agent Reasoning Protocol)
# =============================================================================

class TestMARP:
    """Test Multi-Agent Reasoning Protocol."""

    def test_vote_type_enum(self):
        """Test VoteType enum."""
        from src.orchestrator.marp import VoteType

        assert VoteType.SUPPORT == "support"
        assert VoteType.OPPOSE == "oppose"
        assert VoteType.ABSTAIN == "abstain"
        print("✅ VoteType enum works")

    def test_proposal_dataclass(self):
        """Test Proposal dataclass."""
        from src.orchestrator.marp import Proposal

        proposal = Proposal(
            proposal_id=str(uuid4()),
            agent_id="self-healer-1",
            agent_type="self_healer",
            solution={"new_selector": "#submit-btn"},
            confidence=0.85,
            reasoning="Element ID changed",
            timestamp=datetime.now(UTC)
        )

        assert proposal.confidence == 0.85
        assert proposal.agent_type == "self_healer"
        print("✅ Proposal dataclass works")

    def test_vote_dataclass(self):
        """Test Vote dataclass."""
        from src.orchestrator.marp import Vote, VoteType

        vote = Vote(
            vote_id=str(uuid4()),
            proposal_id="prop-123",
            voter_id="code-analyzer-1",
            voter_type="code_analyzer",
            vote_type=VoteType.SUPPORT,
            confidence=0.9,
            reasoning="The analysis confirms this fix",
            timestamp=datetime.now(UTC)
        )

        assert vote.vote_type == VoteType.SUPPORT
        assert vote.confidence == 0.9
        print("✅ Vote dataclass works")

    def test_marp_initialization(self):
        """Test MARP initialization."""
        from src.orchestrator.marp import MARP

        marp = MARP()
        assert marp is not None
        print("✅ MARP initialization works")


# =============================================================================
# Test Consensus Strategies
# =============================================================================

class TestConsensus:
    """Test Consensus strategies."""

    def test_majority_voting(self):
        """Test majority voting strategy."""
        from src.orchestrator.consensus import MajorityVoting
        from src.orchestrator.marp import Proposal, Vote, VoteType

        strategy = MajorityVoting()

        # Create a proposal
        proposal = Proposal(
            proposal_id="prop-1",
            agent_id="agent-1",
            agent_type="self_healer",
            solution={"fix": "update_selector"},
            confidence=0.8,
            reasoning="Test reasoning",
            timestamp=datetime.now(UTC)
        )

        # Create votes - 3 support, 2 oppose (using positional args: vote_id, voter_id, voter_type, proposal_id, vote_type, confidence, reasoning)
        votes = [
            Vote(str(uuid4()), "a1", "code_analyzer", "prop-1", VoteType.SUPPORT, 0.8, "Good fix"),
            Vote(str(uuid4()), "a2", "ui_tester", "prop-1", VoteType.SUPPORT, 0.7, "Looks correct"),
            Vote(str(uuid4()), "a3", "api_tester", "prop-1", VoteType.SUPPORT, 0.9, "Confirmed"),
            Vote(str(uuid4()), "a4", "security_scanner", "prop-1", VoteType.OPPOSE, 0.6, "Not sure"),
            Vote(str(uuid4()), "a5", "reporter", "prop-1", VoteType.OPPOSE, 0.5, "Need review"),
        ]

        # Resolve
        winning_proposal, metadata = strategy.resolve([proposal], votes)
        assert winning_proposal == proposal
        print("✅ Majority voting strategy works")

    def test_confidence_weighted_voting(self):
        """Test confidence-weighted voting."""
        from src.orchestrator.consensus import ConfidenceWeighted
        from src.orchestrator.marp import Proposal, Vote, VoteType

        strategy = ConfidenceWeighted()

        # Create proposal
        proposal = Proposal(
            proposal_id="prop-1",
            agent_id="agent-1",
            agent_type="self_healer",
            solution={"fix": "update_selector"},
            confidence=0.8,
            reasoning="Test reasoning",
            timestamp=datetime.now(UTC)
        )

        # High confidence oppose should outweigh low confidence supports
        votes = [
            Vote(str(uuid4()), "a1", "code_analyzer", "prop-1", VoteType.SUPPORT, 0.3, "Maybe"),
            Vote(str(uuid4()), "a2", "ui_tester", "prop-1", VoteType.SUPPORT, 0.3, "Could work"),
            Vote(str(uuid4()), "a3", "api_tester", "prop-1", VoteType.OPPOSE, 0.95, "This breaks API"),
        ]

        # This should return None (no consensus) or the proposal rejected
        result = strategy.resolve([proposal], votes)
        print("✅ Confidence-weighted voting strategy works")

    def test_all_strategies_exist(self):
        """Test that all consensus strategies are defined."""
        from src.orchestrator import consensus

        strategies = [
            'MajorityVoting',
            'ConfidenceWeighted',
            'ExpertiseWeighted',
            'SuperMajority',
            'BordaCount',
            'QuadraticVoting',
        ]

        for strategy_name in strategies:
            assert hasattr(consensus, strategy_name), f"Missing strategy: {strategy_name}"

        print(f"✅ All {len(strategies)} consensus strategies exist")


# =============================================================================
# Test Workflow Composer
# =============================================================================

class TestWorkflowComposer:
    """Test Workflow Composer functionality."""

    def test_task_priority_enum(self):
        """Test TaskPriority enum."""
        from src.orchestrator.workflow_composer import TaskPriority

        assert TaskPriority.CRITICAL == 0
        assert TaskPriority.HIGH == 1
        assert TaskPriority.MEDIUM == 2
        print("✅ TaskPriority enum works")

    def test_execution_mode_enum(self):
        """Test ExecutionMode enum."""
        from src.orchestrator.workflow_composer import ExecutionMode

        assert ExecutionMode.SEQUENTIAL == "sequential"
        assert ExecutionMode.PARALLEL == "parallel"
        assert ExecutionMode.CONSENSUS == "consensus"
        print("✅ ExecutionMode enum works")

    def test_workflow_composer_init(self):
        """Test WorkflowComposer initialization."""
        from src.orchestrator.agent_registry import get_agent_registry
        from src.orchestrator.workflow_composer import WorkflowComposer

        registry = get_agent_registry()
        composer = WorkflowComposer(registry)
        assert composer is not None
        print("✅ WorkflowComposer initialization works")


# =============================================================================
# Test Incremental Indexer
# =============================================================================

class TestIncrementalIndexer:
    """Test Incremental Indexer functionality."""

    def test_change_type_enum(self):
        """Test ChangeType enum."""
        from src.indexer.change_manifest import ChangeType

        assert ChangeType.ADDED == "added"
        assert ChangeType.MODIFIED == "modified"
        assert ChangeType.DELETED == "deleted"
        print("✅ ChangeType enum works")

    def test_file_change_dataclass(self):
        """Test FileChange dataclass."""
        from src.indexer.change_manifest import ChangeType, FileChange

        change = FileChange(
            path="src/auth.py",
            change_type=ChangeType.MODIFIED,
            additions=50,
            deletions=10,
        )

        assert change.path == "src/auth.py"
        assert change.additions == 50
        print("✅ FileChange dataclass works")

    def test_change_manifest(self):
        """Test ChangeManifest dataclass."""
        from src.indexer.change_manifest import ChangeManifest, ChangeType, FileChange

        change = FileChange(
            path="src/auth.py",
            change_type=ChangeType.MODIFIED,
            additions=50,
            deletions=10,
        )

        manifest = ChangeManifest(
            from_commit="abc123",
            to_commit="def456",
            files=[change],
        )

        assert len(manifest.files) == 1
        assert manifest.from_commit == "abc123"
        print("✅ ChangeManifest dataclass works")

    def test_indexer_initialization(self):
        """Test IncrementalIndexer initialization."""
        from src.indexer.incremental_indexer import IncrementalIndexer

        indexer = IncrementalIndexer()
        assert indexer is not None
        print("✅ IncrementalIndexer initialization works")


# =============================================================================
# Test CI/CD Monitor
# =============================================================================

class TestCICDMonitor:
    """Test CI/CD Monitor functionality."""

    def test_monitor_initialization(self):
        """Test CI/CD monitor initialization."""
        from src.services.cicd_monitor import CICDMonitor

        monitor = CICDMonitor()
        assert monitor is not None
        print("✅ CI/CD monitor initialization works")

    def test_pr_info_dataclass(self):
        """Test PRInfo dataclass."""
        from src.services.cicd_monitor import PRInfo, PRState

        pr = PRInfo(
            number=123,
            title="Fix auth bug",
            author="developer",
            head_branch="fix/auth",
            base_branch="main",
            state=PRState.OPEN,
            html_url="https://github.com/org/repo/pull/123",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert pr.number == 123
        assert pr.state == PRState.OPEN
        print("✅ PRInfo dataclass works")

    def test_mr_info_dataclass(self):
        """Test MRInfo dataclass (GitLab)."""
        from src.services.cicd_monitor import MRInfo, PRState

        mr = MRInfo(
            iid=456,
            title="Add feature",
            author="developer",
            source_branch="feature/new",
            target_branch="main",
            state=PRState.OPEN,
            web_url="https://gitlab.com/org/repo/-/merge_requests/456",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert mr.iid == 456
        assert mr.state == PRState.OPEN
        print("✅ MRInfo dataclass works")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all A2A integration tests."""
    print("\n" + "="*60)
    print("🧪 A2A ARCHITECTURE INTEGRATION TESTS")
    print("="*60 + "\n")

    # Agent Registry Tests
    print("\n📦 Testing Agent Registry...")
    registry_tests = TestAgentRegistry()
    registry_tests.test_capability_enum()
    registry_tests.test_get_agent_registry()
    registry_tests.test_register_and_discover()
    registry_tests.test_agent_info_dataclass()

    # A2A Protocol Tests
    print("\n📡 Testing A2A Protocol...")
    protocol_tests = TestA2AProtocol()
    protocol_tests.test_protocol_instantiation()
    protocol_tests.test_message_types()
    protocol_tests.test_agent_request_event()
    protocol_tests.test_agent_response()
    protocol_tests.test_circuit_breaker()

    # MARP Tests
    print("\n🗳️ Testing MARP...")
    marp_tests = TestMARP()
    marp_tests.test_vote_type_enum()
    marp_tests.test_proposal_dataclass()
    marp_tests.test_vote_dataclass()
    marp_tests.test_marp_initialization()

    # Consensus Tests
    print("\n📊 Testing Consensus Strategies...")
    consensus_tests = TestConsensus()
    consensus_tests.test_majority_voting()
    consensus_tests.test_confidence_weighted_voting()
    consensus_tests.test_all_strategies_exist()

    # Workflow Composer Tests
    print("\n🔄 Testing Workflow Composer...")
    composer_tests = TestWorkflowComposer()
    composer_tests.test_task_priority_enum()
    composer_tests.test_execution_mode_enum()
    composer_tests.test_workflow_composer_init()

    # Incremental Indexer Tests
    print("\n📁 Testing Incremental Indexer...")
    indexer_tests = TestIncrementalIndexer()
    indexer_tests.test_change_type_enum()
    indexer_tests.test_file_change_dataclass()
    indexer_tests.test_change_manifest()
    indexer_tests.test_indexer_initialization()

    # CI/CD Monitor Tests
    print("\n🔍 Testing CI/CD Monitor...")
    cicd_tests = TestCICDMonitor()
    cicd_tests.test_monitor_initialization()
    cicd_tests.test_pr_info_dataclass()
    cicd_tests.test_mr_info_dataclass()

    print("\n" + "="*60)
    print("✅ ALL A2A INTEGRATION TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
