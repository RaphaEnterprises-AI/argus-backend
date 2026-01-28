"""Multi-Agent Reasoning Protocol (MARP) for consensus decisions.

RAP-235: MARP enables multiple agents to collaborate on complex decisions
through a structured proposal and voting mechanism. This is particularly
useful for:

1. Test strategy decisions requiring multiple agent perspectives
2. Self-healing decisions where UI, API, and code analysis agents
   need to agree on the root cause
3. Quality assessments requiring consensus across different test types
4. Complex debugging scenarios where multiple hypotheses exist

Architecture:
    +---------+     propose()     +---------+
    | Agent A | ----------------> |  MARP   |
    +---------+                   |         |
                                  |         |
    +---------+     propose()     |         |
    | Agent B | ----------------> | Topic:  |
    +---------+                   | "fix"   |
                                  |         |
    +---------+   request_votes() |         |
    | Agent C | <---------------- |         |
    +---------+                   +---------+
         |                             |
         +------ vote() ------------->|
                                      |
                         resolve() -->+

Example usage:
    from src.orchestrator.marp import MARP, Proposal
    from src.orchestrator.consensus import ConfidenceWeighted

    marp = MARP(strategy=ConfidenceWeighted())

    # Agent proposes a solution
    proposal_id = await marp.propose(
        topic="selector_fix",
        agent_id="self_healer_1",
        agent_type="self_healer",
        solution={"new_selector": "#submit-btn", "old_selector": ".submit"},
        confidence=0.85,
        reasoning="The element ID changed from class to ID in recent deploy"
    )

    # Request votes from other agents
    votes = await marp.request_votes(topic="selector_fix", timeout=30.0)

    # Resolve to get winning proposal
    resolution = await marp.resolve(topic="selector_fix")
    if resolution.consensus_reached:
        apply_fix(resolution.winning_proposal.solution)
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol

import structlog

if TYPE_CHECKING:
    from .a2a_protocol import A2AProtocol

logger = structlog.get_logger()


class VoteType(str, Enum):
    """Types of votes an agent can cast."""
    SUPPORT = "support"
    OPPOSE = "oppose"
    ABSTAIN = "abstain"


class ResolutionStatus(str, Enum):
    """Status of a topic resolution."""
    PENDING = "pending"
    CONSENSUS_REACHED = "consensus_reached"
    NO_CONSENSUS = "no_consensus"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"
    TIE = "tie"


@dataclass
class Proposal:
    """A proposed solution from an agent.

    Attributes:
        proposal_id: Unique identifier for this proposal
        agent_id: ID of the agent making the proposal
        agent_type: Type of agent (e.g., "self_healer", "ui_tester", "code_analyzer")
        solution: The proposed solution as a dictionary
        confidence: Agent's confidence in the solution (0.0 to 1.0)
        reasoning: Explanation of why this solution is proposed
        timestamp: When the proposal was created
        metadata: Additional metadata for the proposal
    """
    proposal_id: str
    agent_id: str
    agent_type: str
    solution: dict[str, Any]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate confidence is within bounds."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "solution": self.solution,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        """Create from dictionary representation."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(UTC)

        return cls(
            proposal_id=data["proposal_id"],
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            solution=data["solution"],
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Vote:
    """An agent's vote on a proposal.

    Attributes:
        vote_id: Unique identifier for this vote
        voter_id: ID of the voting agent
        voter_type: Type of voting agent
        proposal_id: ID of the proposal being voted on
        vote_type: Type of vote (support, oppose, abstain)
        confidence: Voter's confidence in their vote (0.0 to 1.0)
        reasoning: Explanation of the vote
        timestamp: When the vote was cast
        expertise_score: Voter's expertise in the relevant domain (0.0 to 1.0)
    """
    vote_id: str
    voter_id: str
    voter_type: str
    proposal_id: str
    vote_type: VoteType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    expertise_score: float = 0.5  # Domain expertise (0.0 to 1.0)

    @property
    def support(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.vote_type == VoteType.SUPPORT

    def __post_init__(self):
        """Validate confidence and expertise are within bounds."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.expertise_score <= 1.0:
            raise ValueError(f"Expertise score must be between 0.0 and 1.0, got {self.expertise_score}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vote_id": self.vote_id,
            "voter_id": self.voter_id,
            "voter_type": self.voter_type,
            "proposal_id": self.proposal_id,
            "vote_type": self.vote_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "expertise_score": self.expertise_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vote:
        """Create from dictionary representation."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(UTC)

        vote_type = data.get("vote_type", VoteType.ABSTAIN)
        if isinstance(vote_type, str):
            vote_type = VoteType(vote_type)

        # Handle legacy 'support' field
        if "support" in data and "vote_type" not in data:
            vote_type = VoteType.SUPPORT if data["support"] else VoteType.OPPOSE

        return cls(
            vote_id=data.get("vote_id", str(uuid.uuid4())),
            voter_id=data["voter_id"],
            voter_type=data["voter_type"],
            proposal_id=data["proposal_id"],
            vote_type=vote_type,
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            timestamp=timestamp,
            expertise_score=data.get("expertise_score", 0.5),
        )


@dataclass
class Resolution:
    """Result of resolving a topic with votes.

    Attributes:
        topic: The topic that was resolved
        status: Resolution status
        winning_proposal: The selected proposal (if consensus reached)
        all_proposals: All proposals submitted for this topic
        all_votes: All votes cast
        total_support: Total weighted support score
        total_opposition: Total weighted opposition score
        participation_rate: Percentage of eligible agents that voted
        resolution_time: How long the resolution took
        reasoning: Explanation of the resolution decision
        audit_trail: Detailed audit information
    """
    topic: str
    status: ResolutionStatus
    winning_proposal: Proposal | None = None
    all_proposals: list[Proposal] = field(default_factory=list)
    all_votes: list[Vote] = field(default_factory=list)
    total_support: float = 0.0
    total_opposition: float = 0.0
    participation_rate: float = 0.0
    resolution_time: timedelta | None = None
    reasoning: str = ""
    audit_trail: dict[str, Any] = field(default_factory=dict)

    @property
    def consensus_reached(self) -> bool:
        """Check if consensus was reached."""
        return self.status == ResolutionStatus.CONSENSUS_REACHED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "topic": self.topic,
            "status": self.status.value,
            "winning_proposal": self.winning_proposal.to_dict() if self.winning_proposal else None,
            "all_proposals": [p.to_dict() for p in self.all_proposals],
            "all_votes": [v.to_dict() for v in self.all_votes],
            "total_support": self.total_support,
            "total_opposition": self.total_opposition,
            "participation_rate": self.participation_rate,
            "resolution_time": str(self.resolution_time) if self.resolution_time else None,
            "consensus_reached": self.consensus_reached,
            "reasoning": self.reasoning,
            "audit_trail": self.audit_trail,
        }


class VoteRequestCallback(Protocol):
    """Protocol for vote request callbacks."""

    async def __call__(
        self,
        topic: str,
        proposals: list[Proposal],
        voter_id: str,
        voter_type: str,
    ) -> Vote | None:
        """Handle a vote request.

        Args:
            topic: Topic being voted on
            proposals: All proposals for consideration
            voter_id: ID of the voter
            voter_type: Type of the voter

        Returns:
            Vote if the voter participates, None to abstain
        """
        ...


class MARP:
    """Multi-Agent Reasoning Protocol for consensus decisions.

    MARP coordinates multiple agents to reach consensus on complex
    decisions through a structured proposal and voting mechanism.

    Features:
    - Multiple consensus strategies (majority, confidence-weighted, expertise-weighted)
    - Configurable quorum and timeout settings
    - Tie-breaking mechanisms
    - Human escalation for unresolved conflicts
    - Full audit trail for debugging and analysis

    Attributes:
        a2a: Optional A2A protocol for inter-agent communication
        proposals: All proposals indexed by topic
        votes: All votes indexed by topic
        strategy: Consensus strategy to use
        quorum_threshold: Minimum participation rate (0.0 to 1.0)
        consensus_threshold: Minimum support ratio for consensus (0.0 to 1.0)
    """

    def __init__(
        self,
        a2a_protocol: A2AProtocol | None = None,
        strategy: ConsensusStrategy | None = None,
        quorum_threshold: float = 0.5,
        consensus_threshold: float = 0.6,
    ):
        """Initialize MARP.

        Args:
            a2a_protocol: Optional A2A protocol for distributed communication
            strategy: Consensus strategy (defaults to ConfidenceWeighted)
            quorum_threshold: Minimum participation for valid resolution
            consensus_threshold: Minimum support ratio for consensus
        """
        self.a2a = a2a_protocol
        self._strategy = strategy
        self.quorum_threshold = quorum_threshold
        self.consensus_threshold = consensus_threshold

        # Storage
        self.proposals: dict[str, list[Proposal]] = {}
        self.votes: dict[str, list[Vote]] = {}
        self._topic_start_times: dict[str, datetime] = {}
        self._registered_voters: dict[str, set[str]] = {}  # topic -> set of voter_ids
        self._vote_callbacks: dict[str, VoteRequestCallback] = {}  # voter_id -> callback

        self.log = logger.bind(component="marp")

    @property
    def strategy(self) -> ConsensusStrategy:
        """Get the consensus strategy, lazy-loading default if needed."""
        if self._strategy is None:
            from .consensus import ConfidenceWeighted
            self._strategy = ConfidenceWeighted()
        return self._strategy

    def register_voter(
        self,
        voter_id: str,
        callback: VoteRequestCallback,
        topics: list[str] | None = None,
    ) -> None:
        """Register a voter with their callback for vote requests.

        Args:
            voter_id: Unique identifier for the voter
            callback: Async callback to handle vote requests
            topics: Optional list of topics to register for (None = all topics)
        """
        self._vote_callbacks[voter_id] = callback
        self.log.info("Registered voter", voter_id=voter_id, topics=topics)

    def unregister_voter(self, voter_id: str) -> None:
        """Unregister a voter.

        Args:
            voter_id: ID of voter to unregister
        """
        if voter_id in self._vote_callbacks:
            del self._vote_callbacks[voter_id]
            self.log.info("Unregistered voter", voter_id=voter_id)

    async def propose(
        self,
        topic: str,
        agent_id: str,
        agent_type: str,
        solution: dict[str, Any],
        confidence: float,
        reasoning: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a proposal for consideration.

        Args:
            topic: Topic/issue being addressed
            agent_id: ID of the proposing agent
            agent_type: Type of the proposing agent
            solution: The proposed solution
            confidence: Agent's confidence in the solution (0.0 to 1.0)
            reasoning: Explanation of the proposal
            metadata: Additional metadata

        Returns:
            Unique proposal ID

        Raises:
            ValueError: If confidence is out of range
        """
        proposal_id = str(uuid.uuid4())

        proposal = Proposal(
            proposal_id=proposal_id,
            agent_id=agent_id,
            agent_type=agent_type,
            solution=solution,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata or {},
        )

        # Store proposal
        if topic not in self.proposals:
            self.proposals[topic] = []
            self._topic_start_times[topic] = datetime.now(UTC)

        self.proposals[topic].append(proposal)

        self.log.info(
            "Proposal submitted",
            topic=topic,
            proposal_id=proposal_id,
            agent_id=agent_id,
            agent_type=agent_type,
            confidence=confidence,
        )

        # Broadcast via A2A if available
        if self.a2a:
            await self.a2a.broadcast(
                topic=f"marp.proposal.{topic}",
                message={
                    "event": "proposal_submitted",
                    "proposal": proposal.to_dict(),
                }
            )

        return proposal_id

    async def vote(
        self,
        topic: str,
        proposal_id: str,
        voter_id: str,
        voter_type: str,
        vote_type: VoteType,
        confidence: float,
        reasoning: str,
        expertise_score: float = 0.5,
    ) -> str:
        """Cast a vote on a proposal.

        Args:
            topic: Topic being voted on
            proposal_id: ID of the proposal
            voter_id: ID of the voting agent
            voter_type: Type of the voting agent
            vote_type: Type of vote
            confidence: Confidence in the vote
            reasoning: Explanation of the vote
            expertise_score: Voter's domain expertise

        Returns:
            Unique vote ID
        """
        vote_id = str(uuid.uuid4())

        vote = Vote(
            vote_id=vote_id,
            voter_id=voter_id,
            voter_type=voter_type,
            proposal_id=proposal_id,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=reasoning,
            expertise_score=expertise_score,
        )

        if topic not in self.votes:
            self.votes[topic] = []

        self.votes[topic].append(vote)

        self.log.info(
            "Vote cast",
            topic=topic,
            vote_id=vote_id,
            proposal_id=proposal_id,
            voter_id=voter_id,
            vote_type=vote_type.value,
            confidence=confidence,
        )

        # Broadcast via A2A if available
        if self.a2a:
            await self.a2a.broadcast(
                topic=f"marp.vote.{topic}",
                message={
                    "event": "vote_cast",
                    "vote": vote.to_dict(),
                }
            )

        return vote_id

    async def request_votes(
        self,
        topic: str,
        timeout: float = 30.0,
        min_votes: int | None = None,
    ) -> list[Vote]:
        """Request votes from registered agents.

        Args:
            topic: Topic to request votes for
            timeout: Maximum time to wait for votes (seconds)
            min_votes: Minimum votes needed (None = quorum based)

        Returns:
            List of votes received
        """
        if topic not in self.proposals:
            self.log.warning("No proposals for topic", topic=topic)
            return []

        proposals = self.proposals[topic]

        self.log.info(
            "Requesting votes",
            topic=topic,
            num_proposals=len(proposals),
            timeout=timeout,
            registered_voters=len(self._vote_callbacks),
        )

        # Request votes from all registered callbacks
        vote_tasks = []
        for voter_id, callback in self._vote_callbacks.items():
            # Skip if voter already voted
            existing_votes = self.votes.get(topic, [])
            if any(v.voter_id == voter_id for v in existing_votes):
                continue

            task = asyncio.create_task(
                self._request_vote_with_timeout(
                    callback=callback,
                    topic=topic,
                    proposals=proposals,
                    voter_id=voter_id,
                    timeout=timeout,
                )
            )
            vote_tasks.append(task)

        # Also request via A2A if available
        if self.a2a:
            await self.a2a.broadcast(
                topic=f"marp.vote_request.{topic}",
                message={
                    "event": "vote_requested",
                    "topic": topic,
                    "proposals": [p.to_dict() for p in proposals],
                    "timeout": timeout,
                }
            )

        # Wait for all vote tasks
        if vote_tasks:
            results = await asyncio.gather(*vote_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Vote):
                    # Vote was cast via callback, already stored
                    pass
                elif isinstance(result, Exception):
                    self.log.warning("Vote task failed", error=str(result))

        return self.votes.get(topic, [])

    async def _request_vote_with_timeout(
        self,
        callback: VoteRequestCallback,
        topic: str,
        proposals: list[Proposal],
        voter_id: str,
        timeout: float,
    ) -> Vote | None:
        """Request a vote with timeout handling.

        Args:
            callback: Vote callback function
            topic: Topic being voted on
            proposals: Proposals to consider
            voter_id: ID of the voter
            timeout: Timeout in seconds

        Returns:
            Vote if received, None otherwise
        """
        try:
            # Get voter type from an existing proposal or default
            voter_type = "unknown"

            vote = await asyncio.wait_for(
                callback(topic, proposals, voter_id, voter_type),
                timeout=timeout,
            )

            if vote:
                # Store the vote
                if topic not in self.votes:
                    self.votes[topic] = []
                self.votes[topic].append(vote)
                return vote

        except asyncio.TimeoutError:
            self.log.warning(
                "Vote request timed out",
                voter_id=voter_id,
                topic=topic,
                timeout=timeout,
            )
        except Exception as e:
            self.log.error(
                "Vote request failed",
                voter_id=voter_id,
                topic=topic,
                error=str(e),
            )

        return None

    async def resolve(
        self,
        topic: str,
        require_quorum: bool = True,
    ) -> Resolution:
        """Aggregate votes and determine winning proposal.

        Uses the configured consensus strategy to determine the winner.

        Args:
            topic: Topic to resolve
            require_quorum: Whether to require minimum participation

        Returns:
            Resolution with winning proposal and audit trail
        """
        proposals = self.proposals.get(topic, [])
        votes = self.votes.get(topic, [])

        if not proposals:
            return Resolution(
                topic=topic,
                status=ResolutionStatus.NO_CONSENSUS,
                reasoning="No proposals submitted for this topic",
            )

        # Calculate resolution time
        start_time = self._topic_start_times.get(topic, datetime.now(UTC))
        resolution_time = datetime.now(UTC) - start_time

        # Calculate participation
        unique_voters = len(set(v.voter_id for v in votes))
        eligible_voters = len(self._vote_callbacks) or 1  # Avoid division by zero
        participation_rate = unique_voters / eligible_voters

        # Check quorum
        if require_quorum and participation_rate < self.quorum_threshold:
            return Resolution(
                topic=topic,
                status=ResolutionStatus.NO_CONSENSUS,
                all_proposals=proposals,
                all_votes=votes,
                participation_rate=participation_rate,
                resolution_time=resolution_time,
                reasoning=f"Quorum not met: {participation_rate:.1%} < {self.quorum_threshold:.1%}",
            )

        # Use strategy to resolve
        try:
            winning_proposal, scores = self.strategy.resolve(proposals, votes)
        except TieError as e:
            return Resolution(
                topic=topic,
                status=ResolutionStatus.TIE,
                all_proposals=proposals,
                all_votes=votes,
                participation_rate=participation_rate,
                resolution_time=resolution_time,
                reasoning=str(e),
                audit_trail={"tied_proposals": [p.proposal_id for p in e.tied_proposals]},
            )

        # Calculate support/opposition
        total_support = scores.get("total_support", 0.0)
        total_opposition = scores.get("total_opposition", 0.0)
        total_votes = total_support + total_opposition or 1  # Avoid division by zero

        # Check if consensus threshold met
        support_ratio = total_support / total_votes
        consensus_reached = support_ratio >= self.consensus_threshold

        status = (
            ResolutionStatus.CONSENSUS_REACHED
            if consensus_reached
            else ResolutionStatus.NO_CONSENSUS
        )

        resolution = Resolution(
            topic=topic,
            status=status,
            winning_proposal=winning_proposal if consensus_reached else None,
            all_proposals=proposals,
            all_votes=votes,
            total_support=total_support,
            total_opposition=total_opposition,
            participation_rate=participation_rate,
            resolution_time=resolution_time,
            reasoning=scores.get("reasoning", ""),
            audit_trail={
                "strategy": self.strategy.__class__.__name__,
                "scores": scores,
                "consensus_threshold": self.consensus_threshold,
                "support_ratio": support_ratio,
            },
        )

        self.log.info(
            "Topic resolved",
            topic=topic,
            status=status.value,
            winning_proposal=winning_proposal.proposal_id if winning_proposal else None,
            support_ratio=support_ratio,
            participation_rate=participation_rate,
        )

        # Broadcast resolution via A2A
        if self.a2a:
            await self.a2a.broadcast(
                topic=f"marp.resolution.{topic}",
                message={
                    "event": "topic_resolved",
                    "resolution": resolution.to_dict(),
                }
            )

        return resolution

    def get_proposals(self, topic: str) -> list[Proposal]:
        """Get all proposals for a topic.

        Args:
            topic: Topic to get proposals for

        Returns:
            List of proposals
        """
        return self.proposals.get(topic, [])

    def get_votes(self, topic: str) -> list[Vote]:
        """Get all votes for a topic.

        Args:
            topic: Topic to get votes for

        Returns:
            List of votes
        """
        return self.votes.get(topic, [])

    def clear_topic(self, topic: str) -> None:
        """Clear all data for a topic.

        Args:
            topic: Topic to clear
        """
        if topic in self.proposals:
            del self.proposals[topic]
        if topic in self.votes:
            del self.votes[topic]
        if topic in self._topic_start_times:
            del self._topic_start_times[topic]

        self.log.info("Topic cleared", topic=topic)


class TieError(Exception):
    """Raised when there's a tie between proposals."""

    def __init__(self, message: str, tied_proposals: list[Proposal]):
        super().__init__(message)
        self.tied_proposals = tied_proposals


class ConsensusStrategy(ABC):
    """Abstract base class for consensus strategies.

    Consensus strategies determine how votes are aggregated and
    which proposal wins based on the voting results.
    """

    @abstractmethod
    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve proposals based on votes.

        Args:
            proposals: List of proposals to consider
            votes: List of votes cast

        Returns:
            Tuple of (winning_proposal, scores_dict)

        Raises:
            TieError: If there's an unresolvable tie
        """
        pass
