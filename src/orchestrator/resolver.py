"""Conflict resolver for Multi-Agent Reasoning Protocol.

RAP-235: Provides mechanisms for resolving conflicts when MARP
cannot reach consensus through normal voting.

Resolution strategies:
1. Tie resolution - Multiple proposals with equal support
2. Human escalation - When automated resolution isn't appropriate
3. Time-based - Prefer earlier/later proposals
4. Fallback - Use default or most conservative option

The resolver also maintains a complete audit trail for debugging
and compliance purposes.

Example usage:
    from src.orchestrator.resolver import ConflictResolver
    from src.orchestrator.marp import MARP, Resolution, ResolutionStatus

    resolver = ConflictResolver()
    marp = MARP()

    # ... proposals and voting ...

    resolution = await marp.resolve("topic")

    if resolution.status == ResolutionStatus.TIE:
        # Attempt to break the tie
        resolved = await resolver.resolve_tie(resolution.all_proposals)
        if resolved:
            print(f"Tie broken: {resolved.proposal_id}")
        else:
            # Escalate to human
            await resolver.escalate_to_human("topic", resolution.all_proposals)

    # Create audit trail
    audit = resolver.create_audit_trail("topic", resolution)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol

import structlog

from .marp import Proposal, Resolution, ResolutionStatus, Vote, VoteType

if TYPE_CHECKING:
    from .marp import MARP

logger = structlog.get_logger()


class TieBreakStrategy(str, Enum):
    """Strategies for breaking ties between proposals."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    EARLIEST_TIMESTAMP = "earliest_timestamp"
    LATEST_TIMESTAMP = "latest_timestamp"
    MOST_EXPERIENCED = "most_experienced"
    RANDOM = "random"
    FALLBACK_TO_FIRST = "fallback_to_first"
    HUMAN_DECISION = "human_decision"


class EscalationLevel(str, Enum):
    """Levels of human escalation."""
    INFO = "info"  # Informational only
    WARNING = "warning"  # Needs attention
    CRITICAL = "critical"  # Requires immediate decision
    BLOCKER = "blocker"  # Blocks further progress


@dataclass
class EscalationRequest:
    """Request for human intervention.

    Attributes:
        request_id: Unique identifier
        topic: Topic that needs resolution
        proposals: Proposals that are in conflict
        level: Escalation urgency level
        reason: Why escalation is needed
        context: Additional context for the human
        created_at: When the escalation was created
        resolved_at: When resolved (None if pending)
        resolution: Human's decision (None if pending)
        resolver_id: ID of human who resolved (None if pending)
    """
    request_id: str
    topic: str
    proposals: list[Proposal]
    level: EscalationLevel
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    resolution: Proposal | None = None
    resolver_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "topic": self.topic,
            "proposals": [p.to_dict() for p in self.proposals],
            "level": self.level.value,
            "reason": self.reason,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "resolver_id": self.resolver_id,
        }


@dataclass
class AuditEntry:
    """Entry in the audit trail.

    Attributes:
        entry_id: Unique identifier
        topic: Topic being audited
        timestamp: When this entry was created
        event_type: Type of event (proposal, vote, resolution, escalation)
        actor_id: Who triggered the event
        actor_type: Type of actor (agent, human, system)
        details: Event-specific details
        resolution: Final resolution (if this is a resolution event)
    """
    entry_id: str
    topic: str
    timestamp: datetime
    event_type: str
    actor_id: str
    actor_type: str
    details: dict[str, Any]
    resolution: Resolution | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "topic": self.topic,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "details": self.details,
            "resolution": self.resolution.to_dict() if self.resolution else None,
        }


class EscalationHandler(Protocol):
    """Protocol for escalation handlers.

    Implementations handle sending escalations to the appropriate
    human reviewers (Slack, email, dashboard, etc.)
    """

    async def send_escalation(
        self,
        request: EscalationRequest,
    ) -> bool:
        """Send an escalation request to human reviewers.

        Args:
            request: The escalation request

        Returns:
            True if successfully sent, False otherwise
        """
        ...

    async def check_resolution(
        self,
        request_id: str,
    ) -> EscalationRequest | None:
        """Check if an escalation has been resolved.

        Args:
            request_id: ID of the escalation request

        Returns:
            Updated request if resolved, None if still pending
        """
        ...


class ConflictResolver:
    """Resolves conflicts when MARP consensus cannot be reached.

    Provides multiple strategies for breaking ties and handling
    situations where normal voting doesn't produce a clear winner.

    The resolver maintains a complete audit trail for debugging,
    compliance, and learning purposes.
    """

    def __init__(
        self,
        default_tie_break: TieBreakStrategy = TieBreakStrategy.HIGHEST_CONFIDENCE,
        escalation_handler: EscalationHandler | None = None,
        auto_escalate_after_retries: int = 3,
    ):
        """Initialize conflict resolver.

        Args:
            default_tie_break: Default strategy for breaking ties
            escalation_handler: Handler for human escalations
            auto_escalate_after_retries: Auto-escalate after this many failed resolutions
        """
        self.default_tie_break = default_tie_break
        self.escalation_handler = escalation_handler
        self.auto_escalate_after_retries = auto_escalate_after_retries

        # Storage
        self._audit_trail: dict[str, list[AuditEntry]] = {}
        self._escalations: dict[str, EscalationRequest] = {}
        self._resolution_attempts: dict[str, int] = {}

        self.log = logger.bind(component="conflict_resolver")

    async def resolve_tie(
        self,
        proposals: list[Proposal],
        votes: list[Vote] | None = None,
        strategy: TieBreakStrategy | None = None,
    ) -> Proposal | None:
        """Attempt to resolve a tie between proposals.

        Args:
            proposals: Tied proposals
            votes: Optional votes for additional context
            strategy: Strategy to use (defaults to configured default)

        Returns:
            Winning proposal if tie broken, None if escalation needed
        """
        if not proposals:
            return None

        if len(proposals) == 1:
            return proposals[0]

        strategy = strategy or self.default_tie_break

        self.log.info(
            "Attempting to resolve tie",
            num_proposals=len(proposals),
            strategy=strategy.value,
        )

        winner: Proposal | None = None

        if strategy == TieBreakStrategy.HIGHEST_CONFIDENCE:
            winner = self._tie_break_by_confidence(proposals)

        elif strategy == TieBreakStrategy.EARLIEST_TIMESTAMP:
            winner = self._tie_break_by_timestamp(proposals, earliest=True)

        elif strategy == TieBreakStrategy.LATEST_TIMESTAMP:
            winner = self._tie_break_by_timestamp(proposals, earliest=False)

        elif strategy == TieBreakStrategy.MOST_EXPERIENCED:
            winner = self._tie_break_by_experience(proposals, votes or [])

        elif strategy == TieBreakStrategy.RANDOM:
            winner = self._tie_break_random(proposals)

        elif strategy == TieBreakStrategy.FALLBACK_TO_FIRST:
            winner = proposals[0]

        elif strategy == TieBreakStrategy.HUMAN_DECISION:
            # Don't auto-resolve, return None to trigger escalation
            return None

        if winner:
            self.log.info(
                "Tie resolved",
                strategy=strategy.value,
                winner_id=winner.proposal_id,
                winner_agent=winner.agent_id,
            )

        return winner

    def _tie_break_by_confidence(self, proposals: list[Proposal]) -> Proposal:
        """Break tie by selecting proposal with highest confidence."""
        return max(proposals, key=lambda p: p.confidence)

    def _tie_break_by_timestamp(
        self,
        proposals: list[Proposal],
        earliest: bool = True,
    ) -> Proposal:
        """Break tie by timestamp (earliest or latest)."""
        sorted_proposals = sorted(proposals, key=lambda p: p.timestamp)
        return sorted_proposals[0] if earliest else sorted_proposals[-1]

    def _tie_break_by_experience(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> Proposal:
        """Break tie by proposer's apparent experience.

        Experience is inferred from:
        1. Number of votes received for their proposal
        2. Average confidence of supporting votes
        3. Proposal confidence
        """
        proposal_scores: dict[str, float] = {}

        for proposal in proposals:
            # Get votes for this proposal
            proposal_votes = [v for v in votes if v.proposal_id == proposal.proposal_id]

            # Calculate experience score
            num_votes = len(proposal_votes)
            avg_voter_confidence = (
                sum(v.confidence for v in proposal_votes) / num_votes
                if num_votes > 0
                else 0.5
            )

            # Combined score
            score = (
                proposal.confidence * 0.4 +  # Proposer's confidence
                avg_voter_confidence * 0.3 +  # Voter confidence
                min(num_votes / 10, 1.0) * 0.3  # Engagement (capped)
            )

            proposal_scores[proposal.proposal_id] = score

        # Return proposal with highest score
        best_proposal_id = max(proposal_scores, key=proposal_scores.get)  # type: ignore
        return next(p for p in proposals if p.proposal_id == best_proposal_id)

    def _tie_break_random(self, proposals: list[Proposal]) -> Proposal:
        """Break tie randomly (using deterministic hash for reproducibility)."""
        import hashlib

        # Create deterministic but seemingly random selection
        combined_ids = "".join(sorted(p.proposal_id for p in proposals))
        hash_val = int(hashlib.sha256(combined_ids.encode()).hexdigest(), 16)
        index = hash_val % len(proposals)

        return sorted(proposals, key=lambda p: p.proposal_id)[index]

    async def escalate_to_human(
        self,
        topic: str,
        proposals: list[Proposal],
        level: EscalationLevel = EscalationLevel.WARNING,
        reason: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> EscalationRequest:
        """Escalate to human when automated resolution fails.

        Args:
            topic: Topic that needs resolution
            proposals: Proposals to choose from
            level: Urgency level
            reason: Why escalation is needed
            context: Additional context

        Returns:
            EscalationRequest object

        Note:
            If no escalation_handler is configured, the request is
            stored but not sent. Use check_escalation() to poll.
        """
        request_id = str(uuid.uuid4())

        request = EscalationRequest(
            request_id=request_id,
            topic=topic,
            proposals=proposals,
            level=level,
            reason=reason or f"Unable to reach consensus on {topic}",
            context=context or {},
        )

        self._escalations[request_id] = request

        self.log.warning(
            "Escalating to human",
            request_id=request_id,
            topic=topic,
            level=level.value,
            num_proposals=len(proposals),
        )

        # Record in audit trail
        self._record_audit(
            topic=topic,
            event_type="escalation",
            actor_id="system",
            actor_type="system",
            details={
                "request_id": request_id,
                "level": level.value,
                "reason": request.reason,
                "proposals": [p.proposal_id for p in proposals],
            },
        )

        # Send via handler if available
        if self.escalation_handler:
            try:
                sent = await self.escalation_handler.send_escalation(request)
                if sent:
                    self.log.info("Escalation sent", request_id=request_id)
                else:
                    self.log.warning("Escalation handler returned False", request_id=request_id)
            except Exception as e:
                self.log.error(
                    "Failed to send escalation",
                    request_id=request_id,
                    error=str(e),
                )

        return request

    async def resolve_escalation(
        self,
        request_id: str,
        selected_proposal: Proposal,
        resolver_id: str,
        reason: str | None = None,
    ) -> EscalationRequest:
        """Record human resolution of an escalation.

        Args:
            request_id: ID of the escalation request
            selected_proposal: Proposal selected by human
            resolver_id: ID of the human who resolved it
            reason: Optional reason for the decision

        Returns:
            Updated escalation request

        Raises:
            KeyError: If request_id not found
        """
        if request_id not in self._escalations:
            raise KeyError(f"Escalation request not found: {request_id}")

        request = self._escalations[request_id]
        request.resolved_at = datetime.now(UTC)
        request.resolution = selected_proposal
        request.resolver_id = resolver_id

        self.log.info(
            "Escalation resolved by human",
            request_id=request_id,
            resolver_id=resolver_id,
            selected_proposal=selected_proposal.proposal_id,
        )

        # Record in audit trail
        self._record_audit(
            topic=request.topic,
            event_type="escalation_resolved",
            actor_id=resolver_id,
            actor_type="human",
            details={
                "request_id": request_id,
                "selected_proposal": selected_proposal.proposal_id,
                "reason": reason,
            },
        )

        return request

    def get_escalation(self, request_id: str) -> EscalationRequest | None:
        """Get an escalation request by ID.

        Args:
            request_id: ID of the escalation

        Returns:
            EscalationRequest if found, None otherwise
        """
        return self._escalations.get(request_id)

    def get_pending_escalations(
        self,
        topic: str | None = None,
    ) -> list[EscalationRequest]:
        """Get all pending (unresolved) escalations.

        Args:
            topic: Optional topic filter

        Returns:
            List of pending escalation requests
        """
        pending = [
            req for req in self._escalations.values()
            if req.resolved_at is None
        ]

        if topic:
            pending = [req for req in pending if req.topic == topic]

        return pending

    def create_audit_trail(
        self,
        topic: str,
        resolution: Resolution,
    ) -> AuditEntry:
        """Create an audit entry for a resolution.

        Records the complete resolution including all proposals,
        votes, and the final decision for compliance and debugging.

        Args:
            topic: Topic that was resolved
            resolution: The resolution object

        Returns:
            AuditEntry for the resolution
        """
        entry = self._record_audit(
            topic=topic,
            event_type="resolution",
            actor_id="marp",
            actor_type="system",
            details={
                "status": resolution.status.value,
                "winning_proposal": (
                    resolution.winning_proposal.proposal_id
                    if resolution.winning_proposal
                    else None
                ),
                "total_proposals": len(resolution.all_proposals),
                "total_votes": len(resolution.all_votes),
                "total_support": resolution.total_support,
                "total_opposition": resolution.total_opposition,
                "participation_rate": resolution.participation_rate,
                "resolution_time": str(resolution.resolution_time),
                "reasoning": resolution.reasoning,
            },
            resolution=resolution,
        )

        self.log.info(
            "Audit trail created",
            topic=topic,
            entry_id=entry.entry_id,
            status=resolution.status.value,
        )

        return entry

    def _record_audit(
        self,
        topic: str,
        event_type: str,
        actor_id: str,
        actor_type: str,
        details: dict[str, Any],
        resolution: Resolution | None = None,
    ) -> AuditEntry:
        """Record an audit entry.

        Args:
            topic: Topic for the entry
            event_type: Type of event
            actor_id: Who triggered the event
            actor_type: Type of actor
            details: Event details
            resolution: Optional resolution object

        Returns:
            Created AuditEntry
        """
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            topic=topic,
            timestamp=datetime.now(UTC),
            event_type=event_type,
            actor_id=actor_id,
            actor_type=actor_type,
            details=details,
            resolution=resolution,
        )

        if topic not in self._audit_trail:
            self._audit_trail[topic] = []

        self._audit_trail[topic].append(entry)

        return entry

    def get_audit_trail(
        self,
        topic: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[AuditEntry]:
        """Get audit trail entries.

        Args:
            topic: Optional topic filter
            event_type: Optional event type filter
            since: Only entries after this time
            limit: Maximum entries to return

        Returns:
            List of audit entries matching criteria
        """
        if topic:
            entries = self._audit_trail.get(topic, [])
        else:
            entries = [
                entry
                for topic_entries in self._audit_trail.values()
                for entry in topic_entries
            ]

        # Apply filters
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        # Sort by timestamp (newest first)
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        if limit:
            entries = entries[:limit]

        return entries

    def export_audit_trail(
        self,
        topic: str | None = None,
        format: str = "dict",
    ) -> list[dict[str, Any]] | str:
        """Export audit trail for external storage or analysis.

        Args:
            topic: Optional topic filter
            format: Output format ("dict" or "json")

        Returns:
            Audit trail as list of dicts or JSON string
        """
        entries = self.get_audit_trail(topic=topic)
        data = [entry.to_dict() for entry in entries]

        if format == "json":
            import json
            return json.dumps(data, indent=2)

        return data


class SlackEscalationHandler:
    """Example escalation handler that sends to Slack.

    This is a reference implementation. In production, configure
    with your Slack workspace and channel settings.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        mention_users: list[str] | None = None,
    ):
        """Initialize Slack handler.

        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override
            mention_users: User IDs to mention on escalations
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.mention_users = mention_users or []
        self._pending: dict[str, EscalationRequest] = {}
        self.log = logger.bind(handler="slack_escalation")

    async def send_escalation(self, request: EscalationRequest) -> bool:
        """Send escalation to Slack.

        Args:
            request: Escalation request

        Returns:
            True if sent successfully
        """
        import httpx

        # Build message
        level_emoji = {
            EscalationLevel.INFO: ":information_source:",
            EscalationLevel.WARNING: ":warning:",
            EscalationLevel.CRITICAL: ":rotating_light:",
            EscalationLevel.BLOCKER: ":no_entry:",
        }

        mentions = " ".join(f"<@{uid}>" for uid in self.mention_users)

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{level_emoji.get(request.level, ':question:')} MARP Escalation: {request.topic}",
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reason:* {request.reason}\n{mentions}",
                }
            },
            {"type": "divider"},
        ]

        # Add proposals
        for i, proposal in enumerate(request.proposals, 1):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Option {i}: {proposal.agent_type}*\n"
                        f"Confidence: {proposal.confidence:.0%}\n"
                        f"```{proposal.reasoning[:300]}```"
                    ),
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": f"Select #{i}"},
                    "value": f"{request.request_id}:{proposal.proposal_id}",
                    "action_id": f"marp_select_{i}",
                },
            })

        payload = {
            "blocks": blocks,
        }

        if self.channel:
            payload["channel"] = self.channel

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0,
                )
                response.raise_for_status()

            self._pending[request.request_id] = request
            return True

        except Exception as e:
            self.log.error("Failed to send Slack escalation", error=str(e))
            return False

    async def check_resolution(
        self,
        request_id: str,
    ) -> EscalationRequest | None:
        """Check if Slack escalation has been resolved.

        Note: This would typically poll a database or webhook endpoint
        that receives Slack interactive component payloads.

        Args:
            request_id: ID of the escalation

        Returns:
            Updated request if resolved, None if pending
        """
        # In real implementation, check database/cache for Slack button clicks
        request = self._pending.get(request_id)
        if request and request.resolved_at:
            return request
        return None


class DashboardEscalationHandler:
    """Escalation handler that records to a dashboard/database.

    For production use with a web dashboard where humans can
    review and resolve escalations.
    """

    def __init__(self, db_url: str | None = None):
        """Initialize dashboard handler.

        Args:
            db_url: Database connection URL (optional, uses in-memory storage if None)
        """
        self.db_url = db_url
        self._storage: dict[str, EscalationRequest] = {}
        self.log = logger.bind(handler="dashboard_escalation")

    async def send_escalation(self, request: EscalationRequest) -> bool:
        """Store escalation for dashboard display.

        Args:
            request: Escalation request

        Returns:
            True (always succeeds for in-memory storage)
        """
        self._storage[request.request_id] = request
        self.log.info(
            "Escalation stored for dashboard",
            request_id=request.request_id,
            topic=request.topic,
        )
        return True

    async def check_resolution(
        self,
        request_id: str,
    ) -> EscalationRequest | None:
        """Check if escalation has been resolved via dashboard.

        Args:
            request_id: ID of the escalation

        Returns:
            Updated request if resolved, None if pending
        """
        request = self._storage.get(request_id)
        if request and request.resolved_at:
            return request
        return None

    def resolve_via_dashboard(
        self,
        request_id: str,
        selected_proposal_id: str,
        resolver_id: str,
    ) -> bool:
        """Called by dashboard to resolve an escalation.

        Args:
            request_id: ID of the escalation
            selected_proposal_id: ID of the selected proposal
            resolver_id: ID of the human resolver

        Returns:
            True if resolved, False if not found
        """
        request = self._storage.get(request_id)
        if not request:
            return False

        # Find the selected proposal
        selected = next(
            (p for p in request.proposals if p.proposal_id == selected_proposal_id),
            None,
        )
        if not selected:
            return False

        request.resolved_at = datetime.now(UTC)
        request.resolution = selected
        request.resolver_id = resolver_id

        self.log.info(
            "Escalation resolved via dashboard",
            request_id=request_id,
            resolver_id=resolver_id,
            selected_proposal=selected_proposal_id,
        )

        return True

    def get_pending_for_dashboard(self) -> list[dict[str, Any]]:
        """Get pending escalations formatted for dashboard display.

        Returns:
            List of pending escalation data
        """
        pending = [
            req for req in self._storage.values()
            if req.resolved_at is None
        ]

        return [
            {
                "request_id": req.request_id,
                "topic": req.topic,
                "level": req.level.value,
                "reason": req.reason,
                "created_at": req.created_at.isoformat(),
                "proposals": [
                    {
                        "proposal_id": p.proposal_id,
                        "agent_type": p.agent_type,
                        "confidence": p.confidence,
                        "reasoning": p.reasoning,
                        "solution": p.solution,
                    }
                    for p in req.proposals
                ],
            }
            for req in pending
        ]
