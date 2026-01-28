"""Consensus strategies for Multi-Agent Reasoning Protocol.

RAP-235: Provides different strategies for aggregating votes and
determining winning proposals in multi-agent decision making.

Strategies:
    - MajorityVoting: Simple majority wins
    - ConfidenceWeighted: Weight votes by confidence scores
    - ExpertiseWeighted: Weight votes by agent expertise in the domain
    - SuperMajority: Require supermajority (2/3) for consensus
    - BordaCount: Rank-based voting using Borda count method
    - QuadraticVoting: Square root weighting to prevent dominance

Example usage:
    from src.orchestrator.consensus import (
        MajorityVoting,
        ConfidenceWeighted,
        ExpertiseWeighted,
    )
    from src.orchestrator.marp import MARP

    # Simple majority voting
    marp_simple = MARP(strategy=MajorityVoting())

    # Confidence-weighted for uncertain decisions
    marp_confidence = MARP(strategy=ConfidenceWeighted())

    # Expertise-weighted for domain-specific decisions
    marp_expert = MARP(strategy=ExpertiseWeighted(
        expertise_domains={
            "self_healer": ["selector_fix", "assertion_fix"],
            "ui_tester": ["visual_change", "layout_issue"],
            "api_tester": ["schema_change", "endpoint_fix"],
            "code_analyzer": ["refactoring", "dependency_update"],
        }
    ))
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import structlog

from .marp import ConsensusStrategy, Proposal, TieError, Vote, VoteType

logger = structlog.get_logger()


class MajorityVoting(ConsensusStrategy):
    """Simple majority wins strategy.

    Each vote counts equally. The proposal with the most support
    votes wins. Ties are broken by proposal confidence.

    Suitable for:
    - Simple decisions with clear alternatives
    - Situations where all voters have equal authority
    - Quick decisions that don't require weighted input
    """

    def __init__(self, tie_breaker: str = "confidence"):
        """Initialize majority voting strategy.

        Args:
            tie_breaker: How to break ties ("confidence", "timestamp", "none")
        """
        self.tie_breaker = tie_breaker
        self.log = logger.bind(strategy="majority_voting")

    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve using simple majority voting.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Tuple of (winning_proposal, scores)

        Raises:
            TieError: If tie cannot be broken
        """
        if not proposals:
            raise ValueError("No proposals to resolve")

        # If only one proposal, it wins by default
        if len(proposals) == 1:
            proposal = proposals[0]
            support_votes = [v for v in votes if v.vote_type == VoteType.SUPPORT]
            oppose_votes = [v for v in votes if v.vote_type == VoteType.OPPOSE]

            return proposal, {
                "total_support": len(support_votes),
                "total_opposition": len(oppose_votes),
                "reasoning": "Single proposal - wins by default",
                "vote_breakdown": {
                    proposal.proposal_id: {
                        "support": len(support_votes),
                        "oppose": len(oppose_votes),
                    }
                },
            }

        # Count votes per proposal
        vote_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"support": 0, "oppose": 0, "abstain": 0}
        )

        for vote in votes:
            vote_counts[vote.proposal_id][vote.vote_type.value] += 1

        # Calculate net support for each proposal
        proposal_scores: list[tuple[Proposal, int]] = []
        for proposal in proposals:
            counts = vote_counts[proposal.proposal_id]
            net_support = counts["support"] - counts["oppose"]
            proposal_scores.append((proposal, net_support))

        # Sort by net support (highest first)
        proposal_scores.sort(key=lambda x: x[1], reverse=True)

        # Check for tie
        if len(proposal_scores) > 1 and proposal_scores[0][1] == proposal_scores[1][1]:
            tied = [p for p, s in proposal_scores if s == proposal_scores[0][1]]

            if self.tie_breaker == "confidence":
                # Break tie by proposal confidence
                tied.sort(key=lambda p: p.confidence, reverse=True)
                winner = tied[0]
                reasoning = f"Tie broken by proposal confidence ({winner.confidence:.2f})"
            elif self.tie_breaker == "timestamp":
                # Break tie by earliest proposal
                tied.sort(key=lambda p: p.timestamp)
                winner = tied[0]
                reasoning = "Tie broken by earliest proposal timestamp"
            else:
                raise TieError(
                    f"Tie between {len(tied)} proposals with no tie-breaker",
                    tied,
                )
        else:
            winner = proposal_scores[0][0]
            reasoning = f"Won with {proposal_scores[0][1]} net support"

        # Calculate totals
        total_support = sum(vote_counts[p.proposal_id]["support"] for p in proposals)
        total_opposition = sum(vote_counts[p.proposal_id]["oppose"] for p in proposals)

        return winner, {
            "total_support": total_support,
            "total_opposition": total_opposition,
            "reasoning": reasoning,
            "vote_breakdown": dict(vote_counts),
            "net_scores": {p.proposal_id: s for p, s in proposal_scores},
        }


class ConfidenceWeighted(ConsensusStrategy):
    """Weight votes by confidence scores.

    Votes with higher confidence have more weight in the decision.
    This is useful when some votes are more certain than others.

    Weight formula: vote_weight = confidence^power

    Suitable for:
    - Uncertain situations where confidence varies
    - Decisions involving AI agents with varying certainty
    - Prioritizing well-reasoned votes over guesses
    """

    def __init__(
        self,
        power: float = 1.5,
        min_confidence: float = 0.1,
        tie_breaker: str = "confidence",
    ):
        """Initialize confidence-weighted strategy.

        Args:
            power: Exponent for confidence weighting (higher = more weight to confident votes)
            min_confidence: Minimum confidence threshold for vote to count
            tie_breaker: How to break ties
        """
        self.power = power
        self.min_confidence = min_confidence
        self.tie_breaker = tie_breaker
        self.log = logger.bind(strategy="confidence_weighted")

    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve using confidence-weighted voting.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Tuple of (winning_proposal, scores)

        Raises:
            TieError: If tie cannot be broken
        """
        if not proposals:
            raise ValueError("No proposals to resolve")

        # Filter votes below minimum confidence
        valid_votes = [v for v in votes if v.confidence >= self.min_confidence]

        if len(proposals) == 1:
            proposal = proposals[0]
            support_weight = sum(
                v.confidence ** self.power
                for v in valid_votes
                if v.vote_type == VoteType.SUPPORT
            )
            oppose_weight = sum(
                v.confidence ** self.power
                for v in valid_votes
                if v.vote_type == VoteType.OPPOSE
            )

            return proposal, {
                "total_support": support_weight,
                "total_opposition": oppose_weight,
                "reasoning": "Single proposal - wins by default",
                "power": self.power,
            }

        # Calculate weighted scores per proposal
        proposal_weights: dict[str, dict[str, float]] = defaultdict(
            lambda: {"support": 0.0, "oppose": 0.0}
        )

        for vote in valid_votes:
            weight = vote.confidence ** self.power
            if vote.vote_type == VoteType.SUPPORT:
                proposal_weights[vote.proposal_id]["support"] += weight
            elif vote.vote_type == VoteType.OPPOSE:
                proposal_weights[vote.proposal_id]["oppose"] += weight

        # Calculate net weighted support
        proposal_scores: list[tuple[Proposal, float]] = []
        for proposal in proposals:
            weights = proposal_weights[proposal.proposal_id]
            net_support = weights["support"] - weights["oppose"]
            proposal_scores.append((proposal, net_support))

        # Sort by net weighted support
        proposal_scores.sort(key=lambda x: x[1], reverse=True)

        # Check for tie (within small epsilon due to floating point)
        epsilon = 0.001
        if len(proposal_scores) > 1:
            top_score = proposal_scores[0][1]
            tied = [p for p, s in proposal_scores if abs(s - top_score) < epsilon]

            if len(tied) > 1:
                if self.tie_breaker == "confidence":
                    tied.sort(key=lambda p: p.confidence, reverse=True)
                    winner = tied[0]
                    reasoning = f"Tie broken by proposal confidence ({winner.confidence:.2f})"
                else:
                    raise TieError(f"Tie between {len(tied)} proposals", tied)
            else:
                winner = proposal_scores[0][0]
                reasoning = f"Won with {proposal_scores[0][1]:.3f} weighted net support"
        else:
            winner = proposal_scores[0][0]
            reasoning = f"Won with {proposal_scores[0][1]:.3f} weighted net support"

        # Calculate totals
        total_support = sum(
            proposal_weights[p.proposal_id]["support"] for p in proposals
        )
        total_opposition = sum(
            proposal_weights[p.proposal_id]["oppose"] for p in proposals
        )

        return winner, {
            "total_support": total_support,
            "total_opposition": total_opposition,
            "reasoning": reasoning,
            "power": self.power,
            "weighted_breakdown": {
                p.proposal_id: dict(proposal_weights[p.proposal_id])
                for p in proposals
            },
            "net_scores": {p.proposal_id: s for p, s in proposal_scores},
            "filtered_votes": len(votes) - len(valid_votes),
        }


class ExpertiseWeighted(ConsensusStrategy):
    """Weight votes by agent expertise in the domain.

    Different agent types have different expertise levels for
    different topics. This strategy gives more weight to agents
    with relevant expertise.

    Suitable for:
    - Domain-specific decisions
    - Multi-agent systems with specialized agents
    - Decisions where expertise matters more than quantity
    """

    def __init__(
        self,
        expertise_domains: dict[str, list[str]] | None = None,
        base_weight: float = 0.5,
        expertise_boost: float = 1.5,
        confidence_factor: float = 0.5,
    ):
        """Initialize expertise-weighted strategy.

        Args:
            expertise_domains: Mapping of agent_type -> list of topics they're expert in
            base_weight: Base weight for non-expert votes
            expertise_boost: Multiplier for expert votes
            confidence_factor: How much to factor in confidence (0 to 1)
        """
        self.expertise_domains = expertise_domains or self._default_expertise_domains()
        self.base_weight = base_weight
        self.expertise_boost = expertise_boost
        self.confidence_factor = confidence_factor
        self.log = logger.bind(strategy="expertise_weighted")

    def _default_expertise_domains(self) -> dict[str, list[str]]:
        """Get default expertise domains for testing agents."""
        return {
            "self_healer": [
                "selector_fix", "assertion_fix", "healing_strategy",
                "test_stability", "flaky_test",
            ],
            "ui_tester": [
                "visual_change", "layout_issue", "interaction",
                "form_validation", "navigation",
            ],
            "api_tester": [
                "schema_change", "endpoint_fix", "authentication",
                "response_validation", "api_contract",
            ],
            "code_analyzer": [
                "refactoring", "dependency_update", "code_structure",
                "test_coverage", "code_quality",
            ],
            "reporter": [
                "test_summary", "report_format", "notification",
                "priority_assessment",
            ],
        }

    def _calculate_expertise(
        self,
        voter_type: str,
        topic: str,
        voter_expertise_score: float,
    ) -> float:
        """Calculate expertise weight for a voter on a topic.

        Args:
            voter_type: Type of the voting agent
            topic: Topic being voted on
            voter_expertise_score: Voter's self-reported expertise

        Returns:
            Expertise weight multiplier
        """
        # Check if voter type has expertise in any topic keywords
        expert_topics = self.expertise_domains.get(voter_type, [])

        is_expert = any(
            expert_topic.lower() in topic.lower() or
            topic.lower() in expert_topic.lower()
            for expert_topic in expert_topics
        )

        if is_expert:
            # Expert gets boost plus their self-reported expertise
            return self.expertise_boost + (voter_expertise_score * 0.5)
        else:
            # Non-expert gets base weight modified by their expertise score
            return self.base_weight + (voter_expertise_score * 0.3)

    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve using expertise-weighted voting.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Tuple of (winning_proposal, scores)

        Raises:
            TieError: If tie cannot be broken
        """
        if not proposals:
            raise ValueError("No proposals to resolve")

        # Infer topic from proposals
        topic = self._infer_topic(proposals)

        if len(proposals) == 1:
            proposal = proposals[0]
            support_weight = 0.0
            oppose_weight = 0.0

            for vote in votes:
                expertise = self._calculate_expertise(
                    vote.voter_type, topic, vote.expertise_score
                )
                # Combine expertise with confidence
                weight = expertise * (
                    1.0 + self.confidence_factor * vote.confidence
                )

                if vote.vote_type == VoteType.SUPPORT:
                    support_weight += weight
                elif vote.vote_type == VoteType.OPPOSE:
                    oppose_weight += weight

            return proposal, {
                "total_support": support_weight,
                "total_opposition": oppose_weight,
                "reasoning": "Single proposal - wins by default",
                "inferred_topic": topic,
            }

        # Calculate expertise-weighted scores per proposal
        proposal_weights: dict[str, dict[str, float]] = defaultdict(
            lambda: {"support": 0.0, "oppose": 0.0}
        )
        voter_weights: dict[str, float] = {}  # For audit trail

        for vote in votes:
            expertise = self._calculate_expertise(
                vote.voter_type, topic, vote.expertise_score
            )
            weight = expertise * (1.0 + self.confidence_factor * vote.confidence)
            voter_weights[vote.voter_id] = weight

            if vote.vote_type == VoteType.SUPPORT:
                proposal_weights[vote.proposal_id]["support"] += weight
            elif vote.vote_type == VoteType.OPPOSE:
                proposal_weights[vote.proposal_id]["oppose"] += weight

        # Calculate net weighted support
        proposal_scores: list[tuple[Proposal, float]] = []
        for proposal in proposals:
            weights = proposal_weights[proposal.proposal_id]
            net_support = weights["support"] - weights["oppose"]
            proposal_scores.append((proposal, net_support))

        # Sort by net weighted support
        proposal_scores.sort(key=lambda x: x[1], reverse=True)

        # Handle ties
        epsilon = 0.001
        if len(proposal_scores) > 1:
            top_score = proposal_scores[0][1]
            tied = [p for p, s in proposal_scores if abs(s - top_score) < epsilon]

            if len(tied) > 1:
                # Break tie by proposer's expertise
                def proposer_expertise(p: Proposal) -> float:
                    return self._calculate_expertise(p.agent_type, topic, p.confidence)

                tied.sort(key=proposer_expertise, reverse=True)
                winner = tied[0]
                reasoning = "Tie broken by proposer expertise"
            else:
                winner = proposal_scores[0][0]
                reasoning = f"Won with {proposal_scores[0][1]:.3f} expertise-weighted support"
        else:
            winner = proposal_scores[0][0]
            reasoning = f"Won with {proposal_scores[0][1]:.3f} expertise-weighted support"

        # Calculate totals
        total_support = sum(
            proposal_weights[p.proposal_id]["support"] for p in proposals
        )
        total_opposition = sum(
            proposal_weights[p.proposal_id]["oppose"] for p in proposals
        )

        return winner, {
            "total_support": total_support,
            "total_opposition": total_opposition,
            "reasoning": reasoning,
            "inferred_topic": topic,
            "voter_weights": voter_weights,
            "weighted_breakdown": {
                p.proposal_id: dict(proposal_weights[p.proposal_id])
                for p in proposals
            },
            "net_scores": {p.proposal_id: s for p, s in proposal_scores},
        }

    def _infer_topic(self, proposals: list[Proposal]) -> str:
        """Infer topic from proposal metadata and reasoning.

        Args:
            proposals: List of proposals

        Returns:
            Inferred topic string
        """
        # Try to get topic from metadata
        for proposal in proposals:
            if "topic" in proposal.metadata:
                return proposal.metadata["topic"]

        # Otherwise, use the most common words from reasoning
        # (simplified - in practice, use NLP)
        if proposals:
            return proposals[0].reasoning.split()[0].lower()

        return "unknown"


class SuperMajority(ConsensusStrategy):
    """Require supermajority (configurable threshold) for consensus.

    More conservative than simple majority - requires a higher
    threshold for a proposal to be accepted.

    Suitable for:
    - High-stakes decisions
    - Changes that affect multiple components
    - Decisions requiring strong agreement
    """

    def __init__(
        self,
        threshold: float = 0.67,  # 2/3 majority
        confidence_weighted: bool = True,
    ):
        """Initialize supermajority strategy.

        Args:
            threshold: Required support ratio (0.67 = 2/3 majority)
            confidence_weighted: Whether to weight by confidence
        """
        self.threshold = threshold
        self.confidence_weighted = confidence_weighted
        self.log = logger.bind(strategy="super_majority")

    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve requiring supermajority support.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Tuple of (winning_proposal, scores)

        Raises:
            TieError: If no proposal meets threshold
        """
        if not proposals:
            raise ValueError("No proposals to resolve")

        # Calculate support for each proposal
        proposal_support: dict[str, float] = defaultdict(float)
        proposal_opposition: dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for vote in votes:
            weight = vote.confidence if self.confidence_weighted else 1.0
            total_weight += weight

            if vote.vote_type == VoteType.SUPPORT:
                proposal_support[vote.proposal_id] += weight
            elif vote.vote_type == VoteType.OPPOSE:
                proposal_opposition[vote.proposal_id] += weight

        if total_weight == 0:
            # No votes - use proposal with highest confidence
            proposals_sorted = sorted(proposals, key=lambda p: p.confidence, reverse=True)
            return proposals_sorted[0], {
                "total_support": 0,
                "total_opposition": 0,
                "reasoning": "No votes cast - selected highest confidence proposal",
            }

        # Find proposal meeting threshold
        meeting_threshold: list[tuple[Proposal, float]] = []
        for proposal in proposals:
            support = proposal_support[proposal.proposal_id]
            support_ratio = support / total_weight

            if support_ratio >= self.threshold:
                meeting_threshold.append((proposal, support_ratio))

        if not meeting_threshold:
            # No proposal meets threshold - still return highest
            best_proposal = max(
                proposals,
                key=lambda p: proposal_support[p.proposal_id] / total_weight if total_weight else 0,
            )
            best_ratio = proposal_support[best_proposal.proposal_id] / total_weight

            return best_proposal, {
                "total_support": sum(proposal_support.values()),
                "total_opposition": sum(proposal_opposition.values()),
                "reasoning": f"No proposal met {self.threshold:.0%} threshold. Best had {best_ratio:.1%}",
                "threshold": self.threshold,
                "support_ratios": {
                    p.proposal_id: proposal_support[p.proposal_id] / total_weight
                    for p in proposals
                },
            }

        # Sort by support ratio
        meeting_threshold.sort(key=lambda x: x[1], reverse=True)
        winner = meeting_threshold[0][0]

        return winner, {
            "total_support": proposal_support[winner.proposal_id],
            "total_opposition": proposal_opposition[winner.proposal_id],
            "reasoning": f"Won with {meeting_threshold[0][1]:.1%} support (threshold: {self.threshold:.0%})",
            "threshold": self.threshold,
            "support_ratios": {
                p.proposal_id: proposal_support[p.proposal_id] / total_weight
                for p in proposals
            },
        }


class BordaCount(ConsensusStrategy):
    """Rank-based voting using Borda count method.

    Voters rank proposals, and points are assigned based on rank.
    Highest ranked gets most points, lowest gets least.

    Suitable for:
    - Multiple competing proposals
    - Situations where preference ordering matters
    - Finding compromise solutions
    """

    def __init__(self, confidence_weighted: bool = True):
        """Initialize Borda count strategy.

        Args:
            confidence_weighted: Whether to weight rankings by confidence
        """
        self.confidence_weighted = confidence_weighted
        self.log = logger.bind(strategy="borda_count")

    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve using Borda count method.

        Since our votes are per-proposal (not ranked), we derive
        rankings from vote patterns: support = high rank, oppose = low.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Tuple of (winning_proposal, scores)
        """
        if not proposals:
            raise ValueError("No proposals to resolve")

        n = len(proposals)

        # Group votes by voter
        voter_votes: dict[str, list[Vote]] = defaultdict(list)
        for vote in votes:
            voter_votes[vote.voter_id].append(vote)

        # Calculate Borda scores
        borda_scores: dict[str, float] = defaultdict(float)

        for voter_id, voter_vote_list in voter_votes.items():
            # Get confidence for weighting (use average if multiple votes)
            weight = 1.0
            if self.confidence_weighted and voter_vote_list:
                weight = sum(v.confidence for v in voter_vote_list) / len(voter_vote_list)

            # Assign points based on vote type
            # Support = n points, Abstain = n/2 points, Oppose = 0 points
            for vote in voter_vote_list:
                if vote.vote_type == VoteType.SUPPORT:
                    points = n * weight
                elif vote.vote_type == VoteType.ABSTAIN:
                    points = (n / 2) * weight
                else:
                    points = 0

                borda_scores[vote.proposal_id] += points

        # For proposals with no votes, give neutral score
        for proposal in proposals:
            if proposal.proposal_id not in borda_scores:
                borda_scores[proposal.proposal_id] = (n / 2)

        # Sort by score
        proposal_scores = [
            (proposal, borda_scores[proposal.proposal_id])
            for proposal in proposals
        ]
        proposal_scores.sort(key=lambda x: x[1], reverse=True)

        winner = proposal_scores[0][0]
        total_support = sum(
            s for _, s in proposal_scores if s > n / 2
        )
        total_opposition = sum(
            n - s for _, s in proposal_scores if s < n / 2
        )

        return winner, {
            "total_support": total_support,
            "total_opposition": total_opposition,
            "reasoning": f"Won with {proposal_scores[0][1]:.1f} Borda points",
            "borda_scores": {p.proposal_id: s for p, s in proposal_scores},
            "num_proposals": n,
        }


class QuadraticVoting(ConsensusStrategy):
    """Square root weighting to prevent vote concentration.

    Uses quadratic voting principles to prevent any single voter
    from having too much influence, while still allowing
    strength of preference to be expressed.

    Weight = sqrt(confidence) * sqrt(expertise)

    Suitable for:
    - Preventing dominant agents from controlling decisions
    - Balanced multi-agent voting
    - Encouraging participation from all agents
    """

    def __init__(self, expertise_factor: float = 0.5):
        """Initialize quadratic voting strategy.

        Args:
            expertise_factor: How much to weight expertise (0 to 1)
        """
        self.expertise_factor = expertise_factor
        self.log = logger.bind(strategy="quadratic_voting")

    def resolve(
        self,
        proposals: list[Proposal],
        votes: list[Vote],
    ) -> tuple[Proposal, dict[str, Any]]:
        """Resolve using quadratic voting.

        Args:
            proposals: List of proposals
            votes: List of votes

        Returns:
            Tuple of (winning_proposal, scores)
        """
        if not proposals:
            raise ValueError("No proposals to resolve")

        # Calculate quadratic-weighted scores
        proposal_scores: dict[str, float] = defaultdict(float)
        vote_weights: dict[str, float] = {}

        for vote in votes:
            # Quadratic weighting
            confidence_weight = math.sqrt(vote.confidence)
            expertise_weight = math.sqrt(
                self.expertise_factor * vote.expertise_score + (1 - self.expertise_factor)
            )
            weight = confidence_weight * expertise_weight

            vote_weights[vote.vote_id] = weight

            if vote.vote_type == VoteType.SUPPORT:
                proposal_scores[vote.proposal_id] += weight
            elif vote.vote_type == VoteType.OPPOSE:
                proposal_scores[vote.proposal_id] -= weight
            # Abstain doesn't affect score

        # Ensure all proposals have a score
        for proposal in proposals:
            if proposal.proposal_id not in proposal_scores:
                proposal_scores[proposal.proposal_id] = 0.0

        # Sort by score
        scored_proposals = [
            (proposal, proposal_scores[proposal.proposal_id])
            for proposal in proposals
        ]
        scored_proposals.sort(key=lambda x: x[1], reverse=True)

        winner = scored_proposals[0][0]

        # Calculate totals (positive scores = support, negative = opposition)
        total_support = sum(s for _, s in scored_proposals if s > 0)
        total_opposition = abs(sum(s for _, s in scored_proposals if s < 0))

        return winner, {
            "total_support": total_support,
            "total_opposition": total_opposition,
            "reasoning": f"Won with {scored_proposals[0][1]:.3f} quadratic score",
            "quadratic_scores": {p.proposal_id: s for p, s in scored_proposals},
            "vote_weights": vote_weights,
        }


# Factory function for creating strategies
def create_strategy(
    name: str,
    **kwargs: Any,
) -> ConsensusStrategy:
    """Create a consensus strategy by name.

    Args:
        name: Strategy name (majority, confidence, expertise, supermajority, borda, quadratic)
        **kwargs: Additional arguments for the strategy

    Returns:
        ConsensusStrategy instance

    Raises:
        ValueError: If unknown strategy name
    """
    strategies = {
        "majority": MajorityVoting,
        "confidence": ConfidenceWeighted,
        "expertise": ExpertiseWeighted,
        "supermajority": SuperMajority,
        "borda": BordaCount,
        "quadratic": QuadraticVoting,
    }

    if name.lower() not in strategies:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(strategies.keys())}"
        )

    return strategies[name.lower()](**kwargs)
