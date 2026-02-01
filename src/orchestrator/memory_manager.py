"""
Advanced Memory Management for Agentic AI Systems.

Implements MemGPT / A-MEM pattern with three memory tiers:
1. Working Memory - Active context (via ContextManager)
2. Episodic Memory - Specific past events/interactions
3. Procedural Memory - Learned patterns/skills with Zettelkasten linking

RAP-252: Episodic and Procedural Memory Implementation

References:
- MemGPT (Packer et al. 2023): Hierarchical memory for LLM agents
- A-MEM (2024): Agentic memory with semantic linking
- Zettelkasten: Bidirectional knowledge graph approach
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class MemoryType(str, Enum):
    """Types of memories in the hierarchical system."""
    EPISODIC = "episodic"      # Past events/interactions
    PROCEDURAL = "procedural"  # Learned patterns/skills
    SEMANTIC = "semantic"      # Factual knowledge
    WORKING = "working"        # Active context


class MemoryImportance(str, Enum):
    """Importance levels for memory retention."""
    CRITICAL = "critical"    # Never forget (security incidents, etc.)
    HIGH = "high"           # Long retention (successful fixes)
    MEDIUM = "medium"       # Standard retention
    LOW = "low"             # Short retention (routine operations)


@dataclass
class EpisodicMemory:
    """A specific past event or interaction.

    Episodic memories capture:
    - What happened (event details)
    - When it happened (timestamp)
    - What was the outcome
    - Emotional valence (success/failure)
    """
    id: str
    event_type: str
    timestamp: datetime
    summary: str
    details: dict[str, Any]
    outcome: str  # "success", "failure", "partial"
    importance: MemoryImportance

    # Linking
    related_memories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Retrieval metadata
    access_count: int = 0
    last_accessed: datetime | None = None
    embedding: list[float] | None = None

    # Context
    org_id: str | None = None
    project_id: str | None = None
    agent_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "details": self.details,
            "outcome": self.outcome,
            "importance": self.importance.value,
            "related_memories": self.related_memories,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodicMemory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            summary=data["summary"],
            details=data.get("details", {}),
            outcome=data["outcome"],
            importance=MemoryImportance(data.get("importance", "medium")),
            related_memories=data.get("related_memories", []),
            tags=data.get("tags", []),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            org_id=data.get("org_id"),
            project_id=data.get("project_id"),
            agent_id=data.get("agent_id"),
        )


@dataclass
class ProceduralMemory:
    """A learned pattern, skill, or procedure.

    Procedural memories capture:
    - How to do something (steps/procedure)
    - When to apply it (trigger conditions)
    - Success rate and confidence
    - Links to related procedures (Zettelkasten)
    """
    id: str
    name: str
    description: str
    trigger_conditions: list[str]  # When to apply this procedure
    steps: list[dict[str, Any]]    # Procedure steps

    # Learning metadata
    success_count: int = 0
    failure_count: int = 0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    confidence: float = 0.5  # 0.0-1.0

    # Zettelkasten-style linking
    links: list[dict[str, str]] = field(default_factory=list)  # {"target_id": "...", "relationship": "..."}
    backlinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Context
    domain: str = "general"  # "healing", "testing", "analysis", etc.
    importance: MemoryImportance = MemoryImportance.MEDIUM

    # Retrieval
    embedding: list[float] | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def record_outcome(self, success: bool) -> None:
        """Record a procedure execution outcome."""
        now = datetime.utcnow()
        if success:
            self.success_count += 1
            self.last_success = now
            # Increase confidence on success
            self.confidence = min(1.0, self.confidence + 0.1 * (1 - self.confidence))
        else:
            self.failure_count += 1
            self.last_failure = now
            # Decrease confidence on failure
            self.confidence = max(0.0, self.confidence - 0.15 * self.confidence)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "steps": self.steps,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "confidence": self.confidence,
            "links": self.links,
            "backlinks": self.backlinks,
            "tags": self.tags,
            "domain": self.domain,
            "importance": self.importance.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProceduralMemory":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            trigger_conditions=data.get("trigger_conditions", []),
            steps=data.get("steps", []),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            last_success=datetime.fromisoformat(data["last_success"]) if data.get("last_success") else None,
            last_failure=datetime.fromisoformat(data["last_failure"]) if data.get("last_failure") else None,
            confidence=data.get("confidence", 0.5),
            links=data.get("links", []),
            backlinks=data.get("backlinks", []),
            tags=data.get("tags", []),
            domain=data.get("domain", "general"),
            importance=MemoryImportance(data.get("importance", "medium")),
        )


@dataclass
class MemorySearchResult:
    """Result from memory search."""
    memory: EpisodicMemory | ProceduralMemory
    score: float
    match_type: str  # "semantic", "keyword", "tag", "link"


class MemoryManager:
    """
    Advanced memory management for agentic AI systems.

    Implements hierarchical memory with:
    - Episodic Memory: Past events and interactions
    - Procedural Memory: Learned patterns with Zettelkasten linking
    - Memory consolidation and forgetting
    - Semantic search via embeddings

    Usage:
        ```python
        memory = MemoryManager(org_id="org_123", project_id="proj_456")

        # Store episodic memory
        await memory.store_episode(
            event_type="test_failure_healed",
            summary="Fixed login button selector after UI refactor",
            details={"old_selector": "#login", "new_selector": "[data-testid='login']"},
            outcome="success",
            importance=MemoryImportance.HIGH,
        )

        # Store procedural memory
        await memory.store_procedure(
            name="heal_selector_by_testid",
            description="When selector fails, try data-testid attribute",
            trigger_conditions=["selector_not_found", "element_missing"],
            steps=[
                {"action": "extract_element_context", "params": {}},
                {"action": "search_by_testid", "params": {}},
                {"action": "validate_selector", "params": {}},
            ],
        )

        # Retrieve relevant memories
        results = await memory.search(
            query="login button not found",
            memory_types=[MemoryType.EPISODIC, MemoryType.PROCEDURAL],
            limit=5,
        )
        ```
    """

    def __init__(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
        database_url: str | None = None,
        embeddings_model: str = "text-embedding-3-small",
    ):
        self.org_id = org_id
        self.project_id = project_id
        self.database_url = database_url
        self.embeddings_model = embeddings_model
        self.log = logger.bind(
            component="memory_manager",
            org_id=org_id,
            project_id=project_id,
        )

        # In-memory caches (for development/testing)
        self._episodic_cache: dict[str, EpisodicMemory] = {}
        self._procedural_cache: dict[str, ProceduralMemory] = {}

        # Database client (lazy initialization)
        self._db_client = None
        self._embeddings_client = None

    # =========================================================================
    # Episodic Memory Operations
    # =========================================================================

    async def store_episode(
        self,
        event_type: str,
        summary: str,
        details: dict[str, Any],
        outcome: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: list[str] | None = None,
        related_memories: list[str] | None = None,
        agent_id: str | None = None,
    ) -> EpisodicMemory:
        """Store a new episodic memory.

        Args:
            event_type: Type of event (e.g., "test_failure", "healing_success")
            summary: Brief summary of what happened
            details: Detailed information about the event
            outcome: "success", "failure", or "partial"
            importance: Memory importance level
            tags: Optional tags for categorization
            related_memories: IDs of related memories
            agent_id: ID of the agent that created this memory

        Returns:
            The created EpisodicMemory
        """
        memory_id = f"ep_{uuid4().hex[:12]}"

        episode = EpisodicMemory(
            id=memory_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            summary=summary,
            details=details,
            outcome=outcome,
            importance=importance,
            tags=tags or [],
            related_memories=related_memories or [],
            org_id=self.org_id,
            project_id=self.project_id,
            agent_id=agent_id,
        )

        # Generate embedding for semantic search
        episode.embedding = await self._generate_embedding(
            f"{event_type}: {summary}. {json.dumps(details)}"
        )

        # Store in cache and database
        self._episodic_cache[memory_id] = episode
        await self._persist_episode(episode)

        self.log.info(
            "Stored episodic memory",
            memory_id=memory_id,
            event_type=event_type,
            importance=importance.value,
        )

        return episode

    async def get_episode(self, memory_id: str) -> EpisodicMemory | None:
        """Retrieve an episodic memory by ID."""
        # Check cache first
        if memory_id in self._episodic_cache:
            memory = self._episodic_cache[memory_id]
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            return memory

        # Load from database
        return await self._load_episode(memory_id)

    async def search_episodes(
        self,
        query: str | None = None,
        event_types: list[str] | None = None,
        outcome: str | None = None,
        importance: MemoryImportance | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """Search episodic memories with filters.

        Args:
            query: Semantic search query
            event_types: Filter by event types
            outcome: Filter by outcome
            importance: Filter by minimum importance
            tags: Filter by tags (any match)
            since: Filter by timestamp
            limit: Maximum results

        Returns:
            List of matching memories with scores
        """
        results = []

        # Generate query embedding for semantic search
        query_embedding = None
        if query:
            query_embedding = await self._generate_embedding(query)

        # Search through cache (in production, use database with pgvector)
        for memory in self._episodic_cache.values():
            # Apply filters
            if event_types and memory.event_type not in event_types:
                continue
            if outcome and memory.outcome != outcome:
                continue
            if importance and self._importance_rank(memory.importance) < self._importance_rank(importance):
                continue
            if tags and not any(t in memory.tags for t in tags):
                continue
            if since and memory.timestamp < since:
                continue

            # Calculate score
            score = 1.0
            match_type = "filter"

            if query_embedding and memory.embedding:
                score = self._cosine_similarity(query_embedding, memory.embedding)
                match_type = "semantic"

            results.append(MemorySearchResult(
                memory=memory,
                score=score,
                match_type=match_type,
            ))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # =========================================================================
    # Procedural Memory Operations
    # =========================================================================

    async def store_procedure(
        self,
        name: str,
        description: str,
        trigger_conditions: list[str],
        steps: list[dict[str, Any]],
        domain: str = "general",
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: list[str] | None = None,
        links: list[dict[str, str]] | None = None,
    ) -> ProceduralMemory:
        """Store a new procedural memory (learned skill/pattern).

        Args:
            name: Procedure name
            description: What this procedure does
            trigger_conditions: When to apply this procedure
            steps: List of steps with actions and parameters
            domain: Domain (healing, testing, analysis, etc.)
            importance: Memory importance
            tags: Optional tags
            links: Zettelkasten-style links to other procedures

        Returns:
            The created ProceduralMemory
        """
        memory_id = f"pr_{uuid4().hex[:12]}"

        procedure = ProceduralMemory(
            id=memory_id,
            name=name,
            description=description,
            trigger_conditions=trigger_conditions,
            steps=steps,
            domain=domain,
            importance=importance,
            tags=tags or [],
            links=links or [],
        )

        # Generate embedding
        embed_text = f"{name}: {description}. Triggers: {', '.join(trigger_conditions)}"
        procedure.embedding = await self._generate_embedding(embed_text)

        # Update backlinks in linked procedures
        if links:
            for link in links:
                target_id = link.get("target_id")
                if target_id and target_id in self._procedural_cache:
                    self._procedural_cache[target_id].backlinks.append(memory_id)

        # Store
        self._procedural_cache[memory_id] = procedure
        await self._persist_procedure(procedure)

        self.log.info(
            "Stored procedural memory",
            memory_id=memory_id,
            name=name,
            domain=domain,
        )

        return procedure

    async def get_procedure(self, memory_id: str) -> ProceduralMemory | None:
        """Retrieve a procedural memory by ID."""
        return self._procedural_cache.get(memory_id)

    async def find_applicable_procedures(
        self,
        context: str,
        domain: str | None = None,
        min_confidence: float = 0.3,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        """Find procedures applicable to the current context.

        Uses semantic similarity to match context against trigger conditions.

        Args:
            context: Current situation/error/context
            domain: Optional domain filter
            min_confidence: Minimum procedure confidence
            limit: Maximum results

        Returns:
            List of applicable procedures ranked by relevance
        """
        results = []
        context_embedding = await self._generate_embedding(context)

        for procedure in self._procedural_cache.values():
            # Apply filters
            if domain and procedure.domain != domain:
                continue
            if procedure.confidence < min_confidence:
                continue

            # Calculate relevance score
            score = 0.0
            if procedure.embedding and context_embedding:
                score = self._cosine_similarity(context_embedding, procedure.embedding)

            # Boost by confidence
            score = score * (0.5 + 0.5 * procedure.confidence)

            results.append(MemorySearchResult(
                memory=procedure,
                score=score,
                match_type="semantic",
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def record_procedure_outcome(
        self,
        procedure_id: str,
        success: bool,
        context: str | None = None,
    ) -> None:
        """Record the outcome of applying a procedure.

        Updates confidence and creates an episodic memory of the application.

        Args:
            procedure_id: ID of the procedure
            success: Whether the application was successful
            context: Optional context about the application
        """
        procedure = self._procedural_cache.get(procedure_id)
        if not procedure:
            return

        procedure.record_outcome(success)

        # Create episodic memory of the application
        await self.store_episode(
            event_type="procedure_applied",
            summary=f"Applied procedure '{procedure.name}': {'success' if success else 'failure'}",
            details={
                "procedure_id": procedure_id,
                "procedure_name": procedure.name,
                "success": success,
                "context": context,
                "new_confidence": procedure.confidence,
            },
            outcome="success" if success else "failure",
            importance=MemoryImportance.MEDIUM,
            related_memories=[procedure_id],
        )

        self.log.info(
            "Recorded procedure outcome",
            procedure_id=procedure_id,
            success=success,
            new_confidence=procedure.confidence,
        )

    # =========================================================================
    # Unified Search
    # =========================================================================

    async def search(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> list[MemorySearchResult]:
        """Search across all memory types.

        Args:
            query: Search query
            memory_types: Types to search (default: all)
            limit: Maximum total results

        Returns:
            Combined and ranked results
        """
        types = memory_types or [MemoryType.EPISODIC, MemoryType.PROCEDURAL]
        all_results = []

        if MemoryType.EPISODIC in types:
            episodes = await self.search_episodes(query=query, limit=limit)
            all_results.extend(episodes)

        if MemoryType.PROCEDURAL in types:
            procedures = await self.find_applicable_procedures(context=query, limit=limit)
            all_results.extend(procedures)

        # Merge and rank
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:limit]

    # =========================================================================
    # Memory Consolidation and Forgetting
    # =========================================================================

    async def consolidate_memories(self) -> dict[str, int]:
        """Consolidate and clean up memories.

        - Merge similar episodic memories
        - Decay old, unused memories
        - Strengthen frequently accessed memories

        Returns:
            Statistics about consolidation
        """
        stats = {
            "episodes_decayed": 0,
            "episodes_removed": 0,
            "procedures_updated": 0,
        }

        now = datetime.utcnow()

        # Decay episodic memories based on importance and access
        for memory_id, memory in list(self._episodic_cache.items()):
            age_days = (now - memory.timestamp).days

            # Calculate decay threshold based on importance
            decay_thresholds = {
                MemoryImportance.CRITICAL: 365 * 10,  # 10 years
                MemoryImportance.HIGH: 365,           # 1 year
                MemoryImportance.MEDIUM: 90,          # 90 days
                MemoryImportance.LOW: 30,             # 30 days
            }

            threshold = decay_thresholds.get(memory.importance, 90)

            # Extend threshold based on access frequency
            if memory.access_count > 10:
                threshold *= 2

            if age_days > threshold:
                # Remove from cache (in production, archive to cold storage)
                del self._episodic_cache[memory_id]
                stats["episodes_removed"] += 1
            elif age_days > threshold / 2:
                stats["episodes_decayed"] += 1

        # Update procedural memory confidence based on recency
        for procedure in self._procedural_cache.values():
            if procedure.last_success:
                days_since_success = (now - procedure.last_success).days
                if days_since_success > 30:
                    # Slight confidence decay for unused procedures
                    procedure.confidence = max(0.3, procedure.confidence * 0.99)
                    stats["procedures_updated"] += 1

        self.log.info("Memory consolidation complete", **stats)
        return stats

    # =========================================================================
    # Zettelkasten-style Linking
    # =========================================================================

    async def link_procedures(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
    ) -> bool:
        """Create a Zettelkasten-style link between procedures.

        Args:
            source_id: Source procedure ID
            target_id: Target procedure ID
            relationship: Type of relationship (e.g., "prerequisite", "alternative", "extension")

        Returns:
            True if link was created
        """
        source = self._procedural_cache.get(source_id)
        target = self._procedural_cache.get(target_id)

        if not source or not target:
            return False

        # Add forward link
        source.links.append({
            "target_id": target_id,
            "relationship": relationship,
        })

        # Add backlink
        if source_id not in target.backlinks:
            target.backlinks.append(source_id)

        self.log.debug(
            "Linked procedures",
            source=source_id,
            target=target_id,
            relationship=relationship,
        )

        return True

    async def get_linked_procedures(
        self,
        procedure_id: str,
        include_backlinks: bool = True,
    ) -> list[tuple[ProceduralMemory, str]]:
        """Get procedures linked to a given procedure.

        Args:
            procedure_id: Procedure ID
            include_backlinks: Include backward links

        Returns:
            List of (procedure, relationship) tuples
        """
        procedure = self._procedural_cache.get(procedure_id)
        if not procedure:
            return []

        results = []

        # Forward links
        for link in procedure.links:
            target = self._procedural_cache.get(link["target_id"])
            if target:
                results.append((target, link["relationship"]))

        # Backlinks
        if include_backlinks:
            for backlink_id in procedure.backlinks:
                source = self._procedural_cache.get(backlink_id)
                if source:
                    # Find the relationship
                    rel = "backlink"
                    for link in source.links:
                        if link["target_id"] == procedure_id:
                            rel = f"backlink:{link['relationship']}"
                            break
                    results.append((source, rel))

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using the embeddings model."""
        try:
            # Try to use OpenAI embeddings
            if self._embeddings_client is None:
                try:
                    from openai import AsyncOpenAI
                    from ..config import get_settings
                    settings = get_settings()
                    if settings.openai_api_key:
                        self._embeddings_client = AsyncOpenAI(
                            api_key=settings.openai_api_key.get_secret_value()
                        )
                except Exception:
                    pass

            if self._embeddings_client:
                response = await self._embeddings_client.embeddings.create(
                    model=self.embeddings_model,
                    input=text[:8000],  # Truncate to avoid limits
                )
                return response.data[0].embedding

            return None

        except Exception as e:
            self.log.warning("Failed to generate embedding", error=str(e))
            return None

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _importance_rank(self, importance: MemoryImportance) -> int:
        """Convert importance to numeric rank."""
        ranks = {
            MemoryImportance.LOW: 1,
            MemoryImportance.MEDIUM: 2,
            MemoryImportance.HIGH: 3,
            MemoryImportance.CRITICAL: 4,
        }
        return ranks.get(importance, 2)

    async def _persist_episode(self, episode: EpisodicMemory) -> None:
        """Persist episodic memory to database."""
        # In production, save to Supabase
        # For now, cache only
        pass

    async def _load_episode(self, memory_id: str) -> EpisodicMemory | None:
        """Load episodic memory from database."""
        # In production, load from Supabase
        return None

    async def _persist_procedure(self, procedure: ProceduralMemory) -> None:
        """Persist procedural memory to database."""
        # In production, save to Supabase
        pass

    # =========================================================================
    # Statistics and Diagnostics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        episode_count = len(self._episodic_cache)
        procedure_count = len(self._procedural_cache)

        # Calculate average procedure confidence
        avg_confidence = 0.0
        if procedure_count > 0:
            avg_confidence = sum(p.confidence for p in self._procedural_cache.values()) / procedure_count

        return {
            "episodic_memories": episode_count,
            "procedural_memories": procedure_count,
            "average_procedure_confidence": round(avg_confidence, 3),
            "org_id": self.org_id,
            "project_id": self.project_id,
        }


# Singleton instance for global access
_memory_manager: MemoryManager | None = None


def get_memory_manager(
    org_id: str | None = None,
    project_id: str | None = None,
) -> MemoryManager:
    """Get or create a memory manager instance."""
    global _memory_manager

    if _memory_manager is None or (org_id and _memory_manager.org_id != org_id):
        _memory_manager = MemoryManager(org_id=org_id, project_id=project_id)

    return _memory_manager
