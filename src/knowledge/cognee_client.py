"""
Cognee Knowledge Client - Unified knowledge layer for Argus.

This module provides a unified interface to Cognee for:
- Storing and retrieving knowledge (key-value with embeddings)
- Semantic search across all knowledge types
- Failure pattern storage and retrieval
- Multi-hop graph reasoning

This replaces the custom MemoryStore and GraphStore with a single
Cognee-powered layer, reducing code complexity and gaining:
- ECL (Extract, Cognify, Load) pipeline
- 30+ data connectors
- Incremental learning
- Graph + vector hybrid search
- Multi-hop reasoning
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

# Cognee is optional for development/testing
try:
    import cognee

    COGNEE_AVAILABLE = True
except ImportError:
    cognee = None  # type: ignore
    COGNEE_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class SimilarFailure:
    """A similar failure pattern found via semantic search."""

    id: str
    error_message: str
    error_type: Optional[str]
    original_selector: Optional[str]
    healed_selector: str
    healing_method: str
    success_count: int
    failure_count: int
    success_rate: float
    similarity: float
    metadata: dict


class CogneeKnowledgeClient:
    """
    Unified knowledge client powered by Cognee.

    This client provides a compatibility layer that matches the interface
    of the deprecated MemoryStore and GraphStore, while using Cognee's
    superior ECL pipeline and graph capabilities under the hood.

    Key features:
    - Drop-in replacement for MemoryStore.get/put/search
    - Drop-in replacement for MemoryStore failure pattern methods
    - Graph-based reasoning via Cognee's knowledge graph
    - Multi-tenant isolation via dataset namespacing

    Example:
        ```python
        client = CogneeKnowledgeClient(org_id="org123", project_id="proj456")

        # Store a value
        await client.put(
            namespace=["patterns", "selectors"],
            key="login-button",
            value={"selector": "#login-btn", "success_rate": 0.95},
            embed_text="login button selector pattern"
        )

        # Semantic search
        results = await client.search(
            namespace=["patterns"],
            query="button for user login",
            limit=5
        )

        # Find similar failures
        similar = await client.find_similar_failures(
            error_message="Element not found: #submit-btn"
        )
        ```
    """

    def __init__(
        self,
        org_id: str,
        project_id: str,
    ):
        """
        Initialize the Cognee knowledge client.

        Args:
            org_id: Organization ID for multi-tenant isolation
            project_id: Project ID for multi-tenant isolation
        """
        self.org_id = org_id
        self.project_id = project_id
        self._namespace_prefix = f"{org_id}:{project_id}"
        self._log = logger.bind(
            component="cognee_client",
            org_id=org_id,
            project_id=project_id,
        )

    def _ensure_cognee_available(self) -> None:
        """Raise ImportError if cognee module is not available."""
        if not COGNEE_AVAILABLE:
            raise ImportError(
                "The cognee package is required for CogneeKnowledgeClient. "
                "Install it with: pip install cognee"
            )

    def _build_namespace(self, parts: list[str] | tuple[str, ...]) -> str:
        """Build a full namespace string from parts.

        Args:
            parts: Namespace parts like ["patterns", "selectors"]

        Returns:
            Full namespace like "org_id:project_id:patterns:selectors"
        """
        if parts:
            return f"{self._namespace_prefix}:{':'.join(parts)}"
        return self._namespace_prefix

    def _get_dataset_name(self, namespace: list[str] | tuple[str, ...]) -> str:
        """Generate tenant-scoped dataset name from namespace.

        Args:
            namespace: Hierarchical namespace like ["patterns", "selectors"]

        Returns:
            Dataset name like "org_abc_project_xyz_patterns_selectors"
        """
        namespace_str = "_".join(namespace) if namespace else "default"
        return f"org_{self.org_id}_project_{self.project_id}_{namespace_str}"

    def _generate_key_id(self, namespace: list[str] | tuple[str, ...], key: str) -> str:
        """Generate a unique ID for a namespace+key combination.

        Args:
            namespace: Hierarchical namespace
            key: Key within namespace

        Returns:
            Unique hash ID
        """
        full_key = f"{self.org_id}:{self.project_id}:{':'.join(namespace)}:{key}"
        return hashlib.sha256(full_key.encode()).hexdigest()[:32]

    # =========================================================================
    # Key-Value Store Interface (replaces MemoryStore.get/put)
    # =========================================================================

    async def put(
        self,
        namespace: list[str] | tuple[str, ...],
        key: str,
        value: dict[str, Any],
        embed_text: Optional[str] = None,
    ) -> None:
        """Store a value with optional embedding.

        This provides compatibility with the MemoryStore.put() interface
        while using Cognee's ECL pipeline for storage.

        Args:
            namespace: Hierarchical namespace (e.g., ["patterns", "selectors"])
            key: Unique key within namespace
            value: JSON-serializable value
            embed_text: Optional text for embedding generation
        """
        self._ensure_cognee_available()
        dataset_name = self._get_dataset_name(namespace)
        key_id = self._generate_key_id(namespace, key)

        # Build content document with metadata
        content = {
            "_id": key_id,
            "_key": key,
            "_namespace": list(namespace),
            "_org_id": self.org_id,
            "_project_id": self.project_id,
            "_created_at": datetime.now(timezone.utc).isoformat(),
            "_embed_text": embed_text,
            **value,
        }

        # Add to Cognee
        await cognee.add(
            json.dumps(content),
            dataset_name=dataset_name,
        )

        # Run cognify to extract knowledge and create embeddings
        await cognee.cognify(dataset_name=dataset_name)

        self._log.debug(
            "Stored value in Cognee",
            namespace=namespace,
            key=key,
            dataset=dataset_name,
        )

    async def get(
        self,
        namespace: list[str] | tuple[str, ...],
        key: str,
    ) -> Optional[dict[str, Any]]:
        """Get a value by namespace and key.

        This provides compatibility with the MemoryStore.get() interface.

        Args:
            namespace: Hierarchical namespace
            key: Key within namespace

        Returns:
            Stored value or None if not found
        """
        self._ensure_cognee_available()
        dataset_name = self._get_dataset_name(namespace)
        key_id = self._generate_key_id(namespace, key)

        try:
            # Search for the specific key
            results = await cognee.search(
                query=f"_id:{key_id}",
                dataset_name=dataset_name,
                top_k=1,
            )

            if results and len(results) > 0:
                result = results[0]
                # Parse the result and extract the value
                if isinstance(result, str):
                    return json.loads(result)
                elif isinstance(result, dict):
                    # Remove internal fields
                    return {k: v for k, v in result.items() if not k.startswith("_")}

            return None

        except Exception as e:
            self._log.warning(
                "Failed to get value from Cognee",
                namespace=namespace,
                key=key,
                error=str(e),
            )
            return None

    async def delete(
        self,
        namespace: list[str] | tuple[str, ...],
        key: str,
    ) -> bool:
        """Delete a value by namespace and key.

        Args:
            namespace: Hierarchical namespace
            key: Key within namespace

        Returns:
            True if deleted, False otherwise
        """
        # Note: Cognee doesn't have a direct delete API in v0.3
        # We mark the item as deleted instead
        await self.put(
            namespace=namespace,
            key=key,
            value={"_deleted": True, "_deleted_at": datetime.now(timezone.utc).isoformat()},
        )
        return True

    async def search(
        self,
        namespace: list[str] | tuple[str, ...],
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Semantic search within a namespace.

        This provides compatibility with the MemoryStore.search() interface
        while leveraging Cognee's hybrid graph+vector search.

        Args:
            namespace: Namespace to search in
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity (0-1)

        Returns:
            List of matching items with similarity scores
        """
        self._ensure_cognee_available()
        dataset_name = self._get_dataset_name(namespace)

        try:
            results = await cognee.search(
                query=query,
                dataset_name=dataset_name,
                top_k=limit,
            )

            parsed_results = []
            for i, result in enumerate(results or []):
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                    except json.JSONDecodeError:
                        parsed = {"content": result}
                else:
                    parsed = result

                # Add similarity score (Cognee results are ordered by relevance)
                parsed["similarity"] = 1.0 - (i * 0.1)  # Approximate score
                parsed_results.append(parsed)

            return parsed_results

        except Exception as e:
            self._log.warning(
                "Search failed",
                namespace=namespace,
                query=query[:50],
                error=str(e),
            )
            return []

    # =========================================================================
    # Failure Pattern Interface (replaces MemoryStore failure methods)
    # =========================================================================

    async def store_failure_pattern(
        self,
        error_message: str,
        healed_selector: str,
        healing_method: str,
        test_id: Optional[str] = None,
        error_type: Optional[str] = None,
        original_selector: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a failure pattern for future learning.

        This stores a successful healing pattern in Cognee's knowledge graph,
        allowing semantic search to find similar failures and their solutions.

        Args:
            error_message: The error message from the failure
            healed_selector: The selector that fixed the issue
            healing_method: Method used (e.g., "semantic_match", "code_aware")
            test_id: Optional test ID
            error_type: Type of error
            original_selector: The original broken selector
            metadata: Additional metadata

        Returns:
            Pattern ID
        """
        self._ensure_cognee_available()
        pattern_id = hashlib.sha256(
            f"{error_message}:{original_selector}:{healed_selector}".encode()
        ).hexdigest()[:32]

        pattern = {
            "pattern_id": pattern_id,
            "error_message": error_message,
            "error_type": error_type,
            "original_selector": original_selector,
            "healed_selector": healed_selector,
            "healing_method": healing_method,
            "test_id": test_id,
            "success_count": 1,
            "failure_count": 0,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.put(
            namespace=["failure_patterns"],
            key=pattern_id,
            value=pattern,
            embed_text=f"{error_message} {error_type or ''} {original_selector or ''}",
        )

        self._log.info(
            "Stored failure pattern",
            pattern_id=pattern_id,
            error_type=error_type,
            healing_method=healing_method,
        )

        return pattern_id

    async def find_similar_failures(
        self,
        error_message: str,
        limit: int = 5,
        threshold: float = 0.7,
        error_type: Optional[str] = None,
    ) -> list[SimilarFailure]:
        """Find similar past failures with their healing solutions.

        Uses Cognee's semantic search to find failures with similar
        error messages and returns the healing solutions that worked.

        Args:
            error_message: Current error message
            limit: Maximum results
            threshold: Minimum similarity
            error_type: Optional filter

        Returns:
            List of similar failures with solutions
        """
        query = error_message
        if error_type:
            query = f"{error_type}: {error_message}"

        results = await self.search(
            namespace=["failure_patterns"],
            query=query,
            limit=limit,
            threshold=threshold,
        )

        similar_failures = []
        for result in results:
            if result.get("_deleted"):
                continue

            success_count = result.get("success_count", 0)
            failure_count = result.get("failure_count", 0)
            total = success_count + failure_count

            similar_failures.append(SimilarFailure(
                id=result.get("pattern_id", ""),
                error_message=result.get("error_message", ""),
                error_type=result.get("error_type"),
                original_selector=result.get("original_selector"),
                healed_selector=result.get("healed_selector", ""),
                healing_method=result.get("healing_method", ""),
                success_count=success_count,
                failure_count=failure_count,
                success_rate=success_count / total if total > 0 else 0,
                similarity=result.get("similarity", 0),
                metadata=result.get("metadata", {}),
            ))

        self._log.debug(
            "Found similar failures",
            query=error_message[:50],
            count=len(similar_failures),
        )

        return similar_failures

    async def record_healing_outcome(
        self,
        pattern_id: str,
        success: bool,
    ) -> None:
        """Record the outcome of applying a healing pattern.

        Updates success/failure counts for continuous learning.

        Args:
            pattern_id: Pattern ID
            success: Whether healing worked
        """
        existing = await self.get(
            namespace=["failure_patterns"],
            key=pattern_id,
        )

        if not existing:
            self._log.warning("Pattern not found", pattern_id=pattern_id)
            return

        if success:
            existing["success_count"] = existing.get("success_count", 0) + 1
        else:
            existing["failure_count"] = existing.get("failure_count", 0) + 1

        existing["last_used_at"] = datetime.now(timezone.utc).isoformat()

        await self.put(
            namespace=["failure_patterns"],
            key=pattern_id,
            value=existing,
        )

        self._log.debug(
            "Recorded healing outcome",
            pattern_id=pattern_id,
            success=success,
        )

    # =========================================================================
    # Graph Reasoning Interface (replaces GraphStore)
    # =========================================================================

    async def add_to_knowledge_graph(
        self,
        content: str | dict,
        content_type: str = "general",
    ) -> None:
        """Add content to the knowledge graph via Cognee's ECL pipeline.

        This is the primary way to add knowledge that will be:
        1. Extracted (entities, relationships)
        2. Cognified (embeddings, graph connections)
        3. Loaded (stored in graph + vector store)

        Args:
            content: Text or structured content to add
            content_type: Type hint for better extraction
        """
        self._ensure_cognee_available()
        dataset_name = self._get_dataset_name([content_type])

        if isinstance(content, dict):
            content = json.dumps(content)

        await cognee.add(content, dataset_name=dataset_name)
        await cognee.cognify(dataset_name=dataset_name)

        self._log.debug("Added to knowledge graph", content_type=content_type)

    async def query_knowledge_graph(
        self,
        query: str,
        content_types: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the knowledge graph with natural language.

        Uses Cognee's hybrid search to find relevant knowledge
        across graph and vector stores.

        Args:
            query: Natural language query
            content_types: Optional filter by content type
            limit: Maximum results

        Returns:
            Relevant knowledge items
        """
        self._ensure_cognee_available()
        all_results = []

        # Search across specified content types or all
        types_to_search = content_types or ["codebase", "tests", "failures", "general"]

        for content_type in types_to_search:
            dataset_name = self._get_dataset_name([content_type])
            try:
                results = await cognee.search(
                    query=query,
                    dataset_name=dataset_name,
                    top_k=limit // len(types_to_search),
                )
                if results:
                    for result in results:
                        if isinstance(result, str):
                            all_results.append({"content": result, "type": content_type})
                        else:
                            result["type"] = content_type
                            all_results.append(result)
            except Exception as e:
                self._log.warning(
                    "Failed to search content type",
                    content_type=content_type,
                    error=str(e),
                )

        return all_results[:limit]

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: Optional[list[str]] = None,
        hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Get entities related to a given entity via graph traversal.

        Uses Cognee's graph capabilities for multi-hop reasoning.

        Args:
            entity_id: Starting entity ID
            relationship_types: Filter by relationship types
            hops: Maximum traversal depth

        Returns:
            Related entities with relationship info
        """
        # Build a query that leverages Cognee's graph search
        query = f"entities related to {entity_id}"
        if relationship_types:
            query += f" via {', '.join(relationship_types)}"

        results = await self.query_knowledge_graph(
            query=query,
            limit=20,
        )

        return results

    # =========================================================================
    # Discovery Patterns Interface (replaces PatternService + CloudflareVectorize)
    # =========================================================================

    async def store_discovery_pattern(
        self,
        pattern_type: str,
        pattern_name: str,
        pattern_signature: str,
        pattern_data: dict[str, Any],
        source_url: Optional[str] = None,
        source_project_id: Optional[str] = None,
    ) -> str:
        """Store a UI discovery pattern for cross-project learning.

        This replaces PatternService._store_pattern() and uses Cognee's ECL pipeline
        for embedding generation and storage instead of Cloudflare Vectorize.

        Args:
            pattern_type: Type of pattern (page_layout, navigation, form, authentication, etc.)
            pattern_name: Human-readable pattern name
            pattern_signature: Hash for deduplication
            pattern_data: Full pattern details including features
            source_url: URL where pattern was discovered
            source_project_id: Project ID where pattern was found

        Returns:
            Pattern ID
        """
        # Create searchable text for embedding
        embed_parts = [
            f"Pattern type: {pattern_type}",
            f"Pattern name: {pattern_name}",
        ]

        features = pattern_data.get("features", {})
        for key, value in features.items():
            if value:
                embed_parts.append(f"{key}: {value}")

        embed_text = " | ".join(embed_parts)

        # Generate pattern ID
        pattern_id = f"discovery_{pattern_signature}"

        # Store via Cognee ECL pipeline
        value = {
            "id": pattern_id,
            "pattern_type": pattern_type,
            "pattern_name": pattern_name,
            "pattern_signature": pattern_signature,
            "pattern_data": pattern_data,
            "source_url": source_url,
            "source_project_id": source_project_id,
            "times_seen": 1,
            "projects_seen": 1,
            "test_success_rate": 0.0,
            "self_heal_success_rate": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.put(
            namespace=["discovery_patterns", pattern_type],
            key=pattern_id,
            value=value,
            embed_text=embed_text,
        )

        self._log.info(
            "Stored discovery pattern",
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            pattern_name=pattern_name,
        )

        return pattern_id

    async def find_similar_discovery_patterns(
        self,
        query_text: str,
        pattern_type: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Find similar UI patterns for cross-project learning.

        This replaces PatternService.find_similar_patterns() using Cognee's
        semantic search capabilities.

        Args:
            query_text: Text description or pattern to match
            pattern_type: Optional filter by pattern type
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching patterns with similarity scores
        """
        # Build namespace for search
        if pattern_type:
            namespaces = [["discovery_patterns", pattern_type]]
        else:
            # Search across all pattern types
            namespaces = [["discovery_patterns"]]

        results = []
        for ns in namespaces:
            try:
                matches = await self.search(
                    namespace=ns,
                    query=query_text,
                    limit=limit,
                )
                results.extend(matches)
            except Exception as e:
                self._log.warning(
                    "Failed to search discovery patterns",
                    namespace=ns,
                    error=str(e),
                )

        # Filter by similarity and deduplicate
        filtered = []
        seen_ids = set()
        for match in results:
            if isinstance(match, dict):
                pattern_id = match.get("id", match.get("key", ""))
                similarity = match.get("similarity", match.get("score", 0.8))

                if pattern_id not in seen_ids and similarity >= min_similarity:
                    seen_ids.add(pattern_id)
                    match["similarity"] = similarity
                    filtered.append(match)

        # Sort by similarity descending
        filtered.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        return filtered[:limit]

    async def increment_pattern_times_seen(
        self,
        pattern_id: str,
        pattern_type: str,
    ) -> bool:
        """Increment times_seen counter for an existing pattern.

        Args:
            pattern_id: Pattern ID
            pattern_type: Pattern type (for namespace lookup)

        Returns:
            True if updated successfully
        """
        try:
            # Get current pattern
            current = await self.get(
                namespace=["discovery_patterns", pattern_type],
                key=pattern_id,
            )

            if current:
                current["times_seen"] = current.get("times_seen", 0) + 1
                current["updated_at"] = datetime.now(timezone.utc).isoformat()

                await self.put(
                    namespace=["discovery_patterns", pattern_type],
                    key=pattern_id,
                    value=current,
                )
                return True
        except Exception as e:
            self._log.warning(
                "Failed to increment pattern times_seen",
                pattern_id=pattern_id,
                error=str(e),
            )

        return False

    async def update_discovery_pattern_stats(
        self,
        pattern_id: str,
        pattern_type: str,
        test_passed: bool,
        self_healed: bool = False,
    ) -> bool:
        """Update pattern success rates after test execution.

        This enables learning: patterns with high success rates
        are prioritized in future discoveries.

        Args:
            pattern_id: Pattern ID
            pattern_type: Pattern type (for namespace lookup)
            test_passed: Whether the test passed
            self_healed: Whether self-healing was applied

        Returns:
            True if updated successfully
        """
        try:
            # Get current pattern
            current = await self.get(
                namespace=["discovery_patterns", pattern_type],
                key=pattern_id,
            )

            if not current:
                return False

            times_seen = current.get("times_seen", 1)
            current_test_rate = float(current.get("test_success_rate", 0) or 0)
            current_heal_rate = float(current.get("self_heal_success_rate", 0) or 0)

            # Calculate new rolling average
            new_test_rate = (
                (current_test_rate * (times_seen - 1)) + (100 if test_passed else 0)
            ) / times_seen

            new_heal_rate = current_heal_rate
            if self_healed:
                new_heal_rate = (
                    (current_heal_rate * (times_seen - 1)) + 100
                ) / times_seen

            current["test_success_rate"] = round(new_test_rate, 2)
            current["self_heal_success_rate"] = round(new_heal_rate, 2)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()

            await self.put(
                namespace=["discovery_patterns", pattern_type],
                key=pattern_id,
                value=current,
            )

            self._log.info(
                "Updated discovery pattern stats",
                pattern_id=pattern_id,
                test_success_rate=new_test_rate,
                self_heal_success_rate=new_heal_rate,
            )

            return True
        except Exception as e:
            self._log.warning(
                "Failed to update pattern stats",
                pattern_id=pattern_id,
                error=str(e),
            )

        return False

    async def get_discovery_pattern_insights(
        self,
        pattern_type: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get insights about stored discovery patterns.

        Returns statistics about pattern types, success rates, etc.

        Args:
            pattern_type: Optional filter by pattern type

        Returns:
            Insights dictionary with statistics
        """
        # Search all patterns
        if pattern_type:
            namespaces = [["discovery_patterns", pattern_type]]
        else:
            namespaces = [["discovery_patterns"]]

        patterns = []
        for ns in namespaces:
            try:
                # Use a generic query to get all patterns
                results = await self.search(
                    namespace=ns,
                    query="pattern",  # Generic query to match all
                    limit=1000,
                )
                patterns.extend(results)
            except Exception:
                pass

        # Calculate insights
        by_type = {}
        total_patterns = len(patterns)
        total_seen = 0
        avg_test_success = 0
        avg_heal_success = 0

        for p in patterns:
            if not isinstance(p, dict):
                continue

            ptype = p.get("pattern_type", "unknown")
            if ptype not in by_type:
                by_type[ptype] = {"count": 0, "total_seen": 0}
            by_type[ptype]["count"] += 1
            by_type[ptype]["total_seen"] += p.get("times_seen", 0)
            total_seen += p.get("times_seen", 0)
            avg_test_success += float(p.get("test_success_rate", 0) or 0)
            avg_heal_success += float(p.get("self_heal_success_rate", 0) or 0)

        if total_patterns > 0:
            avg_test_success /= total_patterns
            avg_heal_success /= total_patterns

        return {
            "total_patterns": total_patterns,
            "total_occurrences": total_seen,
            "by_type": by_type,
            "avg_test_success_rate": round(avg_test_success, 2),
            "avg_self_heal_success_rate": round(avg_heal_success, 2),
        }


# =========================================================================
# Global Instance Management
# =========================================================================

_cognee_client: Optional[CogneeKnowledgeClient] = None


def get_cognee_client(
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> CogneeKnowledgeClient:
    """Get or create the global Cognee client instance.

    Args:
        org_id: Organization ID (defaults to DEFAULT_ORG_ID env var)
        project_id: Project ID (defaults to DEFAULT_PROJECT_ID env var)

    Returns:
        CogneeKnowledgeClient instance
    """
    global _cognee_client

    org = org_id or os.environ.get("DEFAULT_ORG_ID", "default")
    proj = project_id or os.environ.get("DEFAULT_PROJECT_ID", "default")

    if _cognee_client is None or (
        _cognee_client.org_id != org or _cognee_client.project_id != proj
    ):
        _cognee_client = CogneeKnowledgeClient(org_id=org, project_id=proj)

    return _cognee_client


def reset_cognee_client() -> None:
    """Reset the global Cognee client instance."""
    global _cognee_client
    _cognee_client = None


async def init_cognee_client(
    org_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> CogneeKnowledgeClient:
    """Initialize and return the Cognee client.

    Args:
        org_id: Organization ID
        project_id: Project ID

    Returns:
        Initialized CogneeKnowledgeClient
    """
    client = get_cognee_client(org_id=org_id, project_id=project_id)
    logger.info(
        "Cognee client initialized",
        org_id=client.org_id,
        project_id=client.project_id,
    )
    return client
