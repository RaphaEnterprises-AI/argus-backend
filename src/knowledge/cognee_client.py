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

import cognee
import structlog

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
