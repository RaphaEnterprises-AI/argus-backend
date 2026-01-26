"""Apache AGE Graph Store Client.

Provides a high-level interface for interacting with the Apache AGE knowledge graph,
enabling multi-hop reasoning about test relationships.
"""

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from .schema import EdgeType, EntityType, GraphEntity, GraphRelationship

logger = structlog.get_logger(__name__)


class GraphStore:
    """Client for Apache AGE knowledge graph operations.

    This class provides methods to:
    - Add vertices (entities) to the graph
    - Create edges (relationships) between vertices
    - Execute Cypher queries for graph traversal
    - Perform multi-hop reasoning queries
    - Find related entities efficiently

    Example:
        ```python
        graph = GraphStore()

        # Add test vertex
        test_id = await graph.add_test(
            test_id="uuid-123",
            name="login_test",
            file_path="tests/auth/test_login.py"
        )

        # Find related failures
        failures = await graph.find_related_failures(test_id)
        ```
    """

    def __init__(
        self,
        database_url: str | None = None,
        graph_name: str = "argus_knowledge",
    ):
        """Initialize the graph store.

        Args:
            database_url: PostgreSQL connection URL. Defaults to DATABASE_URL env var.
            graph_name: Name of the AGE graph. Defaults to 'argus_knowledge'.
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.graph_name = graph_name
        self._pool = None
        self._log = logger.bind(component="graph_store")

    async def _get_pool(self):
        """Get or create the connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for graph store. Install with: pip install asyncpg"
                )

            if not self.database_url:
                raise ValueError(
                    "DATABASE_URL environment variable or database_url parameter required"
                )

            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=10,
            )
            self._log.info("Created graph database connection pool")

        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._log.info("Closed graph database connection pool")

    # =========================================================================
    # Core Graph Operations
    # =========================================================================

    async def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query on the graph.

        Args:
            cypher: Cypher query string
            params: Optional parameters for the query

        Returns:
            List of result records as dictionaries

        Example:
            ```python
            results = await graph.query(
                "MATCH (t:Test)-[:USES]->(s:Selector) "
                "WHERE s.selector = $selector "
                "RETURN t",
                params={"selector": "#login-button"}
            )
            ```
        """
        pool = await self._get_pool()

        # Wrap Cypher query in AGE SQL function
        # AGE requires queries to be wrapped in cypher() function
        age_query = f"""
            SELECT * FROM cypher('{self.graph_name}', $$
                {cypher}
            $$) AS (result agtype);
        """

        self._log.debug("Executing Cypher query", cypher=cypher[:100])

        async with pool.acquire() as conn:
            # Set search path to include ag_catalog
            await conn.execute("SET search_path = ag_catalog, \"$user\", public;")

            try:
                rows = await conn.fetch(age_query)

                # Parse agtype results
                results = []
                for row in rows:
                    # AGE returns results as agtype, need to parse JSON
                    result = json.loads(row["result"])
                    results.append(result)

                self._log.debug("Query executed successfully", result_count=len(results))
                return results

            except Exception as e:
                self._log.error("Cypher query failed", error=str(e), cypher=cypher[:200])
                raise

    async def add_vertex(
        self,
        label: str | EntityType,
        properties: dict[str, Any],
    ) -> int:
        """Add a vertex to the graph.

        Args:
            label: Vertex label (entity type)
            properties: Vertex properties

        Returns:
            Vertex ID (AGE internal ID)

        Example:
            ```python
            vertex_id = await graph.add_vertex(
                label=EntityType.TEST,
                properties={"name": "login_test", "status": "passed"}
            )
            ```
        """
        if isinstance(label, EntityType):
            label = label.value

        entity = GraphEntity(
            entity_type=EntityType(label),
            entity_id=str(uuid.uuid4()),
            properties=properties,
        )

        props_str = entity.to_cypher_properties()

        cypher = f"""
            CREATE (v:{label} {props_str})
            RETURN id(v)
        """

        results = await self.query(cypher)

        if not results:
            raise RuntimeError(f"Failed to create vertex with label {label}")

        # AGE returns vertex ID as integer
        vertex_id = results[0]
        self._log.debug("Created vertex", label=label, vertex_id=vertex_id)

        return vertex_id

    async def add_edge(
        self,
        from_vertex_id: int,
        to_vertex_id: int,
        relationship: str | EdgeType,
        properties: dict[str, Any] | None = None,
    ) -> int:
        """Create an edge between two vertices.

        Args:
            from_vertex_id: Source vertex ID
            to_vertex_id: Target vertex ID
            relationship: Relationship type
            properties: Optional edge properties

        Returns:
            Edge ID (AGE internal ID)

        Example:
            ```python
            edge_id = await graph.add_edge(
                from_vertex_id=test_id,
                to_vertex_id=selector_id,
                relationship=EdgeType.USES,
                properties={"step": 5, "action": "click"}
            )
            ```
        """
        if isinstance(relationship, EdgeType):
            relationship = relationship.value

        props = properties or {}
        rel = GraphRelationship(
            edge_type=EdgeType(relationship),
            from_vertex_id=from_vertex_id,
            to_vertex_id=to_vertex_id,
            properties=props,
        )

        props_str = rel.to_cypher_properties()

        # Use MATCH to find vertices by their internal IDs
        cypher = f"""
            MATCH (a), (b)
            WHERE id(a) = {from_vertex_id} AND id(b) = {to_vertex_id}
            CREATE (a)-[r:{relationship} {props_str}]->(b)
            RETURN id(r)
        """

        results = await self.query(cypher)

        if not results:
            raise RuntimeError(
                f"Failed to create edge from {from_vertex_id} to {to_vertex_id}"
            )

        edge_id = results[0]
        self._log.debug(
            "Created edge",
            relationship=relationship,
            edge_id=edge_id,
            from_vertex_id=from_vertex_id,
            to_vertex_id=to_vertex_id,
        )

        # Store edge mapping
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_edges (edge_id, edge_type, from_vertex_id, to_vertex_id, properties)
                VALUES ($1, $2, $3, $4, $5)
                """,
                edge_id,
                relationship,
                from_vertex_id,
                to_vertex_id,
                json.dumps(props),
            )

        return edge_id

    async def get_neighborhood(
        self,
        vertex_id: int,
        hops: int = 2,
        edge_types: list[str | EdgeType] | None = None,
    ) -> dict[str, Any]:
        """Get the neighborhood of a vertex up to N hops away.

        Args:
            vertex_id: Starting vertex ID
            hops: Maximum number of hops to traverse
            edge_types: Optional filter for specific edge types

        Returns:
            Dictionary with vertices and edges in the neighborhood

        Example:
            ```python
            # Get all connected entities within 2 hops
            neighborhood = await graph.get_neighborhood(test_id, hops=2)
            ```
        """
        edge_filter = ""
        if edge_types:
            types = [t.value if isinstance(t, EdgeType) else t for t in edge_types]
            types_str = "|".join(types)
            edge_filter = f":{types_str}"

        cypher = f"""
            MATCH path = (start)-[{edge_filter}*1..{hops}]-(connected)
            WHERE id(start) = {vertex_id}
            RETURN start, connected, relationships(path) AS rels
        """

        results = await self.query(cypher)

        vertices = {}
        edges = []

        for result in results:
            # Parse start vertex
            if "start" in result:
                start_id = result["start"]["id"]
                vertices[start_id] = result["start"]

            # Parse connected vertex
            if "connected" in result:
                conn_id = result["connected"]["id"]
                vertices[conn_id] = result["connected"]

            # Parse relationships
            if "rels" in result:
                edges.extend(result["rels"])

        return {
            "vertices": list(vertices.values()),
            "edges": edges,
            "total_vertices": len(vertices),
            "total_edges": len(edges),
        }

    # =========================================================================
    # High-Level Entity Operations
    # =========================================================================

    async def add_test(
        self,
        test_id: str,
        name: str,
        file_path: str,
        status: str = "pending",
        **kwargs,
    ) -> int:
        """Add a Test vertex to the graph.

        Args:
            test_id: Test UUID
            name: Test name
            file_path: Path to test file
            status: Test status (passed, failed, skipped)
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        properties = {
            "test_id": test_id,
            "name": name,
            "file_path": file_path,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.TEST, properties)

        # Store mapping
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_test_vertices (test_id, vertex_id, properties)
                VALUES ($1, $2, $3)
                ON CONFLICT (test_id) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                uuid.UUID(test_id),
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    async def add_selector(
        self,
        selector: str,
        selector_type: str = "css",
        **kwargs,
    ) -> int:
        """Add a Selector vertex to the graph.

        Args:
            selector: CSS/XPath selector string
            selector_type: Type of selector (css, xpath)
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        # Check if selector already exists
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT vertex_id FROM graph_selector_vertices WHERE selector_text = $1",
                selector,
            )

            if existing:
                return existing["vertex_id"]

        properties = {
            "selector": selector,
            "selector_type": selector_type,
            "created_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.SELECTOR, properties)

        # Store mapping
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_selector_vertices (selector_text, vertex_id, properties)
                VALUES ($1, $2, $3)
                ON CONFLICT (selector_text) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                selector,
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    async def add_failure(
        self,
        failure_id: str,
        test_run_id: str,
        error_message: str,
        error_type: str,
        **kwargs,
    ) -> int:
        """Add a Failure vertex to the graph.

        Args:
            failure_id: Failure UUID
            test_run_id: Test run UUID
            error_message: Error message
            error_type: Type of error
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        properties = {
            "failure_id": failure_id,
            "test_run_id": test_run_id,
            "error_message": error_message,
            "error_type": error_type,
            "occurred_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.FAILURE, properties)

        # Store mapping
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_failure_vertices (failure_id, test_run_id, vertex_id, properties)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (failure_id) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                uuid.UUID(failure_id),
                uuid.UUID(test_run_id),
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    async def add_code_change(
        self,
        commit_sha: str,
        project_id: str,
        commit_message: str,
        author: str,
        **kwargs,
    ) -> int:
        """Add a CodeChange vertex to the graph.

        Args:
            commit_sha: Git commit SHA
            project_id: Project UUID
            commit_message: Commit message
            author: Commit author
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        # Check if code change already exists
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT vertex_id FROM graph_code_change_vertices
                WHERE project_id = $1 AND commit_sha = $2
                """,
                uuid.UUID(project_id),
                commit_sha,
            )

            if existing:
                return existing["vertex_id"]

        properties = {
            "commit_sha": commit_sha,
            "commit_message": commit_message,
            "author": author,
            "committed_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.CODE_CHANGE, properties)

        # Store mapping
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_code_change_vertices (commit_sha, project_id, vertex_id, properties)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (project_id, commit_sha) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                commit_sha,
                uuid.UUID(project_id),
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    async def add_healing_pattern(
        self,
        pattern_id: str,
        original_selector: str,
        healed_selector: str,
        **kwargs,
    ) -> int:
        """Add a HealingPattern vertex to the graph.

        Args:
            pattern_id: Healing pattern UUID
            original_selector: Original broken selector
            healed_selector: Fixed selector
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        properties = {
            "pattern_id": pattern_id,
            "original_selector": original_selector,
            "healed_selector": healed_selector,
            "created_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.HEALING_PATTERN, properties)

        # Store mapping
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_healing_pattern_vertices (pattern_id, vertex_id, properties)
                VALUES ($1, $2, $3)
                ON CONFLICT (pattern_id) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                uuid.UUID(pattern_id),
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    async def add_page(
        self,
        url: str,
        project_id: str,
        page_title: str | None = None,
        **kwargs,
    ) -> int:
        """Add a Page vertex to the graph.

        Args:
            url: Page URL
            project_id: Project UUID
            page_title: Page title
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        # Check if page already exists
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT vertex_id FROM graph_page_vertices
                WHERE project_id = $1 AND page_url = $2
                """,
                uuid.UUID(project_id),
                url,
            )

            if existing:
                return existing["vertex_id"]

        properties = {
            "url": url,
            "page_title": page_title or url,
            "created_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.PAGE, properties)

        # Store mapping
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_page_vertices (page_url, project_id, vertex_id, properties)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (project_id, page_url) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                url,
                uuid.UUID(project_id),
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    async def add_project(
        self,
        project_id: str,
        name: str,
        **kwargs,
    ) -> int:
        """Add a Project vertex to the graph.

        Args:
            project_id: Project UUID
            name: Project name
            **kwargs: Additional properties

        Returns:
            Vertex ID
        """
        properties = {
            "project_id": project_id,
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            **kwargs,
        }

        vertex_id = await self.add_vertex(EntityType.PROJECT, properties)

        # Store mapping
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO graph_project_vertices (project_id, vertex_id, properties)
                VALUES ($1, $2, $3)
                ON CONFLICT (project_id) DO UPDATE SET
                    vertex_id = EXCLUDED.vertex_id,
                    properties = EXCLUDED.properties,
                    updated_at = NOW()
                """,
                uuid.UUID(project_id),
                vertex_id,
                json.dumps(properties),
            )

        return vertex_id

    # =========================================================================
    # High-Level Query Operations
    # =========================================================================

    async def find_related_failures(
        self,
        test_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find failures related to a test.

        Args:
            test_id: Test UUID
            limit: Maximum number of results

        Returns:
            List of related failure records

        Example:
            Find all failures for a test and their healing patterns.
        """
        # Get vertex ID for test
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
                uuid.UUID(test_id),
            )

            if not row:
                return []

            test_vertex_id = row["vertex_id"]

        cypher = f"""
            MATCH (t:Test)-[:BROKE]-(f:Failure)
            WHERE id(t) = {test_vertex_id}
            OPTIONAL MATCH (f)-[:FIXES]-(hp:HealingPattern)
            RETURN f AS failure, hp AS healing_pattern
            LIMIT {limit}
        """

        results = await self.query(cypher)
        return results

    async def find_code_change_impact(
        self,
        commit_sha: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Find the impact of a code change on tests.

        Multi-hop query to find:
        - Tests affected by the change
        - Failures caused by the change
        - Selectors modified by the change

        Args:
            commit_sha: Git commit SHA
            project_id: Project UUID

        Returns:
            Dictionary with affected tests, failures, and selectors

        Example:
            Impact analysis: CodeChange -> Modified Selectors -> Used by Tests -> Failures
        """
        # Get vertex ID for code change
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT vertex_id FROM graph_code_change_vertices
                WHERE project_id = $1 AND commit_sha = $2
                """,
                uuid.UUID(project_id),
                commit_sha,
            )

            if not row:
                return {
                    "affected_tests": [],
                    "caused_failures": [],
                    "modified_selectors": [],
                }

            change_vertex_id = row["vertex_id"]

        # Multi-hop query for impact analysis
        cypher = f"""
            MATCH (cc:CodeChange)
            WHERE id(cc) = {change_vertex_id}

            OPTIONAL MATCH (cc)-[:MODIFIED]->(s:Selector)<-[:USES]-(t:Test)
            OPTIONAL MATCH (cc)-[:CAUSED]->(f:Failure)
            OPTIONAL MATCH (cc)-[:AFFECTED]->(t2:Test)

            RETURN
                collect(DISTINCT t) AS affected_via_selector,
                collect(DISTINCT t2) AS affected_directly,
                collect(DISTINCT f) AS caused_failures,
                collect(DISTINCT s) AS modified_selectors
        """

        results = await self.query(cypher)

        if not results:
            return {
                "affected_tests": [],
                "caused_failures": [],
                "modified_selectors": [],
            }

        result = results[0]

        # Combine affected tests
        affected_tests = []
        if result.get("affected_via_selector"):
            affected_tests.extend(result["affected_via_selector"])
        if result.get("affected_directly"):
            affected_tests.extend(result["affected_directly"])

        # Remove duplicates
        seen = set()
        unique_tests = []
        for test in affected_tests:
            test_id = test.get("test_id")
            if test_id and test_id not in seen:
                seen.add(test_id)
                unique_tests.append(test)

        return {
            "affected_tests": unique_tests,
            "caused_failures": result.get("caused_failures", []),
            "modified_selectors": result.get("modified_selectors", []),
        }

    async def find_fragile_selectors(
        self,
        project_id: str,
        min_failures: int = 3,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find the most fragile selectors in a project.

        Identifies selectors that:
        - Are frequently modified by code changes
        - Cause multiple test failures
        - Have low healing success rates

        Args:
            project_id: Project UUID
            min_failures: Minimum number of failures to consider
            limit: Maximum number of results

        Returns:
            List of fragile selectors with statistics
        """
        cypher = f"""
            MATCH (p:Project)<-[:BELONGS_TO]-(t:Test)-[:USES]->(s:Selector)
            WHERE p.project_id = '{project_id}'

            OPTIONAL MATCH (s)<-[:MODIFIED]-(cc:CodeChange)
            OPTIONAL MATCH (t)-[:BROKE]-(f:Failure)

            WITH s,
                 count(DISTINCT cc) AS modification_count,
                 count(DISTINCT f) AS failure_count,
                 count(DISTINCT t) AS usage_count

            WHERE failure_count >= {min_failures}

            RETURN s.selector AS selector,
                   s.selector_type AS selector_type,
                   modification_count,
                   failure_count,
                   usage_count,
                   (failure_count::float / NULLIF(usage_count, 0)) AS fragility_score
            ORDER BY fragility_score DESC, failure_count DESC
            LIMIT {limit}
        """

        results = await self.query(cypher)
        return results

    async def find_tests_using_selector(
        self,
        selector: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find all tests that use a specific selector.

        Args:
            selector: CSS/XPath selector
            limit: Maximum number of results

        Returns:
            List of tests using the selector
        """
        cypher = f"""
            MATCH (t:Test)-[u:USES]->(s:Selector)
            WHERE s.selector = '{selector}'
            RETURN t, u.step_number AS step, u.action AS action
            ORDER BY t.name
            LIMIT {limit}
        """

        results = await self.query(cypher)
        return results

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with vertex and edge counts
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            metadata = await conn.fetchrow(
                "SELECT * FROM graph_metadata WHERE graph_name = $1",
                self.graph_name,
            )

            if not metadata:
                return {
                    "total_vertices": 0,
                    "total_edges": 0,
                    "graph_name": self.graph_name,
                }

            return {
                "total_vertices": metadata["total_vertices"],
                "total_edges": metadata["total_edges"],
                "graph_name": metadata["graph_name"],
                "schema_version": metadata["schema_version"],
                "last_compaction_at": metadata["last_compaction_at"],
            }


# =========================================================================
# Global Instance Management
# =========================================================================

_graph_store: GraphStore | None = None


def get_graph_store(
    database_url: str | None = None,
    graph_name: str = "argus_knowledge",
) -> GraphStore:
    """Get or create the global graph store instance.

    Args:
        database_url: Optional database URL
        graph_name: Graph name

    Returns:
        GraphStore instance
    """
    global _graph_store

    if _graph_store is None:
        _graph_store = GraphStore(
            database_url=database_url,
            graph_name=graph_name,
        )

    return _graph_store


def reset_graph_store() -> None:
    """Reset the global graph store instance."""
    global _graph_store
    _graph_store = None


async def init_graph_store(
    database_url: str | None = None,
    graph_name: str = "argus_knowledge",
) -> GraphStore:
    """Initialize and return the graph store.

    Args:
        database_url: Optional database URL
        graph_name: Graph name

    Returns:
        Initialized GraphStore instance
    """
    store = get_graph_store(database_url=database_url, graph_name=graph_name)

    # Verify connection
    await store._get_pool()

    logger.info("Graph store initialized successfully", graph_name=graph_name)
    return store
