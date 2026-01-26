"""Data migration utilities for populating the knowledge graph.

Migrates existing data from relational tables to the knowledge graph:
- Tests -> Test vertices
- Healing patterns -> HealingPattern vertices
- Test failures -> Failure vertices
- Creates relationships based on existing data
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from .graph_store import GraphStore
from .schema import EdgeType

logger = structlog.get_logger(__name__)


class GraphMigration:
    """Handles migration of existing data to the knowledge graph."""

    def __init__(self, graph_store: GraphStore):
        """Initialize migration with graph store.

        Args:
            graph_store: GraphStore instance
        """
        self.graph = graph_store
        self._log = logger.bind(component="graph_migration")

    async def migrate_all(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Migrate all data to the knowledge graph.

        Args:
            project_id: Optional project ID to migrate (if None, migrates all)
            batch_size: Number of records to process per batch

        Returns:
            Dictionary with migration statistics
        """
        stats = {
            "projects": 0,
            "tests": 0,
            "selectors": 0,
            "failures": 0,
            "healing_patterns": 0,
            "pages": 0,
            "code_changes": 0,
            "edges": 0,
        }

        self._log.info("Starting graph migration", project_id=project_id)

        # Migrate in order (to satisfy dependencies)
        stats["projects"] = await self.migrate_projects(project_id, batch_size)
        stats["tests"] = await self.migrate_tests(project_id, batch_size)
        stats["healing_patterns"] = await self.migrate_healing_patterns(
            project_id, batch_size
        )
        stats["failures"] = await self.migrate_test_failures(project_id, batch_size)
        stats["selectors"] = await self.migrate_selectors(project_id, batch_size)
        stats["pages"] = await self.migrate_pages(project_id, batch_size)

        # Create relationships
        stats["edges"] += await self.create_test_relationships(project_id, batch_size)
        stats["edges"] += await self.create_failure_relationships(
            project_id, batch_size
        )
        stats["edges"] += await self.create_healing_relationships(
            project_id, batch_size
        )

        self._log.info("Graph migration completed", stats=stats)
        return stats

    async def migrate_projects(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Migrate projects to graph.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of projects migrated
        """
        pool = await self.graph._get_pool()

        where_clause = ""
        params = []
        if project_id:
            where_clause = "WHERE id = $1"
            params = [uuid.UUID(project_id)]

        async with pool.acquire() as conn:
            query = f"""
                SELECT id, name, repository_url, created_at
                FROM projects
                {where_clause}
                LIMIT {batch_size}
            """

            rows = await conn.fetch(query, *params)

            count = 0
            for row in rows:
                try:
                    await self.graph.add_project(
                        project_id=str(row["id"]),
                        name=row["name"],
                        repository_url=row.get("repository_url"),
                    )
                    count += 1
                except Exception as e:
                    self._log.error(
                        "Failed to migrate project",
                        project_id=str(row["id"]),
                        error=str(e),
                    )

            self._log.info("Migrated projects", count=count)
            return count

    async def migrate_tests(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Migrate tests to graph.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of tests migrated
        """
        pool = await self.graph._get_pool()

        where_clause = ""
        params = []
        if project_id:
            where_clause = "WHERE project_id = $1"
            params = [uuid.UUID(project_id)]

        async with pool.acquire() as conn:
            query = f"""
                SELECT id, project_id, name, file_path, status, created_at
                FROM tests
                {where_clause}
                LIMIT {batch_size}
            """

            rows = await conn.fetch(query, *params)

            count = 0
            for row in rows:
                try:
                    test_vertex_id = await self.graph.add_test(
                        test_id=str(row["id"]),
                        name=row["name"],
                        file_path=row.get("file_path", ""),
                        status=row.get("status", "pending"),
                    )

                    # Create BELONGS_TO relationship with project
                    project_vertex_id = await self._get_project_vertex_id(
                        str(row["project_id"])
                    )
                    if project_vertex_id:
                        await self.graph.add_edge(
                            from_vertex_id=test_vertex_id,
                            to_vertex_id=project_vertex_id,
                            relationship=EdgeType.BELONGS_TO,
                        )

                    count += 1
                except Exception as e:
                    self._log.error(
                        "Failed to migrate test",
                        test_id=str(row["id"]),
                        error=str(e),
                    )

            self._log.info("Migrated tests", count=count)
            return count

    async def migrate_healing_patterns(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Migrate healing patterns to graph.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of patterns migrated
        """
        pool = await self.graph._get_pool()

        where_clause = ""
        params = []
        if project_id:
            where_clause = "WHERE project_id = $1"
            params = [uuid.UUID(project_id)]

        async with pool.acquire() as conn:
            query = f"""
                SELECT id, original_selector, healed_selector, error_type,
                       success_count, failure_count, confidence, metadata, created_at
                FROM healing_patterns
                {where_clause}
                LIMIT {batch_size}
            """

            rows = await conn.fetch(query, *params)

            count = 0
            for row in rows:
                try:
                    await self.graph.add_healing_pattern(
                        pattern_id=str(row["id"]),
                        original_selector=row["original_selector"],
                        healed_selector=row["healed_selector"],
                        confidence=float(row.get("confidence", 0.5)),
                        success_count=row.get("success_count", 0),
                        failure_count=row.get("failure_count", 0),
                        error_type=row.get("error_type"),
                    )
                    count += 1
                except Exception as e:
                    self._log.error(
                        "Failed to migrate healing pattern",
                        pattern_id=str(row["id"]),
                        error=str(e),
                    )

            self._log.info("Migrated healing patterns", count=count)
            return count

    async def migrate_test_failures(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Migrate test failures to graph.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of failures migrated
        """
        pool = await self.graph._get_pool()

        where_clause = ""
        params = []
        if project_id:
            where_clause = "WHERE tr.project_id = $1"
            params = [uuid.UUID(project_id)]

        async with pool.acquire() as conn:
            # Query test_failure_patterns table
            query = f"""
                SELECT tfp.id, tfp.test_id, tfp.error_message, tfp.error_type,
                       tfp.selector, tfp.healed_selector, tfp.created_at
                FROM test_failure_patterns tfp
                LIMIT {batch_size}
            """

            rows = await conn.fetch(query)

            count = 0
            for row in rows:
                try:
                    failure_id = str(row["id"])
                    test_run_id = str(row["test_id"]) if row["test_id"] else str(
                        uuid.uuid4()
                    )

                    await self.graph.add_failure(
                        failure_id=failure_id,
                        test_run_id=test_run_id,
                        error_message=row["error_message"],
                        error_type=row.get("error_type", "unknown"),
                        selector=row.get("selector"),
                        healed_selector=row.get("healed_selector"),
                    )
                    count += 1
                except Exception as e:
                    self._log.error(
                        "Failed to migrate failure",
                        failure_id=str(row["id"]),
                        error=str(e),
                    )

            self._log.info("Migrated test failures", count=count)
            return count

    async def migrate_selectors(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Extract and migrate selectors from test steps.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of selectors migrated
        """
        pool = await self.graph._get_pool()

        # Extract unique selectors from healing_patterns
        async with pool.acquire() as conn:
            query = """
                SELECT DISTINCT original_selector AS selector
                FROM healing_patterns
                WHERE original_selector IS NOT NULL
                UNION
                SELECT DISTINCT healed_selector AS selector
                FROM healing_patterns
                WHERE healed_selector IS NOT NULL
                LIMIT $1
            """

            rows = await conn.fetch(query, batch_size)

            count = 0
            for row in rows:
                try:
                    if row["selector"]:
                        await self.graph.add_selector(
                            selector=row["selector"],
                            selector_type="css",  # Default to CSS
                        )
                        count += 1
                except Exception as e:
                    self._log.debug(
                        "Failed to migrate selector",
                        selector=row["selector"],
                        error=str(e),
                    )

            self._log.info("Migrated selectors", count=count)
            return count

    async def migrate_pages(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Migrate pages from healing patterns.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of pages migrated
        """
        pool = await self.graph._get_pool()

        where_clause = ""
        params = [batch_size]
        if project_id:
            where_clause = "WHERE project_id = $2"
            params.append(uuid.UUID(project_id))

        async with pool.acquire() as conn:
            query = f"""
                SELECT DISTINCT page_url, project_id
                FROM healing_patterns
                WHERE page_url IS NOT NULL
                {where_clause}
                LIMIT $1
            """

            rows = await conn.fetch(query, *params)

            count = 0
            for row in rows:
                try:
                    if row["page_url"] and row["project_id"]:
                        await self.graph.add_page(
                            url=row["page_url"],
                            project_id=str(row["project_id"]),
                        )
                        count += 1
                except Exception as e:
                    self._log.debug(
                        "Failed to migrate page",
                        url=row["page_url"],
                        error=str(e),
                    )

            self._log.info("Migrated pages", count=count)
            return count

    async def create_test_relationships(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Create USES relationships between tests and selectors.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of relationships created
        """
        # This would require parsing test steps to extract selectors
        # For now, we'll create relationships based on healing patterns
        pool = await self.graph._get_pool()

        count = 0
        async with pool.acquire() as conn:
            # Get tests and their associated selectors from healing patterns
            query = """
                SELECT DISTINCT t.id AS test_id, hp.original_selector, hp.healed_selector
                FROM tests t
                JOIN test_failure_patterns tfp ON tfp.test_id = t.id
                JOIN healing_patterns hp ON hp.original_selector = tfp.selector
                LIMIT $1
            """

            rows = await conn.fetch(query, batch_size)

            for row in rows:
                try:
                    test_vertex_id = await self._get_test_vertex_id(
                        str(row["test_id"])
                    )
                    if not test_vertex_id:
                        continue

                    # Create USES edge for original selector
                    if row["original_selector"]:
                        selector_vertex_id = await self._get_selector_vertex_id(
                            row["original_selector"]
                        )
                        if selector_vertex_id:
                            await self.graph.add_edge(
                                from_vertex_id=test_vertex_id,
                                to_vertex_id=selector_vertex_id,
                                relationship=EdgeType.USES,
                            )
                            count += 1

                except Exception as e:
                    self._log.debug("Failed to create test relationship", error=str(e))

        self._log.info("Created test relationships", count=count)
        return count

    async def create_failure_relationships(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Create BROKE relationships between failures and tests.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of relationships created
        """
        pool = await self.graph._get_pool()

        count = 0
        async with pool.acquire() as conn:
            query = """
                SELECT tfp.id AS failure_id, tfp.test_id
                FROM test_failure_patterns tfp
                WHERE tfp.test_id IS NOT NULL
                LIMIT $1
            """

            rows = await conn.fetch(query, batch_size)

            for row in rows:
                try:
                    failure_vertex_id = await self._get_failure_vertex_id(
                        str(row["failure_id"])
                    )
                    test_vertex_id = await self._get_test_vertex_id(
                        str(row["test_id"])
                    )

                    if failure_vertex_id and test_vertex_id:
                        await self.graph.add_edge(
                            from_vertex_id=failure_vertex_id,
                            to_vertex_id=test_vertex_id,
                            relationship=EdgeType.BROKE,
                            properties={"occurred_at": datetime.now(UTC).isoformat()},
                        )
                        count += 1

                except Exception as e:
                    self._log.debug(
                        "Failed to create failure relationship", error=str(e)
                    )

        self._log.info("Created failure relationships", count=count)
        return count

    async def create_healing_relationships(
        self,
        project_id: str | None = None,
        batch_size: int = 100,
    ) -> int:
        """Create REPLACES relationships for healing patterns.

        Args:
            project_id: Optional specific project ID
            batch_size: Batch size

        Returns:
            Number of relationships created
        """
        pool = await self.graph._get_pool()

        count = 0
        async with pool.acquire() as conn:
            query = """
                SELECT id, original_selector, healed_selector, confidence
                FROM healing_patterns
                WHERE original_selector IS NOT NULL AND healed_selector IS NOT NULL
                LIMIT $1
            """

            rows = await conn.fetch(query, batch_size)

            for row in rows:
                try:
                    pattern_vertex_id = await self._get_healing_pattern_vertex_id(
                        str(row["id"])
                    )
                    old_selector_vertex_id = await self._get_selector_vertex_id(
                        row["original_selector"]
                    )
                    new_selector_vertex_id = await self._get_selector_vertex_id(
                        row["healed_selector"]
                    )

                    if pattern_vertex_id and old_selector_vertex_id:
                        # HealingPattern -> REPLACES -> old Selector
                        await self.graph.add_edge(
                            from_vertex_id=pattern_vertex_id,
                            to_vertex_id=old_selector_vertex_id,
                            relationship=EdgeType.REPLACES,
                            properties={
                                "old_selector": row["original_selector"],
                                "new_selector": row["healed_selector"],
                                "confidence": float(row.get("confidence", 0.5)),
                            },
                        )
                        count += 1

                except Exception as e:
                    self._log.debug(
                        "Failed to create healing relationship", error=str(e)
                    )

        self._log.info("Created healing relationships", count=count)
        return count

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_test_vertex_id(self, test_id: str) -> int | None:
        """Get vertex ID for a test."""
        pool = await self.graph._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
                uuid.UUID(test_id),
            )
            return row["vertex_id"] if row else None

    async def _get_selector_vertex_id(self, selector: str) -> int | None:
        """Get vertex ID for a selector."""
        pool = await self.graph._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_selector_vertices WHERE selector_text = $1",
                selector,
            )
            return row["vertex_id"] if row else None

    async def _get_failure_vertex_id(self, failure_id: str) -> int | None:
        """Get vertex ID for a failure."""
        pool = await self.graph._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_failure_vertices WHERE failure_id = $1",
                uuid.UUID(failure_id),
            )
            return row["vertex_id"] if row else None

    async def _get_healing_pattern_vertex_id(self, pattern_id: str) -> int | None:
        """Get vertex ID for a healing pattern."""
        pool = await self.graph._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_healing_pattern_vertices WHERE pattern_id = $1",
                uuid.UUID(pattern_id),
            )
            return row["vertex_id"] if row else None

    async def _get_project_vertex_id(self, project_id: str) -> int | None:
        """Get vertex ID for a project."""
        pool = await self.graph._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_project_vertices WHERE project_id = $1",
                uuid.UUID(project_id),
            )
            return row["vertex_id"] if row else None


async def run_migration(
    database_url: str | None = None,
    project_id: str | None = None,
) -> dict[str, int]:
    """Run the complete graph migration.

    Args:
        database_url: Optional database URL
        project_id: Optional project ID to migrate

    Returns:
        Migration statistics
    """
    from . import init_graph_store

    graph = await init_graph_store(database_url=database_url)
    migration = GraphMigration(graph)

    stats = await migration.migrate_all(project_id=project_id)

    # Update graph statistics
    pool = await graph._get_pool()
    async with pool.acquire() as conn:
        await conn.execute("SELECT update_graph_stats()")

    return stats
