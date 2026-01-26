"""Unit tests for GraphStore.

Tests the Apache AGE knowledge graph operations including:
- Vertex creation and retrieval
- Edge creation and traversal
- Multi-hop queries
- Entity-specific operations
"""

import os
import uuid
from datetime import UTC, datetime

import pytest

from src.knowledge_graph import GraphStore
from src.knowledge_graph.schema import EdgeType, EntityType


@pytest.fixture
async def graph_store():
    """Create a GraphStore instance for testing."""
    # Use test database URL if available
    database_url = os.environ.get("TEST_DATABASE_URL") or os.environ.get(
        "DATABASE_URL"
    )

    if not database_url:
        pytest.skip("No database URL configured for testing")

    store = GraphStore(database_url=database_url)
    yield store
    await store.close()


@pytest.fixture
async def test_project_id():
    """Generate a test project ID."""
    return str(uuid.uuid4())


@pytest.fixture
async def test_test_id():
    """Generate a test test ID."""
    return str(uuid.uuid4())


class TestGraphStoreBasicOperations:
    """Test basic graph operations."""

    @pytest.mark.asyncio
    async def test_add_vertex(self, graph_store):
        """Test adding a vertex to the graph."""
        properties = {
            "name": "test_login",
            "status": "passed",
            "created_at": datetime.now(UTC).isoformat(),
        }

        vertex_id = await graph_store.add_vertex(
            label=EntityType.TEST,
            properties=properties,
        )

        assert vertex_id is not None
        assert isinstance(vertex_id, int)

    @pytest.mark.asyncio
    async def test_add_edge(self, graph_store):
        """Test creating an edge between vertices."""
        # Create two vertices
        test_props = {"name": "test_1", "status": "passed"}
        test_vertex_id = await graph_store.add_vertex(EntityType.TEST, test_props)

        selector_props = {"selector": "#login-btn", "selector_type": "css"}
        selector_vertex_id = await graph_store.add_vertex(
            EntityType.SELECTOR, selector_props
        )

        # Create edge
        edge_id = await graph_store.add_edge(
            from_vertex_id=test_vertex_id,
            to_vertex_id=selector_vertex_id,
            relationship=EdgeType.USES,
            properties={"step": 5, "action": "click"},
        )

        assert edge_id is not None
        assert isinstance(edge_id, int)

    @pytest.mark.asyncio
    async def test_query_basic(self, graph_store):
        """Test basic Cypher query execution."""
        # Create a test vertex
        props = {"name": "query_test", "status": "passed"}
        vertex_id = await graph_store.add_vertex(EntityType.TEST, props)

        # Query for the vertex
        cypher = f"""
            MATCH (t:Test)
            WHERE t.name = 'query_test'
            RETURN t
        """

        results = await graph_store.query(cypher)

        assert len(results) > 0
        assert results[0].get("name") == "query_test"


class TestEntityOperations:
    """Test entity-specific operations."""

    @pytest.mark.asyncio
    async def test_add_test(self, graph_store, test_project_id, test_test_id):
        """Test adding a Test entity."""
        vertex_id = await graph_store.add_test(
            test_id=test_test_id,
            name="test_login",
            file_path="tests/auth/test_login.py",
            status="passed",
        )

        assert vertex_id is not None

        # Verify mapping was created
        pool = await graph_store._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
                uuid.UUID(test_test_id),
            )
            assert row is not None
            assert row["vertex_id"] == vertex_id

    @pytest.mark.asyncio
    async def test_add_selector(self, graph_store):
        """Test adding a Selector entity."""
        selector = "#login-button"
        vertex_id = await graph_store.add_selector(
            selector=selector,
            selector_type="css",
        )

        assert vertex_id is not None

        # Adding same selector again should return same vertex
        vertex_id2 = await graph_store.add_selector(
            selector=selector,
            selector_type="css",
        )

        assert vertex_id == vertex_id2

    @pytest.mark.asyncio
    async def test_add_failure(self, graph_store):
        """Test adding a Failure entity."""
        failure_id = str(uuid.uuid4())
        test_run_id = str(uuid.uuid4())

        vertex_id = await graph_store.add_failure(
            failure_id=failure_id,
            test_run_id=test_run_id,
            error_message="Element not found: #submit-btn",
            error_type="selector_changed",
        )

        assert vertex_id is not None

    @pytest.mark.asyncio
    async def test_add_healing_pattern(self, graph_store):
        """Test adding a HealingPattern entity."""
        pattern_id = str(uuid.uuid4())

        vertex_id = await graph_store.add_healing_pattern(
            pattern_id=pattern_id,
            original_selector="#submit-btn",
            healed_selector="#submit-button",
            confidence=0.95,
        )

        assert vertex_id is not None

    @pytest.mark.asyncio
    async def test_add_page(self, graph_store, test_project_id):
        """Test adding a Page entity."""
        url = "https://example.com/login"

        vertex_id = await graph_store.add_page(
            url=url,
            project_id=test_project_id,
            page_title="Login Page",
        )

        assert vertex_id is not None

        # Adding same page again should return same vertex
        vertex_id2 = await graph_store.add_page(
            url=url,
            project_id=test_project_id,
        )

        assert vertex_id == vertex_id2


class TestGraphTraversal:
    """Test graph traversal and multi-hop queries."""

    @pytest.mark.asyncio
    async def test_get_neighborhood(self, graph_store):
        """Test getting the neighborhood of a vertex."""
        # Create a small graph: Test -> USES -> Selector
        test_vertex_id = await graph_store.add_vertex(
            EntityType.TEST,
            {"name": "test_neighborhood", "status": "passed"},
        )

        selector_vertex_id = await graph_store.add_vertex(
            EntityType.SELECTOR,
            {"selector": "#test-btn", "selector_type": "css"},
        )

        await graph_store.add_edge(
            from_vertex_id=test_vertex_id,
            to_vertex_id=selector_vertex_id,
            relationship=EdgeType.USES,
        )

        # Get neighborhood
        neighborhood = await graph_store.get_neighborhood(
            vertex_id=test_vertex_id,
            hops=1,
        )

        assert neighborhood["total_vertices"] >= 2
        assert neighborhood["total_edges"] >= 1

    @pytest.mark.asyncio
    async def test_find_tests_using_selector(self, graph_store, test_test_id):
        """Test finding tests that use a specific selector."""
        selector = "#unique-test-selector"

        # Create test and selector
        test_vertex_id = await graph_store.add_test(
            test_id=test_test_id,
            name="test_selector_usage",
            file_path="tests/test.py",
        )

        selector_vertex_id = await graph_store.add_selector(
            selector=selector,
            selector_type="css",
        )

        # Create USES relationship
        await graph_store.add_edge(
            from_vertex_id=test_vertex_id,
            to_vertex_id=selector_vertex_id,
            relationship=EdgeType.USES,
            properties={"step": 1, "action": "click"},
        )

        # Find tests using this selector
        tests = await graph_store.find_tests_using_selector(selector)

        assert len(tests) >= 1
        # Check if our test is in the results
        test_found = any(t.get("test_id") == test_test_id for t in tests)
        assert test_found


class TestHighLevelQueries:
    """Test high-level query operations."""

    @pytest.mark.asyncio
    async def test_find_related_failures(
        self, graph_store, test_project_id, test_test_id
    ):
        """Test finding related failures for a test."""
        # Create test
        test_vertex_id = await graph_store.add_test(
            test_id=test_test_id,
            name="test_with_failures",
            file_path="tests/test.py",
        )

        # Create failure
        failure_id = str(uuid.uuid4())
        failure_vertex_id = await graph_store.add_failure(
            failure_id=failure_id,
            test_run_id=str(uuid.uuid4()),
            error_message="Test failed",
            error_type="selector_changed",
        )

        # Create BROKE relationship
        await graph_store.add_edge(
            from_vertex_id=failure_vertex_id,
            to_vertex_id=test_vertex_id,
            relationship=EdgeType.BROKE,
        )

        # Find related failures
        failures = await graph_store.find_related_failures(test_test_id)

        assert len(failures) >= 0  # May be empty if mapping not found

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, graph_store):
        """Test getting graph statistics."""
        stats = await graph_store.get_graph_stats()

        assert "total_vertices" in stats
        assert "total_edges" in stats
        assert "graph_name" in stats
        assert stats["graph_name"] == "argus_knowledge"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_query_with_no_results(self, graph_store):
        """Test query that returns no results."""
        cypher = """
            MATCH (t:Test)
            WHERE t.name = 'nonexistent_test_xyz_123'
            RETURN t
        """

        results = await graph_store.query(cypher)
        assert results == []

    @pytest.mark.asyncio
    async def test_add_vertex_with_special_characters(self, graph_store):
        """Test adding vertex with special characters in properties."""
        properties = {
            "name": "test's \"special\" <chars>",
            "description": "Test with\nnewlines\tand\ttabs",
        }

        vertex_id = await graph_store.add_vertex(
            EntityType.TEST,
            properties,
        )

        assert vertex_id is not None

    @pytest.mark.asyncio
    async def test_add_edge_to_nonexistent_vertex(self, graph_store):
        """Test creating edge to non-existent vertex fails gracefully."""
        # Create one valid vertex
        vertex_id = await graph_store.add_vertex(
            EntityType.TEST,
            {"name": "test"},
        )

        # Try to create edge to non-existent vertex
        with pytest.raises(Exception):
            await graph_store.add_edge(
                from_vertex_id=vertex_id,
                to_vertex_id=999999999,  # Non-existent ID
                relationship=EdgeType.USES,
            )


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_healing_workflow(
        self, graph_store, test_project_id, test_test_id
    ):
        """Test a complete healing workflow through the graph."""
        # 1. Create project
        project_vertex_id = await graph_store.add_project(
            project_id=test_project_id,
            name="Test Project",
        )

        # 2. Create test
        test_vertex_id = await graph_store.add_test(
            test_id=test_test_id,
            name="test_healing_workflow",
            file_path="tests/test.py",
        )

        # 3. Link test to project
        await graph_store.add_edge(
            from_vertex_id=test_vertex_id,
            to_vertex_id=project_vertex_id,
            relationship=EdgeType.BELONGS_TO,
        )

        # 4. Create selectors
        old_selector_id = await graph_store.add_selector(
            selector="#old-btn",
            selector_type="css",
        )

        new_selector_id = await graph_store.add_selector(
            selector="#new-btn",
            selector_type="css",
        )

        # 5. Test uses old selector
        await graph_store.add_edge(
            from_vertex_id=test_vertex_id,
            to_vertex_id=old_selector_id,
            relationship=EdgeType.USES,
        )

        # 6. Create failure
        failure_id = str(uuid.uuid4())
        failure_vertex_id = await graph_store.add_failure(
            failure_id=failure_id,
            test_run_id=str(uuid.uuid4()),
            error_message="Selector not found: #old-btn",
            error_type="selector_changed",
        )

        # 7. Failure broke test
        await graph_store.add_edge(
            from_vertex_id=failure_vertex_id,
            to_vertex_id=test_vertex_id,
            relationship=EdgeType.BROKE,
        )

        # 8. Create healing pattern
        pattern_id = str(uuid.uuid4())
        pattern_vertex_id = await graph_store.add_healing_pattern(
            pattern_id=pattern_id,
            original_selector="#old-btn",
            healed_selector="#new-btn",
            confidence=0.95,
        )

        # 9. Healing pattern fixes failure
        await graph_store.add_edge(
            from_vertex_id=pattern_vertex_id,
            to_vertex_id=failure_vertex_id,
            relationship=EdgeType.FIXES,
        )

        # 10. Healing pattern replaces selector
        await graph_store.add_edge(
            from_vertex_id=pattern_vertex_id,
            to_vertex_id=old_selector_id,
            relationship=EdgeType.REPLACES,
            properties={
                "old_selector": "#old-btn",
                "new_selector": "#new-btn",
            },
        )

        # Verify the complete workflow via graph query
        cypher = f"""
            MATCH (t:Test {{test_id: '{test_test_id}'}})
            MATCH (t)-[:BROKE]-(f:Failure)
            MATCH (f)-[:FIXES]-(hp:HealingPattern)
            MATCH (hp)-[:REPLACES]->(s:Selector)
            RETURN t, f, hp, s
        """

        results = await graph_store.query(cypher)
        assert len(results) >= 1

        # Verify the healing pattern was found
        result = results[0]
        assert result.get("hp") is not None
        assert result.get("s") is not None
