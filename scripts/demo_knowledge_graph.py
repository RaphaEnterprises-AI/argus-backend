#!/usr/bin/env python3
"""Demo script showing knowledge graph capabilities.

This script demonstrates the power of graph-based reasoning for
test failure analysis and healing.
"""

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_graph import GraphStore, init_graph_store
from src.knowledge_graph.schema import EdgeType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def demo_basic_operations(graph: GraphStore):
    """Demo basic graph operations."""
    logger.info("\n=== DEMO 1: Basic Graph Operations ===\n")

    # Create a project
    project_id = str(uuid.uuid4())
    project_vertex_id = await graph.add_project(
        project_id=project_id,
        name="E-commerce Platform",
        repository_url="https://github.com/example/ecommerce",
    )
    logger.info(f"Created project vertex: {project_vertex_id}")

    # Create a test
    test_id = str(uuid.uuid4())
    test_vertex_id = await graph.add_test(
        test_id=test_id,
        name="test_checkout_flow",
        file_path="tests/e2e/test_checkout.py",
        status="passed",
    )
    logger.info(f"Created test vertex: {test_vertex_id}")

    # Link test to project
    await graph.add_edge(
        from_vertex_id=test_vertex_id,
        to_vertex_id=project_vertex_id,
        relationship=EdgeType.BELONGS_TO,
    )
    logger.info("Linked test to project")

    # Create selectors
    selectors = [
        "#add-to-cart-btn",
        "#checkout-btn",
        "#payment-form",
        "#confirm-order-btn",
    ]

    for idx, selector in enumerate(selectors):
        selector_vertex_id = await graph.add_selector(
            selector=selector,
            selector_type="css",
        )

        await graph.add_edge(
            from_vertex_id=test_vertex_id,
            to_vertex_id=selector_vertex_id,
            relationship=EdgeType.USES,
            properties={"step": idx + 1, "action": "click"},
        )
        logger.info(f"Created selector: {selector}")

    logger.info("\n✓ Basic operations completed\n")
    return project_id, test_id


async def demo_failure_healing_workflow(graph: GraphStore, test_id: str):
    """Demo a complete failure and healing workflow."""
    logger.info("\n=== DEMO 2: Failure & Healing Workflow ===\n")

    # Simulate a failure
    failure_id = str(uuid.uuid4())
    failure_vertex_id = await graph.add_failure(
        failure_id=failure_id,
        test_run_id=str(uuid.uuid4()),
        error_message="Element not found: #checkout-btn",
        error_type="selector_changed",
        auto_healed=False,
    )
    logger.info(f"Created failure: {failure_id}")

    # Get test vertex ID
    pool = await graph._get_pool()
    async with pool.acquire() as conn:
        test_row = await conn.fetchrow(
            "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
            uuid.UUID(test_id),
        )
        test_vertex_id = test_row["vertex_id"]

    # Link failure to test
    await graph.add_edge(
        from_vertex_id=failure_vertex_id,
        to_vertex_id=test_vertex_id,
        relationship=EdgeType.BROKE,
    )
    logger.info("Linked failure to test")

    # Create healing pattern
    pattern_id = str(uuid.uuid4())
    pattern_vertex_id = await graph.add_healing_pattern(
        pattern_id=pattern_id,
        original_selector="#checkout-btn",
        healed_selector="#checkout-button",
        confidence=0.95,
        success_count=1,
        healing_method="code_aware",
    )
    logger.info(f"Created healing pattern: {pattern_id}")

    # Link healing pattern to failure
    await graph.add_edge(
        from_vertex_id=pattern_vertex_id,
        to_vertex_id=failure_vertex_id,
        relationship=EdgeType.FIXES,
        properties={"success": True},
    )
    logger.info("Linked healing pattern to failure")

    # Get old selector vertex
    async with pool.acquire() as conn:
        old_selector_row = await conn.fetchrow(
            "SELECT vertex_id FROM graph_selector_vertices WHERE selector_text = $1",
            "#checkout-btn",
        )
        if old_selector_row:
            old_selector_vertex_id = old_selector_row["vertex_id"]

            # Link healing pattern to replaced selector
            await graph.add_edge(
                from_vertex_id=pattern_vertex_id,
                to_vertex_id=old_selector_vertex_id,
                relationship=EdgeType.REPLACES,
                properties={
                    "old_selector": "#checkout-btn",
                    "new_selector": "#checkout-button",
                },
            )
            logger.info("Linked healing pattern to replaced selector")

    logger.info("\n✓ Failure & healing workflow completed\n")
    return failure_id, pattern_id


async def demo_multi_hop_queries(graph: GraphStore, test_id: str):
    """Demo multi-hop graph queries."""
    logger.info("\n=== DEMO 3: Multi-Hop Graph Queries ===\n")

    # Query 1: Find all selectors used by a test
    logger.info("Query 1: Find all selectors used by test")
    cypher = f"""
        MATCH (t:Test {{test_id: '{test_id}'}})-[u:USES]->(s:Selector)
        RETURN s.selector AS selector, u.step AS step, u.action AS action
        ORDER BY u.step
    """
    results = await graph.query(cypher)
    logger.info(f"Found {len(results)} selectors:")
    for result in results:
        logger.info(f"  Step {result.get('step')}: {result.get('selector')} ({result.get('action')})")

    # Query 2: Find healing patterns for failed selectors
    logger.info("\nQuery 2: Find healing patterns for failures")
    cypher = f"""
        MATCH (t:Test {{test_id: '{test_id}'}})-[:BROKE]-(f:Failure)
        MATCH (f)-[:FIXES]-(hp:HealingPattern)
        RETURN f.error_message AS error,
               hp.original_selector AS old_selector,
               hp.healed_selector AS new_selector,
               hp.confidence AS confidence
    """
    results = await graph.query(cypher)
    logger.info(f"Found {len(results)} healing patterns:")
    for result in results:
        logger.info(f"  Error: {result.get('error')}")
        logger.info(f"  Fix: {result.get('old_selector')} → {result.get('new_selector')}")
        logger.info(f"  Confidence: {result.get('confidence'):.0%}")

    # Query 3: Get test neighborhood
    logger.info("\nQuery 3: Get test neighborhood (2 hops)")
    pool = await graph._get_pool()
    async with pool.acquire() as conn:
        test_row = await conn.fetchrow(
            "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
            uuid.UUID(test_id),
        )
        if test_row:
            neighborhood = await graph.get_neighborhood(
                vertex_id=test_row["vertex_id"],
                hops=2,
            )
            logger.info(f"  Vertices in neighborhood: {neighborhood['total_vertices']}")
            logger.info(f"  Edges in neighborhood: {neighborhood['total_edges']}")

    logger.info("\n✓ Multi-hop queries completed\n")


async def demo_code_change_impact(graph: GraphStore, project_id: str, test_id: str):
    """Demo code change impact analysis."""
    logger.info("\n=== DEMO 4: Code Change Impact Analysis ===\n")

    # Create a code change
    commit_sha = "abc123def456"
    code_change_vertex_id = await graph.add_code_change(
        commit_sha=commit_sha,
        project_id=project_id,
        commit_message="Refactor checkout button selector",
        author="developer@example.com",
        files_changed=["src/components/Checkout.tsx"],
        lines_added=5,
        lines_deleted=3,
    )
    logger.info(f"Created code change: {commit_sha}")

    # Get selector vertex
    pool = await graph._get_pool()
    async with pool.acquire() as conn:
        selector_row = await conn.fetchrow(
            "SELECT vertex_id FROM graph_selector_vertices WHERE selector_text = $1",
            "#checkout-btn",
        )
        if selector_row:
            # Link code change to modified selector
            await graph.add_edge(
                from_vertex_id=code_change_vertex_id,
                to_vertex_id=selector_row["vertex_id"],
                relationship=EdgeType.MODIFIED,
                properties={
                    "change_type": "renamed",
                    "old_value": "#checkout-btn",
                    "new_value": "#checkout-button",
                },
            )
            logger.info("Linked code change to modified selector")

        # Get test vertex
        test_row = await conn.fetchrow(
            "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
            uuid.UUID(test_id),
        )
        if test_row:
            # Link code change to affected test
            await graph.add_edge(
                from_vertex_id=code_change_vertex_id,
                to_vertex_id=test_row["vertex_id"],
                relationship=EdgeType.AFFECTED,
                properties={"impact_type": "direct", "confidence": 0.95},
            )
            logger.info("Linked code change to affected test")

    # Query impact
    impact = await graph.find_code_change_impact(commit_sha, project_id)

    logger.info(f"\nImpact Analysis for {commit_sha}:")
    logger.info(f"  Affected tests: {len(impact['affected_tests'])}")
    logger.info(f"  Caused failures: {len(impact['caused_failures'])}")
    logger.info(f"  Modified selectors: {len(impact['modified_selectors'])}")

    for selector in impact["modified_selectors"]:
        logger.info(f"    - {selector.get('selector')}")

    logger.info("\n✓ Code change impact analysis completed\n")


async def demo_graph_stats(graph: GraphStore):
    """Demo graph statistics."""
    logger.info("\n=== DEMO 5: Graph Statistics ===\n")

    stats = await graph.get_graph_stats()

    logger.info("Knowledge Graph Statistics:")
    logger.info(f"  Graph Name: {stats['graph_name']}")
    logger.info(f"  Total Vertices: {stats['total_vertices']}")
    logger.info(f"  Total Edges: {stats['total_edges']}")
    logger.info(f"  Schema Version: {stats['schema_version']}")

    logger.info("\n✓ Statistics completed\n")


async def main():
    """Run all demos."""
    logger.info("==============================================")
    logger.info("  Knowledge Graph Demo")
    logger.info("==============================================")

    # Initialize graph store
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable required")
        sys.exit(1)

    logger.info(f"\nConnecting to database...\n")
    graph = await init_graph_store(database_url=database_url)

    try:
        # Run demos
        project_id, test_id = await demo_basic_operations(graph)
        failure_id, pattern_id = await demo_failure_healing_workflow(graph, test_id)
        await demo_multi_hop_queries(graph, test_id)
        await demo_code_change_impact(graph, project_id, test_id)
        await demo_graph_stats(graph)

        logger.info("==============================================")
        logger.info("  All Demos Completed Successfully!")
        logger.info("==============================================\n")

        logger.info("Demo Entities Created:")
        logger.info(f"  Project ID: {project_id}")
        logger.info(f"  Test ID: {test_id}")
        logger.info(f"  Failure ID: {failure_id}")
        logger.info(f"  Pattern ID: {pattern_id}")
        logger.info("")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1

    finally:
        await graph.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
