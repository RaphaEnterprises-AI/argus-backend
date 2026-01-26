"""Knowledge Graph Layer for Multi-Hop Reasoning.

This module provides a knowledge graph layer using Apache AGE (PostgreSQL extension)
to enable multi-hop reasoning about relationships between tests, failures, code changes,
and healing patterns.

The knowledge graph enables answering questions like:
- What tests break when we change the payment module?
- What's the relationship between flaky tests and network timeouts?
- Which selectors are most fragile across all projects?
- Show me all failures caused by commit abc123

Key Features:
- Apache AGE graph database (same PostgreSQL instance)
- Cypher query language for graph traversals
- Multi-hop reasoning for complex relationships
- Integration with existing vector search for hybrid queries

Example:
    ```python
    from knowledge_graph import GraphStore

    graph = GraphStore()

    # Add a test node
    test_id = await graph.add_test(
        test_id="uuid-123",
        name="login_test",
        file_path="tests/auth/test_login.py"
    )

    # Add a selector node
    selector_id = await graph.add_selector(
        selector="#login-button",
        selector_type="css"
    )

    # Create relationship
    await graph.add_edge(
        from_id=test_id,
        to_id=selector_id,
        relationship="USES",
        properties={"step": 5}
    )

    # Query: Find all tests that use this selector
    tests = await graph.find_tests_using_selector("#login-button")

    # Multi-hop query: Find all tests affected by a code change
    affected = await graph.find_code_change_impact("abc123")
    ```
"""

from .graph_store import GraphStore, get_graph_store, init_graph_store, reset_graph_store
from .schema import EdgeType, EntityType, GraphEntity, GraphRelationship

__all__ = [
    "GraphStore",
    "get_graph_store",
    "init_graph_store",
    "reset_graph_store",
    "EntityType",
    "EdgeType",
    "GraphEntity",
    "GraphRelationship",
]
