"""
Argus Knowledge Layer - Unified knowledge management powered by Cognee.

This module provides a single, unified interface for all knowledge operations:
- Key-value storage with semantic search
- Failure pattern learning
- Knowledge graph reasoning
- Multi-tenant isolation

This replaces the deprecated:
- src/orchestrator/memory_store.py (MemoryStore)
- src/knowledge_graph/graph_store.py (GraphStore)

Usage:
    ```python
    from src.knowledge import get_cognee_client, CogneeKnowledgeClient

    # Get client for current tenant
    client = get_cognee_client(org_id="org123", project_id="proj456")

    # Store a pattern
    await client.put(
        namespace=["patterns"],
        key="login-button",
        value={"selector": "#login-btn"},
        embed_text="login button selector"
    )

    # Find similar failures
    similar = await client.find_similar_failures(
        error_message="Element not found: #submit-btn"
    )

    # Query knowledge graph
    results = await client.query_knowledge_graph(
        query="What tests use the login button?"
    )
    ```
"""

from .cognee_client import (
    CogneeError,
    CogneeGraphError,
    CogneeKnowledgeClient,
    CogneeRetrievalError,
    CogneeSearchError,
    CogneeStorageError,
    SimilarFailure,
    get_cognee_client,
    init_cognee_client,
    reset_cognee_client,
)

__all__ = [
    # Client
    "CogneeKnowledgeClient",
    "SimilarFailure",
    "get_cognee_client",
    "reset_cognee_client",
    "init_cognee_client",
    # Exceptions
    "CogneeError",
    "CogneeStorageError",
    "CogneeRetrievalError",
    "CogneeSearchError",
    "CogneeGraphError",
]
