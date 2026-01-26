"""Graph schema definitions for the Apache AGE knowledge graph.

Defines entity types (vertices) and relationship types (edges) used in the graph.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityType(Enum):
    """Entity types (vertex labels) in the knowledge graph."""

    TEST = "Test"
    SELECTOR = "Selector"
    FAILURE = "Failure"
    CODE_CHANGE = "CodeChange"
    HEALING_PATTERN = "HealingPattern"
    PAGE = "Page"
    PROJECT = "Project"


class EdgeType(Enum):
    """Relationship types (edge labels) in the knowledge graph."""

    # Test relationships
    USES = "USES"  # Test USES Selector
    TARGETS = "TARGETS"  # Test TARGETS Page
    BELONGS_TO = "BELONGS_TO"  # Test BELONGS_TO Project

    # Failure relationships
    BROKE = "BROKE"  # Failure BROKE Test
    AFFECTED = "AFFECTED"  # CodeChange AFFECTED Test
    CAUSED = "CAUSED"  # CodeChange CAUSED Failure

    # Code change relationships
    MODIFIED = "MODIFIED"  # CodeChange MODIFIED Selector

    # Healing relationships
    FIXES = "FIXES"  # HealingPattern FIXES Failure
    REPLACES = "REPLACES"  # HealingPattern REPLACES Selector (old -> new)

    # Page relationships
    ON = "ON"  # Selector ON Page

    # Dependency relationships
    DEPENDS_ON = "DEPENDS_ON"  # Test DEPENDS_ON Test

    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"  # Selector SIMILAR_TO Selector, Failure SIMILAR_TO Failure


@dataclass
class GraphEntity:
    """Represents an entity (vertex) in the knowledge graph."""

    entity_type: EntityType
    entity_id: str  # UUID or unique identifier
    properties: dict[str, Any] = field(default_factory=dict)
    vertex_id: int | None = None  # AGE vertex ID (set after insertion)

    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher property string.

        Returns:
            Cypher property string like {name: 'test1', status: 'passed'}
        """
        if not self.properties:
            return "{}"

        props = []
        for key, value in self.properties.items():
            # Escape strings and handle different types
            if isinstance(value, str):
                # Escape single quotes
                escaped_value = value.replace("'", "\\'")
                props.append(f"{key}: '{escaped_value}'")
            elif isinstance(value, bool):
                props.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, (int, float)):
                props.append(f"{key}: {value}")
            elif value is None:
                props.append(f"{key}: null")
            else:
                # For complex types, convert to JSON string
                import json

                escaped_value = json.dumps(value).replace("'", "\\'")
                props.append(f"{key}: '{escaped_value}'")

        return "{" + ", ".join(props) + "}"


@dataclass
class GraphRelationship:
    """Represents a relationship (edge) in the knowledge graph."""

    edge_type: EdgeType
    from_vertex_id: int
    to_vertex_id: int
    properties: dict[str, Any] = field(default_factory=dict)
    edge_id: int | None = None  # AGE edge ID (set after insertion)

    def to_cypher_properties(self) -> str:
        """Convert properties to Cypher property string.

        Returns:
            Cypher property string like {weight: 0.95, created_at: '2024-01-26'}
        """
        if not self.properties:
            return "{}"

        props = []
        for key, value in self.properties.items():
            if isinstance(value, str):
                escaped_value = value.replace("'", "\\'")
                props.append(f"{key}: '{escaped_value}'")
            elif isinstance(value, bool):
                props.append(f"{key}: {str(value).lower()}")
            elif isinstance(value, (int, float)):
                props.append(f"{key}: {value}")
            elif value is None:
                props.append(f"{key}: null")
            else:
                import json

                escaped_value = json.dumps(value).replace("'", "\\'")
                props.append(f"{key}: '{escaped_value}'")

        return "{" + ", ".join(props) + "}"


# ============================================================================
# Predefined Entity Schemas
# ============================================================================

TEST_PROPERTIES = {
    "name": str,  # Test name
    "file_path": str,  # Path to test file
    "status": str,  # passed, failed, skipped
    "duration_ms": int,  # Execution duration
    "created_at": str,  # ISO timestamp
}

SELECTOR_PROPERTIES = {
    "selector": str,  # CSS/XPath selector
    "selector_type": str,  # css, xpath
    "fragility_score": float,  # 0-1, how often it breaks
    "usage_count": int,  # How many tests use it
}

FAILURE_PROPERTIES = {
    "error_message": str,  # Error message
    "error_type": str,  # selector_changed, timing_issue, etc.
    "screenshot_url": str,  # URL to screenshot
    "occurred_at": str,  # ISO timestamp
    "auto_healed": bool,  # Whether it was auto-healed
}

CODE_CHANGE_PROPERTIES = {
    "commit_sha": str,  # Git commit SHA
    "commit_message": str,  # Commit message
    "author": str,  # Commit author
    "committed_at": str,  # ISO timestamp
    "files_changed": list,  # List of changed files
    "lines_added": int,  # Lines added
    "lines_deleted": int,  # Lines deleted
}

HEALING_PATTERN_PROPERTIES = {
    "original_selector": str,  # Original broken selector
    "healed_selector": str,  # Fixed selector
    "confidence": float,  # 0-1 confidence score
    "success_count": int,  # Times this pattern worked
    "failure_count": int,  # Times it failed
    "healing_method": str,  # code_aware, semantic_match, etc.
}

PAGE_PROPERTIES = {
    "url": str,  # Page URL
    "page_title": str,  # Page title
    "route": str,  # Route pattern
    "complexity": str,  # simple, medium, complex
}

PROJECT_PROPERTIES = {
    "name": str,  # Project name
    "repository_url": str,  # Git repo URL
    "total_tests": int,  # Total test count
    "pass_rate": float,  # Overall pass rate
}

# ============================================================================
# Relationship Property Schemas
# ============================================================================

USES_PROPERTIES = {
    "step_number": int,  # Which step in the test
    "action": str,  # click, type, etc.
    "usage_count": int,  # How many times used
}

TARGETS_PROPERTIES = {
    "test_count": int,  # How many tests target this page
}

BROKE_PROPERTIES = {
    "occurred_at": str,  # When failure occurred
    "test_run_id": str,  # Test run UUID
}

AFFECTED_PROPERTIES = {
    "impact_type": str,  # direct, indirect
    "confidence": float,  # 0-1 confidence
}

CAUSED_PROPERTIES = {
    "likelihood": float,  # 0-1 probability this change caused failure
    "evidence": list,  # List of evidence
}

MODIFIED_PROPERTIES = {
    "change_type": str,  # renamed, removed, updated
    "old_value": str,  # Old selector
    "new_value": str,  # New selector
}

FIXES_PROPERTIES = {
    "applied_at": str,  # When fix was applied
    "success": bool,  # Whether fix worked
}

REPLACES_PROPERTIES = {
    "old_selector": str,  # Original selector
    "new_selector": str,  # Replacement selector
    "confidence": float,  # 0-1 confidence
}

SIMILAR_TO_PROPERTIES = {
    "similarity_score": float,  # 0-1 similarity score
    "similarity_type": str,  # semantic, structural, behavioral
}
