#!/usr/bin/env python3
"""
Neo4j Aura Schema Initialization Script

Initializes the multi-tenant schema with constraints, indexes, and full-text indexes
for optimal query performance. This script is idempotent - safe to run multiple times.

Usage:
    export NEO4J_URI="neo4j+s://xxx.databases.neo4j.io"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="your-password"
    python init_neo4j_schema.py

Or via Kubernetes Job:
    kubectl apply -f neo4j-schema-init.yaml
"""

import os
import sys
import time
from typing import List, Tuple

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError:
    print("ERROR: neo4j driver not installed. Run: pip install neo4j")
    sys.exit(1)


# =============================================================================
# Schema Definitions
# =============================================================================

# Uniqueness constraints - (node_label, property)
UNIQUENESS_CONSTRAINTS = [
    ("Organization", "id"),
    ("Project", "id"),
    ("Repository", "id"),
    ("CodeFile", "id"),
    ("Function", "id"),
    ("Class", "id"),
    ("TestSuite", "id"),
    ("TestCase", "id"),
    ("TestRun", "id"),
    ("TestFailure", "id"),
    ("HealingPattern", "id"),
    ("CogneeDocument", "id"),
]

# Existence constraints - (node_label, property)
# These ensure org_id is always set for tenant isolation
EXISTENCE_CONSTRAINTS = [
    ("Project", "org_id"),
    ("Repository", "org_id"),
    ("CodeFile", "org_id"),
    ("Function", "org_id"),
    ("Class", "org_id"),
    ("TestSuite", "org_id"),
    ("TestCase", "org_id"),
    ("TestRun", "org_id"),
    ("TestFailure", "org_id"),
    ("HealingPattern", "org_id"),
    ("CogneeDocument", "org_id"),
]

# Composite indexes - (node_label, properties)
# org_id always first for tenant isolation
COMPOSITE_INDEXES = [
    ("Organization", ["name"]),
    ("Project", ["org_id"]),
    ("Project", ["org_id", "name"]),
    ("Repository", ["org_id"]),
    ("Repository", ["org_id", "project_id"]),
    ("Repository", ["org_id", "url"]),
    ("CodeFile", ["org_id"]),
    ("CodeFile", ["org_id", "project_id"]),
    ("CodeFile", ["org_id", "repo_id"]),
    ("CodeFile", ["org_id", "path"]),
    ("CodeFile", ["org_id", "language"]),
    ("Function", ["org_id"]),
    ("Function", ["org_id", "project_id"]),
    ("Function", ["org_id", "name"]),
    ("Class", ["org_id"]),
    ("Class", ["org_id", "project_id"]),
    ("Class", ["org_id", "name"]),
    ("TestSuite", ["org_id"]),
    ("TestSuite", ["org_id", "project_id"]),
    ("TestCase", ["org_id"]),
    ("TestCase", ["org_id", "project_id"]),
    ("TestCase", ["org_id", "status"]),
    ("TestRun", ["org_id"]),
    ("TestRun", ["org_id", "project_id"]),
    ("TestRun", ["org_id", "started_at"]),
    ("TestFailure", ["org_id"]),
    ("TestFailure", ["org_id", "healed"]),
    ("TestFailure", ["org_id", "error_type"]),
    ("HealingPattern", ["org_id"]),
    ("HealingPattern", ["org_id", "success_rate"]),
]

# Full-text indexes - (index_name, labels, properties)
FULLTEXT_INDEXES = [
    ("code_search", ["Function", "Class"], ["name", "docstring", "signature"]),
    ("error_search", ["TestFailure"], ["error_message", "stack_trace"]),
    ("file_search", ["CodeFile"], ["path", "name"]),
]


def generate_constraint_name(label: str, prop: str, type: str) -> str:
    """Generate a valid Neo4j constraint name."""
    return f"{label.lower()}_{prop.lower()}_{type}"


def generate_index_name(label: str, props: List[str]) -> str:
    """Generate a valid Neo4j index name."""
    props_str = "_".join(p.lower() for p in props)
    return f"{label.lower()}_{props_str}_idx"


class SchemaInitializer:
    """Initializes Neo4j schema with constraints and indexes."""

    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.stats = {
            "constraints_created": 0,
            "constraints_existed": 0,
            "indexes_created": 0,
            "indexes_existed": 0,
            "errors": 0,
        }

    def connect(self, max_retries: int = 5, retry_delay: int = 15) -> bool:
        """Connect to Neo4j with retry logic for Aura cold starts."""
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

        for attempt in range(1, max_retries + 1):
            try:
                print(f"Connecting to Neo4j (attempt {attempt}/{max_retries})...")
                with self.driver.session() as session:
                    result = session.run("RETURN 1 AS test")
                    result.single()
                print("Connected to Neo4j Aura")
                return True
            except ServiceUnavailable as e:
                if attempt < max_retries:
                    print(f"  Aura waking up, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"ERROR: Failed to connect after {max_retries} attempts: {e}")
                    return False
        return False

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def _run_cypher(self, cypher: str, description: str) -> Tuple[bool, str]:
        """Run a Cypher statement and return (success, message)."""
        try:
            with self.driver.session() as session:
                session.run(cypher)
            return True, "created"
        except ClientError as e:
            error_code = getattr(e, 'code', '')
            # Handle "already exists" errors gracefully
            if "AlreadyExists" in str(e) or "already exists" in str(e).lower():
                return True, "already exists"
            elif "EquivalentSchemaRuleAlreadyExists" in error_code:
                return True, "already exists"
            else:
                print(f"  ERROR: {description}: {e}")
                self.stats["errors"] += 1
                return False, str(e)

    def create_uniqueness_constraints(self):
        """Create uniqueness constraints for all node types."""
        print("\n=== Creating Uniqueness Constraints ===")
        for label, prop in UNIQUENESS_CONSTRAINTS:
            name = generate_constraint_name(label, prop, "unique")
            cypher = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            success, status = self._run_cypher(cypher, f"{label}.{prop} unique")
            if success:
                if status == "created":
                    self.stats["constraints_created"] += 1
                else:
                    self.stats["constraints_existed"] += 1
                print(f"  {label}.{prop} IS UNIQUE: {status}")

    def create_existence_constraints(self):
        """Create existence constraints for org_id (tenant isolation)."""
        print("\n=== Creating Existence Constraints (org_id) ===")
        for label, prop in EXISTENCE_CONSTRAINTS:
            name = generate_constraint_name(label, prop, "exists")
            cypher = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS NOT NULL"
            success, status = self._run_cypher(cypher, f"{label}.{prop} exists")
            if success:
                if status == "created":
                    self.stats["constraints_created"] += 1
                else:
                    self.stats["constraints_existed"] += 1
                print(f"  {label}.{prop} IS NOT NULL: {status}")

    def create_composite_indexes(self):
        """Create composite indexes for multi-tenant queries."""
        print("\n=== Creating Composite Indexes ===")
        for label, props in COMPOSITE_INDEXES:
            name = generate_index_name(label, props)
            props_str = ", ".join(f"n.{p}" for p in props)
            cypher = f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON ({props_str})"
            success, status = self._run_cypher(cypher, f"{label}({', '.join(props)})")
            if success:
                if status == "created":
                    self.stats["indexes_created"] += 1
                else:
                    self.stats["indexes_existed"] += 1
                print(f"  {label}({', '.join(props)}): {status}")

    def create_fulltext_indexes(self):
        """Create full-text indexes for semantic search."""
        print("\n=== Creating Full-Text Indexes ===")
        for name, labels, props in FULLTEXT_INDEXES:
            labels_str = "|".join(labels)
            props_str = ", ".join(f"n.{p}" for p in props)
            cypher = f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS FOR (n:{labels_str}) ON EACH [{props_str}]"
            success, status = self._run_cypher(cypher, name)
            if success:
                if status == "created":
                    self.stats["indexes_created"] += 1
                else:
                    self.stats["indexes_existed"] += 1
                print(f"  {name}: {status}")

    def verify_schema(self):
        """Verify the schema was created correctly."""
        print("\n=== Verifying Schema ===")
        with self.driver.session() as session:
            # Count constraints
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            print(f"  Total constraints: {len(constraints)}")

            # Count indexes
            result = session.run("SHOW INDEXES")
            indexes = list(result)
            print(f"  Total indexes: {len(indexes)}")

    def run(self) -> bool:
        """Run the full schema initialization."""
        print("=" * 60)
        print("Neo4j Aura Multi-Tenant Schema Initialization")
        print("=" * 60)

        if not self.connect():
            return False

        try:
            self.create_uniqueness_constraints()
            self.create_existence_constraints()
            self.create_composite_indexes()
            self.create_fulltext_indexes()
            self.verify_schema()

            print("\n" + "=" * 60)
            print("Schema Initialization Complete")
            print("=" * 60)
            print(f"  Constraints created: {self.stats['constraints_created']}")
            print(f"  Constraints existed: {self.stats['constraints_existed']}")
            print(f"  Indexes created: {self.stats['indexes_created']}")
            print(f"  Indexes existed: {self.stats['indexes_existed']}")
            print(f"  Errors: {self.stats['errors']}")

            return self.stats["errors"] == 0
        finally:
            self.close()


def main():
    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not uri or not password:
        print("ERROR: NEO4J_URI and NEO4J_PASSWORD environment variables required")
        print("\nUsage:")
        print("  export NEO4J_URI='neo4j+s://xxx.databases.neo4j.io'")
        print("  export NEO4J_USERNAME='neo4j'")
        print("  export NEO4J_PASSWORD='your-password'")
        print("  python init_neo4j_schema.py")
        sys.exit(1)

    initializer = SchemaInitializer(uri, username, password)
    success = initializer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
