# Knowledge Graph Layer - Apache AGE Integration

## Overview

The knowledge graph layer adds multi-hop reasoning capabilities to Argus using Apache AGE (A Graph Extension for PostgreSQL). This enables answering complex questions about relationships between tests, failures, code changes, and healing patterns that would be difficult or impossible with flat relational queries or vector search alone.

## Why Knowledge Graphs?

### Current Limitation (Flat Vectors)
Our `test_failure_patterns` table stores flat vectors for semantic search. While powerful for finding similar failures, it cannot answer relationship questions like:

- **"What tests break when we change the payment module?"**
- **"What's the relationship between flaky tests and network timeouts?"**
- **"Which selectors are most fragile across all projects?"**
- **"Show me all failures caused by commit abc123"**

### Solution: Hybrid Approach
Combining vector search (pgvector) with graph traversal (Apache AGE) gives us the best of both worlds:

1. **Vector Search** - Semantic similarity for finding related failures
2. **Graph Traversal** - Structural relationships for impact analysis
3. **Combined** - Comprehensive understanding of test ecosystem

## Technology: Apache AGE

- **Same Database**: No new infrastructure - uses existing PostgreSQL
- **Cypher Query Language**: Industry-standard graph query language
- **Free & Open Source**: Apache 2.0 license
- **Production Ready**: Developed by Bitnine, based on PostgreSQL internals

## Graph Schema

### Entity Types (Vertices)

| Entity | Description | Key Properties |
|--------|-------------|----------------|
| `Test` | Test case | name, file_path, status |
| `Selector` | CSS/XPath selector | selector, selector_type, fragility_score |
| `Failure` | Test failure | error_message, error_type, occurred_at |
| `CodeChange` | Git commit | commit_sha, commit_message, author |
| `HealingPattern` | Healing solution | original_selector, healed_selector, confidence |
| `Page` | Web page | url, page_title, complexity |
| `Project` | Project/repository | name, repository_url |

### Relationship Types (Edges)

| Relationship | From → To | Description |
|--------------|-----------|-------------|
| `USES` | Test → Selector | Test uses selector in a step |
| `TARGETS` | Test → Page | Test targets a page |
| `BELONGS_TO` | Test → Project | Test belongs to project |
| `BROKE` | Failure → Test | Failure broke a test |
| `AFFECTED` | CodeChange → Test | Code change affected test |
| `CAUSED` | CodeChange → Failure | Code change caused failure |
| `MODIFIED` | CodeChange → Selector | Code change modified selector |
| `FIXES` | HealingPattern → Failure | Healing pattern fixes failure |
| `REPLACES` | HealingPattern → Selector | Healing pattern replaces selector |
| `ON` | Selector → Page | Selector is on page |
| `DEPENDS_ON` | Test → Test | Test depends on another test |
| `SIMILAR_TO` | Entity → Entity | Entities are similar |

## Installation & Setup

### 1. Database Migration

Run the Apache AGE migration to create the graph schema:

```bash
# Apply migration via Supabase CLI
supabase db push

# Or apply manually
psql $DATABASE_URL -f supabase/migrations/20260126000000_apache_age_graph.sql
```

### 2. Migrate Existing Data

Use the migration script to populate the graph with existing data:

```bash
# Migrate all projects
python scripts/migrate_knowledge_graph.py

# Migrate specific project
python scripts/migrate_knowledge_graph.py --project-id abc123-def456-...

# Custom batch size
python scripts/migrate_knowledge_graph.py --batch-size 500
```

## Usage Examples

### Basic Operations

```python
from knowledge_graph import GraphStore

# Initialize
graph = GraphStore()

# Add a test
test_id = await graph.add_test(
    test_id="uuid-123",
    name="login_test",
    file_path="tests/auth/test_login.py",
    status="passed"
)

# Add a selector
selector_id = await graph.add_selector(
    selector="#login-button",
    selector_type="css"
)

# Create relationship
await graph.add_edge(
    from_id=test_id,
    to_id=selector_id,
    relationship=EdgeType.USES,
    properties={"step": 5, "action": "click"}
)
```

### Multi-Hop Queries

#### Find All Tests Using a Selector

```python
tests = await graph.find_tests_using_selector("#login-button")

for test in tests:
    print(f"Test: {test['name']}, Step: {test['step']}")
```

#### Analyze Code Change Impact

```python
impact = await graph.find_code_change_impact(
    commit_sha="abc123",
    project_id="project-uuid"
)

print(f"Affected tests: {len(impact['affected_tests'])}")
print(f"Caused failures: {len(impact['caused_failures'])}")
print(f"Modified selectors: {len(impact['modified_selectors'])}")
```

#### Find Fragile Selectors

```python
fragile = await graph.find_fragile_selectors(
    project_id="project-uuid",
    min_failures=3,
    limit=20
)

for selector_info in fragile:
    print(f"{selector_info['selector']}: "
          f"Fragility={selector_info['fragility_score']:.2f}, "
          f"Failures={selector_info['failure_count']}")
```

### Self-Healer Integration

```python
from agents.self_healer_graph_integration import GraphEnhancedHealer

healer = GraphEnhancedHealer()

# Find related failures via graph
related = await healer.find_related_failures_graph(
    test_id="test-uuid",
    selector="#submit-btn",
    max_hops=2
)

# Analyze selector fragility
fragility = await healer.analyze_selector_fragility(
    selector="#submit-btn",
    project_id="project-uuid"
)

if fragility['fragility_score'] > 0.7:
    print(f"WARNING: {fragility['recommendation']}")

# Get healing suggestions via graph
suggestions = await healer.suggest_healing_via_graph(
    selector="#submit-btn",
    test_id="test-uuid",
    project_id="project-uuid"
)

for suggestion in suggestions:
    print(f"Source: {suggestion['source']}, "
          f"Confidence: {suggestion['confidence']:.2%}")
```

### Raw Cypher Queries

For advanced use cases, execute raw Cypher queries:

```python
# Find all failures in the last week
cypher = """
    MATCH (f:Failure)
    WHERE f.occurred_at > '2024-01-19'
    OPTIONAL MATCH (f)-[:BROKE]->(t:Test)
    OPTIONAL MATCH (f)-[:FIXES]-(hp:HealingPattern)
    RETURN f, t, hp
    ORDER BY f.occurred_at DESC
    LIMIT 50
"""

results = await graph.query(cypher)
```

## Advanced Use Cases

### 1. Test Dependency Analysis

Find which tests depend on each other (useful for parallelization):

```python
cypher = """
    MATCH (t1:Test)-[:DEPENDS_ON*1..3]->(t2:Test)
    WHERE t1.test_id = $test_id
    RETURN t2.name AS dependent_test,
           length(path) AS depth
    ORDER BY depth
"""

deps = await graph.query(cypher, params={"test_id": "test-uuid"})
```

### 2. Blast Radius Analysis

Find all tests that could be affected by changing a selector:

```python
cypher = """
    MATCH (s:Selector {selector: $selector})
    MATCH (s)<-[:USES]-(t:Test)
    OPTIONAL MATCH (t)-[:BELONGS_TO]->(p:Project)
    OPTIONAL MATCH (s)-[:ON]->(page:Page)
    RETURN t, p, page
"""

blast_radius = await graph.query(
    cypher,
    params={"selector": "#login-button"}
)
```

### 3. Healing Success Rate by Method

Analyze which healing methods work best:

```python
cypher = """
    MATCH (hp:HealingPattern)-[:FIXES]->(f:Failure)
    WITH hp.healing_method AS method,
         count(f) AS total_healings,
         avg(hp.confidence) AS avg_confidence,
         sum(hp.success_count) AS successes,
         sum(hp.failure_count) AS failures
    RETURN method,
           total_healings,
           avg_confidence,
           successes::float / NULLIF(successes + failures, 0) AS success_rate
    ORDER BY success_rate DESC
"""

methods = await graph.query(cypher)
```

### 4. Find Tests Similar to a Failed Test

```python
cypher = """
    MATCH (t1:Test {test_id: $test_id})-[:USES]->(s:Selector)
    MATCH (s)<-[:USES]-(t2:Test)
    WHERE t1 <> t2
    WITH t2, count(s) AS shared_selectors
    MATCH (t2)-[:BROKE]-(f:Failure)
    RETURN t2.name AS similar_test,
           shared_selectors,
           count(f) AS failure_count
    ORDER BY shared_selectors DESC, failure_count DESC
    LIMIT 10
"""

similar = await graph.query(cypher, params={"test_id": "test-uuid"})
```

## Performance Considerations

### Indexes

All mapping tables have indexes on:
- `vertex_id` - Fast lookups by graph ID
- Entity IDs - Fast lookups by entity UUID
- Text fields - Fast selector/URL lookups

### Query Optimization

1. **Use `LIMIT`** - Always limit results for exploratory queries
2. **Filter Early** - Add `WHERE` clauses before expensive operations
3. **Index Lookups** - Use indexed properties in `WHERE` clauses
4. **Avoid Deep Traversals** - Limit path length with `*1..3` syntax

### Statistics

Update graph statistics periodically:

```python
pool = await graph._get_pool()
async with pool.acquire() as conn:
    await conn.execute("SELECT update_graph_stats()")
```

## Testing

Run the test suite:

```bash
# All graph tests
pytest tests/knowledge_graph/

# Specific test file
pytest tests/knowledge_graph/test_graph_store.py

# With verbose output
pytest tests/knowledge_graph/ -v

# Integration tests only
pytest tests/knowledge_graph/ -m integration
```

## Architecture Notes

### Hybrid Storage Model

The knowledge graph uses a **hybrid storage model**:

1. **AGE Graph** - Stores vertices and edges with Cypher queries
2. **Relational Mapping Tables** - Maps entity UUIDs to graph vertex IDs
3. **Existing Tables** - Original tables (tests, healing_patterns) remain unchanged

This approach provides:
- Fast graph traversal (Cypher queries)
- Fast entity lookups (UUID indexes)
- No data duplication
- Backward compatibility

### Why Not Neo4j?

We chose Apache AGE over Neo4j because:

1. **Same Database** - No additional infrastructure
2. **Cost** - Free and open source (Neo4j Enterprise is paid)
3. **Integration** - Uses PostgreSQL connection pooling, RLS, and backups
4. **Simplicity** - One database for vectors, relations, and graphs

## Troubleshooting

### AGE Extension Not Found

If you get "extension age does not exist":

```sql
-- Check if AGE is installed
SELECT * FROM pg_available_extensions WHERE name = 'age';

-- Install AGE (requires superuser)
CREATE EXTENSION age;
```

### Search Path Issues

If Cypher queries fail with "function cypher() does not exist":

```python
# Ensure search path includes ag_catalog
async with pool.acquire() as conn:
    await conn.execute("SET search_path = ag_catalog, \"$user\", public;")
```

### Migration Failures

Check logs for specific errors:

```bash
# Run migration with debug logging
python scripts/migrate_knowledge_graph.py --log-level DEBUG
```

## Roadmap

Future enhancements:

- [ ] Auto-sync: Automatically update graph when relational data changes
- [ ] Graph ML: Train GNN models on the graph for failure prediction
- [ ] Visual Explorer: Web UI for exploring the graph
- [ ] Graph Snapshots: Version the graph for time-travel queries
- [ ] Cross-Project Analysis: Find patterns across multiple projects

## References

- [Apache AGE Documentation](https://age.apache.org/)
- [Cypher Query Language](https://neo4j.com/developer/cypher/)
- [Graph Databases Overview](https://en.wikipedia.org/wiki/Graph_database)
