# Knowledge Graph Quick Start Guide

Get started with the Apache AGE knowledge graph layer in 5 minutes.

## Prerequisites

- PostgreSQL 12+ with Apache AGE extension
- Python 3.12+
- Existing Argus installation

## Step 1: Install Apache AGE Extension

```sql
-- Connect to your database as superuser
psql $DATABASE_URL

-- Install AGE extension
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
```

**Note**: If you don't have superuser access, ask your database administrator to install the AGE extension.

## Step 2: Run Database Migration

```bash
# Apply the migration
supabase db push

# Or manually via psql
psql $DATABASE_URL -f supabase/migrations/20260126000000_apache_age_graph.sql
```

This creates:
- The `argus_knowledge` graph
- Vertex mapping tables
- Edge tracking tables
- Helper functions

## Step 3: Migrate Existing Data

```bash
# Migrate all existing data to the graph
python scripts/migrate_knowledge_graph.py

# Or migrate a specific project
python scripts/migrate_knowledge_graph.py --project-id YOUR-PROJECT-UUID
```

Expected output:
```
INFO - Starting graph migration
INFO - Migrated projects: 5
INFO - Migrated tests: 234
INFO - Migrated healing patterns: 89
INFO - Migrated test failures: 156
INFO - Migrated selectors: 312
INFO - Created test relationships: 458
INFO - Created failure relationships: 156
INFO - Created healing relationships: 89
INFO - Graph migration completed
```

## Step 4: Verify Installation

```bash
# Run the demo script
python scripts/demo_knowledge_graph.py
```

Expected output:
```
=== DEMO 1: Basic Graph Operations ===
Created project vertex: 12345
Created test vertex: 12346
...
✓ All Demos Completed Successfully!
```

## Step 5: Use in Your Code

### Basic Example

```python
from knowledge_graph import GraphStore

# Initialize
graph = GraphStore()

# Find tests using a selector
tests = await graph.find_tests_using_selector("#login-button")
print(f"Found {len(tests)} tests using this selector")

# Find fragile selectors
fragile = await graph.find_fragile_selectors(
    project_id="your-project-id",
    min_failures=3,
    limit=10
)

for selector in fragile:
    print(f"{selector['selector']}: "
          f"fragility={selector['fragility_score']:.2f}")
```

### Self-Healer Integration

```python
from agents.self_healer_graph_integration import GraphEnhancedHealer

healer = GraphEnhancedHealer()

# Analyze selector fragility
fragility = await healer.analyze_selector_fragility(
    selector="#submit-btn",
    project_id="your-project-id"
)

if fragility['fragility_score'] > 0.7:
    print(f"⚠️  {fragility['recommendation']}")

# Get healing suggestions
suggestions = await healer.suggest_healing_via_graph(
    selector="#submit-btn",
    test_id="test-uuid",
    project_id="project-uuid"
)

for suggestion in suggestions:
    print(f"Suggestion: {suggestion['healed_selector']} "
          f"(confidence: {suggestion['confidence']:.0%})")
```

### Raw Cypher Queries

```python
# Find all failures in the last 24 hours
cypher = """
    MATCH (f:Failure)
    WHERE f.occurred_at > datetime() - duration('P1D')
    OPTIONAL MATCH (f)-[:BROKE]->(t:Test)
    RETURN f, t
    ORDER BY f.occurred_at DESC
    LIMIT 50
"""

results = await graph.query(cypher)
```

## Common Use Cases

### 1. Find What Breaks When You Change Code

```python
impact = await graph.find_code_change_impact(
    commit_sha="abc123",
    project_id="project-uuid"
)

print(f"This commit affected:")
print(f"  {len(impact['affected_tests'])} tests")
print(f"  {len(impact['modified_selectors'])} selectors")
print(f"  {len(impact['caused_failures'])} failures")
```

### 2. Identify Technical Debt

```python
healer = GraphEnhancedHealer()
fragile = await healer.get_top_fragile_selectors(
    project_id="project-uuid",
    limit=20
)

print("Top fragile selectors:")
for selector in fragile:
    print(f"  {selector['selector']}")
    print(f"    Fragility: {selector['fragility_score']:.2f}")
    print(f"    Failures: {selector['failure_count']}")
    print(f"    Recommendation: {selector['recommendation']}")
```

### 3. Understand Test Dependencies

```python
cypher = """
    MATCH (t1:Test {test_id: $test_id})-[:DEPENDS_ON*1..3]->(t2:Test)
    RETURN t2.name AS dependent_test, length(path) AS depth
    ORDER BY depth
"""

deps = await graph.query(cypher, params={"test_id": "test-uuid"})
print(f"This test depends on {len(deps)} other tests")
```

## Troubleshooting

### "extension age does not exist"

**Solution**: Install the AGE extension:
```sql
CREATE EXTENSION age;
```

### "function cypher() does not exist"

**Solution**: Set search path:
```python
async with pool.acquire() as conn:
    await conn.execute("SET search_path = ag_catalog, \"$user\", public;")
```

### Migration fails

**Solution**: Check logs and run with debug:
```bash
python scripts/migrate_knowledge_graph.py --log-level DEBUG
```

## Next Steps

1. **Read the full documentation**: `src/knowledge_graph/README.md`
2. **Explore examples**: `scripts/demo_knowledge_graph.py`
3. **Run tests**: `pytest tests/knowledge_graph/`
4. **Integrate with self-healer**: See `src/agents/self_healer_graph_integration.py`

## Resources

- [Apache AGE Documentation](https://age.apache.org/)
- [Cypher Query Language Guide](https://neo4j.com/developer/cypher/)
- [Full Implementation Details](KNOWLEDGE_GRAPH_IMPLEMENTATION.md)

## Support

For issues or questions:
1. Check `src/knowledge_graph/README.md`
2. Review test examples in `tests/knowledge_graph/`
3. Run the demo script for working examples

---

**You're ready to use the knowledge graph!** 🚀

The graph enables multi-hop reasoning about test relationships, providing insights that were previously impossible with traditional queries.
