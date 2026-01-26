"""Knowledge Graph Integration for Self-Healer Agent.

This module extends the SelfHealerAgent with knowledge graph capabilities,
enabling multi-hop reasoning about test failures and their relationships.

The graph integration provides:
- Finding related failures across different tests
- Analyzing code change impact through the graph
- Discovering fragile selectors that frequently break
- Multi-hop healing pattern discovery
"""

from typing import Any

import structlog

from ..knowledge_graph import GraphStore, get_graph_store
from ..knowledge_graph.schema import EdgeType

logger = structlog.get_logger(__name__)


class GraphEnhancedHealer:
    """Knowledge graph enhancements for the self-healer.

    This class provides graph-based reasoning capabilities that complement
    the existing vector-based semantic search in the self-healer.

    Hybrid Approach:
    1. Vector search (pgvector) - finds semantically similar failures
    2. Graph traversal (AGE) - finds structurally related failures
    3. Combined results - provides comprehensive healing suggestions
    """

    def __init__(
        self,
        graph_store: GraphStore | None = None,
    ):
        """Initialize graph-enhanced healer.

        Args:
            graph_store: Optional GraphStore instance (uses global if not provided)
        """
        self.graph = graph_store or get_graph_store()
        self._log = logger.bind(component="graph_enhanced_healer")

    async def find_related_failures_graph(
        self,
        test_id: str,
        selector: str | None = None,
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Find related failures using graph traversal.

        Uses multi-hop graph queries to find failures that:
        - Affect the same test
        - Use the same selector
        - Were caused by the same code change
        - Have similar healing patterns

        Args:
            test_id: Test UUID
            selector: Optional selector to focus on
            max_hops: Maximum hops for graph traversal

        Returns:
            List of related failure records with graph context

        Example:
            Test -> BROKE <- Failure -> FIXES <- HealingPattern -> REPLACES -> Selector
        """
        try:
            # Build multi-hop Cypher query
            if selector:
                # Find failures related to this selector
                cypher = f"""
                    MATCH (s:Selector {{selector: '{selector}'}})<-[:USES]-(t:Test)
                    WHERE t.test_id = '{test_id}'

                    MATCH (t)-[:BROKE]-(f:Failure)

                    OPTIONAL MATCH (f)-[:FIXES]-(hp:HealingPattern)
                    OPTIONAL MATCH (hp)-[:REPLACES]->(s2:Selector)
                    OPTIONAL MATCH (cc:CodeChange)-[:CAUSED]->(f)

                    RETURN f, hp, s2, cc
                    LIMIT 20
                """
            else:
                # Find all failures for this test
                cypher = f"""
                    MATCH (t:Test {{test_id: '{test_id}'}})-[:BROKE]-(f:Failure)

                    OPTIONAL MATCH (f)-[:FIXES]-(hp:HealingPattern)
                    OPTIONAL MATCH (hp)-[:REPLACES]->(s:Selector)
                    OPTIONAL MATCH (cc:CodeChange)-[:CAUSED]->(f)

                    RETURN f, hp, s, cc
                    ORDER BY f.occurred_at DESC
                    LIMIT 20
                """

            results = await self.graph.query(cypher)

            # Enrich results with graph context
            enriched = []
            for result in results:
                failure = result.get("f", {})
                healing_pattern = result.get("hp", {})
                selector_node = result.get("s", {}) or result.get("s2", {})
                code_change = result.get("cc", {})

                enriched_failure = {
                    "failure": failure,
                    "healing_pattern": healing_pattern,
                    "related_selector": selector_node,
                    "code_change": code_change,
                    "source": "knowledge_graph",
                }
                enriched.append(enriched_failure)

            self._log.info(
                "Found related failures via graph",
                test_id=test_id,
                count=len(enriched),
            )

            return enriched

        except Exception as e:
            self._log.error(
                "Failed to find related failures via graph",
                test_id=test_id,
                error=str(e),
            )
            return []

    async def analyze_selector_fragility(
        self,
        selector: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Analyze how fragile a selector is using graph data.

        Fragility indicators:
        - Number of failures involving this selector
        - Number of code changes that modified it
        - Number of healing patterns that replaced it
        - Success rate of healings

        Args:
            selector: CSS/XPath selector
            project_id: Project UUID

        Returns:
            Fragility analysis with metrics

        Example:
            Returns fragility score, failure count, healing success rate
        """
        try:
            cypher = f"""
                MATCH (p:Project {{project_id: '{project_id}'}})
                MATCH (s:Selector {{selector: '{selector}'}})

                OPTIONAL MATCH (s)<-[:USES]-(t:Test)-[:BELONGS_TO]->(p)
                OPTIONAL MATCH (t)-[:BROKE]-(f:Failure)
                OPTIONAL MATCH (cc:CodeChange)-[:MODIFIED]->(s)
                OPTIONAL MATCH (hp:HealingPattern)-[:REPLACES]->(s)

                WITH s,
                     count(DISTINCT t) AS usage_count,
                     count(DISTINCT f) AS failure_count,
                     count(DISTINCT cc) AS modification_count,
                     count(DISTINCT hp) AS healing_count,
                     collect(DISTINCT hp.confidence) AS healing_confidences

                RETURN
                    s.selector AS selector,
                    usage_count,
                    failure_count,
                    modification_count,
                    healing_count,
                    CASE WHEN usage_count > 0
                         THEN failure_count::float / usage_count
                         ELSE 0
                    END AS failure_rate,
                    healing_confidences
            """

            results = await self.graph.query(cypher)

            if not results:
                return {
                    "selector": selector,
                    "fragility_score": 0.0,
                    "usage_count": 0,
                    "failure_count": 0,
                    "recommendation": "No data available",
                }

            result = results[0]

            # Calculate fragility score (0-1, higher = more fragile)
            failure_rate = result.get("failure_rate", 0.0)
            modification_count = result.get("modification_count", 0)
            healing_count = result.get("healing_count", 0)

            # Weighted fragility score
            fragility_score = min(
                1.0,
                (
                    failure_rate * 0.5
                    + (modification_count / 10.0) * 0.3
                    + (healing_count / 5.0) * 0.2
                ),
            )

            # Calculate avg healing confidence
            confidences = result.get("healing_confidences", [])
            avg_confidence = (
                sum(float(c) for c in confidences) / len(confidences)
                if confidences
                else 0.0
            )

            # Generate recommendation
            if fragility_score > 0.7:
                recommendation = "CRITICAL: Consider refactoring this selector (highly fragile)"
            elif fragility_score > 0.4:
                recommendation = "WARNING: This selector breaks frequently"
            else:
                recommendation = "OK: Selector appears stable"

            return {
                "selector": selector,
                "fragility_score": round(fragility_score, 3),
                "usage_count": result.get("usage_count", 0),
                "failure_count": result.get("failure_count", 0),
                "modification_count": modification_count,
                "healing_count": healing_count,
                "failure_rate": round(failure_rate, 3),
                "avg_healing_confidence": round(avg_confidence, 3),
                "recommendation": recommendation,
            }

        except Exception as e:
            self._log.error(
                "Failed to analyze selector fragility",
                selector=selector,
                error=str(e),
            )
            return {
                "selector": selector,
                "fragility_score": 0.0,
                "error": str(e),
            }

    async def find_code_change_impact(
        self,
        commit_sha: str,
        project_id: str,
    ) -> dict[str, Any]:
        """Analyze the impact of a code change using graph traversal.

        Multi-hop query:
        CodeChange -> MODIFIED -> Selector <- USES <- Test
        CodeChange -> CAUSED -> Failure -> BROKE -> Test

        Args:
            commit_sha: Git commit SHA
            project_id: Project UUID

        Returns:
            Impact analysis with affected tests and failures
        """
        try:
            impact = await self.graph.find_code_change_impact(commit_sha, project_id)

            # Enrich with fragility analysis for modified selectors
            for selector in impact.get("modified_selectors", []):
                selector_text = selector.get("selector")
                if selector_text:
                    fragility = await self.analyze_selector_fragility(
                        selector_text, project_id
                    )
                    selector["fragility_analysis"] = fragility

            self._log.info(
                "Analyzed code change impact",
                commit_sha=commit_sha[:7],
                affected_tests=len(impact.get("affected_tests", [])),
                failures=len(impact.get("caused_failures", [])),
            )

            return impact

        except Exception as e:
            self._log.error(
                "Failed to analyze code change impact",
                commit_sha=commit_sha,
                error=str(e),
            )
            return {
                "affected_tests": [],
                "caused_failures": [],
                "modified_selectors": [],
                "error": str(e),
            }

    async def suggest_healing_via_graph(
        self,
        selector: str,
        test_id: str,
        project_id: str,
    ) -> list[dict[str, Any]]:
        """Find healing suggestions using graph traversal.

        Traverses the graph to find:
        1. Similar selectors that were successfully healed
        2. Healing patterns for this exact selector
        3. Code changes that modified this selector

        Args:
            selector: Broken selector
            test_id: Test UUID
            project_id: Project UUID

        Returns:
            List of healing suggestions from graph
        """
        try:
            # Multi-hop query to find healing patterns
            cypher = f"""
                MATCH (s:Selector {{selector: '{selector}'}})

                // Find healing patterns that replaced this selector
                OPTIONAL MATCH (hp:HealingPattern)-[:REPLACES]->(s)

                // Find similar selectors that were healed
                OPTIONAL MATCH (s2:Selector)-[:SIMILAR_TO]-(s)
                OPTIONAL MATCH (hp2:HealingPattern)-[:REPLACES]->(s2)

                // Find code changes that modified this selector
                OPTIONAL MATCH (cc:CodeChange)-[:MODIFIED]->(s)

                RETURN
                    collect(DISTINCT {{
                        pattern_id: hp.pattern_id,
                        healed_selector: hp.healed_selector,
                        confidence: hp.confidence,
                        success_count: hp.success_count,
                        source: 'direct_match'
                    }}) AS direct_healings,
                    collect(DISTINCT {{
                        pattern_id: hp2.pattern_id,
                        healed_selector: hp2.healed_selector,
                        confidence: hp2.confidence,
                        similar_selector: s2.selector,
                        source: 'similar_selector'
                    }}) AS similar_healings,
                    collect(DISTINCT {{
                        commit_sha: cc.commit_sha,
                        commit_message: cc.commit_message,
                        author: cc.author,
                        source: 'code_change'
                    }}) AS code_changes
            """

            results = await self.graph.query(cypher)

            if not results:
                return []

            result = results[0]

            suggestions = []

            # Add direct healing patterns
            for healing in result.get("direct_healings", []):
                if healing.get("healed_selector"):
                    suggestions.append({
                        "type": "healing_pattern",
                        "source": "knowledge_graph_direct",
                        "healed_selector": healing["healed_selector"],
                        "confidence": healing.get("confidence", 0.5),
                        "success_count": healing.get("success_count", 0),
                        "pattern_id": healing.get("pattern_id"),
                    })

            # Add similar selector healings with lower confidence
            for healing in result.get("similar_healings", []):
                if healing.get("healed_selector"):
                    suggestions.append({
                        "type": "similar_selector_healing",
                        "source": "knowledge_graph_similar",
                        "healed_selector": healing["healed_selector"],
                        "confidence": healing.get("confidence", 0.3) * 0.8,  # Reduce confidence
                        "similar_to": healing.get("similar_selector"),
                        "pattern_id": healing.get("pattern_id"),
                    })

            # Add code change insights
            for change in result.get("code_changes", []):
                suggestions.append({
                    "type": "code_change_insight",
                    "source": "knowledge_graph_code",
                    "commit_sha": change.get("commit_sha"),
                    "commit_message": change.get("commit_message"),
                    "author": change.get("author"),
                    "confidence": 0.6,
                })

            # Sort by confidence
            suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            self._log.info(
                "Found healing suggestions via graph",
                selector=selector[:50],
                suggestion_count=len(suggestions),
            )

            return suggestions

        except Exception as e:
            self._log.error(
                "Failed to find healing suggestions via graph",
                selector=selector,
                error=str(e),
            )
            return []

    async def record_healing_to_graph(
        self,
        test_id: str,
        failure_id: str,
        healing_pattern_id: str,
        original_selector: str,
        healed_selector: str,
        success: bool,
    ) -> bool:
        """Record a healing attempt to the graph.

        Creates relationships:
        - HealingPattern -> FIXES -> Failure
        - HealingPattern -> REPLACES -> Selector (old)
        - Test -> USES -> Selector (new)

        Args:
            test_id: Test UUID
            failure_id: Failure UUID
            healing_pattern_id: Healing pattern UUID
            original_selector: Original broken selector
            healed_selector: Fixed selector
            success: Whether healing succeeded

        Returns:
            True if recorded successfully
        """
        try:
            # Get vertex IDs from mapping tables
            pool = await self.graph._get_pool()

            async with pool.acquire() as conn:
                # Get test vertex
                test_row = await conn.fetchrow(
                    "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
                    test_id,
                )

                # Get or create failure vertex
                failure_row = await conn.fetchrow(
                    "SELECT vertex_id FROM graph_failure_vertices WHERE failure_id = $1",
                    failure_id,
                )

                # Get or create healing pattern vertex
                pattern_row = await conn.fetchrow(
                    "SELECT vertex_id FROM graph_healing_pattern_vertices WHERE pattern_id = $1",
                    healing_pattern_id,
                )

                # Get or create selector vertices
                old_selector_row = await conn.fetchrow(
                    "SELECT vertex_id FROM graph_selector_vertices WHERE selector_text = $1",
                    original_selector,
                )

                new_selector_row = await conn.fetchrow(
                    "SELECT vertex_id FROM graph_selector_vertices WHERE selector_text = $1",
                    healed_selector,
                )

            # Create edges
            if pattern_row and failure_row:
                # HealingPattern -> FIXES -> Failure
                await self.graph.add_edge(
                    from_vertex_id=pattern_row["vertex_id"],
                    to_vertex_id=failure_row["vertex_id"],
                    relationship=EdgeType.FIXES,
                    properties={"success": success},
                )

            if pattern_row and old_selector_row:
                # HealingPattern -> REPLACES -> OldSelector
                await self.graph.add_edge(
                    from_vertex_id=pattern_row["vertex_id"],
                    to_vertex_id=old_selector_row["vertex_id"],
                    relationship=EdgeType.REPLACES,
                    properties={
                        "old_selector": original_selector,
                        "new_selector": healed_selector,
                    },
                )

            if test_row and new_selector_row and success:
                # Test -> USES -> NewSelector (only if healing succeeded)
                await self.graph.add_edge(
                    from_vertex_id=test_row["vertex_id"],
                    to_vertex_id=new_selector_row["vertex_id"],
                    relationship=EdgeType.USES,
                )

            self._log.info(
                "Recorded healing to graph",
                test_id=test_id,
                healing_pattern_id=healing_pattern_id,
                success=success,
            )

            return True

        except Exception as e:
            self._log.error(
                "Failed to record healing to graph",
                test_id=test_id,
                error=str(e),
            )
            return False

    async def get_top_fragile_selectors(
        self,
        project_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get the most fragile selectors in a project.

        Useful for:
        - Identifying technical debt
        - Prioritizing refactoring efforts
        - Understanding test stability risks

        Args:
            project_id: Project UUID
            limit: Maximum number of selectors to return

        Returns:
            List of fragile selectors sorted by fragility score
        """
        try:
            fragile = await self.graph.find_fragile_selectors(
                project_id=project_id,
                min_failures=3,
                limit=limit,
            )

            self._log.info(
                "Found fragile selectors",
                project_id=project_id,
                count=len(fragile),
            )

            return fragile

        except Exception as e:
            self._log.error(
                "Failed to get fragile selectors",
                project_id=project_id,
                error=str(e),
            )
            return []
