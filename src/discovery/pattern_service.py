"""Discovery Pattern Service - Cross-project pattern learning with Cognee.

This service provides:
1. Pattern extraction from discovered pages/elements/flows
2. Embedding generation via Cognee's ECL pipeline
3. Storage in Cognee knowledge layer with semantic search
4. Cross-project pattern matching to improve discovery

The key insight is that UI patterns (login forms, navigation, etc.) are
similar across projects. Learning from one project helps discover patterns
in new projects automatically.

.. note::
    This module has been migrated to use CogneeKnowledgeClient as part of
    RAP-132 (Cognee Consolidation). The old CloudflareVectorize dependency
    has been removed in favor of Cognee's unified knowledge layer.
"""

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass
from enum import Enum

from src.knowledge import CogneeKnowledgeClient, get_cognee_client
from src.services.cache import (
    cache_discovery_pattern,
    get_cached_pattern,
    set_cached_pattern,
)
from src.services.supabase_client import SupabaseClient, get_supabase_client

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Types of discoverable patterns."""

    PAGE_LAYOUT = "page_layout"
    NAVIGATION = "navigation"
    FORM = "form"
    AUTHENTICATION = "authentication"
    ERROR_HANDLING = "error_handling"
    LOADING_STATE = "loading_state"
    MODAL = "modal"
    LIST_VIEW = "list_view"
    DETAIL_VIEW = "detail_view"
    SEARCH = "search"
    FILTER = "filter"
    PAGINATION = "pagination"
    CUSTOM = "custom"


@dataclass
class PatternMatch:
    """Result of pattern similarity search."""

    id: str
    pattern_type: str
    pattern_name: str
    pattern_data: dict
    times_seen: int
    test_success_rate: float
    similarity: float


@dataclass
class DiscoveryPattern:
    """A discovered UI pattern."""

    pattern_type: PatternType
    pattern_name: str
    pattern_signature: str  # Hash for deduplication
    pattern_data: dict  # Full pattern details
    embedding: list[float] | None = None

    @classmethod
    def from_page(cls, page_data: dict) -> "DiscoveryPattern":
        """Create a pattern from a discovered page."""
        # Extract key features for pattern matching
        features = {
            "category": page_data.get("category", "other"),
            "element_types": list(
                set(e.get("category", "unknown") for e in page_data.get("elements", []))
            ),
            "has_forms": any(e.get("category") == "form" for e in page_data.get("elements", [])),
            "has_auth": any(
                e.get("category") == "authentication" for e in page_data.get("elements", [])
            ),
            "element_count": len(page_data.get("elements", [])),
            "title_pattern": _normalize_title(page_data.get("title", "")),
            "url_pattern": _extract_url_pattern(page_data.get("url", "")),
        }

        # Determine pattern type from page category
        pattern_type = _map_page_to_pattern_type(page_data.get("category", "other"))

        # Create pattern signature for deduplication
        signature = _create_signature(features)

        return cls(
            pattern_type=pattern_type,
            pattern_name=f"{features['category']}_page",
            pattern_signature=signature,
            pattern_data={
                "features": features,
                "source_url": page_data.get("url"),
                "source_title": page_data.get("title"),
                "elements_summary": page_data.get("elements_summary", []),
            },
        )

    @classmethod
    def from_flow(cls, flow_data: dict) -> "DiscoveryPattern":
        """Create a pattern from a discovered flow."""
        features = {
            "category": flow_data.get("category", "user_journey"),
            "step_count": len(flow_data.get("steps", [])),
            "step_types": [
                s.get("type", "unknown")
                for s in flow_data.get("steps", [])[:5]  # First 5 steps
            ],
            "has_auth_steps": any(
                "login" in s.get("instruction", "").lower()
                or "sign" in s.get("instruction", "").lower()
                for s in flow_data.get("steps", [])
            ),
            "priority": flow_data.get("priority", "medium"),
        }

        pattern_type = PatternType(flow_data.get("category", "custom"))

        signature = _create_signature(features)

        return cls(
            pattern_type=pattern_type,
            pattern_name=flow_data.get("name", "unknown_flow"),
            pattern_signature=signature,
            pattern_data={
                "features": features,
                "steps": flow_data.get("steps", [])[:5],  # First 5 steps only
                "success_criteria": flow_data.get("success_criteria"),
            },
        )

    @classmethod
    def from_element(cls, element_data: dict) -> "DiscoveryPattern":
        """Create a pattern from a discovered element."""
        features = {
            "category": element_data.get("category", "unknown"),
            "tag_name": element_data.get("tag_name", ""),
            "role": element_data.get("role", ""),
            "has_label": bool(element_data.get("label")),
            "is_interactive": element_data.get("is_visible", True)
            and element_data.get("is_enabled", True),
            "selector_pattern": _normalize_selector(element_data.get("selector", "")),
        }

        pattern_type = _map_element_to_pattern_type(element_data.get("category", "unknown"))

        signature = _create_signature(features)

        return cls(
            pattern_type=pattern_type,
            pattern_name=element_data.get("purpose", element_data.get("category", "unknown")),
            pattern_signature=signature,
            pattern_data={
                "features": features,
                "selector": element_data.get("selector"),
                "alternative_selectors": element_data.get("alternative_selectors", []),
                "aria_label": element_data.get("aria_label"),
            },
        )

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding."""
        parts = [
            f"Pattern type: {self.pattern_type.value}",
            f"Pattern name: {self.pattern_name}",
        ]

        features = self.pattern_data.get("features", {})
        for key, value in features.items():
            if value:
                parts.append(f"{key}: {value}")

        return " | ".join(parts)


class PatternService:
    """Service for cross-project pattern learning.

    Now powered by CogneeKnowledgeClient for unified knowledge management.
    Supabase is kept for backwards compatibility with dashboard queries.
    """

    def __init__(
        self,
        supabase: SupabaseClient | None = None,
        cognee_client: CogneeKnowledgeClient | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
    ):
        self.supabase = supabase or get_supabase_client()
        self.cognee = cognee_client or get_cognee_client(
            org_id=org_id,
            project_id=project_id,
        )

    async def extract_and_store_patterns(
        self,
        session_id: str,
        project_id: str,
        pages: list[dict],
        flows: list[dict],
        elements: list[dict],
    ) -> dict:
        """Extract patterns from discovery results and store them.

        Args:
            session_id: Discovery session ID
            project_id: Project ID
            pages: Discovered pages
            flows: Discovered flows
            elements: Discovered elements

        Returns:
            Summary of stored patterns
        """
        stored_count = 0
        updated_count = 0
        errors = []

        # Extract patterns from pages
        for page in pages:
            try:
                pattern = DiscoveryPattern.from_page(page)
                result = await self._store_pattern(pattern, project_id)
                if result.get("created"):
                    stored_count += 1
                elif result.get("updated"):
                    updated_count += 1
            except Exception as e:
                logger.error(f"Failed to extract pattern from page: {e}")
                errors.append(f"page:{page.get('url', 'unknown')}")

        # Extract patterns from flows
        for flow in flows:
            try:
                pattern = DiscoveryPattern.from_flow(flow)
                result = await self._store_pattern(pattern, project_id)
                if result.get("created"):
                    stored_count += 1
                elif result.get("updated"):
                    updated_count += 1
            except Exception as e:
                logger.error(f"Failed to extract pattern from flow: {e}")
                errors.append(f"flow:{flow.get('name', 'unknown')}")

        # Extract patterns from key elements (limit to avoid too many)
        key_elements = [
            e for e in elements if e.get("category") in ["authentication", "form", "navigation"]
        ][:50]  # Limit to 50 key elements

        for element in key_elements:
            try:
                pattern = DiscoveryPattern.from_element(element)
                result = await self._store_pattern(pattern, project_id)
                if result.get("created"):
                    stored_count += 1
                elif result.get("updated"):
                    updated_count += 1
            except Exception as e:
                logger.error(f"Failed to extract pattern from element: {e}")
                errors.append(f"element:{element.get('selector', 'unknown')[:50]}")

        return {
            "stored": stored_count,
            "updated": updated_count,
            "errors": errors,
        }

    async def _store_pattern(self, pattern: DiscoveryPattern, project_id: str) -> dict:
        """Store a pattern in the database with embedding.

        Stores in both Cognee (for semantic search) and Supabase (for dashboard queries).
        """
        # Check if pattern already exists by signature
        existing = await self._find_by_signature(pattern.pattern_signature)

        if existing:
            # Update times_seen in both systems
            await self._update_existing_pattern(existing["id"], project_id)
            # Also increment in Cognee
            try:
                await self.cognee.increment_pattern_times_seen(
                    pattern_id=existing["id"],
                    pattern_type=pattern.pattern_type.value,
                )
            except Exception as e:
                logger.warning(f"Failed to increment pattern in Cognee: {e}")
            return {"updated": True, "id": existing["id"]}

        # Store in Cognee first (handles embeddings internally)
        try:
            cognee_id = await self.cognee.store_discovery_pattern(
                pattern_type=pattern.pattern_type.value,
                pattern_name=pattern.pattern_name,
                pattern_signature=pattern.pattern_signature,
                pattern_data=pattern.pattern_data,
                source_url=pattern.pattern_data.get("source_url"),
                source_project_id=project_id,
            )
            logger.debug(f"Stored pattern in Cognee: {cognee_id}")
        except Exception as e:
            logger.warning(f"Failed to store pattern in Cognee: {e}")
            cognee_id = None

        # Also store in Supabase for dashboard backwards compatibility
        # Generate embedding for legacy storage (optional)
        embedding = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                embedding = await self._generate_embedding(pattern)
        except Exception:
            pass  # Embeddings optional for Supabase

        result = await self.supabase.request(
            "/discovery_patterns",
            method="POST",
            body={
                "pattern_type": pattern.pattern_type.value,
                "pattern_name": pattern.pattern_name,
                "pattern_signature": pattern.pattern_signature,
                "pattern_data": pattern.pattern_data,
                "embedding": embedding,
                "times_seen": 1,
                "projects_seen": 1,
            },
        )

        if result.get("error"):
            logger.error(f"Failed to store pattern in Supabase: {result['error']}")
            # Still return success if Cognee worked
            if cognee_id:
                return {"created": True, "id": cognee_id, "cognee_only": True}
            return {"error": result["error"]}

        return {"created": True, "id": result["data"][0]["id"] if result["data"] else cognee_id}

    async def _find_by_signature(self, signature: str) -> dict | None:
        """Find pattern by signature hash."""
        result = await self.supabase.request(
            f"/discovery_patterns?pattern_signature=eq.{signature}&limit=1"
        )

        if result.get("data") and len(result["data"]) > 0:
            return result["data"][0]
        return None

    async def _update_existing_pattern(self, pattern_id: str, project_id: str) -> dict:
        """Increment times_seen for existing pattern using atomic RPC function."""
        # Use PostgreSQL RPC function for atomic increment to prevent race conditions
        result = await self.supabase.request(
            "/rpc/increment_pattern_times_seen", method="POST", body={"pattern_id": pattern_id}
        )

        if result.get("data"):
            return {
                "updated": True,
                "times_seen": result["data"][0].get("times_seen") if result["data"] else None,
            }

        # Log error but don't fail - the pattern still exists
        logger.warning(f"Failed to atomically increment pattern times_seen: {result.get('error')}")
        return {"error": "Failed to update pattern", "details": result.get("error")}

    async def _generate_embedding(self, pattern: DiscoveryPattern) -> list[float] | None:
        """Generate embedding for pattern using available services.

        .. deprecated::
            Cognee handles embeddings internally via its ECL pipeline.
            This method is kept for backwards compatibility with Supabase storage.
            New code should use CogneeKnowledgeClient methods directly.
        """
        warnings.warn(
            "_generate_embedding is deprecated. Cognee handles embeddings internally.",
            DeprecationWarning,
            stacklevel=2,
        )
        text = pattern.to_embedding_text()

        # Try local embedder for Supabase backwards compatibility
        try:
            from src.indexer.local_embedder import get_embedder

            embedder = get_embedder()
            if embedder.is_available:
                result = embedder.embed(text)
                if result:
                    return _pad_embedding(result.embedding, 1536)
        except Exception as e:
            logger.warning(f"Local embedding failed: {e}")

        return None

    async def find_similar_patterns(
        self,
        query_pattern: DiscoveryPattern,
        pattern_type: PatternType | None = None,
        threshold: float = 0.7,
        limit: int = 5,
        use_cache: bool = True,
    ) -> list[PatternMatch]:
        """Find patterns similar to the query pattern.

        Uses Cognee's semantic search for similarity matching with
        Valkey caching for frequently accessed patterns.

        Args:
            query_pattern: Pattern to search for
            pattern_type: Optional filter by pattern type
            threshold: Minimum similarity score (0-1)
            limit: Maximum results
            use_cache: Whether to use Valkey cache (default: True)

        Returns:
            List of matching patterns sorted by similarity
        """
        # Generate query text from pattern
        query_text = query_pattern.to_embedding_text()

        # Generate cache key from query parameters
        cache_key = hashlib.sha256(
            f"{query_text}:{pattern_type}:{threshold}:{limit}".encode()
        ).hexdigest()[:16]

        # Check Valkey cache first (for hot patterns)
        if use_cache:
            cached = await get_cached_pattern(cache_key, "similar_search")
            if cached:
                logger.debug(f"Pattern search cache HIT: {cache_key}")
                return [PatternMatch(**m) for m in cached.get("matches", [])]

        try:
            # Use Cognee's semantic search
            cognee_results = await self.cognee.find_similar_discovery_patterns(
                query_text=query_text,
                pattern_type=pattern_type.value if pattern_type else None,
                limit=limit,
                min_similarity=threshold,
            )

            matches = []
            for row in cognee_results:
                matches.append(
                    PatternMatch(
                        id=row.get("id", ""),
                        pattern_type=row.get("pattern_type", "custom"),
                        pattern_name=row.get("pattern_name", ""),
                        pattern_data=row.get("pattern_data", {}),
                        times_seen=row.get("times_seen", 1),
                        test_success_rate=float(row.get("test_success_rate", 0)),
                        similarity=float(row.get("similarity", 0)),
                    )
                )

            # Cache results in Valkey for fast subsequent lookups
            if use_cache and matches:
                await set_cached_pattern(
                    cache_key,
                    "similar_search",
                    {"matches": [
                        {
                            "id": m.id,
                            "pattern_type": m.pattern_type,
                            "pattern_name": m.pattern_name,
                            "pattern_data": m.pattern_data,
                            "times_seen": m.times_seen,
                            "test_success_rate": m.test_success_rate,
                            "similarity": m.similarity,
                        }
                        for m in matches
                    ]},
                    ttl_seconds=900,  # 15 minutes
                )

            return matches

        except Exception as e:
            logger.warning(f"Cognee pattern search failed: {e}, falling back to Supabase")

            # Fallback to Supabase pgvector search
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                embedding = await self._generate_embedding(query_pattern)

            if not embedding:
                logger.warning("Could not generate embedding for similarity search")
                return []

            params = {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": limit,
            }

            if pattern_type:
                params["pattern_type_filter"] = pattern_type.value

            result = await self.supabase.request(
                "/rpc/search_similar_discovery_patterns",
                method="POST",
                body=params,
            )

            if result.get("error"):
                logger.error(f"Pattern search failed: {result['error']}")
                return []

            matches = []
            for row in result.get("data", []):
                matches.append(
                    PatternMatch(
                        id=row["id"],
                        pattern_type=row["pattern_type"],
                        pattern_name=row["pattern_name"],
                        pattern_data=row["pattern_data"],
                        times_seen=row["times_seen"],
                        test_success_rate=float(row.get("test_success_rate", 0)),
                        similarity=row["similarity"],
                    )
                )

            return matches

    async def get_patterns_for_project(self, project_id: str) -> list[dict]:
        """Get all patterns associated with a project."""
        # Note: This requires the project_id to be stored with patterns
        # Currently we track projects_seen count, not individual project IDs
        # This is a simplified implementation
        result = await self.supabase.request("/discovery_patterns?order=times_seen.desc&limit=100")
        return result.get("data", [])

    async def get_patterns_for_session(self, session_id: str, limit: int = 50) -> list[dict]:
        """Get patterns associated with a specific discovery session.

        Patterns are linked to sessions via the pattern extraction process.
        This enables tracing which patterns came from which exploration.

        Args:
            session_id: Discovery session ID
            limit: Maximum patterns to return

        Returns:
            List of patterns associated with the session
        """
        # Try to get patterns that have this session_id in their metadata
        # This requires patterns to store source session information
        result = await self.supabase.request(
            f"/discovery_patterns?order=times_seen.desc&limit={limit}"
        )

        patterns = result.get("data", [])

        # Filter patterns that might be linked to this session
        # In the full implementation, we would store session_id with patterns
        # For now, return all recent patterns (frontend has fallback)
        return patterns[:limit]

    async def store_pattern(self, pattern_data: dict) -> dict:
        """Store a pattern in the database.

        This is the public interface for creating patterns.
        For internal pattern extraction, use extract_and_store_patterns instead.

        Stores in both Cognee (for semantic search) and Supabase (for dashboard).

        Args:
            pattern_data: Pattern data including pattern_type, pattern_name,
                         pattern_signature, and pattern_data

        Returns:
            Created pattern record or error
        """
        # Check if pattern already exists by signature
        signature = pattern_data.get("pattern_signature", "")
        if signature:
            existing = await self._find_by_signature(signature)
            if existing:
                # Update times_seen in both systems
                project_id = None
                projects = pattern_data.get("projects_seen", [])
                if projects:
                    project_id = projects[0]
                await self._update_existing_pattern(existing["id"], project_id)
                # Also update in Cognee
                try:
                    await self.cognee.increment_pattern_times_seen(
                        pattern_id=existing["id"],
                        pattern_type=pattern_data.get("pattern_type", "custom"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to increment in Cognee: {e}")
                existing["times_seen"] = existing.get("times_seen", 0) + 1
                return existing

        # Store in Cognee first (handles embeddings internally)
        cognee_id = None
        try:
            cognee_id = await self.cognee.store_discovery_pattern(
                pattern_type=pattern_data.get("pattern_type", "custom"),
                pattern_name=pattern_data.get("pattern_name", ""),
                pattern_signature=pattern_data.get("pattern_signature", ""),
                pattern_data=pattern_data.get("pattern_data", {}),
                source_url=pattern_data.get("pattern_data", {}).get("source_url"),
                source_project_id=pattern_data.get("projects_seen", [None])[0],
            )
        except Exception as e:
            logger.warning(f"Failed to store in Cognee: {e}")

        # Generate embedding for legacy Supabase storage
        embedding = None
        if pattern_data.get("pattern_data"):
            try:
                pattern_type_str = pattern_data.get("pattern_type", "custom")
                try:
                    pattern_type = PatternType(pattern_type_str)
                except ValueError:
                    pattern_type = PatternType.CUSTOM

                temp_pattern = DiscoveryPattern(
                    pattern_type=pattern_type,
                    pattern_name=pattern_data.get("pattern_name", ""),
                    pattern_signature=pattern_data.get("pattern_signature", ""),
                    pattern_data=pattern_data.get("pattern_data", {}),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    embedding = await self._generate_embedding(temp_pattern)
            except Exception as e:
                logger.warning(f"Could not generate embedding: {e}")

        # Prepare database record
        db_record = {
            "pattern_type": pattern_data.get("pattern_type"),
            "pattern_name": pattern_data.get("pattern_name"),
            "pattern_signature": pattern_data.get("pattern_signature"),
            "pattern_data": pattern_data.get("pattern_data", {}),
            "times_seen": pattern_data.get("times_seen", 1),
            "test_success_rate": pattern_data.get("test_success_rate"),
            "self_heal_success_rate": pattern_data.get("self_heal_success_rate"),
        }

        if embedding:
            db_record["embedding"] = embedding

        # Store in Supabase for dashboard compatibility
        result = await self.supabase.request(
            "/discovery_patterns",
            method="POST",
            body=db_record,
        )

        if result.get("error"):
            logger.error(f"Failed to store pattern in Supabase: {result['error']}")
            if cognee_id:
                # Return Cognee ID if Supabase failed
                return {"id": cognee_id, "cognee_only": True, **pattern_data}
            raise Exception(f"Failed to store pattern: {result['error']}")

        if result.get("data") and len(result["data"]) > 0:
            return result["data"][0]

        return pattern_data

    async def get_pattern_insights(
        self,
        pattern_type: PatternType | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Get insights about stored patterns.

        Returns statistics about pattern types, success rates, etc.
        Uses Cognee with Supabase fallback and Valkey caching.

        Args:
            pattern_type: Optional filter by pattern type
            use_cache: Whether to use Valkey cache (default: True)

        Returns:
            Dict with pattern insights and statistics
        """
        # Check cache first
        cache_key = f"insights:{pattern_type.value if pattern_type else 'all'}"
        if use_cache:
            cached = await get_cached_pattern(cache_key, "insights")
            if cached:
                logger.debug(f"Pattern insights cache HIT: {cache_key}")
                return cached

        insights = None

        # Try Cognee first
        try:
            cognee_insights = await self.cognee.get_discovery_pattern_insights(
                pattern_type=pattern_type.value if pattern_type else None,
            )
            if cognee_insights and cognee_insights.get("total_patterns", 0) > 0:
                insights = cognee_insights
        except Exception as e:
            logger.warning(f"Cognee insights failed: {e}, falling back to Supabase")

        # Fallback to Supabase if Cognee didn't return results
        if insights is None:
            filters = ""
            if pattern_type:
                filters = f"&pattern_type=eq.{pattern_type.value}"

            result = await self.supabase.request(
                f"/discovery_patterns?select=pattern_type,times_seen,test_success_rate,self_heal_success_rate{filters}"
            )

            if result.get("error"):
                return {"error": result["error"]}

            patterns = result.get("data", [])

            # Calculate insights
            by_type = {}
            total_patterns = len(patterns)
            total_seen = 0
            avg_test_success = 0
            avg_heal_success = 0

            for p in patterns:
                ptype = p["pattern_type"]
                if ptype not in by_type:
                    by_type[ptype] = {"count": 0, "total_seen": 0}
                by_type[ptype]["count"] += 1
                by_type[ptype]["total_seen"] += p.get("times_seen", 0)
                total_seen += p.get("times_seen", 0)
                avg_test_success += float(p.get("test_success_rate", 0) or 0)
                avg_heal_success += float(p.get("self_heal_success_rate", 0) or 0)

            if total_patterns > 0:
                avg_test_success /= total_patterns
                avg_heal_success /= total_patterns

            insights = {
                "total_patterns": total_patterns,
                "total_occurrences": total_seen,
                "by_type": by_type,
                "avg_test_success_rate": round(avg_test_success, 2),
                "avg_self_heal_success_rate": round(avg_heal_success, 2),
            }

        # Cache the results (5 minute TTL for insights)
        if use_cache and insights and not insights.get("error"):
            await set_cached_pattern(cache_key, "insights", insights, ttl_seconds=300)

        return insights

    async def update_pattern_success_rate(
        self,
        pattern_id: str,
        test_passed: bool,
        self_healed: bool = False,
        pattern_type: str | None = None,
    ) -> bool:
        """Update pattern success rates after test execution.

        This enables learning: patterns with high success rates
        are prioritized in future discoveries.

        Updates both Cognee (for semantic search ranking) and Supabase (for dashboard).
        """
        success = True

        # Update in Cognee first
        try:
            await self.cognee.update_discovery_pattern_stats(
                pattern_id=pattern_id,
                pattern_type=pattern_type or "custom",
                test_passed=test_passed,
                self_healed=self_healed,
            )
        except Exception as e:
            logger.warning(f"Failed to update pattern stats in Cognee: {e}")
            success = False

        # Also update in Supabase for backwards compatibility
        try:
            fetch = await self.supabase.request(f"/discovery_patterns?id=eq.{pattern_id}")

            if fetch.get("data") and len(fetch["data"]) > 0:
                current = fetch["data"][0]
                times_seen = current.get("times_seen", 1)
                current_test_rate = float(current.get("test_success_rate", 0) or 0)
                current_heal_rate = float(current.get("self_heal_success_rate", 0) or 0)

                # Calculate new rolling average
                new_test_rate = (
                    (current_test_rate * (times_seen - 1)) + (100 if test_passed else 0)
                ) / times_seen
                new_heal_rate = current_heal_rate

                if self_healed:
                    new_heal_rate = ((current_heal_rate * (times_seen - 1)) + 100) / times_seen

                result = await self.supabase.request(
                    f"/discovery_patterns?id=eq.{pattern_id}",
                    method="PATCH",
                    body={
                        "test_success_rate": round(new_test_rate, 2),
                        "self_heal_success_rate": round(new_heal_rate, 2),
                        "updated_at": "now()",
                    },
                )

                if result.get("error"):
                    logger.warning(f"Failed to update pattern in Supabase: {result['error']}")
                    success = False
        except Exception as e:
            logger.warning(f"Failed to update pattern in Supabase: {e}")
            success = False

        return success


# Utility functions


def _normalize_title(title: str) -> str:
    """Normalize page title to pattern form."""
    # Remove specific words/values, keep structure
    import re

    # Remove UUIDs first (before number replacement mangles them)
    title = re.sub(
        r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "UUID", title, flags=re.I
    )
    # Remove numbers
    title = re.sub(r"\d+", "#", title)
    return title.lower().strip()


def _extract_url_pattern(url: str) -> str:
    """Extract URL pattern (remove IDs, keep structure)."""
    import re
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path

    # Replace numeric IDs with placeholder
    path = re.sub(r"/\d+(?=/|$)", "/:id", path)
    # Replace UUIDs
    path = re.sub(
        r"/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}(?=/|$)",
        "/:uuid",
        path,
        flags=re.I,
    )

    return path


def _normalize_selector(selector: str) -> str:
    """Normalize selector to pattern form."""
    import re

    # Remove specific IDs
    selector = re.sub(r"#[a-zA-Z0-9_-]+", "#ID", selector)
    # Remove specific classes but keep count
    classes = re.findall(r"\.[a-zA-Z0-9_-]+", selector)
    if classes:
        selector = re.sub(r"(\.[a-zA-Z0-9_-]+)+", f".CLASS[{len(classes)}]", selector)

    return selector


def _create_signature(features: dict) -> str:
    """Create a unique signature hash for deduplication."""
    # Sort keys for consistent hashing
    canonical = json.dumps(features, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _map_page_to_pattern_type(category: str) -> PatternType:
    """Map page category to pattern type."""
    mapping = {
        "auth_login": PatternType.AUTHENTICATION,
        "auth_signup": PatternType.AUTHENTICATION,
        "auth_reset": PatternType.AUTHENTICATION,
        "list": PatternType.LIST_VIEW,
        "detail": PatternType.DETAIL_VIEW,
        "form": PatternType.FORM,
        "landing": PatternType.PAGE_LAYOUT,
        "dashboard": PatternType.PAGE_LAYOUT,
        "error": PatternType.ERROR_HANDLING,
        "settings": PatternType.FORM,
        "profile": PatternType.DETAIL_VIEW,
    }
    return mapping.get(category, PatternType.CUSTOM)


def _map_element_to_pattern_type(category: str) -> PatternType:
    """Map element category to pattern type."""
    mapping = {
        "navigation": PatternType.NAVIGATION,
        "form": PatternType.FORM,
        "authentication": PatternType.AUTHENTICATION,
        "interactive": PatternType.CUSTOM,
        "action": PatternType.CUSTOM,
        "content": PatternType.PAGE_LAYOUT,
    }
    return mapping.get(category, PatternType.CUSTOM)


def _pad_embedding(embedding: list[float], target_dim: int) -> list[float]:
    """Pad or truncate embedding to target dimension."""
    current_dim = len(embedding)

    if current_dim == target_dim:
        return embedding
    elif current_dim > target_dim:
        return embedding[:target_dim]
    else:
        # Pad with zeros
        return embedding + [0.0] * (target_dim - current_dim)


# Global instance
_pattern_service: PatternService | None = None


def get_pattern_service() -> PatternService:
    """Get or create global pattern service instance."""
    global _pattern_service
    if _pattern_service is None:
        _pattern_service = PatternService()
    return _pattern_service
