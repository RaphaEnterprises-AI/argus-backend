"""Healing Pattern Service - Business Logic Layer.

Provides centralized healing pattern storage and retrieval for:
- Healing Consumer Worker (Kafka event processing)
- Healing API endpoints
- SelfHealerAgent

This service layer ensures consistent data access, validation,
and storage to both Supabase (for API queries) and Cognee (for semantic search).

RAP-356 / RAP-357: Added unified heal() entry point that orchestrates:
1. Cognee pattern search (semantic search for similar past failures)
2. Code analyzer integration (GitHubCodeAnalyzer for remote repos)
3. SelfHealerAgent LLM fallback
4. Pattern storage on success (learning loop)
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)


# =============================================================================
# RAP-356 / RAP-357: Unified Healing Entry Point Data Classes
# =============================================================================


class DeploymentMode(str, Enum):
    """Deployment mode for code analyzer selection (RAP-357)."""

    AUTO = "auto"  # Auto-detect: prefer local, fallback to remote
    LOCAL = "local"  # Force local git repository
    REMOTE = "remote"  # Force remote API (GitHub/GitLab)


@dataclass
class TestFailure:
    """Input data for a test failure that needs healing.

    Attributes:
        error_type: Type of error (selector_not_found, timeout, assertion_failed, etc.)
        selector: The failed selector (CSS/XPath) - required for selector-based fixes
        error_message: Full error message for semantic search
        page_url: URL where failure occurred (for context)
        test_id: Optional test ID for tracking
        failure_id: Optional failure ID for tracking
        step_index: Optional step index where failure occurred
        dom_snapshot: Optional DOM snapshot at failure time
        screenshot_base64: Optional screenshot for visual analysis
    """
    error_type: str
    selector: str | None = None
    error_message: str | None = None
    page_url: str | None = None
    test_id: str | None = None
    failure_id: str | None = None
    step_index: int | None = None
    dom_snapshot: str | None = None
    screenshot_base64: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "selector": self.selector,
            "error_message": self.error_message,
            "page_url": self.page_url,
            "test_id": self.test_id,
            "failure_id": self.failure_id,
            "step_index": self.step_index,
            "dom_snapshot": self.dom_snapshot,
            "screenshot_base64": self.screenshot_base64,
        }


@dataclass
class HealingResult:
    """Result from the unified healing flow.

    Attributes:
        success: Whether healing produced a viable fix
        fix_applied: The fix that was generated (if any)
        strategy_used: Which strategy produced the fix
        confidence: Confidence score (0.0 to 1.0)
        pattern_id: ID of Cognee pattern used/stored (for learning loop)
        code_context: Git/code context if code-aware healing was used
        error: Error message if healing failed
    """
    success: bool
    fix_applied: dict | None = None
    strategy_used: str | None = None  # "cognee_pattern", "code_aware", "llm_fallback"
    confidence: float = 0.0
    pattern_id: str | None = None
    code_context: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "fix_applied": self.fix_applied,
            "strategy_used": self.strategy_used,
            "confidence": self.confidence,
            "pattern_id": self.pattern_id,
            "code_context": self.code_context,
            "error": self.error,
        }


class HealingPatternCreate(BaseModel):
    """Request model for creating a healing pattern."""

    original_selector: str = Field(..., description="The original/broken selector")
    healed_selector: str = Field(..., description="The fixed selector")
    error_type: str = Field(..., description="Type of error (selector_not_found, etc.)")
    project_id: str = Field(..., description="Project ID the pattern belongs to")
    page_url: str | None = Field(None, description="Page URL where failure occurred")
    element_context: dict | None = Field(None, description="DOM context around element")
    metadata: dict | None = Field(None, description="Additional metadata (test_id, failure_id, etc.)")


class HealingPatternResult(BaseModel):
    """Result model for healing pattern operations."""

    success: bool
    pattern_id: str | None = None
    fingerprint: str | None = None
    is_new: bool = False
    message: str | None = None
    error: str | None = None


class HealingService:
    """
    Service layer for healing pattern operations.

    Handles storage to both Supabase (for API queries) and Cognee (for semantic learning).
    """

    def __init__(
        self,
        org_id: str | None = None,
        project_id: str | None = None,
    ):
        self.org_id = org_id
        self.project_id = project_id
        self._supabase = None

    @property
    def supabase(self):
        """Lazy-load Supabase client."""
        if self._supabase is None:
            self._supabase = get_supabase_client()
        return self._supabase

    def _generate_fingerprint(self, original_selector: str, error_type: str) -> str:
        """Generate a unique fingerprint for deduplication."""
        return hashlib.sha256(
            f"{original_selector}:{error_type}".encode()
        ).hexdigest()[:32]

    async def store_pattern(
        self,
        pattern: HealingPatternCreate,
        store_in_cognee: bool = True,
    ) -> HealingPatternResult:
        """
        Store a healing pattern in both Supabase and Cognee.

        Args:
            pattern: The healing pattern to store
            store_in_cognee: Whether to also store in Cognee for semantic search

        Returns:
            HealingPatternResult with pattern_id and status
        """
        fingerprint = self._generate_fingerprint(pattern.original_selector, pattern.error_type)
        pattern_id = None
        is_new = False

        # 1. Store in Supabase (for API queries)
        try:
            if not self.supabase.is_configured:
                logger.warning("Supabase not configured, skipping pattern storage")
                return HealingPatternResult(
                    success=False,
                    error="Supabase not configured",
                )

            # Check if pattern exists (upsert logic)
            existing = await self.supabase.request(
                f"/healing_patterns?fingerprint=eq.{fingerprint}&select=id,success_count"
            )
            existing_data = existing.get("data", [])

            if existing_data:
                # Update existing pattern - increment success count
                pattern_id = existing_data[0]["id"]
                new_count = existing_data[0].get("success_count", 0) + 1

                update_result = await self.supabase.request(
                    f"/healing_patterns?id=eq.{pattern_id}",
                    method="PATCH",
                    body={
                        "success_count": new_count,
                        "healed_selector": pattern.healed_selector,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )

                if update_result.get("error"):
                    logger.error(
                        "Failed to update healing pattern in Supabase",
                        pattern_id=pattern_id,
                        error=update_result["error"],
                    )
                    return HealingPatternResult(
                        success=False,
                        pattern_id=pattern_id,
                        fingerprint=fingerprint,
                        error=update_result["error"],
                    )

                logger.info(
                    "Updated existing healing pattern",
                    pattern_id=pattern_id,
                    success_count=new_count,
                )
            else:
                # Insert new pattern
                is_new = True
                insert_result = await self.supabase.request(
                    "/healing_patterns",
                    method="POST",
                    body={
                        "fingerprint": fingerprint,
                        "original_selector": pattern.original_selector,
                        "healed_selector": pattern.healed_selector,
                        "error_type": pattern.error_type,
                        "project_id": pattern.project_id,
                        "page_url": pattern.page_url,
                        "element_context": pattern.element_context or {},
                        "metadata": pattern.metadata or {},
                        "success_count": 1,
                        "failure_count": 0,
                    },
                )

                if insert_result.get("error"):
                    logger.error(
                        "Failed to insert healing pattern in Supabase",
                        error=insert_result["error"],
                    )
                    return HealingPatternResult(
                        success=False,
                        fingerprint=fingerprint,
                        error=insert_result["error"],
                    )

                if insert_result.get("data"):
                    pattern_id = insert_result["data"][0]["id"]
                    logger.info(
                        "Created new healing pattern",
                        pattern_id=pattern_id,
                        fingerprint=fingerprint,
                    )

        except Exception as e:
            logger.exception("Exception storing healing pattern in Supabase", error=str(e))
            return HealingPatternResult(
                success=False,
                fingerprint=fingerprint,
                error=str(e),
            )

        # 2. Store in Cognee (for semantic search) if enabled
        if store_in_cognee and self.org_id and self.project_id:
            try:
                from src.knowledge import get_cognee_client

                cognee = get_cognee_client(
                    org_id=self.org_id,
                    project_id=self.project_id,
                )

                if cognee:
                    metadata = pattern.metadata or {}
                    await cognee.store_failure_pattern(
                        error_message=metadata.get("error_message", f"Selector not found: {pattern.original_selector}"),
                        error_type=pattern.error_type,
                        original_selector=pattern.original_selector,
                        healed_selector=pattern.healed_selector,
                        healing_method=metadata.get("strategy_used", "unknown"),
                        test_id=metadata.get("test_id"),
                        metadata={
                            "failure_id": metadata.get("failure_id"),
                            "auto_healed": metadata.get("auto_healed", False),
                            "confidence": metadata.get("confidence", 0.0),
                            "supabase_pattern_id": pattern_id,
                            "fingerprint": fingerprint,
                        },
                    )
                    logger.info(
                        "Stored healing pattern in Cognee",
                        pattern_id=pattern_id,
                        org_id=self.org_id,
                    )

            except Exception as e:
                # Don't fail the whole operation if Cognee fails
                logger.error(
                    "Failed to store healing pattern in Cognee",
                    error=str(e),
                    pattern_id=pattern_id,
                )

        return HealingPatternResult(
            success=True,
            pattern_id=pattern_id,
            fingerprint=fingerprint,
            is_new=is_new,
            message="Pattern stored successfully" if is_new else "Pattern updated successfully",
        )

    # =========================================================================
    # RAP-356 / RAP-357: Unified Healing Entry Point
    # =========================================================================

    async def heal(
        self,
        test_id: str,
        org_id: str,
        project_id: str,
        failure: TestFailure,
        local_repo_path: str | None = None,
        deployment_mode: DeploymentMode = DeploymentMode.AUTO,
        auto_heal_threshold: float = 0.9,
    ) -> HealingResult:
        """Unified healing entry point that orchestrates the full healing flow.

        This method implements the healing priority:
        1. Search Cognee for similar patterns (fastest, highest accuracy)
        2. Use code analyzer (git history for selector changes)
        3. Fall back to LLM analysis (most flexible, slowest)
        4. On success, store the pattern in Cognee for learning

        Args:
            test_id: Unique identifier for the test
            org_id: Organization ID for multi-tenant isolation
            project_id: Project ID for pattern scoping
            failure: TestFailure containing error details
            local_repo_path: Optional path to local git repository
            deployment_mode: How to select code analyzer (AUTO, LOCAL, REMOTE)
            auto_heal_threshold: Minimum confidence to consider healing successful

        Returns:
            HealingResult with fix details or error

        Example:
            ```python
            service = HealingService()
            result = await service.heal(
                test_id="test_123",
                org_id="org_456",
                project_id="proj_789",
                failure=TestFailure(
                    error_type="selector_not_found",
                    selector="#login-btn",
                    error_message="Element not found: #login-btn",
                ),
            )
            if result.success:
                print(f"Fix found: {result.fix_applied}")
                print(f"Strategy: {result.strategy_used}")
            ```
        """
        log = logger.bind(
            test_id=test_id,
            org_id=org_id,
            project_id=project_id,
            error_type=failure.error_type,
        )
        log.info("Starting unified healing flow")

        # Update service context for pattern storage
        self.org_id = org_id
        self.project_id = project_id

        # =====================================================================
        # STEP 1: Search Cognee for similar patterns
        # =====================================================================
        if failure.error_message:
            cognee_result = await self._try_cognee_pattern(
                failure=failure,
                org_id=org_id,
                project_id=project_id,
                auto_heal_threshold=auto_heal_threshold,
                log=log,
            )
            if cognee_result and cognee_result.success:
                return cognee_result

        # =====================================================================
        # STEP 2: Try code-aware healing (git history analysis)
        # =====================================================================
        if failure.selector:
            code_aware_result = await self._try_code_aware_healing(
                failure=failure,
                org_id=org_id,
                project_id=project_id,
                local_repo_path=local_repo_path,
                deployment_mode=deployment_mode,
                auto_heal_threshold=auto_heal_threshold,
                log=log,
            )
            if code_aware_result and code_aware_result.success:
                # Store successful pattern for learning
                await self._store_successful_pattern(
                    failure=failure,
                    result=code_aware_result,
                    project_id=project_id,
                    log=log,
                )
                return code_aware_result

        # =====================================================================
        # STEP 3: Fall back to LLM-based healing
        # =====================================================================
        llm_result = await self._try_llm_healing(
            test_id=test_id,
            failure=failure,
            org_id=org_id,
            project_id=project_id,
            auto_heal_threshold=auto_heal_threshold,
            log=log,
        )
        if llm_result and llm_result.success:
            # Store successful pattern for learning
            await self._store_successful_pattern(
                failure=failure,
                result=llm_result,
                project_id=project_id,
                log=log,
            )
            return llm_result

        # =====================================================================
        # No strategy succeeded
        # =====================================================================
        log.warning("All healing strategies failed")
        return HealingResult(
            success=False,
            error="All healing strategies exhausted",
        )

    async def _try_cognee_pattern(
        self,
        failure: TestFailure,
        org_id: str,
        project_id: str,
        auto_heal_threshold: float,
        log: Any,
    ) -> HealingResult | None:
        """Try to find a healing pattern from Cognee semantic search.

        This is the fastest strategy with highest accuracy because it uses
        patterns that have already been proven to work.
        """
        try:
            from src.knowledge import get_cognee_client

            cognee = get_cognee_client(org_id=org_id, project_id=project_id)
            if not cognee:
                log.debug("Cognee client not available")
                return None

            similar_failures = await cognee.find_similar_failures(
                error_message=failure.error_message or f"Error: {failure.selector}",
                limit=5,
                threshold=0.75,
                error_type=failure.error_type if failure.error_type != "unknown" else None,
            )

            if not similar_failures:
                log.debug("No similar patterns found in Cognee")
                return None

            # Find best match with high success rate
            for match in similar_failures:
                if (
                    match.success_rate >= 0.7
                    and match.success_count >= 1
                    and match.healed_selector
                ):
                    # Calculate confidence from similarity and success rate
                    confidence = min(0.95, match.similarity * match.success_rate + 0.1)

                    if confidence >= auto_heal_threshold:
                        log.info(
                            "Found Cognee pattern match",
                            pattern_id=match.id,
                            similarity=match.similarity,
                            success_rate=match.success_rate,
                            healed_selector=match.healed_selector,
                        )

                        # Record that we used this pattern
                        await cognee.record_healing_outcome(match.id, success=True)

                        return HealingResult(
                            success=True,
                            fix_applied={
                                "fix_type": "update_selector",
                                "old_value": failure.selector,
                                "new_value": match.healed_selector,
                            },
                            strategy_used="cognee_pattern",
                            confidence=confidence,
                            pattern_id=match.id,
                        )

            log.debug(
                "Cognee patterns found but none met threshold",
                count=len(similar_failures),
            )
            return None

        except Exception as e:
            log.warning("Cognee pattern search failed", error=str(e))
            return None

    async def _try_code_aware_healing(
        self,
        failure: TestFailure,
        org_id: str,
        project_id: str,
        local_repo_path: str | None,
        deployment_mode: DeploymentMode,
        auto_heal_threshold: float,
        log: Any,
    ) -> HealingResult | None:
        """Try code-aware healing using git history analysis.

        This uses either local git or remote API (GitHub/GitLab) to find
        when and why a selector changed.
        """
        try:
            from src.services.code_analyzer_factory import (
                CodeAnalyzerFactory,
                get_code_analyzer_factory,
            )

            factory = get_code_analyzer_factory()

            # Determine force_remote based on deployment_mode
            force_remote = deployment_mode == DeploymentMode.REMOTE

            # Skip local if mode is REMOTE
            actual_local_path = None
            if deployment_mode != DeploymentMode.REMOTE:
                actual_local_path = local_repo_path

            analyzer = await factory.get_analyzer(
                project_id=project_id,
                org_id=org_id,
                local_repo_path=actual_local_path,
                force_remote=force_remote,
            )

            if not analyzer:
                log.debug(
                    "No code analyzer available",
                    deployment_mode=deployment_mode.value,
                )
                return None

            # Search git history for selector changes
            selector_change = await analyzer.find_replacement_selector(
                failed_selector=failure.selector,
                file_hint=None,  # Could be enhanced with file context
                days=14,
            )

            if not selector_change or not selector_change.new_selector:
                log.debug("No selector change found in code history")
                return None

            # Build code context for response
            code_context = {
                "commit_sha": getattr(selector_change.commit, "sha", None),
                "commit_message": getattr(selector_change.commit, "message", None),
                "commit_author": getattr(selector_change.commit, "author", None),
                "file_changed": selector_change.file_path,
                "old_selector": selector_change.old_selector,
                "new_selector": selector_change.new_selector,
            }

            # Code-aware healing has high confidence
            confidence = 0.99

            log.info(
                "Code-aware healing found replacement",
                old_selector=failure.selector,
                new_selector=selector_change.new_selector,
                commit=code_context.get("commit_sha", "")[:7] if code_context.get("commit_sha") else None,
            )

            return HealingResult(
                success=True,
                fix_applied={
                    "fix_type": "update_selector",
                    "old_value": failure.selector,
                    "new_value": selector_change.new_selector,
                },
                strategy_used="code_aware",
                confidence=confidence,
                code_context=code_context,
            )

        except Exception as e:
            log.warning("Code-aware healing failed", error=str(e))
            return None

    async def _try_llm_healing(
        self,
        test_id: str,
        failure: TestFailure,
        org_id: str,
        project_id: str,
        auto_heal_threshold: float,
        log: Any,
    ) -> HealingResult | None:
        """Try LLM-based healing as a fallback.

        This uses the SelfHealerAgent with full LLM analysis.
        """
        try:
            from src.agents.self_healer import SelfHealerAgent

            agent = SelfHealerAgent(
                auto_heal_threshold=auto_heal_threshold,
                enable_code_aware=False,  # Already tried code-aware above
                enable_memory_store=False,  # Already tried Cognee above
                org_id=org_id,
                project_id=project_id,
            )

            # Build test_spec and failure_details for agent
            test_spec = {"id": test_id}
            failure_details = {
                "type": failure.error_type,
                "selector": failure.selector,
                "target": failure.selector,
                "message": failure.error_message,
                "error": failure.error_message,
                "url": failure.page_url,
                "step_index": failure.step_index,
                "dom_snapshot": failure.dom_snapshot,
            }

            # Decode screenshot if provided
            screenshot_bytes = None
            if failure.screenshot_base64:
                import base64
                screenshot_bytes = base64.b64decode(failure.screenshot_base64)

            log.info("Starting LLM-based healing via SelfHealerAgent")

            result = await agent.execute(
                test_spec=test_spec,
                failure_details=failure_details,
                screenshot=screenshot_bytes,
            )

            if not result.success or not result.data:
                log.warning(
                    "LLM healing did not produce a fix",
                    success=result.success,
                    has_data=result.data is not None,
                    error=result.error,
                )
                return None

            healing_result = result.data

            # Check if we got a viable fix
            if not healing_result.suggested_fixes:
                log.warning("LLM healing returned no suggested fixes")
                return None

            best_fix = healing_result.suggested_fixes[0]
            if best_fix.confidence < auto_heal_threshold:
                log.warning(
                    "LLM fix below threshold",
                    confidence=best_fix.confidence,
                    threshold=auto_heal_threshold,
                    fix_type=best_fix.fix_type.value,
                )
                return None

            log.info(
                "LLM healing produced fix",
                fix_type=best_fix.fix_type.value,
                confidence=best_fix.confidence,
            )

            return HealingResult(
                success=True,
                fix_applied=best_fix.to_dict(),
                strategy_used="llm_fallback",
                confidence=best_fix.confidence,
            )

        except Exception as e:
            log.warning("LLM healing failed", error=str(e))
            return None

    async def _store_successful_pattern(
        self,
        failure: TestFailure,
        result: HealingResult,
        project_id: str,
        log: Any,
    ) -> None:
        """Store a successful healing pattern for future learning.

        This closes the learning loop by storing patterns that worked.
        """
        if not result.fix_applied:
            return

        # Only store selector-based fixes
        fix_type = result.fix_applied.get("fix_type")
        if fix_type != "update_selector":
            log.debug("Skipping pattern storage for non-selector fix", fix_type=fix_type)
            return

        old_selector = result.fix_applied.get("old_value")
        new_selector = result.fix_applied.get("new_value")

        if not old_selector or not new_selector:
            return

        try:
            pattern = HealingPatternCreate(
                original_selector=old_selector,
                healed_selector=new_selector,
                error_type=failure.error_type,
                project_id=project_id,
                page_url=failure.page_url,
                metadata={
                    "test_id": failure.test_id,
                    "failure_id": failure.failure_id,
                    "strategy_used": result.strategy_used,
                    "confidence": result.confidence,
                    "error_message": failure.error_message,
                    "auto_healed": True,
                },
            )

            await self.store_pattern(pattern, store_in_cognee=True)
            log.info(
                "Stored successful healing pattern",
                original=old_selector[:50],
                healed=new_selector[:50],
                strategy=result.strategy_used,
            )

        except Exception as e:
            log.warning("Failed to store healing pattern", error=str(e))

    async def get_pattern_by_fingerprint(self, fingerprint: str) -> dict | None:
        """Get a pattern by its fingerprint."""
        result = await self.supabase.request(
            f"/healing_patterns?fingerprint=eq.{fingerprint}&select=*"
        )
        data = result.get("data", [])
        return data[0] if data else None

    async def increment_success(self, pattern_id: str) -> bool:
        """Increment the success count for a pattern."""
        try:
            result = await self.supabase.rpc("increment_healing_success", {"pattern_id": pattern_id})
            return result.get("error") is None
        except Exception as e:
            logger.error("Failed to increment success count", pattern_id=pattern_id, error=str(e))
            return False

    async def increment_failure(self, pattern_id: str) -> bool:
        """Increment the failure count for a pattern."""
        try:
            result = await self.supabase.rpc("increment_healing_failure", {"pattern_id": pattern_id})
            return result.get("error") is None
        except Exception as e:
            logger.error("Failed to increment failure count", pattern_id=pattern_id, error=str(e))
            return False


# Global service factory
def get_healing_service(
    org_id: str | None = None,
    project_id: str | None = None,
) -> HealingService:
    """Get a HealingService instance."""
    return HealingService(org_id=org_id, project_id=project_id)
