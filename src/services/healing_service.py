"""Healing Pattern Service - Business Logic Layer.

Provides centralized healing pattern storage and retrieval for:
- Healing Consumer Worker (Kafka event processing)
- Healing API endpoints
- SelfHealerAgent

This service layer ensures consistent data access, validation,
and storage to both Supabase (for API queries) and Cognee (for semantic search).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)


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
