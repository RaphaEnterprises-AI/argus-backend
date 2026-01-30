"""Artifact Store for separating large data from LangGraph state.

This module provides a way to store large artifacts (screenshots, videos, HTML)
separately from the conversation state, preventing context overflow while
maintaining full data availability for the frontend.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐
    │  LangGraph      │     │  Artifact Store │
    │  State          │────▶│  (Supabase/S3)  │
    │  (lightweight)  │     │  (full data)    │
    └─────────────────┘     └─────────────────┘
           │                        │
           │ summary + refs         │ full artifacts
           ▼                        ▼
    ┌─────────────────────────────────────────┐
    │              Frontend                    │
    │  (receives both via streaming)          │
    └─────────────────────────────────────────┘

Memory Backend Safeguards:
    - Size limit: 100MB by default (configurable)
    - Item limit: 1000 artifacts by default
    - LRU eviction when limits are reached
    - Production warning and auto-fallback to cloud storage
"""

import hashlib
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger()

# Memory store limits (can be overridden via environment)
DEFAULT_MEMORY_MAX_SIZE_MB = int(os.getenv("ARTIFACT_STORE_MAX_SIZE_MB", "100"))
DEFAULT_MEMORY_MAX_ITEMS = int(os.getenv("ARTIFACT_STORE_MAX_ITEMS", "1000"))

# Production detection
PRODUCTION_ENVIRONMENTS = {"production", "prod", "live"}


def _is_production() -> bool:
    """Detect if running in production environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    railway_env = os.getenv("RAILWAY_ENVIRONMENT", "").lower()
    return env in PRODUCTION_ENVIRONMENTS or railway_env in PRODUCTION_ENVIRONMENTS


@dataclass
class Artifact:
    """An artifact stored separately from LangGraph state."""
    id: str
    type: str  # screenshot, video, html, json
    content: str  # base64 or URL
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    size_bytes: int = 0  # Estimated size in bytes

    def __post_init__(self):
        """Calculate size if not provided."""
        if self.size_bytes == 0:
            # Estimate size: content length + metadata overhead
            self.size_bytes = len(self.content.encode("utf-8", errors="replace"))

    def to_reference(self) -> dict[str, Any]:
        """Return a lightweight reference for storing in LangGraph state."""
        return {
            "artifact_id": self.id,
            "type": self.type,
            "created_at": self.created_at,
            **self.metadata
        }


class LRUArtifactCache:
    """LRU cache for artifacts with size and item limits.

    Prevents OOM by evicting oldest artifacts when limits are reached.
    Uses OrderedDict for O(1) LRU operations.
    """

    def __init__(
        self,
        max_size_mb: int = DEFAULT_MEMORY_MAX_SIZE_MB,
        max_items: int = DEFAULT_MEMORY_MAX_ITEMS,
    ):
        self._store: OrderedDict[str, Artifact] = OrderedDict()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_items = max_items
        self._current_size_bytes = 0
        self._eviction_count = 0

    @property
    def current_size_mb(self) -> float:
        """Current cache size in megabytes."""
        return self._current_size_bytes / (1024 * 1024)

    @property
    def item_count(self) -> int:
        """Number of items in cache."""
        return len(self._store)

    def get(self, artifact_id: str) -> Artifact | None:
        """Get artifact and move to end (most recently used)."""
        if artifact_id in self._store:
            # Move to end (most recently used)
            self._store.move_to_end(artifact_id)
            return self._store[artifact_id]
        return None

    def put(self, artifact: Artifact) -> None:
        """Add artifact to cache, evicting oldest if limits exceeded."""
        # If artifact already exists, remove it first to update size tracking
        if artifact.id in self._store:
            old_artifact = self._store.pop(artifact.id)
            self._current_size_bytes -= old_artifact.size_bytes

        # Evict oldest items until we have room
        self._evict_if_needed(artifact.size_bytes)

        # Add new artifact
        self._store[artifact.id] = artifact
        self._current_size_bytes += artifact.size_bytes

    def _evict_if_needed(self, new_item_size: int) -> None:
        """Evict oldest items until we have room for new item."""
        # Check if we need to evict based on size
        while (
            self._store
            and (self._current_size_bytes + new_item_size) > self._max_size_bytes
        ):
            self._evict_oldest()

        # Check if we need to evict based on item count
        while self._store and len(self._store) >= self._max_items:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) item."""
        if not self._store:
            return

        oldest_id, oldest_artifact = self._store.popitem(last=False)
        self._current_size_bytes -= oldest_artifact.size_bytes
        self._eviction_count += 1

        logger.debug(
            "Evicted artifact from memory cache (LRU)",
            artifact_id=oldest_id,
            artifact_type=oldest_artifact.type,
            freed_bytes=oldest_artifact.size_bytes,
            total_evictions=self._eviction_count,
        )

    def clear(self) -> None:
        """Clear all items from cache."""
        self._store.clear()
        self._current_size_bytes = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "item_count": len(self._store),
            "current_size_mb": round(self.current_size_mb, 2),
            "max_size_mb": self._max_size_bytes / (1024 * 1024),
            "max_items": self._max_items,
            "eviction_count": self._eviction_count,
            "utilization_percent": round(
                (self._current_size_bytes / self._max_size_bytes) * 100
                if self._max_size_bytes > 0 else 0,
                1
            ),
        }


class ArtifactStore:
    """Store and retrieve artifacts separately from LangGraph state.

    This keeps the conversation state lightweight while preserving
    full artifact data for frontend display and later retrieval.

    IMPORTANT: Memory backend is for testing only. In production, use
    "supabase" or "r2" backends to prevent OOM and data loss on restart.

    Usage:
        store = ArtifactStore()

        # Store screenshot, get reference for LangGraph state
        ref = store.store_screenshot(base64_data, {"step": 1})

        # In ToolMessage, store only the reference
        tool_result = {
            "success": True,
            "summary": "Test passed",
            "screenshots": [ref]  # Lightweight reference
        }

        # Frontend can fetch full artifact
        full_data = store.get(ref["artifact_id"])
    """

    def __init__(
        self,
        backend: str = "auto",
        max_memory_size_mb: int = DEFAULT_MEMORY_MAX_SIZE_MB,
        max_memory_items: int = DEFAULT_MEMORY_MAX_ITEMS,
    ):
        """Initialize artifact store.

        Args:
            backend: Storage backend - "auto", "memory", "supabase", or "r2"
                    "auto" selects cloud storage in production, memory in dev
            max_memory_size_mb: Maximum memory cache size (default: 100MB)
            max_memory_items: Maximum number of items in memory (default: 1000)
        """
        # Auto-select backend based on environment
        if backend == "auto":
            if _is_production():
                # Default to R2 in production (zero egress costs)
                backend = os.getenv("ARTIFACT_STORE_BACKEND", "r2")
                logger.info(
                    "Auto-selected cloud storage for production",
                    backend=backend,
                )
            else:
                backend = "memory"
                logger.debug("Using memory backend for development")

        # Warn if memory backend is used in production
        if backend == "memory" and _is_production():
            logger.warning(
                "MEMORY BACKEND IN PRODUCTION: Artifacts will be lost on restart! "
                "OOM risk with large volumes. Set ARTIFACT_STORE_BACKEND=r2 or supabase.",
                max_size_mb=max_memory_size_mb,
                max_items=max_memory_items,
            )

        self.backend = backend
        self._memory_cache = LRUArtifactCache(
            max_size_mb=max_memory_size_mb,
            max_items=max_memory_items,
        )

        # Log configuration
        if backend == "memory":
            logger.info(
                "Artifact store initialized with memory backend",
                max_size_mb=max_memory_size_mb,
                max_items=max_memory_items,
                is_production=_is_production(),
            )

    def store(
        self,
        content: str,
        artifact_type: str,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Store an artifact and return a lightweight reference.

        Args:
            content: The artifact content (base64, URL, or JSON string)
            artifact_type: Type of artifact (screenshot, video, html, json)
            metadata: Optional metadata to include in reference

        Returns:
            Lightweight reference dict for storing in LangGraph state
        """
        # Generate content-based ID for deduplication
        content_hash = hashlib.sha256(content[:1000].encode()).hexdigest()[:12]
        artifact_id = f"{artifact_type}_{content_hash}_{uuid.uuid4().hex[:8]}"

        artifact = Artifact(
            id=artifact_id,
            type=artifact_type,
            content=content,
            metadata=metadata or {}
        )

        # Store based on backend
        if self.backend == "memory":
            self._memory_cache.put(artifact)
            # Log warning periodically when cache is getting full
            cache_stats = self._memory_cache.stats()
            if cache_stats["utilization_percent"] > 80:
                logger.warning(
                    "Memory artifact cache is >80% full, LRU eviction active",
                    **cache_stats,
                )
        elif self.backend == "supabase":
            self._store_supabase(artifact)
        elif self.backend in ("s3", "r2"):
            self._store_r2(artifact)

        logger.info(
            "Stored artifact",
            artifact_id=artifact_id,
            type=artifact_type,
            size_kb=artifact.size_bytes // 1024,
            backend=self.backend,
        )

        return artifact.to_reference()

    def store_screenshot(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Convenience method for storing screenshots."""
        return self.store(base64_data, "screenshot", metadata)

    def store_test_result(
        self,
        result: dict[str, Any],
        extract_artifacts: bool = True
    ) -> dict[str, Any]:
        """Store a test result, extracting large artifacts.

        Args:
            result: Full test result with potential large artifacts
            extract_artifacts: Whether to extract and store artifacts separately

        Returns:
            Lightweight result with artifact references instead of full data
        """
        if not extract_artifacts:
            return result

        lightweight = dict(result)
        artifact_refs: list[dict[str, Any]] = []

        # Extract screenshots
        if "screenshot" in lightweight and lightweight["screenshot"]:
            ref = self.store_screenshot(
                lightweight["screenshot"],
                {"source": "final"}
            )
            artifact_refs.append(ref)
            lightweight["screenshot"] = ref["artifact_id"]

        if "finalScreenshot" in lightweight and lightweight["finalScreenshot"]:
            ref = self.store_screenshot(
                lightweight["finalScreenshot"],
                {"source": "final"}
            )
            artifact_refs.append(ref)
            lightweight["finalScreenshot"] = ref["artifact_id"]

        if "screenshots" in lightweight and isinstance(lightweight["screenshots"], list):
            screenshot_refs = []
            for i, screenshot in enumerate(lightweight["screenshots"]):
                if isinstance(screenshot, str) and len(screenshot) > 1000:
                    ref = self.store_screenshot(screenshot, {"step": i})
                    screenshot_refs.append(ref["artifact_id"])
                    artifact_refs.append(ref)
                else:
                    screenshot_refs.append(screenshot)
            lightweight["screenshots"] = screenshot_refs

        # Extract step screenshots
        if "steps" in lightweight and isinstance(lightweight["steps"], list):
            for step in lightweight["steps"]:
                if isinstance(step, dict) and "screenshot" in step:
                    if isinstance(step["screenshot"], str) and len(step["screenshot"]) > 1000:
                        ref = self.store_screenshot(
                            step["screenshot"],
                            {"step": step.get("instruction", "unknown")}
                        )
                        step["screenshot"] = ref["artifact_id"]
                        artifact_refs.append(ref)

        # Add artifact references for frontend retrieval
        lightweight["_artifact_refs"] = artifact_refs

        # Create summary for Claude context
        lightweight["_summary"] = self._create_summary(result)

        return lightweight

    def _create_summary(self, result: dict[str, Any]) -> str:
        """Create a concise summary of a test result for Claude context."""
        parts = []

        if "success" in result:
            parts.append(f"Success: {result['success']}")

        if "steps" in result and isinstance(result["steps"], list):
            total = len(result["steps"])
            passed = sum(1 for s in result["steps"] if s.get("success"))
            parts.append(f"Steps: {passed}/{total} passed")

        if "error" in result:
            # Truncate long errors
            error = str(result["error"])[:200]
            parts.append(f"Error: {error}")

        if "message" in result:
            parts.append(f"Message: {result['message'][:100]}")

        return " | ".join(parts) if parts else "No summary available"

    def get(self, artifact_id: str) -> Artifact | None:
        """Retrieve a full artifact by ID."""
        if self.backend == "memory":
            return self._memory_cache.get(artifact_id)
        elif self.backend == "supabase":
            return self._get_supabase(artifact_id)
        elif self.backend in ("s3", "r2"):
            return self._get_r2(artifact_id)
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics (memory backend only)."""
        if self.backend == "memory":
            return self._memory_cache.stats()
        return {
            "backend": self.backend,
            "note": "Stats only available for memory backend",
        }

    def get_content(self, artifact_id: str) -> str | None:
        """Get just the content of an artifact."""
        artifact = self.get(artifact_id)
        return artifact.content if artifact else None

    def _store_supabase(self, artifact: Artifact):
        """Store artifact in Supabase Storage."""
        # TODO: Implement Supabase storage
        # from src.services.supabase_client import get_supabase_client
        # client = get_supabase_client()
        # client.storage.from_("artifacts").upload(...)
        logger.debug(
            "Supabase storage not yet implemented, falling back to memory cache",
            artifact_id=artifact.id,
        )
        self._memory_cache.put(artifact)

    def _store_r2(self, artifact: Artifact):
        """Store artifact in Cloudflare R2.

        R2 is preferred for production due to zero egress costs.
        Falls back to memory cache if R2 is not configured.
        """
        try:
            from src.services.cloudflare_storage import get_cloudflare_storage

            storage = get_cloudflare_storage()
            # Store in R2 bucket under artifacts/ prefix
            storage.put_object(
                key=f"artifacts/{artifact.id}",
                data=artifact.content.encode("utf-8"),
                content_type=self._get_content_type(artifact.type),
                metadata={
                    "artifact_type": artifact.type,
                    "created_at": artifact.created_at,
                    **artifact.metadata,
                },
            )
            logger.debug("Stored artifact in R2", artifact_id=artifact.id)
        except Exception as e:
            logger.warning(
                "Failed to store in R2, falling back to memory cache",
                artifact_id=artifact.id,
                error=str(e),
            )
            self._memory_cache.put(artifact)

    def _get_supabase(self, artifact_id: str) -> Artifact | None:
        """Retrieve artifact from Supabase Storage."""
        # TODO: Implement Supabase retrieval
        # Falls back to memory cache
        return self._memory_cache.get(artifact_id)

    def _get_r2(self, artifact_id: str) -> Artifact | None:
        """Retrieve artifact from Cloudflare R2."""
        try:
            from src.services.cloudflare_storage import get_cloudflare_storage

            storage = get_cloudflare_storage()
            result = storage.get_object(f"artifacts/{artifact_id}")
            if result:
                return Artifact(
                    id=artifact_id,
                    type=result.get("metadata", {}).get("artifact_type", "unknown"),
                    content=result["data"].decode("utf-8"),
                    metadata=result.get("metadata", {}),
                    created_at=result.get("metadata", {}).get(
                        "created_at", datetime.now(UTC).isoformat()
                    ),
                )
        except Exception as e:
            logger.warning(
                "Failed to retrieve from R2, checking memory cache",
                artifact_id=artifact_id,
                error=str(e),
            )
        # Fall back to memory cache
        return self._memory_cache.get(artifact_id)

    def _get_content_type(self, artifact_type: str) -> str:
        """Get MIME content type for artifact type."""
        content_types = {
            "screenshot": "image/png",
            "video": "video/webm",
            "html": "text/html",
            "json": "application/json",
        }
        return content_types.get(artifact_type, "application/octet-stream")

    def cleanup_old(self, max_age_hours: int = 24):
        """Clean up artifacts older than specified age."""
        # TODO: Implement cleanup for production
        pass


# Global artifact store instance
_artifact_store: ArtifactStore | None = None


def get_artifact_store() -> ArtifactStore:
    """Get the global artifact store instance.

    The backend is automatically selected based on the environment:
    - Production: R2 (Cloudflare) or Supabase storage
    - Development: Memory with LRU eviction

    Override with ARTIFACT_STORE_BACKEND environment variable:
    - "memory": In-memory with LRU eviction (testing only!)
    - "r2": Cloudflare R2 (recommended for production)
    - "supabase": Supabase Storage
    - "auto": Auto-select based on environment (default)
    """
    global _artifact_store
    if _artifact_store is None:
        backend = os.getenv("ARTIFACT_STORE_BACKEND", "auto")
        _artifact_store = ArtifactStore(backend=backend)
    return _artifact_store


def reset_artifact_store() -> None:
    """Reset the global artifact store (for testing)."""
    global _artifact_store
    _artifact_store = None
