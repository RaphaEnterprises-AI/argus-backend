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
"""

import uuid
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
import hashlib
import structlog

logger = structlog.get_logger()


@dataclass
class Artifact:
    """An artifact stored separately from LangGraph state."""
    id: str
    type: str  # screenshot, video, html, json
    content: str  # base64 or URL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_reference(self) -> Dict[str, Any]:
        """Return a lightweight reference for storing in LangGraph state."""
        return {
            "artifact_id": self.id,
            "type": self.type,
            "created_at": self.created_at,
            **self.metadata
        }


class ArtifactStore:
    """Store and retrieve artifacts separately from LangGraph state.

    This keeps the conversation state lightweight while preserving
    full artifact data for frontend display and later retrieval.

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

    def __init__(self, backend: str = "memory"):
        """Initialize artifact store.

        Args:
            backend: Storage backend - "memory", "supabase", or "s3"
        """
        self.backend = backend
        self._memory_store: Dict[str, Artifact] = {}

    def store(
        self,
        content: str,
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            self._memory_store[artifact_id] = artifact
        elif self.backend == "supabase":
            self._store_supabase(artifact)
        elif self.backend == "s3":
            self._store_s3(artifact)

        logger.info(
            "Stored artifact",
            artifact_id=artifact_id,
            type=artifact_type,
            size_kb=len(content) // 1024
        )

        return artifact.to_reference()

    def store_screenshot(
        self,
        base64_data: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method for storing screenshots."""
        return self.store(base64_data, "screenshot", metadata)

    def store_test_result(
        self,
        result: Dict[str, Any],
        extract_artifacts: bool = True
    ) -> Dict[str, Any]:
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
        artifact_refs: List[Dict[str, Any]] = []

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

    def _create_summary(self, result: Dict[str, Any]) -> str:
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

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve a full artifact by ID."""
        if self.backend == "memory":
            return self._memory_store.get(artifact_id)
        elif self.backend == "supabase":
            return self._get_supabase(artifact_id)
        elif self.backend == "s3":
            return self._get_s3(artifact_id)
        return None

    def get_content(self, artifact_id: str) -> Optional[str]:
        """Get just the content of an artifact."""
        artifact = self.get(artifact_id)
        return artifact.content if artifact else None

    def _store_supabase(self, artifact: Artifact):
        """Store artifact in Supabase Storage."""
        # TODO: Implement Supabase storage
        # from src.services.supabase_client import get_supabase_client
        # client = get_supabase_client()
        # client.storage.from_("artifacts").upload(...)
        self._memory_store[artifact.id] = artifact

    def _store_s3(self, artifact: Artifact):
        """Store artifact in S3."""
        # TODO: Implement S3 storage
        self._memory_store[artifact.id] = artifact

    def _get_supabase(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact from Supabase Storage."""
        return self._memory_store.get(artifact_id)

    def _get_s3(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact from S3."""
        return self._memory_store.get(artifact_id)

    def cleanup_old(self, max_age_hours: int = 24):
        """Clean up artifacts older than specified age."""
        # TODO: Implement cleanup for production
        pass


# Global artifact store instance
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get the global artifact store instance."""
    global _artifact_store
    if _artifact_store is None:
        # TODO: Configure backend based on settings
        _artifact_store = ArtifactStore(backend="memory")
    return _artifact_store
