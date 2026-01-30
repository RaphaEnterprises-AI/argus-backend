"""Abstract Storage Provider Interface.

This module defines the abstract base class for all storage providers,
enabling pluggable storage backends for air-gap deployments.

Supported backends:
- Cloudflare R2 (cloud deployment)
- MinIO (self-hosted / air-gap)
- S3-compatible (AWS, DigitalOcean Spaces, etc.)

RAP-297: Air-Gap Foundation - MinIO Storage Adapter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class StorageBackend(str, Enum):
    """Supported storage backends."""
    CLOUDFLARE_R2 = "cloudflare"
    MINIO = "minio"
    S3 = "s3"
    LOCAL = "local"  # For development/testing


class ArtifactType(str, Enum):
    """Types of artifacts stored."""
    SCREENSHOT = "screenshot"
    VIDEO = "video"
    LOG = "log"
    REPORT = "report"
    TEST_RESULT = "test_result"
    OTHER = "other"


@dataclass
class ArtifactRef:
    """Reference to a stored artifact.

    This is the unified response format for all storage operations,
    regardless of the underlying storage backend.
    """
    artifact_id: str
    artifact_type: ArtifactType
    storage_backend: StorageBackend
    storage_key: str  # Full path/key in storage
    url: str  # Public or presigned URL for access
    presigned_url: str | None = None  # Separate presigned URL if different
    url_expiry_seconds: int | None = None
    content_type: str = "application/octet-stream"
    file_size_bytes: int | None = None
    checksum: str | None = None  # SHA256 or MD5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "artifact_id": self.artifact_id,
            "type": self.artifact_type.value,
            "storage": self.storage_backend.value,
            "key": self.storage_key,
            "url": self.url,
            "presigned_url": self.presigned_url,
            "url_expiry_seconds": self.url_expiry_seconds,
            "content_type": self.content_type,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class StorageConfig:
    """Base configuration for storage providers."""
    bucket: str = "argus-artifacts"
    prefix: str = ""  # Optional path prefix for all objects
    public_url_base: str | None = None  # Base URL for public access
    presigned_url_expiry: int = 3600  # Default 1 hour

    # Connection settings
    max_retries: int = 3
    timeout_seconds: int = 30


class StorageProvider(ABC):
    """Abstract base class for storage providers.

    All storage backends must implement this interface to ensure
    consistent behavior across cloud and self-hosted deployments.

    Usage:
        provider = get_storage_provider()  # Returns configured provider

        # Store a screenshot
        ref = await provider.store_screenshot(base64_data, metadata)
        print(f"Stored at: {ref.url}")

        # Retrieve
        data = await provider.get_artifact(ref.artifact_id)
    """

    backend: StorageBackend
    config: StorageConfig

    @abstractmethod
    async def store_screenshot(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a screenshot image.

        Args:
            base64_data: Base64 encoded image data (with or without data URI prefix)
            metadata: Optional metadata (test_id, step, organization_id, etc.)

        Returns:
            ArtifactRef with storage location and access URL
        """
        pass

    @abstractmethod
    async def store_video(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a video recording.

        Args:
            base64_data: Base64 encoded video data
            metadata: Optional metadata

        Returns:
            ArtifactRef with storage location and access URL
        """
        pass

    @abstractmethod
    async def store_artifact(
        self,
        data: bytes,
        artifact_type: ArtifactType,
        content_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a generic artifact.

        Args:
            data: Raw bytes to store
            artifact_type: Type of artifact
            content_type: MIME type
            metadata: Optional metadata

        Returns:
            ArtifactRef with storage location and access URL
        """
        pass

    @abstractmethod
    async def get_artifact(self, artifact_id: str) -> bytes | None:
        """Retrieve an artifact by ID.

        Args:
            artifact_id: The artifact ID (e.g., screenshot_xxx_yyy)

        Returns:
            Raw bytes of the artifact, or None if not found
        """
        pass

    @abstractmethod
    async def get_artifact_url(
        self,
        artifact_id: str,
        expiry_seconds: int | None = None,
    ) -> str | None:
        """Get a URL for accessing an artifact.

        For public storage, returns the public URL.
        For private storage, returns a presigned URL.

        Args:
            artifact_id: The artifact ID
            expiry_seconds: URL expiry time (for presigned URLs)

        Returns:
            URL string, or None if artifact not found
        """
        pass

    @abstractmethod
    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: The artifact ID

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_artifacts(
        self,
        prefix: str | None = None,
        artifact_type: ArtifactType | None = None,
        limit: int = 100,
    ) -> list[ArtifactRef]:
        """List artifacts in storage.

        Args:
            prefix: Optional path prefix to filter
            artifact_type: Optional type filter
            limit: Maximum number of results

        Returns:
            List of ArtifactRef objects
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check storage backend health.

        Returns:
            Health status dict with 'healthy' bool and details
        """
        pass

    # Helper methods (non-abstract, can be overridden)

    def _generate_artifact_id(
        self,
        artifact_type: ArtifactType,
        content_hash: str | None = None,
    ) -> str:
        """Generate a unique artifact ID.

        Format: {type}_{hash}_{timestamp}
        """
        import hashlib
        import uuid

        hash_part = content_hash[:16] if content_hash else uuid.uuid4().hex[:16]
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"{artifact_type.value}_{hash_part}_{timestamp}"

    def _get_storage_key(
        self,
        artifact_id: str,
        artifact_type: ArtifactType,
        extension: str = "",
    ) -> str:
        """Get the full storage key/path for an artifact.

        Format: {prefix}/{type}s/{artifact_id}.{ext}
        """
        type_folder = f"{artifact_type.value}s"  # screenshots, videos, etc.
        key = f"{type_folder}/{artifact_id}"
        if extension:
            key = f"{key}.{extension.lstrip('.')}"
        if self.config.prefix:
            key = f"{self.config.prefix.rstrip('/')}/{key}"
        return key

    def _decode_base64(self, base64_data: str) -> bytes:
        """Decode base64 data, handling data URI prefix."""
        import base64

        # Handle data URI format: data:image/png;base64,xxxxx
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]

        return base64.b64decode(base64_data)

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        import hashlib
        return hashlib.sha256(data).hexdigest()
