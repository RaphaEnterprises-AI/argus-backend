"""Pluggable Storage Backend for Argus.

This package provides an abstract storage interface with multiple
backend implementations for cloud and self-hosted deployments.

RAP-297: Air-Gap Foundation - MinIO Storage Adapter

Supported Backends:
- Cloudflare R2 (default for cloud deployments)
- MinIO (for self-hosted / air-gap deployments)
- S3-compatible (AWS S3, DigitalOcean Spaces, etc.)

Usage:
    from src.services.storage import get_storage_provider

    # Get the configured storage provider (singleton)
    provider = get_storage_provider()

    # Store a screenshot
    ref = await provider.store_screenshot(base64_data, {"test_id": "123"})
    print(f"Stored at: {ref.url}")

    # Retrieve
    data = await provider.get_artifact(ref.artifact_id)

Configuration:
    # Cloud deployment (default)
    STORAGE_PROVIDER=cloudflare

    # Self-hosted deployment
    STORAGE_PROVIDER=minio
    MINIO_ENDPOINT=minio.local:9000
    MINIO_ACCESS_KEY=xxx
    MINIO_SECRET_KEY=xxx
    MINIO_BUCKET=argus-artifacts
"""

from src.services.storage.base import (
    ArtifactRef,
    ArtifactType,
    StorageBackend,
    StorageConfig,
    StorageProvider,
)
from src.services.storage.factory import (
    create_storage_provider,
    ensure_storage_ready,
    get_storage_backend,
    get_storage_provider,
    reset_storage_provider,
)

__all__ = [
    # Base classes
    "StorageProvider",
    "StorageConfig",
    "StorageBackend",
    "ArtifactType",
    "ArtifactRef",
    # Factory functions
    "get_storage_provider",
    "create_storage_provider",
    "get_storage_backend",
    "reset_storage_provider",
    "ensure_storage_ready",
]
