"""Storage Provider Factory.

This module provides a factory function to create the appropriate
storage provider based on configuration.

RAP-297: Air-Gap Foundation - MinIO Storage Adapter

Configuration:
    STORAGE_PROVIDER=cloudflare  # Default for cloud
    STORAGE_PROVIDER=minio       # For air-gap deployments
    STORAGE_PROVIDER=s3          # For AWS S3

Usage:
    from src.services.storage import get_storage_provider

    # Get configured provider (singleton)
    provider = get_storage_provider()

    # Store a screenshot
    ref = await provider.store_screenshot(base64_data, metadata)
"""

import os
from typing import TYPE_CHECKING

import structlog

from src.services.storage.base import StorageBackend, StorageProvider

if TYPE_CHECKING:
    from src.services.storage.minio_provider import MinIOStorageProvider

logger = structlog.get_logger()

# Singleton instance
_storage_provider: StorageProvider | None = None


def get_storage_backend() -> StorageBackend:
    """Get the configured storage backend from settings.

    Configuration: storage_provider setting or STORAGE_PROVIDER env var
    Values: cloudflare, minio, s3, local

    Returns:
        StorageBackend enum value
    """
    from src.config import get_settings

    settings = get_settings()
    provider_str = getattr(settings, "storage_provider", "cloudflare").lower()

    backend_map = {
        "cloudflare": StorageBackend.CLOUDFLARE_R2,
        "r2": StorageBackend.CLOUDFLARE_R2,
        "minio": StorageBackend.MINIO,
        "s3": StorageBackend.S3,
        "local": StorageBackend.LOCAL,
    }

    return backend_map.get(provider_str, StorageBackend.CLOUDFLARE_R2)


def create_storage_provider(backend: StorageBackend | None = None) -> StorageProvider:
    """Create a storage provider instance.

    Args:
        backend: Optional backend override. If not provided, uses STORAGE_PROVIDER env var.

    Returns:
        Configured StorageProvider instance
    """
    if backend is None:
        backend = get_storage_backend()

    logger.info("Creating storage provider", backend=backend.value)

    if backend == StorageBackend.MINIO or backend == StorageBackend.S3:
        from src.services.storage.minio_provider import MinIOConfig, MinIOStorageProvider

        config = MinIOConfig.from_settings()
        return MinIOStorageProvider(config)

    elif backend == StorageBackend.CLOUDFLARE_R2:
        from src.services.storage.cloudflare_provider import (
            CloudflareR2Config,
            CloudflareR2Provider,
        )

        config = CloudflareR2Config.from_settings()
        return CloudflareR2Provider(config)

    elif backend == StorageBackend.LOCAL:
        from src.services.storage.local_provider import LocalStorageProvider

        return LocalStorageProvider()

    else:
        raise ValueError(f"Unsupported storage backend: {backend}")


def get_storage_provider() -> StorageProvider:
    """Get the singleton storage provider instance.

    This is the recommended way to get a storage provider.
    It ensures only one instance is created and reused.

    Returns:
        Configured StorageProvider instance
    """
    global _storage_provider

    if _storage_provider is None:
        _storage_provider = create_storage_provider()

    return _storage_provider


def reset_storage_provider() -> None:
    """Reset the singleton storage provider.

    Useful for testing or when configuration changes.
    """
    global _storage_provider
    _storage_provider = None


async def ensure_storage_ready() -> bool:
    """Ensure storage is configured and accessible.

    For MinIO: Creates bucket if it doesn't exist.
    For Cloudflare: Validates credentials.

    Returns:
        True if storage is ready
    """
    provider = get_storage_provider()

    # For MinIO, ensure bucket exists
    if provider.backend == StorageBackend.MINIO:
        from src.services.storage.minio_provider import MinIOStorageProvider

        if isinstance(provider, MinIOStorageProvider):
            return await provider.ensure_bucket()

    # For other providers, do health check
    health = await provider.health_check()
    return health.get("healthy", False)
