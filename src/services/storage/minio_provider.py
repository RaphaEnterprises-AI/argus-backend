"""MinIO Storage Provider for Air-Gap Deployments.

This module implements the StorageProvider interface using MinIO,
enabling fully self-hosted, air-gapped deployments without external
cloud dependencies.

MinIO is S3-compatible, so this provider also works with:
- AWS S3
- DigitalOcean Spaces
- Any S3-compatible storage

RAP-297: Air-Gap Foundation - MinIO Storage Adapter

Configuration:
    STORAGE_PROVIDER=minio
    MINIO_ENDPOINT=minio.local:9000
    MINIO_ACCESS_KEY=minioadmin
    MINIO_SECRET_KEY=minioadmin
    MINIO_BUCKET=argus-artifacts
    MINIO_SECURE=false  # Use HTTPS
    MINIO_REGION=us-east-1
"""

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import structlog

from src.services.storage.base import (
    ArtifactRef,
    ArtifactType,
    StorageBackend,
    StorageConfig,
    StorageProvider,
)

logger = structlog.get_logger()


@dataclass
class MinIOConfig(StorageConfig):
    """MinIO-specific configuration."""
    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = False  # Use HTTPS
    region: str = "us-east-1"

    # Optional: Custom public URL for serving artifacts
    # If not set, presigned URLs are used
    public_url_base: str | None = None

    @classmethod
    def from_env(cls) -> "MinIOConfig":
        """Load configuration from environment variables."""
        import os

        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", ""),
            secret_key=os.getenv("MINIO_SECRET_KEY", ""),
            bucket=os.getenv("MINIO_BUCKET", "argus-artifacts"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            region=os.getenv("MINIO_REGION", "us-east-1"),
            public_url_base=os.getenv("MINIO_PUBLIC_URL_BASE"),
            prefix=os.getenv("MINIO_PREFIX", ""),
            presigned_url_expiry=int(os.getenv("MINIO_PRESIGNED_URL_EXPIRY", "3600")),
        )

    @classmethod
    def from_settings(cls) -> "MinIOConfig":
        """Load configuration from application settings."""
        from src.config import get_settings

        settings = get_settings()

        secret_key = ""
        if hasattr(settings, "minio_secret_key") and settings.minio_secret_key:
            secret_key = settings.minio_secret_key.get_secret_value()

        return cls(
            endpoint=getattr(settings, "minio_endpoint", "localhost:9000"),
            access_key=getattr(settings, "minio_access_key", ""),
            secret_key=secret_key,
            bucket=getattr(settings, "minio_bucket", "argus-artifacts"),
            secure=getattr(settings, "minio_secure", False),
            region=getattr(settings, "minio_region", "us-east-1"),
            public_url_base=getattr(settings, "minio_public_url_base", None),
            prefix=getattr(settings, "minio_prefix", ""),
            presigned_url_expiry=getattr(settings, "minio_presigned_url_expiry", 3600),
        )


class MinIOStorageProvider(StorageProvider):
    """MinIO/S3-compatible storage provider.

    Implements the StorageProvider interface for MinIO and S3-compatible
    storage backends. Suitable for air-gapped and self-hosted deployments.

    Usage:
        config = MinIOConfig.from_env()
        provider = MinIOStorageProvider(config)

        # Ensure bucket exists
        await provider.ensure_bucket()

        # Store screenshot
        ref = await provider.store_screenshot(base64_data, {"test_id": "123"})
        print(f"URL: {ref.url}")
    """

    backend = StorageBackend.MINIO

    def __init__(self, config: MinIOConfig):
        """Initialize MinIO provider.

        Args:
            config: MinIO configuration
        """
        self.config = config
        self._client = None
        self._initialized = False

    def _get_client(self):
        """Get or create MinIO client (lazy initialization)."""
        if self._client is None:
            try:
                from minio import Minio

                self._client = Minio(
                    self.config.endpoint,
                    access_key=self.config.access_key,
                    secret_key=self.config.secret_key,
                    secure=self.config.secure,
                    region=self.config.region,
                )
                logger.info(
                    "MinIO client initialized",
                    endpoint=self.config.endpoint,
                    bucket=self.config.bucket,
                    secure=self.config.secure,
                )
            except ImportError:
                raise ImportError(
                    "minio package not installed. Install with: pip install minio"
                )
        return self._client

    async def ensure_bucket(self) -> bool:
        """Ensure the storage bucket exists, create if not.

        Returns:
            True if bucket exists or was created
        """
        client = self._get_client()

        def _ensure():
            if not client.bucket_exists(self.config.bucket):
                client.make_bucket(self.config.bucket, location=self.config.region)
                logger.info("Created MinIO bucket", bucket=self.config.bucket)
                return True
            return True

        return await asyncio.to_thread(_ensure)

    async def store_screenshot(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a screenshot image in MinIO."""
        metadata = metadata or {}

        # Decode base64 data
        image_bytes = self._decode_base64(base64_data)

        # Generate artifact ID based on content hash
        content_hash = self._compute_checksum(image_bytes)[:16]
        artifact_id = self._generate_artifact_id(ArtifactType.SCREENSHOT, content_hash)

        # Get storage key
        storage_key = self._get_storage_key(artifact_id, ArtifactType.SCREENSHOT, "png")

        # Upload to MinIO
        client = self._get_client()

        def _upload():
            from io import BytesIO

            client.put_object(
                bucket_name=self.config.bucket,
                object_name=storage_key,
                data=BytesIO(image_bytes),
                length=len(image_bytes),
                content_type="image/png",
                metadata={
                    "artifact-id": artifact_id,
                    "artifact-type": "screenshot",
                    **({"test-id": metadata.get("test_id")} if metadata.get("test_id") else {}),
                    **({"thread-id": metadata.get("thread_id")} if metadata.get("thread_id") else {}),
                },
            )

        await asyncio.to_thread(_upload)

        logger.info(
            "Stored screenshot in MinIO",
            artifact_id=artifact_id,
            size_kb=len(image_bytes) // 1024,
            bucket=self.config.bucket,
            key=storage_key,
        )

        # Generate URL
        url = await self.get_artifact_url(artifact_id)

        # Save metadata to Supabase if available
        await self._save_metadata_to_db(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.SCREENSHOT,
            storage_key=storage_key,
            url=url,
            file_size=len(image_bytes),
            content_type="image/png",
            metadata=metadata,
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.SCREENSHOT,
            storage_backend=StorageBackend.MINIO,
            storage_key=storage_key,
            url=url,
            presigned_url=url,  # MinIO uses presigned URLs by default
            url_expiry_seconds=self.config.presigned_url_expiry,
            content_type="image/png",
            file_size_bytes=len(image_bytes),
            checksum=self._compute_checksum(image_bytes),
            metadata=metadata,
        )

    async def store_video(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a video recording in MinIO."""
        metadata = metadata or {}

        # Decode base64 data
        video_bytes = self._decode_base64(base64_data)

        # Generate artifact ID
        content_hash = self._compute_checksum(video_bytes)[:16]
        artifact_id = self._generate_artifact_id(ArtifactType.VIDEO, content_hash)

        # Get storage key
        storage_key = self._get_storage_key(artifact_id, ArtifactType.VIDEO, "webm")

        # Upload to MinIO
        client = self._get_client()

        def _upload():
            from io import BytesIO

            client.put_object(
                bucket_name=self.config.bucket,
                object_name=storage_key,
                data=BytesIO(video_bytes),
                length=len(video_bytes),
                content_type="video/webm",
                metadata={
                    "artifact-id": artifact_id,
                    "artifact-type": "video",
                },
            )

        await asyncio.to_thread(_upload)

        logger.info(
            "Stored video in MinIO",
            artifact_id=artifact_id,
            size_mb=len(video_bytes) // (1024 * 1024),
            bucket=self.config.bucket,
        )

        url = await self.get_artifact_url(artifact_id)

        await self._save_metadata_to_db(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.VIDEO,
            storage_key=storage_key,
            url=url,
            file_size=len(video_bytes),
            content_type="video/webm",
            metadata=metadata,
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.VIDEO,
            storage_backend=StorageBackend.MINIO,
            storage_key=storage_key,
            url=url,
            content_type="video/webm",
            file_size_bytes=len(video_bytes),
            checksum=self._compute_checksum(video_bytes),
            metadata=metadata,
        )

    async def store_artifact(
        self,
        data: bytes,
        artifact_type: ArtifactType,
        content_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a generic artifact in MinIO."""
        metadata = metadata or {}

        # Generate artifact ID
        content_hash = self._compute_checksum(data)[:16]
        artifact_id = self._generate_artifact_id(artifact_type, content_hash)

        # Determine extension from content type
        ext_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "video/webm": "webm",
            "video/mp4": "mp4",
            "application/json": "json",
            "text/plain": "txt",
            "text/html": "html",
        }
        extension = ext_map.get(content_type, "bin")

        storage_key = self._get_storage_key(artifact_id, artifact_type, extension)

        # Upload
        client = self._get_client()

        def _upload():
            from io import BytesIO

            client.put_object(
                bucket_name=self.config.bucket,
                object_name=storage_key,
                data=BytesIO(data),
                length=len(data),
                content_type=content_type,
            )

        await asyncio.to_thread(_upload)

        url = await self.get_artifact_url(artifact_id)

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            storage_backend=StorageBackend.MINIO,
            storage_key=storage_key,
            url=url,
            content_type=content_type,
            file_size_bytes=len(data),
            checksum=self._compute_checksum(data),
            metadata=metadata,
        )

    async def get_artifact(self, artifact_id: str) -> bytes | None:
        """Retrieve an artifact by ID from MinIO."""
        # Determine storage key from artifact ID
        artifact_type = self._infer_type_from_id(artifact_id)
        ext = "png" if artifact_type == ArtifactType.SCREENSHOT else "webm"
        storage_key = self._get_storage_key(artifact_id, artifact_type, ext)

        client = self._get_client()

        def _download():
            try:
                response = client.get_object(self.config.bucket, storage_key)
                data = response.read()
                response.close()
                response.release_conn()
                return data
            except Exception as e:
                if "NoSuchKey" in str(e) or "not found" in str(e).lower():
                    return None
                raise

        try:
            return await asyncio.to_thread(_download)
        except Exception as e:
            logger.error("Failed to get artifact from MinIO", artifact_id=artifact_id, error=str(e))
            return None

    async def get_artifact_url(
        self,
        artifact_id: str,
        expiry_seconds: int | None = None,
    ) -> str | None:
        """Get a presigned URL for an artifact."""
        artifact_type = self._infer_type_from_id(artifact_id)
        ext = "png" if artifact_type == ArtifactType.SCREENSHOT else "webm"
        storage_key = self._get_storage_key(artifact_id, artifact_type, ext)

        # If public URL base is configured, use it
        if self.config.public_url_base:
            return f"{self.config.public_url_base.rstrip('/')}/{storage_key}"

        # Generate presigned URL
        expiry = expiry_seconds or self.config.presigned_url_expiry
        client = self._get_client()

        def _presign():
            return client.presigned_get_object(
                bucket_name=self.config.bucket,
                object_name=storage_key,
                expires=timedelta(seconds=expiry),
            )

        try:
            return await asyncio.to_thread(_presign)
        except Exception as e:
            logger.error("Failed to generate presigned URL", artifact_id=artifact_id, error=str(e))
            return None

    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from MinIO."""
        artifact_type = self._infer_type_from_id(artifact_id)
        ext = "png" if artifact_type == ArtifactType.SCREENSHOT else "webm"
        storage_key = self._get_storage_key(artifact_id, artifact_type, ext)

        client = self._get_client()

        def _delete():
            client.remove_object(self.config.bucket, storage_key)
            return True

        try:
            await asyncio.to_thread(_delete)
            logger.info("Deleted artifact from MinIO", artifact_id=artifact_id)
            return True
        except Exception as e:
            logger.error("Failed to delete artifact", artifact_id=artifact_id, error=str(e))
            return False

    async def list_artifacts(
        self,
        prefix: str | None = None,
        artifact_type: ArtifactType | None = None,
        limit: int = 100,
    ) -> list[ArtifactRef]:
        """List artifacts in MinIO."""
        client = self._get_client()

        # Build prefix for listing
        list_prefix = self.config.prefix or ""
        if artifact_type:
            type_folder = f"{artifact_type.value}s"
            list_prefix = f"{list_prefix}/{type_folder}" if list_prefix else type_folder
        if prefix:
            list_prefix = f"{list_prefix}/{prefix}" if list_prefix else prefix

        def _list():
            objects = client.list_objects(
                bucket_name=self.config.bucket,
                prefix=list_prefix,
                recursive=True,
            )
            results = []
            for obj in objects:
                if len(results) >= limit:
                    break
                # Parse artifact info from object name
                artifact_id = obj.object_name.split("/")[-1].rsplit(".", 1)[0]
                inferred_type = self._infer_type_from_id(artifact_id)
                results.append(
                    ArtifactRef(
                        artifact_id=artifact_id,
                        artifact_type=inferred_type,
                        storage_backend=StorageBackend.MINIO,
                        storage_key=obj.object_name,
                        url="",  # URL generated on demand
                        file_size_bytes=obj.size,
                        created_at=obj.last_modified.isoformat() if obj.last_modified else "",
                    )
                )
            return results

        return await asyncio.to_thread(_list)

    async def health_check(self) -> dict[str, Any]:
        """Check MinIO connection health."""
        client = self._get_client()

        def _check():
            bucket_exists = client.bucket_exists(self.config.bucket)
            return {
                "healthy": bucket_exists,
                "backend": "minio",
                "endpoint": self.config.endpoint,
                "bucket": self.config.bucket,
                "bucket_exists": bucket_exists,
            }

        try:
            return await asyncio.to_thread(_check)
        except Exception as e:
            return {
                "healthy": False,
                "backend": "minio",
                "endpoint": self.config.endpoint,
                "error": str(e),
            }

    def _infer_type_from_id(self, artifact_id: str) -> ArtifactType:
        """Infer artifact type from ID prefix."""
        if artifact_id.startswith("screenshot"):
            return ArtifactType.SCREENSHOT
        elif artifact_id.startswith("video"):
            return ArtifactType.VIDEO
        elif artifact_id.startswith("log"):
            return ArtifactType.LOG
        elif artifact_id.startswith("report"):
            return ArtifactType.REPORT
        else:
            return ArtifactType.OTHER

    async def _save_metadata_to_db(
        self,
        artifact_id: str,
        artifact_type: ArtifactType,
        storage_key: str,
        url: str,
        file_size: int,
        content_type: str,
        metadata: dict[str, Any],
    ) -> None:
        """Save artifact metadata to Supabase for querying."""
        try:
            from src.integrations.supabase import get_supabase

            supabase = await get_supabase()
            if supabase:
                await supabase.insert(
                    "artifacts",
                    {
                        "id": artifact_id,
                        "organization_id": metadata.get("organization_id"),
                        "project_id": metadata.get("project_id"),
                        "user_id": metadata.get("user_id", "anonymous"),
                        "type": artifact_type.value,
                        "storage_backend": "minio",
                        "storage_key": storage_key,
                        "storage_url": url,
                        "test_id": metadata.get("test_id"),
                        "thread_id": metadata.get("thread_id"),
                        "step_index": metadata.get("step"),
                        "action_description": metadata.get("action"),
                        "file_size_bytes": file_size,
                        "content_type": content_type,
                        "metadata": metadata,
                    },
                )
                logger.debug("Saved artifact metadata to Supabase", artifact_id=artifact_id)
        except Exception as e:
            # Don't fail the upload if metadata save fails
            logger.warning("Failed to save artifact metadata to database", error=str(e))
