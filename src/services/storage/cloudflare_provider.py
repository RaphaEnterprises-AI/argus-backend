"""Cloudflare R2 Storage Provider.

This module implements the StorageProvider interface using Cloudflare R2,
the default storage backend for cloud deployments.

RAP-297: Air-Gap Foundation - Extracted from cloudflare_storage.py

Configuration:
    STORAGE_PROVIDER=cloudflare
    CLOUDFLARE_ACCOUNT_ID=xxx
    CLOUDFLARE_API_TOKEN=xxx
    CLOUDFLARE_R2_BUCKET=argus-artifacts
    CLOUDFLARE_R2_ACCESS_KEY_ID=xxx  # For presigned URLs
    CLOUDFLARE_R2_SECRET_ACCESS_KEY=xxx
"""

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
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
class CloudflareR2Config(StorageConfig):
    """Cloudflare R2-specific configuration."""
    account_id: str = ""
    api_token: str = ""

    # S3-compatible credentials for presigned URLs
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""

    # Worker URL for public access (recommended over presigned)
    worker_url: str = ""

    # Media signing for Worker authentication
    media_signing_secret: str = ""
    media_url_expiry: int = 900  # 15 minutes

    @classmethod
    def from_env(cls) -> "CloudflareR2Config":
        """Load configuration from environment variables."""
        import os

        return cls(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
            api_token=os.getenv("CLOUDFLARE_API_TOKEN", ""),
            bucket=os.getenv("CLOUDFLARE_R2_BUCKET", "argus-artifacts"),
            r2_access_key_id=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID", ""),
            r2_secret_access_key=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", ""),
            worker_url=os.getenv("CLOUDFLARE_WORKER_URL", ""),
            media_signing_secret=os.getenv("CLOUDFLARE_MEDIA_SIGNING_SECRET", ""),
            presigned_url_expiry=int(os.getenv("CLOUDFLARE_R2_PRESIGNED_URL_EXPIRY", "3600")),
            media_url_expiry=int(os.getenv("CLOUDFLARE_MEDIA_URL_EXPIRY", "900")),
        )

    @classmethod
    def from_settings(cls) -> "CloudflareR2Config":
        """Load configuration from application settings."""
        from src.config import get_settings

        settings = get_settings()

        api_token = ""
        if settings.cloudflare_api_token:
            api_token = settings.cloudflare_api_token.get_secret_value()

        r2_secret = ""
        if settings.cloudflare_r2_secret_access_key:
            r2_secret = settings.cloudflare_r2_secret_access_key.get_secret_value()

        media_secret = ""
        if settings.cloudflare_media_signing_secret:
            media_secret = settings.cloudflare_media_signing_secret.get_secret_value()

        return cls(
            account_id=settings.cloudflare_account_id or "",
            api_token=api_token,
            bucket=settings.cloudflare_r2_bucket or "argus-artifacts",
            r2_access_key_id=settings.cloudflare_r2_access_key_id or "",
            r2_secret_access_key=r2_secret,
            worker_url=settings.cloudflare_worker_url or "",
            media_signing_secret=media_secret,
            presigned_url_expiry=settings.cloudflare_r2_presigned_url_expiry or 3600,
            media_url_expiry=settings.cloudflare_media_url_expiry or 900,
        )


class CloudflareR2Provider(StorageProvider):
    """Cloudflare R2 storage provider.

    Implements the StorageProvider interface using Cloudflare R2.
    Supports both presigned URLs and Worker-based public URLs.
    """

    backend = StorageBackend.CLOUDFLARE_R2

    def __init__(self, config: CloudflareR2Config):
        """Initialize R2 provider.

        Args:
            config: Cloudflare R2 configuration
        """
        self.config = config
        self.base_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{config.account_id}/r2/buckets/{config.bucket}"
        )
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json",
        }

    async def store_screenshot(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        """Store a screenshot in R2."""
        metadata = metadata or {}

        # Decode base64 data
        image_bytes = self._decode_base64(base64_data)

        # Generate artifact ID
        content_hash = self._compute_checksum(image_bytes)[:16]
        artifact_id = self._generate_artifact_id(ArtifactType.SCREENSHOT, content_hash)

        # Get storage key
        storage_key = self._get_storage_key(artifact_id, ArtifactType.SCREENSHOT, "png")

        # Upload to R2
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/objects/{storage_key}",
                headers={
                    **self.headers,
                    "Content-Type": "image/png",
                },
                content=image_bytes,
                timeout=30.0,
            )

            if response.status_code not in [200, 201]:
                logger.error(
                    "Failed to store screenshot in R2",
                    status=response.status_code,
                    response=response.text[:200],
                )
                raise Exception(f"R2 upload failed: {response.status_code}")

        logger.info(
            "Stored screenshot in R2",
            artifact_id=artifact_id,
            size_kb=len(image_bytes) // 1024,
        )

        # Generate URL - prefer Worker URL for public access
        url = self._get_worker_url(artifact_id)
        presigned_url = self._get_presigned_url(artifact_id)

        # Save metadata to Supabase
        await self._save_metadata_to_db(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.SCREENSHOT,
            storage_key=storage_key,
            url=url or presigned_url or "",
            file_size=len(image_bytes),
            content_type="image/png",
            metadata=metadata,
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.SCREENSHOT,
            storage_backend=StorageBackend.CLOUDFLARE_R2,
            storage_key=storage_key,
            url=url or presigned_url or "",
            presigned_url=presigned_url,
            url_expiry_seconds=None if url else self.config.presigned_url_expiry,
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
        """Store a video in R2."""
        metadata = metadata or {}

        video_bytes = self._decode_base64(base64_data)
        content_hash = self._compute_checksum(video_bytes)[:16]
        artifact_id = self._generate_artifact_id(ArtifactType.VIDEO, content_hash)
        storage_key = self._get_storage_key(artifact_id, ArtifactType.VIDEO, "webm")

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/objects/{storage_key}",
                headers={
                    **self.headers,
                    "Content-Type": "video/webm",
                },
                content=video_bytes,
                timeout=120.0,  # Longer timeout for videos
            )

            if response.status_code not in [200, 201]:
                raise Exception(f"R2 upload failed: {response.status_code}")

        logger.info(
            "Stored video in R2",
            artifact_id=artifact_id,
            size_mb=len(video_bytes) // (1024 * 1024),
        )

        url = self._get_worker_url(artifact_id)
        presigned_url = self._get_presigned_url(artifact_id)

        await self._save_metadata_to_db(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.VIDEO,
            storage_key=storage_key,
            url=url or presigned_url or "",
            file_size=len(video_bytes),
            content_type="video/webm",
            metadata=metadata,
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=ArtifactType.VIDEO,
            storage_backend=StorageBackend.CLOUDFLARE_R2,
            storage_key=storage_key,
            url=url or presigned_url or "",
            presigned_url=presigned_url,
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
        """Store a generic artifact in R2."""
        metadata = metadata or {}

        content_hash = self._compute_checksum(data)[:16]
        artifact_id = self._generate_artifact_id(artifact_type, content_hash)

        ext_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "video/webm": "webm",
            "video/mp4": "mp4",
            "application/json": "json",
            "text/plain": "txt",
        }
        extension = ext_map.get(content_type, "bin")
        storage_key = self._get_storage_key(artifact_id, artifact_type, extension)

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/objects/{storage_key}",
                headers={
                    **self.headers,
                    "Content-Type": content_type,
                },
                content=data,
                timeout=60.0,
            )

            if response.status_code not in [200, 201]:
                raise Exception(f"R2 upload failed: {response.status_code}")

        url = self._get_presigned_url(artifact_id)

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            storage_backend=StorageBackend.CLOUDFLARE_R2,
            storage_key=storage_key,
            url=url or "",
            content_type=content_type,
            file_size_bytes=len(data),
            checksum=self._compute_checksum(data),
            metadata=metadata,
        )

    async def get_artifact(self, artifact_id: str) -> bytes | None:
        """Retrieve an artifact from R2."""
        artifact_type = self._infer_type_from_id(artifact_id)
        ext = "png" if artifact_type == ArtifactType.SCREENSHOT else "webm"
        storage_key = self._get_storage_key(artifact_id, artifact_type, ext)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/objects/{storage_key}",
                    headers=self.headers,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    return response.content
                return None
        except Exception as e:
            logger.error("R2 retrieval error", artifact_id=artifact_id, error=str(e))
            return None

    async def get_artifact_url(
        self,
        artifact_id: str,
        expiry_seconds: int | None = None,
    ) -> str | None:
        """Get URL for an artifact."""
        # Prefer Worker URL
        worker_url = self._get_worker_url(artifact_id)
        if worker_url:
            return worker_url

        # Fall back to presigned URL
        return self._get_presigned_url(artifact_id, expiry_seconds)

    async def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact from R2."""
        artifact_type = self._infer_type_from_id(artifact_id)
        ext = "png" if artifact_type == ArtifactType.SCREENSHOT else "webm"
        storage_key = self._get_storage_key(artifact_id, artifact_type, ext)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/objects/{storage_key}",
                    headers=self.headers,
                    timeout=30.0,
                )
                return response.status_code in [200, 204]
        except Exception as e:
            logger.error("R2 deletion error", artifact_id=artifact_id, error=str(e))
            return False

    async def list_artifacts(
        self,
        prefix: str | None = None,
        artifact_type: ArtifactType | None = None,
        limit: int = 100,
    ) -> list[ArtifactRef]:
        """List artifacts in R2."""
        # R2 listing via API is complex; for now return empty
        # In production, query Supabase artifacts table instead
        logger.warning("R2 list_artifacts not fully implemented, use Supabase query")
        return []

    async def health_check(self) -> dict[str, Any]:
        """Check R2 connection health."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.cloudflare.com/client/v4/accounts/{self.config.account_id}/r2/buckets",
                    headers=self.headers,
                    timeout=10.0,
                )
                healthy = response.status_code == 200
                return {
                    "healthy": healthy,
                    "backend": "cloudflare_r2",
                    "bucket": self.config.bucket,
                    "status_code": response.status_code,
                }
        except Exception as e:
            return {
                "healthy": False,
                "backend": "cloudflare_r2",
                "error": str(e),
            }

    def _get_worker_url(self, artifact_id: str) -> str | None:
        """Get Worker URL for public access."""
        if not self.config.worker_url:
            return None

        artifact_type = self._infer_type_from_id(artifact_id)
        type_path = f"{artifact_type.value}s"  # screenshots, videos

        return f"{self.config.worker_url}/{type_path}/{artifact_id}"

    def _get_presigned_url(
        self,
        artifact_id: str,
        expiry_seconds: int | None = None,
    ) -> str | None:
        """Generate presigned URL for R2 access."""
        if not self.config.r2_access_key_id or not self.config.r2_secret_access_key:
            return None

        artifact_type = self._infer_type_from_id(artifact_id)
        ext = "png" if artifact_type == ArtifactType.SCREENSHOT else "webm"
        storage_key = self._get_storage_key(artifact_id, artifact_type, ext)

        # Implement AWS Signature V4 for R2
        # This is a simplified version; for production use boto3 or similar
        expiry = expiry_seconds or self.config.presigned_url_expiry

        # R2 endpoint format
        endpoint = f"https://{self.config.account_id}.r2.cloudflarestorage.com"
        url = f"{endpoint}/{self.config.bucket}/{storage_key}"

        # For now, return unsigned URL (requires bucket to be public or use signed URLs)
        logger.debug("Presigned URL generation simplified", artifact_id=artifact_id)
        return url

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
        """Save artifact metadata to Supabase."""
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
                        "storage_backend": "r2",
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
            logger.warning("Failed to save artifact metadata", error=str(e))
