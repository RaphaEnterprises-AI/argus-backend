"""Cloudflare Storage Services for Argus.

This module provides a unified interface to Cloudflare's storage ecosystem:
- R2: Object storage for screenshots, videos, artifacts (zero egress!)
- Vectorize: Vector database for failure patterns & self-healing memory
- D1: SQL database for test history & metadata
- KV: Fast key-value cache for sessions & selectors

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Cloudflare Storage                        │
    ├──────────────┬──────────────┬──────────────┬────────────────┤
    │      R2      │   Vectorize  │      D1      │       KV       │
    │  Artifacts   │   Patterns   │   History    │     Cache      │
    └──────────────┴──────────────┴──────────────┴────────────────┘

Vectorize Index Types (RAP-248):
    - FAILURE_PATTERNS: Self-healing memory for error patterns
    - CODE_EMBEDDINGS: Code snippets and function signatures
    - TEST_EMBEDDINGS: Test cases and assertions
    - DOCUMENTATION: API docs, guides, comments
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


# =============================================================================
# Vectorize Index Types (RAP-248)
# =============================================================================

class VectorizeIndexType(str, Enum):
    """Supported Vectorize index types for different embedding categories.

    Each index type maps to a separate Vectorize index for optimal search
    performance and isolation. Index names are derived from the base index
    name with a suffix.

    Example:
        Base index: argus-patterns
        - FAILURE_PATTERNS -> argus-patterns (default, no suffix)
        - CODE_EMBEDDINGS -> argus-patterns-code
        - TEST_EMBEDDINGS -> argus-patterns-tests
        - DOCUMENTATION -> argus-patterns-docs
    """
    FAILURE_PATTERNS = "failure_patterns"
    CODE_EMBEDDINGS = "code_embeddings"
    TEST_EMBEDDINGS = "test_embeddings"
    DOCUMENTATION = "documentation"


@dataclass
class VectorSearchResult:
    """Result from a Vectorize search operation."""
    id: str
    score: float
    index_type: VectorizeIndexType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "score": self.score,
            "index_type": self.index_type.value,
            "metadata": self.metadata,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CloudflareConfig:
    """Cloudflare API configuration."""
    account_id: str
    api_token: str
    r2_bucket: str = "argus-artifacts"
    vectorize_index: str = "argus-patterns"
    d1_database_id: str = ""
    kv_namespace_id: str = ""
    ai_gateway_id: str = "argus-gateway"
    # R2 S3-compatible credentials for presigned URLs
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_presigned_url_expiry: int = 3600  # 1 hour default
    # Worker URL for screenshot access
    worker_url: str = "https://argus-api.samuelvinay-kumar.workers.dev"
    # Media URL signing (HMAC-SHA256)
    # Must match MEDIA_SIGNING_SECRET in Cloudflare Worker
    media_signing_secret: str = ""
    media_url_expiry: int = 900  # 15 minutes default

    @classmethod
    def from_env(cls) -> "CloudflareConfig":
        """Load config from environment variables."""
        return cls(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID", ""),
            api_token=os.getenv("CLOUDFLARE_API_TOKEN", ""),
            r2_bucket=os.getenv("CLOUDFLARE_R2_BUCKET", "argus-artifacts"),
            vectorize_index=os.getenv("CLOUDFLARE_VECTORIZE_INDEX", "argus-patterns"),
            d1_database_id=os.getenv("CLOUDFLARE_D1_DATABASE_ID", ""),
            kv_namespace_id=os.getenv("CLOUDFLARE_KV_NAMESPACE_ID", ""),
            ai_gateway_id=os.getenv("CLOUDFLARE_AI_GATEWAY_ID", "argus-gateway"),
            r2_access_key_id=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID", ""),
            r2_secret_access_key=os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", ""),
            r2_presigned_url_expiry=int(os.getenv("CLOUDFLARE_R2_PRESIGNED_URL_EXPIRY", "3600")),
            worker_url=os.getenv("CLOUDFLARE_WORKER_URL", "https://argus-api.samuelvinay-kumar.workers.dev"),
            media_signing_secret=os.getenv("CLOUDFLARE_MEDIA_SIGNING_SECRET", ""),
            media_url_expiry=int(os.getenv("CLOUDFLARE_MEDIA_URL_EXPIRY", "900")),
        )

    @classmethod
    def from_settings(cls) -> "CloudflareConfig":
        """Load config from application settings (recommended)."""
        from src.config import get_settings
        settings = get_settings()

        api_token = ""
        if settings.cloudflare_api_token:
            api_token = settings.cloudflare_api_token.get_secret_value()

        r2_secret = ""
        if settings.cloudflare_r2_secret_access_key:
            r2_secret = settings.cloudflare_r2_secret_access_key.get_secret_value()

        media_signing_secret = ""
        if settings.cloudflare_media_signing_secret:
            media_signing_secret = settings.cloudflare_media_signing_secret.get_secret_value()

        return cls(
            account_id=settings.cloudflare_account_id or "",
            api_token=api_token,
            r2_bucket=settings.cloudflare_r2_bucket or "argus-artifacts",
            vectorize_index=settings.cloudflare_vectorize_index or "argus-patterns",
            d1_database_id=settings.cloudflare_d1_database_id or "",
            kv_namespace_id=settings.cloudflare_kv_namespace_id or "",
            ai_gateway_id=settings.cloudflare_gateway_id or "argus-gateway",
            r2_access_key_id=settings.cloudflare_r2_access_key_id or "",
            r2_secret_access_key=r2_secret,
            r2_presigned_url_expiry=settings.cloudflare_r2_presigned_url_expiry,
            worker_url=settings.cloudflare_worker_url,
            media_signing_secret=media_signing_secret,
            media_url_expiry=settings.cloudflare_media_url_expiry,
        )


# =============================================================================
# R2 Object Storage (Screenshots, Videos, Artifacts)
# =============================================================================

class R2Storage:
    """Cloudflare R2 object storage for test artifacts.

    Perfect for:
    - Screenshots (PNG, base64)
    - Video recordings
    - HTML snapshots
    - Large test results

    Benefits:
    - Zero egress fees (huge cost savings!)
    - S3-compatible API
    - Global distribution
    """

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/r2/buckets/{config.r2_bucket}"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }

    async def store_screenshot(
        self,
        base64_data: str,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Store a screenshot in R2 and save metadata to Supabase.

        Args:
            base64_data: Base64 encoded image data
            metadata: Optional metadata (step, test_id, etc.)

        Returns:
            Reference object with artifact_id and URL
        """
        # Generate content-based ID for deduplication
        content_hash = hashlib.sha256(base64_data[:2000].encode()).hexdigest()[:16]
        artifact_id = f"screenshot_{content_hash}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        # Store in R2
        key = f"screenshots/{artifact_id}.png"
        metadata = metadata or {}

        try:
            async with httpx.AsyncClient() as client:
                # Convert base64 to bytes for upload
                import base64
                image_bytes = base64.b64decode(base64_data.split(",")[-1] if "," in base64_data else base64_data)

                response = await client.put(
                    f"{self.base_url}/objects/{key}",
                    headers={
                        **self.headers,
                        "Content-Type": "image/png"
                    },
                    content=image_bytes,
                    timeout=30.0
                )

                if response.status_code in [200, 201]:
                    logger.info("Stored screenshot in R2", artifact_id=artifact_id, size_kb=len(image_bytes) // 1024)

                    # Generate URL - prefer Worker URL (public, no auth) over presigned URLs
                    # Worker URL format: {worker_url}/screenshots/{artifact_id}
                    worker_url = f"{self.config.worker_url}/screenshots/{artifact_id}"

                    # Fallback to presigned URL if Worker URL not configured
                    presigned_url = None
                    if not self.config.worker_url:
                        presigned_url = self.get_presigned_url(artifact_id)

                    artifact_ref = {
                        "artifact_id": artifact_id,
                        "type": "screenshot",
                        "storage": "r2",
                        "key": key,
                        "url": worker_url,  # Use Worker URL for public access
                        "presigned_url": presigned_url,
                        "url_expiry_seconds": None,  # Worker URLs don't expire
                        "metadata": metadata,
                        "created_at": datetime.now(UTC).isoformat()
                    }

                    # Also save metadata to Supabase for listing/querying
                    try:
                        from src.integrations.supabase import get_supabase
                        supabase = await get_supabase()
                        if supabase:
                            await supabase.insert("artifacts", {
                                "id": artifact_id,
                                "organization_id": metadata.get("organization_id"),
                                "project_id": metadata.get("project_id"),
                                "user_id": metadata.get("user_id", "anonymous"),
                                "type": "screenshot",
                                "storage_backend": "r2",
                                "storage_key": key,
                                "storage_url": artifact_ref["url"],
                                "test_id": metadata.get("test_id"),
                                "thread_id": metadata.get("thread_id"),
                                "step_index": metadata.get("step"),
                                "action_description": metadata.get("action"),
                                "file_size_bytes": len(image_bytes),
                                "content_type": "image/png",
                                "metadata": metadata,
                            })
                            logger.info("Saved artifact metadata to Supabase", artifact_id=artifact_id)
                    except Exception as db_error:
                        logger.warning("Failed to save artifact metadata to Supabase", error=str(db_error))

                    return artifact_ref
                else:
                    logger.error("Failed to store screenshot in R2", status=response.status_code)
                    raise Exception(f"R2 upload failed: {response.status_code}")

        except Exception as e:
            logger.exception("R2 storage error", error=str(e))
            # Fallback: return base64 reference (not ideal but works)
            return {
                "artifact_id": artifact_id,
                "type": "screenshot",
                "storage": "inline",
                "data": base64_data[:100] + "...[truncated]",
                "error": str(e)
            }

    async def get_screenshot(self, artifact_id: str) -> str | None:
        """Retrieve a screenshot from R2."""
        key = f"screenshots/{artifact_id}.png"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/objects/{key}",
                    headers=self.headers,
                    timeout=30.0
                )

                if response.status_code == 200:
                    import base64
                    return base64.b64encode(response.content).decode()
                return None
        except Exception as e:
            logger.exception("R2 retrieval error", error=str(e))
            return None

    def get_presigned_url(
        self,
        artifact_id: str,
        expiry_seconds: int | None = None
    ) -> str | None:
        """Generate a presigned URL for direct R2 access.

        Args:
            artifact_id: The artifact ID (e.g., screenshot_xxx_yyy)
            expiry_seconds: URL expiry time (default from config)

        Returns:
            Presigned URL string or None if R2 credentials not configured
        """
        # Check if S3-compatible credentials are configured
        if not self.config.r2_access_key_id or not self.config.r2_secret_access_key:
            logger.debug("R2 access keys not configured, presigned URLs unavailable")
            return None

        try:
            import boto3
            from botocore.config import Config

            # R2 S3-compatible endpoint
            endpoint_url = f"https://{self.config.account_id}.r2.cloudflarestorage.com"

            # Create S3 client for R2
            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.config.r2_access_key_id,
                aws_secret_access_key=self.config.r2_secret_access_key,
                config=Config(signature_version="s3v4"),
                region_name="auto",  # R2 uses 'auto' region
            )

            # Determine the object key
            key = f"screenshots/{artifact_id}.png"
            expiry = expiry_seconds or self.config.r2_presigned_url_expiry

            # Generate presigned URL
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.config.r2_bucket,
                    "Key": key,
                },
                ExpiresIn=expiry,
            )

            logger.info("Generated presigned URL", artifact_id=artifact_id, expiry_seconds=expiry)
            return url

        except ImportError:
            logger.warning("boto3 not installed, presigned URLs unavailable")
            return None
        except Exception as e:
            logger.exception("Failed to generate presigned URL", error=str(e))
            return None

    def generate_signed_url(
        self,
        artifact_id: str,
        artifact_type: str = "screenshot",
        expiry_seconds: int | None = None
    ) -> str | None:
        """Generate an HMAC-signed URL for authenticated media access.

        This method creates a short-lived signed URL that can be verified
        by the Cloudflare Worker. Unlike S3 presigned URLs, these are
        simpler and faster to generate.

        Args:
            artifact_id: The artifact ID (e.g., screenshot_xxx_yyy)
            artifact_type: Type of artifact ('screenshot' or 'video')
            expiry_seconds: URL expiry time (default from config, typically 15 min)

        Returns:
            Signed URL string or None if signing secret not configured
        """
        if not self.config.media_signing_secret:
            logger.debug("Media signing secret not configured, signed URLs unavailable")
            return None

        try:
            expiry = expiry_seconds or self.config.media_url_expiry
            exp_timestamp = int(time.time()) + expiry

            # Generate HMAC-SHA256 signature: HMAC(artifact_id:expiration, secret)
            message = f"{artifact_id}:{exp_timestamp}"
            signature = hmac.new(
                self.config.media_signing_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            # Build URL based on artifact type
            if artifact_type == "video":
                path = f"/videos/{artifact_id}"
            else:
                path = f"/screenshots/{artifact_id}"

            signed_url = f"{self.config.worker_url}{path}?sig={signature}&exp={exp_timestamp}"

            logger.info(
                "Generated signed media URL",
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                expiry_seconds=expiry
            )
            return signed_url

        except Exception as e:
            logger.exception("Failed to generate signed URL", error=str(e))
            return None

    # =========================================================================
    # Video Storage Methods (Following Official R2 Documentation)
    # https://developers.cloudflare.com/r2/api/s3/presigned-urls/
    # https://developers.cloudflare.com/r2/objects/upload-objects/
    # =========================================================================

    def generate_video_upload_url(
        self,
        artifact_id: str,
        content_type: str = "video/webm",
        expiry_seconds: int = 3600,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Generate a presigned PUT URL for direct video upload to R2.

        This follows the official R2 presigned URL pattern where external services
        (like BrowserPool) can upload directly to R2 without exposing credentials.

        Per Cloudflare docs:
        - Presigned URLs support GET, PUT, HEAD, DELETE operations
        - Max expiry: 7 days (604,800 seconds)
        - Must use signature_version='s3v4'

        Args:
            artifact_id: Unique identifier for the video (e.g., video_abc123_timestamp)
            content_type: MIME type (video/webm, video/mp4). Must match upload Content-Type header.
            expiry_seconds: URL validity (default 1 hour, max 604,800 = 7 days)
            metadata: Optional metadata to associate with the upload

        Returns:
            Dict with upload_url, artifact_id, storage_key, and expiry info
            None if R2 credentials not configured
        """
        if not self.config.r2_access_key_id or not self.config.r2_secret_access_key:
            logger.warning("R2 access keys not configured, cannot generate upload URL")
            return None

        try:
            import boto3
            from botocore.config import Config

            # R2 S3-compatible endpoint (per official docs)
            endpoint_url = f"https://{self.config.account_id}.r2.cloudflarestorage.com"

            # Create S3 client with proper R2 configuration
            # Per docs: signature_version='s3v4' is required
            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.config.r2_access_key_id,
                aws_secret_access_key=self.config.r2_secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
                region_name="auto",  # R2 uses 'auto' region
            )

            # Determine file extension from content type
            ext = "webm" if "webm" in content_type else "mp4"
            storage_key = f"videos/{artifact_id}.{ext}"

            # Enforce max expiry per R2 docs (7 days = 604,800 seconds)
            expiry = min(expiry_seconds, 604800)

            # Generate presigned PUT URL
            # Per docs: ContentType in Params restricts upload to matching Content-Type header
            upload_url = s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.config.r2_bucket,
                    "Key": storage_key,
                    "ContentType": content_type,
                },
                ExpiresIn=expiry,
            )

            logger.info(
                "Generated video upload presigned URL",
                artifact_id=artifact_id,
                storage_key=storage_key,
                content_type=content_type,
                expiry_seconds=expiry,
            )

            return {
                "upload_url": upload_url,
                "artifact_id": artifact_id,
                "storage_key": storage_key,
                "content_type": content_type,
                "bucket": self.config.r2_bucket,
                "expiry_seconds": expiry,
                "metadata": metadata or {},
            }

        except ImportError:
            logger.error("boto3 not installed, presigned URLs unavailable")
            return None
        except Exception as e:
            logger.exception("Failed to generate video upload URL", error=str(e))
            return None

    async def confirm_video_upload(
        self,
        artifact_id: str,
        storage_key: str,
        file_size_bytes: int | None = None,
        duration_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Confirm video upload and save metadata to Supabase.

        Call this after BrowserPool successfully uploads to the presigned URL.
        This saves the artifact reference to Supabase for querying/serving.

        Args:
            artifact_id: The artifact ID used when generating upload URL
            storage_key: The R2 object key (e.g., videos/video_abc123.webm)
            file_size_bytes: Size of uploaded video (optional but recommended)
            duration_seconds: Video duration (optional)
            metadata: Additional metadata (session_id, project_id, etc.)

        Returns:
            Artifact reference dict with URLs
        """
        metadata = metadata or {}

        # Determine content type from storage key
        content_type = "video/webm" if storage_key.endswith(".webm") else "video/mp4"

        # Generate serving URL via Worker (preferred - no expiry)
        worker_url = f"{self.config.worker_url}/videos/{artifact_id}"

        # Also generate presigned GET URL as fallback
        presigned_url = self._generate_video_presigned_get_url(storage_key)

        artifact_ref = {
            "artifact_id": artifact_id,
            "type": "video",
            "storage": "r2",
            "storage_key": storage_key,
            "url": worker_url,
            "presigned_url": presigned_url,
            "content_type": content_type,
            "file_size_bytes": file_size_bytes,
            "duration_seconds": duration_seconds,
            "metadata": metadata,
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Save metadata to Supabase artifacts table
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
                        "type": "video",
                        "storage_backend": "r2",
                        "storage_key": storage_key,
                        "storage_url": worker_url,
                        "test_id": metadata.get("test_id"),
                        "test_run_id": metadata.get("test_run_id"),
                        "thread_id": metadata.get("thread_id"),
                        "step_index": metadata.get("step_index"),
                        "action_description": metadata.get("description", "Session recording"),
                        "file_size_bytes": file_size_bytes,
                        "content_type": content_type,
                        "metadata": {
                            **metadata,
                            "duration_seconds": duration_seconds,
                        },
                    },
                )
                logger.info(
                    "Saved video artifact metadata to Supabase",
                    artifact_id=artifact_id,
                    file_size_kb=file_size_bytes // 1024 if file_size_bytes else None,
                )
        except Exception as db_error:
            logger.warning(
                "Failed to save video artifact metadata to Supabase",
                artifact_id=artifact_id,
                error=str(db_error),
            )

        return artifact_ref

    def _generate_video_presigned_get_url(
        self,
        storage_key: str,
        expiry_seconds: int | None = None,
    ) -> str | None:
        """Generate presigned GET URL for video download."""
        if not self.config.r2_access_key_id or not self.config.r2_secret_access_key:
            return None

        try:
            import boto3
            from botocore.config import Config

            endpoint_url = f"https://{self.config.account_id}.r2.cloudflarestorage.com"

            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.config.r2_access_key_id,
                aws_secret_access_key=self.config.r2_secret_access_key,
                config=Config(signature_version="s3v4"),
                region_name="auto",
            )

            expiry = expiry_seconds or self.config.r2_presigned_url_expiry

            return s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.config.r2_bucket,
                    "Key": storage_key,
                },
                ExpiresIn=expiry,
            )
        except Exception as e:
            logger.warning("Failed to generate video presigned GET URL", error=str(e))
            return None

    async def store_video_from_bytes(
        self,
        video_bytes: bytes,
        content_type: str = "video/webm",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store video directly from bytes (for smaller videos < 5GB).

        Use this when you have the video data in memory.
        For larger videos or external uploads, use generate_video_upload_url().

        Per R2 docs: Single-part upload limit is 5GB (4.995 GiB).

        Args:
            video_bytes: Raw video data
            content_type: MIME type (video/webm, video/mp4)
            metadata: Optional metadata

        Returns:
            Artifact reference dict with URLs
        """
        metadata = metadata or {}

        # Generate content-based ID for deduplication
        content_hash = hashlib.sha256(video_bytes[:10000]).hexdigest()[:16]
        artifact_id = f"video_{content_hash}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        # Determine extension
        ext = "webm" if "webm" in content_type else "mp4"
        storage_key = f"videos/{artifact_id}.{ext}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.base_url}/objects/{storage_key}",
                    headers={
                        **self.headers,
                        "Content-Type": content_type,
                    },
                    content=video_bytes,
                    timeout=120.0,  # Longer timeout for video uploads
                )

                if response.status_code in [200, 201]:
                    logger.info(
                        "Stored video in R2",
                        artifact_id=artifact_id,
                        size_mb=len(video_bytes) / (1024 * 1024),
                    )

                    # Confirm upload and save metadata
                    return await self.confirm_video_upload(
                        artifact_id=artifact_id,
                        storage_key=storage_key,
                        file_size_bytes=len(video_bytes),
                        metadata=metadata,
                    )
                else:
                    logger.error(
                        "Failed to store video in R2",
                        status=response.status_code,
                        response=response.text[:200],
                    )
                    raise Exception(f"R2 video upload failed: {response.status_code}")

        except Exception as e:
            logger.exception("R2 video storage error", error=str(e))
            return {
                "artifact_id": artifact_id,
                "type": "video",
                "storage": "failed",
                "error": str(e),
            }

    def _convert_buffer_to_base64(self, screenshot: Any) -> str | None:
        """Convert various screenshot formats to base64 string.

        Handles:
        - base64 string (pass through)
        - Buffer dict: {type: "Buffer", data: [bytes]}
        - bytes/bytearray
        """
        import base64 as b64

        if isinstance(screenshot, str) and len(screenshot) > 100:
            return screenshot
        elif isinstance(screenshot, dict) and screenshot.get("type") == "Buffer":
            # Node.js Buffer format from browser pool
            data = screenshot.get("data", [])
            if isinstance(data, list):
                byte_data = bytes(data)
                return b64.b64encode(byte_data).decode("utf-8")
        elif isinstance(screenshot, (bytes, bytearray)):
            return b64.b64encode(screenshot).decode("utf-8")
        return None

    async def store_test_result(
        self,
        result: dict[str, Any],
        test_id: str
    ) -> dict[str, Any]:
        """Store full test result, extracting screenshots to R2.

        Returns lightweight result with R2 references.
        """
        lightweight = dict(result)
        artifact_refs = []

        # Extract and store screenshots (handle both string and Buffer formats)
        for field in ["screenshot", "finalScreenshot"]:
            if field in lightweight and lightweight[field]:
                base64_data = self._convert_buffer_to_base64(lightweight[field])
                if base64_data and len(base64_data) > 1000:
                    ref = await self.store_screenshot(base64_data, {"source": field, "test_id": test_id})
                    artifact_refs.append(ref)
                    lightweight[field] = ref["artifact_id"]

        # Extract screenshots array
        if "screenshots" in lightweight and isinstance(lightweight["screenshots"], list):
            new_screenshots = []
            for i, screenshot in enumerate(lightweight["screenshots"]):
                base64_data = self._convert_buffer_to_base64(screenshot)
                if base64_data and len(base64_data) > 1000:
                    ref = await self.store_screenshot(base64_data, {"step": i, "test_id": test_id})
                    artifact_refs.append(ref)
                    new_screenshots.append(ref["artifact_id"])
                else:
                    new_screenshots.append(screenshot)
            lightweight["screenshots"] = new_screenshots

        # Extract step screenshots
        if "steps" in lightweight and isinstance(lightweight["steps"], list):
            for i, step in enumerate(lightweight["steps"]):
                if isinstance(step, dict) and "screenshot" in step and step["screenshot"]:
                    base64_data = self._convert_buffer_to_base64(step["screenshot"])
                    if base64_data and len(base64_data) > 1000:
                        ref = await self.store_screenshot(
                            base64_data,
                            {"step": i, "instruction": step.get("instruction", ""), "test_id": test_id}
                        )
                        artifact_refs.append(ref)
                        step["screenshot"] = ref["artifact_id"]

        lightweight["_artifact_refs"] = artifact_refs
        lightweight["_r2_stored"] = True

        return lightweight


# =============================================================================
# Vectorize (Self-Healing Memory) - Enhanced for RAP-248
# =============================================================================

# Index suffix mapping for multiple index types
_INDEX_SUFFIXES: dict[VectorizeIndexType, str] = {
    VectorizeIndexType.FAILURE_PATTERNS: "",  # Default index, no suffix
    VectorizeIndexType.CODE_EMBEDDINGS: "-code",
    VectorizeIndexType.TEST_EMBEDDINGS: "-tests",
    VectorizeIndexType.DOCUMENTATION: "-docs",
}


class VectorizeMemory:
    """Cloudflare Vectorize for semantic failure pattern memory (Enhanced RAP-248).

    Enables self-healing by:
    1. Storing failure patterns with their solutions
    2. Finding similar past failures via semantic search
    3. Suggesting healing strategies based on experience
    4. Parallel search across multiple index types
    5. Fallback to Cognee for comprehensive search

    This is the "learning" component of the agent.

    Index Types:
        - FAILURE_PATTERNS: Self-healing memory for error patterns (default)
        - CODE_EMBEDDINGS: Code snippets, function signatures, component patterns
        - TEST_EMBEDDINGS: Test cases, assertions, test scenarios
        - DOCUMENTATION: API docs, guides, inline comments

    Hot Path Optimization:
        For failure pattern matching, we use an in-memory LRU cache to avoid
        repeated embeddings and queries for common error patterns.
    """

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self._base_index = config.vectorize_index
        # Use bge-large-en-v1.5 (1024 dimensions) to match existing Vectorize index
        # Must match the dimension of the argus-patterns index
        self.ai_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/ai/run/@cf/baai/bge-large-en-v1.5"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }
        # Hot path cache for frequent failure pattern lookups (LRU-style)
        self._embedding_cache: dict[str, list[float]] = {}
        self._cache_max_size = 100
        # Shared HTTP client for connection reuse
        self._client: httpx.AsyncClient | None = None

    def _get_index_url(self, index_type: VectorizeIndexType) -> str:
        """Get the Vectorize API URL for a specific index type.

        Args:
            index_type: The type of index to access

        Returns:
            Full API URL for the index
        """
        suffix = _INDEX_SUFFIXES.get(index_type, "")
        index_name = f"{self._base_index}{suffix}"
        return f"https://api.cloudflare.com/client/v4/accounts/{self.config.account_id}/vectorize/v2/indexes/{index_name}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared HTTP client for connection reuse."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """Generate embedding using Workers AI with optional caching.

        Args:
            text: Text to generate embedding for
            use_cache: Whether to use the embedding cache (default: True)

        Returns:
            1024-dimensional embedding vector
        """
        # Hot path: check cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:32]
        if use_cache and cache_key in self._embedding_cache:
            logger.debug("Embedding cache hit", cache_key=cache_key[:8])
            return self._embedding_cache[cache_key]

        client = await self._get_client()
        response = await client.post(
            self.ai_url,
            headers=self.headers,
            json={"text": text},
        )

        if response.status_code == 200:
            data = response.json()
            embedding = data["result"]["data"][0]

            # Cache the embedding (with LRU eviction)
            if use_cache:
                if len(self._embedding_cache) >= self._cache_max_size:
                    # Simple eviction: remove oldest entries (FIFO approximation)
                    keys_to_remove = list(self._embedding_cache.keys())[:10]
                    for key in keys_to_remove:
                        del self._embedding_cache[key]
                self._embedding_cache[cache_key] = embedding

            return embedding
        else:
            raise Exception(f"Embedding generation failed: {response.status_code}")

    # =========================================================================
    # Core CRUD Operations with Index Type Support
    # =========================================================================

    async def upsert_embedding(
        self,
        index_type: VectorizeIndexType,
        id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Upsert an embedding into a specific index.

        Args:
            index_type: Which index to store in
            id: Unique identifier for the vector
            vector: 1024-dimensional embedding vector
            metadata: Optional metadata to store with the vector

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()
        index_url = self._get_index_url(index_type)

        try:
            response = await client.post(
                f"{index_url}/upsert",
                headers=self.headers,
                json={
                    "vectors": [{
                        "id": id,
                        "values": vector,
                        "metadata": metadata or {},
                    }]
                },
            )

            if response.status_code in [200, 201]:
                result = response.json()
                success = result.get("success", False)
                if success:
                    logger.info(
                        "Upserted embedding",
                        id=id,
                        index_type=index_type.value,
                    )
                return success
            else:
                logger.error(
                    "Failed to upsert embedding",
                    status=response.status_code,
                    index_type=index_type.value,
                )
                return False
        except Exception as e:
            logger.exception("Upsert embedding error", error=str(e))
            return False

    async def delete_embedding(
        self,
        index_type: VectorizeIndexType,
        id: str,
    ) -> bool:
        """Delete an embedding from a specific index.

        Args:
            index_type: Which index to delete from
            id: Unique identifier for the vector to delete

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()
        index_url = self._get_index_url(index_type)

        try:
            response = await client.post(
                f"{index_url}/delete-by-ids",
                headers=self.headers,
                json={"ids": [id]},
            )

            if response.status_code == 200:
                logger.info(
                    "Deleted embedding",
                    id=id,
                    index_type=index_type.value,
                )
                return True
            else:
                logger.error(
                    "Failed to delete embedding",
                    status=response.status_code,
                    index_type=index_type.value,
                )
                return False
        except Exception as e:
            logger.exception("Delete embedding error", error=str(e))
            return False

    async def _search_single_index(
        self,
        index_type: VectorizeIndexType,
        vector: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search a single index for similar vectors.

        Args:
            index_type: Which index to search
            vector: Query vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of search results
        """
        client = await self._get_client()
        index_url = self._get_index_url(index_type)

        payload: dict[str, Any] = {
            "vector": vector,
            "topK": top_k,
            "returnMetadata": True,
        }
        if filter_dict:
            payload["filter"] = filter_dict

        try:
            response = await client.post(
                f"{index_url}/query",
                headers=self.headers,
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                matches = data.get("result", {}).get("matches", [])

                results = []
                for match in matches:
                    results.append(VectorSearchResult(
                        id=match.get("id", ""),
                        score=match.get("score", 0.0),
                        index_type=index_type,
                        metadata=match.get("metadata", {}),
                    ))
                return results
            else:
                logger.warning(
                    "Index search failed",
                    index_type=index_type.value,
                    status=response.status_code,
                )
                return []
        except Exception as e:
            logger.exception(
                "Index search error",
                index_type=index_type.value,
                error=str(e),
            )
            return []

    # =========================================================================
    # Parallel Search Across Multiple Indexes (RAP-248)
    # =========================================================================

    async def search_parallel(
        self,
        query: str,
        indexes: list[VectorizeIndexType],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search multiple indexes in parallel and merge results.

        This enables cross-index semantic search for comprehensive retrieval.
        Results are merged and sorted by score descending.

        Args:
            query: Search query text
            indexes: List of index types to search
            top_k: Number of results per index (final results may be less after merging)
            min_score: Minimum similarity score to include
            filter_dict: Optional metadata filter applied to all indexes

        Returns:
            Merged and sorted list of search results from all indexes

        Example:
            results = await vectorize.search_parallel(
                query="button not found error",
                indexes=[
                    VectorizeIndexType.FAILURE_PATTERNS,
                    VectorizeIndexType.TEST_EMBEDDINGS,
                ],
                top_k=5,
                min_score=0.7,
            )
        """
        if not indexes:
            return []

        # Generate embedding once for all indexes
        embedding = await self._get_embedding(query)

        # Search all indexes in parallel
        search_tasks = [
            self._search_single_index(
                index_type=index_type,
                vector=embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )
            for index_type in indexes
        ]

        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results, filtering out errors
        merged: list[VectorSearchResult] = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Parallel search failed for index",
                    index=indexes[i].value,
                    error=str(result),
                )
                continue
            merged.extend(result)

        # Filter by min_score and sort by score descending
        filtered = [r for r in merged if r.score >= min_score]
        filtered.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "Parallel search complete",
            indexes=[i.value for i in indexes],
            total_results=len(filtered),
        )

        return filtered

    # =========================================================================
    # Fallback Search with Cognee Integration (RAP-248)
    # =========================================================================

    async def search_with_fallback(
        self,
        query: str,
        index_type: VectorizeIndexType,
        top_k: int = 5,
        min_score: float = 0.7,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search Vectorize first, fall back to Cognee if no results.

        This provides a robust search that leverages the edge-optimized Vectorize
        for speed, but falls back to Cognee's more comprehensive knowledge graph
        when Vectorize doesn't have relevant results.

        Args:
            query: Search query text
            index_type: Primary index to search
            top_k: Number of results to return
            min_score: Minimum similarity score
            filter_dict: Optional metadata filter

        Returns:
            List of search results as dictionaries

        Example:
            results = await vectorize.search_with_fallback(
                query="element not interactable",
                index_type=VectorizeIndexType.FAILURE_PATTERNS,
                top_k=5,
            )
        """
        # Try Vectorize first (fast edge search)
        try:
            embedding = await self._get_embedding(query)
            vectorize_results = await self._search_single_index(
                index_type=index_type,
                vector=embedding,
                top_k=top_k,
                filter_dict=filter_dict,
            )

            # Filter by min_score
            filtered = [r for r in vectorize_results if r.score >= min_score]

            if filtered:
                logger.info(
                    "Vectorize search returned results",
                    count=len(filtered),
                    index_type=index_type.value,
                )
                return [r.to_dict() for r in filtered]

            logger.info(
                "Vectorize search returned no results, falling back to Cognee",
                index_type=index_type.value,
            )
        except Exception as e:
            logger.warning(
                "Vectorize search failed, falling back to Cognee",
                error=str(e),
                index_type=index_type.value,
            )

        # Fall back to Cognee
        try:
            from src.knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client()

            # Map index type to Cognee namespace
            namespace_map = {
                VectorizeIndexType.FAILURE_PATTERNS: ["failure_patterns"],
                VectorizeIndexType.CODE_EMBEDDINGS: ["code_embeddings"],
                VectorizeIndexType.TEST_EMBEDDINGS: ["test_embeddings"],
                VectorizeIndexType.DOCUMENTATION: ["documentation"],
            }
            namespace = namespace_map.get(index_type, ["general"])

            cognee_results = await cognee.search(
                namespace=namespace,
                query=query,
                limit=top_k,
                threshold=min_score,
            )

            # Normalize Cognee results to match our format
            normalized = []
            for result in cognee_results:
                normalized.append({
                    "id": result.get("_id", result.get("id", "")),
                    "score": result.get("similarity", 0.8),
                    "index_type": index_type.value,
                    "metadata": result,
                    "source": "cognee",
                })

            logger.info(
                "Cognee fallback returned results",
                count=len(normalized),
                index_type=index_type.value,
            )
            return normalized

        except ImportError:
            logger.warning("Cognee not available for fallback search")
            return []
        except Exception as e:
            logger.exception("Cognee fallback search failed", error=str(e))
            return []

    # =========================================================================
    # Hot Path Optimization for Failure Pattern Matching (RAP-248)
    # =========================================================================

    async def find_failure_pattern_fast(
        self,
        error_message: str,
        selector: str,
        top_k: int = 3,
        min_score: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Hot path optimized failure pattern matching.

        This method is optimized for the most common use case: finding
        similar failure patterns during self-healing. It uses:
        - Embedding cache to avoid repeated AI calls
        - Connection reuse for lower latency
        - Smaller top_k and higher min_score for faster response

        Args:
            error_message: The error message from the failed step
            selector: The CSS selector that failed
            top_k: Number of results (default 3 for speed)
            min_score: Higher threshold (0.8) for confident matches

        Returns:
            List of high-confidence healing suggestions
        """
        # Build search text
        search_text = f"{error_message} selector:{selector}"

        # Use cached embedding when possible (hot path optimization)
        embedding = await self._get_embedding(search_text, use_cache=True)

        # Query only the failure patterns index with healed filter
        results = await self._search_single_index(
            index_type=VectorizeIndexType.FAILURE_PATTERNS,
            vector=embedding,
            top_k=top_k,
            filter_dict={"healed": {"$eq": True}},
        )

        # Filter by min_score and format response
        suggestions = []
        for match in results:
            if match.score >= min_score:
                metadata = match.metadata
                suggestions.append({
                    "pattern_id": match.id,
                    "score": match.score,
                    "original_error": metadata.get("error_message"),
                    "failed_selector": metadata.get("failed_selector"),
                    "healed_selector": metadata.get("healed_selector"),
                    "success_count": metadata.get("success_count", 0),
                    "confidence": min(0.95, match.score * metadata.get("success_count", 1) / 10),
                })

        logger.debug(
            "Fast failure pattern match",
            query_len=len(search_text),
            results=len(suggestions),
            cache_size=len(self._embedding_cache),
        )

        return suggestions

    # =========================================================================
    # Legacy Methods (Backward Compatibility)
    # =========================================================================

    async def store_failure_pattern(
        self,
        error_message: str,
        failed_selector: str,
        healed_selector: str | None = None,
        context: dict[str, Any] | None = None
    ) -> str:
        """Store a failure pattern for future self-healing.

        Args:
            error_message: The error that occurred
            failed_selector: The selector that failed
            healed_selector: The selector that worked (if healed)
            context: Additional context (URL, page structure, etc.)

        Returns:
            Pattern ID
        """
        # Create searchable text
        search_text = f"{error_message} selector:{failed_selector}"
        if context:
            search_text += f" url:{context.get('url', '')} element:{context.get('element_type', '')}"

        # Generate embedding
        embedding = await self._get_embedding(search_text)

        # Create pattern ID
        pattern_id = f"pattern_{hashlib.sha256(search_text.encode()).hexdigest()[:12]}"

        # Prepare metadata
        metadata = {
            "error_message": error_message[:500],
            "failed_selector": failed_selector,
            "healed_selector": healed_selector,
            "healed": healed_selector is not None,
            "url": context.get("url", "") if context else "",
            "element_type": context.get("element_type", "") if context else "",
            "created_at": datetime.now(UTC).isoformat(),
            "success_count": 1 if healed_selector else 0,
        }

        # Use the new upsert_embedding method
        success = await self.upsert_embedding(
            index_type=VectorizeIndexType.FAILURE_PATTERNS,
            id=pattern_id,
            vector=embedding,
            metadata=metadata,
        )

        if success:
            logger.info("Stored failure pattern", pattern_id=pattern_id, healed=healed_selector is not None)
            return pattern_id
        else:
            raise Exception("Failed to store failure pattern")

    async def find_similar_failures(
        self,
        error_message: str,
        selector: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> list[dict[str, Any]]:
        """Find similar past failures for self-healing suggestions.

        Args:
            error_message: Current error message
            selector: Current failed selector
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of similar patterns with healing suggestions
        """
        search_text = f"{error_message} selector:{selector}"
        embedding = await self._get_embedding(search_text)

        results = await self._search_single_index(
            index_type=VectorizeIndexType.FAILURE_PATTERNS,
            vector=embedding,
            top_k=top_k,
            filter_dict={"healed": {"$eq": True}},
        )

        suggestions = []
        for match in results:
            if match.score >= min_score:
                metadata = match.metadata
                suggestions.append({
                    "pattern_id": match.id,
                    "score": match.score,
                    "original_error": metadata.get("error_message"),
                    "failed_selector": metadata.get("failed_selector"),
                    "healed_selector": metadata.get("healed_selector"),
                    "success_count": metadata.get("success_count", 0),
                    "confidence": min(0.95, match.score * metadata.get("success_count", 1) / 10)
                })

        logger.info("Found similar failures", count=len(suggestions))
        return suggestions

    async def record_healing_success(self, pattern_id: str):
        """Record that a healing suggestion worked, increasing its confidence."""
        # In production, this would update the metadata to increase success_count
        # Vectorize doesn't support direct updates, so we'd need to fetch, delete, reinsert
        logger.info("Recording healing success", pattern_id=pattern_id)


# =============================================================================
# D1 Database (Test History & Metadata)
# =============================================================================

class D1Database:
    """Cloudflare D1 for test history and structured data.

    Stores:
    - Test run history
    - Conversation logs
    - User/project metadata
    - Analytics data
    """

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/d1/database/{config.d1_database_id}"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }

    async def execute(self, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
        """Execute a SQL query."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json={
                    "sql": sql,
                    "params": params or []
                },
                timeout=30.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"D1 query failed: {response.status_code}")

    async def store_test_run(
        self,
        test_id: str,
        project_id: str,
        result: dict[str, Any]
    ) -> str:
        """Store a test run in the database."""
        await self.execute(
            """
            INSERT INTO test_runs (id, project_id, success, steps_total, steps_passed, duration_ms, created_at, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                test_id,
                project_id,
                result.get("success", False),
                len(result.get("steps", [])),
                sum(1 for s in result.get("steps", []) if s.get("success")),
                result.get("duration_ms", 0),
                datetime.now(UTC).isoformat(),
                json.dumps(result)
            ]
        )
        return test_id

    async def get_test_history(
        self,
        project_id: str,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get recent test history for a project."""
        result = await self.execute(
            """
            SELECT id, success, steps_total, steps_passed, duration_ms, created_at
            FROM test_runs
            WHERE project_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [project_id, limit]
        )
        return result.get("result", [])


# =============================================================================
# KV Cache (Fast Session Data)
# =============================================================================

class KVCache:
    """Cloudflare KV for fast session caching.

    Stores:
    - Session state (fast retrieval)
    - Discovered page elements (cache)
    - User preferences
    """

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/storage/kv/namespaces/{config.kv_namespace_id}"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }

    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set a value in KV with TTL."""
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/values/{key}",
                headers=self.headers,
                params={"expiration_ttl": ttl_seconds},
                content=json.dumps(value) if not isinstance(value, str) else value,
                timeout=10.0
            )
            return response.status_code == 200

    async def get(self, key: str) -> Any | None:
        """Get a value from KV."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/values/{key}",
                headers=self.headers,
                timeout=10.0
            )
            if response.status_code == 200:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return response.text
            return None

    async def cache_page_elements(self, url: str, elements: list[dict[str, Any]]) -> bool:
        """Cache discovered page elements for faster subsequent access."""
        key = f"elements:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        return await self.set(key, elements, ttl_seconds=300)  # 5 minute cache

    async def get_cached_elements(self, url: str) -> list[dict[str, Any]] | None:
        """Get cached page elements."""
        key = f"elements:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        return await self.get(key)


# =============================================================================
# Workers AI (Edge Inference)
# =============================================================================

class WorkersAI:
    """Cloudflare Workers AI for edge inference.

    Use for:
    - Query intent classification (<10ms)
    - Confidence scoring
    - Simple extractions

    Cost: ~$0.001/1K tokens (cheaper than Claude Haiku)

    Available models:
    - @cf/meta/llama-3.1-8b-instruct: Fast instruction following (8B params)
    - @cf/meta/llama-3.2-3b-instruct: Ultra-fast for simple tasks (3B params)
    - @cf/mistral/mistral-7b-instruct-v0.1: Good for classification
    """

    # Intent classification categories
    INTENTS = [
        "run_test",           # User wants to run/execute a test
        "create_test",        # User wants to create/generate a new test
        "view_results",       # User wants to see test results
        "debug_failure",      # User wants to debug/analyze a failure
        "heal_test",          # User wants to fix/heal a broken test
        "list_tests",         # User wants to list available tests
        "configure",          # User wants to configure settings
        "explain",            # User wants explanation/documentation
        "cancel",             # User wants to cancel/stop something
        "other",              # Doesn't fit other categories
    ]

    # Entity types for extraction
    ENTITY_TYPES = [
        "test_name",          # Name of a test
        "file_path",          # Path to a file
        "error_type",         # Type of error (timeout, assertion, etc.)
        "selector",           # CSS/XPath selector
        "url",                # URL or endpoint
        "project_name",       # Project identifier
        "step_number",        # Step index in a test
    ]

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/ai/run"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }
        # Default to Llama 3.1 8B for good balance of speed and quality
        self.default_model = "@cf/meta/llama-3.1-8b-instruct"
        # Faster model for simple classification
        self.fast_model = "@cf/meta/llama-3.2-3b-instruct"

    async def classify_intent(
        self,
        query: str,
        model: str | None = None
    ) -> tuple[str, float]:
        """Classify query intent using Llama model.

        Uses a structured prompt to classify user queries into predefined
        intent categories. Optimized for <10ms edge inference.

        Args:
            query: The user's query text
            model: Optional model override (defaults to fast_model for speed)

        Returns:
            (intent, confidence) tuple where:
            - intent: One of the INTENTS categories
            - confidence: Score from 0.0 to 1.0

        Example:
            >>> intent, conf = await workers_ai.classify_intent("run my login test")
            >>> print(intent, conf)  # "run_test", 0.95
        """
        model = model or self.fast_model

        # Structured prompt for fast, reliable classification
        system_prompt = """You are a query intent classifier for a testing agent.
Classify the user query into exactly ONE of these categories:
- run_test: User wants to run/execute a test
- create_test: User wants to create/generate a new test
- view_results: User wants to see test results
- debug_failure: User wants to debug/analyze a failure
- heal_test: User wants to fix/heal a broken test
- list_tests: User wants to list available tests
- configure: User wants to configure settings
- explain: User wants explanation/documentation
- cancel: User wants to cancel/stop something
- other: Doesn't fit other categories

Respond ONLY with JSON: {"intent": "<category>", "confidence": <0.0-1.0>}"""

        user_prompt = f"Query: {query}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/{model}",
                    headers=self.headers,
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 50,  # Keep response short
                        "temperature": 0.1,  # Low temp for consistency
                    },
                    timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    result_text = data.get("result", {}).get("response", "")

                    # Parse JSON response
                    try:
                        # Handle potential markdown code blocks
                        if "```" in result_text:
                            result_text = result_text.split("```")[1]
                            if result_text.startswith("json"):
                                result_text = result_text[4:]

                        result = json.loads(result_text.strip())
                        intent = result.get("intent", "other")
                        confidence = float(result.get("confidence", 0.5))

                        # Validate intent is in known list
                        if intent not in self.INTENTS:
                            intent = "other"
                            confidence = max(0.3, confidence - 0.2)

                        # Clamp confidence to valid range
                        confidence = max(0.0, min(1.0, confidence))

                        logger.info(
                            "Classified intent",
                            query=query[:50],
                            intent=intent,
                            confidence=confidence
                        )
                        return (intent, confidence)

                    except (json.JSONDecodeError, KeyError, ValueError) as parse_error:
                        logger.warning(
                            "Failed to parse intent response",
                            response=result_text[:100],
                            error=str(parse_error)
                        )
                        # Fallback: try to extract intent from raw text
                        for known_intent in self.INTENTS:
                            if known_intent in result_text.lower():
                                return (known_intent, 0.5)
                        return ("other", 0.3)

                else:
                    logger.error(
                        "Workers AI intent classification failed",
                        status=response.status_code,
                        response=response.text[:200]
                    )
                    return ("other", 0.0)

        except httpx.TimeoutException:
            logger.warning("Workers AI timeout during intent classification")
            return ("other", 0.0)
        except Exception as e:
            logger.exception("Workers AI error during intent classification", error=str(e))
            return ("other", 0.0)

    async def extract_entities(
        self,
        text: str,
        entity_types: list[str] | None = None,
        model: str | None = None
    ) -> dict[str, list[str]]:
        """Extract entities from text.

        Uses LLM to identify and extract structured entities from
        unstructured text. Useful for parsing user queries or error messages.

        Args:
            text: Input text to extract from
            entity_types: Types to extract (defaults to ENTITY_TYPES)
                - test_name: Names of tests
                - file_path: File paths
                - error_type: Error categories (timeout, assertion, etc.)
                - selector: CSS/XPath selectors
                - url: URLs or endpoints
                - project_name: Project identifiers
                - step_number: Step indices

        Returns:
            Dict mapping entity_type to list of extracted values

        Example:
            >>> entities = await workers_ai.extract_entities(
            ...     "The login test failed at step 3 with selector #submit-btn",
            ...     ["test_name", "step_number", "selector"]
            ... )
            >>> print(entities)
            {"test_name": ["login test"], "step_number": ["3"], "selector": ["#submit-btn"]}
        """
        model = model or self.default_model
        entity_types = entity_types or self.ENTITY_TYPES

        # Build entity type descriptions
        entity_descriptions = {
            "test_name": "name or identifier of a test",
            "file_path": "file system path (e.g., /src/tests/login.spec.ts)",
            "error_type": "type of error (timeout, assertion, element_not_found, network, etc.)",
            "selector": "CSS selector or XPath (e.g., #submit-btn, .login-form, //button[@type='submit'])",
            "url": "URL or endpoint (e.g., https://example.com, /api/users)",
            "project_name": "project or repository name",
            "step_number": "step index or number",
        }

        types_to_find = [
            f"- {t}: {entity_descriptions.get(t, t)}"
            for t in entity_types
            if t in entity_descriptions
        ]

        system_prompt = f"""You are an entity extractor. Extract the following entity types from the text:
{chr(10).join(types_to_find)}

Respond ONLY with JSON mapping entity_type to list of extracted values.
If no entities found for a type, use empty list.
Example: {{"test_name": ["login test"], "selector": ["#submit-btn"], "step_number": []}}"""

        user_prompt = f"Text: {text}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/{model}",
                    headers=self.headers,
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.1,
                    },
                    timeout=15.0
                )

                if response.status_code == 200:
                    data = response.json()
                    result_text = data.get("result", {}).get("response", "")

                    try:
                        # Handle potential markdown code blocks
                        if "```" in result_text:
                            result_text = result_text.split("```")[1]
                            if result_text.startswith("json"):
                                result_text = result_text[4:]

                        entities = json.loads(result_text.strip())

                        # Ensure all requested types are present
                        result: dict[str, list[str]] = {}
                        for entity_type in entity_types:
                            extracted = entities.get(entity_type, [])
                            # Normalize to list
                            if isinstance(extracted, str):
                                extracted = [extracted] if extracted else []
                            elif not isinstance(extracted, list):
                                extracted = []
                            result[entity_type] = extracted

                        logger.info(
                            "Extracted entities",
                            text=text[:50],
                            entity_count=sum(len(v) for v in result.values())
                        )
                        return result

                    except (json.JSONDecodeError, KeyError) as parse_error:
                        logger.warning(
                            "Failed to parse entity extraction response",
                            response=result_text[:100],
                            error=str(parse_error)
                        )
                        # Return empty results
                        return {t: [] for t in entity_types}

                else:
                    logger.error(
                        "Workers AI entity extraction failed",
                        status=response.status_code
                    )
                    return {t: [] for t in entity_types}

        except httpx.TimeoutException:
            logger.warning("Workers AI timeout during entity extraction")
            return {t: [] for t in entity_types}
        except Exception as e:
            logger.exception("Workers AI error during entity extraction", error=str(e))
            return {t: [] for t in entity_types}

    async def compute_confidence(
        self,
        query: str,
        results: list[dict],
        model: str | None = None
    ) -> float:
        """Compute confidence score for search results.

        Evaluates how well search results match the user's query.
        Useful for deciding whether to return results or ask for clarification.

        Args:
            query: The original user query
            results: List of search results (dicts with 'title', 'content', etc.)

        Returns:
            Confidence score from 0.0 to 1.0

        Example:
            >>> conf = await workers_ai.compute_confidence(
            ...     "login test failure",
            ...     [{"title": "Login Test", "content": "Failed at step 3"}]
            ... )
            >>> print(conf)  # 0.85
        """
        model = model or self.fast_model

        if not results:
            return 0.0

        # Summarize results for context
        results_summary = []
        for i, r in enumerate(results[:5]):  # Limit to top 5
            title = r.get("title", r.get("name", f"Result {i+1}"))
            content = r.get("content", r.get("description", ""))[:100]
            results_summary.append(f"{i+1}. {title}: {content}")

        system_prompt = """You evaluate how well search results match a query.
Score from 0.0 to 1.0:
- 1.0: Perfect match, exactly what user asked for
- 0.7-0.9: Good match, relevant results
- 0.4-0.6: Partial match, some relevance
- 0.1-0.3: Poor match, barely relevant
- 0.0: No match, completely irrelevant

Respond ONLY with JSON: {"confidence": <score>, "reason": "<brief reason>"}"""

        user_prompt = f"""Query: {query}

Results:
{chr(10).join(results_summary)}"""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/{model}",
                    headers=self.headers,
                    json={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 80,
                        "temperature": 0.1,
                    },
                    timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    result_text = data.get("result", {}).get("response", "")

                    try:
                        # Handle potential markdown code blocks
                        if "```" in result_text:
                            result_text = result_text.split("```")[1]
                            if result_text.startswith("json"):
                                result_text = result_text[4:]

                        result = json.loads(result_text.strip())
                        confidence = float(result.get("confidence", 0.5))
                        reason = result.get("reason", "")

                        # Clamp to valid range
                        confidence = max(0.0, min(1.0, confidence))

                        logger.info(
                            "Computed result confidence",
                            query=query[:30],
                            confidence=confidence,
                            reason=reason[:50] if reason else ""
                        )
                        return confidence

                    except (json.JSONDecodeError, KeyError, ValueError) as parse_error:
                        logger.warning(
                            "Failed to parse confidence response",
                            response=result_text[:100],
                            error=str(parse_error)
                        )
                        # Fallback: return moderate confidence if we got results
                        return 0.5 if results else 0.0

                else:
                    logger.error(
                        "Workers AI confidence computation failed",
                        status=response.status_code
                    )
                    return 0.5 if results else 0.0

        except httpx.TimeoutException:
            logger.warning("Workers AI timeout during confidence computation")
            return 0.5 if results else 0.0
        except Exception as e:
            logger.exception("Workers AI error during confidence computation", error=str(e))
            return 0.5 if results else 0.0

    async def batch_classify(
        self,
        queries: list[str],
        model: str | None = None
    ) -> list[tuple[str, float]]:
        """Classify multiple queries in parallel.

        Uses asyncio.gather for concurrent classification requests.
        For truly high-volume batching, consider using Workers AI batch API.

        Args:
            queries: List of query strings to classify

        Returns:
            List of (intent, confidence) tuples
        """
        tasks = [self.classify_intent(query, model) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Batch classification error", error=str(result))
                processed_results.append(("other", 0.0))
            else:
                processed_results.append(result)

        return processed_results

    async def health_check(self) -> dict[str, Any]:
        """Check Workers AI availability and latency.

        Returns:
            Dict with status, latency_ms, and model info
        """
        start_time = time.time()

        try:
            # Simple classification to test connectivity
            intent, confidence = await self.classify_intent("test query", self.fast_model)
            latency_ms = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "model": self.fast_model,
                "test_result": {"intent": intent, "confidence": confidence}
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "error": str(e)
            }


# =============================================================================
# AI Gateway (LLM Routing with Caching)
# =============================================================================

@dataclass
class AIGatewayCacheConfig:
    """Configuration for AI Gateway caching.

    Cloudflare AI Gateway supports response caching to reduce costs
    and improve latency for repeated queries.

    Attributes:
        enabled: Whether caching is enabled (default: True)
        ttl_seconds: Cache time-to-live in seconds (default: 3600 = 1 hour)
        skip_cache_header: Header name to bypass cache (default: cf-skip-cache)
    """
    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour
    skip_cache_header: str = "cf-skip-cache"  # Header to bypass cache


class AIGateway:
    """Cloudflare AI Gateway for LLM call routing with caching.

    Benefits:
    - Unified billing across providers
    - Caching (save costs on repeated queries)
    - Rate limiting
    - Observability & logging
    - Fallback routing

    Caching:
    - Responses are cached based on prompt content hash
    - Cache TTL configurable (default 1 hour)
    - Use cf-skip-cache header to bypass cache for fresh responses
    - Analytics track cache hit/miss rates

    Example usage:
        gateway = AIGateway(config)

        # Get URL with cache headers
        url = gateway.get_anthropic_url()
        headers = gateway.get_cache_headers(skip_cache=False)

        # Build cache key for deduplication
        cache_key = gateway.build_cache_key(prompt_hash, model)

        # Log cache analytics
        gateway.log_cache_hit(hit=True, latency_ms=150)
    """

    def __init__(
        self,
        config: CloudflareConfig,
        cache_config: AIGatewayCacheConfig | None = None
    ):
        self.config = config
        self.cache_config = cache_config or AIGatewayCacheConfig()
        self.gateway_url = f"https://gateway.ai.cloudflare.com/v1/{config.account_id}/{config.ai_gateway_id}"

        # Analytics tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_latency_ms = 0
        self._request_count = 0

    def get_anthropic_url(self) -> str:
        """Get the AI Gateway URL for Anthropic."""
        return f"{self.gateway_url}/anthropic/v1/messages"

    def get_openai_url(self) -> str:
        """Get the AI Gateway URL for OpenAI."""
        return f"{self.gateway_url}/openai/v1/chat/completions"

    def get_cached_completion_url(self, provider: str) -> str:
        """Get the AI Gateway URL for a provider with cache parameters.

        Args:
            provider: The LLM provider ('anthropic', 'openai', 'workers-ai')

        Returns:
            Full gateway URL for the provider

        Raises:
            ValueError: If provider is not supported
        """
        provider_paths = {
            "anthropic": "/anthropic/v1/messages",
            "openai": "/openai/v1/chat/completions",
            "workers-ai": "/workers-ai",
        }

        if provider not in provider_paths:
            supported = ", ".join(provider_paths.keys())
            raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")

        return f"{self.gateway_url}{provider_paths[provider]}"

    def build_cache_key(self, prompt_hash: str, model: str) -> str:
        """Generate a cache key for a prompt/model combination.

        The cache key is used by AI Gateway to deduplicate requests.
        Same prompt + model = same cache key = cached response.

        Args:
            prompt_hash: SHA256 hash of the prompt content
            model: Model identifier (e.g., 'claude-sonnet-4-5-20250514')

        Returns:
            Cache key string in format: argus:{model}:{prompt_hash[:32]}
        """
        # Normalize model name (remove version suffixes for broader cache hits)
        model_normalized = model.split("-202")[0] if "-202" in model else model

        # Truncate hash to 32 chars for reasonable key length
        hash_truncated = prompt_hash[:32]

        cache_key = f"argus:{model_normalized}:{hash_truncated}"

        logger.debug(
            "Built cache key",
            cache_key=cache_key,
            model=model,
            prompt_hash_prefix=prompt_hash[:8]
        )

        return cache_key

    def get_cache_headers(self, skip_cache: bool = False) -> dict[str, str]:
        """Get HTTP headers for cache control.

        Args:
            skip_cache: If True, include header to bypass cache

        Returns:
            Dict of cache-related headers to include in request
        """
        headers = {}

        if not self.cache_config.enabled:
            # Caching disabled globally - always skip
            headers[self.cache_config.skip_cache_header] = "true"
        elif skip_cache:
            # Caching enabled but caller wants to skip for this request
            headers[self.cache_config.skip_cache_header] = "true"

        # Add cache TTL hint header (AI Gateway uses this for cache duration)
        if self.cache_config.enabled and not skip_cache:
            headers["cf-cache-ttl"] = str(self.cache_config.ttl_seconds)

        return headers

    def log_cache_hit(self, hit: bool, latency_ms: int) -> None:
        """Log cache hit/miss analytics for monitoring.

        Call this after each LLM request to track cache effectiveness.

        Args:
            hit: True if response was served from cache
            latency_ms: Request latency in milliseconds
        """
        self._request_count += 1
        self._total_latency_ms += latency_ms

        if hit:
            self._cache_hits += 1
            logger.info(
                "AI Gateway cache HIT",
                latency_ms=latency_ms,
                cache_hit_rate=self.get_cache_hit_rate(),
            )
        else:
            self._cache_misses += 1
            logger.info(
                "AI Gateway cache MISS",
                latency_ms=latency_ms,
                cache_hit_rate=self.get_cache_hit_rate(),
            )

    def get_cache_hit_rate(self) -> float:
        """Get the current cache hit rate as a percentage.

        Returns:
            Cache hit rate (0.0 to 100.0), or 0.0 if no requests yet
        """
        if self._request_count == 0:
            return 0.0
        return (self._cache_hits / self._request_count) * 100.0

    def get_analytics(self) -> dict[str, Any]:
        """Get comprehensive cache analytics.

        Returns:
            Dict with cache statistics for monitoring/dashboards
        """
        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count > 0
            else 0.0
        )

        return {
            "cache_enabled": self.cache_config.enabled,
            "cache_ttl_seconds": self.cache_config.ttl_seconds,
            "total_requests": self._request_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_percent": self.get_cache_hit_rate(),
            "avg_latency_ms": avg_latency,
            "total_latency_ms": self._total_latency_ms,
            "estimated_cost_savings_percent": self.get_cache_hit_rate(),  # Rough estimate
        }

    def reset_analytics(self) -> None:
        """Reset all analytics counters."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_latency_ms = 0
        self._request_count = 0
        logger.info("AI Gateway analytics reset")


# =============================================================================
# Unified Cloudflare Client
# =============================================================================

class CloudflareClient:
    """Unified client for all Cloudflare services.

    Provides access to:
    - R2: Object storage for screenshots, videos, artifacts
    - Vectorize: Vector database for failure patterns & memory
    - D1: SQL database for test history & metadata
    - KV: Fast key-value cache for sessions
    - AI Gateway: LLM routing with caching
    - Workers AI: Edge inference for intent classification (<10ms)
    """

    def __init__(self, config: CloudflareConfig | None = None):
        self.config = config or CloudflareConfig.from_env()
        self.r2 = R2Storage(self.config)
        self.vectorize = VectorizeMemory(self.config)
        self.d1 = D1Database(self.config)
        self.kv = KVCache(self.config)
        self.ai_gateway = AIGateway(self.config)
        self.workers_ai = WorkersAI(self.config)

    async def store_test_artifacts(
        self,
        result: dict[str, Any],
        test_id: str,
        project_id: str
    ) -> dict[str, Any]:
        """Store all test artifacts across Cloudflare services.

        1. Screenshots → R2
        2. Result metadata → D1
        3. Failure patterns → Vectorize (if failed)

        Returns lightweight result for LangGraph state.
        """
        # Store screenshots in R2
        lightweight = await self.r2.store_test_result(result, test_id)

        # Store metadata in D1
        if self.config.d1_database_id:
            await self.d1.store_test_run(test_id, project_id, lightweight)

        # If test failed, store pattern in Vectorize for learning
        if not result.get("success") and result.get("error"):
            if self.config.vectorize_index:
                # Extract failed step info
                failed_steps = [s for s in result.get("steps", []) if not s.get("success")]
                for step in failed_steps:
                    await self.vectorize.store_failure_pattern(
                        error_message=step.get("error", result.get("error", "")),
                        failed_selector=step.get("selector", "unknown"),
                        context={"url": result.get("url", ""), "instruction": step.get("instruction", "")}
                    )

        return lightweight

    async def get_healing_suggestions(
        self,
        error_message: str,
        selector: str,
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Get self-healing suggestions from Vectorize memory.

        Queries the VectorizeMemory for similar past failures and returns
        healing suggestions that have worked before.

        Args:
            error_message: The error message from the failed step
            selector: The CSS selector that failed
            limit: Maximum number of suggestions to return

        Returns:
            List of healing suggestions with confidence scores
        """
        if not self.config.vectorize_index:
            return []

        suggestions = await self.vectorize.find_similar_failures(error_message, selector)
        return suggestions[:limit] if suggestions else []


# =============================================================================
# Global Instance
# =============================================================================

_cloudflare_client: CloudflareClient | None = None


def get_cloudflare_client() -> CloudflareClient:
    """Get the global Cloudflare client instance.

    Uses settings-based configuration by default.
    """
    global _cloudflare_client
    if _cloudflare_client is None:
        # Prefer settings-based config over raw environment variables
        try:
            config = CloudflareConfig.from_settings()
            _cloudflare_client = CloudflareClient(config)
        except Exception as e:
            logger.warning("Failed to load Cloudflare config from settings, using env vars", error=str(e))
            _cloudflare_client = CloudflareClient()
    return _cloudflare_client


def is_cloudflare_configured() -> bool:
    """Check if Cloudflare services are properly configured."""
    try:
        config = CloudflareConfig.from_settings()
        return bool(config.account_id and config.api_token)
    except Exception:
        return bool(os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN"))
