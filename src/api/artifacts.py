"""Artifacts API endpoint for retrieving stored screenshots and other artifacts.

This module provides endpoints for fetching artifact content that was stored
during test execution. Screenshots are stored in Cloudflare R2 (or memory
fallback) and referenced by ID in the chat stream.

The frontend can use these endpoints to fetch actual image data when it
receives artifact IDs in tool results.
"""

import base64
import hashlib
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.orchestrator.artifact_store import get_artifact_store
from src.services.cloudflare_storage import get_cloudflare_client, is_cloudflare_configured

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/artifacts", tags=["Artifacts"])


class ArtifactResponse(BaseModel):
    """Response containing artifact metadata and content."""
    artifact_id: str
    type: str
    content: str = Field(..., description="Base64 encoded content or URL")
    content_type: str = Field(default="base64", description="Either 'base64' or 'url'")
    metadata: dict | None = None


class SignedUrlResponse(BaseModel):
    """Response containing a signed URL for authenticated media access."""
    artifact_id: str
    url: str = Field(..., description="Signed URL with HMAC signature and expiration")
    expires_in: int = Field(..., description="URL expiry time in seconds")
    url_type: str = Field(default="signed", description="Type of URL (signed or presigned)")


class ArtifactListResponse(BaseModel):
    """Response listing multiple artifacts."""
    artifacts: list[dict]
    total: int


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: str,
    format: str = "base64",
    user: UserContext = Depends(get_current_user),
):
    """
    Get artifact content by ID.

    Args:
        artifact_id: The artifact ID (e.g., screenshot_07d3a7894e3944d1_20260113_141415)
        format: Return format - 'base64' (default) or 'url' (presigned URL if available)

    Returns:
        Artifact content as base64 or URL depending on format parameter.

    The artifact ID format is: {type}_{content_hash}_{timestamp}
    Common types: screenshot, video, html

    SECURITY: Requires authentication. Note: Full org-based isolation requires
    adding organization_id to artifact storage (TODO: implement proper multi-tenant
    artifact isolation).
    """
    logger.info("Fetching artifact", artifact_id=artifact_id, format=format)

    # First, try memory artifact store (for non-R2 stored artifacts)
    artifact_store = get_artifact_store()
    artifact = artifact_store.get(artifact_id)

    if artifact:
        logger.info("Found artifact in memory store", artifact_id=artifact_id, type=artifact.type)
        return ArtifactResponse(
            artifact_id=artifact.id,
            type=artifact.type,
            content=artifact.content,
            content_type="base64",
            metadata=artifact.metadata,
        )

    # If not in memory, try Cloudflare R2
    if is_cloudflare_configured():
        try:
            cf_client = get_cloudflare_client()

            # Determine the artifact type from the ID prefix
            artifact_type = artifact_id.split("_")[0] if "_" in artifact_id else "screenshot"

            if artifact_type == "screenshot":
                # If URL format requested, try to generate signed URL first
                if format == "url":
                    # Prefer HMAC signed URL (faster, simpler)
                    signed_url = cf_client.r2.generate_signed_url(artifact_id, artifact_type="screenshot")
                    if signed_url:
                        logger.info("Generated signed URL for artifact", artifact_id=artifact_id)
                        return ArtifactResponse(
                            artifact_id=artifact_id,
                            type="screenshot",
                            content=signed_url,
                            content_type="url",
                            metadata={"storage": "r2", "url_type": "signed", "expiry_seconds": cf_client.r2.config.media_url_expiry},
                        )

                    # Fallback to S3v4 presigned URL
                    presigned_url = cf_client.r2.get_presigned_url(artifact_id)
                    if presigned_url:
                        logger.info("Generated presigned URL for artifact", artifact_id=artifact_id)
                        return ArtifactResponse(
                            artifact_id=artifact_id,
                            type="screenshot",
                            content=presigned_url,
                            content_type="url",
                            metadata={"storage": "r2", "url_type": "presigned", "expiry_seconds": cf_client.r2.config.r2_presigned_url_expiry},
                        )

                    # Fall back to base64 if URL generation failed
                    logger.warning("URL generation failed, falling back to base64", artifact_id=artifact_id)

                # Fetch from R2 as base64
                base64_content = await cf_client.r2.get_screenshot(artifact_id)

                if base64_content:
                    logger.info("Found artifact in R2", artifact_id=artifact_id)
                    return ArtifactResponse(
                        artifact_id=artifact_id,
                        type="screenshot",
                        content=base64_content,
                        content_type="base64",
                        metadata={"storage": "r2"},
                    )

            logger.warning("Artifact not found in R2", artifact_id=artifact_id)

        except Exception as e:
            logger.exception("Error fetching from R2", artifact_id=artifact_id, error=str(e))
            # Fall through to 404

    # Artifact not found
    raise HTTPException(
        status_code=404,
        detail=f"Artifact not found: {artifact_id}. It may have expired or was stored in a different session."
    )


@router.get("/{artifact_id}/url", response_model=SignedUrlResponse)
async def get_artifact_signed_url(
    artifact_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Generate a signed URL for authenticated artifact access.

    This endpoint generates a time-limited signed URL that can be used
    to fetch the artifact directly from the CDN without requiring
    authentication headers on each request.

    Args:
        artifact_id: The artifact ID (e.g., screenshot_xxx_yyy)

    Returns:
        SignedUrlResponse containing:
        - url: The signed URL (valid for ~15 minutes)
        - expires_in: Seconds until URL expires
        - url_type: 'signed' (HMAC) or 'presigned' (S3v4 fallback)

    The signed URL format: /screenshots/{id}?sig=SIGNATURE&exp=EXPIRATION

    SECURITY: Requires authentication. The signed URL verifies both
    the artifact ID and expiration time via HMAC-SHA256.
    """
    logger.info("Generating signed URL for artifact", artifact_id=artifact_id, user_id=user.user_id)

    # Validate artifact ID format (basic check)
    if not artifact_id or "_" not in artifact_id:
        raise HTTPException(status_code=400, detail="Invalid artifact ID format")

    if not is_cloudflare_configured():
        raise HTTPException(
            status_code=503,
            detail="Cloudflare storage not configured"
        )

    try:
        cf_client = get_cloudflare_client()

        # Determine artifact type from ID prefix
        artifact_type = artifact_id.split("_")[0] if "_" in artifact_id else "screenshot"

        # Try signed URL first (preferred - HMAC-based, lightweight)
        signed_url = cf_client.r2.generate_signed_url(artifact_id, artifact_type=artifact_type)

        if signed_url:
            return SignedUrlResponse(
                artifact_id=artifact_id,
                url=signed_url,
                expires_in=cf_client.r2.config.media_url_expiry,
                url_type="signed",
            )

        # Fallback to S3v4 presigned URL if signing secret not configured
        presigned_url = cf_client.r2.get_presigned_url(artifact_id)

        if presigned_url:
            return SignedUrlResponse(
                artifact_id=artifact_id,
                url=presigned_url,
                expires_in=cf_client.r2.config.r2_presigned_url_expiry,
                url_type="presigned",
            )

        # Neither signing method available - return error
        raise HTTPException(
            status_code=503,
            detail="URL signing not configured. Set CLOUDFLARE_MEDIA_SIGNING_SECRET or R2 S3 credentials."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating signed URL", artifact_id=artifact_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate signed URL: {str(e)}"
        )


@router.get("/{artifact_id}/raw")
async def get_artifact_raw(
    artifact_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Get artifact content as raw image bytes.

    This endpoint returns the actual image file with proper content-type headers,
    suitable for authenticated image delivery.

    Args:
        artifact_id: The artifact ID

    Returns:
        Raw image bytes with appropriate Content-Type header.

    SECURITY: Requires authentication. All artifacts are customer-specific
    and confidential.
    """
    # Validate artifact ID format to prevent enumeration
    import re
    if not re.match(r'^(screenshot|video|html)_[a-f0-9]{16}_\d{8}_\d{6}$', artifact_id):
        raise HTTPException(status_code=400, detail="Invalid artifact ID format")
    logger.info("Fetching raw artifact", artifact_id=artifact_id)

    # First, try memory artifact store
    artifact_store = get_artifact_store()
    artifact = artifact_store.get(artifact_id)

    content_bytes = None

    if artifact:
        try:
            # Content might be base64 encoded
            content = artifact.content
            if content.startswith("data:"):
                # Remove data URL prefix
                content = content.split(",", 1)[1]
            content_bytes = base64.b64decode(content)
        except Exception as e:
            logger.warning("Failed to decode artifact content", error=str(e))

    # Try Cloudflare R2
    if content_bytes is None and is_cloudflare_configured():
        try:
            cf_client = get_cloudflare_client()
            base64_content = await cf_client.r2.get_screenshot(artifact_id)

            if base64_content:
                content_bytes = base64.b64decode(base64_content)
        except Exception as e:
            logger.exception("Error fetching raw from R2", artifact_id=artifact_id, error=str(e))

    if content_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact not found: {artifact_id}"
        )

    # Determine content type from artifact ID
    artifact_type = artifact_id.split("_")[0] if "_" in artifact_id else "screenshot"
    content_type = "image/png" if artifact_type == "screenshot" else "application/octet-stream"

    return Response(
        content=content_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{artifact_id}.png"',
            "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
        }
    )


@router.get("/")
async def list_recent_artifacts(
    limit: int = 20,
    type: str | None = None,
    user: UserContext = Depends(get_current_user),
):
    """
    List recent artifacts from Supabase and memory store.

    Args:
        limit: Maximum number of artifacts to return
        type: Filter by artifact type (screenshot, video, html)

    Returns:
        List of artifact references (without full content).
    """
    artifacts = []

    # First, try to get artifacts from Supabase
    try:
        from src.integrations.supabase import get_supabase
        supabase = await get_supabase()
        if supabase:
            query_params = {"limit": limit}
            if type:
                query_params["type"] = type
            if user.organization_id:
                query_params["organization_id"] = user.organization_id

            db_artifacts = await supabase.select(
                "artifacts",
                columns="id, type, storage_backend, storage_url, test_id, thread_id, action_description, file_size_bytes, metadata, created_at",
                order_by="created_at",
                ascending=False,
                **query_params
            )

            for artifact in db_artifacts or []:
                artifacts.append({
                    "artifact_id": artifact["id"],
                    "type": artifact["type"],
                    "storage": artifact["storage_backend"],
                    "url": artifact.get("storage_url"),
                    "test_id": artifact.get("test_id"),
                    "thread_id": artifact.get("thread_id"),
                    "action": artifact.get("action_description"),
                    "file_size_bytes": artifact.get("file_size_bytes"),
                    "created_at": artifact.get("created_at"),
                    "metadata": artifact.get("metadata", {}),
                })
    except Exception as e:
        logger.warning("Failed to fetch artifacts from Supabase", error=str(e))

    # Also check memory store for any in-memory artifacts
    artifact_store = get_artifact_store()
    for artifact_id, artifact in artifact_store._memory_store.items():
        if type and artifact.type != type:
            continue
        # Avoid duplicates
        if not any(a["artifact_id"] == artifact_id for a in artifacts):
            artifacts.append({
                "artifact_id": artifact.id,
                "type": artifact.type,
                "storage": "memory",
                "created_at": artifact.created_at,
                "metadata": artifact.metadata,
            })

    # Sort by created_at descending
    artifacts.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return ArtifactListResponse(
        artifacts=artifacts[:limit],
        total=len(artifacts),
    )


# =============================================================================
# Video Upload Endpoints (Following Official R2 Documentation)
# https://developers.cloudflare.com/r2/api/s3/presigned-urls/
# =============================================================================


class VideoUploadRequest(BaseModel):
    """Request for generating a video upload presigned URL."""
    content_type: str = Field(default="video/webm", description="Video MIME type (video/webm or video/mp4)")
    file_size_bytes: int | None = Field(default=None, description="Expected file size for validation")
    session_id: str | None = Field(default=None, description="Discovery session ID")
    project_id: str | None = Field(default=None, description="Project ID")
    metadata: dict | None = Field(default=None, description="Additional metadata")


class VideoUploadUrlResponse(BaseModel):
    """Response with presigned URL for video upload."""
    artifact_id: str
    upload_url: str = Field(..., description="Presigned PUT URL for direct R2 upload")
    storage_key: str = Field(..., description="R2 object key")
    content_type: str
    expiry_seconds: int
    callback_url: str = Field(..., description="URL to call after upload completes")


class VideoUploadConfirmRequest(BaseModel):
    """Request to confirm video upload completion."""
    artifact_id: str
    storage_key: str
    file_size_bytes: int | None = None
    duration_seconds: float | None = None
    session_id: str | None = None
    project_id: str | None = None
    metadata: dict | None = None


class VideoArtifactResponse(BaseModel):
    """Response with video artifact details."""
    artifact_id: str
    type: str = "video"
    url: str = Field(..., description="Worker URL for video access")
    presigned_url: str | None = Field(None, description="Presigned GET URL (backup)")
    content_type: str
    file_size_bytes: int | None = None
    duration_seconds: float | None = None
    metadata: dict | None = None


@router.post("/videos/upload-url", response_model=VideoUploadUrlResponse)
async def generate_video_upload_url(
    request: VideoUploadRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Generate a presigned URL for direct video upload to R2.

    This follows the official Cloudflare R2 presigned URL pattern:
    1. External service (BrowserPool) requests upload URL
    2. Service uploads directly to R2 using presigned PUT URL
    3. Service calls callback URL to confirm upload

    Per R2 docs:
    - Presigned URLs valid for up to 7 days (604,800 seconds)
    - Content-Type in URL must match upload header
    - Single-part upload limit: 5GB

    Args:
        request: Video upload parameters including content type

    Returns:
        Presigned PUT URL and callback endpoint
    """
    if not is_cloudflare_configured():
        raise HTTPException(
            status_code=503,
            detail="Cloudflare R2 storage not configured"
        )

    # Validate content type
    valid_types = ["video/webm", "video/mp4", "video/x-matroska"]
    if request.content_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type. Supported: {', '.join(valid_types)}"
        )

    # Check file size if provided (R2 single-part limit is 5GB)
    max_size = 5 * 1024 * 1024 * 1024  # 5GB
    if request.file_size_bytes and request.file_size_bytes > max_size:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum single-part upload: 5GB. Use multipart for larger files."
        )

    try:
        cf_client = get_cloudflare_client()

        # Generate unique artifact ID
        unique_str = f"{user.user_id}:{request.session_id or 'none'}:{datetime.now(UTC).isoformat()}"
        content_hash = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
        artifact_id = f"video_{content_hash}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        # Prepare metadata
        metadata = request.metadata or {}
        metadata.update({
            "user_id": user.user_id,
            "organization_id": user.organization_id,
            "session_id": request.session_id,
            "project_id": request.project_id,
        })

        # Generate presigned PUT URL
        upload_info = cf_client.r2.generate_video_upload_url(
            artifact_id=artifact_id,
            content_type=request.content_type,
            expiry_seconds=3600,  # 1 hour to upload
            metadata=metadata,
        )

        if not upload_info:
            raise HTTPException(
                status_code=503,
                detail="Failed to generate upload URL. R2 credentials may not be configured."
            )

        logger.info(
            "Generated video upload URL",
            artifact_id=artifact_id,
            user_id=user.user_id,
            content_type=request.content_type,
        )

        # Build callback URL for upload confirmation
        import os
        base_url = os.getenv("API_BASE_URL", "https://argus-brain-production.up.railway.app")
        callback_url = f"{base_url}/api/v1/artifacts/videos/confirm"

        return VideoUploadUrlResponse(
            artifact_id=artifact_id,
            upload_url=upload_info["upload_url"],
            storage_key=upload_info["storage_key"],
            content_type=request.content_type,
            expiry_seconds=upload_info["expiry_seconds"],
            callback_url=callback_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating video upload URL", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate upload URL: {str(e)}"
        )


@router.post("/videos/confirm", response_model=VideoArtifactResponse)
async def confirm_video_upload(
    request: VideoUploadConfirmRequest,
    user: UserContext = Depends(get_current_user),
):
    """
    Confirm video upload completion and save metadata to database.

    Call this endpoint after successfully uploading to the presigned URL.
    This saves the artifact reference to Supabase for querying and generates
    the serving URL via Cloudflare Worker.

    Args:
        request: Upload confirmation with artifact details

    Returns:
        Video artifact with serving URLs
    """
    if not is_cloudflare_configured():
        raise HTTPException(
            status_code=503,
            detail="Cloudflare R2 storage not configured"
        )

    try:
        cf_client = get_cloudflare_client()

        # Prepare metadata
        metadata = request.metadata or {}
        metadata.update({
            "user_id": user.user_id,
            "organization_id": user.organization_id,
            "session_id": request.session_id,
            "project_id": request.project_id,
        })

        # Confirm upload and save to Supabase
        artifact_ref = await cf_client.r2.confirm_video_upload(
            artifact_id=request.artifact_id,
            storage_key=request.storage_key,
            file_size_bytes=request.file_size_bytes,
            duration_seconds=request.duration_seconds,
            metadata=metadata,
        )

        logger.info(
            "Confirmed video upload",
            artifact_id=request.artifact_id,
            file_size_bytes=request.file_size_bytes,
            duration_seconds=request.duration_seconds,
        )

        return VideoArtifactResponse(
            artifact_id=artifact_ref["artifact_id"],
            url=artifact_ref["url"],
            presigned_url=artifact_ref.get("presigned_url"),
            content_type=artifact_ref["content_type"],
            file_size_bytes=artifact_ref.get("file_size_bytes"),
            duration_seconds=artifact_ref.get("duration_seconds"),
            metadata=artifact_ref.get("metadata"),
        )

    except Exception as e:
        logger.exception("Error confirming video upload", artifact_id=request.artifact_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to confirm upload: {str(e)}"
        )


@router.get("/videos/{artifact_id}", response_model=VideoArtifactResponse)
async def get_video_artifact(
    artifact_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Get video artifact details and serving URL.

    Args:
        artifact_id: The video artifact ID (e.g., video_abc123_20260122_123456)

    Returns:
        Video artifact with Worker URL and optional presigned URL
    """
    # Validate artifact ID format
    if not artifact_id.startswith("video_"):
        raise HTTPException(status_code=400, detail="Invalid video artifact ID format")

    # Try to fetch from Supabase first
    try:
        from src.integrations.supabase import get_supabase
        supabase = await get_supabase()
        if supabase:
            artifacts = await supabase.select(
                "artifacts",
                columns="*",
                filters={"id": artifact_id},
                limit=1,
            )
            if artifacts:
                artifact = artifacts[0]

                # Generate fresh presigned URL if needed
                presigned_url = None
                if is_cloudflare_configured():
                    cf_client = get_cloudflare_client()
                    presigned_url = cf_client.r2._generate_video_presigned_get_url(
                        artifact.get("storage_key", f"videos/{artifact_id}.webm")
                    )

                return VideoArtifactResponse(
                    artifact_id=artifact["id"],
                    url=artifact.get("storage_url", ""),
                    presigned_url=presigned_url,
                    content_type=artifact.get("content_type", "video/webm"),
                    file_size_bytes=artifact.get("file_size_bytes"),
                    duration_seconds=artifact.get("metadata", {}).get("duration_seconds"),
                    metadata=artifact.get("metadata"),
                )
    except Exception as e:
        logger.warning("Failed to fetch video from Supabase", artifact_id=artifact_id, error=str(e))

    # Fallback: construct URLs directly
    if is_cloudflare_configured():
        cf_client = get_cloudflare_client()
        worker_url = f"{cf_client.r2.config.worker_url}/videos/{artifact_id}"
        presigned_url = cf_client.r2._generate_video_presigned_get_url(f"videos/{artifact_id}.webm")

        return VideoArtifactResponse(
            artifact_id=artifact_id,
            url=worker_url,
            presigned_url=presigned_url,
            content_type="video/webm",
            metadata={"source": "fallback"},
        )

    raise HTTPException(status_code=404, detail=f"Video artifact not found: {artifact_id}")


@router.get("/videos/{artifact_id}/url", response_model=SignedUrlResponse)
async def get_video_signed_url(
    artifact_id: str,
    user: UserContext = Depends(get_current_user),
):
    """
    Generate a signed URL for authenticated video access.

    Similar to screenshot signed URLs but for video content.

    Args:
        artifact_id: The video artifact ID

    Returns:
        Signed URL with expiration
    """
    if not artifact_id.startswith("video_"):
        raise HTTPException(status_code=400, detail="Invalid video artifact ID format")

    if not is_cloudflare_configured():
        raise HTTPException(status_code=503, detail="Cloudflare storage not configured")

    try:
        cf_client = get_cloudflare_client()

        # Try HMAC signed URL first
        signed_url = cf_client.r2.generate_signed_url(artifact_id, artifact_type="video")

        if signed_url:
            return SignedUrlResponse(
                artifact_id=artifact_id,
                url=signed_url,
                expires_in=cf_client.r2.config.media_url_expiry,
                url_type="signed",
            )

        # Fallback to presigned URL
        presigned_url = cf_client.r2._generate_video_presigned_get_url(f"videos/{artifact_id}.webm")

        if presigned_url:
            return SignedUrlResponse(
                artifact_id=artifact_id,
                url=presigned_url,
                expires_in=cf_client.r2.config.r2_presigned_url_expiry,
                url_type="presigned",
            )

        raise HTTPException(
            status_code=503,
            detail="URL signing not configured"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating video signed URL", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")


@router.post("/resolve")
async def resolve_artifact_refs(artifact_refs: list[dict]):
    """
    Resolve multiple artifact references to their content.

    This is useful when you have _artifact_refs from a tool result
    and want to fetch all artifacts in a single request.

    Args:
        artifact_refs: List of artifact reference objects with artifact_id field

    Returns:
        List of resolved artifacts with their content.
    """
    results = []

    for ref in artifact_refs:
        artifact_id = ref.get("artifact_id")
        if not artifact_id:
            continue

        try:
            # Try to get each artifact
            artifact_store = get_artifact_store()
            artifact = artifact_store.get(artifact_id)

            if artifact:
                results.append({
                    "artifact_id": artifact.id,
                    "type": artifact.type,
                    "content": artifact.content,
                    "metadata": artifact.metadata,
                    "found": True,
                })
            elif is_cloudflare_configured():
                # Try R2
                cf_client = get_cloudflare_client()
                base64_content = await cf_client.r2.get_screenshot(artifact_id)

                if base64_content:
                    results.append({
                        "artifact_id": artifact_id,
                        "type": "screenshot",
                        "content": base64_content,
                        "metadata": {"storage": "r2"},
                        "found": True,
                    })
                else:
                    results.append({
                        "artifact_id": artifact_id,
                        "found": False,
                        "error": "Not found in R2",
                    })
            else:
                results.append({
                    "artifact_id": artifact_id,
                    "found": False,
                    "error": "Not found",
                })
        except Exception as e:
            results.append({
                "artifact_id": artifact_id,
                "found": False,
                "error": str(e),
            })

    return {
        "artifacts": results,
        "resolved": sum(1 for r in results if r.get("found")),
        "total": len(artifact_refs),
    }
