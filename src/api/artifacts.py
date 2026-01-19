"""Artifacts API endpoint for retrieving stored screenshots and other artifacts.

This module provides endpoints for fetching artifact content that was stored
during test execution. Screenshots are stored in Cloudflare R2 (or memory
fallback) and referenced by ID in the chat stream.

The frontend can use these endpoints to fetch actual image data when it
receives artifact IDs in tool results.
"""

import base64

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
                # Fetch from R2
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


@router.get("/{artifact_id}/raw")
async def get_artifact_raw(
    artifact_id: str,
):
    """
    Get artifact content as raw image bytes.

    This endpoint returns the actual image file with proper content-type headers,
    suitable for direct use in <img> tags.

    Args:
        artifact_id: The artifact ID

    Returns:
        Raw image bytes with appropriate Content-Type header.

    SECURITY: Publicly accessible for image delivery (used in <img> tags).
    Security is maintained via:
    - Artifact IDs contain content hashes (hard to guess)
    - Artifacts auto-expire after 24 hours
    - Rate limiting on the endpoint
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
async def list_recent_artifacts(limit: int = 20, type: str | None = None):
    """
    List recent artifacts from the memory store.

    Note: This only lists artifacts in the current process memory.
    R2-stored artifacts are not listed here.

    Args:
        limit: Maximum number of artifacts to return
        type: Filter by artifact type (screenshot, video, html)

    Returns:
        List of artifact references (without full content).
    """
    artifact_store = get_artifact_store()

    # Get all artifacts from memory store
    artifacts = []
    for artifact_id, artifact in artifact_store._memory_store.items():
        if type and artifact.type != type:
            continue

        artifacts.append({
            "artifact_id": artifact.id,
            "type": artifact.type,
            "created_at": artifact.created_at,
            "metadata": artifact.metadata,
            # Don't include full content in list response
            "content_preview": f"{artifact.content[:50]}..." if len(artifact.content) > 50 else artifact.content,
        })

    # Sort by created_at descending
    artifacts.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return ArtifactListResponse(
        artifacts=artifacts[:limit],
        total=len(artifacts),
    )


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
