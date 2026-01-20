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
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


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
    # Worker URL for public screenshot access (preferred over presigned URLs)
    worker_url: str = "https://argus-api.anthropic.workers.dev"

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
            worker_url=os.getenv("CLOUDFLARE_WORKER_URL", "https://argus-api.anthropic.workers.dev"),
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
            worker_url=getattr(settings, 'cloudflare_worker_url', None) or "https://argus-api.anthropic.workers.dev",
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
# Vectorize (Self-Healing Memory)
# =============================================================================

class VectorizeMemory:
    """Cloudflare Vectorize for semantic failure pattern memory.

    Enables self-healing by:
    1. Storing failure patterns with their solutions
    2. Finding similar past failures via semantic search
    3. Suggesting healing strategies based on experience

    This is the "learning" component of the agent.
    """

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/vectorize/v2/indexes/{config.vectorize_index}"
        # Use bge-large-en-v1.5 (1024 dimensions) to match existing Vectorize index
        # Must match the dimension of the argus-patterns index
        self.ai_url = f"https://api.cloudflare.com/client/v4/accounts/{config.account_id}/ai/run/@cf/baai/bge-large-en-v1.5"
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using Workers AI."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ai_url,
                headers=self.headers,
                json={"text": text},
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                return data["result"]["data"][0]
            else:
                raise Exception(f"Embedding generation failed: {response.status_code}")

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

        # Store in Vectorize
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/insert",
                headers=self.headers,
                json={
                    "vectors": [{
                        "id": pattern_id,
                        "values": embedding,
                        "metadata": {
                            "error_message": error_message[:500],
                            "failed_selector": failed_selector,
                            "healed_selector": healed_selector,
                            "healed": healed_selector is not None,
                            "url": context.get("url", "") if context else "",
                            "element_type": context.get("element_type", "") if context else "",
                            "created_at": datetime.now(UTC).isoformat(),
                            "success_count": 1 if healed_selector else 0,
                        }
                    }]
                },
                timeout=30.0
            )

            if response.status_code in [200, 201]:
                logger.info("Stored failure pattern", pattern_id=pattern_id, healed=healed_selector is not None)
                return pattern_id
            else:
                logger.error("Failed to store pattern", status=response.status_code)
                raise Exception(f"Vectorize insert failed: {response.status_code}")

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

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json={
                    "vector": embedding,
                    "topK": top_k,
                    "returnMetadata": True,
                    "filter": {"healed": {"$eq": True}}  # Only return patterns that were successfully healed
                },
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                matches = data.get("result", {}).get("matches", [])

                suggestions = []
                for match in matches:
                    if match.get("score", 0) >= min_score:
                        metadata = match.get("metadata", {})
                        suggestions.append({
                            "pattern_id": match.get("id"),
                            "score": match.get("score"),
                            "original_error": metadata.get("error_message"),
                            "failed_selector": metadata.get("failed_selector"),
                            "healed_selector": metadata.get("healed_selector"),
                            "success_count": metadata.get("success_count", 0),
                            "confidence": min(0.95, match.get("score", 0) * metadata.get("success_count", 1) / 10)
                        })

                logger.info("Found similar failures", count=len(suggestions))
                return suggestions
            else:
                logger.error("Vectorize query failed", status=response.status_code)
                return []

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
# AI Gateway (LLM Routing)
# =============================================================================

class AIGateway:
    """Cloudflare AI Gateway for LLM call routing.

    Benefits:
    - Unified billing across providers
    - Caching (save costs on repeated queries)
    - Rate limiting
    - Observability & logging
    - Fallback routing
    """

    def __init__(self, config: CloudflareConfig):
        self.config = config
        self.gateway_url = f"https://gateway.ai.cloudflare.com/v1/{config.account_id}/{config.ai_gateway_id}"

    def get_anthropic_url(self) -> str:
        """Get the AI Gateway URL for Anthropic."""
        return f"{self.gateway_url}/anthropic/v1/messages"

    def get_openai_url(self) -> str:
        """Get the AI Gateway URL for OpenAI."""
        return f"{self.gateway_url}/openai/v1/chat/completions"


# =============================================================================
# Unified Cloudflare Client
# =============================================================================

class CloudflareClient:
    """Unified client for all Cloudflare services."""

    def __init__(self, config: CloudflareConfig | None = None):
        self.config = config or CloudflareConfig.from_env()
        self.r2 = R2Storage(self.config)
        self.vectorize = VectorizeMemory(self.config)
        self.d1 = D1Database(self.config)
        self.kv = KVCache(self.config)
        self.ai_gateway = AIGateway(self.config)

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
