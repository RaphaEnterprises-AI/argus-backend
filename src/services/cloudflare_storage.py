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

import os
import json
import hashlib
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass
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
        )

    @classmethod
    def from_settings(cls) -> "CloudflareConfig":
        """Load config from application settings (recommended)."""
        from src.config import get_settings
        settings = get_settings()

        api_token = ""
        if settings.cloudflare_api_token:
            api_token = settings.cloudflare_api_token.get_secret_value()

        return cls(
            account_id=settings.cloudflare_account_id or "",
            api_token=api_token,
            r2_bucket=settings.cloudflare_r2_bucket or "argus-artifacts",
            vectorize_index=settings.cloudflare_vectorize_index or "argus-patterns",
            d1_database_id=settings.cloudflare_d1_database_id or "",
            kv_namespace_id=settings.cloudflare_kv_namespace_id or "",
            ai_gateway_id=settings.cloudflare_gateway_id or "argus-gateway",
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a screenshot in R2.

        Args:
            base64_data: Base64 encoded image data
            metadata: Optional metadata (step, test_id, etc.)

        Returns:
            Reference object with artifact_id and URL
        """
        # Generate content-based ID for deduplication
        content_hash = hashlib.sha256(base64_data[:2000].encode()).hexdigest()[:16]
        artifact_id = f"screenshot_{content_hash}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Store in R2
        key = f"screenshots/{artifact_id}.png"

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
                    return {
                        "artifact_id": artifact_id,
                        "type": "screenshot",
                        "storage": "r2",
                        "key": key,
                        "url": f"https://{self.config.r2_bucket}.r2.cloudflarestorage.com/{key}",
                        "metadata": metadata or {},
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
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

    async def get_screenshot(self, artifact_id: str) -> Optional[str]:
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

    async def store_test_result(
        self,
        result: Dict[str, Any],
        test_id: str
    ) -> Dict[str, Any]:
        """Store full test result, extracting screenshots to R2.

        Returns lightweight result with R2 references.
        """
        lightweight = dict(result)
        artifact_refs = []

        # Extract and store screenshots
        for field in ["screenshot", "finalScreenshot"]:
            if field in lightweight and isinstance(lightweight[field], str) and len(lightweight[field]) > 1000:
                ref = await self.store_screenshot(lightweight[field], {"source": field, "test_id": test_id})
                artifact_refs.append(ref)
                lightweight[field] = ref["artifact_id"]

        # Extract screenshots array
        if "screenshots" in lightweight and isinstance(lightweight["screenshots"], list):
            new_screenshots = []
            for i, screenshot in enumerate(lightweight["screenshots"]):
                if isinstance(screenshot, str) and len(screenshot) > 1000:
                    ref = await self.store_screenshot(screenshot, {"step": i, "test_id": test_id})
                    artifact_refs.append(ref)
                    new_screenshots.append(ref["artifact_id"])
                else:
                    new_screenshots.append(screenshot)
            lightweight["screenshots"] = new_screenshots

        # Extract step screenshots
        if "steps" in lightweight and isinstance(lightweight["steps"], list):
            for i, step in enumerate(lightweight["steps"]):
                if isinstance(step, dict) and "screenshot" in step:
                    if isinstance(step["screenshot"], str) and len(step["screenshot"]) > 1000:
                        ref = await self.store_screenshot(
                            step["screenshot"],
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

    async def _get_embedding(self, text: str) -> List[float]:
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
        healed_selector: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
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
                            "created_at": datetime.now(timezone.utc).isoformat(),
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
    ) -> List[Dict[str, Any]]:
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

    async def execute(self, sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
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
        result: Dict[str, Any]
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
                datetime.now(timezone.utc).isoformat(),
                json.dumps(result)
            ]
        )
        return test_id

    async def get_test_history(
        self,
        project_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
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

    async def get(self, key: str) -> Optional[Any]:
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

    async def cache_page_elements(self, url: str, elements: List[Dict[str, Any]]) -> bool:
        """Cache discovered page elements for faster subsequent access."""
        key = f"elements:{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        return await self.set(key, elements, ttl_seconds=300)  # 5 minute cache

    async def get_cached_elements(self, url: str) -> Optional[List[Dict[str, Any]]]:
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

    def __init__(self, config: Optional[CloudflareConfig] = None):
        self.config = config or CloudflareConfig.from_env()
        self.r2 = R2Storage(self.config)
        self.vectorize = VectorizeMemory(self.config)
        self.d1 = D1Database(self.config)
        self.kv = KVCache(self.config)
        self.ai_gateway = AIGateway(self.config)

    async def store_test_artifacts(
        self,
        result: Dict[str, Any],
        test_id: str,
        project_id: str
    ) -> Dict[str, Any]:
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
        selector: str
    ) -> List[Dict[str, Any]]:
        """Get self-healing suggestions from Vectorize memory."""
        if not self.config.vectorize_index:
            return []

        return await self.vectorize.find_similar_failures(error_message, selector)


# =============================================================================
# Global Instance
# =============================================================================

_cloudflare_client: Optional[CloudflareClient] = None


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
