"""
Cloudflare Vectorize REST API client.

Used for semantic search on error patterns.
"""

import logging

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


class CloudflareVectorizeClient:
    """Cloudflare Vectorize REST API client."""

    def __init__(self, account_id: str, index_name: str, api_token: str):
        self.account_id = account_id
        self.index_name = index_name
        self.api_token = api_token
        # Use v2 API for Vectorize (v2 indexes require v2 API)
        self.base_url = f"{CF_API_BASE}/{account_id}/vectorize/v2/indexes/{index_name}"
        # Use bge-large-en-v1.5 for better quality embeddings (1024 dimensions)
        self.ai_url = f"{CF_API_BASE}/{account_id}/ai/run/@cf/baai/bge-large-en-v1.5"
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def generate_embedding(self, text: str) -> list[float] | None:
        """
        Generate text embedding using Cloudflare Workers AI.

        Uses @cf/baai/bge-large-en-v1.5 model (1024 dimensions) for better quality.
        API Ref: https://developers.cloudflare.com/workers-ai/models/bge-large-en-v1.5/
        """
        try:
            client = await self._get_client()
            # Workers AI expects text as an array
            response = await client.post(
                self.ai_url,
                headers=self._get_headers(),
                json={"text": [text]}
            )

            if response.status_code == 200:
                result = response.json()
                # Workers AI returns {"result": {"data": [[...embedding...]]}}
                if result.get("success") and result.get("result"):
                    data = result["result"].get("data", [])
                    if data and len(data) > 0:
                        return data[0]
                logger.warning(f"Embedding response missing data: {result}")
                return None
            else:
                logger.warning(f"Embedding generation error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.warning(f"Embedding generation exception: {e}")
            return None

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
        return_values: bool = False,
        return_metadata: bool = True
    ) -> list[dict]:
        """
        Query the Vectorize index for similar vectors.

        Args:
            vector: Query vector (1024 dimensions for bge-large-en-v1.5)
            top_k: Number of results to return
            filter: Optional metadata filter
            return_values: Whether to return vector values (unused in v2)
            return_metadata: Whether to return metadata (unused in v2)

        Returns:
            List of matches with id, score, and optional metadata/values

        API Reference: https://developers.cloudflare.com/api/resources/vectorize/
        """
        try:
            client = await self._get_client()

            # Vectorize v2 API - simplified payload
            # See: https://developers.cloudflare.com/api/resources/vectorize/subresources/indexes/methods/query/
            payload = {
                "vector": vector,
                "topK": top_k
            }
            if filter:
                payload["filter"] = filter

            response = await client.post(
                f"{self.base_url}/query",
                headers=self._get_headers(),
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return result.get("result", {}).get("matches", [])
                else:
                    logger.warning(f"Vectorize query failed: {result.get('errors')}")
                    return []
            else:
                logger.warning(f"Vectorize query error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.warning(f"Vectorize query exception: {e}")
            return []

    async def insert(
        self,
        vectors: list[dict]
    ) -> bool:
        """
        Insert vectors into the index.

        Args:
            vectors: List of dicts with id, values, and optional metadata
                     e.g. [{"id": "error-123", "values": [...], "metadata": {...}}]

        Returns:
            True if successful
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.base_url}/insert",
                headers=self._get_headers(),
                json={"vectors": vectors}
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                logger.warning(f"Vectorize insert error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.warning(f"Vectorize insert exception: {e}")
            return False

    async def upsert(
        self,
        vectors: list[dict]
    ) -> bool:
        """
        Upsert vectors into the index (insert or update).

        Args:
            vectors: List of dicts with id, values, and optional metadata

        Returns:
            True if successful
        """
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.base_url}/upsert",
                headers=self._get_headers(),
                json={"vectors": vectors}
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("success", False)
            else:
                logger.warning(f"Vectorize upsert error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.warning(f"Vectorize upsert exception: {e}")
            return False

    async def delete_by_ids(self, ids: list[str]) -> bool:
        """Delete vectors by their IDs."""
        try:
            client = await self._get_client()

            response = await client.post(
                f"{self.base_url}/delete-by-ids",
                headers=self._get_headers(),
                json={"ids": ids}
            )

            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Vectorize delete exception: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global Vectorize client (lazy initialized)
_vectorize_client: CloudflareVectorizeClient | None = None


def get_vectorize_client() -> CloudflareVectorizeClient | None:
    """Get or create Cloudflare Vectorize client (lazy initialization)."""
    global _vectorize_client

    if _vectorize_client is not None:
        return _vectorize_client

    settings = get_settings()

    if not settings.cloudflare_api_token or not settings.cloudflare_account_id:
        logger.warning("Cloudflare Vectorize not configured - semantic search disabled")
        return None

    if not settings.cloudflare_vectorize_index:
        logger.warning("Cloudflare Vectorize index not configured - semantic search disabled")
        return None

    try:
        api_token = settings.cloudflare_api_token
        if hasattr(api_token, 'get_secret_value'):
            api_token = api_token.get_secret_value()

        _vectorize_client = CloudflareVectorizeClient(
            account_id=settings.cloudflare_account_id,
            index_name=settings.cloudflare_vectorize_index,
            api_token=api_token
        )
        logger.info("Cloudflare Vectorize client initialized")
        return _vectorize_client
    except Exception as e:
        logger.error(f"Failed to initialize Cloudflare Vectorize client: {e}")
        return None


async def semantic_search_errors(
    error_text: str,
    limit: int = 5,
    min_score: float = 0.7
) -> list[dict]:
    """
    Search for similar error patterns using semantic similarity.

    Args:
        error_text: Error text to search for
        limit: Maximum results to return
        min_score: Minimum similarity score (0-1)

    Returns:
        List of similar patterns with scores
    """
    vectorize = get_vectorize_client()

    if vectorize is None:
        logger.warning("Vectorize not available, using fallback search")
        return []

    # Generate embedding for the query text
    embedding = await vectorize.generate_embedding(error_text)

    if embedding is None:
        logger.warning("Failed to generate embedding, using fallback search")
        return []

    # Query the vector index
    matches = await vectorize.query(
        vector=embedding,
        top_k=limit * 2,  # Get more to filter by score
        return_metadata=True
    )

    # Filter by minimum score and format results
    results = []
    for match in matches:
        score = match.get("score", 0)
        if score >= min_score:
            results.append({
                "id": match.get("id"),
                "score": score,
                "metadata": match.get("metadata", {})
            })

    return results[:limit]


async def index_error_pattern(
    error_id: str,
    error_text: str,
    metadata: dict | None = None
) -> bool:
    """
    Index an error pattern for future semantic search.

    Args:
        error_id: Unique identifier for the error
        error_text: Error text to embed
        metadata: Additional metadata to store

    Returns:
        True if successful
    """
    vectorize = get_vectorize_client()

    if vectorize is None:
        logger.warning("Vectorize not available, cannot index error")
        return False

    # Generate embedding
    embedding = await vectorize.generate_embedding(error_text)

    if embedding is None:
        logger.warning("Failed to generate embedding for indexing")
        return False

    # Upsert the vector
    return await vectorize.upsert([{
        "id": error_id,
        "values": embedding,
        "metadata": metadata or {}
    }])


async def index_production_event(event: dict) -> bool:
    """
    Index a production event for semantic search.

    Called automatically when new errors come in via webhooks.

    Args:
        event: Production event dict with id, title, message, etc.

    Returns:
        True if indexed successfully
    """
    event_id = event.get("id")
    if not event_id:
        logger.warning("Cannot index event without ID")
        return False

    # Build searchable text from event fields
    title = event.get("title", "")
    message = event.get("message", "")
    component = event.get("component", "")
    stack_trace = event.get("stack_trace", "")

    # Combine text for embedding (limit to avoid token limits)
    search_text = f"{title} {message}"
    if component:
        search_text += f" in {component}"
    if stack_trace:
        # Just include first 500 chars of stack trace
        search_text += f" {stack_trace[:500]}"

    search_text = search_text.strip()[:2000]  # Limit total length

    if not search_text:
        logger.warning(f"No text to index for event {event_id}")
        return False

    # Build metadata for filtering and display
    # Use `or` pattern to handle None values (event.get returns None if key exists with None value)
    url = event.get("url") or ""
    metadata = {
        "title": title[:200] if title else "",
        "message": message[:500] if message else "",
        "severity": event.get("severity") or "error",
        "source": event.get("source") or "unknown",
        "component": component[:100] if component else "",
        "url": url[:200],
        "fingerprint": event.get("fingerprint") or "",
        "project_id": event.get("project_id") or "",
    }

    success = await index_error_pattern(event_id, search_text, metadata)

    if success:
        logger.info(f"Indexed production event {event_id} for semantic search")
    else:
        logger.warning(f"Failed to index production event {event_id}")

    return success
