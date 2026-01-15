"""Supabase client for Quality Intelligence data storage."""

from typing import Any, Optional
import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class SupabaseClient:
    """Client for Supabase REST API operations."""

    def __init__(self, url: Optional[str] = None, service_key: Optional[str] = None):
        settings = get_settings()
        self.url = url or settings.supabase_url
        self.service_key = service_key or (
            settings.supabase_service_key.get_secret_value()
            if settings.supabase_service_key
            else None
        )
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is configured."""
        return bool(self.url and self.service_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"{self.url}/rest/v1",
                headers={
                    "apikey": self.service_key,
                    "Authorization": f"Bearer {self.service_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def request(
        self,
        path: str,
        method: str = "GET",
        body: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make a request to Supabase REST API.

        Args:
            path: REST API path (e.g., "/production_events")
            method: HTTP method
            body: Request body for POST/PATCH
            headers: Additional headers

        Returns:
            {"data": ..., "error": ...}
        """
        if not self.is_configured:
            return {"data": None, "error": "Supabase not configured"}

        client = await self._get_client()

        request_headers = {"Prefer": "return=representation"}
        if method == "POST":
            request_headers["Prefer"] = "return=representation,resolution=merge-duplicates"
        if headers:
            request_headers.update(headers)

        try:
            response = await client.request(
                method=method,
                url=path,
                json=body if body else None,
                headers=request_headers,
            )

            if not response.is_success:
                error_text = response.text
                logger.error(
                    "Supabase request failed",
                    path=path,
                    status=response.status_code,
                    error=error_text,
                )
                return {"data": None, "error": error_text}

            data = response.json() if response.text else None
            return {"data": data, "error": None}

        except Exception as e:
            logger.exception("Supabase request error", path=path, error=str(e))
            return {"data": None, "error": str(e)}

    # Convenience methods
    async def insert(self, table: str, data: dict) -> dict[str, Any]:
        """Insert a record."""
        return await self.request(f"/{table}", method="POST", body=data)

    async def select(
        self, table: str, columns: str = "*", filters: Optional[dict] = None
    ) -> dict[str, Any]:
        """Select records with optional filters."""
        path = f"/{table}?select={columns}"
        if filters:
            for key, value in filters.items():
                path += f"&{key}={value}"
        return await self.request(path)

    async def update(
        self, table: str, filters: dict, data: dict
    ) -> dict[str, Any]:
        """Update records matching filters."""
        path = f"/{table}?"
        path += "&".join(f"{k}={v}" for k, v in filters.items())
        return await self.request(path, method="PATCH", body=data)

    async def rpc(self, function_name: str, params: dict) -> dict[str, Any]:
        """Call a PostgreSQL function via PostgREST RPC.

        Args:
            function_name: Name of the PostgreSQL function to call
            params: Parameters to pass to the function

        Returns:
            {"data": ..., "error": ...}
        """
        return await self.request(f"/rpc/{function_name}", method="POST", body=params)


# Global instance
_supabase_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create global Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client
