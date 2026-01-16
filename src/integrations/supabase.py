"""Supabase integration for audit logs, analytics, and persistence.

This module provides a Supabase client wrapper for storing audit events,
test results, and other persistent data.
"""

import os
from typing import Any, Optional
import structlog

logger = structlog.get_logger(__name__)

# Global client instance (lazy initialized)
_supabase_client: Optional["SupabaseClient"] = None


class SupabaseClient:
    """Async-compatible Supabase client wrapper."""

    def __init__(self, url: str, key: str):
        """Initialize Supabase client.

        Args:
            url: Supabase project URL
            key: Supabase service role key
        """
        self.url = url
        self.key = key
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        """Lazily initialize the Supabase client."""
        if self._initialized:
            return self._client is not None

        try:
            from supabase import create_client, Client
            self._client: Client = create_client(self.url, self.key)
            self._initialized = True
            logger.info("Supabase client initialized", url=self.url[:50] + "...")
            return True
        except ImportError:
            logger.warning("supabase-py not installed, audit persistence disabled")
            self._initialized = True
            return False
        except Exception as e:
            logger.error("Failed to initialize Supabase client", error=str(e))
            self._initialized = True
            return False

    async def insert(self, table: str, records: list[dict[str, Any]]) -> bool:
        """Insert records into a table.

        Args:
            table: Table name
            records: List of records to insert

        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_initialized():
            return False

        if not self._client:
            return False

        try:
            result = self._client.table(table).insert(records).execute()
            logger.debug("Inserted records into Supabase", table=table, count=len(records))
            return True
        except Exception as e:
            # Log the first record's keys for debugging schema mismatches
            record_keys = list(records[0].keys()) if records else []
            logger.error(
                "Failed to insert into Supabase",
                table=table,
                error=str(e),
                record_keys=record_keys,
                error_type=type(e).__name__,
            )
            return False

    async def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        """Select records from a table.

        Args:
            table: Table name
            columns: Columns to select (default: "*")
            filters: Dictionary of column=value filters
            limit: Maximum number of records
            order_by: Column to order by
            ascending: Sort order

        Returns:
            List of records
        """
        if not await self._ensure_initialized():
            return []

        if not self._client:
            return []

        try:
            query = self._client.table(table).select(columns)

            if filters:
                for column, value in filters.items():
                    query = query.eq(column, value)

            if order_by:
                query = query.order(order_by, desc=not ascending)

            if limit:
                query = query.limit(limit)

            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error("Failed to select from Supabase", table=table, error=str(e))
            return []

    async def update(
        self,
        table: str,
        values: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """Update records in a table.

        Args:
            table: Table name
            values: Dictionary of column=value to update
            filters: Dictionary of column=value filters

        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_initialized():
            return False

        if not self._client:
            return False

        try:
            query = self._client.table(table).update(values)

            for column, value in filters.items():
                query = query.eq(column, value)

            query.execute()
            return True
        except Exception as e:
            logger.error("Failed to update Supabase", table=table, error=str(e))
            return False

    async def delete(self, table: str, filters: dict[str, Any]) -> bool:
        """Delete records from a table.

        Args:
            table: Table name
            filters: Dictionary of column=value filters

        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_initialized():
            return False

        if not self._client:
            return False

        try:
            query = self._client.table(table).delete()

            for column, value in filters.items():
                query = query.eq(column, value)

            query.execute()
            return True
        except Exception as e:
            logger.error("Failed to delete from Supabase", table=table, error=str(e))
            return False


async def get_supabase() -> Optional[SupabaseClient]:
    """Get the global Supabase client instance.

    Returns:
        SupabaseClient instance or None if not configured
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    # Try to get from config
    try:
        from src.config import get_settings
        settings = get_settings()
        url = settings.supabase_url
        key = settings.supabase_service_key.get_secret_value() if settings.supabase_service_key else None
    except Exception:
        # Fallback to environment variables
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        logger.debug("Supabase not configured, audit persistence disabled")
        return None

    _supabase_client = SupabaseClient(url, key)
    return _supabase_client


def is_supabase_configured() -> bool:
    """Check if Supabase is configured.

    Returns:
        True if Supabase URL and key are set
    """
    try:
        from src.config import get_settings
        settings = get_settings()
        return bool(settings.supabase_url and settings.supabase_service_key)
    except Exception:
        return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))
