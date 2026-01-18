"""
Discovery Repository - Database-first persistence layer.

This module provides the DiscoveryRepository class that manages all database
operations for discovery sessions, pages, flows, and elements. It uses Supabase
as the primary source of truth with an in-memory cache for performance.

Key Design Decisions:
- Database-first: All reads prefer database, writes go to DB first
- Write-through cache: In-memory cache updated after successful DB writes
- Retry logic: Transient failures are retried with exponential backoff
- Graceful degradation: Falls back to cache if DB is temporarily unavailable
"""

import asyncio
import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

import structlog

from src.discovery.models import (
    DiscoveredElement,
    DiscoveredFlow,
    DiscoveredPage,
    DiscoverySession,
    DiscoveryStatus,
)

logger = structlog.get_logger()

T = TypeVar("T")


class RepositoryError(Exception):
    """Base exception for repository operations."""

    pass


class DatabaseUnavailableError(RepositoryError):
    """Raised when the database is temporarily unavailable."""

    pass


class RecordNotFoundError(RepositoryError):
    """Raised when a record is not found."""

    pass


class DiscoveryRepository:
    """
    Repository for discovery data persistence.

    Manages CRUD operations for discovery sessions, pages, flows, and elements.
    Uses Supabase as the primary data store with an in-memory cache layer.

    Attributes:
        supabase: Supabase client for database operations
        max_retries: Maximum retry attempts for transient failures
        retry_delay: Base delay between retries (exponential backoff)

    Example:
        repo = DiscoveryRepository(supabase_client)

        # Create a session
        await repo.save_session(session)

        # Get a session (from DB or cache)
        session = await repo.get_session(session_id)

        # List sessions with filtering
        sessions = await repo.list_sessions(project_id=project_id, status="running")
    """

    def __init__(
        self,
        supabase_client=None,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        """
        Initialize the repository.

        Args:
            supabase_client: Supabase client instance. If None, uses cache-only mode.
            max_retries: Maximum retry attempts for failed operations.
            retry_delay: Base delay in seconds between retries (exponential backoff).
        """
        self.supabase = supabase_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log = logger.bind(component="discovery_repository")

        # In-memory cache (write-through)
        self._session_cache: dict[str, DiscoverySession] = {}
        self._pages_cache: dict[str, list[DiscoveredPage]] = {}  # session_id -> pages
        self._flows_cache: dict[str, list[DiscoveredFlow]] = {}  # session_id -> flows
        self._elements_cache: dict[str, list[DiscoveredElement]] = {}  # page_id -> elements

        # Track database availability
        self._db_available = True
        self._last_db_error: str | None = None

    @property
    def is_database_available(self) -> bool:
        """Check if database is available."""
        return self.supabase is not None and self._db_available

    async def _retry_operation(
        self,
        operation: Callable,
        operation_name: str,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: Async callable to execute
            operation_name: Name for logging
            **kwargs: Arguments to pass to the operation

        Returns:
            Result from the operation

        Raises:
            DatabaseUnavailableError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await asyncio.to_thread(operation, **kwargs)
                self._db_available = True
                return result
            except Exception as e:
                last_error = e
                self.log.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries})",
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        self._db_available = False
        self._last_db_error = str(last_error)
        self.log.error(
            f"{operation_name} failed after all retries",
            error=str(last_error),
        )
        raise DatabaseUnavailableError(f"{operation_name} failed: {last_error}")

    # =========================================================================
    # Session Operations
    # =========================================================================

    async def save_session(self, session: DiscoverySession) -> DiscoverySession:
        """
        Save a discovery session to the database.

        Writes to database first, then updates cache on success.

        Args:
            session: The session to save

        Returns:
            The saved session

        Raises:
            DatabaseUnavailableError: If database save fails
        """
        # Prepare data for database
        data = self._session_to_db_record(session)

        if self.supabase:
            try:

                def do_upsert():
                    return self.supabase.table("discovery_sessions").upsert(data).execute()

                await self._retry_operation(do_upsert, "save_session")
            except DatabaseUnavailableError:
                # Log but continue - we'll update cache anyway
                self.log.warning("Database unavailable, session saved to cache only")

        # Update cache
        self._session_cache[session.id] = session

        return session

    async def get_session(self, session_id: str) -> DiscoverySession | None:
        """
        Get a session by ID.

        Tries database first, falls back to cache.

        Args:
            session_id: The session ID to retrieve

        Returns:
            The session or None if not found
        """
        # Try database first
        if self.supabase and self._db_available:
            try:

                def do_select():
                    return (
                        self.supabase.table("discovery_sessions")
                        .select("*")
                        .eq("id", session_id)
                        .single()
                        .execute()
                    )

                response = await self._retry_operation(do_select, "get_session")
                if response.data:
                    session = self._db_record_to_session(response.data)
                    # Update cache
                    self._session_cache[session_id] = session
                    return session
            except DatabaseUnavailableError:
                self.log.warning("Database unavailable, checking cache")
            except Exception as e:
                if "PGRST116" in str(e):  # No rows returned
                    return None
                self.log.warning("Failed to get session from DB", error=str(e))

        # Fall back to cache
        return self._session_cache.get(session_id)

    async def list_sessions(
        self,
        project_id: str | None = None,
        status: DiscoveryStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DiscoverySession]:
        """
        List sessions with optional filtering.

        Args:
            project_id: Filter by project ID
            status: Filter by status
            limit: Maximum results
            offset: Skip first N results

        Returns:
            List of matching sessions
        """
        # Try database first
        if self.supabase and self._db_available:
            try:

                def do_query():
                    query = self.supabase.table("discovery_sessions").select("*")

                    if project_id:
                        query = query.eq("project_id", project_id)
                    if status:
                        query = query.eq("status", status.value)

                    return (
                        query.order("started_at", desc=True)
                        .range(offset, offset + limit - 1)
                        .execute()
                    )

                response = await self._retry_operation(do_query, "list_sessions")
                if response.data:
                    sessions = [self._db_record_to_session(r) for r in response.data]
                    # Update cache
                    for s in sessions:
                        self._session_cache[s.id] = s
                    return sessions
            except DatabaseUnavailableError:
                self.log.warning("Database unavailable, using cache")

        # Fall back to cache
        sessions = list(self._session_cache.values())

        if project_id:
            sessions = [s for s in sessions if s.project_id == project_id]
        if status:
            sessions = [s for s in sessions if s.status == status]

        sessions.sort(key=lambda s: s.started_at or datetime.min, reverse=True)
        return sessions[offset : offset + limit]

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all related data.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        if self.supabase:
            try:

                def do_delete():
                    return (
                        self.supabase.table("discovery_sessions")
                        .delete()
                        .eq("id", session_id)
                        .execute()
                    )

                await self._retry_operation(do_delete, "delete_session")
            except DatabaseUnavailableError:
                self.log.warning("Database unavailable during delete")

        # Clear from cache
        self._session_cache.pop(session_id, None)
        self._pages_cache.pop(session_id, None)
        self._flows_cache.pop(session_id, None)

        return True

    # =========================================================================
    # Page Operations
    # =========================================================================

    async def save_pages(
        self,
        session_id: str,
        project_id: str,
        pages: list[DiscoveredPage],
    ) -> list[DiscoveredPage]:
        """
        Save discovered pages to the database.

        Args:
            session_id: Parent session ID
            project_id: Parent project ID
            pages: List of pages to save

        Returns:
            The saved pages
        """
        if not pages:
            return pages

        # Prepare records
        records = []
        for page in pages:
            record = self._page_to_db_record(page, session_id, project_id)
            records.append(record)

        if self.supabase:
            try:

                def do_upsert():
                    return self.supabase.table("discovered_pages").upsert(records).execute()

                await self._retry_operation(do_upsert, "save_pages")

                # Also save elements
                all_elements = []
                for page in pages:
                    for element in page.elements:
                        elem_record = self._element_to_db_record(
                            element, page.id, session_id
                        )
                        all_elements.append(elem_record)

                if all_elements:

                    def do_upsert_elements():
                        return (
                            self.supabase.table("discovered_elements")
                            .upsert(all_elements)
                            .execute()
                        )

                    await self._retry_operation(do_upsert_elements, "save_elements")

            except DatabaseUnavailableError:
                self.log.warning("Database unavailable, pages saved to cache only")

        # Update cache
        self._pages_cache[session_id] = pages

        return pages

    async def get_pages(self, session_id: str) -> list[DiscoveredPage]:
        """
        Get all pages for a session.

        Args:
            session_id: Session ID

        Returns:
            List of discovered pages
        """
        # Try database first
        if self.supabase and self._db_available:
            try:

                def do_select():
                    return (
                        self.supabase.table("discovered_pages")
                        .select("*")
                        .eq("discovery_session_id", session_id)
                        .execute()
                    )

                response = await self._retry_operation(do_select, "get_pages")
                if response.data:
                    pages = [self._db_record_to_page(r) for r in response.data]

                    # Fetch elements for each page
                    for page in pages:
                        elements = await self._get_elements_for_page(page.id)
                        page.elements = elements

                    # Update cache
                    self._pages_cache[session_id] = pages
                    return pages
            except DatabaseUnavailableError:
                self.log.warning("Database unavailable, using cache")

        # Fall back to cache
        return self._pages_cache.get(session_id, [])

    async def _get_elements_for_page(self, page_id: str) -> list[DiscoveredElement]:
        """Get elements for a specific page."""
        if self.supabase and self._db_available:
            try:

                def do_select():
                    return (
                        self.supabase.table("discovered_elements")
                        .select("*")
                        .eq("page_id", page_id)
                        .execute()
                    )

                response = await self._retry_operation(do_select, "get_elements")
                if response.data:
                    return [self._db_record_to_element(r) for r in response.data]
            except DatabaseUnavailableError:
                pass

        return self._elements_cache.get(page_id, [])

    # =========================================================================
    # Flow Operations
    # =========================================================================

    async def save_flows(
        self,
        session_id: str,
        project_id: str,
        flows: list[DiscoveredFlow],
    ) -> list[DiscoveredFlow]:
        """
        Save discovered flows to the database.

        Args:
            session_id: Parent session ID
            project_id: Parent project ID
            flows: List of flows to save

        Returns:
            The saved flows
        """
        if not flows:
            return flows

        # Prepare records
        records = []
        for flow in flows:
            record = self._flow_to_db_record(flow, session_id, project_id)
            records.append(record)

        if self.supabase:
            try:

                def do_upsert():
                    return self.supabase.table("discovered_flows").upsert(records).execute()

                await self._retry_operation(do_upsert, "save_flows")
            except DatabaseUnavailableError:
                self.log.warning("Database unavailable, flows saved to cache only")

        # Update cache
        self._flows_cache[session_id] = flows

        return flows

    async def get_flows(self, session_id: str) -> list[DiscoveredFlow]:
        """
        Get all flows for a session.

        Args:
            session_id: Session ID

        Returns:
            List of discovered flows
        """
        # Try database first
        if self.supabase and self._db_available:
            try:

                def do_select():
                    return (
                        self.supabase.table("discovered_flows")
                        .select("*")
                        .eq("discovery_session_id", session_id)
                        .execute()
                    )

                response = await self._retry_operation(do_select, "get_flows")
                if response.data:
                    flows = [self._db_record_to_flow(r) for r in response.data]
                    # Update cache
                    self._flows_cache[session_id] = flows
                    return flows
            except DatabaseUnavailableError:
                self.log.warning("Database unavailable, using cache")

        # Fall back to cache
        return self._flows_cache.get(session_id, [])

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def get_session_with_data(
        self, session_id: str
    ) -> dict[str, Any] | None:
        """
        Get a session with all its pages and flows.

        Args:
            session_id: Session ID

        Returns:
            Dict with session, pages, and flows, or None if not found
        """
        session = await self.get_session(session_id)
        if not session:
            return None

        pages = await self.get_pages(session_id)
        flows = await self.get_flows(session_id)

        return {
            "session": session,
            "pages": pages,
            "flows": flows,
        }

    async def refresh_cache_from_db(self, project_id: str | None = None) -> int:
        """
        Refresh in-memory cache from database.

        Args:
            project_id: Optional project to refresh (None = all)

        Returns:
            Number of sessions refreshed
        """
        if not self.supabase:
            return 0

        try:
            sessions = await self.list_sessions(project_id=project_id, limit=1000)

            for session in sessions:
                await self.get_pages(session.id)
                await self.get_flows(session.id)

            self.log.info(
                "Cache refreshed from database",
                sessions_count=len(sessions),
            )
            return len(sessions)
        except Exception as e:
            self.log.error("Failed to refresh cache", error=str(e))
            return 0

    # =========================================================================
    # Data Conversion Helpers
    # =========================================================================

    def _session_to_db_record(self, session: DiscoverySession) -> dict[str, Any]:
        """Convert session dataclass to database record."""
        data = session.to_dict()

        # Ensure datetime fields are ISO strings
        if data.get("started_at") and hasattr(data["started_at"], "isoformat"):
            data["started_at"] = data["started_at"].isoformat()
        if data.get("completed_at") and hasattr(data["completed_at"], "isoformat"):
            data["completed_at"] = data["completed_at"].isoformat()

        # Map fields to DB column names
        db_record = {
            "id": data["id"],
            "project_id": data["project_id"],
            "name": f"Discovery {data['id'][:8]}",  # Generate name
            "status": data["status"],
            "start_url": data.get("current_page", ""),
            "mode": data["mode"],
            "strategy": data["strategy"],
            "config": data["config"],
            "progress_percentage": int(data.get("progress_percentage", 0)),
            "current_page": data.get("current_page"),
            "pages_discovered": data.get("pages_found", 0),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
        }

        return db_record

    def _db_record_to_session(self, record: dict[str, Any]) -> DiscoverySession:
        """Convert database record to session dataclass."""
        # Map DB column names back to dataclass fields
        data = {
            "id": record["id"],
            "project_id": record["project_id"],
            "status": record.get("status", "pending"),
            "mode": record.get("mode", "standard_crawl"),
            "strategy": record.get("strategy", "breadth_first"),
            "config": record.get("config"),
            "progress_percentage": float(record.get("progress_percentage", 0)),
            "current_page": record.get("current_page"),
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
            "pages_found": record.get("pages_discovered", 0),
            "flows_found": 0,  # Not stored in DB
            "elements_found": 0,  # Not stored in DB
        }

        return DiscoverySession.from_dict(data)

    def _page_to_db_record(
        self, page: DiscoveredPage, session_id: str, project_id: str
    ) -> dict[str, Any]:
        """Convert page dataclass to database record."""
        # Generate URL hash for deduplication
        url_hash = hashlib.sha256(page.url.encode()).hexdigest()[:16]

        return {
            "id": page.id,
            "discovery_session_id": session_id,
            "url": page.url,
            "url_hash": url_hash,
            "title": page.title,
            "page_type": self._map_page_category_to_type(page.category.value),
            "category": page.category.value,
            "element_count": len(page.elements),
            "interactive_element_count": len(
                [e for e in page.elements if e.category.value in ("form", "action", "navigation")]
            ),
            "outgoing_links": list(page.outgoing_links),
            "incoming_links": list(page.incoming_links),
            "depth_from_start": page.depth,
            "importance_score": page.importance_score * 100,  # Scale to 0-100
            "risk_score": page.risk_score * 100,
            "coverage_score": page.coverage_score * 100,
            "requires_auth": page.requires_auth,
            "load_time_ms": page.load_time_ms,
            "screenshot_url": None,  # Screenshot stored separately
            "elements_summary": {
                "categories": self._summarize_element_categories(page.elements)
            },
        }

    def _map_page_category_to_type(self, category: str) -> str:
        """Map page category to database page_type enum."""
        mapping = {
            "landing": "landing",
            "auth_login": "auth",
            "auth_signup": "auth",
            "auth_reset": "auth",
            "dashboard": "dashboard",
            "list": "list",
            "detail": "detail",
            "form": "form",
            "settings": "settings",
            "profile": "content",
            "checkout": "form",
            "error": "error",
            "other": "unknown",
        }
        return mapping.get(category, "unknown")

    def _summarize_element_categories(
        self, elements: list[DiscoveredElement]
    ) -> dict[str, int]:
        """Summarize element categories for storage."""
        summary: dict[str, int] = {}
        for element in elements:
            cat = element.category.value
            summary[cat] = summary.get(cat, 0) + 1
        return summary

    def _db_record_to_page(self, record: dict[str, Any]) -> DiscoveredPage:
        """Convert database record to page dataclass."""
        from src.discovery.models import PageCategory

        # Map page_type back to category
        category_str = record.get("category", record.get("page_type", "other"))
        try:
            category = PageCategory(category_str)
        except ValueError:
            category = PageCategory.other

        return DiscoveredPage(
            id=record["id"],
            url=record["url"],
            title=record.get("title"),
            description=None,
            category=category,
            elements=[],  # Elements loaded separately
            outgoing_links=set(record.get("outgoing_links", [])),
            incoming_links=set(record.get("incoming_links", [])),
            importance_score=float(record.get("importance_score", 50)) / 100,
            coverage_score=float(record.get("coverage_score", 0)) / 100,
            risk_score=float(record.get("risk_score", 50)) / 100,
            depth=record.get("depth_from_start", 0),
            screenshot_base64=None,
            dom_snapshot_url=record.get("dom_snapshot_url"),
            load_time_ms=record.get("load_time_ms"),
            requires_auth=record.get("requires_auth", False),
        )

    def _element_to_db_record(
        self, element: DiscoveredElement, page_id: str, session_id: str
    ) -> dict[str, Any]:
        """Convert element dataclass to database record."""
        return {
            "id": element.id,
            "page_id": page_id,
            "discovery_session_id": session_id,
            "selector": element.selector,
            "xpath": element.xpath,
            "tag_name": element.tag_name,
            "category": self._map_element_category(element.category.value),
            "purpose": element.purpose,
            "label": element.label,
            "bounds": element.bounds.to_dict() if element.bounds else None,
            "is_visible": element.is_visible,
            "is_enabled": element.is_enabled,
            "is_required": element.is_required,
            "aria_label": element.aria_label,
            "role": element.role,
            "importance_score": element.importance_score * 100,
            "stability_score": element.stability_score * 100,
            "alternative_selectors": element.alternative_selectors,
            "html_attributes": element.html_attributes,
        }

    def _map_element_category(self, category: str) -> str:
        """Map element category to database enum."""
        mapping = {
            "navigation": "link",
            "form": "input",
            "action": "button",
            "content": "other",
            "interactive": "custom",
            "authentication": "input",
            "commerce": "button",
            "social": "button",
        }
        return mapping.get(category, "other")

    def _db_record_to_element(self, record: dict[str, Any]) -> DiscoveredElement:
        """Convert database record to element dataclass."""
        from src.discovery.models import ElementBounds, ElementCategory

        # Map DB category back
        category_str = record.get("category", "content")
        reverse_mapping = {
            "button": "action",
            "link": "navigation",
            "input": "form",
            "select": "form",
            "checkbox": "form",
            "radio": "form",
            "textarea": "form",
            "form": "form",
            "navigation": "navigation",
            "modal": "interactive",
            "menu": "navigation",
            "table": "content",
            "list": "content",
            "image": "content",
            "video": "content",
            "custom": "interactive",
            "other": "content",
        }
        mapped_category = reverse_mapping.get(category_str, "content")

        try:
            category = ElementCategory(mapped_category)
        except ValueError:
            category = ElementCategory.content

        bounds = None
        if record.get("bounds"):
            bounds = ElementBounds.from_dict(record["bounds"])

        return DiscoveredElement(
            id=record["id"],
            page_url="",  # Not stored in DB
            selector=record["selector"],
            xpath=record.get("xpath"),
            category=category,
            purpose=record.get("purpose"),
            label=record.get("label"),
            bounds=bounds,
            importance_score=float(record.get("importance_score", 50)) / 100,
            stability_score=float(record.get("stability_score", 50)) / 100,
            alternative_selectors=record.get("alternative_selectors", []),
            tag_name=record.get("tag_name", "div"),
            html_attributes=record.get("html_attributes", {}),
            is_visible=record.get("is_visible", True),
            is_enabled=record.get("is_enabled", True),
            is_required=record.get("is_required", False),
            aria_label=record.get("aria_label"),
            role=record.get("role"),
        )

    def _flow_to_db_record(
        self, flow: DiscoveredFlow, session_id: str, project_id: str
    ) -> dict[str, Any]:
        """Convert flow dataclass to database record."""
        # Map flow category to DB enum
        flow_type_mapping = {
            "authentication": "authentication",
            "registration": "registration",
            "navigation": "navigation",
            "search": "search",
            "crud": "crud",
            "checkout": "checkout",
            "profile": "form_submission",
            "settings": "form_submission",
            "social": "custom",
            "admin": "custom",
        }

        return {
            "id": flow.id,
            "discovery_session_id": session_id,
            "name": flow.name,
            "description": flow.description,
            "flow_type": flow_type_mapping.get(flow.category.value, "custom"),
            "category": flow.category.value,
            "steps": [s.to_dict() for s in flow.steps],
            "entry_points": [{"url": flow.start_url}] if flow.start_url else [],
            "success_criteria": {"checks": flow.success_criteria},
            "failure_indicators": {"checks": flow.failure_indicators},
            "complexity_score": flow.complexity_score * 100,
            "business_value_score": flow.business_value_score * 100,
            "confidence_score": flow.confidence_score * 100,
            "validated": flow.validated,
        }

    def _db_record_to_flow(self, record: dict[str, Any]) -> DiscoveredFlow:
        """Convert database record to flow dataclass."""
        from src.discovery.models import FlowCategory, FlowStep

        # Map back flow type
        category_str = record.get("category", record.get("flow_type", "navigation"))
        try:
            category = FlowCategory(category_str)
        except ValueError:
            category = FlowCategory.navigation

        # Parse steps
        steps_data = record.get("steps", [])
        steps = [FlowStep.from_dict(s) for s in steps_data]

        # Parse entry points
        entry_points = record.get("entry_points", [])
        start_url = entry_points[0].get("url") if entry_points else None

        # Parse criteria
        success_criteria = record.get("success_criteria", {})
        failure_indicators = record.get("failure_indicators", {})

        return DiscoveredFlow(
            id=record["id"],
            name=record["name"],
            description=record.get("description"),
            category=category,
            priority=5,  # Default
            start_url=start_url,
            pages=list(record.get("page_ids", [])) if record.get("page_ids") else [],
            steps=steps,
            success_criteria=success_criteria.get("checks", []),
            failure_indicators=failure_indicators.get("checks", []),
            complexity_score=float(record.get("complexity_score", 50)) / 100,
            business_value_score=float(record.get("business_value_score", 50)) / 100,
            confidence_score=float(record.get("confidence_score", 50)) / 100,
            validated=record.get("validated", False),
        )

    # =========================================================================
    # Health & Diagnostics
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """
        Check repository health status.

        Returns:
            Dict with health status information
        """
        status = {
            "database_configured": self.supabase is not None,
            "database_available": self._db_available,
            "last_error": self._last_db_error,
            "cache_stats": {
                "sessions": len(self._session_cache),
                "pages": sum(len(p) for p in self._pages_cache.values()),
                "flows": sum(len(f) for f in self._flows_cache.values()),
            },
        }

        # Try a simple query to verify connection
        if self.supabase:
            try:

                def do_ping():
                    return (
                        self.supabase.table("discovery_sessions")
                        .select("id")
                        .limit(1)
                        .execute()
                    )

                await asyncio.to_thread(do_ping)
                status["database_available"] = True
                self._db_available = True
            except Exception as e:
                status["database_available"] = False
                status["ping_error"] = str(e)

        return status
