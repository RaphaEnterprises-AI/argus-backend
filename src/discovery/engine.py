"""Main Discovery Intelligence Engine for autonomous app exploration.

This module provides the core orchestration for the discovery process, including:
- Managing discovery sessions
- Coordinating crawling operations
- AI-powered flow inference
- Real-time progress streaming via SSE
- Database persistence (Supabase) with in-memory fallback
"""

import asyncio
import json
import re
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import anthropic
import structlog

from src.config import get_settings
from src.discovery.crawlers.crawlee_bridge import CrawleeBridge, CrawlProgress
from src.discovery.models import (
    CrawlResult,
    DiscoveredFlow,
    DiscoveredPage,
    DiscoveryConfig,
    DiscoverySession,
    DiscoveryStatus,
    FlowCategory,
    FlowStep,
    PageCategory,
)
from src.discovery.repository import DiscoveryRepository

logger = structlog.get_logger()


class DiscoveryError(Exception):
    """Exception raised when discovery operations fail."""

    pass


class DiscoveryEngine:
    """Main Discovery Intelligence Engine.

    Orchestrates the autonomous discovery process including crawling,
    element extraction, flow inference, and result persistence.

    Uses DiscoveryRepository for all database operations, providing:
    - Database-first persistence (Supabase as source of truth)
    - Write-through caching for performance
    - Automatic retry logic for transient failures
    - Graceful degradation to cache when DB unavailable

    Example:
        engine = DiscoveryEngine(supabase_client=supabase)

        session = await engine.start_discovery(
            project_id="proj-123",
            app_url="https://example.com",
        )

        # Stream events
        async for event in engine.discovery_events(session.id):
            print(f"Event: {event['type']}")

    Attributes:
        bridge: CrawleeBridge for crawling operations
        repository: DiscoveryRepository for data persistence
        current_session: Currently active discovery session
    """

    def __init__(
        self,
        supabase_client=None,
        use_crawlee: bool = False,
        repository: DiscoveryRepository | None = None,
    ):
        """Initialize the Discovery Engine.

        Args:
            supabase_client: Optional Supabase client for database persistence.
                            If not provided, uses in-memory storage only.
            use_crawlee: Whether to use Crawlee for crawling (requires Node.js)
            repository: Optional DiscoveryRepository instance. If not provided,
                       one will be created using the supabase_client.
        """
        self.bridge = CrawleeBridge(use_crawlee=use_crawlee)
        self.current_session: DiscoverySession | None = None
        self.log = logger.bind(component="discovery_engine")

        # Initialize repository for persistence
        # Can be injected for testing or use the default with supabase_client
        self.repository = repository or DiscoveryRepository(supabase_client=supabase_client)

        # Keep supabase reference for backward compatibility with other operations
        self.supabase = supabase_client

        # Event streaming
        self._event_subscribers: dict[str, list[asyncio.Queue]] = {}
        self._cancellation_flags: dict[str, bool] = {}

        # Initialize Claude client for flow inference
        settings = get_settings()
        api_key = settings.anthropic_api_key
        if api_key:
            if hasattr(api_key, "get_secret_value"):
                api_key = api_key.get_secret_value()
            self.claude = anthropic.Anthropic(api_key=api_key)
        else:
            self.claude = None
            self.log.warning("Anthropic API key not configured, flow inference disabled")

    async def start_discovery(
        self,
        project_id: str,
        app_url: str,
        config: DiscoveryConfig | None = None,
    ) -> DiscoverySession:
        """Start a new discovery session.

        Creates a new session, initializes it in the database, and begins
        the discovery crawl process.

        Args:
            project_id: ID of the project to discover
            app_url: Base URL of the application to explore
            config: Optional discovery configuration

        Returns:
            DiscoverySession with session details

        Raises:
            DiscoveryError: If session creation fails
        """
        self.log.info(
            "Starting discovery session",
            project_id=project_id,
            app_url=app_url,
        )

        # Use default config if not provided
        if config is None:
            config = DiscoveryConfig()

        # Create session
        session_id = str(uuid.uuid4())
        session = DiscoverySession(
            id=session_id,
            project_id=project_id,
            status=DiscoveryStatus.running,
            mode=config.mode,
            strategy=config.strategy,
            config=config,
            progress_percentage=0.0,
            started_at=datetime.now(UTC),
        )

        # Store session using repository (database-first)
        await self.repository.save_session(session)
        self.current_session = session
        self._cancellation_flags[session_id] = False

        # Emit start event
        await self._emit_event(
            session_id,
            "start",
            {
                "session_id": session_id,
                "project_id": project_id,
                "app_url": app_url,
                "config": config.to_dict(),
            },
        )

        # Run discovery in background
        asyncio.create_task(
            self._run_discovery_with_error_handling(session, app_url, config)
        )

        return session

    async def _run_discovery_with_error_handling(
        self,
        session: DiscoverySession,
        app_url: str,
        config: DiscoveryConfig,
    ) -> None:
        """Run discovery with error handling wrapper.

        Args:
            session: Discovery session
            app_url: Application URL
            config: Discovery configuration
        """
        try:
            await self.run_discovery(session, app_url, config)
        except Exception as e:
            self.log.exception(
                "Discovery failed",
                session_id=session.id,
                error=str(e),
            )
            session.status = DiscoveryStatus.failed
            session.completed_at = datetime.now(UTC)
            await self.repository.save_session(session)
            await self._emit_event(
                session.id,
                "error",
                {"error": str(e), "error_type": type(e).__name__},
            )

    async def run_discovery(
        self,
        session: DiscoverySession,
        app_url: str,
        config: DiscoveryConfig,
    ) -> CrawlResult:
        """Execute the discovery crawl.

        Coordinates the crawling process, tracks progress, saves results,
        and infers user flows.

        Args:
            session: Discovery session to update
            config: Discovery configuration

        Returns:
            CrawlResult with discovered pages and metadata
        """
        self.log.info(
            "Running discovery crawl",
            session_id=session.id,
            app_url=app_url,
        )

        # Set up progress tracking
        def on_progress(progress: CrawlProgress):
            asyncio.create_task(
                self._update_progress(
                    session.id,
                    progress.progress_percent,
                    progress.current_url,
                )
            )

        self.bridge.set_progress_callback(on_progress)

        # Run the crawl
        result = await self.bridge.run_crawl(app_url, config)

        # Check for cancellation
        if self._cancellation_flags.get(session.id, False):
            self.log.info("Discovery cancelled", session_id=session.id)
            session.status = DiscoveryStatus.cancelled
            session.completed_at = datetime.now(UTC)
            await self.repository.save_session(session)
            await self._emit_event(session.id, "cancelled", {})
            return result

        # Save discovered pages using repository
        pages = list(result.pages.values())
        await self.repository.save_pages(session.id, session.project_id, pages)

        session.pages_found = len(pages)
        session.elements_found = sum(len(p.elements) for p in pages)

        # Emit individual page discovery events
        for page in pages:
            await self._emit_event(
                session.id,
                "page_discovered",
                {
                    "page_id": page.id,
                    "url": page.url,
                    "title": page.title,
                    "category": page.category.value,
                    "elements_count": len(page.elements),
                },
            )

        await self._emit_event(
            session.id,
            "pages_discovered",
            {
                "count": len(pages),
                "elements_count": session.elements_found,
            },
        )

        # Infer user flows using AI
        flows = await self._infer_flows(pages)
        await self.repository.save_flows(session.id, session.project_id, flows)

        session.flows_found = len(flows)

        await self._emit_event(
            session.id,
            "flows_inferred",
            {
                "count": len(flows),
                "flows": [
                    {
                        "id": f.id,
                        "name": f.name,
                        "category": f.category.value,
                    }
                    for f in flows
                ],
            },
        )

        # Update session as completed
        session.status = DiscoveryStatus.completed
        session.progress_percentage = 100.0
        session.completed_at = datetime.now(UTC)
        await self.repository.save_session(session)

        await self._emit_event(
            session.id,
            "complete",
            {
                "pages_found": session.pages_found,
                "flows_found": session.flows_found,
                "elements_found": session.elements_found,
                "duration_ms": result.duration_ms,
                "errors": len(result.errors),
            },
        )

        self.log.info(
            "Discovery completed",
            session_id=session.id,
            pages=session.pages_found,
            flows=session.flows_found,
        )

        return result

    async def pause_discovery(self, session_id: str) -> bool:
        """Pause a running discovery.

        Args:
            session_id: ID of the session to pause

        Returns:
            True if successfully paused
        """
        session = await self.get_session_status(session_id)
        if not session:
            raise DiscoveryError(f"Session not found: {session_id}")

        if session.status != DiscoveryStatus.running:
            raise DiscoveryError(f"Session is not running: {session.status.value}")

        # Set cancellation flag (crawler will check this)
        self._cancellation_flags[session_id] = True

        session.status = DiscoveryStatus.paused
        await self.repository.save_session(session)

        await self._emit_event(
            session_id,
            "paused",
            {"paused_at": datetime.now(UTC).isoformat()},
        )

        self.log.info("Discovery paused", session_id=session_id)
        return True

    async def resume_discovery(self, session_id: str) -> DiscoverySession:
        """Resume a paused discovery.

        Args:
            session_id: ID of the session to resume

        Returns:
            Updated DiscoverySession

        Raises:
            DiscoveryError: If session cannot be resumed
        """
        session = await self.get_session_status(session_id)
        if not session:
            raise DiscoveryError(f"Session not found: {session_id}")

        if session.status != DiscoveryStatus.paused:
            raise DiscoveryError(f"Session is not paused: {session.status.value}")

        # Clear cancellation flag
        self._cancellation_flags[session_id] = False

        session.status = DiscoveryStatus.running
        await self.repository.save_session(session)

        await self._emit_event(
            session_id,
            "resumed",
            {"resumed_at": datetime.now(UTC).isoformat()},
        )

        # TODO: Resume crawl from last checkpoint
        # For now, this is a placeholder - full implementation would
        # require tracking crawl state and resuming from checkpoints

        self.log.info("Discovery resumed", session_id=session_id)
        return session

    async def cancel_discovery(self, session_id: str) -> bool:
        """Cancel a running discovery.

        Args:
            session_id: ID of the session to cancel

        Returns:
            True if successfully cancelled
        """
        session = await self.get_session_status(session_id)
        if not session:
            raise DiscoveryError(f"Session not found: {session_id}")

        if session.status not in [DiscoveryStatus.running, DiscoveryStatus.paused]:
            raise DiscoveryError(f"Session cannot be cancelled: {session.status.value}")

        # Set cancellation flag
        self._cancellation_flags[session_id] = True

        session.status = DiscoveryStatus.cancelled
        session.completed_at = datetime.now(UTC)
        await self.repository.save_session(session)

        await self._emit_event(
            session_id,
            "cancelled",
            {"cancelled_at": datetime.now(UTC).isoformat()},
        )

        self.log.info("Discovery cancelled", session_id=session_id)
        return True

    async def get_session_status(self, session_id: str) -> DiscoverySession | None:
        """Get current session status.

        Uses the repository for database-first access with caching.

        Args:
            session_id: ID of the session

        Returns:
            DiscoverySession or None if not found
        """
        return await self.repository.get_session(session_id)

    async def get_session_pages(
        self,
        session_id: str,
    ) -> list[DiscoveredPage]:
        """Get pages discovered in a session.

        Uses the repository for database-first access with caching.

        Args:
            session_id: ID of the session

        Returns:
            List of DiscoveredPage objects
        """
        return await self.repository.get_pages(session_id)

    async def get_session_flows(
        self,
        session_id: str,
    ) -> list[DiscoveredFlow]:
        """Get flows discovered in a session.

        Uses the repository for database-first access with caching.

        Args:
            session_id: ID of the session

        Returns:
            List of DiscoveredFlow objects
        """
        return await self.repository.get_flows(session_id)

    # Note: _save_session, _save_pages, _save_flows methods removed.
    # All persistence is now handled by self.repository (DiscoveryRepository)
    # which provides database-first storage with write-through caching.

    async def _infer_flows(
        self,
        pages: list[DiscoveredPage],
    ) -> list[DiscoveredFlow]:
        """Use AI to infer user flows from discovered pages.

        Analyzes page structure, elements, and relationships to identify
        common user journeys and flows.

        Args:
            pages: List of discovered pages

        Returns:
            List of inferred DiscoveredFlow objects
        """
        if not pages:
            return []

        if not self.claude:
            self.log.warning("Claude client not available, skipping flow inference")
            return self._infer_flows_heuristic(pages)

        return await self._infer_flows_with_ai(pages)

    async def _infer_flows_with_ai(
        self,
        pages: list[DiscoveredPage],
    ) -> list[DiscoveredFlow]:
        """Use Claude to infer user flows from pages.

        Args:
            pages: List of discovered pages

        Returns:
            List of DiscoveredFlow objects
        """
        # Prepare page summaries for Claude
        page_summaries = []
        for page in pages:
            auth_elements = [
                e
                for e in page.elements
                if "login" in (e.label or "").lower()
                or "password" in (e.html_attributes.get("type") or "").lower()
            ]
            form_elements = [e for e in page.elements if e.tag_name in ["input", "textarea", "select"]]
            button_elements = [e for e in page.elements if e.tag_name == "button"]

            page_summaries.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "category": page.category.value,
                    "has_auth": len(auth_elements) > 0,
                    "form_fields": len(form_elements),
                    "buttons": len(button_elements),
                    "outgoing_links_count": len(page.outgoing_links),
                }
            )

        prompt = f"""Analyze these discovered web pages and identify user flows (journeys).

DISCOVERED PAGES:
{json.dumps(page_summaries, indent=2)}

Identify complete user flows. For each flow:
1. Give it a clear name
2. Describe what the user accomplishes
3. List the steps (which pages, what actions)
4. Assign a category and priority

Respond with JSON only (no markdown):
{{
    "flows": [
        {{
            "id": "flow-1",
            "name": "User Login Flow",
            "description": "User authenticates to access protected content",
            "category": "authentication|registration|navigation|search|crud|checkout|profile|settings|social|admin",
            "priority": 1-10,
            "start_url": "/login",
            "pages": ["/login", "/dashboard"],
            "steps": [
                {{"order": 1, "page_url": "/login", "action": "navigate", "description": "Go to login page"}},
                {{"order": 2, "page_url": "/login", "action": "fill", "target": "email input", "description": "Enter email"}},
                {{"order": 3, "page_url": "/login", "action": "fill", "target": "password input", "description": "Enter password"}},
                {{"order": 4, "page_url": "/login", "action": "click", "target": "submit button", "description": "Click login"}}
            ],
            "success_criteria": ["Redirected to dashboard", "User name visible"],
            "complexity_score": 0.5
        }}
    ]
}}

Prioritize:
- Authentication flows (highest priority)
- Core business flows
- Settings/profile
- Edge cases (lowest)
"""

        try:
            from src.core.model_registry import get_model_id

            model = get_model_id("claude-sonnet-4-5")

            response = self.claude.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            data = self._parse_json_response(content)

            flows = []
            for f_data in data.get("flows", []):
                steps = []
                for s_data in f_data.get("steps", []):
                    steps.append(
                        FlowStep(
                            order=s_data.get("order", 0),
                            page_url=s_data.get("page_url", ""),
                            action=s_data.get("action", ""),
                            element_selector=s_data.get("target"),
                            expected_result=s_data.get("description"),
                        )
                    )

                try:
                    category = FlowCategory(f_data.get("category", "navigation"))
                except ValueError:
                    category = FlowCategory.navigation

                flow = DiscoveredFlow(
                    id=f_data.get("id", str(uuid.uuid4())),
                    name=f_data.get("name", ""),
                    description=f_data.get("description"),
                    category=category,
                    priority=f_data.get("priority", 5),
                    start_url=f_data.get("start_url"),
                    pages=f_data.get("pages", []),
                    steps=steps,
                    success_criteria=f_data.get("success_criteria", []),
                    complexity_score=f_data.get("complexity_score", 0.5),
                )
                flows.append(flow)

            self.log.info("AI flow inference completed", flows_count=len(flows))
            return flows

        except Exception as e:
            self.log.warning("AI flow inference failed, using heuristics", error=str(e))
            return self._infer_flows_heuristic(pages)

    def _infer_flows_heuristic(
        self,
        pages: list[DiscoveredPage],
    ) -> list[DiscoveredFlow]:
        """Infer flows using heuristic rules when AI is not available.

        Args:
            pages: List of discovered pages

        Returns:
            List of DiscoveredFlow objects
        """
        flows = []

        # Find login flow
        login_pages = [
            p
            for p in pages
            if p.category in [PageCategory.auth_login, PageCategory.auth_signup]
        ]
        if login_pages:
            login_page = login_pages[0]
            flows.append(
                DiscoveredFlow(
                    id=str(uuid.uuid4()),
                    name="User Authentication",
                    description="User login flow",
                    category=FlowCategory.authentication,
                    priority=1,
                    start_url=login_page.url,
                    pages=[login_page.url],
                    steps=[
                        FlowStep(
                            order=1,
                            page_url=login_page.url,
                            action="navigate",
                        ),
                        FlowStep(
                            order=2,
                            page_url=login_page.url,
                            action="fill",
                            expected_result="Enter credentials",
                        ),
                        FlowStep(
                            order=3,
                            page_url=login_page.url,
                            action="click",
                            expected_result="Submit login",
                        ),
                    ],
                )
            )

        # Find registration flow
        signup_pages = [p for p in pages if p.category == PageCategory.auth_signup]
        if signup_pages:
            signup_page = signup_pages[0]
            flows.append(
                DiscoveredFlow(
                    id=str(uuid.uuid4()),
                    name="User Registration",
                    description="New user signup flow",
                    category=FlowCategory.registration,
                    priority=2,
                    start_url=signup_page.url,
                    pages=[signup_page.url],
                )
            )

        # Find form submission flows
        form_pages = [p for p in pages if p.category == PageCategory.form]
        for form_page in form_pages[:3]:  # Limit to 3 form flows
            flows.append(
                DiscoveredFlow(
                    id=str(uuid.uuid4()),
                    name=f"Form: {form_page.title or 'Unknown'}",
                    description=f"Form submission on {form_page.url}",
                    category=FlowCategory.crud,
                    priority=5,
                    start_url=form_page.url,
                    pages=[form_page.url],
                )
            )

        # Find navigation flows
        landing_pages = [p for p in pages if p.category == PageCategory.landing]
        if landing_pages:
            flows.append(
                DiscoveredFlow(
                    id=str(uuid.uuid4()),
                    name="Main Navigation",
                    description="Primary site navigation flow",
                    category=FlowCategory.navigation,
                    priority=3,
                    start_url=landing_pages[0].url,
                    pages=[p.url for p in landing_pages[:5]],
                )
            )

        return flows

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response with error handling.

        Args:
            content: Raw LLM response content

        Returns:
            Parsed JSON dictionary
        """
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        content = content.strip()

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas
        content = re.sub(r",\s*([}\]])", r"\1", content)

        # Remove comments
        content = re.sub(r"//.*?$", "", content, flags=re.MULTILINE)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass

        self.log.warning("Failed to parse JSON from LLM", content_preview=content[:200])
        return {}

    async def _update_progress(
        self,
        session_id: str,
        progress: float,
        current_page: str,
    ) -> None:
        """Update session progress in database.

        Args:
            session_id: Session ID
            progress: Progress percentage (0-100)
            current_page: URL of current page being processed
        """
        # Use current_session for efficiency (avoid DB round-trip for progress updates)
        # Fall back to repository lookup if needed
        session = self.current_session if self.current_session and self.current_session.id == session_id else None
        if not session:
            session = await self.repository.get_session(session_id)

        if session:
            session.progress_percentage = progress
            session.current_page = current_page
            await self.repository.save_session(session)

        await self._emit_event(
            session_id,
            "progress",
            {
                "progress_percent": progress,
                "current_page": current_page,
            },
        )

    async def _emit_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Emit an event to all subscribers.

        Args:
            session_id: Session ID
            event_type: Type of event
            data: Event data
        """
        event = {
            "event": event_type,
            "data": {
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                **data,
            },
        }

        # Send to all subscribers
        subscribers = self._event_subscribers.get(session_id, [])
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.log.warning("Event queue full", session_id=session_id)

    async def discovery_events(
        self,
        session_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate SSE events for discovery progress.

        Creates an async generator that yields events as they occur
        during the discovery process.

        Args:
            session_id: Session ID to monitor

        Yields:
            Dictionary events with type and data
        """
        # Create queue for this subscriber
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Register subscriber
        if session_id not in self._event_subscribers:
            self._event_subscribers[session_id] = []
        self._event_subscribers[session_id].append(queue)

        try:
            # Check if session exists
            session = await self.get_session_status(session_id)
            if not session:
                yield {
                    "event": "error",
                    "data": {
                        "session_id": session_id,
                        "error": "Session not found",
                    },
                }
                return

            # Yield initial status
            yield {
                "event": "status",
                "data": {
                    "session_id": session_id,
                    "status": session.status.value,
                    "progress_percent": session.progress_percentage,
                    "pages_found": session.pages_found,
                    "flows_found": session.flows_found,
                },
            }

            # Stream events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event

                    # Check for completion events
                    if event.get("event") in ["complete", "cancelled", "error"]:
                        break

                except TimeoutError:
                    # Send keepalive
                    yield {
                        "event": "keepalive",
                        "data": {
                            "session_id": session_id,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    }

                    # Check session status
                    session = await self.get_session_status(session_id)
                    if session and session.status in [
                        DiscoveryStatus.completed,
                        DiscoveryStatus.failed,
                        DiscoveryStatus.cancelled,
                    ]:
                        yield {
                            "event": "status",
                            "data": {
                                "session_id": session_id,
                                "status": session.status.value,
                                "final": True,
                            },
                        }
                        break

        finally:
            # Unregister subscriber
            if session_id in self._event_subscribers:
                self._event_subscribers[session_id].remove(queue)
                if not self._event_subscribers[session_id]:
                    del self._event_subscribers[session_id]

    async def list_sessions(
        self,
        project_id: str | None = None,
        status: DiscoveryStatus | None = None,
        limit: int = 50,
    ) -> list[DiscoverySession]:
        """List discovery sessions with optional filtering.

        Uses the repository for database-first access with caching.

        Args:
            project_id: Filter by project ID
            status: Filter by status
            limit: Maximum number of sessions to return

        Returns:
            List of DiscoverySession objects
        """
        return await self.repository.list_sessions(
            project_id=project_id,
            status=status.value if status else None,
            limit=limit,
        )


# Factory function for convenience
def create_discovery_engine(supabase_client=None) -> DiscoveryEngine:
    """Create a DiscoveryEngine instance.

    Args:
        supabase_client: Optional Supabase client

    Returns:
        Configured DiscoveryEngine instance
    """
    return DiscoveryEngine(supabase_client=supabase_client)
