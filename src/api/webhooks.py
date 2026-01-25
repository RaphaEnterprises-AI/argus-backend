"""Webhook handlers for observability platforms.

Handles incoming webhooks from:
- Sentry (error tracking)
- Datadog (monitoring)
- FullStory (session replay)
- LogRocket (frontend monitoring)
- NewRelic (APM)
- Bugsnag (error tracking)
- Rollbar (error tracking)

These webhooks create production_events in Supabase for Quality Intelligence analysis.
"""

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.services.supabase_client import get_supabase_client
from src.services.vectorize import index_production_event

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/webhooks", tags=["Webhooks"])


# =============================================================================
# Pydantic Models
# =============================================================================

class SentryException(BaseModel):
    type: str | None = None
    value: str | None = None


class SentryMetadata(BaseModel):
    type: str | None = None
    value: str | None = None
    filename: str | None = None
    function: str | None = None


class SentryIssue(BaseModel):
    id: str
    title: str
    culprit: str | None = None
    level: str = "error"
    message: str | None = None
    metadata: SentryMetadata | None = None
    platform: str | None = None
    project: str | None = None
    url: str | None = None
    shortId: str | None = None
    count: str = "1"
    userCount: int = 1
    firstSeen: str | None = None
    lastSeen: str | None = None
    tags: list[dict[str, str]] = Field(default_factory=list)


class SentryWebhookPayload(BaseModel):
    action: str
    data: dict[str, Any]
    installation: dict | None = None
    actor: dict | None = None


class DatadogError(BaseModel):
    type: str | None = None
    message: str | None = None
    stack: str | None = None
    source: str | None = None


class DatadogView(BaseModel):
    url: str | None = None
    name: str | None = None


class DatadogEvent(BaseModel):
    id: str | None = None
    event_type: str | None = None
    title: str
    message: str
    date_happened: int | None = None
    priority: str = "normal"
    host: str | None = None
    tags: list[str] = Field(default_factory=list)
    alert_type: str = "error"
    source_type_name: str | None = None
    aggregation_key: str | None = None
    url: str | None = None
    error: DatadogError | None = None
    view: DatadogView | None = None
    user: dict | None = None
    context: dict | None = None


class ProductionEvent(BaseModel):
    """Normalized production event for storage."""
    project_id: str
    source: Literal["sentry", "datadog", "fullstory", "logrocket", "newrelic", "bugsnag", "rollbar"]
    external_id: str
    external_url: str | None = None
    event_type: Literal["error", "exception", "performance", "session", "rage_click", "dead_click"]
    severity: Literal["fatal", "error", "warning", "info"]
    title: str
    message: str | None = None
    stack_trace: str | None = None
    fingerprint: str
    url: str | None = None
    component: str | None = None
    browser: str | None = None
    os: str | None = None
    device_type: Literal["desktop", "mobile", "tablet"] | None = None
    occurrence_count: int = 1
    affected_users: int = 1
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    status: Literal["new", "processing", "resolved", "ignored"] = "new"
    raw_payload: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    success: bool
    message: str
    event_id: str | None = None
    fingerprint: str | None = None


# =============================================================================
# Helper Functions
# =============================================================================

def parse_severity(level: str) -> Literal["fatal", "error", "warning", "info"]:
    """Parse severity level to normalized format."""
    level_lower = level.lower()
    if level_lower == "fatal":
        return "fatal"
    elif level_lower in ("error", "high"):
        return "error"
    elif level_lower in ("warning", "warn", "normal"):
        return "warning"
    return "info"


def extract_component_from_stack(stack_trace: str | None) -> str | None:
    """Extract component name from stack trace."""
    if not stack_trace:
        return None

    # React component pattern
    react_match = re.search(r"at\s+([A-Z][a-zA-Z0-9]*)\s+\(", stack_trace)
    if react_match:
        return react_match.group(1)

    # Vue component pattern
    vue_match = re.search(r"VueComponent\.([a-zA-Z0-9_]+)", stack_trace)
    if vue_match:
        return vue_match.group(1)

    # Angular component pattern
    angular_match = re.search(r"([A-Z][a-zA-Z0-9]*Component)\.", stack_trace)
    if angular_match:
        return angular_match.group(1)

    return None


def generate_fingerprint(
    error_type: str,
    message: str,
    component: str | None,
    url: str | None,
) -> str:
    """Generate a fingerprint for error grouping."""
    parts = [error_type, message[:100] if message else ""]
    if component:
        parts.append(component)
    if url:
        # Normalize URL by removing query params and IDs
        # NOTE: UUID normalization MUST come before numeric ID normalization
        # to prevent UUIDs starting with digits (e.g., "11111111-...") from being
        # partially matched by the numeric pattern first.
        normalized_url = re.sub(r"\?.*$", "", url)
        normalized_url = re.sub(r"/[a-f0-9-]{36}", "/:uuid", normalized_url)
        normalized_url = re.sub(r"/\d+", "/:id", normalized_url)
        parts.append(normalized_url)

    combined = "|".join(parts)
    hash_value = hashlib.sha256(combined.encode()).hexdigest()[:12]
    return hash_value


async def get_default_project_id(supabase, organization_id: str) -> str | None:
    """Get the first available project ID for a specific organization.

    SECURITY: Always filter by organization_id to prevent cross-tenant data access.
    """
    if not organization_id:
        return None

    result = await supabase.request(
        f"/projects?organization_id=eq.{organization_id}&select=id&limit=1"
    )
    if result.get("data") and len(result["data"]) > 0:
        return result["data"][0]["id"]
    return None


async def validate_project_org(supabase, project_id: str, organization_id: str) -> bool:
    """Validate that a project belongs to the specified organization.

    SECURITY: Prevents IDOR by ensuring users can only access their own org's projects.
    """
    if not project_id or not organization_id:
        return False

    result = await supabase.request(
        f"/projects?id=eq.{project_id}&organization_id=eq.{organization_id}&select=id"
    )
    return bool(result.get("data") and len(result["data"]) > 0)


async def log_webhook(
    supabase,
    webhook_id: str,
    source: str,
    request: Request,
    body: dict,
    status: str = "processing",
) -> None:
    """Log incoming webhook for debugging."""
    await supabase.insert(
        "webhook_logs",
        {
            "id": webhook_id,
            "source": source,
            "method": request.method,
            "headers": dict(request.headers),
            "body": body,
            "status": status,
        },
    )


async def update_webhook_log(
    supabase, webhook_id: str, status: str, error_message: str | None = None, event_id: str | None = None
) -> None:
    """Update webhook log status."""
    update_data = {"status": status, "processed_at": datetime.now(UTC).isoformat()}
    if error_message:
        update_data["error_message"] = error_message
    if event_id:
        update_data["processed_event_id"] = event_id
    await supabase.update("webhook_logs", {"id": f"eq.{webhook_id}"}, update_data)


# =============================================================================
# Sentry Webhook
# =============================================================================

@router.post("/sentry", response_model=WebhookResponse)
async def handle_sentry_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Sentry webhook events.

    Sentry sends webhooks for new issues, resolved issues, etc.
    We normalize these into production_events for Quality Intelligence.

    SECURITY: organization_id is required to ensure data is stored in the correct tenant.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if project_id is provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "sentry", request, body)

        payload = SentryWebhookPayload(**body)
        action = payload.action
        data = payload.data or {}  # Ensure data is never None

        # Handle issue creation/triggering
        if action in ("issue", "created", "triggered"):
            issue_data = data.get("issue", {}) or {}
            event_data = data.get("event", {}) or {}

            if not issue_data and not event_data:
                raise HTTPException(status_code=400, detail="No issue or event data")

            # Extract error details
            exception_values = event_data.get("event", {}).get("exception", {}).get("values", [])
            error_details = exception_values[0] if exception_values else {}

            stack_frames = error_details.get("stacktrace", {}).get("frames", [])
            stack_trace = "\n".join(
                f"  at {f.get('function', 'anonymous')} ({f.get('filename')}:{f.get('lineno')}:{f.get('colno')})"
                for f in reversed(stack_frames)
            ) if stack_frames else None

            page_url = (
                event_data.get("event", {}).get("request", {}).get("url")
                or event_data.get("url")
            )
            component = extract_component_from_stack(stack_trace)
            title = issue_data.get("title") or error_details.get("type") or "Unknown Error"
            message = issue_data.get("message") or error_details.get("value") or ""

            fingerprint = generate_fingerprint(
                error_details.get("type", "Error"),
                message,
                component,
                page_url,
            )

            # Determine device type from contexts
            contexts = event_data.get("event", {}).get("contexts", {})
            device_type = None
            if contexts.get("device", {}).get("family"):
                family = contexts["device"]["family"].lower()
                if "iphone" in family or "android" in family:
                    device_type = "mobile"
                elif "ipad" in family or "tablet" in family:
                    device_type = "tablet"
                else:
                    device_type = "desktop"

            # Get project ID (SECURITY: filtered by organization_id)
            if not project_id:
                project_id = await get_default_project_id(supabase, organization_id)
                if not project_id:
                    raise HTTPException(
                        status_code=400,
                        detail="No project found. Please specify project_id query parameter.",
                    )

            # Create production event
            browser_info = contexts.get("browser", {})
            os_info = contexts.get("os", {})

            # Parse occurrence count safely
            try:
                occurrence_count = int(issue_data.get("count", 1))
            except (ValueError, TypeError):
                occurrence_count = 1

            # Parse tags safely - ensure each tag has key and value
            raw_tags = issue_data.get("tags", []) or []
            tags = []
            for t in raw_tags:
                if isinstance(t, dict) and "key" in t and "value" in t:
                    tags.append(f"{t['key']}:{t['value']}")

            production_event = ProductionEvent(
                project_id=project_id,
                source="sentry",
                external_id=issue_data.get("id") or event_data.get("event_id") or str(uuid.uuid4()),
                external_url=issue_data.get("url") or event_data.get("issue_url"),
                event_type="error",
                severity=parse_severity(issue_data.get("level", "error")),
                title=title,
                message=message,
                stack_trace=stack_trace,
                fingerprint=fingerprint,
                url=page_url,
                component=component,
                browser=f"{browser_info.get('name', '')} {browser_info.get('version', '')}".strip() or None,
                os=f"{os_info.get('name', '')} {os_info.get('version', '')}".strip() or None,
                device_type=device_type,
                occurrence_count=occurrence_count,
                affected_users=issue_data.get("userCount", 1),
                first_seen_at=issue_data.get("firstSeen"),
                last_seen_at=issue_data.get("lastSeen"),
                status="new",
                raw_payload=body,
                tags=tags,
                metadata={
                    "sentry_project": event_data.get("project_name") or issue_data.get("project"),
                    "sentry_platform": event_data.get("platform") or issue_data.get("platform"),
                    "sentry_short_id": issue_data.get("shortId"),
                },
            )

            result = await supabase.insert("production_events", production_event.model_dump())

            if result.get("error"):
                await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
                raise HTTPException(status_code=500, detail="Failed to process event")

            event_id = result["data"][0]["id"] if result["data"] else None
            await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

            # Auto-index for semantic search
            if result.get("data"):
                await index_production_event(result["data"][0])

            logger.info("Sentry webhook processed", event_id=event_id, fingerprint=fingerprint)

            return WebhookResponse(
                success=True,
                message="Event processed successfully",
                event_id=event_id,
                fingerprint=fingerprint,
            )

        # Handle resolved/ignored status updates
        if action in ("resolved", "ignored"):
            issue = data.get("issue", {})
            if issue.get("id"):
                await supabase.update(
                    "production_events",
                    {"external_id": f"eq.{issue['id']}", "source": "eq.sentry"},
                    {
                        "status": "resolved" if action == "resolved" else "ignored",
                        "resolved_at": datetime.now(UTC).isoformat(),
                    },
                )
            await update_webhook_log(supabase, webhook_id, "processed")
            return WebhookResponse(success=True, message=f"Issue {action}")

        await update_webhook_log(supabase, webhook_id, "processed")
        return WebhookResponse(success=True, message="Webhook received")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Sentry webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Datadog Webhook
# =============================================================================

@router.post("/datadog", response_model=WebhookResponse)
async def handle_datadog_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Datadog webhook events.

    Datadog sends webhooks for alerts, errors, and monitoring events.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "datadog", request, body)

        events = body if isinstance(body, list) else [body]

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        processed_events = []

        for event_data in events:
            event = DatadogEvent(**event_data)

            # Determine event type
            event_type: Literal["error", "exception", "performance"] = "error"
            if event.error and event.error.type:
                event_type = "exception"
            elif event.event_type and ("performance" in event.event_type or (event.source_type_name and "apm" in event.source_type_name)):
                event_type = "performance"

            # Determine severity
            severity: Literal["fatal", "error", "warning", "info"] = "error"
            if event.alert_type == "error" or event.priority == "high":
                severity = "error"
            elif event.alert_type == "warning" or event.priority == "normal":
                severity = "warning"
            elif event.alert_type == "info" or event.priority == "low":
                severity = "info"

            page_url = (
                (event.view.url if event.view else None)
                or (event.error.source if event.error else None)
                or event.url
            )
            component = extract_component_from_stack(event.error.stack if event.error else None)

            fingerprint = generate_fingerprint(
                event.error.type if event.error else (event.event_type or "Error"),
                event.error.message if event.error else event.message,
                component,
                page_url,
            )

            production_event = ProductionEvent(
                project_id=project_id,
                source="datadog",
                external_id=event.id or event.aggregation_key or str(uuid.uuid4()),
                external_url=event.url,
                event_type=event_type,
                severity=severity,
                title=event.title,
                message=event.error.message if event.error else event.message,
                stack_trace=event.error.stack if event.error else None,
                fingerprint=fingerprint,
                url=page_url,
                component=component,
                occurrence_count=1,
                affected_users=1,
                status="new",
                raw_payload=event_data,
                tags=event.tags,
                metadata={
                    "datadog_host": event.host,
                    "datadog_source": event.source_type_name,
                    "datadog_priority": event.priority,
                    "datadog_view_name": event.view.name if event.view else None,
                },
            )

            result = await supabase.insert("production_events", production_event.model_dump())
            if not result.get("error") and result.get("data"):
                processed_events.append(result["data"][0]["id"])
                # Auto-index for semantic search
                await index_production_event(result["data"][0])

        await update_webhook_log(supabase, webhook_id, "processed")

        logger.info("Datadog webhook processed", event_count=len(processed_events))

        return WebhookResponse(
            success=True,
            message=f"Processed {len(processed_events)} events",
            event_id=processed_events[0] if processed_events else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Datadog webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FullStory Webhook
# =============================================================================

@router.post("/fullstory", response_model=WebhookResponse)
async def handle_fullstory_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle FullStory webhook events.

    FullStory sends webhooks for rage clicks, dead clicks, and error events.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "fullstory", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        # FullStory webhook structure varies by event type
        event_type_raw = body.get("type") or body.get("event_type") or "error"
        event_type: Literal["error", "rage_click", "dead_click", "session"] = "error"
        if "rage" in event_type_raw.lower():
            event_type = "rage_click"
        elif "dead" in event_type_raw.lower():
            event_type = "dead_click"

        # Handle both nested and flat payload structures
        session_data = body.get("session", {})
        page_data = body.get("page", {})
        element_data = body.get("element", {})

        session_url = (session_data.get("url") if isinstance(session_data, dict) else None) or body.get("sessionUrl") or body.get("session_url")
        page_url = (page_data.get("url") if isinstance(page_data, dict) else None) or body.get("pageUrl") or body.get("page_url")
        element_selector = (element_data.get("selector") if isinstance(element_data, dict) else element_data) or body.get("selector") or body.get("element")

        title = body.get("title") or f"FullStory {event_type.replace('_', ' ').title()}"
        message = body.get("message") or element_selector or ""

        fingerprint = generate_fingerprint(event_type, message, element_selector, page_url)

        production_event = ProductionEvent(
            project_id=project_id,
            source="fullstory",
            external_id=body.get("id") or str(uuid.uuid4()),
            external_url=session_url,
            event_type=event_type,
            severity="warning" if event_type in ("rage_click", "dead_click") else "error",
            title=title,
            message=message,
            fingerprint=fingerprint,
            url=page_url,
            component=element_selector,
            occurrence_count=body.get("count", 1),
            affected_users=body.get("userCount", 1),
            status="new",
            raw_payload=body,
            metadata={
                "fullstory_session_url": session_url,
                "element_selector": element_selector,
            },
        )

        result = await supabase.insert("production_events", production_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to process event")

        event_id = result["data"][0]["id"] if result["data"] else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        # Auto-index for semantic search
        if result.get("data"):
            await index_production_event(result["data"][0])

        logger.info("FullStory webhook processed", event_id=event_id)

        return WebhookResponse(
            success=True,
            message="Event processed successfully",
            event_id=event_id,
            fingerprint=fingerprint,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("FullStory webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LogRocket Webhook
# =============================================================================

@router.post("/logrocket", response_model=WebhookResponse)
async def handle_logrocket_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle LogRocket webhook events.

    LogRocket sends webhooks for errors and session events.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "logrocket", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        error = body.get("error", {})
        session = body.get("session", {})

        title = error.get("type") or error.get("name") or body.get("title") or "LogRocket Error"
        message = error.get("message") or body.get("message") or ""
        stack_trace = error.get("stack") or error.get("stackTrace")
        page_url = session.get("url") or body.get("url")
        component = extract_component_from_stack(stack_trace)

        fingerprint = generate_fingerprint(
            error.get("type", "Error"),
            message,
            component,
            page_url,
        )

        production_event = ProductionEvent(
            project_id=project_id,
            source="logrocket",
            external_id=body.get("id") or session.get("id") or str(uuid.uuid4()),
            external_url=session.get("sessionUrl"),
            event_type="error",
            severity=parse_severity(body.get("severity", "error")),
            title=title,
            message=message,
            stack_trace=stack_trace,
            fingerprint=fingerprint,
            url=page_url,
            component=component,
            browser=session.get("browser"),
            os=session.get("os"),
            occurrence_count=body.get("count", 1),
            affected_users=1,
            status="new",
            raw_payload=body,
            metadata={
                "logrocket_session_url": session.get("sessionUrl"),
                "logrocket_app_id": body.get("appId"),
            },
        )

        result = await supabase.insert("production_events", production_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to process event")

        event_id = result["data"][0]["id"] if result["data"] else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        # Auto-index for semantic search
        if result.get("data"):
            await index_production_event(result["data"][0])

        logger.info("LogRocket webhook processed", event_id=event_id)

        return WebhookResponse(
            success=True,
            message="Event processed successfully",
            event_id=event_id,
            fingerprint=fingerprint,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LogRocket webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NewRelic Webhook
# =============================================================================

@router.post("/newrelic", response_model=WebhookResponse)
async def handle_newrelic_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle NewRelic webhook events.

    NewRelic sends webhooks for APM alerts and errors.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "newrelic", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        # NewRelic can send different payload formats
        incident = body.get("incident", body)
        title = incident.get("incident_title") or incident.get("condition_name") or body.get("title") or "NewRelic Alert"
        message = incident.get("details") or body.get("message") or ""

        # Extract URL from details or target
        targets = incident.get("targets", [])
        page_url = targets[0].get("link") if targets else body.get("url")

        fingerprint = generate_fingerprint("newrelic_alert", message, None, page_url)

        # Map NewRelic priority to severity
        priority = incident.get("priority", "HIGH")
        severity: Literal["fatal", "error", "warning", "info"] = "error"
        if priority == "CRITICAL":
            severity = "fatal"
        elif priority == "HIGH":
            severity = "error"
        elif priority == "MEDIUM":
            severity = "warning"
        else:
            severity = "info"

        production_event = ProductionEvent(
            project_id=project_id,
            source="newrelic",
            external_id=str(incident.get("incident_id") or body.get("id") or uuid.uuid4()),
            external_url=incident.get("incident_url"),
            event_type="error",
            severity=severity,
            title=title,
            message=message,
            fingerprint=fingerprint,
            url=page_url,
            occurrence_count=1,
            affected_users=1,
            status="new",
            raw_payload=body,
            metadata={
                "newrelic_account_id": incident.get("account_id"),
                "newrelic_condition_name": incident.get("condition_name"),
                "newrelic_policy_name": incident.get("policy_name"),
            },
        )

        result = await supabase.insert("production_events", production_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to process event")

        event_id = result["data"][0]["id"] if result["data"] else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        # Auto-index for semantic search
        if result.get("data"):
            await index_production_event(result["data"][0])

        logger.info("NewRelic webhook processed", event_id=event_id)

        return WebhookResponse(
            success=True,
            message="Event processed successfully",
            event_id=event_id,
            fingerprint=fingerprint,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("NewRelic webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Bugsnag Webhook
# =============================================================================

@router.post("/bugsnag", response_model=WebhookResponse)
async def handle_bugsnag_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Bugsnag webhook events.

    Bugsnag sends webhooks for new errors and regressions.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "bugsnag", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        error = body.get("error", {})
        trigger = body.get("trigger", {})

        title = error.get("errorClass") or error.get("exceptionClass") or body.get("title") or "Bugsnag Error"
        message = error.get("message") or ""

        # Extract stack trace
        stacktrace = error.get("stacktrace", [])
        stack_trace = "\n".join(
            f"  at {frame.get('method', 'anonymous')} ({frame.get('file')}:{frame.get('lineNumber')})"
            for frame in stacktrace
        ) if stacktrace else None

        page_url = error.get("context") or error.get("url")
        component = extract_component_from_stack(stack_trace)

        fingerprint = generate_fingerprint(
            error.get("errorClass", "Error"),
            message,
            component,
            page_url,
        )

        production_event = ProductionEvent(
            project_id=project_id,
            source="bugsnag",
            external_id=error.get("id") or str(uuid.uuid4()),
            external_url=error.get("url"),
            event_type="error",
            severity=parse_severity(error.get("severity", "error")),
            title=title,
            message=message,
            stack_trace=stack_trace,
            fingerprint=fingerprint,
            url=page_url,
            component=component,
            occurrence_count=error.get("eventsCount", 1),
            affected_users=error.get("usersCount", 1),
            first_seen_at=error.get("firstSeen"),
            last_seen_at=error.get("lastSeen"),
            status="new",
            raw_payload=body,
            metadata={
                "bugsnag_project_id": body.get("project", {}).get("id"),
                "bugsnag_project_name": body.get("project", {}).get("name"),
                "bugsnag_trigger": trigger.get("type"),
            },
        )

        result = await supabase.insert("production_events", production_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to process event")

        event_id = result["data"][0]["id"] if result["data"] else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        # Auto-index for semantic search
        if result.get("data"):
            await index_production_event(result["data"][0])

        logger.info("Bugsnag webhook processed", event_id=event_id)

        return WebhookResponse(
            success=True,
            message="Event processed successfully",
            event_id=event_id,
            fingerprint=fingerprint,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Bugsnag webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Rollbar Webhook
# =============================================================================

# =============================================================================
# GitHub Actions CI/CD Webhook
# =============================================================================

class GitHubWorkflowRun(BaseModel):
    """GitHub Actions workflow run data."""
    id: int
    name: str
    head_branch: str
    head_sha: str
    status: str  # queued, in_progress, completed
    conclusion: str | None = None  # success, failure, cancelled, skipped
    html_url: str
    created_at: str
    updated_at: str
    run_number: int
    workflow_id: int


class GitHubRepository(BaseModel):
    """GitHub repository data."""
    id: int
    name: str
    full_name: str
    html_url: str
    default_branch: str = "main"


class GitHubActionsPayload(BaseModel):
    """GitHub Actions webhook payload."""
    action: str  # requested, completed, in_progress
    workflow_run: dict | None = None
    workflow_job: dict | None = None
    repository: dict
    sender: dict | None = None


class CIEvent(BaseModel):
    """Normalized CI/CD event for storage."""
    project_id: str
    source: Literal["github_actions", "gitlab_ci", "circleci", "jenkins"]
    external_id: str
    external_url: str | None = None
    event_type: Literal["workflow_run", "workflow_job", "test_run", "coverage_report"]
    status: Literal["pending", "running", "success", "failure", "cancelled", "skipped"]
    workflow_name: str
    branch: str
    commit_sha: str
    run_number: int
    duration_seconds: int | None = None
    test_results: dict | None = None  # {passed: X, failed: Y, skipped: Z}
    coverage_percent: float | None = None
    raw_payload: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


@router.post("/github-actions", response_model=WebhookResponse)
async def handle_github_actions_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle GitHub Actions webhook events.

    GitHub sends webhooks for workflow runs and jobs.
    Configure in GitHub: Settings → Webhooks → Add webhook
    Events: workflow_run, workflow_job

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "github_actions", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        payload = GitHubActionsPayload(**body)
        action = payload.action
        repo = payload.repository

        # Handle workflow_run events
        if payload.workflow_run:
            run = payload.workflow_run

            # Map GitHub status/conclusion to our status
            status: Literal["pending", "running", "success", "failure", "cancelled", "skipped"] = "pending"
            if run.get("status") == "completed":
                conclusion = run.get("conclusion", "").lower()
                if conclusion == "success":
                    status = "success"
                elif conclusion in ("failure", "timed_out"):
                    status = "failure"
                elif conclusion == "cancelled":
                    status = "cancelled"
                elif conclusion == "skipped":
                    status = "skipped"
            elif run.get("status") == "in_progress":
                status = "running"

            # Calculate duration if completed
            duration_seconds = None
            if run.get("updated_at") and run.get("created_at"):
                try:
                    from datetime import datetime as dt
                    created = dt.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                    updated = dt.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                    duration_seconds = int((updated - created).total_seconds())
                except Exception:
                    pass

            ci_event = CIEvent(
                project_id=project_id,
                source="github_actions",
                external_id=str(run.get("id")),
                external_url=run.get("html_url"),
                event_type="workflow_run",
                status=status,
                workflow_name=run.get("name", "Unknown"),
                branch=run.get("head_branch", "unknown"),
                commit_sha=run.get("head_sha", ""),
                run_number=run.get("run_number", 0),
                duration_seconds=duration_seconds,
                raw_payload=body,
                metadata={
                    "repository": repo.get("full_name"),
                    "workflow_id": run.get("workflow_id"),
                    "event": run.get("event"),
                    "actor": run.get("actor", {}).get("login") if run.get("actor") else None,
                    "conclusion": run.get("conclusion"),
                },
            )

            result = await supabase.insert("ci_events", ci_event.model_dump())

            if result.get("error"):
                await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
                raise HTTPException(status_code=500, detail="Failed to store CI event")

            event_id = result["data"][0]["id"] if result.get("data") else None
            await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

            logger.info(
                "GitHub Actions webhook processed",
                event_id=event_id,
                workflow=run.get("name"),
                status=status,
            )

            return WebhookResponse(
                success=True,
                message=f"Workflow run {action}: {run.get('name')} ({status})",
                event_id=event_id,
            )

        # Handle workflow_job events
        if payload.workflow_job:
            job = payload.workflow_job
            await update_webhook_log(supabase, webhook_id, "processed")

            return WebhookResponse(
                success=True,
                message=f"Workflow job {action}: {job.get('name')}",
            )

        await update_webhook_log(supabase, webhook_id, "processed")
        return WebhookResponse(success=True, message="Webhook received")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GitHub Actions webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Coverage Report Upload Endpoint
# =============================================================================

class CoverageReport(BaseModel):
    """Coverage report upload."""
    organization_id: str  # SECURITY: Required for multi-tenant isolation
    project_id: str | None = None
    branch: str = "main"
    commit_sha: str
    format: Literal["lcov", "istanbul", "cobertura", "clover"] = "lcov"
    report_data: str  # Raw coverage report content
    ci_run_id: str | None = None  # Link to CI event


class CoverageSummary(BaseModel):
    """Parsed coverage summary."""
    lines_total: int = 0
    lines_covered: int = 0
    lines_percent: float = 0.0
    branches_total: int = 0
    branches_covered: int = 0
    branches_percent: float = 0.0
    functions_total: int = 0
    functions_covered: int = 0
    functions_percent: float = 0.0
    files: list[dict] = Field(default_factory=list)


def parse_lcov(report_data: str) -> CoverageSummary:
    """Parse LCOV coverage report format."""
    lines_total = 0
    lines_covered = 0
    branches_total = 0
    branches_covered = 0
    functions_total = 0
    functions_covered = 0
    files = []

    current_file = None
    file_lines_total = 0
    file_lines_covered = 0

    for line in report_data.split("\n"):
        line = line.strip()

        if line.startswith("SF:"):
            current_file = line[3:]
            file_lines_total = 0
            file_lines_covered = 0
        elif line.startswith("LF:"):
            file_lines_total = int(line[3:])
            lines_total += file_lines_total
        elif line.startswith("LH:"):
            file_lines_covered = int(line[3:])
            lines_covered += file_lines_covered
        elif line.startswith("BRF:"):
            branches_total += int(line[4:])
        elif line.startswith("BRH:"):
            branches_covered += int(line[4:])
        elif line.startswith("FNF:"):
            functions_total += int(line[4:])
        elif line.startswith("FNH:"):
            functions_covered += int(line[4:])
        elif line == "end_of_record":
            if current_file:
                file_percent = (file_lines_covered / file_lines_total * 100) if file_lines_total > 0 else 0
                files.append({
                    "path": current_file,
                    "lines_total": file_lines_total,
                    "lines_covered": file_lines_covered,
                    "coverage_percent": round(file_percent, 2),
                })
            current_file = None

    return CoverageSummary(
        lines_total=lines_total,
        lines_covered=lines_covered,
        lines_percent=round(lines_covered / lines_total * 100, 2) if lines_total > 0 else 0,
        branches_total=branches_total,
        branches_covered=branches_covered,
        branches_percent=round(branches_covered / branches_total * 100, 2) if branches_total > 0 else 0,
        functions_total=functions_total,
        functions_covered=functions_covered,
        functions_percent=round(functions_covered / functions_total * 100, 2) if functions_total > 0 else 0,
        files=files,
    )


def parse_istanbul_json(report_data: str) -> CoverageSummary:
    """Parse Istanbul JSON coverage report format."""
    import json

    try:
        data = json.loads(report_data)
    except json.JSONDecodeError:
        return CoverageSummary()

    lines_total = 0
    lines_covered = 0
    branches_total = 0
    branches_covered = 0
    functions_total = 0
    functions_covered = 0
    files = []

    for file_path, file_data in data.items():
        # Handle both istanbul formats
        if isinstance(file_data, dict):
            # Standard istanbul format
            s = file_data.get("s", {})  # statements
            b = file_data.get("b", {})  # branches
            f = file_data.get("f", {})  # functions

            file_lines_total = len(s)
            file_lines_covered = sum(1 for v in s.values() if v > 0)
            file_branches_total = sum(len(br) for br in b.values())
            file_branches_covered = sum(sum(1 for v in br if v > 0) for br in b.values())
            file_functions_total = len(f)
            file_functions_covered = sum(1 for v in f.values() if v > 0)

            lines_total += file_lines_total
            lines_covered += file_lines_covered
            branches_total += file_branches_total
            branches_covered += file_branches_covered
            functions_total += file_functions_total
            functions_covered += file_functions_covered

            file_percent = (file_lines_covered / file_lines_total * 100) if file_lines_total > 0 else 0
            files.append({
                "path": file_path,
                "lines_total": file_lines_total,
                "lines_covered": file_lines_covered,
                "coverage_percent": round(file_percent, 2),
            })

    return CoverageSummary(
        lines_total=lines_total,
        lines_covered=lines_covered,
        lines_percent=round(lines_covered / lines_total * 100, 2) if lines_total > 0 else 0,
        branches_total=branches_total,
        branches_covered=branches_covered,
        branches_percent=round(branches_covered / branches_total * 100, 2) if branches_total > 0 else 0,
        functions_total=functions_total,
        functions_covered=functions_covered,
        functions_percent=round(functions_covered / functions_total * 100, 2) if functions_total > 0 else 0,
        files=files,
    )


@router.post("/coverage", response_model=WebhookResponse)
async def upload_coverage_report(
    report: CoverageReport,
):
    """
    Upload a coverage report.

    Supports LCOV, Istanbul JSON, Cobertura, and Clover formats.

    Example curl:
        curl -X POST /api/v1/webhooks/coverage \\
            -H "Content-Type: application/json" \\
            -d '{"organization_id": "...", "commit_sha": "abc123", "format": "lcov", "report_data": "..."}'

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if report.project_id:
        if not await validate_project_org(supabase, report.project_id, report.organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        project_id = report.project_id
        if not project_id:
            project_id = await get_default_project_id(supabase, report.organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id.",
                )

        # Parse coverage based on format
        if report.format == "lcov":
            summary = parse_lcov(report.report_data)
        elif report.format == "istanbul":
            summary = parse_istanbul_json(report.report_data)
        else:
            # For cobertura/clover, use basic parsing or return error
            raise HTTPException(
                status_code=400,
                detail=f"Format '{report.format}' not yet supported. Use 'lcov' or 'istanbul'.",
            )

        # Store coverage report
        coverage_data = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "branch": report.branch,
            "commit_sha": report.commit_sha,
            "ci_run_id": report.ci_run_id,
            "format": report.format,
            "lines_total": summary.lines_total,
            "lines_covered": summary.lines_covered,
            "lines_percent": summary.lines_percent,
            "branches_total": summary.branches_total,
            "branches_covered": summary.branches_covered,
            "branches_percent": summary.branches_percent,
            "functions_total": summary.functions_total,
            "functions_covered": summary.functions_covered,
            "functions_percent": summary.functions_percent,
            "files": summary.files,
            "created_at": datetime.now(UTC).isoformat(),
        }

        result = await supabase.insert("coverage_reports", coverage_data)

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to store coverage report")

        event_id = result["data"][0]["id"] if result.get("data") else coverage_data["id"]

        logger.info(
            "Coverage report uploaded",
            event_id=event_id,
            lines_percent=summary.lines_percent,
            files_count=len(summary.files),
        )

        return WebhookResponse(
            success=True,
            message=f"Coverage report processed: {summary.lines_percent}% line coverage",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Coverage upload error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Rollbar Webhook (existing)
# =============================================================================

@router.post("/rollbar", response_model=WebhookResponse)
async def handle_rollbar_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Rollbar webhook events.

    Rollbar sends webhooks for new errors and reactivated items.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "rollbar", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        event_name = body.get("event_name", "new_item")
        data = body.get("data", {})
        item = data.get("item", {})
        occurrence = data.get("occurrence", item.get("last_occurrence", {}))

        title = item.get("title") or occurrence.get("exception", {}).get("class") or "Rollbar Error"
        message = occurrence.get("exception", {}).get("message") or item.get("message") or ""

        # Extract stack trace
        frames = occurrence.get("exception", {}).get("frames", [])
        stack_trace = "\n".join(
            f"  at {frame.get('method', 'anonymous')} ({frame.get('filename')}:{frame.get('lineno')})"
            for frame in reversed(frames)
        ) if frames else None

        page_url = occurrence.get("request", {}).get("url") or occurrence.get("context")
        component = extract_component_from_stack(stack_trace)

        fingerprint = generate_fingerprint(
            occurrence.get("exception", {}).get("class", "Error"),
            message,
            component,
            page_url,
        )

        production_event = ProductionEvent(
            project_id=project_id,
            source="rollbar",
            external_id=str(item.get("id") or occurrence.get("id") or uuid.uuid4()),
            external_url=item.get("public_item_handle"),
            event_type="error",
            severity=parse_severity(item.get("level", "error")),
            title=title,
            message=message,
            stack_trace=stack_trace,
            fingerprint=fingerprint,
            url=page_url,
            component=component,
            browser=occurrence.get("client", {}).get("browser"),
            os=occurrence.get("client", {}).get("os"),
            occurrence_count=item.get("total_occurrences", 1),
            affected_users=item.get("unique_occurrences", 1),
            first_seen_at=datetime.fromtimestamp(item["first_occurrence_timestamp"]).isoformat() if item.get("first_occurrence_timestamp") else None,
            last_seen_at=datetime.fromtimestamp(item["last_occurrence_timestamp"]).isoformat() if item.get("last_occurrence_timestamp") else None,
            status="new",
            raw_payload=body,
            metadata={
                "rollbar_environment": item.get("environment"),
                "rollbar_framework": item.get("framework"),
                "rollbar_event": event_name,
            },
        )

        result = await supabase.insert("production_events", production_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to process event")

        event_id = result["data"][0]["id"] if result["data"] else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        # Auto-index for semantic search
        if result.get("data"):
            await index_production_event(result["data"][0])

        logger.info("Rollbar webhook processed", event_id=event_id)

        return WebhookResponse(
            success=True,
            message="Event processed successfully",
            event_id=event_id,
            fingerprint=fingerprint,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Rollbar webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GitLab CI/CD Webhook
# =============================================================================

class GitLabPipeline(BaseModel):
    """GitLab CI pipeline data."""
    id: int
    status: str  # pending, running, success, failed, canceled, skipped
    ref: str  # branch name
    sha: str
    source: str | None = None  # push, web, trigger, schedule, api
    duration: int | None = None


class GitLabProject(BaseModel):
    """GitLab project data."""
    id: int
    name: str
    path_with_namespace: str
    web_url: str


class GitLabPipelinePayload(BaseModel):
    """GitLab CI webhook payload for pipeline events."""
    object_kind: str  # pipeline
    object_attributes: dict
    project: dict
    user: dict | None = None
    commit: dict | None = None
    builds: list[dict] = Field(default_factory=list)


@router.post("/gitlab-ci", response_model=WebhookResponse)
async def handle_gitlab_ci_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle GitLab CI/CD webhook events.

    GitLab sends webhooks for pipeline and job events.
    Configure in GitLab: Settings → Webhooks → Add webhook
    Events: Pipeline events

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "gitlab_ci", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        object_kind = body.get("object_kind")

        if object_kind != "pipeline":
            await update_webhook_log(supabase, webhook_id, "processed")
            return WebhookResponse(success=True, message=f"Ignored event type: {object_kind}")

        payload = GitLabPipelinePayload(**body)
        pipeline = payload.object_attributes
        gitlab_project = payload.project

        # Map GitLab status to our status
        gitlab_status = pipeline.get("status", "").lower()
        status: Literal["pending", "running", "success", "failure", "cancelled", "skipped"] = "pending"
        if gitlab_status == "success":
            status = "success"
        elif gitlab_status in ("failed", "timeout"):
            status = "failure"
        elif gitlab_status in ("canceled", "cancelled"):
            status = "cancelled"
        elif gitlab_status == "skipped":
            status = "skipped"
        elif gitlab_status == "running":
            status = "running"

        # Get workflow name from builds or use pipeline ID
        workflow_name = "Pipeline"
        if payload.builds:
            stages = list(set(b.get("stage", "") for b in payload.builds))
            workflow_name = f"Pipeline ({', '.join(stages[:3])})"

        ci_event = CIEvent(
            project_id=project_id,
            source="gitlab_ci",
            external_id=str(pipeline.get("id")),
            external_url=f"{gitlab_project.get('web_url')}/-/pipelines/{pipeline.get('id')}",
            event_type="workflow_run",
            status=status,
            workflow_name=workflow_name,
            branch=pipeline.get("ref", "unknown"),
            commit_sha=pipeline.get("sha", ""),
            run_number=pipeline.get("id", 0),
            duration_seconds=pipeline.get("duration"),
            raw_payload=body,
            metadata={
                "repository": gitlab_project.get("path_with_namespace"),
                "source": pipeline.get("source"),
                "user": payload.user.get("username") if payload.user else None,
                "stages": [b.get("stage") for b in payload.builds],
                "failed_jobs": [b.get("name") for b in payload.builds if b.get("status") == "failed"],
            },
        )

        result = await supabase.insert("ci_events", ci_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to store CI event")

        event_id = result["data"][0]["id"] if result.get("data") else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        logger.info(
            "GitLab CI webhook processed",
            event_id=event_id,
            pipeline_id=pipeline.get("id"),
            status=status,
        )

        return WebhookResponse(
            success=True,
            message=f"Pipeline {pipeline.get('id')}: {status}",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("GitLab CI webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CircleCI Webhook
# =============================================================================

class CircleCIPipeline(BaseModel):
    """CircleCI pipeline data."""
    id: str
    number: int
    state: str  # created, pending, running, errored, failed, canceled, success
    created_at: str


class CircleCIWorkflow(BaseModel):
    """CircleCI workflow data."""
    id: str
    name: str
    status: str
    created_at: str
    stopped_at: str | None = None


class CircleCIPayload(BaseModel):
    """CircleCI webhook payload."""
    type: str  # workflow-completed, job-completed
    id: str
    happened_at: str
    webhook: dict | None = None
    project: dict | None = None
    organization: dict | None = None
    pipeline: dict | None = None
    workflow: dict | None = None
    job: dict | None = None


@router.post("/circleci", response_model=WebhookResponse)
async def handle_circleci_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle CircleCI webhook events.

    CircleCI sends webhooks for workflow and job completions.
    Configure in CircleCI: Project Settings → Webhooks → Add webhook

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "circleci", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        payload = CircleCIPayload(**body)
        event_type = payload.type

        # Only process workflow-completed events
        if event_type != "workflow-completed":
            await update_webhook_log(supabase, webhook_id, "processed")
            return WebhookResponse(success=True, message=f"Ignored event type: {event_type}")

        workflow = payload.workflow or {}
        pipeline = payload.pipeline or {}
        project_data = payload.project or {}

        # Map CircleCI status to our status
        circleci_status = workflow.get("status", "").lower()
        status: Literal["pending", "running", "success", "failure", "cancelled", "skipped"] = "pending"
        if circleci_status == "success":
            status = "success"
        elif circleci_status in ("failed", "error", "infrastructure_fail", "timedout"):
            status = "failure"
        elif circleci_status in ("canceled", "cancelled"):
            status = "cancelled"
        elif circleci_status == "not_run":
            status = "skipped"
        elif circleci_status == "running":
            status = "running"

        # Calculate duration
        duration_seconds = None
        if workflow.get("stopped_at") and workflow.get("created_at"):
            try:
                from datetime import datetime as dt
                created = dt.fromisoformat(workflow["created_at"].replace("Z", "+00:00"))
                stopped = dt.fromisoformat(workflow["stopped_at"].replace("Z", "+00:00"))
                duration_seconds = int((stopped - created).total_seconds())
            except Exception:
                pass

        # Get VCS info from pipeline
        vcs = pipeline.get("vcs", {})
        branch = vcs.get("branch", "unknown")
        commit_sha = vcs.get("revision", "")

        ci_event = CIEvent(
            project_id=project_id,
            source="circleci",
            external_id=workflow.get("id", payload.id),
            external_url=workflow.get("url"),
            event_type="workflow_run",
            status=status,
            workflow_name=workflow.get("name", "Workflow"),
            branch=branch,
            commit_sha=commit_sha,
            run_number=pipeline.get("number", 0),
            duration_seconds=duration_seconds,
            raw_payload=body,
            metadata={
                "repository": project_data.get("slug"),
                "pipeline_id": pipeline.get("id"),
                "trigger_type": pipeline.get("trigger", {}).get("type"),
                "vcs_provider": vcs.get("provider_name"),
            },
        )

        result = await supabase.insert("ci_events", ci_event.model_dump())

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to store CI event")

        event_id = result["data"][0]["id"] if result.get("data") else None
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        logger.info(
            "CircleCI webhook processed",
            event_id=event_id,
            workflow=workflow.get("name"),
            status=status,
        )

        return WebhookResponse(
            success=True,
            message=f"Workflow {workflow.get('name')}: {status}",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CircleCI webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Test Results Upload (JUnit XML)
# =============================================================================

class TestResultsUpload(BaseModel):
    """Test results upload request."""
    organization_id: str  # SECURITY: Required for multi-tenant isolation
    project_id: str | None = None
    branch: str = "main"
    commit_sha: str
    format: Literal["junit", "pytest", "jest"] = "junit"
    report_data: str  # XML/JSON content
    ci_run_id: str | None = None


class TestResultSummary(BaseModel):
    """Parsed test results summary."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errored: int = 0
    duration_seconds: float = 0.0
    test_suites: list[dict] = Field(default_factory=list)
    failures: list[dict] = Field(default_factory=list)


def parse_junit_xml(report_data: str) -> TestResultSummary:
    """Parse JUnit XML test results.

    Uses defusedxml to prevent XXE (XML External Entity) attacks.
    """
    import defusedxml.ElementTree as ET

    try:
        root = ET.fromstring(report_data)
    except ET.ParseError:
        return TestResultSummary()

    total = 0
    passed = 0
    failed = 0
    skipped = 0
    errored = 0
    duration = 0.0
    test_suites = []
    failures = []

    # Handle both <testsuites> and single <testsuite>
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    elif root.tag == "testsuite":
        suites = [root]
    else:
        suites = []

    for suite in suites:
        suite_name = suite.get("name", "Unknown")
        suite_tests = int(suite.get("tests", 0))
        suite_failures = int(suite.get("failures", 0))
        suite_errors = int(suite.get("errors", 0))
        suite_skipped = int(suite.get("skipped", 0))
        suite_time = float(suite.get("time", 0))

        total += suite_tests
        failed += suite_failures
        errored += suite_errors
        skipped += suite_skipped
        duration += suite_time

        test_suites.append({
            "name": suite_name,
            "tests": suite_tests,
            "failures": suite_failures,
            "errors": suite_errors,
            "skipped": suite_skipped,
            "time": suite_time,
        })

        # Extract failure details
        for testcase in suite.findall("testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")

            if failure is not None or error is not None:
                fail_elem = failure if failure is not None else error
                failures.append({
                    "test_name": testcase.get("name", "Unknown"),
                    "class_name": testcase.get("classname", ""),
                    "type": fail_elem.get("type", "AssertionError"),
                    "message": fail_elem.get("message", ""),
                    "details": (fail_elem.text or "")[:1000],  # Truncate
                })

    passed = total - failed - errored - skipped

    return TestResultSummary(
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errored=errored,
        duration_seconds=duration,
        test_suites=test_suites,
        failures=failures,
    )


def parse_pytest_json(report_data: str) -> TestResultSummary:
    """Parse pytest JSON test results."""
    import json

    try:
        data = json.loads(report_data)
    except json.JSONDecodeError:
        return TestResultSummary()

    summary = data.get("summary", {})
    tests = data.get("tests", [])

    failures = []
    for test in tests:
        if test.get("outcome") == "failed":
            call = test.get("call", {})
            failures.append({
                "test_name": test.get("nodeid", "Unknown"),
                "class_name": "",
                "type": "AssertionError",
                "message": call.get("longrepr", "")[:500],
                "details": call.get("longrepr", "")[:1000],
            })

    return TestResultSummary(
        total=summary.get("total", 0),
        passed=summary.get("passed", 0),
        failed=summary.get("failed", 0),
        skipped=summary.get("skipped", 0),
        errored=summary.get("error", 0),
        duration_seconds=data.get("duration", 0),
        failures=failures,
    )


@router.post("/test-results", response_model=WebhookResponse)
async def upload_test_results(
    results: TestResultsUpload,
):
    """
    Upload test results from CI/CD pipeline.

    Supports JUnit XML, pytest JSON formats.

    Example curl:
        curl -X POST /api/v1/webhooks/test-results \\
            -H "Content-Type: application/json" \\
            -d '{"organization_id": "...", "commit_sha": "abc123", "format": "junit", "report_data": "<testsuites>..."}'

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if results.project_id:
        if not await validate_project_org(supabase, results.project_id, results.organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        project_id = results.project_id
        if not project_id:
            project_id = await get_default_project_id(supabase, results.organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id.",
                )

        # Parse test results based on format
        if results.format == "junit":
            summary = parse_junit_xml(results.report_data)
        elif results.format == "pytest":
            summary = parse_pytest_json(results.report_data)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Format '{results.format}' not yet supported. Use 'junit' or 'pytest'.",
            )

        # Determine status based on results
        status: Literal["pending", "running", "success", "failure", "cancelled", "skipped"] = "success"
        if summary.failed > 0 or summary.errored > 0:
            status = "failure"
        elif summary.total == 0:
            status = "skipped"

        # Create CI event for test run
        ci_event = CIEvent(
            project_id=project_id,
            source="github_actions",  # Default, can be overridden
            external_id=f"test-{results.commit_sha[:8]}-{uuid.uuid4().hex[:8]}",
            event_type="test_run",
            status=status,
            workflow_name="Test Results",
            branch=results.branch,
            commit_sha=results.commit_sha,
            run_number=0,
            duration_seconds=int(summary.duration_seconds),
            test_results={
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "skipped": summary.skipped,
                "errored": summary.errored,
            },
            raw_payload={"format": results.format, "ci_run_id": results.ci_run_id},
            metadata={
                "test_suites": summary.test_suites,
                "failures": summary.failures[:10],  # Limit stored failures
            },
        )

        result = await supabase.insert("ci_events", ci_event.model_dump())

        if result.get("error"):
            raise HTTPException(status_code=500, detail="Failed to store test results")

        event_id = result["data"][0]["id"] if result.get("data") else None

        # Link to existing CI run if provided
        if results.ci_run_id:
            await supabase.update(
                "ci_events",
                {"id": f"eq.{results.ci_run_id}"},
                {
                    "test_results": ci_event.test_results,
                    "metadata": {"linked_test_run": event_id},
                },
            )

        logger.info(
            "Test results uploaded",
            event_id=event_id,
            total=summary.total,
            passed=summary.passed,
            failed=summary.failed,
        )

        return WebhookResponse(
            success=True,
            message=f"Test results: {summary.passed}/{summary.total} passed ({summary.failed} failed)",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Test results upload error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Vercel Deployment Webhook
# =============================================================================

class VercelDeploymentPayload(BaseModel):
    """Vercel deployment webhook payload."""
    id: str
    type: str  # deployment, deployment-ready, deployment-error, deployment-canceled
    createdAt: int
    payload: dict = Field(default_factory=dict)


class DeploymentEvent(BaseModel):
    """Normalized deployment event for storage."""
    project_id: str
    source: Literal["vercel", "netlify", "cloudflare", "aws_amplify"]
    external_id: str
    external_url: str | None = None
    status: Literal["pending", "building", "ready", "error", "canceled"]
    deployment_url: str | None = None
    preview_url: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    commit_message: str | None = None
    environment: Literal["production", "preview", "development"] = "preview"
    duration_seconds: int | None = None
    raw_payload: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


@router.post("/vercel", response_model=WebhookResponse)
async def handle_vercel_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Vercel deployment webhook events.

    Vercel sends webhooks for deployment lifecycle events:
    - deployment: Deployment created
    - deployment-ready: Deployment succeeded
    - deployment-error: Deployment failed
    - deployment-canceled: Deployment was canceled

    Configure in Vercel: Project Settings → Git → Deploy Hooks OR Webhooks

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "vercel", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        # Parse the webhook payload
        event_type = body.get("type", "deployment")
        payload = body.get("payload", body)  # Vercel wraps data in payload

        # Extract deployment details
        deployment = payload.get("deployment", payload)
        deployment_id = deployment.get("id") or body.get("id") or str(uuid.uuid4())

        # Map Vercel event types to our status
        status: Literal["pending", "building", "ready", "error", "canceled"] = "pending"
        if event_type == "deployment":
            status = "building"
        elif event_type == "deployment-ready":
            status = "ready"
        elif event_type == "deployment-error":
            status = "error"
        elif event_type == "deployment-canceled":
            status = "canceled"

        # Extract URLs
        deployment_url = deployment.get("url")
        if deployment_url and not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"

        # Determine environment
        target = deployment.get("target") or deployment.get("meta", {}).get("target")
        environment: Literal["production", "preview", "development"] = "preview"
        if target == "production":
            environment = "production"

        # Extract git info
        git_source = deployment.get("gitSource", {}) or payload.get("git", {})
        meta = deployment.get("meta", {})

        branch = (
            git_source.get("ref")
            or meta.get("githubCommitRef")
            or meta.get("gitlabCommitRef")
            or meta.get("bitbucketCommitRef")
        )
        commit_sha = (
            git_source.get("sha")
            or meta.get("githubCommitSha")
            or meta.get("gitlabCommitSha")
            or meta.get("bitbucketCommitSha")
        )
        commit_message = (
            meta.get("githubCommitMessage")
            or meta.get("gitlabCommitMessage")
            or meta.get("bitbucketCommitMessage")
        )

        # Calculate duration if available
        duration_seconds = None
        if deployment.get("buildingAt") and deployment.get("ready"):
            try:
                building_at = deployment["buildingAt"] / 1000 if deployment["buildingAt"] > 1e12 else deployment["buildingAt"]
                ready_at = deployment["ready"] / 1000 if deployment["ready"] > 1e12 else deployment["ready"]
                duration_seconds = int(ready_at - building_at)
            except Exception:
                pass

        deployment_event = DeploymentEvent(
            project_id=project_id,
            source="vercel",
            external_id=deployment_id,
            external_url=f"https://vercel.com/{deployment.get('ownerId')}/{deployment.get('name')}/{deployment_id}",
            status=status,
            deployment_url=deployment_url,
            preview_url=deployment_url if environment == "preview" else None,
            branch=branch,
            commit_sha=commit_sha,
            commit_message=commit_message,
            environment=environment,
            duration_seconds=duration_seconds,
            raw_payload=body,
            metadata={
                "vercel_project_id": deployment.get("projectId"),
                "vercel_project_name": deployment.get("name"),
                "vercel_team_id": deployment.get("ownerId"),
                "vercel_target": target,
                "vercel_regions": deployment.get("regions", []),
                "framework": meta.get("framework"),
            },
        )

        # Check if deployment already exists (update vs insert)
        existing = await supabase.request(
            f"/deployment_events?external_id=eq.{deployment_id}&source=eq.vercel&select=id"
        )

        if existing.get("data") and len(existing["data"]) > 0:
            # Update existing deployment
            result = await supabase.update(
                "deployment_events",
                {"external_id": f"eq.{deployment_id}", "source": "eq.vercel"},
                {
                    "status": status,
                    "deployment_url": deployment_url,
                    "duration_seconds": duration_seconds,
                    "raw_payload": body,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            event_id = existing["data"][0]["id"]
        else:
            # Insert new deployment
            result = await supabase.insert("deployment_events", deployment_event.model_dump())
            if result.get("error"):
                await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
                raise HTTPException(status_code=500, detail="Failed to store deployment event")
            event_id = result["data"][0]["id"] if result.get("data") else None

        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        logger.info(
            "Vercel webhook processed",
            event_id=event_id,
            deployment_id=deployment_id,
            status=status,
            environment=environment,
        )

        return WebhookResponse(
            success=True,
            message=f"Deployment {deployment_id}: {status} ({environment})",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vercel webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Slack Interactive Webhook
# =============================================================================

class SlackInteractivePayload(BaseModel):
    """Slack interactive message payload."""
    type: str  # block_actions, view_submission, shortcut
    user: dict
    team: dict | None = None
    channel: dict | None = None
    message: dict | None = None
    response_url: str | None = None
    trigger_id: str | None = None
    actions: list[dict] = Field(default_factory=list)
    view: dict | None = None


@router.post("/slack/interactive", response_model=WebhookResponse)
async def handle_slack_interactive(
    request: Request,
):
    """
    Handle Slack interactive message events.

    Slack sends interactive payloads when users:
    - Click buttons in messages
    - Select options from menus
    - Submit modal views
    - Use shortcuts

    Configure in Slack App: Interactivity & Shortcuts → Request URL

    Note: Slack sends this as application/x-www-form-urlencoded with a 'payload' field.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        # Slack sends form-encoded data with 'payload' JSON
        form_data = await request.form()
        payload_str = form_data.get("payload", "{}")
        body = __import__("json").loads(payload_str)

        await log_webhook(supabase, webhook_id, "slack_interactive", request, body)

        payload = SlackInteractivePayload(**body)
        interaction_type = payload.type
        user_id = payload.user.get("id")
        user_name = payload.user.get("username") or payload.user.get("name")

        # Process different interaction types
        if interaction_type == "block_actions":
            # Handle button clicks and menu selections
            for action in payload.actions:
                action_id = action.get("action_id", "")
                action_value = action.get("value", "")

                # Parse action ID to determine what to do
                if action_id.startswith("rerun_tests_"):
                    job_id = action_id.replace("rerun_tests_", "")
                    logger.info(
                        "Slack rerun tests requested",
                        job_id=job_id,
                        user=user_name,
                    )
                    # TODO: Trigger test rerun via job queue
                    # For now, just acknowledge
                    await update_webhook_log(supabase, webhook_id, "processed")
                    return WebhookResponse(
                        success=True,
                        message=f"Test rerun requested for job {job_id}",
                    )

                elif action_id.startswith("view_failure_"):
                    test_id = action_id.replace("view_failure_", "")
                    logger.info(
                        "Slack view failure requested",
                        test_id=test_id,
                        user=user_name,
                    )
                    # Could open a modal with failure details
                    await update_webhook_log(supabase, webhook_id, "processed")
                    return WebhookResponse(
                        success=True,
                        message=f"Viewing failure for test {test_id}",
                    )

                elif action_id.startswith("ignore_failure_"):
                    test_id = action_id.replace("ignore_failure_", "")
                    logger.info(
                        "Slack ignore failure requested",
                        test_id=test_id,
                        user=user_name,
                    )
                    # TODO: Mark failure as ignored in database
                    await update_webhook_log(supabase, webhook_id, "processed")
                    return WebhookResponse(
                        success=True,
                        message=f"Failure ignored for test {test_id}",
                    )

                elif action_id.startswith("run_now_"):
                    schedule_id = action_id.replace("run_now_", "")
                    logger.info(
                        "Slack run now requested",
                        schedule_id=schedule_id,
                        user=user_name,
                    )
                    # TODO: Trigger immediate schedule run
                    await update_webhook_log(supabase, webhook_id, "processed")
                    return WebhookResponse(
                        success=True,
                        message=f"Immediate run triggered for schedule {schedule_id}",
                    )

                elif action_id.startswith("skip_run_"):
                    schedule_id = action_id.replace("skip_run_", "")
                    logger.info(
                        "Slack skip run requested",
                        schedule_id=schedule_id,
                        user=user_name,
                    )
                    # TODO: Skip next scheduled run
                    await update_webhook_log(supabase, webhook_id, "processed")
                    return WebhookResponse(
                        success=True,
                        message=f"Next run skipped for schedule {schedule_id}",
                    )

                elif action_id.startswith("generate_tests_"):
                    project_id = action_id.replace("generate_tests_", "")
                    logger.info(
                        "Slack generate tests requested",
                        project_id=project_id,
                        user=user_name,
                    )
                    # TODO: Trigger test generation
                    await update_webhook_log(supabase, webhook_id, "processed")
                    return WebhookResponse(
                        success=True,
                        message=f"Test generation started for project {project_id}",
                    )

                else:
                    logger.info(
                        "Unknown Slack action",
                        action_id=action_id,
                        action_value=action_value,
                        user=user_name,
                    )

        elif interaction_type == "view_submission":
            # Handle modal submissions
            view = payload.view or {}
            callback_id = view.get("callback_id", "")
            logger.info(
                "Slack modal submission",
                callback_id=callback_id,
                user=user_name,
            )
            # Process based on callback_id

        elif interaction_type == "shortcut":
            # Handle shortcuts
            callback_id = body.get("callback_id", "")
            logger.info(
                "Slack shortcut triggered",
                callback_id=callback_id,
                user=user_name,
            )

        await update_webhook_log(supabase, webhook_id, "processed")
        return WebhookResponse(
            success=True,
            message=f"Interactive event processed: {interaction_type}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Slack interactive webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Jira Webhook
# =============================================================================

class JiraWebhookPayload(BaseModel):
    """Jira webhook payload for issue events."""
    webhookEvent: str  # jira:issue_created, jira:issue_updated, etc.
    timestamp: int | None = None
    user: dict | None = None
    issue: dict | None = None
    changelog: dict | None = None
    comment: dict | None = None


@router.post("/jira", response_model=WebhookResponse)
async def handle_jira_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Jira webhook events.

    Jira sends webhooks for:
    - Issue created/updated/deleted
    - Comment added/updated/deleted
    - Sprint events
    - Worklog events

    Configure in Jira: Settings → System → Webhooks

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "jira", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        payload = JiraWebhookPayload(**body)
        event_type = payload.webhookEvent

        # Only process issue events
        if not event_type.startswith("jira:issue"):
            await update_webhook_log(supabase, webhook_id, "processed")
            return WebhookResponse(success=True, message=f"Ignored event type: {event_type}")

        issue = payload.issue or {}
        fields = issue.get("fields", {})

        # Extract issue details
        issue_key = issue.get("key", "")
        issue_id = issue.get("id", "")
        summary = fields.get("summary", "")
        description = fields.get("description", "")
        issue_type = fields.get("issuetype", {}).get("name", "")
        status = fields.get("status", {}).get("name", "")
        priority = fields.get("priority", {}).get("name", "")
        assignee = fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None
        reporter = fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None
        jira_project = fields.get("project", {})

        # Map Jira issue type to our event type
        sdlc_event_type = "task"
        if issue_type.lower() == "bug":
            sdlc_event_type = "bug"
        elif issue_type.lower() in ("story", "user story"):
            sdlc_event_type = "story"
        elif issue_type.lower() == "epic":
            sdlc_event_type = "epic"

        # Store as SDLC event
        sdlc_event = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "source_platform": "jira",
            "event_type": sdlc_event_type,
            "external_id": issue_id,
            "external_key": issue_key,
            "external_url": f"{issue.get('self', '').split('/rest/')[0]}/browse/{issue_key}" if issue.get("self") else None,
            "title": summary,
            "description": description[:2000] if description else None,  # Truncate long descriptions
            "status": status,
            "priority": priority,
            "assignee": assignee,
            "reporter": reporter,
            "raw_payload": body,
            "metadata": {
                "jira_project_key": jira_project.get("key"),
                "jira_project_name": jira_project.get("name"),
                "issue_type": issue_type,
                "labels": fields.get("labels", []),
                "components": [c.get("name") for c in fields.get("components", [])],
                "sprint": fields.get("sprint", {}).get("name") if fields.get("sprint") else None,
            },
            "created_at": fields.get("created"),
            "updated_at": fields.get("updated"),
        }

        # Check if issue already exists (update vs insert)
        existing = await supabase.request(
            f"/sdlc_events?external_id=eq.{issue_id}&source_platform=eq.jira&select=id"
        )

        if existing.get("data") and len(existing["data"]) > 0:
            # Update existing issue
            result = await supabase.update(
                "sdlc_events",
                {"external_id": f"eq.{issue_id}", "source_platform": "eq.jira"},
                {
                    "title": summary,
                    "description": description[:2000] if description else None,
                    "status": status,
                    "priority": priority,
                    "assignee": assignee,
                    "raw_payload": body,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            event_id = existing["data"][0]["id"]
        else:
            # Insert new issue
            result = await supabase.insert("sdlc_events", sdlc_event)
            if result.get("error"):
                await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
                raise HTTPException(status_code=500, detail="Failed to store SDLC event")
            event_id = result["data"][0]["id"] if result.get("data") else sdlc_event["id"]

        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        logger.info(
            "Jira webhook processed",
            event_id=event_id,
            issue_key=issue_key,
            event_type=event_type,
        )

        return WebhookResponse(
            success=True,
            message=f"Jira {event_type}: {issue_key}",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Jira webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PagerDuty Webhook
# =============================================================================

class PagerDutyWebhookEvent(BaseModel):
    """PagerDuty webhook event."""
    routing_key: str | None = None
    event_type: str | None = None


class PagerDutyIncidentPayload(BaseModel):
    """PagerDuty incident payload."""
    id: str
    incident_number: int | None = None
    title: str
    description: str | None = None
    status: str | None = None
    urgency: str | None = None
    priority: dict | None = None
    html_url: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None
    service: dict | None = None
    escalation_policy: dict | None = None
    assignments: list[dict] = Field(default_factory=list)
    acknowledgements: list[dict] = Field(default_factory=list)


class PagerDutyWebhookPayload(BaseModel):
    """PagerDuty webhook payload (v3 format)."""
    event: dict
    messages: list[dict] = Field(default_factory=list)


@router.post("/pagerduty", response_model=WebhookResponse)
async def handle_pagerduty_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle PagerDuty webhook events.

    PagerDuty sends webhooks for incident lifecycle events:
    - incident.triggered
    - incident.acknowledged
    - incident.resolved
    - incident.escalated
    - incident.reassigned
    - incident.annotated

    Configure in PagerDuty: Service → Integrations → Add Generic Webhook.
    Use v3 webhook format.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "pagerduty", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        # Handle both v2 and v3 webhook formats
        messages = body.get("messages", [])
        if not messages:
            # v3 format with single event
            event = body.get("event", {})
            if event:
                messages = [{"event": event, "incident": event.get("data", {}).get("incident", {})}]

        processed_count = 0
        last_event_id = None

        for message in messages:
            event_type = message.get("event", {}).get("event_type", "")
            incident_data = message.get("incident", {}) or message.get("event", {}).get("data", {}).get("incident", {})

            if not incident_data.get("id"):
                continue

            incident_id = incident_data.get("id", "")
            incident_number = incident_data.get("incident_number", 0)
            title = incident_data.get("title", "PagerDuty Incident")
            description = incident_data.get("description")
            status = incident_data.get("status", "triggered")
            urgency = incident_data.get("urgency", "high")
            priority_data = incident_data.get("priority")
            priority = priority_data.get("summary") if priority_data else None
            html_url = incident_data.get("html_url", "")
            created_at = incident_data.get("created_at")
            resolved_at = incident_data.get("resolved_at")

            # Service info
            service = incident_data.get("service", {})
            service_id = service.get("id", "")
            service_name = service.get("summary", "Unknown Service")

            # Calculate duration if resolved
            duration_seconds = None
            if resolved_at and created_at:
                try:
                    from datetime import datetime as dt
                    created = dt.fromisoformat(created_at.replace("Z", "+00:00"))
                    resolved = dt.fromisoformat(resolved_at.replace("Z", "+00:00"))
                    duration_seconds = int((resolved - created).total_seconds())
                except Exception:
                    pass

            # Determine severity from urgency/priority
            severity = "high" if urgency == "high" else "medium"
            if priority and "P1" in priority:
                severity = "critical"
            elif priority and "P2" in priority:
                severity = "high"

            # Map status to our normalized status
            status_mapping = {
                "triggered": "open",
                "acknowledged": "in_progress",
                "resolved": "resolved",
            }
            normalized_status = status_mapping.get(status, status)

            # Check if we already have this incident
            existing = await supabase.request(
                f"/sdlc_events?external_id=eq.{incident_id}&source_platform=eq.pagerduty"
            )

            sdlc_event = {
                "id": str(uuid.uuid4()),
                "project_id": project_id,
                "source_platform": "pagerduty",
                "event_type": "incident",
                "external_id": incident_id,
                "external_key": str(incident_number),
                "external_url": html_url,
                "title": title,
                "description": description,
                "status": normalized_status,
                "priority": priority,
                "severity": severity,
                "incident_id": incident_id,
                "occurred_at": created_at,
                "resolved_at": resolved_at,
                "raw_payload": body,
                "metadata": {
                    "service_id": service_id,
                    "service_name": service_name,
                    "urgency": urgency,
                    "duration_seconds": duration_seconds,
                    "event_type": event_type,
                    "assignments": [a.get("assignee", {}).get("summary") for a in incident_data.get("assignments", [])],
                },
            }

            if existing.get("data"):
                # Update existing incident
                result = await supabase.update(
                    "sdlc_events",
                    {"external_id": f"eq.{incident_id}", "source_platform": "eq.pagerduty"},
                    {
                        "status": normalized_status,
                        "priority": priority,
                        "severity": severity,
                        "resolved_at": resolved_at,
                        "raw_payload": body,
                        "metadata": sdlc_event["metadata"],
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )
                last_event_id = existing["data"][0]["id"]
            else:
                # Insert new incident
                result = await supabase.insert("sdlc_events", sdlc_event)
                if result.get("error"):
                    logger.error("Failed to store PagerDuty incident", error=result["error"])
                    continue
                last_event_id = result["data"][0]["id"] if result.get("data") else sdlc_event["id"]

            processed_count += 1

        await update_webhook_log(supabase, webhook_id, "processed", event_id=last_event_id)

        logger.info(
            "PagerDuty webhook processed",
            event_id=last_event_id,
            processed_count=processed_count,
        )

        return WebhookResponse(
            success=True,
            message=f"Processed {processed_count} PagerDuty incident(s)",
            event_id=last_event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PagerDuty webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LaunchDarkly Webhook
# =============================================================================

class LaunchDarklyWebhookPayload(BaseModel):
    """LaunchDarkly webhook payload."""
    _links: dict = Field(default_factory=dict)
    _id: str | None = None
    kind: str | None = None
    name: str | None = None
    description: str | None = None
    date: int | None = None
    member: dict | None = None
    titleVerb: str | None = None
    title: str | None = None
    target: dict | None = None
    currentVersion: dict | None = None
    previousVersion: dict | None = None


@router.post("/launchdarkly", response_model=WebhookResponse)
async def handle_launchdarkly_webhook(
    request: Request,
    organization_id: str = Query(..., description="Organization ID (required for security)"),
    project_id: str | None = Query(None, description="Project ID to associate events with"),
):
    """
    Handle LaunchDarkly webhook events.

    LaunchDarkly sends webhooks for:
    - Flag created/updated/deleted
    - Flag targeting changes (on/off, rules, segments)
    - Environment changes
    - Audit log events

    Configure in LaunchDarkly: Settings → Integrations → Webhooks.

    SECURITY: organization_id is required for multi-tenant data isolation.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    # SECURITY: Validate project belongs to organization if provided
    if project_id:
        if not await validate_project_org(supabase, project_id, organization_id):
            raise HTTPException(
                status_code=403,
                detail="Project does not belong to the specified organization",
            )

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "launchdarkly", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase, organization_id)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found in organization. Please specify project_id query parameter.",
                )

        # Extract event details
        kind = body.get("kind", "unknown")
        title = body.get("title", "")
        title_verb = body.get("titleVerb", "")
        description = body.get("description", "")
        date_ms = body.get("date", 0)
        member = body.get("member", {})
        target = body.get("target", {})
        current_version = body.get("currentVersion", {})
        previous_version = body.get("previousVersion", {})

        # Determine event type based on kind and verb
        event_type = "flag_change"
        if "created" in title_verb.lower():
            event_type = "flag_created"
        elif "deleted" in title_verb.lower():
            event_type = "flag_deleted"
        elif "turned on" in title.lower() or "enabled" in title.lower():
            event_type = "flag_enabled"
        elif "turned off" in title.lower() or "disabled" in title.lower():
            event_type = "flag_disabled"
        elif "targeting" in title.lower() or "rule" in title.lower():
            event_type = "flag_targeting_changed"
        elif "rollout" in title.lower() or "percentage" in title.lower():
            event_type = "flag_rollout_changed"

        # Extract flag key from target resources
        resources = target.get("resources", [])
        flag_key = None
        environment = None
        project_key = None

        for resource in resources:
            # Format: proj/{project}/env/{env}/flags/{flag_key}
            # or: proj/{project}/flags/{flag_key}
            if isinstance(resource, str):
                parts = resource.split("/")
                if "flags" in parts:
                    flag_idx = parts.index("flags")
                    if flag_idx + 1 < len(parts):
                        flag_key = parts[flag_idx + 1]
                if "env" in parts:
                    env_idx = parts.index("env")
                    if env_idx + 1 < len(parts):
                        environment = parts[env_idx + 1]
                if "proj" in parts:
                    proj_idx = parts.index("proj")
                    if proj_idx + 1 < len(parts):
                        project_key = parts[proj_idx + 1]

        if not flag_key:
            # Try to extract from other fields
            flag_key = body.get("name") or target.get("name") or "unknown"

        # Parse timestamp
        occurred_at = None
        if date_ms:
            try:
                occurred_at = datetime.fromtimestamp(date_ms / 1000, tz=UTC).isoformat()
            except Exception:
                pass

        # Extract member info
        member_email = member.get("email", "unknown")
        member_name = member.get("name") or member.get("firstName", "")

        # Determine flag state from current version
        flag_on = None
        rollout_percentage = None
        if current_version:
            flag_on = current_version.get("on")
            fallthrough = current_version.get("fallthrough", {})
            rollout = fallthrough.get("rollout", {})
            if rollout.get("variations"):
                # Calculate rollout percentage
                variations = rollout.get("variations", [])
                if variations:
                    first_weight = variations[0].get("weight", 0)
                    total_weight = sum(v.get("weight", 0) for v in variations)
                    if total_weight > 0:
                        rollout_percentage = round((first_weight / total_weight) * 100, 2)

        # Construct SDLC event
        sdlc_event = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "source_platform": "launchdarkly",
            "event_type": event_type,
            "external_id": body.get("_id", str(uuid.uuid4())),
            "external_key": flag_key,
            "external_url": body.get("_links", {}).get("canonical", {}).get("href"),
            "title": title or f"Flag {flag_key} {title_verb}",
            "description": description,
            "status": "on" if flag_on else "off" if flag_on is not None else None,
            "flag_key": flag_key,
            "occurred_at": occurred_at,
            "raw_payload": body,
            "metadata": {
                "kind": kind,
                "title_verb": title_verb,
                "member_email": member_email,
                "member_name": member_name,
                "environment": environment,
                "project_key": project_key,
                "flag_on": flag_on,
                "rollout_percentage": rollout_percentage,
                "previous_on": previous_version.get("on") if previous_version else None,
                "resources": resources,
            },
        }

        # Store the event (don't deduplicate - each audit log entry is unique)
        result = await supabase.insert("sdlc_events", sdlc_event)

        if result.get("error"):
            await update_webhook_log(supabase, webhook_id, "failed", str(result["error"]))
            raise HTTPException(status_code=500, detail="Failed to store SDLC event")

        event_id = result["data"][0]["id"] if result.get("data") else sdlc_event["id"]
        await update_webhook_log(supabase, webhook_id, "processed", event_id=event_id)

        logger.info(
            "LaunchDarkly webhook processed",
            event_id=event_id,
            flag_key=flag_key,
            event_type=event_type,
            environment=environment,
        )

        return WebhookResponse(
            success=True,
            message=f"LaunchDarkly {event_type}: {flag_key}",
            event_id=event_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LaunchDarkly webhook error", error=str(e))
        await update_webhook_log(supabase, webhook_id, "failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))
