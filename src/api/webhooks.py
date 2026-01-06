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
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/webhooks", tags=["Webhooks"])


# =============================================================================
# Pydantic Models
# =============================================================================

class SentryException(BaseModel):
    type: Optional[str] = None
    value: Optional[str] = None


class SentryMetadata(BaseModel):
    type: Optional[str] = None
    value: Optional[str] = None
    filename: Optional[str] = None
    function: Optional[str] = None


class SentryIssue(BaseModel):
    id: str
    title: str
    culprit: Optional[str] = None
    level: str = "error"
    message: Optional[str] = None
    metadata: Optional[SentryMetadata] = None
    platform: Optional[str] = None
    project: Optional[str] = None
    url: Optional[str] = None
    shortId: Optional[str] = None
    count: str = "1"
    userCount: int = 1
    firstSeen: Optional[str] = None
    lastSeen: Optional[str] = None
    tags: list[dict[str, str]] = Field(default_factory=list)


class SentryWebhookPayload(BaseModel):
    action: str
    data: dict[str, Any]
    installation: Optional[dict] = None
    actor: Optional[dict] = None


class DatadogError(BaseModel):
    type: Optional[str] = None
    message: Optional[str] = None
    stack: Optional[str] = None
    source: Optional[str] = None


class DatadogView(BaseModel):
    url: Optional[str] = None
    name: Optional[str] = None


class DatadogEvent(BaseModel):
    id: Optional[str] = None
    event_type: Optional[str] = None
    title: str
    message: str
    date_happened: Optional[int] = None
    priority: str = "normal"
    host: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    alert_type: str = "error"
    source_type_name: Optional[str] = None
    aggregation_key: Optional[str] = None
    url: Optional[str] = None
    error: Optional[DatadogError] = None
    view: Optional[DatadogView] = None
    user: Optional[dict] = None
    context: Optional[dict] = None


class ProductionEvent(BaseModel):
    """Normalized production event for storage."""
    project_id: str
    source: Literal["sentry", "datadog", "fullstory", "logrocket", "newrelic", "bugsnag", "rollbar"]
    external_id: str
    external_url: Optional[str] = None
    event_type: Literal["error", "exception", "performance", "session", "rage_click", "dead_click"]
    severity: Literal["fatal", "error", "warning", "info"]
    title: str
    message: Optional[str] = None
    stack_trace: Optional[str] = None
    fingerprint: str
    url: Optional[str] = None
    component: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None
    device_type: Optional[Literal["desktop", "mobile", "tablet"]] = None
    occurrence_count: int = 1
    affected_users: int = 1
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    status: Literal["new", "processing", "resolved", "ignored"] = "new"
    raw_payload: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    success: bool
    message: str
    event_id: Optional[str] = None
    fingerprint: Optional[str] = None


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


def extract_component_from_stack(stack_trace: Optional[str]) -> Optional[str]:
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
    component: Optional[str],
    url: Optional[str],
) -> str:
    """Generate a fingerprint for error grouping."""
    parts = [error_type, message[:100] if message else ""]
    if component:
        parts.append(component)
    if url:
        # Normalize URL by removing query params and IDs
        normalized_url = re.sub(r"\?.*$", "", url)
        normalized_url = re.sub(r"/\d+", "/:id", normalized_url)
        normalized_url = re.sub(r"/[a-f0-9-]{36}", "/:uuid", normalized_url)
        parts.append(normalized_url)

    combined = "|".join(parts)
    hash_value = hashlib.sha256(combined.encode()).hexdigest()[:12]
    return hash_value


async def get_default_project_id(supabase) -> Optional[str]:
    """Get the first available project ID."""
    result = await supabase.select("projects", columns="id", filters={"limit": "1"})
    if result.get("data") and len(result["data"]) > 0:
        return result["data"][0]["id"]
    return None


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
    supabase, webhook_id: str, status: str, error_message: Optional[str] = None, event_id: Optional[str] = None
) -> None:
    """Update webhook log status."""
    update_data = {"status": status, "processed_at": datetime.utcnow().isoformat()}
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
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Sentry webhook events.

    Sentry sends webhooks for new issues, resolved issues, etc.
    We normalize these into production_events for Quality Intelligence.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "sentry", request, body)

        payload = SentryWebhookPayload(**body)
        action = payload.action
        data = payload.data

        # Handle issue creation/triggering
        if action in ("issue", "created", "triggered"):
            issue_data = data.get("issue", {})
            event_data = data.get("event", {})

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

            # Get project ID
            if not project_id:
                project_id = await get_default_project_id(supabase)
                if not project_id:
                    raise HTTPException(
                        status_code=400,
                        detail="No project found. Please specify project_id query parameter.",
                    )

            # Create production event
            browser_info = contexts.get("browser", {})
            os_info = contexts.get("os", {})

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
                occurrence_count=int(issue_data.get("count", "1")),
                affected_users=issue_data.get("userCount", 1),
                first_seen_at=issue_data.get("firstSeen"),
                last_seen_at=issue_data.get("lastSeen"),
                status="new",
                raw_payload=body,
                tags=[f"{t['key']}:{t['value']}" for t in issue_data.get("tags", [])],
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
                        "resolved_at": datetime.utcnow().isoformat(),
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
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Datadog webhook events.

    Datadog sends webhooks for alerts, errors, and monitoring events.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "datadog", request, body)

        events = body if isinstance(body, list) else [body]

        if not project_id:
            project_id = await get_default_project_id(supabase)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found. Please specify project_id query parameter.",
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
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle FullStory webhook events.

    FullStory sends webhooks for rage clicks, dead clicks, and error events.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "fullstory", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found. Please specify project_id query parameter.",
                )

        # FullStory webhook structure varies by event type
        event_type_raw = body.get("type", "error")
        event_type: Literal["error", "rage_click", "dead_click", "session"] = "error"
        if "rage" in event_type_raw.lower():
            event_type = "rage_click"
        elif "dead" in event_type_raw.lower():
            event_type = "dead_click"

        session_url = body.get("session", {}).get("url") or body.get("sessionUrl")
        page_url = body.get("page", {}).get("url") or body.get("pageUrl")
        element_selector = body.get("element", {}).get("selector") or body.get("selector")

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
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle LogRocket webhook events.

    LogRocket sends webhooks for errors and session events.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "logrocket", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found. Please specify project_id query parameter.",
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
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle NewRelic webhook events.

    NewRelic sends webhooks for APM alerts and errors.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "newrelic", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found. Please specify project_id query parameter.",
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
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Bugsnag webhook events.

    Bugsnag sends webhooks for new errors and regressions.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "bugsnag", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found. Please specify project_id query parameter.",
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

@router.post("/rollbar", response_model=WebhookResponse)
async def handle_rollbar_webhook(
    request: Request,
    project_id: Optional[str] = Query(None, description="Project ID to associate events with"),
):
    """
    Handle Rollbar webhook events.

    Rollbar sends webhooks for new errors and reactivated items.
    """
    webhook_id = str(uuid.uuid4())
    supabase = get_supabase_client()

    try:
        body = await request.json()
        await log_webhook(supabase, webhook_id, "rollbar", request, body)

        if not project_id:
            project_id = await get_default_project_id(supabase)
            if not project_id:
                raise HTTPException(
                    status_code=400,
                    detail="No project found. Please specify project_id query parameter.",
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
            first_seen_at=item.get("first_occurrence_timestamp"),
            last_seen_at=item.get("last_occurrence_timestamp"),
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
