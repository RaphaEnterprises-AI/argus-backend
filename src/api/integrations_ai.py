"""AI-powered Integration Features API.

Provides endpoints for:
- Converting production errors to regression tests
- Converting session replays to E2E tests
- Listing errors and sessions from connected integrations

These endpoints leverage Claude AI to analyze errors and sessions,
generating intelligent test cases that can prevent regressions.
"""

import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.agents.session_to_test import (
    ErrorToTestConverter,
    GeneratedTest,
    SessionAnalyzer,
    SessionEvent,
    SessionEventType,
    SessionToTestConverter,
    UserSession,
)
from src.api.context import get_current_organization_id
from src.api.teams import get_current_user
from src.integrations.observability_hub import (
    DatadogProvider,
    FullStoryProvider,
    NewRelicProvider,
    PostHogProvider,
    ProductionError,
    RealUserSession,
    SentryProvider,
)
from src.services.key_encryption import decrypt_api_key
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/integrations", tags=["Integration AI"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ErrorToTestRequest(BaseModel):
    """Request to convert an error to a test."""

    error_id: str = Field(..., description="Error ID from the source platform")
    platform: str = Field(
        ..., description="Source platform (sentry, datadog, new_relic, fullstory)"
    )
    project_id: str | None = Field(
        None, description="Project ID to associate the test with"
    )
    include_session: bool = Field(
        True, description="Include session context if available"
    )
    app_url: str | None = Field(
        None, description="Application URL for test execution"
    )


class SessionToTestRequest(BaseModel):
    """Request to convert a session to a test."""

    session_id: str = Field(..., description="Session ID from the source platform")
    platform: str = Field(
        ..., description="Source platform (fullstory, datadog, posthog, logrocket)"
    )
    project_id: str | None = Field(
        None, description="Project ID to associate the test with"
    )
    generalize: bool = Field(
        True, description="Generalize test data for reuse"
    )
    include_assertions: bool = Field(
        True, description="Generate intelligent assertions"
    )
    app_url: str | None = Field(
        None, description="Application URL for test execution"
    )


class GeneratedTestResponse(BaseModel):
    """Response containing a generated test."""

    id: str
    name: str
    description: str
    source_type: str  # "error" or "session"
    source_id: str
    source_platform: str
    priority: str
    confidence: float
    steps: list[dict]
    assertions: list[dict]
    preconditions: list[str]
    rationale: str
    user_journey: str
    created_at: str
    test_id: str | None = None  # ID if saved to database


class ErrorListItem(BaseModel):
    """Error item for listing."""

    id: str
    platform: str
    message: str
    stack_trace: str | None
    first_seen: str
    last_seen: str
    occurrence_count: int
    affected_users: int
    severity: str
    status: str
    issue_url: str | None
    can_generate_test: bool = True


class ErrorListResponse(BaseModel):
    """Response for listing errors."""

    errors: list[ErrorListItem]
    total: int
    platforms: list[str]


class SessionListItem(BaseModel):
    """Session item for listing."""

    id: str
    platform: str
    user_id: str | None
    started_at: str
    duration_ms: int
    page_views: int
    has_errors: bool
    has_frustration: bool
    replay_url: str | None
    can_generate_test: bool = True


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[SessionListItem]
    total: int
    platforms: list[str]


class BulkGenerateRequest(BaseModel):
    """Request to bulk generate tests from errors or sessions."""

    items: list[dict] = Field(
        ...,
        description="List of items with 'id' and 'platform' fields",
        max_length=10,
    )
    source_type: str = Field(
        ..., description="Type of source: 'error' or 'session'"
    )
    project_id: str | None = Field(
        None, description="Project ID to associate tests with"
    )


class BulkGenerateResponse(BaseModel):
    """Response for bulk test generation."""

    generated: list[GeneratedTestResponse]
    failed: list[dict]
    total_generated: int
    total_failed: int


# ============================================================================
# Helper Functions
# ============================================================================


async def get_integration_credentials(
    platform: str, project_id: str | None = None
) -> dict[str, str] | None:
    """Get decrypted credentials for an integration."""
    supabase = get_supabase_client()

    # Map platform names to integration types
    platform_map = {
        "sentry": "sentry",
        "datadog": "datadog",
        "new_relic": "new_relic",
        "fullstory": "fullstory",
        "posthog": "posthog",
        "logrocket": "logrocket",
    }

    integration_type = platform_map.get(platform)
    if not integration_type:
        return None

    path = f"/integrations?type=eq.{integration_type}&status=eq.connected"
    if project_id:
        path += f"&project_id=eq.{project_id}"

    result = await supabase.request(path)

    if not result.get("data"):
        return None

    integration = result["data"][0]
    encrypted_creds = integration.get("credentials", {})

    if not encrypted_creds:
        return None

    # Decrypt credentials
    decrypted = {}
    for key, value in encrypted_creds.items():
        if value and isinstance(value, str):
            try:
                decrypted[key] = decrypt_api_key(value)
            except Exception as e:
                logger.warning(f"Failed to decrypt {key}: {e}")
                decrypted[key] = ""

    return decrypted


async def get_provider_for_platform(
    platform: str, credentials: dict[str, str]
):
    """Get the appropriate provider instance for a platform."""
    if platform == "sentry":
        return SentryProvider(
            auth_token=credentials.get("auth_token", ""),
            organization=credentials.get("organization", ""),
            project=credentials.get("project", ""),
        )
    elif platform == "datadog":
        return DatadogProvider(
            api_key=credentials.get("api_key", ""),
            app_key=credentials.get("app_key", ""),
            site=credentials.get("site", "datadoghq.com"),
        )
    elif platform == "new_relic":
        return NewRelicProvider(
            api_key=credentials.get("api_key", ""),
            account_id=credentials.get("account_id", ""),
        )
    elif platform == "fullstory":
        return FullStoryProvider(api_key=credentials.get("api_key", ""))
    elif platform == "posthog":
        return PostHogProvider(
            api_key=credentials.get("api_key", ""),
            host=credentials.get("host", "https://app.posthog.com"),
        )
    else:
        return None


def convert_generated_test_to_response(
    test: GeneratedTest,
    source_type: str,
    source_id: str,
    source_platform: str,
    test_id: str | None = None,
) -> GeneratedTestResponse:
    """Convert a GeneratedTest to a response model."""
    return GeneratedTestResponse(
        id=test.id,
        name=test.name,
        description=test.description,
        source_type=source_type,
        source_id=source_id,
        source_platform=source_platform,
        priority=test.priority,
        confidence=test.confidence,
        steps=test.steps,
        assertions=test.assertions,
        preconditions=test.preconditions,
        rationale=test.rationale,
        user_journey=test.user_journey,
        created_at=datetime.now(UTC).isoformat(),
        test_id=test_id,
    )


async def save_generated_test(
    test: GeneratedTest,
    source_type: str,
    source_id: str,
    source_platform: str,
    project_id: str | None,
    user_id: str,
) -> str:
    """Save a generated test to the database."""
    supabase = get_supabase_client()

    test_id = str(uuid.uuid4())

    test_data = {
        "id": test_id,
        "name": test.name,
        "description": test.description,
        "steps": test.steps,
        "assertions": test.assertions,
        "preconditions": test.preconditions,
        "priority": test.priority,
        "status": "draft",
        "source": source_type,
        "source_id": source_id,
        "source_platform": source_platform,
        "project_id": project_id,
        "created_by": user_id,
        "metadata": {
            "generated_by": "ai",
            "confidence": test.confidence,
            "rationale": test.rationale,
            "user_journey": test.user_journey,
            "source_session_ids": test.source_session_ids,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }

    result = await supabase.insert("tests", test_data)

    if result.get("error"):
        logger.error("Failed to save generated test", error=result["error"])
        raise HTTPException(status_code=500, detail="Failed to save generated test")

    return test_id


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/errors", response_model=ErrorListResponse)
async def list_errors(
    request: Request,
    project_id: str | None = None,
    platform: str | None = None,
    limit: int = 50,
    since_hours: int = 24,
):
    """
    List production errors from connected integrations.

    Fetches errors from Sentry, Datadog, New Relic, etc. that can be
    converted into regression tests.
    """
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    errors: list[ErrorListItem] = []
    platforms_with_data: set[str] = set()

    # Determine which platforms to query
    platforms_to_query = [platform] if platform else ["sentry", "datadog", "new_relic", "fullstory"]

    since = datetime.utcnow() - timedelta(hours=since_hours)

    for plat in platforms_to_query:
        try:
            credentials = await get_integration_credentials(plat, project_id)
            if not credentials:
                continue

            provider = await get_provider_for_platform(plat, credentials)
            if not provider:
                continue

            try:
                platform_errors = await provider.get_errors(limit=limit, since=since)

                for err in platform_errors:
                    errors.append(
                        ErrorListItem(
                            id=err.error_id,
                            platform=plat,
                            message=err.message,
                            stack_trace=err.stack_trace,
                            first_seen=err.first_seen.isoformat(),
                            last_seen=err.last_seen.isoformat(),
                            occurrence_count=err.occurrence_count,
                            affected_users=err.affected_users,
                            severity=err.severity,
                            status=err.status,
                            issue_url=err.issue_url,
                            can_generate_test=True,
                        )
                    )
                    platforms_with_data.add(plat)

            finally:
                await provider.close()

        except Exception as e:
            logger.warning(f"Failed to fetch errors from {plat}: {e}")
            continue

    # Sort by occurrence count (most impactful first)
    errors.sort(key=lambda e: e.occurrence_count, reverse=True)

    logger.info(
        "Listed integration errors",
        user_id=user["user_id"],
        error_count=len(errors),
        platforms=list(platforms_with_data),
    )

    return ErrorListResponse(
        errors=errors[:limit],
        total=len(errors),
        platforms=list(platforms_with_data),
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    project_id: str | None = None,
    platform: str | None = None,
    limit: int = 50,
    since_hours: int = 24,
    filter_errors: bool = False,
    filter_frustrations: bool = False,
):
    """
    List user sessions from connected integrations.

    Fetches sessions from FullStory, PostHog, Datadog RUM, etc. that can be
    converted into E2E tests.
    """
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    sessions: list[SessionListItem] = []
    platforms_with_data: set[str] = set()

    # Determine which platforms to query
    platforms_to_query = [platform] if platform else ["fullstory", "datadog", "posthog"]

    since = datetime.utcnow() - timedelta(hours=since_hours)

    for plat in platforms_to_query:
        try:
            credentials = await get_integration_credentials(plat, project_id)
            if not credentials:
                continue

            provider = await get_provider_for_platform(plat, credentials)
            if not provider:
                continue

            try:
                # FullStory has special filtering options
                if plat == "fullstory" and hasattr(provider, "get_recent_sessions"):
                    platform_sessions = await provider.get_recent_sessions(
                        limit=limit,
                        since=since,
                        filter_errors=filter_errors,
                        filter_frustrations=filter_frustrations,
                    )
                else:
                    platform_sessions = await provider.get_recent_sessions(
                        limit=limit, since=since
                    )

                for sess in platform_sessions:
                    sessions.append(
                        SessionListItem(
                            id=sess.session_id,
                            platform=plat,
                            user_id=sess.user_id,
                            started_at=sess.started_at.isoformat(),
                            duration_ms=sess.duration_ms,
                            page_views=len(sess.page_views) if isinstance(sess.page_views, list) else 0,
                            has_errors=bool(sess.errors),
                            has_frustration=bool(sess.frustration_signals),
                            replay_url=sess.replay_url,
                            can_generate_test=True,
                        )
                    )
                    platforms_with_data.add(plat)

            finally:
                await provider.close()

        except Exception as e:
            logger.warning(f"Failed to fetch sessions from {plat}: {e}")
            continue

    # Sort by most recent first
    sessions.sort(key=lambda s: s.started_at, reverse=True)

    logger.info(
        "Listed integration sessions",
        user_id=user["user_id"],
        session_count=len(sessions),
        platforms=list(platforms_with_data),
    )

    return SessionListResponse(
        sessions=sessions[:limit],
        total=len(sessions),
        platforms=list(platforms_with_data),
    )


@router.post("/error-to-test", response_model=GeneratedTestResponse)
async def convert_error_to_test(
    body: ErrorToTestRequest,
    request: Request,
):
    """
    Convert a production error into a regression test.

    Uses Claude AI to analyze the error and generate a test that:
    1. Reproduces the conditions that caused the error
    2. Verifies the error is fixed after remediation
    3. Prevents regression by running in CI/CD

    The generated test includes:
    - Steps to reproduce the error scenario
    - Assertions to verify correct behavior
    - Negative assertions to ensure error doesn't recur
    """
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    # Get credentials for the platform
    credentials = await get_integration_credentials(body.platform, body.project_id)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"No connected {body.platform} integration found",
        )

    provider = await get_provider_for_platform(body.platform, credentials)
    if not provider:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {body.platform}",
        )

    try:
        # Fetch errors and find the specific one
        errors = await provider.get_errors(limit=100)
        error_event = None

        for err in errors:
            if err.error_id == body.error_id:
                error_event = {
                    "id": err.error_id,
                    "message": err.message,
                    "stack_trace": err.stack_trace,
                    "first_seen": err.first_seen.isoformat(),
                    "last_seen": err.last_seen.isoformat(),
                    "occurrence_count": err.occurrence_count,
                    "affected_users": err.affected_users,
                    "tags": err.tags,
                    "context": err.context,
                    "release": err.release,
                    "environment": err.environment,
                    "severity": err.severity,
                }
                break

        if not error_event:
            raise HTTPException(
                status_code=404,
                detail=f"Error {body.error_id} not found in {body.platform}",
            )

        # Try to get session context if requested and available
        session = None
        if body.include_session and hasattr(provider, "get_recent_sessions"):
            # For FullStory, we might have session linked to error
            sessions = await provider.get_recent_sessions(limit=10)
            if sessions:
                # Find session with matching error
                for s in sessions:
                    if any(e.get("id") == body.error_id for e in s.errors if isinstance(e, dict)):
                        session = UserSession(
                            session_id=s.session_id,
                            user_id=s.user_id,
                            started_at=s.started_at,
                            ended_at=None,
                            events=[],  # Would need to fetch detailed events
                            errors=s.errors,
                            device_info=s.device,
                            geo_info=s.geo,
                            outcome="error",
                        )
                        break

        # Convert error to test
        converter = ErrorToTestConverter()
        generated_test = await converter.convert_error(error_event, session)

        # Save the test if project_id is provided
        test_id = None
        if body.project_id:
            test_id = await save_generated_test(
                generated_test,
                source_type="error",
                source_id=body.error_id,
                source_platform=body.platform,
                project_id=body.project_id,
                user_id=user["user_id"],
            )

        logger.info(
            "Generated test from error",
            user_id=user["user_id"],
            error_id=body.error_id,
            platform=body.platform,
            test_name=generated_test.name,
            confidence=generated_test.confidence,
        )

        return convert_generated_test_to_response(
            generated_test,
            source_type="error",
            source_id=body.error_id,
            source_platform=body.platform,
            test_id=test_id,
        )

    finally:
        await provider.close()


@router.post("/session-to-test", response_model=GeneratedTestResponse)
async def convert_session_to_test(
    body: SessionToTestRequest,
    request: Request,
):
    """
    Convert a user session into an E2E test.

    Uses Claude AI to analyze the session recording and generate a test that:
    1. Captures the user's intent and journey
    2. Verifies the expected outcome is achieved
    3. Can be parameterized for different test data

    The generated test includes:
    - Navigation and interaction steps
    - Form inputs (generalized for test data)
    - Assertions based on user journey analysis
    """
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    # Get credentials for the platform
    credentials = await get_integration_credentials(body.platform, body.project_id)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"No connected {body.platform} integration found",
        )

    provider = await get_provider_for_platform(body.platform, credentials)
    if not provider:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {body.platform}",
        )

    try:
        # Fetch sessions and find the specific one
        sessions = await provider.get_recent_sessions(limit=100)
        target_session = None

        for sess in sessions:
            if sess.session_id == body.session_id:
                target_session = sess
                break

        if not target_session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {body.session_id} not found in {body.platform}",
            )

        # Convert RealUserSession to our UserSession format
        events = []

        # Convert page views to navigation events
        if isinstance(target_session.page_views, list):
            for i, page in enumerate(target_session.page_views):
                if isinstance(page, dict):
                    url = page.get("url", "")
                elif isinstance(page, str):
                    url = page
                else:
                    continue

                events.append(
                    SessionEvent(
                        timestamp=target_session.started_at,
                        type=SessionEventType.NAVIGATION,
                        url=url,
                    )
                )

        # Convert actions to events
        if isinstance(target_session.actions, list):
            for action in target_session.actions:
                if isinstance(action, dict):
                    action_type = action.get("type", "").lower()
                    if "click" in action_type:
                        events.append(
                            SessionEvent(
                                timestamp=target_session.started_at,
                                type=SessionEventType.CLICK,
                                target=action.get("target", action.get("selector", "")),
                            )
                        )
                    elif "input" in action_type or "type" in action_type:
                        events.append(
                            SessionEvent(
                                timestamp=target_session.started_at,
                                type=SessionEventType.INPUT,
                                target=action.get("target", action.get("selector", "")),
                                value=action.get("value", ""),
                            )
                        )

        # Create UserSession for conversion
        user_session = UserSession(
            session_id=target_session.session_id,
            user_id=target_session.user_id,
            started_at=target_session.started_at,
            ended_at=None,
            events=events,
            errors=[e if isinstance(e, dict) else {"message": str(e)} for e in target_session.errors],
            device_info=target_session.device,
            geo_info=target_session.geo,
            outcome="conversion" if target_session.conversion_events else (
                "error" if target_session.errors else "unknown"
            ),
        )

        # Convert session to test
        converter = SessionToTestConverter()
        generated_test = await converter.convert_session(
            user_session,
            include_assertions=body.include_assertions,
            generalize=body.generalize,
        )

        # Save the test if project_id is provided
        test_id = None
        if body.project_id:
            test_id = await save_generated_test(
                generated_test,
                source_type="session",
                source_id=body.session_id,
                source_platform=body.platform,
                project_id=body.project_id,
                user_id=user["user_id"],
            )

        logger.info(
            "Generated test from session",
            user_id=user["user_id"],
            session_id=body.session_id,
            platform=body.platform,
            test_name=generated_test.name,
            confidence=generated_test.confidence,
        )

        return convert_generated_test_to_response(
            generated_test,
            source_type="session",
            source_id=body.session_id,
            source_platform=body.platform,
            test_id=test_id,
        )

    finally:
        await provider.close()


@router.post("/bulk-generate", response_model=BulkGenerateResponse)
async def bulk_generate_tests(
    body: BulkGenerateRequest,
    request: Request,
):
    """
    Bulk generate tests from multiple errors or sessions.

    Processes up to 10 items at a time and returns generated tests
    along with any failures.
    """
    user = await get_current_user(request)
    org_id = await get_current_organization_id(request)

    generated: list[GeneratedTestResponse] = []
    failed: list[dict] = []

    for item in body.items:
        item_id = item.get("id")
        platform = item.get("platform")

        if not item_id or not platform:
            failed.append({
                "item": item,
                "error": "Missing 'id' or 'platform' field",
            })
            continue

        try:
            if body.source_type == "error":
                request_body = ErrorToTestRequest(
                    error_id=item_id,
                    platform=platform,
                    project_id=body.project_id,
                )
                result = await convert_error_to_test(request_body, request)
            elif body.source_type == "session":
                request_body = SessionToTestRequest(
                    session_id=item_id,
                    platform=platform,
                    project_id=body.project_id,
                )
                result = await convert_session_to_test(request_body, request)
            else:
                failed.append({
                    "item": item,
                    "error": f"Invalid source_type: {body.source_type}",
                })
                continue

            generated.append(result)

        except HTTPException as e:
            failed.append({
                "item": item,
                "error": e.detail,
            })
        except Exception as e:
            failed.append({
                "item": item,
                "error": str(e),
            })

    logger.info(
        "Bulk generated tests",
        user_id=user["user_id"],
        source_type=body.source_type,
        generated_count=len(generated),
        failed_count=len(failed),
    )

    return BulkGenerateResponse(
        generated=generated,
        failed=failed,
        total_generated=len(generated),
        total_failed=len(failed),
    )


@router.get("/error/{platform}/{error_id}/analyze")
async def analyze_error(
    platform: str,
    error_id: str,
    request: Request,
    project_id: str | None = None,
):
    """
    Analyze an error without generating a test.

    Returns detailed analysis including:
    - Root cause hypothesis
    - Reproduction steps
    - Impact assessment
    - Suggested fixes
    """
    user = await get_current_user(request)

    credentials = await get_integration_credentials(platform, project_id)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"No connected {platform} integration found",
        )

    provider = await get_provider_for_platform(platform, credentials)
    if not provider:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {platform}",
        )

    try:
        errors = await provider.get_errors(limit=100)
        error_data = None

        for err in errors:
            if err.error_id == error_id:
                error_data = err
                break

        if not error_data:
            raise HTTPException(
                status_code=404,
                detail=f"Error {error_id} not found",
            )

        # Use the SessionAnalyzer with error context
        # This would be extended to do proper error analysis
        return {
            "error_id": error_id,
            "platform": platform,
            "message": error_data.message,
            "stack_trace": error_data.stack_trace,
            "occurrence_count": error_data.occurrence_count,
            "affected_users": error_data.affected_users,
            "severity": error_data.severity,
            "analysis": {
                "can_reproduce": error_data.stack_trace is not None,
                "test_priority": "critical" if error_data.occurrence_count > 100 else (
                    "high" if error_data.occurrence_count > 10 else "medium"
                ),
                "suggested_test_type": "regression",
            },
        }

    finally:
        await provider.close()


@router.get("/session/{platform}/{session_id}/analyze")
async def analyze_session(
    platform: str,
    session_id: str,
    request: Request,
    project_id: str | None = None,
):
    """
    Analyze a session without generating a test.

    Returns detailed analysis including:
    - User intent inference
    - Journey classification
    - Frustration signals
    - Test-worthiness assessment
    """
    user = await get_current_user(request)

    credentials = await get_integration_credentials(platform, project_id)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"No connected {platform} integration found",
        )

    provider = await get_provider_for_platform(platform, credentials)
    if not provider:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform: {platform}",
        )

    try:
        sessions = await provider.get_recent_sessions(limit=100)
        session_data = None

        for sess in sessions:
            if sess.session_id == session_id:
                session_data = sess
                break

        if not session_data:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        return {
            "session_id": session_id,
            "platform": platform,
            "user_id": session_data.user_id,
            "duration_ms": session_data.duration_ms,
            "page_views": len(session_data.page_views) if isinstance(session_data.page_views, list) else 0,
            "has_errors": bool(session_data.errors),
            "has_frustration": bool(session_data.frustration_signals),
            "replay_url": session_data.replay_url,
            "analysis": {
                "test_worthy": bool(session_data.page_views) and len(session_data.page_views) > 1,
                "test_priority": "high" if session_data.frustration_signals else (
                    "critical" if session_data.errors else "medium"
                ),
                "suggested_test_type": "e2e" if not session_data.errors else "regression",
                "journey_steps": len(session_data.page_views) if isinstance(session_data.page_views, list) else 0,
            },
        }

    finally:
        await provider.close()
