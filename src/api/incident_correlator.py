"""Incident Correlator & Root Cause Analysis Module.

This module provides AI-powered incident correlation that automatically identifies
root causes by tracing production errors back through the SDLC timeline.

Features:
- Sentry webhook handler for error ingestion
- Stack trace parsing to extract affected files/functions
- Correlation algorithm to find likely commits that caused errors
- Probability scoring based on file overlap, time proximity, and commit size
- AI-generated root cause analysis using Claude
- Incident timeline generation

API Endpoints:
- POST /webhooks/sentry - Receive Sentry error webhooks
- POST /api/v1/incidents/correlate - Manually trigger correlation for an error
- GET /api/v1/incidents/{id} - Get incident with correlation data
- GET /api/v1/incidents/{id}/timeline - Get SDLC timeline leading to incident
- GET /api/v1/incidents/recent - List recent incidents with correlations
"""

import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.security.auth import UserContext, get_current_user
from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(tags=["Incident Correlator"])

# =============================================================================
# Models
# =============================================================================


class ErrorEvent(BaseModel):
    """An error event from a monitoring platform like Sentry."""

    message: str = Field(..., description="Error message")
    stack_trace: str = Field(default="", description="Full stack trace")
    affected_files: list[str] = Field(default_factory=list, description="Files mentioned in stack trace")
    timestamp: datetime = Field(..., description="When the error occurred")
    environment: str = Field(default="production", description="Environment (production, staging)")
    release: str | None = Field(None, description="Release/version tag")
    sentry_event_id: str = Field(..., description="Sentry event ID")
    sentry_issue_id: str | None = Field(None, description="Sentry issue ID (group)")
    severity: str = Field(default="error", description="Error severity level")
    user_count: int = Field(default=1, description="Number of affected users")
    occurrence_count: int = Field(default=1, description="Number of occurrences")
    tags: dict = Field(default_factory=dict, description="Additional tags")
    context: dict = Field(default_factory=dict, description="Additional context")


class CorrelationCandidate(BaseModel):
    """A candidate commit/deploy that may have caused an error."""

    event_id: str = Field(..., description="SDLC event ID")
    event_type: str = Field(..., description="Event type (commit, deploy, pr)")
    commit_sha: str | None = Field(None, description="Commit SHA if applicable")
    pr_number: int | None = Field(None, description="PR number if applicable")
    deploy_id: str | None = Field(None, description="Deploy ID if applicable")
    title: str | None = Field(None, description="Event title/description")
    author: str | None = Field(None, description="Author of the change")
    occurred_at: datetime = Field(..., description="When the event occurred")
    probability: float = Field(..., description="Probability this caused the error (0-1)")
    factors: list[dict] = Field(default_factory=list, description="Factors contributing to score")
    files_changed: list[str] = Field(default_factory=list, description="Files changed in this event")
    file_overlap: list[str] = Field(default_factory=list, description="Files overlapping with error")


class RootCauseAnalysis(BaseModel):
    """AI-generated root cause analysis."""

    most_likely_cause: str = Field(..., description="Most likely root cause explanation")
    how_caused: str = Field(..., description="How the code change caused the error")
    recommended_fix: str = Field(..., description="Recommended fix approach")
    verification_steps: list[str] = Field(default_factory=list, description="Steps to verify the fix")
    confidence: float = Field(..., description="Confidence in this analysis (0-1)")


class IncidentCorrelation(BaseModel):
    """A correlation between an error and its likely cause."""

    incident_id: str = Field(..., description="The error/incident event ID")
    candidates: list[CorrelationCandidate] = Field(default_factory=list, description="Ranked candidates")
    most_likely: CorrelationCandidate | None = Field(None, description="Most likely candidate")
    root_cause_analysis: RootCauseAnalysis | None = Field(None, description="AI analysis")
    confidence: float = Field(default=0.0, description="Overall confidence (0-1)")
    correlated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class IncidentReport(BaseModel):
    """Generated incident report with timeline."""

    incident_id: str
    error_message: str
    environment: str
    confidence: float
    most_likely_commit: str | None = None
    most_likely_pr: int | None = None
    most_likely_author: str | None = None
    time_since_deploy: str | None = None
    timeline: list[dict] = Field(default_factory=list)
    root_cause_analysis: RootCauseAnalysis | None = None
    report_markdown: str


class SentryWebhookResponse(BaseModel):
    """Response for Sentry webhook processing."""

    success: bool
    message: str
    event_id: str | None = None
    sdlc_event_id: str | None = None
    correlation_triggered: bool = False


class CorrelateRequest(BaseModel):
    """Request to correlate an error with likely causes."""

    error_event_id: str = Field(..., description="SDLC event ID of the error to correlate")
    hours_back: int = Field(24, ge=1, le=168, description="Hours to look back for causes")
    include_ai_analysis: bool = Field(True, description="Include AI root cause analysis")


class IncidentListResponse(BaseModel):
    """Response for listing incidents."""

    incidents: list[dict]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================


def parse_stack_trace(stack_trace: str) -> list[str]:
    """Extract file paths from a stack trace.

    Handles various stack trace formats:
    - Python: File "/path/to/file.py", line 42
    - JavaScript/Node: at function (/path/to/file.js:42:10)
    - Java: at com.example.Class.method(Class.java:42)
    - Go: /path/to/file.go:42

    Args:
        stack_trace: Raw stack trace string

    Returns:
        List of unique file paths mentioned
    """
    if not stack_trace:
        return []

    files: set[str] = set()

    # Python pattern: File "path", line N
    python_pattern = r'File "([^"]+)"'
    for match in re.finditer(python_pattern, stack_trace):
        file_path = match.group(1)
        # Filter out standard library and site-packages
        if "/site-packages/" not in file_path and "/lib/python" not in file_path:
            files.add(normalize_file_path(file_path))

    # JavaScript/TypeScript pattern: at ... (path:line:col) or at ... path:line:col
    js_patterns = [
        r"at\s+\S+\s+\(([^:)]+):\d+:\d+\)",  # at func (file:line:col)
        r"at\s+([^:(\s]+\.[jt]sx?):\d+:\d+",  # at file:line:col
        r"\(([^:)]+\.[jt]sx?):\d+:\d+\)",  # (file:line:col)
    ]
    for pattern in js_patterns:
        for match in re.finditer(pattern, stack_trace):
            file_path = match.group(1)
            if "/node_modules/" not in file_path:
                files.add(normalize_file_path(file_path))

    # Java pattern: at package.Class.method(File.java:line)
    java_pattern = r"at\s+[\w.$]+\(([\w]+\.java):\d+\)"
    for match in re.finditer(java_pattern, stack_trace):
        files.add(match.group(1))

    # Go pattern: /path/to/file.go:line
    go_pattern = r"(/[\w/.-]+\.go):\d+"
    for match in re.finditer(go_pattern, stack_trace):
        files.add(normalize_file_path(match.group(1)))

    # Ruby pattern: from /path/to/file.rb:line
    ruby_pattern = r"from\s+([^\s:]+\.rb):\d+"
    for match in re.finditer(ruby_pattern, stack_trace):
        file_path = match.group(1)
        if "/gems/" not in file_path:
            files.add(normalize_file_path(file_path))

    # Generic pattern for common extensions
    generic_pattern = r"([/\w.-]+\.(py|js|ts|tsx|jsx|java|go|rb|php|cs|swift|kt)):\d+"
    for match in re.finditer(generic_pattern, stack_trace):
        file_path = match.group(1)
        # Filter out node_modules and other common library paths
        if "/node_modules/" not in file_path and "/site-packages/" not in file_path:
            files.add(normalize_file_path(file_path))

    return list(files)


def normalize_file_path(path: str) -> str:
    """Normalize a file path for comparison.

    Removes leading slashes, common prefixes, and standardizes format.
    """
    # Remove leading slashes
    path = path.lstrip("/")

    # Remove common prefixes
    prefixes_to_remove = [
        "app/", "src/", "lib/", "packages/", "services/",
        "var/task/", "home/", "usr/", "opt/",
    ]

    for prefix in prefixes_to_remove:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    return path


def parse_isoformat_safe(timestamp_str: str) -> datetime:
    """Safely parse an ISO format timestamp string.

    Handles:
    - Timestamps ending in Z (convert to +00:00)
    - Timestamps already having timezone info
    - Timestamps without timezone info (assume UTC)
    - Timestamps with duplicate timezone suffixes (e.g., +00:00+00:00)

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        datetime object with timezone info
    """
    if not timestamp_str:
        return datetime.now(UTC)

    # If it ends with Z, replace with +00:00
    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1] + "+00:00"

    # Check for and fix duplicate timezone suffixes (e.g., +00:00+00:00)
    # This can happen when timestamps are processed multiple times
    import re
    tz_pattern = r'[+-]\d{2}:\d{2}'
    tz_matches = list(re.finditer(tz_pattern, timestamp_str))

    if len(tz_matches) > 1:
        # Multiple timezone patterns found - keep only the first one
        first_tz_end = tz_matches[0].end()
        timestamp_str = timestamp_str[:first_tz_end]

    # Check if already has timezone info by looking for +HH:MM or -HH:MM at end
    has_timezone = bool(re.search(tz_pattern + r'$', timestamp_str))

    if not has_timezone:
        # No timezone, add UTC
        timestamp_str = timestamp_str + "+00:00"

    return datetime.fromisoformat(timestamp_str)


def calculate_file_overlap_score(
    error_files: list[str],
    changed_files: list[str],
) -> tuple[float, list[str]]:
    """Calculate file overlap score between error stack trace and changed files.

    Args:
        error_files: Files mentioned in the error stack trace
        changed_files: Files changed in the commit/deploy

    Returns:
        Tuple of (score 0-0.5, list of overlapping files)
    """
    if not error_files or not changed_files:
        return 0.0, []

    # Normalize all paths
    normalized_error = {normalize_file_path(f) for f in error_files}
    normalized_changed = {normalize_file_path(f) for f in changed_files}

    # Find exact matches
    exact_matches = normalized_error & normalized_changed

    # Find partial matches (same filename, different path)
    partial_matches: set[str] = set()
    error_basenames = {f.split("/")[-1]: f for f in normalized_error}
    changed_basenames = {f.split("/")[-1]: f for f in normalized_changed}

    for basename, error_path in error_basenames.items():
        if basename in changed_basenames and error_path not in exact_matches:
            partial_matches.add(error_path)

    # Score: exact matches worth more than partial
    num_error_files = len(normalized_error)
    exact_score = len(exact_matches) / num_error_files if num_error_files > 0 else 0
    partial_score = len(partial_matches) / num_error_files if num_error_files > 0 else 0

    # Max score is 0.5 for file overlap
    total_score = min(0.5, (exact_score * 0.4) + (partial_score * 0.1))

    overlapping = list(exact_matches | partial_matches)
    return total_score, overlapping


def calculate_time_proximity_score(
    error_time: datetime,
    event_time: datetime,
    max_hours: int = 24,
) -> float:
    """Calculate time proximity score.

    Closer events get higher scores, max 0.2.

    Args:
        error_time: When the error occurred
        event_time: When the potential cause event occurred
        max_hours: Maximum time window to consider

    Returns:
        Score from 0 to 0.2
    """
    if event_time >= error_time:
        return 0.0  # Event happened after error, can't be the cause

    time_diff = error_time - event_time
    hours_diff = time_diff.total_seconds() / 3600

    if hours_diff > max_hours:
        return 0.0

    # Linear decay: closer = higher score
    proximity_ratio = 1 - (hours_diff / max_hours)
    return proximity_ratio * 0.2


def calculate_commit_size_score(
    lines_added: int,
    lines_deleted: int,
    files_changed: int,
) -> float:
    """Calculate risk score based on commit size.

    Larger commits are riskier and more likely to introduce bugs.
    Max score is 0.15.

    Args:
        lines_added: Lines added in the commit
        lines_deleted: Lines deleted
        files_changed: Number of files changed

    Returns:
        Score from 0 to 0.15
    """
    total_lines = lines_added + lines_deleted

    # Thresholds for "large" commits
    if total_lines > 500 or files_changed > 20:
        return 0.15
    if total_lines > 200 or files_changed > 10:
        return 0.10
    if total_lines > 50 or files_changed > 5:
        return 0.05

    return 0.0


def calculate_historical_pattern_score(
    author_email: str | None,
    file_paths: list[str],
    historical_failures: list[dict] | None = None,
) -> float:
    """Calculate score based on historical failure patterns.

    Checks if this author or these files have been associated with
    past failures. Max score is 0.15.

    Args:
        author_email: Author of the change
        file_paths: Files changed
        historical_failures: Past failure data (if available)

    Returns:
        Score from 0 to 0.15
    """
    # TODO: Implement historical pattern matching
    # This would query past correlations to see if:
    # - This author has introduced bugs before
    # - These files are frequently buggy
    # - This combination has failed before

    # For now, return a base score
    return 0.0


def calculate_probability(
    error_files: list[str],
    error_time: datetime,
    event_data: dict,
    max_hours: int = 24,
) -> tuple[float, list[dict], list[str]]:
    """Calculate the probability that an event caused an error.

    Factors:
    - File overlap (0-0.5) - strongest signal
    - Time proximity (0-0.2) - closer = more likely
    - Commit size (0-0.15) - larger = riskier
    - Historical patterns (0-0.15)

    Args:
        error_files: Files from the error stack trace
        error_time: When the error occurred
        event_data: Event data from sdlc_events
        max_hours: Maximum hours to look back

    Returns:
        Tuple of (probability 0-1, factors list, overlapping files)
    """
    factors: list[dict] = []

    # Get event timestamp
    event_time_str = event_data.get("occurred_at")
    if isinstance(event_time_str, str):
        event_time = parse_isoformat_safe(event_time_str)
    else:
        event_time = event_time_str or datetime.now(UTC)

    # Get files changed from event data
    data = event_data.get("data") or {}
    changed_files = data.get("changed_files", [])

    # If it's a commit/PR, also check the commit data
    if not changed_files:
        changed_files = data.get("files", [])
        if not changed_files and data.get("files_changed"):
            # May be stored as filenames only
            changed_files = []

    # 1. File overlap score (0-0.5)
    file_score, overlapping = calculate_file_overlap_score(error_files, changed_files)
    if file_score > 0:
        factors.append({
            "factor": "file_overlap",
            "score": round(file_score, 3),
            "description": f"{len(overlapping)} file(s) overlap with error stack trace",
            "overlapping_files": overlapping[:5],  # Limit for display
        })

    # 2. Time proximity score (0-0.2)
    time_score = calculate_time_proximity_score(error_time, event_time, max_hours)
    if time_score > 0:
        hours_ago = (error_time - event_time).total_seconds() / 3600
        factors.append({
            "factor": "time_proximity",
            "score": round(time_score, 3),
            "description": f"Occurred {hours_ago:.1f}h before error",
        })

    # 3. Commit size score (0-0.15)
    lines_added = data.get("lines_added", 0)
    lines_deleted = data.get("lines_deleted", 0)
    files_changed_count = data.get("files_changed", len(changed_files))

    size_score = calculate_commit_size_score(lines_added, lines_deleted, files_changed_count)
    if size_score > 0:
        factors.append({
            "factor": "commit_size",
            "score": round(size_score, 3),
            "description": f"Large change: +{lines_added}/-{lines_deleted} lines, {files_changed_count} files",
        })

    # 4. Historical patterns score (0-0.15)
    author = data.get("author") or data.get("pusher") or data.get("creator")
    history_score = calculate_historical_pattern_score(author, changed_files)
    if history_score > 0:
        factors.append({
            "factor": "historical_pattern",
            "score": round(history_score, 3),
            "description": "Matches historical failure patterns",
        })

    # Sum all factors
    total_probability = file_score + time_score + size_score + history_score

    # Apply boost if we have strong file overlap
    if file_score >= 0.3:
        total_probability = min(1.0, total_probability * 1.2)

    return round(min(1.0, total_probability), 3), factors, overlapping


async def generate_ai_root_cause_analysis(
    error_event: dict,
    likely_cause: dict,
) -> RootCauseAnalysis | None:
    """Use Claude to generate a human-readable root cause analysis.

    Args:
        error_event: The error event data
        likely_cause: The most likely cause event data

    Returns:
        RootCauseAnalysis or None if AI unavailable
    """
    settings = get_settings()

    if not settings.anthropic_api_key:
        logger.warning("Anthropic API key not configured, skipping AI analysis")
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

        # Build the prompt
        error_data = error_event.get("data") or {}
        cause_data = likely_cause.get("data") or {}

        prompt = f"""You are an expert software engineer analyzing a production incident.

## Error Details
- **Message**: {error_event.get('title', 'Unknown error')}
- **Environment**: {error_data.get('environment', 'production')}
- **First Seen**: {error_event.get('occurred_at')}
- **Affected Files**: {', '.join(error_data.get('affected_files', [])[:5])}
- **Stack Trace** (excerpt):
```
{error_data.get('stack_trace', 'Not available')[:1000]}
```

## Most Likely Cause
- **Event Type**: {likely_cause.get('event_type')}
- **Title**: {likely_cause.get('title', 'Unknown')}
- **Commit SHA**: {likely_cause.get('commit_sha', 'N/A')}
- **PR Number**: {likely_cause.get('pr_number', 'N/A')}
- **Time**: {likely_cause.get('occurred_at')}
- **Files Changed**: {', '.join(cause_data.get('changed_files', [])[:10])}
- **Lines Added**: {cause_data.get('lines_added', 0)}
- **Lines Deleted**: {cause_data.get('lines_deleted', 0)}

## Analysis Request
Based on the correlation between the error and the identified change, provide:

1. **Most Likely Root Cause**: A clear, concise explanation of what likely went wrong
2. **How It Caused the Error**: The mechanism by which the code change led to the error
3. **Recommended Fix**: A specific, actionable fix approach
4. **Verification Steps**: How to verify the fix works

Respond in JSON format:
{{
  "most_likely_cause": "...",
  "how_caused": "...",
  "recommended_fix": "...",
  "verification_steps": ["step1", "step2", ...],
  "confidence": 0.0-1.0
}}
"""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response
        import json
        try:
            response_text = response.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            analysis_data = json.loads(response_text)
            return RootCauseAnalysis(**analysis_data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse AI response as JSON", error=str(e))
            return None

    except Exception as e:
        logger.exception("Failed to generate AI root cause analysis", error=str(e))
        return None


def generate_incident_report_markdown(
    error_event: dict,
    correlation: IncidentCorrelation,
) -> str:
    """Generate a markdown incident report.

    Args:
        error_event: The error event data
        correlation: The correlation results

    Returns:
        Markdown-formatted incident report
    """
    error_data = error_event.get("data") or {}
    error_title = error_event.get("title") or error_data.get("message", "Unknown Error")

    report = f"""## Incident Analysis: {error_title}

### Root Cause Identified
**Confidence**: {correlation.confidence * 100:.0f}%

"""

    if correlation.most_likely and correlation.most_likely.probability > 0.3:
        ml = correlation.most_likely
        report += """The error was most likely introduced by """

        if ml.commit_sha:
            report += f"commit `{ml.commit_sha[:8]}`"
        if ml.pr_number:
            report += f" in PR #{ml.pr_number}"
        if ml.author:
            report += f" by @{ml.author}"

        # Calculate time since
        error_time = error_event.get("occurred_at")
        if isinstance(error_time, str):
            error_time = parse_isoformat_safe(error_time)

        if error_time and ml.occurred_at:
            time_diff = error_time - ml.occurred_at
            hours = time_diff.total_seconds() / 3600
            if hours < 1:
                time_str = f"{int(hours * 60)} minutes"
            elif hours < 24:
                time_str = f"{hours:.1f} hours"
            else:
                time_str = f"{hours / 24:.1f} days"
            report += f", deployed {time_str} before the error.\n\n"
        else:
            report += ".\n\n"

        # Show file overlap
        if ml.file_overlap:
            report += "**Affected Files** (overlapping with error):\n"
            for f in ml.file_overlap[:5]:
                report += f"- `{f}`\n"
            report += "\n"

        # Show factors
        if ml.factors:
            report += "**Contributing Factors**:\n"
            for factor in ml.factors:
                report += f"- {factor.get('description', '')} (score: {factor.get('score', 0):.2f})\n"
            report += "\n"
    else:
        report += """No definitive root cause identified. The error may be:
- Caused by external factors (infrastructure, third-party services)
- A latent bug triggered by specific conditions
- Unrelated to recent code changes

"""

    # Add AI analysis if available
    if correlation.root_cause_analysis:
        rca = correlation.root_cause_analysis
        report += f"""### AI Analysis

**Root Cause**: {rca.most_likely_cause}

**How It Happened**: {rca.how_caused}

**Recommended Fix**: {rca.recommended_fix}

**Verification Steps**:
"""
        for i, step in enumerate(rca.verification_steps, 1):
            report += f"{i}. {step}\n"
        report += "\n"

    # Add timeline if candidates exist
    if correlation.candidates:
        report += """### Timeline

| Time | Event | Details |
|------|-------|---------|
"""
        # Sort by time
        sorted_candidates = sorted(correlation.candidates, key=lambda c: c.occurred_at)
        for candidate in sorted_candidates[:5]:
            time_str = candidate.occurred_at.strftime("%H:%M")
            event_type = candidate.event_type.upper()
            details = candidate.title or f"SHA: {candidate.commit_sha[:8]}" if candidate.commit_sha else ""
            report += f"| {time_str} | {event_type} | {details} |\n"

        # Add error event
        error_time = error_event.get("occurred_at")
        if isinstance(error_time, str):
            error_time = parse_isoformat_safe(error_time)
        if error_time:
            report += f"| {error_time.strftime('%H:%M')} | ERROR | First error reported |\n"

    return report


def verify_sentry_signature(
    payload_body: bytes,
    signature_header: str | None,
    secret: str,
) -> bool:
    """Verify Sentry webhook signature.

    Args:
        payload_body: Raw request body bytes
        signature_header: Sentry-Hook-Signature header value
        secret: Webhook signing secret

    Returns:
        True if signature is valid
    """
    if not signature_header:
        logger.warning("Missing Sentry-Hook-Signature header")
        return False

    # Calculate expected signature
    mac = hmac.new(
        secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    )
    expected_signature = mac.hexdigest()

    return secrets.compare_digest(signature_header, expected_signature)


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/webhooks/sentry", response_model=SentryWebhookResponse)
async def receive_sentry_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    project_id: str = Query(..., description="Project ID for this webhook"),
    sentry_hook_resource: str | None = Header(None, alias="Sentry-Hook-Resource"),
    sentry_hook_signature: str | None = Header(None, alias="Sentry-Hook-Signature"),
):
    """Receive and process Sentry error webhooks.

    This endpoint receives webhooks from Sentry when new errors occur.
    It parses the error, extracts affected files from the stack trace,
    stores the event, and optionally triggers correlation.

    Supported Sentry webhook resources:
    - issue: Issue created/resolved
    - event_alert: Event alert triggered
    - error: Error event

    Security: Optionally verifies HMAC signature using SENTRY_WEBHOOK_SECRET env var.
    """
    import os

    # Read raw body
    body = await request.body()

    # Verify signature if configured
    webhook_secret = os.environ.get("SENTRY_WEBHOOK_SECRET")
    if webhook_secret and sentry_hook_signature:
        if not verify_sentry_signature(body, sentry_hook_signature, webhook_secret):
            logger.warning("Invalid Sentry webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    try:
        payload = await request.json()
    except Exception as e:
        logger.error("Failed to parse Sentry webhook payload", error=str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info(
        "Received Sentry webhook",
        resource=sentry_hook_resource,
        project_id=project_id,
    )

    # Extract error data based on resource type
    error_data: dict[str, Any] = {}
    sentry_event_id = None

    if sentry_hook_resource == "event_alert":
        # Event alert webhook
        event = payload.get("data", {}).get("event", {})
        sentry_event_id = event.get("event_id")
        error_data = {
            "message": event.get("title") or payload.get("data", {}).get("triggered_rule"),
            "stack_trace": extract_sentry_stacktrace(event),
            "environment": event.get("environment", "production"),
            "release": event.get("release"),
            "severity": event.get("level", "error"),
            "tags": event.get("tags", {}),
            "context": event.get("contexts", {}),
            "sentry_issue_id": payload.get("data", {}).get("issue", {}).get("id"),
        }

    elif sentry_hook_resource == "issue":
        # Issue webhook
        issue = payload.get("data", {}).get("issue", {})
        sentry_event_id = issue.get("id")
        error_data = {
            "message": issue.get("title"),
            "stack_trace": issue.get("culprit", ""),
            "environment": issue.get("project", {}).get("name", "production"),
            "severity": issue.get("level", "error"),
            "user_count": issue.get("userCount", 1),
            "occurrence_count": issue.get("count", 1),
            "sentry_issue_id": issue.get("id"),
        }

    elif sentry_hook_resource == "error":
        # Direct error webhook
        sentry_event_id = payload.get("event_id")
        error_data = {
            "message": payload.get("message") or payload.get("title"),
            "stack_trace": extract_sentry_stacktrace(payload),
            "environment": payload.get("environment", "production"),
            "release": payload.get("release"),
            "severity": payload.get("level", "error"),
            "tags": payload.get("tags", {}),
            "context": payload.get("contexts", {}),
        }

    else:
        # Unknown resource type, try to extract what we can
        sentry_event_id = (
            payload.get("event_id") or
            payload.get("data", {}).get("event", {}).get("event_id") or
            payload.get("data", {}).get("issue", {}).get("id") or
            f"unknown-{datetime.now(UTC).timestamp()}"
        )
        error_data = {
            "message": payload.get("message") or "Unknown Sentry event",
            "stack_trace": "",
            "environment": "production",
            "severity": "error",
        }

    # Parse affected files from stack trace
    affected_files = parse_stack_trace(error_data.get("stack_trace", ""))
    error_data["affected_files"] = affected_files

    # Store as SDLC event
    supabase = get_supabase_client()

    sdlc_record = {
        "project_id": project_id,
        "event_type": "error",
        "source_platform": "sentry",
        "external_id": str(sentry_event_id),
        "external_url": payload.get("url") or payload.get("data", {}).get("issue", {}).get("permalink"),
        "title": error_data.get("message"),
        "occurred_at": datetime.now(UTC).isoformat(),
        "data": error_data,
    }

    result = await supabase.insert("sdlc_events", sdlc_record)

    if result.get("error"):
        # Check for duplicate
        if "duplicate" in str(result["error"]).lower():
            logger.info("Duplicate Sentry event, skipping", event_id=sentry_event_id)
            return SentryWebhookResponse(
                success=True,
                message="Duplicate event already processed",
                event_id=sentry_event_id,
            )

        logger.error("Failed to store Sentry event", error=result["error"])
        raise HTTPException(status_code=500, detail="Failed to store error event")

    stored_event = result.get("data", [{}])[0]
    sdlc_event_id = stored_event.get("id")

    logger.info(
        "Stored Sentry error event",
        sentry_event_id=sentry_event_id,
        sdlc_event_id=sdlc_event_id,
        affected_files=len(affected_files),
    )

    # Optionally trigger correlation in background
    # For now, don't auto-trigger to avoid excessive AI costs
    correlation_triggered = False

    return SentryWebhookResponse(
        success=True,
        message="Error event stored successfully",
        event_id=sentry_event_id,
        sdlc_event_id=sdlc_event_id,
        correlation_triggered=correlation_triggered,
    )


def extract_sentry_stacktrace(event: dict) -> str:
    """Extract stack trace from Sentry event data."""
    # Check for exception stacktrace
    exception = event.get("exception")
    if exception and isinstance(exception, dict):
        values = exception.get("values", [])
        if values:
            stacktrace = values[0].get("stacktrace", {})
            frames = stacktrace.get("frames", [])
            if frames:
                # Build stacktrace string
                lines = []
                for frame in frames:
                    filename = frame.get("filename") or frame.get("abs_path", "")
                    lineno = frame.get("lineno", 0)
                    function = frame.get("function", "")
                    if filename:
                        lines.append(f'  File "{filename}", line {lineno}, in {function}')
                return "\n".join(lines)

    # Fallback to culprit
    return event.get("culprit", "")


@router.post("/api/v1/incidents/correlate", response_model=IncidentCorrelation)
async def correlate_error(
    request_body: CorrelateRequest,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Manually trigger correlation for an error event.

    Finds recent deploys and commits, scores them based on file overlap,
    time proximity, and commit size, then optionally generates AI analysis.
    """
    supabase = get_supabase_client()

    # Get the error event
    error_result = await supabase.request(
        f"/sdlc_events?id=eq.{request_body.error_event_id}&project_id=eq.{project_id}"
    )

    if error_result.get("error") or not error_result.get("data"):
        raise HTTPException(status_code=404, detail="Error event not found")

    error_event = error_result["data"][0]

    if error_event.get("event_type") != "error":
        raise HTTPException(status_code=400, detail="Event is not an error type")

    error_data = error_event.get("data") or {}
    error_files = error_data.get("affected_files", [])
    error_time_str = error_event.get("occurred_at")

    if isinstance(error_time_str, str):
        error_time = parse_isoformat_safe(error_time_str)
    else:
        error_time = error_time_str or datetime.now(UTC)

    # Calculate time window
    lookback_time = error_time - timedelta(hours=request_body.hours_back)

    # Find recent deploys and commits
    candidates_result = await supabase.request(
        f"/sdlc_events?project_id=eq.{project_id}"
        f"&event_type=in.(deploy,deployment_status,commit,push,pr)"
        f"&occurred_at=gte.{lookback_time.isoformat()}"
        f"&occurred_at=lt.{error_time.isoformat()}"
        f"&order=occurred_at.desc"
        f"&limit=50"
    )

    if candidates_result.get("error"):
        logger.error("Failed to fetch candidate events", error=candidates_result["error"])
        raise HTTPException(status_code=500, detail="Failed to fetch candidate events")

    candidate_events = candidates_result.get("data") or []

    # Score each candidate
    candidates: list[CorrelationCandidate] = []

    for event in candidate_events:
        probability, factors, overlapping = calculate_probability(
            error_files,
            error_time,
            event,
            max_hours=request_body.hours_back,
        )

        if probability > 0:
            event_data = event.get("data") or {}
            candidates.append(CorrelationCandidate(
                event_id=event["id"],
                event_type=event["event_type"],
                commit_sha=event.get("commit_sha"),
                pr_number=event.get("pr_number"),
                deploy_id=event.get("deploy_id"),
                title=event.get("title"),
                author=event_data.get("author") or event_data.get("pusher"),
                occurred_at=parse_isoformat_safe(event["occurred_at"])
                if isinstance(event["occurred_at"], str) else event["occurred_at"],
                probability=probability,
                factors=factors,
                files_changed=event_data.get("changed_files", []),
                file_overlap=overlapping,
            ))

    # Sort by probability descending
    candidates.sort(key=lambda c: c.probability, reverse=True)

    # Get most likely
    most_likely = candidates[0] if candidates else None

    # Generate AI analysis if requested and we have a likely cause
    root_cause_analysis = None
    if request_body.include_ai_analysis and most_likely and most_likely.probability >= 0.3:
        # Get the full event data for the most likely cause
        likely_event_result = await supabase.request(
            f"/sdlc_events?id=eq.{most_likely.event_id}"
        )
        if likely_event_result.get("data"):
            root_cause_analysis = await generate_ai_root_cause_analysis(
                error_event,
                likely_event_result["data"][0],
            )

    # Calculate overall confidence
    confidence = most_likely.probability if most_likely else 0.0
    if root_cause_analysis:
        confidence = (confidence + root_cause_analysis.confidence) / 2

    correlation = IncidentCorrelation(
        incident_id=request_body.error_event_id,
        candidates=candidates[:10],  # Top 10 candidates
        most_likely=most_likely,
        root_cause_analysis=root_cause_analysis,
        confidence=confidence,
    )

    # Store the correlation
    if most_likely:
        corr_record = {
            "source_event_id": most_likely.event_id,
            "target_event_id": request_body.error_event_id,
            "correlation_type": "caused_by",
            "confidence": most_likely.probability,
            "correlation_method": "automatic",
        }

        await supabase.request(
            "/event_correlations",
            method="POST",
            body=corr_record,
            headers={"Prefer": "resolution=merge-duplicates"},
        )

    logger.info(
        "Correlated error with candidates",
        error_id=request_body.error_event_id,
        candidates_found=len(candidates),
        most_likely_probability=most_likely.probability if most_likely else 0,
        user_id=user.user_id,
    )

    return correlation


@router.get("/api/v1/incidents/recent", response_model=IncidentListResponse)
async def list_recent_incidents(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum incidents to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    environment: str | None = Query(None, description="Filter by environment"),
    severity: str | None = Query(None, description="Filter by severity"),
    user: UserContext = Depends(get_current_user),
):
    """List recent incidents (errors) with their correlation status."""
    supabase = get_supabase_client()

    # Build query
    query_path = (
        f"/sdlc_events?project_id=eq.{project_id}"
        f"&event_type=eq.error"
        f"&order=occurred_at.desc"
        f"&limit={limit}"
        f"&offset={offset}"
    )

    result = await supabase.request(query_path)

    if result.get("error"):
        error_msg = str(result.get("error", ""))
        if "does not exist" in error_msg or "42P01" in error_msg:
            return IncidentListResponse(incidents=[], total=0)
        raise HTTPException(status_code=500, detail="Failed to fetch incidents")

    incidents_data = result.get("data") or []

    # Filter by environment/severity in data field
    filtered_incidents = []
    for incident in incidents_data:
        data = incident.get("data") or {}
        if environment and data.get("environment") != environment:
            continue
        if severity and data.get("severity") != severity:
            continue
        filtered_incidents.append(incident)

    # Fetch correlation status for each incident
    enriched_incidents = []
    for incident in filtered_incidents:
        corr_result = await supabase.request(
            f"/event_correlations?target_event_id=eq.{incident['id']}"
            "&correlation_type=eq.caused_by"
            "&limit=1"
        )

        has_correlation = bool(corr_result.get("data"))
        correlation_confidence = 0.0
        if corr_result.get("data"):
            correlation_confidence = float(corr_result["data"][0].get("confidence", 0))

        enriched_incidents.append({
            **incident,
            "has_correlation": has_correlation,
            "correlation_confidence": correlation_confidence,
        })

    return IncidentListResponse(
        incidents=enriched_incidents,
        total=len(filtered_incidents),
    )


@router.get("/api/v1/incidents/{incident_id}")
async def get_incident(
    incident_id: str,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Get an incident (error event) with its correlation data."""
    supabase = get_supabase_client()

    # Get the error event
    event_result = await supabase.request(
        f"/sdlc_events?id=eq.{incident_id}&project_id=eq.{project_id}"
    )

    if event_result.get("error") or not event_result.get("data"):
        raise HTTPException(status_code=404, detail="Incident not found")

    incident = event_result["data"][0]

    # Get correlations
    corr_result = await supabase.request(
        f"/event_correlations?target_event_id=eq.{incident_id}"
        "&order=confidence.desc"
    )

    correlations = corr_result.get("data") or []

    # Fetch source events for correlations
    enriched_correlations = []
    for corr in correlations:
        source_result = await supabase.request(
            f"/sdlc_events?id=eq.{corr['source_event_id']}"
        )
        if source_result.get("data"):
            source_event = source_result["data"][0]
            enriched_correlations.append({
                **corr,
                "source_event": source_event,
            })

    return {
        "incident": incident,
        "correlations": enriched_correlations,
    }


@router.get("/api/v1/incidents/{incident_id}/timeline")
async def get_incident_timeline(
    incident_id: str,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    hours_before: int = Query(24, ge=1, le=168, description="Hours before incident"),
    user: UserContext = Depends(get_current_user),
):
    """Get the SDLC timeline leading to an incident.

    Returns all events (deploys, commits, PRs) that occurred before
    the incident within the specified time window.
    """
    supabase = get_supabase_client()

    # Get the incident
    event_result = await supabase.request(
        f"/sdlc_events?id=eq.{incident_id}&project_id=eq.{project_id}"
    )

    if event_result.get("error") or not event_result.get("data"):
        raise HTTPException(status_code=404, detail="Incident not found")

    incident = event_result["data"][0]

    # Use the RPC function if available, otherwise manual query
    try:
        timeline_result = await supabase.rpc(
            "get_event_timeline",
            {
                "target_event_id": incident_id,
                "hours_before": hours_before,
                "hours_after": 0,
            },
        )
        if timeline_result.get("data"):
            return {
                "incident": incident,
                "timeline": timeline_result["data"],
            }
    except Exception:
        pass

    # Manual fallback
    incident_time_str = incident.get("occurred_at")
    if isinstance(incident_time_str, str):
        incident_time = parse_isoformat_safe(incident_time_str)
    else:
        incident_time = incident_time_str or datetime.now(UTC)

    lookback_time = incident_time - timedelta(hours=hours_before)

    timeline_result = await supabase.request(
        f"/sdlc_events?project_id=eq.{project_id}"
        f"&occurred_at=gte.{lookback_time.isoformat()}"
        f"&occurred_at=lte.{incident_time.isoformat()}"
        f"&order=occurred_at.asc"
        f"&limit=100"
    )

    return {
        "incident": incident,
        "timeline": timeline_result.get("data") or [],
    }


@router.get("/api/v1/incidents/{incident_id}/report")
async def get_incident_report(
    incident_id: str,
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    user: UserContext = Depends(get_current_user),
):
    """Generate a human-readable incident report.

    Returns a markdown-formatted report with root cause analysis,
    timeline, and recommended actions.
    """
    supabase = get_supabase_client()

    # Get incident
    event_result = await supabase.request(
        f"/sdlc_events?id=eq.{incident_id}&project_id=eq.{project_id}"
    )

    if event_result.get("error") or not event_result.get("data"):
        raise HTTPException(status_code=404, detail="Incident not found")

    incident = event_result["data"][0]

    # Get correlations
    corr_result = await supabase.request(
        f"/event_correlations?target_event_id=eq.{incident_id}"
        "&correlation_type=eq.caused_by"
        "&order=confidence.desc"
    )

    correlations = corr_result.get("data") or []

    # Build correlation object
    candidates: list[CorrelationCandidate] = []
    most_likely = None

    for corr in correlations:
        source_result = await supabase.request(
            f"/sdlc_events?id=eq.{corr['source_event_id']}"
        )
        if source_result.get("data"):
            source_event = source_result["data"][0]
            source_data = source_event.get("data") or {}

            candidate = CorrelationCandidate(
                event_id=source_event["id"],
                event_type=source_event["event_type"],
                commit_sha=source_event.get("commit_sha"),
                pr_number=source_event.get("pr_number"),
                deploy_id=source_event.get("deploy_id"),
                title=source_event.get("title"),
                author=source_data.get("author") or source_data.get("pusher"),
                occurred_at=parse_isoformat_safe(source_event["occurred_at"])
                if isinstance(source_event["occurred_at"], str) else source_event["occurred_at"],
                probability=float(corr.get("confidence", 0)),
                factors=[],
                files_changed=source_data.get("changed_files", []),
                file_overlap=[],
            )
            candidates.append(candidate)

            if not most_likely:
                most_likely = candidate

    correlation = IncidentCorrelation(
        incident_id=incident_id,
        candidates=candidates,
        most_likely=most_likely,
        root_cause_analysis=None,
        confidence=most_likely.probability if most_likely else 0.0,
    )

    # Generate report
    report_markdown = generate_incident_report_markdown(incident, correlation)

    incident_data = incident.get("data") or {}

    return IncidentReport(
        incident_id=incident_id,
        error_message=incident.get("title") or incident_data.get("message", "Unknown"),
        environment=incident_data.get("environment", "production"),
        confidence=correlation.confidence,
        most_likely_commit=most_likely.commit_sha if most_likely else None,
        most_likely_pr=most_likely.pr_number if most_likely else None,
        most_likely_author=most_likely.author if most_likely else None,
        timeline=[c.model_dump() for c in candidates],
        root_cause_analysis=correlation.root_cause_analysis,
        report_markdown=report_markdown,
    )
