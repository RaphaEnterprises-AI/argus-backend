"""
Sentry Error Event Handler for Cognee Worker

Processes Sentry error events and indexes them into Cognee for correlation
with test failures. This enables semantic search across production errors,
identifying patterns, and correlating errors with specific tests.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cognee
from prometheus_client import Counter, Histogram

logger = logging.getLogger("cognee-worker.sentry-handler")

# =============================================================================
# Prometheus Metrics
# =============================================================================

SENTRY_EVENTS_PROCESSED = Counter(
    "cognee_sentry_events_processed_total",
    "Total number of Sentry events processed",
    ["status"],  # success, error
)

SENTRY_PROCESSING_DURATION = Histogram(
    "cognee_sentry_processing_duration_seconds",
    "Time spent processing Sentry events",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

SENTRY_ERRORS_INDEXED = Counter(
    "cognee_sentry_errors_indexed_total",
    "Total number of Sentry errors indexed",
    ["error_type"],
)


class SentryHandler:
    """
    Handler for Sentry error events.

    Indexes Sentry error information into Cognee for:
    - Semantic search across production errors
    - Correlating test failures with production issues
    - Identifying error patterns and trends
    - Understanding error context from stack traces
    """

    def __init__(self):
        """Initialize the Sentry handler."""
        self.logger = logger

    def _get_dataset_name(self, org_id: str, project_id: str) -> str:
        """Generate tenant-scoped dataset name for Sentry data.

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Dataset name like 'org_abc123_project_xyz789_sentry'
        """
        return f"org_{org_id}_project_{project_id}_sentry"

    def _extract_tenant_context(self, event: dict[str, Any]) -> tuple[str, str]:
        """Extract org_id and project_id from event.

        Args:
            event: Event payload

        Returns:
            Tuple of (org_id, project_id)

        Raises:
            ValueError: If org_id or project_id is missing
        """
        tenant = event.get("tenant", {})

        org_id = tenant.get("org_id") or event.get("org_id")
        project_id = tenant.get("project_id") or event.get("project_id")

        if not org_id:
            raise ValueError("Missing org_id in event (required for multi-tenant isolation)")
        if not project_id:
            raise ValueError("Missing project_id in event (required for multi-tenant isolation)")

        return org_id, project_id

    def _format_stack_trace(self, stack_trace: str | list[dict[str, Any]] | None) -> str:
        """Format stack trace for indexing.

        Args:
            stack_trace: Stack trace as string or list of frame objects

        Returns:
            Formatted string representation of stack trace
        """
        if not stack_trace:
            return "No stack trace available"

        if isinstance(stack_trace, str):
            return stack_trace

        if isinstance(stack_trace, list):
            formatted_frames = []
            for frame in stack_trace:
                if isinstance(frame, dict):
                    filename = frame.get("filename", frame.get("abs_path", "unknown"))
                    lineno = frame.get("lineno", "?")
                    function = frame.get("function", "unknown")
                    context_line = frame.get("context_line", "")
                    module = frame.get("module", "")

                    frame_text = f"  at {function} ({filename}:{lineno})"
                    if module:
                        frame_text = f"  at {module}.{function} ({filename}:{lineno})"
                    if context_line:
                        frame_text += f"\n    > {context_line.strip()}"

                    formatted_frames.append(frame_text)
                elif isinstance(frame, str):
                    formatted_frames.append(f"  {frame}")

            return "\n".join(formatted_frames)

        return str(stack_trace)

    def _format_tags(self, tags: dict[str, Any] | list[tuple[str, Any]] | None) -> str:
        """Format tags for indexing.

        Args:
            tags: Tags as dict or list of tuples

        Returns:
            Formatted string representation of tags
        """
        if not tags:
            return "No tags"

        if isinstance(tags, dict):
            return "\n".join(f"- {k}: {v}" for k, v in tags.items())

        if isinstance(tags, list):
            return "\n".join(f"- {t[0]}: {t[1]}" for t in tags if len(t) >= 2)

        return str(tags)

    def _format_breadcrumbs(self, breadcrumbs: list[dict[str, Any]] | None) -> str:
        """Format breadcrumbs for indexing.

        Args:
            breadcrumbs: List of breadcrumb objects

        Returns:
            Formatted string representation of breadcrumbs
        """
        if not breadcrumbs:
            return "No breadcrumbs"

        formatted = []
        for crumb in breadcrumbs[-20:]:  # Only include last 20 breadcrumbs
            if isinstance(crumb, dict):
                timestamp = crumb.get("timestamp", "")
                category = crumb.get("category", "default")
                message = crumb.get("message", "")
                level = crumb.get("level", "info")
                data = crumb.get("data", {})

                crumb_text = f"- [{level}] {category}"
                if timestamp:
                    crumb_text = f"- {timestamp} [{level}] {category}"
                if message:
                    crumb_text += f": {message[:200]}"
                if data:
                    data_str = json.dumps(data) if isinstance(data, dict) else str(data)
                    crumb_text += f" | data: {data_str[:100]}"

                formatted.append(crumb_text)

        return "\n".join(formatted)

    async def process(self, event: dict[str, Any]) -> None:
        """Process a Sentry error event.

        Extracts error information and indexes it into Cognee for semantic search
        and correlation with test failures.

        Args:
            event: Sentry error event payload with fields:
                - org_id: Organization ID (from tenant or top-level)
                - project_id: Project ID (from tenant or top-level)
                - error_message: The error message
                - stack_trace: Stack trace (string or list of frames)
                - fingerprint: Error fingerprint for grouping
                - tags: Error tags (dict or list of tuples)
                - timestamp: When the error occurred (ISO format)
                - Additional optional fields:
                    - error_type: Type/class of the error
                    - level: Error level (error, warning, etc.)
                    - platform: Platform (python, javascript, etc.)
                    - environment: Environment (production, staging, etc.)
                    - release: Release version
                    - user: User context
                    - contexts: Additional contexts
                    - breadcrumbs: Event breadcrumbs
                    - event_id: Sentry event ID
                    - issue_id: Sentry issue ID
                    - issue_url: URL to the Sentry issue

        Raises:
            ValueError: If required fields are missing
        """
        start_time = time.time()

        try:
            # Extract tenant context
            org_id, project_id = self._extract_tenant_context(event)

            # Extract required fields
            error_message = event.get("error_message", "")
            if not error_message:
                raise ValueError("Missing error_message in Sentry event")

            stack_trace = event.get("stack_trace")
            fingerprint = event.get("fingerprint", [])
            tags = event.get("tags", {})
            timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())

            # Extract optional fields
            error_type = event.get("error_type", "Error")
            level = event.get("level", "error")
            platform = event.get("platform", "unknown")
            environment = event.get("environment", "production")
            release = event.get("release", "")
            user = event.get("user", {})
            contexts = event.get("contexts", {})
            breadcrumbs = event.get("breadcrumbs", [])
            event_id = event.get("event_id", "")
            issue_id = event.get("issue_id", "")
            issue_url = event.get("issue_url", "")
            transaction = event.get("transaction", "")
            server_name = event.get("server_name", "")

            # Generate tenant-scoped dataset name
            dataset_name = self._get_dataset_name(org_id, project_id)

            self.logger.info(
                f"Processing Sentry event for tenant org={org_id}, project={project_id}"
            )
            self.logger.info(f"Using dataset: {dataset_name}")

            # Format complex fields
            formatted_stack_trace = self._format_stack_trace(stack_trace)
            formatted_tags = self._format_tags(tags)
            formatted_breadcrumbs = self._format_breadcrumbs(breadcrumbs)

            # Format fingerprint
            if isinstance(fingerprint, list):
                fingerprint_str = " | ".join(str(f) for f in fingerprint)
            else:
                fingerprint_str = str(fingerprint)

            # Format user context
            user_info = "No user context"
            if user:
                user_parts = []
                if user.get("id"):
                    user_parts.append(f"ID: {user.get('id')}")
                if user.get("email"):
                    user_parts.append(f"Email: {user.get('email')}")
                if user.get("username"):
                    user_parts.append(f"Username: {user.get('username')}")
                if user.get("ip_address"):
                    user_parts.append(f"IP: {user.get('ip_address')}")
                user_info = ", ".join(user_parts) if user_parts else "No user details"

            # Build comprehensive error document for indexing
            error_document = f"""# Sentry Error Event

## Metadata
- Organization: {org_id}
- Project: {project_id}
- Event ID: {event_id}
- Issue ID: {issue_id}
- Issue URL: {issue_url}
- Timestamp: {timestamp}
- Level: {level}
- Platform: {platform}
- Environment: {environment}
- Release: {release}
- Server: {server_name}
- Transaction: {transaction}

## Error
- Type: {error_type}
- Message: {error_message}

## Fingerprint
{fingerprint_str}

## Stack Trace
{formatted_stack_trace}

## Tags
{formatted_tags}

## User Context
{user_info}

## Breadcrumbs (Last 20)
{formatted_breadcrumbs}
"""

            # Add error document to Cognee
            await cognee.add(
                error_document,
                dataset_name=dataset_name,
            )

            # Track errors indexed by type
            SENTRY_ERRORS_INDEXED.labels(error_type=error_type[:50]).inc()

            # Run Cognee's cognify pipeline to extract knowledge
            self.logger.info(f"Running Cognee cognify for dataset {dataset_name}...")
            await cognee.cognify(dataset_name=dataset_name)

            # Track successful processing
            duration = time.time() - start_time
            SENTRY_PROCESSING_DURATION.observe(duration)
            SENTRY_EVENTS_PROCESSED.labels(status="success").inc()

            self.logger.info(
                f"Completed indexing Sentry event in {duration:.2f}s "
                f"(org={org_id}, project={project_id}, type={error_type})"
            )

        except Exception as e:
            # Track failed processing
            duration = time.time() - start_time
            SENTRY_PROCESSING_DURATION.observe(duration)
            SENTRY_EVENTS_PROCESSED.labels(status="error").inc()

            self.logger.error(
                f"Error processing Sentry event: {e}",
                exc_info=True,
            )
            raise

    async def search_similar_errors(
        self,
        org_id: str,
        project_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """Search for errors similar to a query.

        Useful for finding related production errors, understanding
        error patterns, and correlating with test failures.

        Args:
            org_id: Organization ID
            project_id: Project ID
            query: Search query (error message, stack trace snippet, etc.)
            top_k: Number of results to return

        Returns:
            List of related error documents from Cognee
        """
        dataset_name = self._get_dataset_name(org_id, project_id)

        try:
            results = await cognee.search(
                query=query,
                dataset_name=dataset_name,
                top_k=top_k,
            )
            return results
        except Exception as e:
            self.logger.warning(f"Error searching Sentry errors: {e}")
            return []

    async def correlate_with_test_failure(
        self,
        org_id: str,
        project_id: str,
        test_error_message: str,
        test_stack_trace: str | None = None,
        top_k: int = 3,
    ) -> list[Any]:
        """Find production errors that may correlate with a test failure.

        Args:
            org_id: Organization ID
            project_id: Project ID
            test_error_message: Error message from the failed test
            test_stack_trace: Stack trace from the failed test (optional)
            top_k: Number of results to return

        Returns:
            List of potentially correlated Sentry errors
        """
        # Build a search query combining the test error and stack trace
        query_parts = [test_error_message]
        if test_stack_trace:
            # Extract key parts of the stack trace for search
            query_parts.append(test_stack_trace[:500])

        query = " ".join(query_parts)

        return await self.search_similar_errors(
            org_id=org_id,
            project_id=project_id,
            query=query,
            top_k=top_k,
        )
