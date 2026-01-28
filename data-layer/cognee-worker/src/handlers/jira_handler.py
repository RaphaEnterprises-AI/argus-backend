"""
Jira Issue Handler for Cognee Worker

Processes Jira issue events and indexes them into Cognee for linking test failures
to tickets. This enables semantic search across issue history for understanding
failure patterns, finding related bugs, and tracking issue resolution.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cognee
from prometheus_client import Counter, Histogram

logger = logging.getLogger("cognee-worker.jira-handler")

# =============================================================================
# Prometheus Metrics
# =============================================================================

JIRA_EVENTS_PROCESSED = Counter(
    "cognee_jira_events_processed_total",
    "Total number of Jira issue events processed",
    ["status"],  # success, error
)

JIRA_PROCESSING_DURATION = Histogram(
    "cognee_jira_processing_duration_seconds",
    "Time spent processing Jira issue events",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

JIRA_COMMENTS_INDEXED = Counter(
    "cognee_jira_comments_indexed_total",
    "Total number of comments indexed from Jira issues",
)


class JiraHandler:
    """
    Handler for Jira issue events.

    Indexes issue metadata, descriptions, and comments into Cognee for:
    - Semantic search across issue history
    - Linking test failures to related bugs/tickets
    - Understanding failure patterns and resolutions
    - Tracking issue status and priority
    """

    def __init__(self):
        """Initialize the Jira handler."""
        self.logger = logger

    def _get_dataset_name(self, org_id: str, project_id: str) -> str:
        """Generate tenant-scoped dataset name for Jira data.

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Dataset name like 'org_abc123_project_xyz789_jira'
        """
        return f"org_{org_id}_project_{project_id}_jira"

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

    def _format_comments(self, comments: list[dict[str, Any]]) -> str:
        """Format comment data for indexing.

        Args:
            comments: List of comment objects

        Returns:
            Formatted string representation of comments
        """
        if not comments:
            return "No comments"

        formatted = []
        for i, comment in enumerate(comments, 1):
            author = comment.get("author", "Unknown")
            body = comment.get("body", "")
            created = comment.get("created", "")
            updated = comment.get("updated", "")

            comment_text = f"### Comment {i} by {author}"
            if created:
                comment_text += f" (created: {created}"
                if updated and updated != created:
                    comment_text += f", updated: {updated}"
                comment_text += ")"
            comment_text += f"\n{body[:2000]}"  # Truncate long comment bodies

            formatted.append(comment_text)

        return "\n\n".join(formatted)

    def _format_labels(self, labels: list[str]) -> str:
        """Format labels for display.

        Args:
            labels: List of label strings

        Returns:
            Comma-separated labels or 'No labels'
        """
        if not labels:
            return "No labels"
        return ", ".join(labels)

    async def process(self, event: dict[str, Any]) -> None:
        """Process a Jira issue event.

        Extracts issue information and indexes it into Cognee for semantic search.

        Args:
            event: Jira issue event payload with fields:
                - org_id: Organization ID
                - project_id: Project ID
                - issue_key: Jira issue key (e.g., 'PROJ-123')
                - summary: Issue summary/title
                - description: Issue description
                - status: Current issue status (e.g., 'Open', 'In Progress', 'Done')
                - priority: Issue priority (e.g., 'Critical', 'High', 'Medium', 'Low')
                - labels: List of labels/tags
                - comments: List of comment objects with author, body, created, updated
                - assignee: Assigned user

        Raises:
            ValueError: If required fields are missing
        """
        start_time = time.time()

        try:
            # Extract tenant context
            org_id, project_id = self._extract_tenant_context(event)

            # Extract issue fields
            issue_key = event.get("issue_key")
            if not issue_key:
                raise ValueError("Missing issue_key in Jira issue event")

            summary = event.get("summary", "")
            description = event.get("description", "")
            status = event.get("status", "Unknown")
            priority = event.get("priority", "Unknown")
            labels = event.get("labels", [])
            comments = event.get("comments", [])
            assignee = event.get("assignee", "Unassigned")

            # Additional optional fields
            reporter = event.get("reporter", "Unknown")
            issue_type = event.get("issue_type", "Unknown")
            created = event.get("created", "")
            updated = event.get("updated", "")
            resolved = event.get("resolved", "")
            resolution = event.get("resolution", "")
            components = event.get("components", [])
            fix_versions = event.get("fix_versions", [])
            affected_versions = event.get("affected_versions", [])
            epic_key = event.get("epic_key", "")
            parent_key = event.get("parent_key", "")
            sprint = event.get("sprint", "")
            story_points = event.get("story_points")

            # Generate tenant-scoped dataset name
            dataset_name = self._get_dataset_name(org_id, project_id)

            self.logger.info(
                f"Processing Jira issue {issue_key} for tenant org={org_id}, project={project_id}"
            )
            self.logger.info(f"Using dataset: {dataset_name}")

            # Format comments for indexing
            formatted_comments = self._format_comments(comments)

            # Build comprehensive issue document for indexing
            # Include all metadata in a structured format for semantic search
            issue_document = f"""# Jira Issue {issue_key}

## Metadata
- Organization: {org_id}
- Project: {project_id}
- Issue Key: {issue_key}
- Type: {issue_type}
- Status: {status}
- Priority: {priority}
- Reporter: {reporter}
- Assignee: {assignee}
- Created: {created}
- Updated: {updated}
- Resolved: {resolved or 'Not resolved'}
- Resolution: {resolution or 'None'}

## Summary
{summary}

## Description
{description or 'No description provided'}

## Labels
{self._format_labels(labels)}

## Components
{', '.join(components) if components else 'No components'}

## Versions
- Affected: {', '.join(affected_versions) if affected_versions else 'None'}
- Fix: {', '.join(fix_versions) if fix_versions else 'None'}

## Agile
- Epic: {epic_key or 'None'}
- Parent: {parent_key or 'None'}
- Sprint: {sprint or 'None'}
- Story Points: {story_points if story_points is not None else 'Not estimated'}

## Comments ({len(comments)} total)
{formatted_comments}
"""

            # Add issue document to Cognee
            await cognee.add(
                issue_document,
                dataset_name=dataset_name,
            )

            # Track comments indexed
            JIRA_COMMENTS_INDEXED.inc(len(comments))

            # Run Cognee's cognify pipeline to extract knowledge
            self.logger.info(f"Running Cognee cognify for dataset {dataset_name}...")
            await cognee.cognify(dataset_name=dataset_name)

            # Track successful processing
            duration = time.time() - start_time
            JIRA_PROCESSING_DURATION.observe(duration)
            JIRA_EVENTS_PROCESSED.labels(status="success").inc()

            self.logger.info(
                f"Completed indexing Jira issue {issue_key} in {duration:.2f}s "
                f"(org={org_id}, project={project_id}, comments={len(comments)})"
            )

        except Exception as e:
            # Track failed processing
            duration = time.time() - start_time
            JIRA_PROCESSING_DURATION.observe(duration)
            JIRA_EVENTS_PROCESSED.labels(status="error").inc()

            self.logger.error(
                f"Error processing Jira issue event: {e}",
                exc_info=True,
            )
            raise

    async def search_related_issues(
        self,
        org_id: str,
        project_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """Search for Jira issues related to a query.

        Useful for finding similar bugs, related issues, or
        understanding failure patterns.

        Args:
            org_id: Organization ID
            project_id: Project ID
            query: Search query (error message, feature description, etc.)
            top_k: Number of results to return

        Returns:
            List of related issue documents from Cognee
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
            self.logger.warning(f"Error searching Jira issues: {e}")
            return []

    async def find_issues_by_status(
        self,
        org_id: str,
        project_id: str,
        status: str,
        top_k: int = 10,
    ) -> list[Any]:
        """Find Jira issues by status.

        Args:
            org_id: Organization ID
            project_id: Project ID
            status: Issue status to search for (e.g., 'Open', 'In Progress')
            top_k: Number of results to return

        Returns:
            List of matching issue documents
        """
        query = f"Jira issue with status {status}"
        return await self.search_related_issues(org_id, project_id, query, top_k)

    async def find_issues_by_priority(
        self,
        org_id: str,
        project_id: str,
        priority: str,
        top_k: int = 10,
    ) -> list[Any]:
        """Find Jira issues by priority.

        Args:
            org_id: Organization ID
            project_id: Project ID
            priority: Issue priority to search for (e.g., 'Critical', 'High')
            top_k: Number of results to return

        Returns:
            List of matching issue documents
        """
        query = f"Jira issue with priority {priority}"
        return await self.search_related_issues(org_id, project_id, query, top_k)
