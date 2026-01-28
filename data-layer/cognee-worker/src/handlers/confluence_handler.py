"""
Confluence Page Handler for Cognee Worker

Processes Confluence page events and indexes them into Cognee for documentation search.
This enables semantic search across documentation, finding related pages, and
understanding product/technical context.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cognee
from prometheus_client import Counter, Histogram

logger = logging.getLogger("cognee-worker.confluence-handler")

# =============================================================================
# Prometheus Metrics
# =============================================================================

CONFLUENCE_EVENTS_PROCESSED = Counter(
    "cognee_confluence_events_processed_total",
    "Total number of Confluence events processed",
    ["status"],  # success, error
)

CONFLUENCE_PROCESSING_DURATION = Histogram(
    "cognee_confluence_processing_duration_seconds",
    "Time spent processing Confluence events",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

CONFLUENCE_PAGES_INDEXED = Counter(
    "cognee_confluence_pages_indexed_total",
    "Total number of Confluence pages indexed",
    ["space_key"],
)


class ConfluenceHandler:
    """
    Handler for Confluence page events.

    Indexes Confluence pages into Cognee for:
    - Semantic search across documentation
    - Finding related documentation for test context
    - Understanding product requirements and specifications
    - Correlating documentation with code changes
    """

    def __init__(self):
        """Initialize the Confluence handler."""
        self.logger = logger

    def _get_dataset_name(self, org_id: str, project_id: str) -> str:
        """Generate tenant-scoped dataset name for Confluence data.

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Dataset name like 'org_abc123_project_xyz789_confluence'
        """
        return f"org_{org_id}_project_{project_id}_confluence"

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

    def _format_labels(self, labels: list[str] | None) -> str:
        """Format labels for indexing.

        Args:
            labels: List of label strings

        Returns:
            Formatted string representation of labels
        """
        if not labels:
            return "No labels"

        return ", ".join(labels)

    def _clean_content(self, content: str) -> str:
        """Clean and normalize Confluence content for indexing.

        Confluence content may contain HTML or storage format markup.
        This method strips common markup while preserving meaningful content.

        Args:
            content: Raw content from Confluence

        Returns:
            Cleaned content string
        """
        if not content:
            return ""

        # Basic cleanup - more sophisticated parsing could be added
        # For now, we preserve the content as-is since Cognee handles various formats
        cleaned = content.strip()

        # Truncate extremely long content to prevent memory issues
        max_length = 100000  # 100K characters
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "\n\n[Content truncated due to length]"

        return cleaned

    async def process(self, event: dict[str, Any]) -> None:
        """Process a Confluence page event.

        Extracts page information and indexes it into Cognee for semantic search.

        Args:
            event: Confluence page event payload with fields:
                - org_id: Organization ID
                - project_id: Project ID
                - page_id: Confluence page ID
                - title: Page title
                - content: Page content (HTML or storage format)
                - space_key: Confluence space key
                - labels: List of page labels
                - author: Page author/creator
                - last_modified: Last modification timestamp (ISO format)
                - parent_page_id: Parent page ID (optional)
                - page_url: URL to the page (optional)
                - version: Page version number (optional)

        Raises:
            ValueError: If required fields are missing
        """
        start_time = time.time()

        try:
            # Extract tenant context
            org_id, project_id = self._extract_tenant_context(event)

            # Extract required fields
            page_id = event.get("page_id")
            if not page_id:
                raise ValueError("Missing page_id in Confluence event")

            title = event.get("title")
            if not title:
                raise ValueError("Missing title in Confluence event")

            content = event.get("content", "")

            # Extract optional fields with defaults
            space_key = event.get("space_key", "unknown")
            labels = event.get("labels", [])
            author = event.get("author", "unknown")
            last_modified = event.get("last_modified", "")
            parent_page_id = event.get("parent_page_id")
            page_url = event.get("page_url", "")
            version = event.get("version", 1)

            # Generate tenant-scoped dataset name
            dataset_name = self._get_dataset_name(org_id, project_id)

            self.logger.info(
                f"Processing Confluence page '{title}' (ID: {page_id}) "
                f"for tenant org={org_id}, project={project_id}"
            )
            self.logger.info(f"Using dataset: {dataset_name}")

            # Clean and prepare content
            cleaned_content = self._clean_content(content)
            formatted_labels = self._format_labels(labels)

            # Build comprehensive page document for indexing
            # Include all metadata in a structured format for semantic search
            page_document = f"""# Confluence Page: {title}

## Metadata
- Organization: {org_id}
- Project: {project_id}
- Page ID: {page_id}
- Space: {space_key}
- Author: {author}
- Last Modified: {last_modified}
- Version: {version}
- URL: {page_url or 'Not available'}
- Parent Page ID: {parent_page_id or 'None (top-level page)'}

## Labels
{formatted_labels}

## Title
{title}

## Content
{cleaned_content}
"""

            # Add page document to Cognee
            await cognee.add(
                page_document,
                dataset_name=dataset_name,
            )

            # Track pages indexed by space
            CONFLUENCE_PAGES_INDEXED.labels(space_key=space_key).inc()

            # Run Cognee's cognify pipeline to extract knowledge
            self.logger.info(f"Running Cognee cognify for dataset {dataset_name}...")
            await cognee.cognify(dataset_name=dataset_name)

            # Track successful processing
            duration = time.time() - start_time
            CONFLUENCE_PROCESSING_DURATION.observe(duration)
            CONFLUENCE_EVENTS_PROCESSED.labels(status="success").inc()

            self.logger.info(
                f"Completed indexing Confluence page '{title}' (ID: {page_id}) in {duration:.2f}s "
                f"(org={org_id}, project={project_id}, space={space_key})"
            )

        except Exception as e:
            # Track failed processing
            duration = time.time() - start_time
            CONFLUENCE_PROCESSING_DURATION.observe(duration)
            CONFLUENCE_EVENTS_PROCESSED.labels(status="error").inc()

            self.logger.error(
                f"Error processing Confluence event: {e}",
                exc_info=True,
            )
            raise

    async def search_documentation(
        self,
        org_id: str,
        project_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """Search for documentation related to a query.

        Useful for finding relevant documentation for test context,
        understanding requirements, or finding related pages.

        Args:
            org_id: Organization ID
            project_id: Project ID
            query: Search query (feature name, error message, concept, etc.)
            top_k: Number of results to return

        Returns:
            List of related documentation from Cognee
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
            self.logger.warning(f"Error searching Confluence documentation: {e}")
            return []

    async def search_by_label(
        self,
        org_id: str,
        project_id: str,
        label: str,
        top_k: int = 10,
    ) -> list[Any]:
        """Search for pages with a specific label.

        Args:
            org_id: Organization ID
            project_id: Project ID
            label: Label to search for
            top_k: Number of results to return

        Returns:
            List of pages with the specified label
        """
        # Search using label as query - Cognee will find pages mentioning this label
        query = f"label:{label}"
        return await self.search_documentation(org_id, project_id, query, top_k)
