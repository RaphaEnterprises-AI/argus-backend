"""
GitHub Pull Request Handler for Cognee Worker

Processes GitHub PR events and indexes them into Cognee for code change context.
This enables semantic search across PR history for understanding code evolution,
finding related changes, and learning from past reviews.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cognee
from prometheus_client import Counter, Histogram

logger = logging.getLogger("cognee-worker.github-pr-handler")

# =============================================================================
# Prometheus Metrics
# =============================================================================

GITHUB_PR_EVENTS_PROCESSED = Counter(
    "cognee_github_pr_events_processed_total",
    "Total number of GitHub PR events processed",
    ["status"],  # success, error
)

GITHUB_PR_PROCESSING_DURATION = Histogram(
    "cognee_github_pr_processing_duration_seconds",
    "Time spent processing GitHub PR events",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

GITHUB_PR_FILES_INDEXED = Counter(
    "cognee_github_pr_files_indexed_total",
    "Total number of changed files indexed from GitHub PRs",
)


class GitHubPRHandler:
    """
    Handler for GitHub Pull Request events.

    Indexes PR metadata and changed files into Cognee for:
    - Semantic search across code changes
    - Understanding code evolution patterns
    - Learning from past reviews and discussions
    - Correlating test failures with code changes
    """

    def __init__(self):
        """Initialize the GitHub PR handler."""
        self.logger = logger

    def _get_dataset_name(self, org_id: str, project_id: str) -> str:
        """Generate tenant-scoped dataset name for GitHub data.

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            Dataset name like 'org_abc123_project_xyz789_github'
        """
        return f"org_{org_id}_project_{project_id}_github"

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

    def _format_reviews(self, reviews: list[dict[str, Any]]) -> str:
        """Format review data for indexing.

        Args:
            reviews: List of review objects

        Returns:
            Formatted string representation of reviews
        """
        if not reviews:
            return "No reviews"

        formatted = []
        for review in reviews:
            reviewer = review.get("reviewer", "Unknown")
            state = review.get("state", "PENDING")
            body = review.get("body", "")
            submitted_at = review.get("submitted_at", "")

            review_text = f"- {reviewer} ({state})"
            if submitted_at:
                review_text += f" at {submitted_at}"
            if body:
                review_text += f": {body[:500]}"  # Truncate long review bodies

            formatted.append(review_text)

        return "\n".join(formatted)

    def _format_changed_files(self, changed_files: list[dict[str, Any]]) -> str:
        """Format changed files for indexing.

        Args:
            changed_files: List of changed file objects

        Returns:
            Formatted string representation of changed files
        """
        if not changed_files:
            return "No changed files"

        formatted = []
        for file_info in changed_files:
            if isinstance(file_info, str):
                # Simple string path
                formatted.append(f"- {file_info}")
            elif isinstance(file_info, dict):
                # Detailed file info
                filename = file_info.get("filename", file_info.get("path", "unknown"))
                status = file_info.get("status", "modified")
                additions = file_info.get("additions", 0)
                deletions = file_info.get("deletions", 0)
                patch = file_info.get("patch", "")

                file_text = f"- {filename} ({status}, +{additions}/-{deletions})"
                if patch:
                    # Include truncated patch for context
                    patch_preview = patch[:1000] if len(patch) > 1000 else patch
                    file_text += f"\n  Patch:\n  {patch_preview}"

                formatted.append(file_text)

        return "\n".join(formatted)

    async def process(self, event: dict[str, Any]) -> None:
        """Process a GitHub PR event.

        Extracts PR information and indexes it into Cognee for semantic search.

        Args:
            event: GitHub PR event payload with fields:
                - org_id: Organization ID
                - project_id: Project ID
                - pr_number: Pull request number
                - title: PR title
                - description: PR description/body
                - author: PR author username
                - changed_files: List of changed file paths or file objects
                - additions: Total lines added
                - deletions: Total lines deleted
                - reviews: List of review objects
                - merged_at: Merge timestamp (ISO format) or None
                - base_branch: Target branch for the PR

        Raises:
            ValueError: If required fields are missing
        """
        start_time = time.time()

        try:
            # Extract tenant context
            org_id, project_id = self._extract_tenant_context(event)

            # Extract PR fields
            pr_number = event.get("pr_number")
            if not pr_number:
                raise ValueError("Missing pr_number in GitHub PR event")

            title = event.get("title", "")
            description = event.get("description", "")
            author = event.get("author", "unknown")
            changed_files = event.get("changed_files", [])
            additions = event.get("additions", 0)
            deletions = event.get("deletions", 0)
            reviews = event.get("reviews", [])
            merged_at = event.get("merged_at")
            base_branch = event.get("base_branch", "main")

            # Additional optional fields
            head_branch = event.get("head_branch", "")
            repository = event.get("repository", "")
            created_at = event.get("created_at", "")
            updated_at = event.get("updated_at", "")
            state = event.get("state", "open")
            labels = event.get("labels", [])
            assignees = event.get("assignees", [])
            reviewers = event.get("reviewers", [])

            # Generate tenant-scoped dataset name
            dataset_name = self._get_dataset_name(org_id, project_id)

            self.logger.info(
                f"Processing GitHub PR #{pr_number} for tenant org={org_id}, project={project_id}"
            )
            self.logger.info(f"Using dataset: {dataset_name}")

            # Format reviews and changed files for indexing
            formatted_reviews = self._format_reviews(reviews)
            formatted_files = self._format_changed_files(changed_files)

            # Build comprehensive PR document for indexing
            # Include all metadata in a structured format for semantic search
            pr_document = f"""# GitHub Pull Request #{pr_number}

## Metadata
- Organization: {org_id}
- Project: {project_id}
- Repository: {repository}
- Author: {author}
- State: {state}
- Base Branch: {base_branch}
- Head Branch: {head_branch}
- Created: {created_at}
- Updated: {updated_at}
- Merged: {merged_at or 'Not merged'}

## Title
{title}

## Description
{description or 'No description provided'}

## Statistics
- Lines Added: {additions}
- Lines Deleted: {deletions}
- Files Changed: {len(changed_files)}

## Labels
{', '.join(labels) if labels else 'No labels'}

## Assignees
{', '.join(assignees) if assignees else 'No assignees'}

## Requested Reviewers
{', '.join(reviewers) if reviewers else 'No reviewers requested'}

## Changed Files
{formatted_files}

## Reviews
{formatted_reviews}
"""

            # Add PR document to Cognee
            await cognee.add(
                pr_document,
                dataset_name=dataset_name,
            )

            # Track files indexed
            GITHUB_PR_FILES_INDEXED.inc(len(changed_files))

            # Optionally index individual file changes for more granular search
            # Only do this for PRs with a reasonable number of files
            if len(changed_files) <= 50:
                for file_info in changed_files:
                    if isinstance(file_info, dict) and file_info.get("patch"):
                        filename = file_info.get("filename", file_info.get("path", "unknown"))
                        patch = file_info.get("patch", "")
                        status = file_info.get("status", "modified")

                        file_document = f"""# File Change: {filename}
## Context
- Pull Request: #{pr_number}
- Title: {title}
- Author: {author}
- Status: {status}
- Organization: {org_id}
- Project: {project_id}

## Diff
```
{patch}
```
"""
                        await cognee.add(
                            file_document,
                            dataset_name=dataset_name,
                        )

            # Run Cognee's cognify pipeline to extract knowledge
            self.logger.info(f"Running Cognee cognify for dataset {dataset_name}...")
            await cognee.cognify(dataset_name=dataset_name)

            # Track successful processing
            duration = time.time() - start_time
            GITHUB_PR_PROCESSING_DURATION.observe(duration)
            GITHUB_PR_EVENTS_PROCESSED.labels(status="success").inc()

            self.logger.info(
                f"Completed indexing GitHub PR #{pr_number} in {duration:.2f}s "
                f"(org={org_id}, project={project_id}, files={len(changed_files)})"
            )

        except Exception as e:
            # Track failed processing
            duration = time.time() - start_time
            GITHUB_PR_PROCESSING_DURATION.observe(duration)
            GITHUB_PR_EVENTS_PROCESSED.labels(status="error").inc()

            self.logger.error(
                f"Error processing GitHub PR event: {e}",
                exc_info=True,
            )
            raise

    async def search_related_prs(
        self,
        org_id: str,
        project_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """Search for PRs related to a query.

        Useful for finding similar code changes, related PRs, or
        understanding code evolution patterns.

        Args:
            org_id: Organization ID
            project_id: Project ID
            query: Search query (error message, file path, feature description, etc.)
            top_k: Number of results to return

        Returns:
            List of related PR documents from Cognee
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
            self.logger.warning(f"Error searching GitHub PRs: {e}")
            return []
