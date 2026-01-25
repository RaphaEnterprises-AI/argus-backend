"""
Jira Integration for requirements and issue tracking.

Syncs user stories, bugs, and acceptance criteria for AI-powered test generation.

API Docs: https://developer.atlassian.com/cloud/jira/platform/rest/v3/

This is a Tier 1 mandatory integration that enables:
- Fetching user stories and acceptance criteria
- Syncing bugs and tasks
- Mapping requirements to generated tests
- Understanding sprint/epic context for test prioritization
"""

import base64
import re
from dataclasses import dataclass, field
from datetime import datetime

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class JiraIssue:
    """A Jira issue (story, bug, task)."""

    issue_id: str
    key: str  # e.g., "PROJ-123"
    issue_type: str  # story, bug, task, epic
    summary: str
    description: str | None
    status: str
    priority: str | None
    assignee: str | None
    reporter: str | None
    created_at: datetime
    updated_at: datetime

    # For stories
    acceptance_criteria: str | None
    story_points: int | None

    # Relationships
    epic_key: str | None
    sprint_name: str | None
    labels: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)

    # Links
    url: str = ""


class AcceptanceCriteriaExtractor:
    """
    Helper class to extract acceptance criteria from Jira descriptions.

    Acceptance criteria can appear in various formats:
    - h3. Acceptance Criteria (Confluence/Jira markup)
    - ## Acceptance Criteria (Markdown)
    - **Acceptance Criteria:** (Bold text)
    - Given/When/Then format (Gherkin)
    - Numbered or bulleted lists after AC header
    """

    # Common patterns for acceptance criteria headers
    AC_HEADER_PATTERNS = [
        r"h[23]\.\s*acceptance\s*criteria",  # Jira markup: h2. or h3.
        r"#{2,3}\s*acceptance\s*criteria",  # Markdown: ## or ###
        r"\*{1,2}acceptance\s*criteria\*{0,2}\s*:?",  # Bold: *AC* or **AC**
        r"acceptance\s*criteria\s*:",  # Plain text with colon
        r"ac\s*:",  # Abbreviated
    ]

    # Gherkin keywords for Given/When/Then format
    GHERKIN_KEYWORDS = ["given", "when", "then", "and", "but"]

    @classmethod
    def extract(cls, description: str | None) -> str | None:
        """
        Extract acceptance criteria from a Jira issue description.

        Args:
            description: The full issue description

        Returns:
            Extracted acceptance criteria or None if not found
        """
        if not description:
            return None

        description = description.strip()

        # Try to find AC section by header
        ac_section = cls._extract_by_header(description)
        if ac_section:
            return ac_section

        # Try to find Gherkin-style AC
        gherkin_ac = cls._extract_gherkin(description)
        if gherkin_ac:
            return gherkin_ac

        # Try to find numbered/bulleted criteria lists
        list_ac = cls._extract_criteria_list(description)
        if list_ac:
            return list_ac

        return None

    @classmethod
    def _extract_by_header(cls, description: str) -> str | None:
        """Extract AC section that starts with a recognizable header."""
        description_lower = description.lower()

        for pattern in cls.AC_HEADER_PATTERNS:
            match = re.search(pattern, description_lower, re.IGNORECASE)
            if match:
                # Find the start position
                start = match.end()

                # Find the end - next header or end of text
                # Look for next h2/h3 header or ## header
                end_patterns = [
                    r"\nh[23]\.",  # Jira markup header
                    r"\n#{2,3}\s+[A-Z]",  # Markdown header
                    r"\n\*{2}[A-Z][a-z]+\*{2}:",  # Bold section header
                ]

                end = len(description)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, description[start:], re.IGNORECASE)
                    if end_match:
                        end = min(end, start + end_match.start())

                ac_text = description[start:end].strip()
                if ac_text:
                    return cls._clean_text(ac_text)

        return None

    @classmethod
    def _extract_gherkin(cls, description: str) -> str | None:
        """Extract Gherkin-style acceptance criteria (Given/When/Then)."""
        lines = description.split("\n")
        gherkin_lines = []
        in_gherkin_block = False

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Check if line starts with a Gherkin keyword
            starts_with_gherkin = any(
                line_lower.startswith(kw) or line_lower.startswith(f"- {kw}")
                for kw in cls.GHERKIN_KEYWORDS
            )

            if starts_with_gherkin:
                in_gherkin_block = True
                gherkin_lines.append(line_stripped)
            elif in_gherkin_block:
                # Continue if it's a continuation (indented or empty)
                if line_stripped == "" or line.startswith(" ") or line.startswith("\t"):
                    if line_stripped:
                        gherkin_lines.append(line_stripped)
                else:
                    # Check if this starts a new Gherkin sequence
                    if not any(
                        line_lower.startswith(kw) for kw in cls.GHERKIN_KEYWORDS
                    ):
                        break

        if len(gherkin_lines) >= 2:  # Need at least Given and Then
            return "\n".join(gherkin_lines)

        return None

    @classmethod
    def _extract_criteria_list(cls, description: str) -> str | None:
        """Extract numbered or bulleted criteria that look like AC."""
        lines = description.split("\n")
        criteria_lines = []
        in_criteria_block = False

        # Patterns that indicate a criteria list item
        list_patterns = [
            r"^\s*[\-\*\+]\s+",  # Bullet: -, *, +
            r"^\s*\d+[\.\)]\s+",  # Numbered: 1. or 1)
            r"^\s*\[\s*\]\s+",  # Checkbox: [ ]
            r"^\s*\[x\]\s+",  # Checked: [x]
        ]

        for line in lines:
            is_list_item = any(re.match(p, line) for p in list_patterns)

            if is_list_item:
                # Check if it looks like a criterion (contains action verbs)
                criterion_indicators = [
                    "should",
                    "must",
                    "shall",
                    "can",
                    "will",
                    "able to",
                    "verify",
                    "ensure",
                    "check",
                    "validate",
                    "display",
                    "show",
                    "allow",
                    "prevent",
                    "user",
                    "system",
                ]

                line_lower = line.lower()
                if any(indicator in line_lower for indicator in criterion_indicators):
                    in_criteria_block = True
                    criteria_lines.append(line.strip())
                elif in_criteria_block:
                    criteria_lines.append(line.strip())

        if len(criteria_lines) >= 2:
            return "\n".join(criteria_lines)

        return None

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Clean up extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()


class JiraIntegration:
    """
    Jira Cloud Integration.

    API Docs: https://developer.atlassian.com/cloud/jira/platform/rest/v3/

    Features:
    - Fetch issues by JQL query
    - Get issue details with acceptance criteria
    - Map stories to tests
    - Search across projects

    Authentication:
    - Uses Basic Auth with email and API token
    - API tokens can be generated at: https://id.atlassian.com/manage-profile/security/api-tokens

    Usage:
        jira = JiraIntegration(
            domain="yourcompany.atlassian.net",
            email="user@company.com",
            api_token="your-api-token"
        )

        # Get issues updated in the last week
        issues = await jira.get_issues(jql="project = PROJ AND updated >= -7d")

        # Get a specific issue
        issue = await jira.get_issue("PROJ-123")

        # Search for issues
        results = await jira.search_issues("login authentication")
    """

    def __init__(
        self,
        domain: str,  # e.g., "yourcompany.atlassian.net"
        email: str,
        api_token: str,
    ):
        self.domain = domain
        self.base_url = f"https://{domain}/rest/api/3"
        self.browse_url = f"https://{domain}/browse"

        # Basic auth: email:api_token base64 encoded
        credentials = f"{email}:{api_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.auth_header = f"Basic {encoded}"

        self.http = httpx.AsyncClient(timeout=30.0)
        self.log = logger.bind(component="jira", domain=domain)

        # Custom field IDs (these may vary per Jira instance)
        # Common custom field names that need to be discovered
        self._custom_fields: dict[str, str] = {}
        self._fields_loaded = False

    @property
    def headers(self) -> dict:
        """Get headers for Jira API requests."""
        return {
            "Authorization": self.auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _load_custom_fields(self) -> None:
        """Load custom field mappings from Jira."""
        if self._fields_loaded:
            return

        try:
            response = await self.http.get(
                f"{self.base_url}/field",
                headers=self.headers,
            )
            response.raise_for_status()
            fields = response.json()

            # Map common custom field names to their IDs
            field_mapping = {
                "story points": "story_points",
                "story point estimate": "story_points",
                "acceptance criteria": "acceptance_criteria",
                "epic link": "epic_link",
                "epic name": "epic_name",
                "sprint": "sprint",
            }

            for field in fields:
                field_name = field.get("name", "").lower()
                field_id = field.get("id", "")

                for key, mapped_name in field_mapping.items():
                    if key in field_name:
                        self._custom_fields[mapped_name] = field_id

            self._fields_loaded = True
            self.log.debug("Loaded custom fields", fields=self._custom_fields)

        except Exception as e:
            self.log.warning("Failed to load custom fields", error=str(e))
            self._fields_loaded = True  # Prevent repeated attempts

    def _parse_issue(self, issue_data: dict) -> JiraIssue:
        """Parse a Jira API issue response into a JiraIssue object."""
        fields = issue_data.get("fields", {})

        # Parse dates
        created_str = fields.get("created", "")
        updated_str = fields.get("updated", "")

        created_at = (
            datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            if created_str
            else datetime.now()
        )
        updated_at = (
            datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
            if updated_str
            else datetime.now()
        )

        # Get assignee and reporter
        assignee_data = fields.get("assignee")
        assignee = (
            assignee_data.get("displayName") or assignee_data.get("emailAddress")
            if assignee_data
            else None
        )

        reporter_data = fields.get("reporter")
        reporter = (
            reporter_data.get("displayName") or reporter_data.get("emailAddress")
            if reporter_data
            else None
        )

        # Get priority
        priority_data = fields.get("priority")
        priority = priority_data.get("name") if priority_data else None

        # Get status
        status_data = fields.get("status", {})
        status = status_data.get("name", "Unknown")

        # Get issue type
        type_data = fields.get("issuetype", {})
        issue_type = type_data.get("name", "").lower()

        # Get description (can be in ADF format or plain text)
        description = self._parse_description(fields.get("description"))

        # Extract acceptance criteria from description or custom field
        acceptance_criteria = None
        if "acceptance_criteria" in self._custom_fields:
            ac_field = fields.get(self._custom_fields["acceptance_criteria"])
            if ac_field:
                acceptance_criteria = self._parse_description(ac_field)

        # If not in custom field, try to extract from description
        if not acceptance_criteria and description:
            acceptance_criteria = AcceptanceCriteriaExtractor.extract(description)

        # Get story points
        story_points = None
        if "story_points" in self._custom_fields:
            story_points = fields.get(self._custom_fields["story_points"])
            if story_points is not None:
                story_points = int(story_points)

        # Get epic key
        epic_key = None
        if "epic_link" in self._custom_fields:
            epic_key = fields.get(self._custom_fields["epic_link"])
        # Also check parent for next-gen projects
        parent = fields.get("parent")
        if parent and not epic_key:
            parent_type = parent.get("fields", {}).get("issuetype", {}).get("name", "")
            if parent_type.lower() == "epic":
                epic_key = parent.get("key")

        # Get sprint name
        sprint_name = None
        if "sprint" in self._custom_fields:
            sprint_data = fields.get(self._custom_fields["sprint"])
            if sprint_data and isinstance(sprint_data, list) and len(sprint_data) > 0:
                # Get the most recent/active sprint
                for sprint in sprint_data:
                    if isinstance(sprint, dict):
                        if sprint.get("state") == "active":
                            sprint_name = sprint.get("name")
                            break
                        elif not sprint_name:
                            sprint_name = sprint.get("name")

        # Get labels
        labels = fields.get("labels", [])

        # Get components
        components = [c.get("name", "") for c in fields.get("components", [])]

        # Build issue URL
        issue_key = issue_data.get("key", "")
        url = f"{self.browse_url}/{issue_key}"

        return JiraIssue(
            issue_id=issue_data.get("id", ""),
            key=issue_key,
            issue_type=issue_type,
            summary=fields.get("summary", ""),
            description=description,
            status=status,
            priority=priority,
            assignee=assignee,
            reporter=reporter,
            created_at=created_at,
            updated_at=updated_at,
            acceptance_criteria=acceptance_criteria,
            story_points=story_points,
            epic_key=epic_key,
            sprint_name=sprint_name,
            labels=labels,
            components=components,
            url=url,
        )

    def _parse_description(self, description: dict | str | None) -> str | None:
        """
        Parse description which can be in ADF (Atlassian Document Format) or plain text.

        ADF is a JSON format used by Jira Cloud for rich text.
        """
        if description is None:
            return None

        if isinstance(description, str):
            return description

        if isinstance(description, dict):
            # This is ADF format - extract text content
            return self._extract_text_from_adf(description)

        return None

    def _extract_text_from_adf(self, adf: dict) -> str:
        """Extract plain text from Atlassian Document Format."""
        if not isinstance(adf, dict):
            return ""

        content = adf.get("content", [])
        text_parts = []

        for block in content:
            block_text = self._extract_text_from_adf_block(block)
            if block_text:
                text_parts.append(block_text)

        return "\n\n".join(text_parts)

    def _extract_text_from_adf_block(self, block: dict) -> str:
        """Extract text from a single ADF block."""
        block_type = block.get("type", "")
        content = block.get("content", [])

        if block_type == "text":
            return block.get("text", "")

        if block_type == "paragraph":
            texts = [self._extract_text_from_adf_block(c) for c in content]
            return "".join(texts)

        if block_type == "heading":
            level = block.get("attrs", {}).get("level", 1)
            texts = [self._extract_text_from_adf_block(c) for c in content]
            heading_text = "".join(texts)
            return f"{'#' * level} {heading_text}"

        if block_type in ("bulletList", "orderedList"):
            items = []
            for i, item in enumerate(content):
                item_text = self._extract_text_from_adf_block(item)
                prefix = "- " if block_type == "bulletList" else f"{i + 1}. "
                items.append(f"{prefix}{item_text}")
            return "\n".join(items)

        if block_type == "listItem":
            texts = [self._extract_text_from_adf_block(c) for c in content]
            return " ".join(texts)

        if block_type == "codeBlock":
            texts = [self._extract_text_from_adf_block(c) for c in content]
            code = "".join(texts)
            return f"```\n{code}\n```"

        if block_type == "blockquote":
            texts = [self._extract_text_from_adf_block(c) for c in content]
            return "> " + "\n> ".join("".join(texts).split("\n"))

        if block_type == "hardBreak":
            return "\n"

        if block_type == "mention":
            return f"@{block.get('attrs', {}).get('text', '')}"

        if block_type == "emoji":
            return block.get("attrs", {}).get("shortName", "")

        # Recursively process any nested content
        if content:
            texts = [self._extract_text_from_adf_block(c) for c in content]
            return "".join(texts)

        return ""

    async def get_issues(
        self,
        jql: str = "updated >= -7d ORDER BY updated DESC",
        limit: int = 50,
        start_at: int = 0,
    ) -> list[JiraIssue]:
        """
        Fetch issues matching a JQL query.

        Args:
            jql: Jira Query Language string
            limit: Maximum number of issues to return (max 100)
            start_at: Index to start at for pagination

        Returns:
            List of JiraIssue objects

        Example JQL queries:
            - "project = PROJ AND issuetype = Story"
            - "sprint in openSprints() AND assignee = currentUser()"
            - "labels = 'needs-e2e-test' AND status != Done"
        """
        await self._load_custom_fields()

        # Build fields to fetch
        fields = [
            "summary",
            "description",
            "status",
            "priority",
            "assignee",
            "reporter",
            "created",
            "updated",
            "issuetype",
            "labels",
            "components",
            "parent",
        ]

        # Add custom fields
        fields.extend(self._custom_fields.values())

        params = {
            "jql": jql,
            "maxResults": min(limit, 100),
            "startAt": start_at,
            "fields": ",".join(fields),
        }

        try:
            response = await self.http.get(
                f"{self.base_url}/search",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            issues = []
            for issue_data in data.get("issues", []):
                try:
                    issue = self._parse_issue(issue_data)
                    issues.append(issue)
                except Exception as e:
                    self.log.warning(
                        "Failed to parse issue",
                        issue_key=issue_data.get("key"),
                        error=str(e),
                    )

            self.log.info(
                "Fetched issues",
                count=len(issues),
                total=data.get("total", 0),
                jql=jql[:100],
            )

            return issues

        except httpx.HTTPStatusError as e:
            self.log.error(
                "HTTP error fetching issues",
                status_code=e.response.status_code,
                detail=e.response.text[:500],
            )
            return []
        except Exception as e:
            self.log.error("Failed to fetch issues", error=str(e))
            return []

    async def get_issue(self, issue_key: str) -> JiraIssue | None:
        """
        Get a single issue by key.

        Args:
            issue_key: The issue key (e.g., "PROJ-123")

        Returns:
            JiraIssue object or None if not found
        """
        await self._load_custom_fields()

        # Build fields to fetch
        fields = [
            "summary",
            "description",
            "status",
            "priority",
            "assignee",
            "reporter",
            "created",
            "updated",
            "issuetype",
            "labels",
            "components",
            "parent",
        ]
        fields.extend(self._custom_fields.values())

        params = {"fields": ",".join(fields)}

        try:
            response = await self.http.get(
                f"{self.base_url}/issue/{issue_key}",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            issue = self._parse_issue(data)
            self.log.info("Fetched issue", key=issue_key)
            return issue

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self.log.warning("Issue not found", key=issue_key)
            else:
                self.log.error(
                    "HTTP error fetching issue",
                    key=issue_key,
                    status_code=e.response.status_code,
                )
            return None
        except Exception as e:
            self.log.error("Failed to fetch issue", key=issue_key, error=str(e))
            return None

    async def get_project_issues(
        self,
        project_key: str,
        issue_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[JiraIssue]:
        """
        Get all issues for a project.

        Args:
            project_key: The project key (e.g., "PROJ")
            issue_types: Optional list of issue types to filter (e.g., ["Story", "Bug"])
            limit: Maximum number of issues to return

        Returns:
            List of JiraIssue objects
        """
        jql_parts = [f"project = {project_key}"]

        if issue_types:
            types_str = ", ".join(f'"{t}"' for t in issue_types)
            jql_parts.append(f"issuetype IN ({types_str})")

        jql_parts.append("ORDER BY updated DESC")
        jql = " AND ".join(jql_parts[:2]) + " " + jql_parts[-1]

        return await self.get_issues(jql=jql, limit=limit)

    async def search_issues(
        self,
        text: str,
        limit: int = 20,
    ) -> list[JiraIssue]:
        """
        Search issues by text.

        Uses Jira's text search which searches summary, description, and comments.

        Args:
            text: Search text
            limit: Maximum number of results

        Returns:
            List of matching JiraIssue objects
        """
        # Escape special JQL characters
        escaped_text = text.replace('"', '\\"')
        jql = f'text ~ "{escaped_text}" ORDER BY updated DESC'

        return await self.get_issues(jql=jql, limit=limit)

    async def get_stories_for_testing(
        self,
        project_key: str,
        sprint: str | None = None,
        labels: list[str] | None = None,
        limit: int = 50,
    ) -> list[JiraIssue]:
        """
        Get user stories that need test coverage.

        This is optimized for the test generation use case.

        Args:
            project_key: The project key
            sprint: Optional sprint name to filter by
            labels: Optional labels to filter by
            limit: Maximum number of stories

        Returns:
            List of JiraIssue objects with stories that have acceptance criteria
        """
        jql_parts = [
            f"project = {project_key}",
            'issuetype IN ("Story", "User Story")',
            "status != Done",
        ]

        if sprint:
            jql_parts.append(f'sprint = "{sprint}"')

        if labels:
            labels_str = ", ".join(f'"{l}"' for l in labels)
            jql_parts.append(f"labels IN ({labels_str})")

        jql = " AND ".join(jql_parts) + " ORDER BY priority DESC, updated DESC"

        issues = await self.get_issues(jql=jql, limit=limit)

        # Filter to only return issues with acceptance criteria
        stories_with_ac = [
            issue for issue in issues if issue.acceptance_criteria is not None
        ]

        self.log.info(
            "Found stories for testing",
            total=len(issues),
            with_ac=len(stories_with_ac),
        )

        return stories_with_ac

    async def get_bugs(
        self,
        project_key: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[JiraIssue]:
        """
        Get bugs from a project.

        Useful for generating regression tests from bug reports.

        Args:
            project_key: The project key
            status: Optional status filter (e.g., "Open", "In Progress", "Done")
            limit: Maximum number of bugs

        Returns:
            List of JiraIssue objects
        """
        jql_parts = [
            f"project = {project_key}",
            "issuetype = Bug",
        ]

        if status:
            jql_parts.append(f'status = "{status}"')

        jql = " AND ".join(jql_parts) + " ORDER BY priority DESC, created DESC"

        return await self.get_issues(jql=jql, limit=limit)

    async def test_connection(self) -> bool:
        """
        Test if the credentials are valid by fetching server info.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = await self.http.get(
                f"{self.base_url}/serverInfo",
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()

            self.log.info(
                "Jira connection successful",
                version=data.get("version"),
                deployment_type=data.get("deploymentType"),
            )
            return True

        except httpx.HTTPStatusError as e:
            self.log.error(
                "Jira connection failed",
                status_code=e.response.status_code,
                detail=e.response.text[:200],
            )
            return False
        except Exception as e:
            self.log.error("Jira connection failed", error=str(e))
            return False

    async def get_epic_issues(
        self,
        epic_key: str,
        limit: int = 100,
    ) -> list[JiraIssue]:
        """
        Get all issues belonging to an epic.

        Args:
            epic_key: The epic's issue key
            limit: Maximum number of issues

        Returns:
            List of JiraIssue objects
        """
        await self._load_custom_fields()

        # Try both classic and next-gen project epic linking
        jql_parts = []

        if "epic_link" in self._custom_fields:
            epic_link_field = self._custom_fields["epic_link"]
            jql_parts.append(f'"{epic_link_field}" = {epic_key}')

        # For next-gen projects, epic is the parent
        jql_parts.append(f"parent = {epic_key}")

        jql = " OR ".join(jql_parts) + " ORDER BY rank ASC"

        return await self.get_issues(jql=jql, limit=limit)

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self.http.aclose()
        self.log.debug("Jira client closed")


def create_jira_integration(
    domain: str,
    email: str,
    api_token: str,
) -> JiraIntegration:
    """
    Factory function for creating a JiraIntegration instance.

    Args:
        domain: Jira domain (e.g., "yourcompany.atlassian.net")
        email: User email for authentication
        api_token: API token (generate at https://id.atlassian.com/manage-profile/security/api-tokens)

    Returns:
        Configured JiraIntegration instance
    """
    return JiraIntegration(domain=domain, email=email, api_token=api_token)
