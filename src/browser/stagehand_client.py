"""
Stagehand Client - AI-powered browser automation client.

This module provides a client for interacting with Stagehand-style browser
automation services that use natural language instructions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx


class StagehandAction(str, Enum):
    """Types of actions that can be performed."""
    ACT = "act"
    EXTRACT = "extract"
    OBSERVE = "observe"


@dataclass
class ActionResult:
    """Result of a Stagehand action."""
    success: bool
    action: StagehandAction
    instruction: str
    result: Any | None = None
    error: str | None = None
    cached: bool = False
    healed: bool = False
    duration_ms: int = 0
    tokens_used: int = 0


@dataclass
class ExtractionSchema:
    """Schema for data extraction."""
    fields: dict[str, str]

    def to_zod_schema(self) -> dict:
        """Convert to Zod-compatible schema format."""
        type_mapping = {
            "string": "string",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
        }

        properties = {}
        for field_name, field_type in self.fields.items():
            properties[field_name] = {
                "type": type_mapping.get(field_type, "string")
            }

        return {
            "type": "object",
            "properties": properties,
        }


class StagehandPage:
    """Represents a page in a Stagehand browser session."""

    def __init__(self, client: "StagehandClient", page_id: str, url: str):
        self.client = client
        self.page_id = page_id
        self.url = url
        self._action_history: list[ActionResult] = []

    async def act(self, instruction: str) -> ActionResult:
        """Perform an action on the page using natural language."""
        try:
            response = await self.client._execute_action(
                page_id=self.page_id,
                action=StagehandAction.ACT,
                instruction=instruction,
            )

            result = ActionResult(
                success=response.get("success", True),
                action=StagehandAction.ACT,
                instruction=instruction,
                result=response.get("action_taken"),
                cached=response.get("cached", False),
                healed=response.get("healed", False),
            )
            self._action_history.append(result)
            return result

        except Exception as e:
            result = ActionResult(
                success=False,
                action=StagehandAction.ACT,
                instruction=instruction,
                error=str(e),
            )
            self._action_history.append(result)
            return result

    async def extract(self, schema: dict[str, str]) -> ActionResult:
        """Extract data from the page according to a schema."""
        try:
            response = await self.client._execute_action(
                page_id=self.page_id,
                action=StagehandAction.EXTRACT,
                instruction="",
                schema=schema,
            )

            result = ActionResult(
                success=response.get("success", True),
                action=StagehandAction.EXTRACT,
                instruction="extract",
                result=response.get("data"),
                cached=response.get("cached", False),
            )
            self._action_history.append(result)
            return result

        except Exception as e:
            result = ActionResult(
                success=False,
                action=StagehandAction.EXTRACT,
                instruction="extract",
                error=str(e),
            )
            self._action_history.append(result)
            return result

    async def observe(self, question: str) -> ActionResult:
        """Observe and describe what's on the page."""
        try:
            response = await self.client._execute_action(
                page_id=self.page_id,
                action=StagehandAction.OBSERVE,
                instruction=question,
            )

            result = ActionResult(
                success=response.get("success", True),
                action=StagehandAction.OBSERVE,
                instruction=question,
                result=response.get("observation", ""),
            )
            self._action_history.append(result)
            return result

        except Exception as e:
            result = ActionResult(
                success=False,
                action=StagehandAction.OBSERVE,
                instruction=question,
                error=str(e),
            )
            self._action_history.append(result)
            return result

    def get_stats(self) -> dict:
        """Get statistics for actions performed on this page."""
        total_actions = len(self._action_history)
        cached_actions = sum(1 for a in self._action_history if a.cached)
        healed_actions = sum(1 for a in self._action_history if a.healed)
        total_tokens = sum(a.tokens_used for a in self._action_history)

        return {
            "total_actions": total_actions,
            "cached_actions": cached_actions,
            "healed_actions": healed_actions,
            "total_tokens": total_tokens,
        }


class StagehandClient:
    """Client for Stagehand browser automation service."""

    def __init__(
        self,
        endpoint: str,
        api_token: str | None = None,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint
        self.api_token = api_token
        self.timeout = timeout
        self._connected = False
        self._http_client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict:
        """Get HTTP headers for requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    async def _connect(self) -> None:
        """Establish connection to the Stagehand service."""
        try:
            self._http_client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            # Verify connection with a health check
            await self._http_client.get("/health")
            self._connected = True
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to Stagehand: {e}")

    async def _disconnect(self) -> None:
        """Close the connection."""
        if self._http_client:
            await self._http_client.aclose()
        self._connected = False

    async def __aenter__(self) -> "StagehandClient":
        """Async context manager entry."""
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self._disconnect()

    async def new_page(self, url: str) -> StagehandPage:
        """Create a new page and navigate to a URL."""
        if not self._connected:
            raise RuntimeError("Client not connected. Call connect() first.")

        response = await self._http_client.post(
            "/page/new",
            json={"url": url},
        )
        response.raise_for_status()
        data = response.json()

        return StagehandPage(
            client=self,
            page_id=data.get("page_id", "default"),
            url=url,
        )

    async def _execute_action(
        self,
        page_id: str,
        action: StagehandAction,
        instruction: str,
        schema: dict | None = None,
    ) -> dict:
        """Execute an action on a page."""
        payload = {
            "page_id": page_id,
            "action": action.value,
            "instruction": instruction,
        }
        if schema:
            payload["schema"] = schema

        response = await self._http_client.post(
            "/action",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def _get_page_state(self, page_id: str) -> dict:
        """Get the current state of a page."""
        response = await self._http_client.get(
            f"/page/{page_id}/state",
        )
        response.raise_for_status()
        return response.json()

    async def _capture_screenshot(self, page_id: str) -> bytes:
        """Capture a screenshot of the page."""
        response = await self._http_client.get(
            f"/page/{page_id}/screenshot",
        )
        response.raise_for_status()
        return response.content

    async def run_test(
        self,
        url: str,
        steps: list[str],
        assertions: list[dict] | None = None,
    ) -> dict:
        """Run a complete test with multiple steps."""
        response = await self._http_client.post(
            "/test/run",
            json={
                "url": url,
                "steps": steps,
                "assertions": assertions or [],
            },
        )
        response.raise_for_status()
        return response.json()


async def run_test_with_stagehand(
    url: str,
    steps: list[str],
    endpoint: str | None = None,
    api_token: str | None = None,
) -> dict:
    """Convenience function to run a test with Stagehand."""
    endpoint = endpoint or "https://stagehand.workers.dev"

    async with StagehandClient(endpoint=endpoint, api_token=api_token) as client:
        return await client.run_test(url=url, steps=steps)
