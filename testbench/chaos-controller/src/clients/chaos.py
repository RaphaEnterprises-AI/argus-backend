import httpx

from src.config import settings


class ChaosClient:
    """Client for Chaos App bug toggle API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.chaos_app_url
        self.timeout = httpx.Timeout(30.0, connect=10.0)

    async def enable_bug(self, bug_id: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chaos/bugs/{bug_id}/enable"
            )
            resp.raise_for_status()
            return resp.json()

    async def disable_bug(self, bug_id: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chaos/bugs/{bug_id}/disable"
            )
            resp.raise_for_status()
            return resp.json()

    async def reset_all(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/chaos/bugs/reset")
            resp.raise_for_status()
            return resp.json()

    async def enable_scenario(self, scenario_name: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chaos/bugs/scenario/{scenario_name}"
            )
            resp.raise_for_status()
            return resp.json()

    async def list_bugs(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/chaos/bugs")
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/chaos/health")
            resp.raise_for_status()
            return resp.json()
