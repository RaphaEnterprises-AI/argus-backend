import httpx

from src.config import settings


class ArgusClient:
    """Client for Argus API -- simulates a real customer."""

    def __init__(self):
        self.base_url = settings.argus_url
        self.headers = {
            "X-API-Key": settings.argus_api_key,
            "X-Organization-Id": settings.argus_org_id,
            "Content-Type": "application/json",
        }
        self.timeout = httpx.Timeout(300.0, connect=15.0)

    async def _get(self, path: str, params: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}{path}",
                headers=self.headers,
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, json: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}{path}",
                headers=self.headers,
                json=json,
            )
            resp.raise_for_status()
            return resp.json()

    # Project management
    async def create_project(self, name: str, url: str) -> dict:
        return await self._post(
            "/api/v1/projects",
            json={
                "name": name,
                "url": url,
                "description": f"Testbench: {name}",
            },
        )

    # Discovery
    async def run_discovery(self, project_id: str, url: str) -> dict:
        return await self._post(
            "/api/v1/discovery/sessions",
            json={
                "projectId": project_id,
                "url": url,
                "maxPages": 20,
            },
        )

    async def get_discovery_status(self, session_id: str) -> dict:
        return await self._get(f"/api/v1/discovery/sessions/{session_id}")

    # Test generation
    async def generate_tests(
        self, project_id: str, surfaces: list | None = None
    ) -> dict:
        return await self._post(
            f"/api/v1/testgen/{settings.argus_org_id}/{project_id}/generate",
            json={
                "requirements": "Generate comprehensive tests for all discovered pages and flows",
                "framework": "playwright",
            },
        )

    # Test execution
    async def run_tests(self, project_id: str, url: str) -> dict:
        return await self._post(
            "/api/v1/browser/test",
            json={"projectId": project_id, "url": url},
        )

    async def get_test_results(self, run_id: str) -> dict:
        return await self._get(f"/api/v1/test-runs/{run_id}/results")

    # Visual AI
    async def visual_capture(self, project_id: str, url: str) -> dict:
        return await self._post(
            "/api/v1/visual/capture",
            json={"projectId": project_id, "url": url},
        )

    async def visual_compare(
        self, project_id: str, baseline_id: str, url: str
    ) -> dict:
        return await self._post(
            "/api/v1/visual/compare",
            json={
                "projectId": project_id,
                "baselineId": baseline_id,
                "url": url,
            },
        )

    # Performance
    async def run_performance_test(self, project_id: str, url: str) -> dict:
        return await self._post(
            "/api/v1/performance/tests",
            json={"projectId": project_id, "url": url},
        )

    # Accessibility
    async def run_accessibility_audit(self, project_id: str, url: str) -> dict:
        return await self._post(
            "/api/v1/accessibility/audit",
            json={"projectId": project_id, "url": url},
        )

    # API Testing
    async def run_api_discovery(self, project_id: str, spec_url: str) -> dict:
        return await self._post(
            "/api/v1/api-testing/discover",
            json={
                "projectId": project_id,
                "specUrl": spec_url,
                "discoveryMethod": "openapi",
            },
        )

    async def run_api_tests(self, project_id: str) -> dict:
        return await self._post(
            "/api/v1/api-testing/run",
            json={"projectId": project_id},
        )

    # Security
    async def run_pentest(self, project_id: str, url: str) -> dict:
        return await self._post(
            "/api/v1/pentest/scan",
            json={
                "projectId": project_id,
                "targetUrl": url,
                "scanType": "full",
            },
        )

    async def get_pentest_status(self, scan_id: str) -> dict:
        return await self._get(f"/api/v1/pentest/scan/{scan_id}")

    # Self-healing
    async def get_healing_patterns(self, org_id: str) -> dict:
        return await self._get(
            f"/api/v1/healing/organizations/{org_id}/patterns"
        )

    async def get_healing_stats(self, org_id: str) -> dict:
        return await self._get(
            f"/api/v1/healing/organizations/{org_id}/stats"
        )

    # Flaky detection
    async def get_flaky_tests(self, project_id: str) -> dict:
        return await self._get(
            "/api/v1/flaky", params={"projectId": project_id}
        )

    # Reporting
    async def generate_report(self, run_id: str) -> dict:
        return await self._post(f"/api/v1/test-runs/{run_id}/report")

    # Health
    async def health(self) -> dict:
        return await self._get("/health")
