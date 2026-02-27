"""Onboarding scenarios -- verify Argus can discover, generate, and run tests."""

import asyncio

from src.config import settings
from src.scenarios.base import BaseScenario


class OnboardConduit(BaseScenario):
    name = "onboard-conduit"
    description = "Onboard Conduit (RealWorld) app: discover, generate tests, execute, report"
    category = "onboarding"
    steps_count = 6

    async def execute(self, run) -> None:
        url = settings.conduit_url

        project = await self.step(
            run,
            "Create project",
            f"Create Argus project for Conduit at {url}",
            lambda: self.argus.create_project("Conduit Testbench", url),
        )
        project_id = project.get("id") or project.get("projectId") or settings.argus_project_id

        discovery = await self.step(
            run,
            "Run discovery",
            "Auto-crawl Conduit UI to discover pages and flows",
            lambda: self.argus.run_discovery(project_id, url),
        )
        session_id = discovery.get("sessionId") or discovery.get("id", "")

        # Poll until discovery completes (up to 120s)
        async def wait_for_discovery():
            for _ in range(24):
                status = await self.argus.get_discovery_status(session_id)
                state = status.get("status", "")
                if state in ("completed", "done", "finished"):
                    return status
                if state in ("failed", "error"):
                    raise RuntimeError(f"Discovery failed: {status.get('error', 'unknown')}")
                await asyncio.sleep(5)
            raise TimeoutError("Discovery did not complete within 120s")

        await self.step(
            run,
            "Wait for discovery",
            "Poll discovery session until crawl completes",
            wait_for_discovery,
        )

        await self.step(
            run,
            "Generate tests",
            "Generate test cases from discovered surfaces",
            lambda: self.argus.generate_tests(project_id),
        )

        test_run = await self.step(
            run,
            "Execute tests",
            "Run generated tests against Conduit -- expect all pass",
            lambda: self.argus.run_tests(project_id, url),
        )
        run_id = test_run.get("runId") or test_run.get("id", "")

        await self.step(
            run,
            "Generate report",
            "Generate a human-readable report for the test run",
            lambda: self.argus.generate_report(run_id),
        )


class OnboardPlane(BaseScenario):
    name = "onboard-plane"
    description = "Onboard Plane (project management) app: discover, generate tests, execute, report"
    category = "onboarding"
    steps_count = 6

    async def execute(self, run) -> None:
        url = settings.plane_url

        project = await self.step(
            run,
            "Create project",
            f"Create Argus project for Plane at {url}",
            lambda: self.argus.create_project("Plane Testbench", url),
        )
        project_id = project.get("id") or project.get("projectId") or settings.argus_project_id

        discovery = await self.step(
            run,
            "Run discovery",
            "Auto-crawl Plane UI to discover pages and flows",
            lambda: self.argus.run_discovery(project_id, url),
        )
        session_id = discovery.get("sessionId") or discovery.get("id", "")

        async def wait_for_discovery():
            for _ in range(24):
                status = await self.argus.get_discovery_status(session_id)
                state = status.get("status", "")
                if state in ("completed", "done", "finished"):
                    return status
                if state in ("failed", "error"):
                    raise RuntimeError(f"Discovery failed: {status.get('error', 'unknown')}")
                await asyncio.sleep(5)
            raise TimeoutError("Discovery did not complete within 120s")

        await self.step(
            run,
            "Wait for discovery",
            "Poll discovery session until crawl completes",
            wait_for_discovery,
        )

        await self.step(
            run,
            "Generate tests",
            "Generate test cases from discovered surfaces",
            lambda: self.argus.generate_tests(project_id),
        )

        test_run = await self.step(
            run,
            "Execute tests",
            "Run generated tests against Plane -- expect all pass",
            lambda: self.argus.run_tests(project_id, url),
        )
        run_id = test_run.get("runId") or test_run.get("id", "")

        await self.step(
            run,
            "Generate report",
            "Generate a human-readable report for the test run",
            lambda: self.argus.generate_report(run_id),
        )


class OnboardAPI(BaseScenario):
    name = "onboard-api"
    description = "Onboard JSON API server: OpenAPI discovery, generate API tests, execute, report"
    category = "onboarding"
    steps_count = 5

    async def execute(self, run) -> None:
        api_url = settings.json_api_url
        spec_url = f"{api_url}/api/v1/docs/openapi.yaml"

        project = await self.step(
            run,
            "Create project",
            f"Create Argus project for JSON API at {api_url}",
            lambda: self.argus.create_project("JSON API Testbench", api_url),
        )
        project_id = project.get("id") or project.get("projectId") or settings.argus_project_id

        await self.step(
            run,
            "API discovery",
            f"Discover endpoints from OpenAPI spec at {spec_url}",
            lambda: self.argus.run_api_discovery(project_id, spec_url),
        )

        await self.step(
            run,
            "Run API tests",
            "Execute generated API tests -- expect all pass",
            lambda: self.argus.run_api_tests(project_id),
        )

        test_run = await self.step(
            run,
            "Execute browser tests",
            "Run any UI-level tests against the API docs page",
            lambda: self.argus.run_tests(project_id, api_url),
        )
        run_id = test_run.get("runId") or test_run.get("id", "")

        await self.step(
            run,
            "Generate report",
            "Generate a human-readable report for the test run",
            lambda: self.argus.generate_report(run_id),
        )
