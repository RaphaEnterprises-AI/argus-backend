"""Golden path scenario -- full integration journey through all Argus capabilities."""

import asyncio

from src.config import settings
from src.scenarios.base import BaseScenario


class GoldenPath(BaseScenario):
    name = "golden-path"
    description = (
        "Full integration journey: create project, discover, generate tests, "
        "execute, visual baseline, inject 5 bugs, detect failures, heal, "
        "visual regression, performance test, security scan, report, reset"
    )
    category = "integration"
    steps_count = 14

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        org_id = settings.argus_org_id

        # Step 1 -- Create project
        project = await self.step(
            run,
            "Create project",
            f"Create Argus project for Chaos App at {url}",
            lambda: self.argus.create_project("Golden Path Testbench", url),
        )
        project_id = project.get("id") or project.get("projectId") or settings.argus_project_id

        # Step 2 -- Discovery
        discovery = await self.step(
            run,
            "Run discovery",
            "Auto-crawl the Chaos App to discover pages and flows",
            lambda: self.argus.run_discovery(project_id, url),
        )
        session_id = discovery.get("sessionId") or discovery.get("id", "")

        # Poll until discovery completes
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

        # Step 3 -- Generate tests
        await self.step(
            run,
            "Generate tests",
            "Generate test cases from discovered surfaces",
            lambda: self.argus.generate_tests(project_id),
        )

        # Step 4 -- Execute (all pass)
        clean_run = await self.step(
            run,
            "Execute tests (clean)",
            "Run tests with no bugs enabled -- expect all pass",
            lambda: self.argus.run_tests(project_id, url),
        )
        clean_run_id = clean_run.get("runId") or clean_run.get("id", "")

        # Step 5 -- Capture visual baseline
        baseline = await self.step(
            run,
            "Capture visual baseline",
            "Take a visual screenshot baseline of the Chaos App",
            lambda: self.argus.visual_capture(project_id, url),
        )
        baseline_id = baseline.get("baselineId") or baseline.get("id", "")

        # Step 6 -- Enable 5 bugs (one per category)
        async def enable_five_bugs():
            bugs = [
                "selector-login-id",    # healing
                "a11y-low-contrast",    # visual
                "perf-slow-api",        # performance
                "sec-xss-reflected",    # security
                "flaky-timing",         # flaky
            ]
            results = {}
            for bug_id in bugs:
                results[bug_id] = await self.chaos.enable_bug(bug_id)
            return results

        await self.step(
            run,
            "Enable 5 bugs",
            "Inject one bug per category: selector, visual, perf, security, flaky",
            enable_five_bugs,
        )

        # Step 7 -- Execute (failures expected)
        broken_run = await self.step(
            run,
            "Execute tests (broken)",
            "Run tests with 5 bugs active -- expect failures",
            lambda: self.argus.run_tests(project_id, url),
        )

        # Step 8 -- Check healing
        async def check_healing():
            # Give healing time to process
            for _ in range(12):
                patterns = await self.argus.get_healing_patterns(org_id)
                items = patterns.get("patterns", patterns.get("items", []))
                if isinstance(items, list) and len(items) > 0:
                    return patterns
                await asyncio.sleep(5)
            # Return whatever we have even if empty
            return await self.argus.get_healing_patterns(org_id)

        await self.step(
            run,
            "Check healing",
            "Wait for Argus self-healing to produce fix patterns (up to 60s)",
            check_healing,
        )

        # Step 9 -- Visual regression
        await self.step(
            run,
            "Visual regression check",
            "Compare current screenshots against the clean baseline",
            lambda: self.argus.visual_compare(project_id, baseline_id, url),
        )

        # Step 10 -- Performance test
        await self.step(
            run,
            "Performance test",
            "Run performance test with perf-slow-api bug active",
            lambda: self.argus.run_performance_test(project_id, url),
        )

        # Step 11 -- Security scan
        await self.step(
            run,
            "Security scan",
            "Run pentest with sec-xss-reflected bug active",
            lambda: self.argus.run_pentest(project_id, url),
        )

        # Step 12 -- Generate report
        broken_run_id = broken_run.get("runId") or broken_run.get("id", "") or clean_run_id

        await self.step(
            run,
            "Generate report",
            "Generate a comprehensive report covering all findings",
            lambda: self.argus.generate_report(broken_run_id),
        )

        # Step 13 -- Reset all
        await self.step(
            run,
            "Reset all bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )

        # Step 14 -- Execute (all pass again)
        await self.step(
            run,
            "Execute tests (clean again)",
            "Run tests after reset -- confirm everything passes again",
            lambda: self.argus.run_tests(project_id, url),
        )
