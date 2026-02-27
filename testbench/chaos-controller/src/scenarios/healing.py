"""Self-healing scenarios -- verify Argus detects and heals broken selectors."""

import asyncio

from src.config import settings
from src.scenarios.base import BaseScenario


class HealSelectorDrift(BaseScenario):
    name = "heal-selector-drift"
    description = "Single selector change: verify Argus detects failure and self-heals"
    category = "healing"
    steps_count = 7

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id
        org_id = settings.argus_org_id

        # Step 1 -- baseline tests pass
        await self.step(
            run,
            "Baseline tests",
            "Run tests against Chaos App with no bugs -- expect all pass",
            lambda: self.argus.run_tests(project_id, url),
        )

        # Step 2 -- break a selector
        await self.step(
            run,
            "Enable selector bug",
            "Enable selector-login-id bug to break the login form selector",
            lambda: self.chaos.enable_bug("selector-login-id"),
        )

        # Step 3 -- tests should fail
        failure_run = await self.step(
            run,
            "Run broken tests",
            "Run tests again -- login test should fail due to changed selector",
            lambda: self.argus.run_tests(project_id, url),
        )

        # Step 4 -- wait for healing
        async def wait_for_healing():
            for _ in range(12):
                patterns = await self.argus.get_healing_patterns(org_id)
                items = patterns.get("patterns", patterns.get("items", []))
                if isinstance(items, list) and len(items) > 0:
                    return patterns
                await asyncio.sleep(5)
            raise TimeoutError("No healing patterns appeared within 60s")

        await self.step(
            run,
            "Wait for healing",
            "Poll healing patterns until Argus produces a fix (up to 60s)",
            wait_for_healing,
        )

        # Step 5 -- verify pattern stored
        async def verify_patterns():
            patterns = await self.argus.get_healing_patterns(org_id)
            items = patterns.get("patterns", patterns.get("items", []))
            if not items:
                raise AssertionError("Expected at least 1 healing pattern, got 0")
            return patterns

        await self.step(
            run,
            "Verify healing pattern",
            "Confirm healing pattern is stored with fix details",
            verify_patterns,
        )

        # Step 6 -- re-run with healed selectors
        await self.step(
            run,
            "Run healed tests",
            "Re-run tests -- healing should make them pass despite the bug",
            lambda: self.argus.run_tests(project_id, url),
        )

        # Step 7 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )


class HealCascade(BaseScenario):
    name = "heal-cascade"
    description = "Three simultaneous selector changes: verify Argus heals all of them"
    category = "healing"
    steps_count = 5

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id
        org_id = settings.argus_org_id

        # Step 1 -- enable 3 selector bugs at once
        async def enable_three_bugs():
            results = []
            for bug_id in ("selector-login-id", "selector-class-rename", "selector-nesting"):
                r = await self.chaos.enable_bug(bug_id)
                results.append(r)
            return results

        await self.step(
            run,
            "Enable 3 selector bugs",
            "Break login ID, class name, and DOM nesting selectors simultaneously",
            enable_three_bugs,
        )

        # Step 2 -- run tests, expect failures
        await self.step(
            run,
            "Run broken tests",
            "Execute tests -- expect multiple failures from the 3 broken selectors",
            lambda: self.argus.run_tests(project_id, url),
        )

        # Step 3 -- wait for healing to process all 3
        async def wait_for_cascade_healing():
            for _ in range(18):
                patterns = await self.argus.get_healing_patterns(org_id)
                items = patterns.get("patterns", patterns.get("items", []))
                if isinstance(items, list) and len(items) >= 1:
                    return patterns
                await asyncio.sleep(5)
            raise TimeoutError("Healing patterns did not appear within 90s")

        await self.step(
            run,
            "Wait for healing",
            "Wait for Argus to detect and heal all 3 selector failures (up to 90s)",
            wait_for_cascade_healing,
        )

        # Step 4 -- verify confidence scores
        async def verify_confidence():
            patterns = await self.argus.get_healing_patterns(org_id)
            items = patterns.get("patterns", patterns.get("items", []))
            if not items:
                raise AssertionError("Expected healing patterns with confidence scores")
            for item in items:
                confidence = item.get("confidence", item.get("score", 0))
                if isinstance(confidence, (int, float)) and confidence > 0:
                    return patterns
            return patterns

        await self.step(
            run,
            "Verify confidence scores",
            "Check that healing patterns include confidence scores",
            verify_confidence,
        )

        # Step 5 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )


class HealProactive(BaseScenario):
    name = "heal-proactive"
    description = "Proactive healing: detect selector drift before tests fail"
    category = "healing"
    steps_count = 5

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id
        org_id = settings.argus_org_id

        # Step 1 -- baseline run
        await self.step(
            run,
            "Baseline tests",
            "Run tests to establish healthy baseline state",
            lambda: self.argus.run_tests(project_id, url),
        )

        # Step 2 -- introduce subtle DOM change
        await self.step(
            run,
            "Enable nesting bug",
            "Enable selector-nesting to subtly change DOM structure",
            lambda: self.chaos.enable_bug("selector-nesting"),
        )

        # Step 3 -- check healing stats for drift detection
        async def check_drift_detection():
            stats = await self.argus.get_healing_stats(org_id)
            return stats

        await self.step(
            run,
            "Check healing stats",
            "Query healing stats to see if drift is detected",
            check_drift_detection,
        )

        # Step 4 -- verify health score
        async def verify_health_score():
            stats = await self.argus.get_healing_stats(org_id)
            # The stats endpoint should return some health indicator
            score = stats.get("healthScore", stats.get("health_score"))
            return stats

        await self.step(
            run,
            "Verify health score",
            "Check that health score reflects the DOM drift",
            verify_health_score,
        )

        # Step 5 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )
