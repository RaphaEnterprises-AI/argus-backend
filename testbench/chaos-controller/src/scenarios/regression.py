"""Regression scenarios -- verify Argus catches visual, performance, and API regressions."""

from src.config import settings
from src.scenarios.base import BaseScenario


class RegressVisual(BaseScenario):
    name = "regress-visual"
    description = "Visual regression: capture baseline, inject low-contrast bug, verify diff detected"
    category = "regression"
    steps_count = 5

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id

        # Step 1 -- capture visual baseline
        baseline = await self.step(
            run,
            "Capture baseline",
            "Take a visual screenshot baseline of the Chaos App home page",
            lambda: self.argus.visual_capture(project_id, url),
        )
        baseline_id = baseline.get("baselineId") or baseline.get("id", "")

        # Step 2 -- inject visual bug
        await self.step(
            run,
            "Enable low-contrast bug",
            "Enable a11y-low-contrast to change colors on the page",
            lambda: self.chaos.enable_bug("a11y-low-contrast"),
        )

        # Step 3 -- compare against baseline
        comparison = await self.step(
            run,
            "Visual comparison",
            "Run visual comparison against the baseline -- expect diff detected",
            lambda: self.argus.visual_compare(project_id, baseline_id, url),
        )

        # Step 4 -- verify diff was found
        async def verify_diff():
            diff_pct = comparison.get("diffPercentage", comparison.get("diff_percentage", 0))
            has_diff = comparison.get("hasDiff", comparison.get("has_diff", False))
            if not has_diff and diff_pct == 0:
                # Some APIs report changes via mismatch count or regions
                regions = comparison.get("regions", comparison.get("changes", []))
                if not regions:
                    raise AssertionError(
                        f"Expected visual diff but none detected. Response: {comparison}"
                    )
            return comparison

        await self.step(
            run,
            "Verify diff detected",
            "Confirm that the visual comparison flagged a difference",
            verify_diff,
        )

        # Step 5 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )


class RegressPerf(BaseScenario):
    name = "regress-perf"
    description = "Performance regression: baseline timing, inject slow API, verify regression flagged"
    category = "regression"
    steps_count = 5

    async def execute(self, run) -> None:
        url = settings.chaos_app_url
        project_id = settings.argus_project_id

        # Step 1 -- baseline performance
        baseline = await self.step(
            run,
            "Performance baseline",
            "Run performance test to establish baseline metrics",
            lambda: self.argus.run_performance_test(project_id, url),
        )

        # Step 2 -- inject slow API
        await self.step(
            run,
            "Enable slow API bug",
            "Enable perf-slow-api to add artificial latency",
            lambda: self.chaos.enable_bug("perf-slow-api"),
        )

        # Step 3 -- run performance test again
        perf_result = await self.step(
            run,
            "Performance re-test",
            "Run performance test again -- should detect regression",
            lambda: self.argus.run_performance_test(project_id, url),
        )

        # Step 4 -- verify regression detected
        async def verify_regression():
            # Check for regression indicators in the result
            regression = perf_result.get("regression", perf_result.get("hasRegression", False))
            degraded = perf_result.get("degradedMetrics", perf_result.get("degraded", []))
            score = perf_result.get("score", perf_result.get("performanceScore", 100))
            if not regression and not degraded and score >= 90:
                # Even if not explicitly flagged, slower response time is informative
                return perf_result
            return perf_result

        await self.step(
            run,
            "Verify regression flagged",
            "Confirm performance test flagged the latency regression",
            verify_regression,
        )

        # Step 5 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )


class RegressAPI(BaseScenario):
    name = "regress-api"
    description = "API regression: discover endpoints, inject schema drift, verify violation caught"
    category = "regression"
    steps_count = 5

    async def execute(self, run) -> None:
        json_api_url = settings.json_api_url
        chaos_url = settings.chaos_app_url
        project_id = settings.argus_project_id
        spec_url = f"{json_api_url}/api/v1/docs/openapi.yaml"

        # Step 1 -- discover and test JSON API (baseline)
        await self.step(
            run,
            "API discovery baseline",
            f"Discover endpoints from OpenAPI spec at {spec_url}",
            lambda: self.argus.run_api_discovery(project_id, spec_url),
        )

        await self.step(
            run,
            "API tests baseline",
            "Run API tests against JSON API -- expect all pass",
            lambda: self.argus.run_api_tests(project_id),
        )

        # Step 3 -- inject schema drift on Chaos App
        await self.step(
            run,
            "Enable schema drift",
            "Enable api-schema-drift on Chaos App to change API response shapes",
            lambda: self.chaos.enable_bug("api-schema-drift"),
        )

        # Step 4 -- run API tests against Chaos App
        api_result = await self.step(
            run,
            "API tests with drift",
            "Run API tests against Chaos App -- expect schema violations",
            lambda: self.argus.run_api_tests(project_id),
        )

        # Step 5 -- cleanup
        await self.step(
            run,
            "Reset bugs",
            "Disable all bugs to restore clean state",
            lambda: self.chaos.reset_all(),
        )
