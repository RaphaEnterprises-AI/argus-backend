"""Playwright Test integration plugin.

Playwright is a framework for end-to-end testing that supports all modern browsers.
This plugin imports test results from Playwright JSON reporter output.

Documentation: https://playwright.dev/docs/test-reporters#json-reporter
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)

logger = logging.getLogger(__name__)


class PlaywrightPlugin(IntegrationPlugin):
    """Playwright Test integration for importing test results.

    Features:
    - Import Playwright JSON reporter output
    - Parse test suites, specs, and assertions
    - Extract screenshots and traces
    - Support for multiple browsers and projects

    Configuration:
    - Report path: Location of JSON report files
    - Artifacts path: Location of screenshots and traces

    Example:
        async with PlaywrightPlugin() as plugin:
            credentials = IntegrationCredentials(
                extra={
                    "report_path": "./playwright-report/results.json",
                    "artifacts_path": "./test-results"
                }
            )
            result = await plugin.test_connection(credentials)
            if result.success:
                data = await plugin.sync_data(credentials)
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get Playwright integration metadata."""
        return IntegrationMetadata(
            slug="playwright",
            name="Playwright Test",
            description="Import test results from Playwright Test framework",
            category=IntegrationCategory.TESTING,
            auth_type=AuthType.API_KEY,  # Using API_KEY type for local config
            icon_url="https://playwright.dev/img/playwright-logo.svg",
            docs_url="https://playwright.dev/docs/test-reporters",
            website_url="https://playwright.dev",
            features=[
                "test_results_import",
                "multi_browser_support",
                "screenshot_capture",
                "trace_viewer",
                "video_recording",
                "parallel_execution",
            ],
            required_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="Report Path",
                    type="text",
                    required=True,
                    placeholder="./playwright-report/results.json",
                    help_text="Path to Playwright JSON report file or directory",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra",
                    label="Configuration",
                    type="text",
                    required=False,
                    help_text="Additional configuration as JSON (artifacts_path, project_filter)",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=[
                "test_suites",
                "test_specs",
                "test_results",
                "screenshots",
                "traces",
            ],
        )

    def _get_config(self, credentials: IntegrationCredentials) -> dict[str, Any]:
        """Get configuration from credentials."""
        config = {
            "report_path": credentials.instance_url or "./playwright-report",
            "artifacts_path": "./test-results",
            "project_filter": None,
        }

        # Merge extra config if provided
        if credentials.extra:
            if isinstance(credentials.extra, str):
                try:
                    extra = json.loads(credentials.extra)
                    config.update(extra)
                except json.JSONDecodeError:
                    pass
            elif isinstance(credentials.extra, dict):
                config.update(credentials.extra)

        return config

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test Playwright configuration by validating report path.

        Args:
            credentials: Configuration with report path.

        Returns:
            ConnectionTestResult with validation status.
        """
        start_time = time.time()

        try:
            config = self._get_config(credentials)
            report_path = Path(config["report_path"])

            # Check if path exists
            if not report_path.exists():
                latency_ms = (time.time() - start_time) * 1000
                return ConnectionTestResult(
                    success=False,
                    message=f"Report path does not exist: {report_path}",
                    latency_ms=latency_ms,
                    error_code="PATH_NOT_FOUND",
                    error_details={"path": str(report_path)},
                )

            # Find JSON report files
            report_files = []
            if report_path.is_file() and report_path.suffix == ".json":
                report_files = [report_path]
            elif report_path.is_dir():
                report_files = list(report_path.glob("**/*.json"))
                # Filter to likely Playwright reports
                report_files = [
                    f for f in report_files
                    if self._is_playwright_report(f)
                ]

            latency_ms = (time.time() - start_time) * 1000

            if not report_files:
                return ConnectionTestResult(
                    success=False,
                    message="No Playwright JSON reports found at specified path",
                    latency_ms=latency_ms,
                    error_code="NO_REPORTS_FOUND",
                    error_details={"path": str(report_path)},
                )

            # Validate first report file
            sample_report = self._read_report(report_files[0])
            if not sample_report:
                return ConnectionTestResult(
                    success=False,
                    message="Failed to parse Playwright report",
                    latency_ms=latency_ms,
                    error_code="PARSE_ERROR",
                    error_details={"file": str(report_files[0])},
                )

            # Extract summary from report
            stats = sample_report.get("stats", {})
            config_data = sample_report.get("config", {})

            return ConnectionTestResult(
                success=True,
                message=f"Found {len(report_files)} Playwright report(s)",
                latency_ms=latency_ms,
                api_version=config_data.get("version", "unknown"),
                permissions=["read:reports", "read:artifacts"],
                account_info={
                    "report_count": len(report_files),
                    "report_files": [str(f) for f in report_files[:5]],
                    "total_tests": stats.get("expected", 0) + stats.get("unexpected", 0),
                    "passed": stats.get("expected", 0),
                    "failed": stats.get("unexpected", 0),
                    "flaky": stats.get("flaky", 0),
                    "skipped": stats.get("skipped", 0),
                    "duration_ms": stats.get("duration", 0),
                    "projects": [
                        p.get("name") for p in config_data.get("projects", [])
                    ],
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Playwright configuration test failed: {e}")
            return ConnectionTestResult(
                success=False,
                message=f"Configuration validation failed: {str(e)}",
                latency_ms=latency_ms,
                error_code="VALIDATION_ERROR",
                error_details={"error": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync test results from Playwright JSON reports.

        Args:
            credentials: Configuration with report path.
            resources: List of resources to sync.
            since: Import reports modified after this datetime.
            limit: Maximum number of items per resource.

        Returns:
            SyncResult with parsed test results.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or [
            "test_suites",
            "test_specs",
            "test_results",
            "screenshots",
        ]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        try:
            config = self._get_config(credentials)
            report_path = Path(config["report_path"])
            artifacts_path = Path(config.get("artifacts_path", "./test-results"))
            project_filter = config.get("project_filter")

            # Find report files
            report_files = []
            if report_path.is_file():
                report_files = [report_path]
            elif report_path.is_dir():
                report_files = [
                    f for f in report_path.glob("**/*.json")
                    if self._is_playwright_report(f)
                ]

            # Filter by modification time if since is provided
            if since:
                report_files = [
                    f for f in report_files
                    if datetime.fromtimestamp(f.stat().st_mtime) > since
                ]

            if not report_files:
                return SyncResult(
                    success=False,
                    message="No Playwright reports found",
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                    errors=["No report files found at specified path"],
                )

            # Parse all reports
            all_suites = []
            all_specs = []
            all_results = []
            all_screenshots = []
            all_traces = []

            for report_file in report_files[:limit]:
                report = self._read_report(report_file)
                if not report:
                    warnings.append(f"Failed to parse report: {report_file}")
                    continue

                # Parse test suites
                suites = report.get("suites", [])
                for suite in suites:
                    parsed_suite, specs, results = self._parse_suite(
                        suite, str(report_file), project_filter
                    )
                    if parsed_suite:
                        all_suites.append(parsed_suite)
                        all_specs.extend(specs)
                        all_results.extend(results)

                # Find artifacts
                if "screenshots" in sync_resources or "traces" in sync_resources:
                    screenshots, traces = await self._find_artifacts(
                        artifacts_path, report
                    )
                    all_screenshots.extend(screenshots)
                    all_traces.extend(traces)

            # Populate data
            if "test_suites" in sync_resources:
                data["test_suites"] = all_suites[:limit]
                total_synced += len(data["test_suites"])

            if "test_specs" in sync_resources:
                data["test_specs"] = all_specs[:limit]
                total_synced += len(data["test_specs"])

            if "test_results" in sync_resources:
                data["test_results"] = all_results[:limit]
                total_synced += len(data["test_results"])

            if "screenshots" in sync_resources:
                data["screenshots"] = all_screenshots[:limit]
                total_synced += len(data["screenshots"])

            if "traces" in sync_resources:
                data["traces"] = all_traces[:limit]
                total_synced += len(data["traces"])

            # Calculate summary statistics
            data["summary"] = self._calculate_summary(all_results)

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from Playwright reports",
                data_points_synced=total_synced,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            logger.error(f"Playwright sync failed: {e}")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    def _is_playwright_report(self, file_path: Path) -> bool:
        """Check if a JSON file is a Playwright report."""
        try:
            with open(file_path, "r") as f:
                # Read just enough to check structure
                content = f.read(1000)
                # Playwright reports have specific keys
                return (
                    '"suites"' in content or
                    '"stats"' in content or
                    '"config"' in content and '"projects"' in content
                )
        except Exception:
            return False

    def _read_report(self, file_path: Path) -> Optional[dict]:
        """Read and parse a Playwright JSON report."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read Playwright report {file_path}: {e}")
            return None

    def _parse_suite(
        self,
        suite: dict,
        report_file: str,
        project_filter: Optional[str] = None,
    ) -> tuple[Optional[dict], list[dict], list[dict]]:
        """Parse a test suite from Playwright report.

        Returns:
            Tuple of (suite_data, specs, results)
        """
        specs = []
        results = []

        # Check project filter
        project = suite.get("project", {})
        project_name = project.get("name", "")
        if project_filter and project_name != project_filter:
            return None, [], []

        suite_id = suite.get("id", f"suite-{hash(suite.get('title', ''))}")
        suite_data = {
            "id": suite_id,
            "title": suite.get("title"),
            "file": suite.get("file"),
            "line": suite.get("line"),
            "column": suite.get("column"),
            "project": project_name,
            "report_file": report_file,
        }

        # Parse specs (test cases)
        for spec in suite.get("specs", []):
            spec_id = spec.get("id", f"spec-{hash(spec.get('title', ''))}")
            spec_data = {
                "id": spec_id,
                "suite_id": suite_id,
                "title": spec.get("title"),
                "file": spec.get("file"),
                "line": spec.get("line"),
                "column": spec.get("column"),
                "tags": spec.get("tags", []),
                "ok": spec.get("ok", True),
            }
            specs.append(spec_data)

            # Parse test results (can have multiple retries)
            for test in spec.get("tests", []):
                for result in test.get("results", []):
                    result_data = self._parse_result(
                        result, spec_id, test, project_name
                    )
                    results.append(result_data)

        # Recursively parse nested suites
        for nested_suite in suite.get("suites", []):
            nested_data, nested_specs, nested_results = self._parse_suite(
                nested_suite, report_file, project_filter
            )
            if nested_data:
                specs.extend(nested_specs)
                results.extend(nested_results)

        return suite_data, specs, results

    def _parse_result(
        self,
        result: dict,
        spec_id: str,
        test: dict,
        project_name: str,
    ) -> dict:
        """Parse a single test result."""
        result_id = f"result-{spec_id}-{result.get('retry', 0)}"

        # Determine status
        status = result.get("status", "unknown")
        if status == "passed":
            outcome = "passed"
        elif status == "failed":
            outcome = "failed"
        elif status == "timedOut":
            outcome = "timeout"
        elif status == "skipped":
            outcome = "skipped"
        else:
            outcome = status

        # Extract error information
        error = result.get("error", {})
        error_message = error.get("message", "")
        error_stack = error.get("stack", "")

        # Extract attachments
        attachments = result.get("attachments", [])
        screenshots = [
            a for a in attachments if a.get("contentType", "").startswith("image/")
        ]
        traces = [
            a for a in attachments if a.get("name") == "trace"
        ]
        videos = [
            a for a in attachments if a.get("contentType", "").startswith("video/")
        ]

        return {
            "id": result_id,
            "spec_id": spec_id,
            "project": project_name,
            "status": status,
            "outcome": outcome,
            "duration_ms": result.get("duration", 0),
            "retry": result.get("retry", 0),
            "worker_index": result.get("workerIndex", 0),
            "parallel_index": result.get("parallelIndex", 0),
            "start_time": result.get("startTime"),
            "error_message": error_message[:500] if error_message else None,
            "error_stack": error_stack[:2000] if error_stack else None,
            "steps": self._parse_steps(result.get("steps", [])),
            "attachments_count": len(attachments),
            "screenshots_count": len(screenshots),
            "has_trace": len(traces) > 0,
            "has_video": len(videos) > 0,
            "stdout": result.get("stdout", [])[:10],  # Limit stdout lines
            "stderr": result.get("stderr", [])[:10],  # Limit stderr lines
        }

    def _parse_steps(self, steps: list) -> list[dict]:
        """Parse test steps from result."""
        parsed = []
        for step in steps[:50]:  # Limit to 50 steps
            parsed.append({
                "title": step.get("title"),
                "category": step.get("category"),
                "duration_ms": step.get("duration", 0),
                "error": step.get("error", {}).get("message") if step.get("error") else None,
            })
        return parsed

    async def _find_artifacts(
        self,
        artifacts_path: Path,
        report: dict,
    ) -> tuple[list[dict], list[dict]]:
        """Find screenshots and traces in artifacts directory."""
        screenshots = []
        traces = []

        if not artifacts_path.exists():
            return screenshots, traces

        # Find screenshot files
        for screenshot_file in artifacts_path.glob("**/*.png"):
            try:
                stat = screenshot_file.stat()
                screenshots.append({
                    "path": str(screenshot_file),
                    "filename": screenshot_file.name,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                })
            except Exception:
                pass

        # Find trace files
        for trace_file in artifacts_path.glob("**/*.zip"):
            if "trace" in trace_file.name.lower():
                try:
                    stat = trace_file.stat()
                    traces.append({
                        "path": str(trace_file),
                        "filename": trace_file.name,
                        "size_bytes": stat.st_size,
                        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    })
                except Exception:
                    pass

        return screenshots, traces

    def _calculate_summary(self, results: list[dict]) -> dict:
        """Calculate summary statistics from results."""
        total = len(results)
        passed = sum(1 for r in results if r["outcome"] == "passed")
        failed = sum(1 for r in results if r["outcome"] == "failed")
        skipped = sum(1 for r in results if r["outcome"] == "skipped")
        timeout = sum(1 for r in results if r["outcome"] == "timeout")
        total_duration_ms = sum(r.get("duration_ms", 0) for r in results)

        # Calculate flaky tests (passed after retry)
        flaky_specs = set()
        for result in results:
            if result["retry"] > 0 and result["outcome"] == "passed":
                flaky_specs.add(result["spec_id"])

        # Group by project
        by_project = {}
        for result in results:
            project = result.get("project", "default")
            if project not in by_project:
                by_project[project] = {"passed": 0, "failed": 0, "total": 0}
            by_project[project]["total"] += 1
            if result["outcome"] == "passed":
                by_project[project]["passed"] += 1
            elif result["outcome"] == "failed":
                by_project[project]["failed"] += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "timeout": timeout,
            "flaky_count": len(flaky_specs),
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration_ms": total_duration_ms,
            "average_duration_ms": (total_duration_ms / total) if total > 0 else 0,
            "by_project": by_project,
        }

    async def parse_report_file(
        self,
        report_path: str,
    ) -> Optional[dict]:
        """Parse a single Playwright report file.

        Args:
            report_path: Path to JSON report file.

        Returns:
            Parsed report data or None.
        """
        path = Path(report_path)
        if not path.exists():
            return None

        report = self._read_report(path)
        if not report:
            return None

        # Parse all suites
        all_specs = []
        all_results = []

        for suite in report.get("suites", []):
            _, specs, results = self._parse_suite(suite, str(path))
            all_specs.extend(specs)
            all_results.extend(results)

        return {
            "config": report.get("config", {}),
            "stats": report.get("stats", {}),
            "specs": all_specs,
            "results": all_results,
            "summary": self._calculate_summary(all_results),
        }
