"""Report generation and export.

Generates comprehensive test reports in multiple formats.
"""

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class TestResultData:
    """Complete test result data for reporting."""
    test_id: str
    name: str
    status: str  # passed, failed, skipped
    duration_seconds: float
    error_message: str | None = None
    screenshots: list[str] = None  # Base64 encoded
    actions_taken: list[dict] = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    healing_applied: dict | None = None

    def __post_init__(self):
        self.screenshots = self.screenshots or []
        self.actions_taken = self.actions_taken or []


@dataclass
class ReportData:
    """Complete report data."""
    run_id: str
    app_url: str
    started_at: str
    completed_at: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    total_cost_usd: float
    test_results: list[TestResultData]
    failures: list[dict]
    security_summary: dict | None = None
    codebase_summary: str = ""


class ReportGenerator:
    """
    Generates test reports in multiple formats.

    Features:
    - JSON reports (machine-readable)
    - HTML reports (human-readable)
    - Markdown reports (for GitHub/docs)
    - Screenshot exports
    - JUnit XML (CI/CD integration)

    Usage:
        reporter = ReportGenerator(output_dir="./test-results")

        # Generate all reports
        paths = reporter.generate_all(report_data)
        print(f"Reports saved to: {paths}")

        # Generate specific format
        html_path = reporter.generate_html(report_data)
    """

    def __init__(
        self,
        output_dir: str = "./test-results",
        include_screenshots: bool = True,
        include_security: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_screenshots = include_screenshots
        self.include_security = include_security
        self.log = logger.bind(component="reporter")

    def generate_all(self, data: ReportData) -> dict[str, Path]:
        """Generate all report formats and return paths."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # JSON report
        paths["json"] = self.generate_json(data, run_dir / "results.json")

        # HTML report
        paths["html"] = self.generate_html(data, run_dir / "report.html")

        # Markdown report
        paths["markdown"] = self.generate_markdown(data, run_dir / "report.md")

        # JUnit XML
        paths["junit"] = self.generate_junit(data, run_dir / "junit.xml")

        # Screenshots
        if self.include_screenshots:
            screenshots_dir = run_dir / "screenshots"
            screenshots_dir.mkdir(exist_ok=True)
            self._save_screenshots(data, screenshots_dir)
            paths["screenshots"] = screenshots_dir

        self.log.info("Reports generated", output_dir=str(run_dir), formats=list(paths.keys()))

        return paths

    def generate_json(self, data: ReportData, output_path: Path | None = None) -> Path:
        """Generate JSON report."""
        output_path = output_path or self.output_dir / "results.json"

        # Convert to dict, excluding base64 screenshots to keep file small
        report_dict = {
            "run_id": data.run_id,
            "app_url": data.app_url,
            "started_at": data.started_at,
            "completed_at": data.completed_at,
            "summary": {
                "total": data.total_tests,
                "passed": data.passed,
                "failed": data.failed,
                "skipped": data.skipped,
                "pass_rate": f"{(data.passed / data.total_tests * 100):.1f}%" if data.total_tests > 0 else "N/A",
                "duration_seconds": data.duration_seconds,
                "total_cost_usd": data.total_cost_usd,
            },
            "test_results": [
                {
                    "test_id": r.test_id,
                    "name": r.name,
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "error_message": r.error_message,
                    "assertions_passed": r.assertions_passed,
                    "assertions_failed": r.assertions_failed,
                    "healing_applied": r.healing_applied,
                    "screenshot_count": len(r.screenshots),
                }
                for r in data.test_results
            ],
            "failures": data.failures,
        }

        if self.include_security and data.security_summary:
            report_dict["security"] = data.security_summary

        output_path.write_text(json.dumps(report_dict, indent=2))
        self.log.debug("Generated JSON report", path=str(output_path))
        return output_path

    def generate_html(self, data: ReportData, output_path: Path | None = None) -> Path:
        """Generate HTML report."""
        output_path = output_path or self.output_dir / "report.html"

        pass_rate = (data.passed / data.total_tests * 100) if data.total_tests > 0 else 0
        status_color = "#28a745" if data.failed == 0 else "#dc3545"
        status_text = "PASSED" if data.failed == 0 else "FAILED"

        # Build test rows
        test_rows = ""
        for r in data.test_results:
            status_badge = {
                "passed": '<span class="badge bg-success">PASSED</span>',
                "failed": '<span class="badge bg-danger">FAILED</span>',
                "skipped": '<span class="badge bg-secondary">SKIPPED</span>',
            }.get(r.status, '<span class="badge bg-warning">UNKNOWN</span>')

            error_cell = f'<small class="text-muted">{r.error_message[:100]}...</small>' if r.error_message else "-"

            test_rows += f"""
            <tr>
                <td><code>{r.test_id}</code></td>
                <td>{r.name}</td>
                <td>{status_badge}</td>
                <td>{r.duration_seconds:.2f}s</td>
                <td>{error_cell}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2E Test Report - {data.run_id[:8]}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .summary-card {{ border-left: 4px solid {status_color}; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; }}
        pre {{ background: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container py-4">
        <header class="mb-4">
            <h1>E2E Test Report</h1>
            <p class="text-muted">
                Run ID: <code>{data.run_id}</code> |
                App: <a href="{data.app_url}">{data.app_url}</a> |
                {data.started_at}
            </p>
        </header>

        <div class="card summary-card mb-4">
            <div class="card-body">
                <div class="row text-center">
                    <div class="col">
                        <div class="stat-value" style="color: {status_color}">{status_text}</div>
                        <div class="text-muted">Status</div>
                    </div>
                    <div class="col">
                        <div class="stat-value">{data.total_tests}</div>
                        <div class="text-muted">Total Tests</div>
                    </div>
                    <div class="col">
                        <div class="stat-value text-success">{data.passed}</div>
                        <div class="text-muted">Passed</div>
                    </div>
                    <div class="col">
                        <div class="stat-value text-danger">{data.failed}</div>
                        <div class="text-muted">Failed</div>
                    </div>
                    <div class="col">
                        <div class="stat-value">{pass_rate:.1f}%</div>
                        <div class="text-muted">Pass Rate</div>
                    </div>
                    <div class="col">
                        <div class="stat-value">{data.duration_seconds:.1f}s</div>
                        <div class="text-muted">Duration</div>
                    </div>
                    <div class="col">
                        <div class="stat-value">${data.total_cost_usd:.4f}</div>
                        <div class="text-muted">Cost</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card mb-4">
            <div class="card-header">
                <h5 class="mb-0">Test Results</h5>
            </div>
            <div class="card-body p-0">
                <table class="table table-hover mb-0">
                    <thead class="table-light">
                        <tr>
                            <th>Test ID</th>
                            <th>Name</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th>Error</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_rows}
                    </tbody>
                </table>
            </div>
        </div>

        {"" if not data.failures else self._generate_failures_html(data.failures)}

        {"" if not data.security_summary else self._generate_security_html(data.security_summary)}

        <footer class="text-center text-muted py-4">
            <small>Generated by E2E Testing Agent</small>
        </footer>
    </div>
</body>
</html>
"""

        output_path.write_text(html)
        self.log.debug("Generated HTML report", path=str(output_path))
        return output_path

    def _generate_failures_html(self, failures: list[dict]) -> str:
        """Generate HTML for failures section."""
        if not failures:
            return ""

        failure_cards = ""
        for f in failures:
            failure_cards += f"""
            <div class="card mb-2">
                <div class="card-body">
                    <h6 class="card-title"><code>{f.get('test_id', 'unknown')}</code></h6>
                    <p class="card-text"><strong>Type:</strong> {f.get('failure_type', 'unknown')}</p>
                    <p class="card-text"><strong>Root Cause:</strong> {f.get('root_cause', 'Unknown')}</p>
                </div>
            </div>
            """

        return f"""
        <div class="card mb-4">
            <div class="card-header bg-danger text-white">
                <h5 class="mb-0">Failures ({len(failures)})</h5>
            </div>
            <div class="card-body">
                {failure_cards}
            </div>
        </div>
        """

    def _generate_security_html(self, security: dict) -> str:
        """Generate HTML for security summary."""
        return f"""
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="mb-0">Security Summary</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <strong>Files Analyzed:</strong> {security.get('files_analyzed', 0)}
                    </div>
                    <div class="col-md-3">
                        <strong>Files Skipped:</strong> {security.get('files_skipped', 0)}
                    </div>
                    <div class="col-md-3">
                        <strong>Secrets Redacted:</strong> {security.get('secrets_redacted', 0)}
                    </div>
                </div>
            </div>
        </div>
        """

    def generate_markdown(self, data: ReportData, output_path: Path | None = None) -> Path:
        """Generate Markdown report."""
        output_path = output_path or self.output_dir / "report.md"

        pass_rate = (data.passed / data.total_tests * 100) if data.total_tests > 0 else 0
        status_emoji = ":white_check_mark:" if data.failed == 0 else ":x:"

        md = f"""# E2E Test Report

{status_emoji} **Status:** {"PASSED" if data.failed == 0 else "FAILED"}

| Metric | Value |
|--------|-------|
| Run ID | `{data.run_id}` |
| App URL | {data.app_url} |
| Started | {data.started_at} |
| Duration | {data.duration_seconds:.1f}s |
| Cost | ${data.total_cost_usd:.4f} |

## Summary

| Total | Passed | Failed | Skipped | Pass Rate |
|-------|--------|--------|---------|-----------|
| {data.total_tests} | {data.passed} | {data.failed} | {data.skipped} | {pass_rate:.1f}% |

## Test Results

| Test ID | Name | Status | Duration |
|---------|------|--------|----------|
"""

        for r in data.test_results:
            status = {"passed": ":white_check_mark:", "failed": ":x:", "skipped": ":fast_forward:"}.get(r.status, "?")
            md += f"| `{r.test_id}` | {r.name} | {status} | {r.duration_seconds:.2f}s |\n"

        if data.failures:
            md += "\n## Failures\n\n"
            for f in data.failures:
                md += f"""### {f.get('test_id', 'Unknown')}

- **Type:** {f.get('failure_type', 'unknown')}
- **Root Cause:** {f.get('root_cause', 'Unknown')}

"""

        if data.security_summary:
            md += f"""
## Security Summary

- **Files Analyzed:** {data.security_summary.get('files_analyzed', 0)}
- **Files Skipped:** {data.security_summary.get('files_skipped', 0)}
- **Secrets Redacted:** {data.security_summary.get('secrets_redacted', 0)}
"""

        md += "\n---\n*Generated by E2E Testing Agent*\n"

        output_path.write_text(md)
        self.log.debug("Generated Markdown report", path=str(output_path))
        return output_path

    def generate_junit(self, data: ReportData, output_path: Path | None = None) -> Path:
        """Generate JUnit XML report for CI/CD integration."""
        output_path = output_path or self.output_dir / "junit.xml"

        test_cases = ""
        for r in data.test_results:
            if r.status == "passed":
                test_cases += f'    <testcase name="{r.name}" classname="{r.test_id}" time="{r.duration_seconds:.3f}"/>\n'
            elif r.status == "failed":
                error_msg = r.error_message or "Unknown error"
                # Escape XML special characters
                error_msg = error_msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                test_cases += f'''    <testcase name="{r.name}" classname="{r.test_id}" time="{r.duration_seconds:.3f}">
        <failure message="Test failed">{error_msg}</failure>
    </testcase>
'''
            else:  # skipped
                test_cases += f'''    <testcase name="{r.name}" classname="{r.test_id}" time="{r.duration_seconds:.3f}">
        <skipped/>
    </testcase>
'''

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="E2E Tests" tests="{data.total_tests}" failures="{data.failed}" skipped="{data.skipped}" time="{data.duration_seconds:.3f}" timestamp="{data.started_at}">
{test_cases}</testsuite>
"""

        output_path.write_text(xml)
        self.log.debug("Generated JUnit XML report", path=str(output_path))
        return output_path

    def _save_screenshots(self, data: ReportData, screenshots_dir: Path) -> None:
        """Save screenshots to files."""
        for r in data.test_results:
            for i, screenshot_b64 in enumerate(r.screenshots):
                filename = f"{r.test_id}_step{i}.png"
                filepath = screenshots_dir / filename
                try:
                    screenshot_bytes = base64.b64decode(screenshot_b64)
                    filepath.write_bytes(screenshot_bytes)
                except Exception as e:
                    self.log.warning("Failed to save screenshot", file=filename, error=str(e))


def create_reporter(output_dir: str = "./test-results") -> ReportGenerator:
    """Factory function for ReportGenerator."""
    return ReportGenerator(output_dir=output_dir)


def create_report_from_state(state: dict) -> ReportData:
    """Create ReportData from orchestrator state."""
    test_results = []
    for r in state.get("test_results", []):
        test_results.append(TestResultData(
            test_id=r.get("test_id", "unknown"),
            name=r.get("test_id", "Unknown Test"),  # Use test_id as name if not available
            status=r.get("status", "unknown"),
            duration_seconds=r.get("duration_seconds", 0),
            error_message=r.get("error_message"),
            screenshots=r.get("screenshots", []),
            actions_taken=r.get("actions_taken", []),
            assertions_passed=r.get("assertions_passed", 0),
            assertions_failed=r.get("assertions_failed", 0),
            healing_applied=r.get("healing_applied"),
        ))

    return ReportData(
        run_id=state.get("run_id", "unknown"),
        app_url=state.get("app_url", ""),
        started_at=state.get("started_at", ""),
        completed_at=datetime.utcnow().isoformat(),
        total_tests=state.get("passed_count", 0) + state.get("failed_count", 0) + state.get("skipped_count", 0),
        passed=state.get("passed_count", 0),
        failed=state.get("failed_count", 0),
        skipped=state.get("skipped_count", 0),
        duration_seconds=state.get("iteration", 0) * 2,  # Rough estimate
        total_cost_usd=state.get("total_cost", 0),
        test_results=test_results,
        failures=state.get("failures", []),
        security_summary=state.get("security_summary"),
        codebase_summary=state.get("codebase_summary", ""),
    )
