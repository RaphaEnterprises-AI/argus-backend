"""Reporter Agent - Generates test reports and notifications.

This agent:
- Generates human-readable test reports
- Creates GitHub issues for failures
- Sends notifications via various channels
- Provides executive summaries
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

from .base import AgentCapability, AgentResult, BaseAgent
from .prompts import get_enhanced_prompt


@dataclass
class TestSummary:
    """Summary of test execution."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total_duration_ms: int = 0
    pass_rate: float = 0.0

    def __post_init__(self):
        if self.total_tests > 0:
            self.pass_rate = (self.passed / self.total_tests) * 100


@dataclass
class FailureReport:
    """Detailed report of a test failure."""

    test_id: str
    test_name: str
    failure_type: str
    error_message: str
    step_failed: int | None = None
    screenshot_path: str | None = None
    suggested_fix: str | None = None
    priority: str = "medium"


@dataclass
class TestReport:
    """Complete test execution report."""

    run_id: str
    timestamp: str
    summary: TestSummary
    failures: list[FailureReport] = field(default_factory=list)
    detailed_results: list[dict] = field(default_factory=list)
    execution_metadata: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    markdown_report: str = ""
    html_report: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "summary": {
                "total_tests": self.summary.total_tests,
                "passed": self.summary.passed,
                "failed": self.summary.failed,
                "errors": self.summary.errors,
                "skipped": self.summary.skipped,
                "pass_rate": self.summary.pass_rate,
                "duration_ms": self.summary.total_duration_ms,
            },
            "failures": [
                {
                    "test_id": f.test_id,
                    "test_name": f.test_name,
                    "error": f.error_message,
                    "priority": f.priority,
                }
                for f in self.failures
            ],
            "recommendations": self.recommendations,
        }


class ReporterAgent(BaseAgent):
    """Agent that generates comprehensive test reports.

    Features:
    - Markdown and HTML report generation
    - Executive summary creation
    - Trend analysis
    - GitHub issue creation
    - Slack/email notifications
    """

    # RAP-231: Agent capabilities for A2A discovery
    # ReporterAgent has no specific testing capabilities to advertise
    CAPABILITIES: list[str] = []

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for reporting."""
        enhanced = get_enhanced_prompt("reporter")
        if enhanced:
            return enhanced

        return """You are an expert test reporting agent. Generate clear, actionable test reports.

When creating reports:
1. Lead with the most important information (pass/fail summary)
2. Highlight critical failures prominently
3. Group related failures together
4. Provide actionable recommendations
5. Include relevant metrics and trends

Reports should be:
- Scannable at a glance
- Detailed where needed
- Actionable with clear next steps
- Professional and objective"""

    async def execute(
        self,
        test_results: list[dict],
        execution_metadata: dict | None = None,
        include_recommendations: bool = True,
    ) -> AgentResult[TestReport]:
        """Generate a test report from results.

        Args:
            test_results: List of test result dictionaries
            execution_metadata: Optional metadata about the run
            include_recommendations: Whether to generate AI recommendations

        Returns:
            AgentResult containing TestReport
        """
        run_id = execution_metadata.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

        self.log.info(
            "Generating test report",
            run_id=run_id,
            result_count=len(test_results),
        )

        # Calculate summary
        summary = self._calculate_summary(test_results)

        # Extract failures
        failures = self._extract_failures(test_results)

        # Generate markdown report
        markdown_report = self._generate_markdown_report(
            run_id, summary, failures, test_results, execution_metadata
        )

        # Generate HTML report
        html_report = self._generate_html_report(
            run_id, summary, failures, test_results, execution_metadata
        )

        # Generate recommendations using Claude
        recommendations = []
        if include_recommendations and failures:
            recommendations = await self._generate_recommendations(
                summary, failures, test_results
            )

        report = TestReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            summary=summary,
            failures=failures,
            detailed_results=test_results,
            execution_metadata=execution_metadata or {},
            recommendations=recommendations,
            markdown_report=markdown_report,
            html_report=html_report,
        )

        self.log.info(
            "Report generated",
            run_id=run_id,
            pass_rate=summary.pass_rate,
            failures_count=len(failures),
        )

        return AgentResult(
            success=True,
            data=report,
        )

    def _calculate_summary(self, results: list[dict]) -> TestSummary:
        """Calculate test summary statistics."""
        total = len(results)
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        errors = sum(1 for r in results if r.get("status") == "error")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        duration = sum(r.get("duration_ms", 0) for r in results)

        return TestSummary(
            total_tests=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total_duration_ms=duration,
        )

    def _extract_failures(self, results: list[dict]) -> list[FailureReport]:
        """Extract failure reports from results."""
        failures = []

        for result in results:
            if result.get("status") in ("failed", "error"):
                failures.append(
                    FailureReport(
                        test_id=result.get("test_id", "unknown"),
                        test_name=result.get("test_name", "Unknown Test"),
                        failure_type=result.get("failure_type", "unknown"),
                        error_message=result.get("error_message", "No error message"),
                        step_failed=result.get("failed_step_index"),
                        screenshot_path=result.get("screenshot_path"),
                        suggested_fix=result.get("suggested_fix"),
                        priority=result.get("priority", "medium"),
                    )
                )

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        failures.sort(key=lambda f: priority_order.get(f.priority, 4))

        return failures

    def _generate_markdown_report(
        self,
        run_id: str,
        summary: TestSummary,
        failures: list[FailureReport],
        results: list[dict],
        metadata: dict | None,
    ) -> str:
        """Generate a markdown test report."""
        lines = [
            "# E2E Test Report",
            "",
            f"**Run ID:** {run_id}",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {summary.total_tests} |",
            f"| Passed | {summary.passed} |",
            f"| Failed | {summary.failed} |",
            f"| Errors | {summary.errors} |",
            f"| Pass Rate | {summary.pass_rate:.1f}% |",
            f"| Duration | {summary.total_duration_ms}ms |",
            "",
        ]

        # Status indicator
        if summary.pass_rate == 100:
            lines.append("> ✅ **All tests passed!**")
        elif summary.pass_rate >= 80:
            lines.append(f"> ⚠️ **{summary.failed} test(s) failed**")
        else:
            lines.append(f"> ❌ **Critical: {summary.failed} test(s) failed ({summary.pass_rate:.1f}% pass rate)**")

        lines.append("")

        # Failures section
        if failures:
            lines.extend([
                f"## Failures ({len(failures)})",
                "",
            ])

            for failure in failures:
                priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(failure.priority, "⚪")

                lines.extend([
                    f"### {priority_emoji} {failure.test_name}",
                    "",
                    f"- **Test ID:** `{failure.test_id}`",
                    f"- **Error:** {failure.error_message}",
                ])

                if failure.step_failed is not None:
                    lines.append(f"- **Failed at step:** {failure.step_failed}")

                if failure.suggested_fix:
                    lines.append(f"- **Suggested fix:** {failure.suggested_fix}")

                lines.append("")

        # Metadata
        if metadata:
            lines.extend([
                "## Execution Details",
                "",
                "```json",
                json.dumps(metadata, indent=2),
                "```",
                "",
            ])

        # Footer
        lines.extend([
            "---",
            "*Generated by E2E Testing Agent*",
        ])

        return "\n".join(lines)

    def _generate_html_report(
        self,
        run_id: str,
        summary: TestSummary,
        failures: list[FailureReport],
        results: list[dict],
        metadata: dict | None,
    ) -> str:
        """Generate an HTML test report."""
        status_color = "#28a745" if summary.pass_rate == 100 else "#dc3545" if summary.pass_rate < 80 else "#ffc107"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>E2E Test Report - {run_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #6c757d; font-size: 14px; }}
        .summary-card .value {{ font-size: 32px; font-weight: bold; }}
        .pass-rate {{ color: {status_color}; }}
        .failure {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }}
        .failure.critical {{ background: #f8d7da; border-left-color: #dc3545; }}
        .failure h4 {{ margin: 0 0 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>E2E Test Report</h1>
    <p><strong>Run ID:</strong> {run_id} | <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <div class="summary-card">
            <h3>Total Tests</h3>
            <div class="value">{summary.total_tests}</div>
        </div>
        <div class="summary-card">
            <h3>Pass Rate</h3>
            <div class="value pass-rate">{summary.pass_rate:.1f}%</div>
        </div>
        <div class="summary-card">
            <h3>Duration</h3>
            <div class="value">{summary.total_duration_ms / 1000:.1f}s</div>
        </div>
    </div>

    <h2>Results</h2>
    <table>
        <tr><th>Test</th><th>Status</th><th>Duration</th></tr>
"""

        for result in results:
            status = result.get("status", "unknown")
            status_class = "passed" if status == "passed" else "failed"
            duration = result.get("duration_ms", 0)

            html += f"""        <tr>
            <td>{result.get('test_name', 'Unknown')}</td>
            <td class="{status_class}">{status.upper()}</td>
            <td>{duration}ms</td>
        </tr>
"""

        html += """    </table>
"""

        if failures:
            html += f"""    <h2>Failures ({len(failures)})</h2>
"""
            for failure in failures:
                failure_class = "failure critical" if failure.priority == "critical" else "failure"
                html += f"""    <div class="{failure_class}">
        <h4>{failure.test_name}</h4>
        <p><strong>Error:</strong> {failure.error_message}</p>
    </div>
"""

        html += """</body>
</html>"""

        return html

    async def _generate_recommendations(
        self,
        summary: TestSummary,
        failures: list[FailureReport],
        results: list[dict],
    ) -> list[str]:
        """Generate AI-powered recommendations."""
        if not self._check_cost_limit():
            return ["Unable to generate recommendations (cost limit reached)"]

        prompt = f"""Analyze these test results and provide actionable recommendations.

SUMMARY:
- Total: {summary.total_tests}, Passed: {summary.passed}, Failed: {summary.failed}
- Pass Rate: {summary.pass_rate:.1f}%

FAILURES:
{json.dumps([{"name": f.test_name, "error": f.error_message, "priority": f.priority} for f in failures[:10]], indent=2)}

Provide 3-5 specific, actionable recommendations to improve test reliability.
Respond with JSON array of strings: ["recommendation 1", "recommendation 2", ...]"""

        try:
            response = self._call_claude(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )

            content = self._extract_text_response(response)
            recommendations = self._parse_json_response(content, [])

            if isinstance(recommendations, list):
                return recommendations[:5]

            return []

        except Exception as e:
            self.log.warning("Failed to generate recommendations", error=str(e))
            return []

    async def create_github_issue(
        self,
        failure: FailureReport,
        repo: str,
        labels: list[str] | None = None,
    ) -> str | None:
        """Create a GitHub issue for a failure.

        Args:
            failure: The failure to report
            repo: Repository in format "owner/repo"
            labels: Optional issue labels

        Returns:
            Issue URL if created, None otherwise
        """
        # This would integrate with GitHub API
        # For now, generate the issue body
        issue_body = f"""## Test Failure Report

**Test:** {failure.test_name}
**Test ID:** `{failure.test_id}`
**Priority:** {failure.priority}

### Error
```
{failure.error_message}
```

### Details
- Failed at step: {failure.step_failed or 'N/A'}
- Failure type: {failure.failure_type}

### Suggested Fix
{failure.suggested_fix or 'No automated suggestion available'}

---
*Created by E2E Testing Agent*
"""
        self.log.info(
            "GitHub issue body generated",
            test_id=failure.test_id,
        )

        return issue_body

    def save_report(
        self,
        report: TestReport,
        output_dir: str,
        formats: list[str] | None = None,
    ) -> dict[str, str]:
        """Save report to files.

        Args:
            report: The report to save
            output_dir: Directory to save to
            formats: List of formats (json, markdown, html)

        Returns:
            Dict of format to file path
        """
        import os

        formats = formats or ["json", "markdown", "html"]
        saved_files = {}

        os.makedirs(output_dir, exist_ok=True)

        if "json" in formats:
            path = os.path.join(output_dir, f"report_{report.run_id}.json")
            with open(path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            saved_files["json"] = path

        if "markdown" in formats:
            path = os.path.join(output_dir, f"report_{report.run_id}.md")
            with open(path, "w") as f:
                f.write(report.markdown_report)
            saved_files["markdown"] = path

        if "html" in formats:
            path = os.path.join(output_dir, f"report_{report.run_id}.html")
            with open(path, "w") as f:
                f.write(report.html_report)
            saved_files["html"] = path

        self.log.info("Reports saved", files=saved_files)
        return saved_files
