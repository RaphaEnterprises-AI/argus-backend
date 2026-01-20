#!/usr/bin/env python3
"""
Real Integration Test Runner

Runs comprehensive integration tests against actual infrastructure:
- Infrastructure health (browser pool, database, memory store, Claude API)
- Orchestrator E2E flows (code analysis, test planning, self-healing, NLP)
- Browser pool operations (observe, act, test, extract)

Usage:
    # Run all tests
    python scripts/run_integration_tests.py --all

    # Run specific categories
    python scripts/run_integration_tests.py --infra
    python scripts/run_integration_tests.py --orchestrator
    python scripts/run_integration_tests.py --browser

    # Generate JSON report
    python scripts/run_integration_tests.py --all --report results.json

    # Monitor performance
    python scripts/run_integration_tests.py --all --monitor

Environment Variables Required:
    ANTHROPIC_API_KEY - For Claude API tests
    DATABASE_URL - For database tests (optional)
    BROWSER_POOL_URL - For browser pool tests (optional)
    BROWSER_POOL_JWT_SECRET - For browser pool auth (optional)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_banner():
    """Print test runner banner."""
    print()
    print("=" * 70)
    print("   🧪 ARGUS E2E TESTING AGENT - INTEGRATION TESTS")
    print("=" * 70)
    print()
    print("   Testing Real Infrastructure & Agent Performance")
    print("   No Mocks - Actual API Calls & Browser Automation")
    print()
    print("-" * 70)


def print_section(title: str, icon: str = "📊"):
    """Print section header."""
    print()
    print(f"{icon} {title}")
    print("-" * 50)


def print_result(name: str, passed: bool, latency_ms: float, details: str = ""):
    """Print individual test result."""
    status = "✅" if passed else "❌"
    latency_str = f"{latency_ms:.0f}ms" if latency_ms > 0 else "N/A"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name:<40} [{latency_str}]{detail_str}")


def print_summary(report: dict):
    """Print test summary."""
    print()
    print("=" * 70)
    print("   📊 TEST SUMMARY")
    print("=" * 70)

    summary = report.get("summary", {})
    print(f"""
   Total Tests:     {summary.get('total_tests', 0)}
   Passed:          {summary.get('passed', 0)}
   Failed:          {summary.get('failed', 0)}
   Pass Rate:       {summary.get('pass_rate', '0%')}

   Performance:
   ├── Total Time:      {summary.get('total_latency_ms', 0):.0f}ms
   ├── Avg Time/Test:   {summary.get('avg_latency_ms', 0):.0f}ms
   └── Total Cost:      {summary.get('total_cost', '$0.0000')}
""")

    # Print by category
    by_category = report.get("by_category", {})
    if by_category:
        print("   Results by Category:")
        for cat, data in by_category.items():
            passed = data.get('passed', 0)
            total = data.get('total', 0)
            print(f"   ├── {cat:<20} {passed}/{total} passed")

    print("=" * 70)


@dataclass
class PerformanceMetrics:
    """Tracks performance metrics during test execution."""
    start_time: float = field(default_factory=time.perf_counter)
    api_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def record_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def record_api_call(self, tokens: int = 0, cost: float = 0.0):
        self.api_calls += 1
        self.total_tokens += tokens
        self.total_cost += cost

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def to_dict(self) -> dict:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "total_cost_usd": f"${self.total_cost:.4f}",
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "test_count": len(self.latencies),
        }


async def run_infrastructure_tests(metrics: PerformanceMetrics) -> dict:
    """Run infrastructure health tests."""
    print_section("INFRASTRUCTURE HEALTH", "🔧")

    from tests.integration.test_infrastructure import InfrastructureTester

    tester = InfrastructureTester()
    report = await tester.run_all()

    # Print results
    for test in report.tests:
        print_result(
            f"{test.component}/{test.test_name}",
            test.passed,
            test.latency_ms,
            test.error[:50] if test.error else "",
        )
        metrics.record_latency(test.latency_ms)

    return {
        "category": "infrastructure",
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "tests": [t.to_dict() for t in report.tests],
    }


async def run_orchestrator_tests(metrics: PerformanceMetrics) -> dict:
    """Run orchestrator E2E tests."""
    print_section("ORCHESTRATOR E2E", "🎭")

    from tests.integration.test_orchestrator_e2e import OrchestratorE2ETester

    tester = OrchestratorE2ETester()
    report = await tester.run_all()

    # Print results
    for test in report.tests:
        print_result(
            f"{test.phase}/{test.test_name}",
            test.passed,
            test.latency_ms,
            f"tokens: {test.tokens_used}" if test.tokens_used else "",
        )
        metrics.record_latency(test.latency_ms)
        metrics.record_api_call(test.tokens_used, test.cost_usd)

    return {
        "category": "orchestrator",
        "total": len(report.tests),
        "passed": report.passed_count,
        "failed": len(report.tests) - report.passed_count,
        "total_tokens": report.total_tokens,
        "total_cost": report.total_cost,
        "tests": [t.to_dict() for t in report.tests],
    }


async def run_browser_tests(metrics: PerformanceMetrics) -> dict:
    """Run browser pool integration tests."""
    print_section("BROWSER POOL", "🌐")

    pool_url = os.environ.get("BROWSER_POOL_URL")
    if not pool_url:
        print("  ⚠️  BROWSER_POOL_URL not configured - skipping")
        return {
            "category": "browser_pool",
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": True,
            "tests": [],
        }

    from tests.integration.test_browser_pool_real import BrowserPoolTester

    tester = BrowserPoolTester()
    report = await tester.run_all()

    # Print results
    for test in report.tests:
        mode_str = f"[{test.execution_mode}]" if test.execution_mode else ""
        print_result(
            f"{test.action_type}/{test.test_name}",
            test.passed,
            test.latency_ms,
            mode_str,
        )
        metrics.record_latency(test.latency_ms)

    return {
        "category": "browser_pool",
        "total": len(report.tests),
        "passed": report.passed_count,
        "failed": len(report.tests) - report.passed_count,
        "execution_modes": report.execution_mode_stats(),
        "tests": [t.to_dict() for t in report.tests],
    }


async def run_all_tests(args) -> dict:
    """Run all integration tests."""
    metrics = PerformanceMetrics()
    results = []
    total_tests = 0
    total_passed = 0
    total_failed = 0

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set. Some tests will be skipped.")
        print()

    try:
        # Infrastructure tests
        if args.all or args.infra:
            result = await run_infrastructure_tests(metrics)
            results.append(result)
            total_tests += result["total"]
            total_passed += result["passed"]
            total_failed += result["failed"]

        # Orchestrator tests
        if args.all or args.orchestrator:
            result = await run_orchestrator_tests(metrics)
            results.append(result)
            total_tests += result["total"]
            total_passed += result["passed"]
            total_failed += result["failed"]

        # Browser pool tests
        if args.all or args.browser:
            result = await run_browser_tests(metrics)
            results.append(result)
            if not result.get("skipped"):
                total_tests += result["total"]
                total_passed += result["passed"]
                total_failed += result["failed"]

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests cancelled by user")

    # Build final report
    report = {
        "summary": {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": f"{total_passed/max(total_tests,1):.1%}",
            "total_latency_ms": sum(metrics.latencies),
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "total_cost": f"${metrics.total_cost:.4f}",
            "total_tokens": metrics.total_tokens,
        },
        "performance": metrics.to_dict(),
        "by_category": {r["category"]: {"passed": r["passed"], "total": r["total"]} for r in results},
        "results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run real integration tests against infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all integration tests",
    )

    parser.add_argument(
        "--infra",
        action="store_true",
        help="Run infrastructure health tests",
    )

    parser.add_argument(
        "--orchestrator",
        action="store_true",
        help="Run orchestrator E2E tests",
    )

    parser.add_argument(
        "--browser",
        action="store_true",
        help="Run browser pool tests",
    )

    parser.add_argument(
        "--report",
        type=str,
        help="Save JSON report to file",
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Show detailed performance monitoring",
    )

    args = parser.parse_args()

    # Default to --all if no category specified
    if not (args.all or args.infra or args.orchestrator or args.browser):
        args.all = True

    print_banner()

    # Show configuration
    print("   Configuration:")
    print(f"   ├── ANTHROPIC_API_KEY: {'✅ Set' if os.environ.get('ANTHROPIC_API_KEY') else '❌ Not set'}")
    print(f"   ├── DATABASE_URL: {'✅ Set' if os.environ.get('DATABASE_URL') else '❌ Not set'}")
    print(f"   ├── BROWSER_POOL_URL: {'✅ Set' if os.environ.get('BROWSER_POOL_URL') else '❌ Not set'}")
    print(f"   └── BROWSER_POOL_JWT_SECRET: {'✅ Set' if os.environ.get('BROWSER_POOL_JWT_SECRET') else '❌ Not set'}")
    print()

    # Run tests
    try:
        report = asyncio.run(run_all_tests(args))
        print_summary(report)

        # Performance monitoring
        if args.monitor:
            print_section("PERFORMANCE METRICS", "📈")
            perf = report.get("performance", {})
            print(f"   Elapsed Time:    {perf.get('elapsed_seconds', 0):.1f}s")
            print(f"   API Calls:       {perf.get('api_calls', 0)}")
            print(f"   Total Tokens:    {perf.get('total_tokens', 0):,}")
            print(f"   Total Cost:      {perf.get('total_cost_usd', '$0.0000')}")
            print(f"   Avg Latency:     {perf.get('avg_latency_ms', 0):.0f}ms")
            print(f"   P95 Latency:     {perf.get('p95_latency_ms', 0):.0f}ms")
            print()

        # Save report
        if args.report:
            report_path = Path(args.report)
            report_path.write_text(json.dumps(report, indent=2, default=str))
            print(f"📄 Report saved to: {args.report}")
            print()

        # Exit code based on results
        summary = report.get("summary", {})
        passed = summary.get("passed", 0)
        total = summary.get("total_tests", 0)

        if total == 0:
            print("⚠️  No tests were run")
            sys.exit(0)
        elif passed == total:
            print("✅ All integration tests PASSED")
            sys.exit(0)
        elif passed / total >= 0.7:
            print("⚠️  Most tests passed, some failures")
            sys.exit(0)
        else:
            print("❌ Integration tests FAILED")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
