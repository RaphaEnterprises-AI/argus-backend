#!/usr/bin/env python3
"""
Agent Evaluation CLI

Run comprehensive evaluation of the multi-agent E2E testing framework.

Usage:
    # Quick smoke test (mocked, fast)
    python scripts/run_agent_evaluation.py --quick

    # Full evaluation with real API calls
    python scripts/run_agent_evaluation.py --full

    # Specific category
    python scripts/run_agent_evaluation.py --category code_analysis

    # Generate detailed report
    python scripts/run_agent_evaluation.py --full --report evaluation_report.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_banner():
    """Print evaluation banner."""
    print()
    print("=" * 70)
    print("   🤖 ARGUS E2E TESTING AGENT - CAPABILITY EVALUATION")
    print("=" * 70)
    print()
    print("   Evaluating AI Agent Intelligence & Performance")
    print("   Based on industry benchmarks: WebArena, AgentBench, BFCL")
    print()
    print("-" * 70)


def print_category_header(category: str):
    """Print category header."""
    icons = {
        "code_analysis": "🔍",
        "test_planning": "📋",
        "self_healing": "🔧",
        "nlp_understanding": "💬",
        "visual_ai": "👁️",
        "ui_execution": "🖱️",
        "orchestration": "🎭",
    }
    icon = icons.get(category, "📊")
    print(f"\n{icon} {category.upper().replace('_', ' ')}")
    print("-" * 40)


def print_score(name: str, score: float, passed: bool):
    """Print individual test score."""
    status = "✅" if passed else "❌"
    bar_length = 20
    filled = int(score * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"  {status} {name:<40} [{bar}] {score:.2f}")


def print_summary(report: dict):
    """Print evaluation summary."""
    summary = report["summary"]
    quality = report["quality_metrics"]
    perf = report["performance"]

    print("\n" + "=" * 70)
    print("   📊 EVALUATION SUMMARY")
    print("=" * 70)

    print(f"""
   Tests Run:      {summary['total_tests']}
   Passed:         {summary['passed']} ({summary['pass_rate']})
   Failed:         {summary['failed']}

   Overall Grade:  {summary['grade']} ({summary['overall_score']})

   Quality Metrics:
   ├── Accuracy:          {quality['accuracy']}
   └── Reasoning Quality: {quality['reasoning_quality']}

   Performance:
   ├── Total Latency:     {perf['total_latency_ms']} ms
   ├── Avg Latency:       {perf['avg_latency_ms']} ms
   └── Total Cost:        {perf['total_cost_usd']}
""")

    # Print by category
    if report.get("by_category"):
        print("   Results by Category:")
        for cat, data in report["by_category"].items():
            passed = data["passed"]
            total = data["count"]
            avg_score = data["avg_score"]
            print(f"   ├── {cat:<20} {passed}/{total} passed (score: {avg_score:.2f})")

    print("=" * 70)


async def run_evaluation(args):
    """Run the evaluation."""
    from tests.evaluation.runner import EvaluationConfig, EvaluationRunner

    # Configure evaluation
    config = EvaluationConfig(
        categories=args.categories if args.categories else None,
        max_tests_per_category=2 if args.quick else None,
        use_real_api=not args.quick and args.real_api,
        verbose=True,
    )

    print(f"   Mode: {'Quick (mocked)' if args.quick else 'Full (real API)' if args.real_api else 'Standard'}")
    print(f"   Categories: {args.categories or 'All'}")
    print()

    # Run evaluation
    runner = EvaluationRunner(config)
    metrics = await runner.run_all()

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run agent capability evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with mocked APIs",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation (all tests)",
    )

    parser.add_argument(
        "--real-api",
        action="store_true",
        default=True,
        help="Use real API calls (default: True)",
    )

    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        choices=[
            "code_analysis",
            "test_planning",
            "self_healing",
            "nlp_understanding",
            "visual_ai",
            "orchestration",
        ],
        help="Run specific category (can be repeated)",
    )

    parser.add_argument(
        "--report",
        type=str,
        help="Save detailed report to JSON file",
    )

    args = parser.parse_args()

    # Check for API key if using real API
    if args.real_api and not args.quick:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("⚠️  Warning: ANTHROPIC_API_KEY not set. Using mocked responses.")
            args.quick = True

    print_banner()

    # Run evaluation
    try:
        metrics = asyncio.run(run_evaluation(args))
        report = metrics.to_report()

        # Print summary
        print_summary(report)

        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.write_text(json.dumps(report, indent=2, default=str))
            print(f"\n📄 Detailed report saved to: {args.report}")

        # Exit code based on pass rate
        pass_rate = metrics.pass_rate
        if pass_rate >= 0.8:
            print("\n✅ Evaluation PASSED - Agent capabilities verified")
            sys.exit(0)
        elif pass_rate >= 0.6:
            print("\n⚠️  Evaluation WARNING - Some capabilities need improvement")
            sys.exit(0)
        else:
            print("\n❌ Evaluation FAILED - Critical capability issues detected")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
