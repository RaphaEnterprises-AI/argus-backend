#!/usr/bin/env python3
"""
World-Class Agent Evaluation CLI.

Run comprehensive agent evaluations aligned with industry standards:
- SWE-bench: Software engineering task completion
- WebArena: Web navigation and automation
- BFCL: Function calling accuracy
- TAU-bench: Multi-turn conversation handling
- Bloom: Behavioral evaluation

Usage:
    # Quick evaluation (mocked, for testing the framework)
    python scripts/run_world_class_eval.py --quick

    # Full evaluation with real API
    python scripts/run_world_class_eval.py --real-api

    # Specific domains
    python scripts/run_world_class_eval.py --domain code_understanding --domain self_healing

    # With detailed report
    python scripts/run_world_class_eval.py --real-api --report results.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env


def print_banner():
    """Print evaluation banner."""
    print()
    print("=" * 70)
    print("   🌍 WORLD-CLASS AGENT EVALUATION FRAMEWORK")
    print("=" * 70)
    print()
    print("   Evaluating Against Industry Benchmarks:")
    print("   ├── SWE-bench     (Code Understanding)")
    print("   ├── WebArena      (Web Navigation)")
    print("   ├── BFCL          (Function Calling)")
    print("   ├── TAU-bench     (Multi-turn Reasoning)")
    print("   └── Bloom         (Behavioral Analysis)")
    print()
    print("-" * 70)


def print_config(args):
    """Print current configuration."""
    print("\n   Configuration:")
    print(f"   ├── Mode:          {'Real API' if args.real_api else 'Mocked'}")
    print(f"   ├── Domains:       {args.domain if args.domain else 'All'}")
    print(f"   ├── Difficulties:  {args.difficulty if args.difficulty else 'All'}")
    print(f"   ├── Attempts/Task: {args.attempts}")
    print(f"   └── Report:        {args.report if args.report else 'Console only'}")

    # Environment check
    print("\n   Environment:")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    browser_pool = os.environ.get("BROWSER_POOL_URL")
    print(f"   ├── ANTHROPIC_API_KEY:  {'✅ Set' if api_key else '❌ Not set'}")
    print(f"   └── BROWSER_POOL_URL:   {'✅ Set' if browser_pool else '❌ Not set'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run world-class agent evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with mocked responses
    python scripts/run_world_class_eval.py --quick

    # Full evaluation with real API
    python scripts/run_world_class_eval.py --real-api --report results.json

    # Test specific capabilities
    python scripts/run_world_class_eval.py --domain code_understanding --real-api

    # Test by difficulty
    python scripts/run_world_class_eval.py --difficulty easy --difficulty medium
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation with mocked responses",
    )

    parser.add_argument(
        "--real-api",
        action="store_true",
        help="Use real API calls (default: mocked)",
    )

    parser.add_argument(
        "--domain",
        action="append",
        choices=[
            "code_understanding",
            "web_navigation",
            "function_calling",
            "multi_turn_reasoning",
            "self_healing",
            "visual_understanding",
            "tool_use",
        ],
        help="Run specific domain(s)",
    )

    parser.add_argument(
        "--difficulty",
        action="append",
        choices=["easy", "medium", "hard", "expert"],
        help="Run specific difficulty level(s)",
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Number of attempts per task for pass@k (default: 5)",
    )

    parser.add_argument(
        "--report",
        type=str,
        help="Save detailed JSON report to file",
    )

    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only show benchmark comparisons without running tests",
    )

    args = parser.parse_args()

    # Quick mode implies mocked
    if args.quick:
        args.real_api = False
        args.attempts = 2  # Fewer attempts for quick mode

    print_banner()
    print_config(args)

    # Check API key if using real API
    if args.real_api and not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY required for --real-api mode")
        print("   Set it with: export ANTHROPIC_API_KEY=your-key")
        sys.exit(1)

    # Import evaluation components
    from tests.evaluation.world_class_metrics import EvalDomain, TaskDifficulty
    from tests.evaluation.world_class_runner import RunConfig, WorldClassRunner

    # Build config
    domains = None
    if args.domain:
        domain_map = {
            "code_understanding": EvalDomain.CODE_UNDERSTANDING,
            "web_navigation": EvalDomain.WEB_NAVIGATION,
            "function_calling": EvalDomain.FUNCTION_CALLING,
            "multi_turn_reasoning": EvalDomain.MULTI_TURN_REASONING,
            "self_healing": EvalDomain.SELF_HEALING,
            "visual_understanding": EvalDomain.VISUAL_UNDERSTANDING,
            "tool_use": EvalDomain.TOOL_USE,
        }
        domains = [domain_map[d] for d in args.domain]

    difficulties = None
    if args.difficulty:
        diff_map = {
            "easy": TaskDifficulty.EASY,
            "medium": TaskDifficulty.MEDIUM,
            "hard": TaskDifficulty.HARD,
            "expert": TaskDifficulty.EXPERT,
        }
        difficulties = [diff_map[d] for d in args.difficulty]

    config = RunConfig(
        use_real_api=args.real_api,
        domains=domains,
        difficulties=difficulties,
        attempts_per_task=args.attempts,
        verbose=True,
    )

    # Run evaluation
    try:
        runner = WorldClassRunner(config)
        report = asyncio.run(runner.run_full_evaluation())

        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_data = report.to_dict()
            report_data["run_config"] = {
                "real_api": args.real_api,
                "domains": args.domain,
                "difficulties": args.difficulty,
                "attempts": args.attempts,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            report_path.write_text(json.dumps(report_data, indent=2, default=str))
            print(f"\n📄 Detailed report saved to: {args.report}")

        # Exit with appropriate code
        grade = report.overall_grade()
        if grade in ["A+", "A", "A-", "B+", "B"]:
            print("\n✅ Evaluation completed - Agent meets world-class standards")
            sys.exit(0)
        elif grade in ["B-", "C+", "C"]:
            print("\n⚠️  Evaluation completed - Agent needs improvement")
            sys.exit(0)
        else:
            print("\n❌ Evaluation completed - Agent below acceptable threshold")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
