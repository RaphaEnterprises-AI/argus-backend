"""Main entry point for E2E Testing Agent."""

import argparse
import asyncio
from pathlib import Path

import structlog

from .config import get_settings
from .orchestrator.graph import TestingOrchestrator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def run_tests(
    codebase_path: str,
    app_url: str,
    output_dir: str,
    pr_number: int | None = None,
    changed_files: list[str] | None = None,
) -> dict:
    """Run the full test suite."""
    settings = get_settings()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    orchestrator = TestingOrchestrator(
        codebase_path=codebase_path,
        app_url=app_url,
        settings=settings,
        pr_number=pr_number,
        changed_files=changed_files,
    )

    # Run tests
    result = await orchestrator.run()

    # Save results
    import json
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Results saved", path=str(results_file))

    # Print summary
    summary = orchestrator.get_run_summary(result)
    print("\n" + "=" * 50)
    print("TEST RUN SUMMARY")
    print("=" * 50)
    print(f"Run ID: {summary['run_id']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Skipped: {summary['skipped']} ⏭️")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Total Cost: ${summary['total_cost']:.4f}")
    print("=" * 50 + "\n")

    return result


def cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Autonomous E2E Testing Agent powered by Claude"
    )
    parser.add_argument(
        "--codebase", "-c",
        required=True,
        help="Path to the codebase to test"
    )
    parser.add_argument(
        "--app-url", "-u",
        required=True,
        help="URL of the running application"
    )
    parser.add_argument(
        "--output", "-o",
        default="./test-results",
        help="Output directory for results (default: ./test-results)"
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="PR number (for CI/CD integration)"
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        help="List of changed files (for targeted testing)"
    )

    args = parser.parse_args()

    # Run async
    result = asyncio.run(run_tests(
        codebase_path=args.codebase,
        app_url=args.app_url,
        output_dir=args.output,
        pr_number=args.pr,
        changed_files=args.changed_files,
    ))

    # Exit with error code if tests failed
    if result.get("failed_count", 0) > 0:
        exit(1)


if __name__ == "__main__":
    cli()
