"""Benchmark Runner — Executes real agents against test datasets.

Runs agents against curated scenarios with known ground truth,
records results via tracked_execute() into agent_evaluations,
then aggregates into agent_benchmarks.

Each scenario runs K times (default 8) for Pass@k calculation.

Usage:
    # Run all benchmarks
    python -m src.services.benchmark_runner

    # Run specific suite
    python -m src.services.benchmark_runner --suite self_healing

    # Run with fewer repetitions (faster, less accurate)
    python -m src.services.benchmark_runner --runs 3
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DATASETS_DIR = Path(__file__).parent.parent.parent / "tests" / "benchmarks" / "datasets"

# Benchmark org/project IDs (NULL org = global benchmarks)
BENCHMARK_ORG_ID = None
BENCHMARK_PROJECT_ID = None


def _should_use_multi_model() -> bool:
    """Check if OpenRouter is available for multi-model routing.

    When OpenRouter API key is not set, agents using _call_ai() will fail
    because the model router only has OpenRouter-backed models.
    In that case, disable multi-model to fall back to direct Anthropic API.
    """
    from src.config import get_settings

    return bool(get_settings().openrouter_api_key)


@dataclass
class ScenarioResult:
    """Result of a single scenario execution."""

    scenario_id: str
    agent_type: str
    success: bool
    score: float  # 0.0 - 1.0
    latency_ms: int
    cost_usd: float
    details: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class SuiteResult:
    """Aggregated results for a benchmark suite."""

    suite_name: str
    agent_type: str
    total_scenarios: int
    total_runs: int
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def pass_sequence(self) -> list[bool]:
        return [r.success for r in self.results]

    @property
    def avg_score(self) -> float:
        scores = [r.score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.latency_ms for r in self.results]
        return sum(lats) / len(lats) if lats else 0.0

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results)


# =============================================================================
# Scoring Functions — Evaluate agent output against ground truth
# =============================================================================


def score_self_healing(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a self-healing result against ground truth.

    Checks if the agent's suggested selector matches or contains
    the expected patterns from ground truth.
    """
    if not result_data or not hasattr(result_data, "data"):
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        suggestions = data.get("suggestions", [])
        healed_selector = data.get("healed_selector", "")
        strategy = data.get("strategy_used", "")
    else:
        suggestions = []
        healed_selector = str(data)
        strategy = ""

    # Combine all text to search
    all_text = f"{healed_selector} {strategy} {json.dumps(suggestions)}".lower()

    must_contain = ground_truth.get("must_contain_in_suggestion", [])
    matches = sum(1 for term in must_contain if term.lower() in all_text)
    match_ratio = matches / len(must_contain) if must_contain else 0.0

    # Success if at least half the expected terms are found
    success = match_ratio >= 0.5
    score = match_ratio

    return success, score, {
        "matched_terms": matches,
        "total_expected": len(must_contain),
        "healed_selector": healed_selector[:200],
        "strategy": strategy,
    }


def score_code_analysis(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a code analysis result against ground truth.

    The CodeAnalyzerAgent returns a CodeAnalysisResult dataclass with:
        summary: str
        testable_surfaces: list[TestableSurface]
        framework_detected: str | None
        recommendations: list[str]
    """
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data

    # Handle CodeAnalysisResult dataclass
    if isinstance(data, dict):
        surfaces = data.get("testable_surfaces", [])
        all_text = json.dumps(data).lower()
    elif hasattr(data, "testable_surfaces"):
        # Dataclass — extract surfaces and convert to searchable text
        surfaces = data.testable_surfaces or []
        # Build text from all fields for keyword matching
        parts = [getattr(data, "summary", "") or ""]
        for s in surfaces:
            parts.append(getattr(s, "name", "") or "")
            parts.append(getattr(s, "description", "") or "")
            parts.append(getattr(s, "path", "") or "")
            parts.extend(getattr(s, "test_scenarios", []) or [])
        parts.extend(getattr(data, "recommendations", []) or [])
        all_text = " ".join(str(p) for p in parts).lower()
    else:
        surfaces = []
        all_text = str(data).lower()

    # Check minimum surfaces found
    min_surfaces = ground_truth.get("min_surfaces", 1)
    surfaces_found = len(surfaces) if isinstance(surfaces, list) else 0

    # Check required elements mentioned
    must_find = ground_truth.get("must_find_elements", [])
    found_elements = sum(1 for elem in must_find if elem.lower() in all_text)
    element_ratio = found_elements / len(must_find) if must_find else 1.0

    # Check API endpoints found
    must_find_apis = ground_truth.get("must_find_api_endpoints", [])
    found_apis = sum(1 for api in must_find_apis if api.lower() in all_text)
    api_ratio = found_apis / len(must_find_apis) if must_find_apis else 1.0

    score = (element_ratio * 0.5 + api_ratio * 0.3 + min(surfaces_found / max(min_surfaces, 1), 1.0) * 0.2)
    success = score >= 0.5 and surfaces_found >= 1

    return success, score, {
        "surfaces_found": surfaces_found,
        "min_expected": min_surfaces,
        "elements_found": found_elements,
        "elements_expected": len(must_find),
        "apis_found": found_apis,
        "apis_expected": len(must_find_apis),
    }


def score_hallucination_detection(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score hallucination detection result.

    The HallucinationDetectorAgent returns:
        severity: none/low/medium/high/critical
        hallucination_rate: 0.0-1.0
        is_reliable: True if severity in (none, low)
        overall_score: 0.0-1.0 (higher = more reliable)
    """
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data

    if isinstance(data, dict):
        # Map agent output to hallucination detection
        severity = data.get("severity", "none")
        hallucination_rate = data.get("hallucination_rate", 0.0)
        is_reliable = data.get("is_reliable", True)
        overall_score = data.get("overall_score", 1.0)

        # Agent considers it a hallucination if not reliable
        # (severity >= medium, or hallucination_rate > 0.3)
        detected_hallucination = not is_reliable or severity in ("medium", "high", "critical")
        confidence = hallucination_rate if detected_hallucination else (1.0 - hallucination_rate)
    else:
        detected_hallucination = False
        confidence = 0.5

    expected = ground_truth.get("is_hallucination", False)

    # Correct detection
    correct = detected_hallucination == expected

    # Score based on confidence in the correct direction
    if correct:
        score = max(confidence, 0.5)  # At least 0.5 for correct answer
    else:
        score = 0.0

    return correct, score, {
        "detected": detected_hallucination,
        "expected": expected,
        "confidence": confidence,
        "severity": data.get("severity") if isinstance(data, dict) else None,
        "hallucination_rate": data.get("hallucination_rate") if isinstance(data, dict) else None,
    }


def score_sre_analysis(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score SRE incident analysis result.

    The SREAgent returns a dict with:
        incident: CorrelatedIncident (dataclass — NOT JSON-serializable)
        analysis: dict with root_cause, confidence, affected_services, etc.
        suggested_actions: list of action dicts

    We extract text from all these fields for keyword matching.
    """
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data

    # Build searchable text from all fields, handling CorrelatedIncident dataclass
    parts = []
    severity_correct = False

    if isinstance(data, dict):
        # Extract from the 'analysis' sub-dict (plain dict, safe to serialize)
        analysis = data.get("analysis", {})
        if isinstance(analysis, dict):
            parts.append(str(analysis.get("root_cause", "")))
            parts.extend(str(s) for s in analysis.get("affected_services", []))
            parts.append(str(analysis.get("blast_radius", "")))

        # Extract from suggested_actions (list of dicts)
        actions = data.get("suggested_actions", [])
        for action in actions:
            if isinstance(action, dict):
                parts.append(str(action))
            else:
                parts.append(str(action))

        # Handle CorrelatedIncident dataclass
        incident = data.get("incident", data)
        if hasattr(incident, "severity"):
            # Dataclass — extract fields directly
            detected_sev = str(getattr(incident, "severity", "")).lower()
            expected_sev = ground_truth.get("severity", "").lower()
            severity_correct = expected_sev in detected_sev or detected_sev in expected_sev
            parts.append(str(getattr(incident, "root_cause_hypothesis", "") or ""))
            parts.append(str(getattr(incident, "title", "") or ""))
            for svc in getattr(incident, "affected_services", []):
                parts.append(str(svc))
            for action in getattr(incident, "suggested_actions", []):
                parts.append(str(action))
        elif isinstance(incident, dict):
            detected_sev = str(incident.get("severity", "")).lower()
            expected_sev = ground_truth.get("severity", "").lower()
            severity_correct = expected_sev in detected_sev or detected_sev in expected_sev
    else:
        parts.append(str(data))

    all_text = " ".join(parts).lower()

    # Check root cause keywords
    root_cause_terms = ground_truth.get("root_cause_must_mention", [])
    rc_found = sum(1 for term in root_cause_terms if term.lower() in all_text)
    rc_ratio = rc_found / len(root_cause_terms) if root_cause_terms else 1.0

    # Check suggested actions
    action_terms = ground_truth.get("suggested_actions_must_include", [])
    actions_found = sum(1 for term in action_terms if term.lower() in all_text)
    action_ratio = actions_found / len(action_terms) if action_terms else 1.0

    score = rc_ratio * 0.4 + action_ratio * 0.4 + (0.2 if severity_correct else 0.0)
    success = score >= 0.4 and rc_found >= 1

    return success, score, {
        "root_cause_found": rc_found,
        "root_cause_expected": len(root_cause_terms),
        "actions_found": actions_found,
        "actions_expected": len(action_terms),
        "severity_correct": severity_correct,
    }


# =============================================================================
# Benchmark Suites — Each suite runs one agent against its dataset
# =============================================================================


async def run_self_healing_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run SelfHealerAgent against healing scenarios."""
    from src.agents.self_healer import SelfHealerAgent

    dataset = _load_dataset("self_healing.json")
    suite = SuiteResult(
        suite_name="self_healing",
        agent_type="SelfHealerAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        fd = scenario["failure_data"]
        # Build test_spec and failure_details from the dataset format
        # to match SelfHealerAgent.execute(test_spec, failure_details, ...)
        test_spec = {
            "id": fd.get("test_id", "benchmark_test"),
            "selector": fd.get("failed_selector", ""),
            "action": fd.get("test_code", ""),
            "url": fd.get("page_url", ""),
        }
        failure_details = {
            "type": "selector_changed" if fd.get("error_type") == "selector_not_found" else fd.get("error_type", "unknown"),
            "selector": fd.get("failed_selector", ""),
            "message": fd.get("error_message", ""),
            "error": fd.get("error_message", ""),
            "page_html": fd.get("page_html_snippet", ""),
            "git_diff": fd.get("git_diff", ""),
            "target": fd.get("failed_selector", ""),
        }

        for run_idx in range(runs_per_scenario):
            agent = SelfHealerAgent(
                org_id=BENCHMARK_ORG_ID,
                project_id=BENCHMARK_PROJECT_ID,
                enable_code_aware=True,
                enable_memory_store=False,  # Pure agent capability test
                enable_hybrid_retrieval=False,
            )

            start = time.time()
            try:
                result = await agent.tracked_execute(
                    test_spec=test_spec,
                    failure_details=failure_details,
                    trigger_source="benchmark",
                    organization_id=BENCHMARK_ORG_ID,
                    project_id=BENCHMARK_PROJECT_ID,
                    eval_metadata={
                        "benchmark_suite": "self_healing",
                        "scenario_id": scenario["id"],
                        "run_index": run_idx,
                    },
                )

                success, score, details = score_self_healing(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                result = None
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="SelfHealerAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="self_healing",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


async def run_code_analysis_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run CodeAnalyzerAgent against code analysis scenarios."""
    from src.agents.code_analyzer import CodeAnalyzerAgent

    dataset = _load_dataset("code_analysis.json")
    suite = SuiteResult(
        suite_name="code_analysis",
        agent_type="CodeAnalyzerAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        for run_idx in range(runs_per_scenario):
            agent = CodeAnalyzerAgent(
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.tracked_execute(
                    codebase_path="/tmp/benchmark",
                    app_url=scenario.get("app_url", "https://example.com"),
                    file_contents=scenario["file_contents"],
                    trigger_source="benchmark",
                    organization_id=BENCHMARK_ORG_ID,
                    project_id=BENCHMARK_PROJECT_ID,
                    eval_metadata={
                        "benchmark_suite": "code_analysis",
                        "scenario_id": scenario["id"],
                        "run_index": run_idx,
                    },
                )

                success, score, details = score_code_analysis(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="CodeAnalyzerAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="code_analysis",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


async def run_hallucination_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run HallucinationDetectorAgent against hallucination scenarios."""
    from src.agents.hallucination_detector import HallucinationDetectorAgent

    dataset = _load_dataset("hallucination_detection.json")
    suite = SuiteResult(
        suite_name="hallucination_detection",
        agent_type="HallucinationDetectorAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        for run_idx in range(runs_per_scenario):
            agent = HallucinationDetectorAgent(
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.tracked_execute(
                    response=scenario["response"],
                    query=scenario["query"],
                    context=scenario["context"],
                    trigger_source="benchmark",
                    organization_id=BENCHMARK_ORG_ID,
                    project_id=BENCHMARK_PROJECT_ID,
                    eval_metadata={
                        "benchmark_suite": "hallucination_detection",
                        "scenario_id": scenario["id"],
                        "run_index": run_idx,
                    },
                )

                success, score, details = score_hallucination_detection(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="HallucinationDetectorAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="hallucination_detection",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


async def run_sre_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run SREAgent against incident analysis scenarios."""
    from src.agents.sre_agent import SREAgent

    dataset = _load_dataset("sre_incidents.json")
    suite = SuiteResult(
        suite_name="sre_incidents",
        agent_type="SREAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        for run_idx in range(runs_per_scenario):
            agent = SREAgent(
                org_id=BENCHMARK_ORG_ID,
                project_id=BENCHMARK_PROJECT_ID,
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.tracked_execute(
                    task_type="analyze",
                    alert_data=scenario["alert_data"],
                    trigger_source="benchmark",
                    organization_id=BENCHMARK_ORG_ID,
                    project_id=BENCHMARK_PROJECT_ID,
                    eval_metadata={
                        "benchmark_suite": "sre_incidents",
                        "scenario_id": scenario["id"],
                        "run_index": run_idx,
                    },
                )

                success, score, details = score_sre_analysis(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="SREAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="sre_incidents",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


# =============================================================================
# Suite Registry & Runner
# =============================================================================

SUITES = {
    "self_healing": run_self_healing_suite,
    "code_analysis": run_code_analysis_suite,
    "hallucination_detection": run_hallucination_suite,
    "sre_incidents": run_sre_suite,
}


def _load_dataset(filename: str) -> list[dict]:
    """Load a JSON dataset file."""
    path = DATASETS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


async def run_all_benchmarks(
    runs_per_scenario: int = 8,
    suites: list[str] | None = None,
    aggregate: bool = True,
) -> dict[str, SuiteResult]:
    """Run all (or selected) benchmark suites.

    Args:
        runs_per_scenario: How many times to run each scenario (for Pass@k).
        suites: List of suite names to run, or None for all.
        aggregate: Whether to run benchmark aggregation after.

    Returns:
        Dict of suite name → SuiteResult.
    """
    suite_names = suites or list(SUITES.keys())
    results: dict[str, SuiteResult] = {}

    total_start = time.time()

    for name in suite_names:
        if name not in SUITES:
            logger.warning("Unknown benchmark suite", suite=name)
            continue

        logger.info("benchmark_suite_starting", suite=name, runs_per_scenario=runs_per_scenario)
        suite_start = time.time()

        try:
            result = await SUITES[name](runs_per_scenario=runs_per_scenario)
            results[name] = result

            suite_time = int(time.time() - suite_start)
            pass_count = sum(1 for r in result.results if r.success)
            total_count = len(result.results)

            logger.info(
                "benchmark_suite_complete",
                suite=name,
                agent=result.agent_type,
                passed=pass_count,
                total=total_count,
                pass_rate=round(pass_count / total_count, 3) if total_count else 0,
                avg_score=round(result.avg_score, 3),
                avg_latency_ms=round(result.avg_latency_ms),
                total_cost=round(result.total_cost, 4),
                duration_seconds=suite_time,
            )
        except Exception as e:
            logger.error("benchmark_suite_failed", suite=name, error=str(e))

    total_time = int(time.time() - total_start)
    total_cost = sum(r.total_cost for r in results.values())

    logger.info(
        "benchmark_run_complete",
        suites_run=len(results),
        total_duration_seconds=total_time,
        total_cost_usd=round(total_cost, 4),
    )

    # Aggregate into agent_benchmarks
    if aggregate and results:
        try:
            from src.services.benchmark_aggregator import aggregate_benchmarks
            benchmarks = await aggregate_benchmarks(
                period_days=1,
                organization_id=BENCHMARK_ORG_ID,
            )
            logger.info(
                "benchmark_aggregation_complete",
                benchmarks_created=len(benchmarks),
            )
        except Exception as e:
            logger.error("benchmark_aggregation_failed", error=str(e))

    return results


def print_report(results: dict[str, SuiteResult]) -> None:
    """Print a human-readable benchmark report."""
    from src.services.benchmark_aggregator import compute_pass_at_k

    print("\n" + "=" * 70)
    print("  AGENT RELIABILITY BENCHMARK REPORT")
    print("=" * 70)

    for name, suite in results.items():
        seq = suite.pass_sequence
        p1 = compute_pass_at_k(seq, 1)
        p3 = compute_pass_at_k(seq, 3)
        p8 = compute_pass_at_k(seq, 8)

        print(f"\n--- {suite.agent_type} ({name}) ---")
        print(f"  Scenarios: {suite.total_scenarios}")
        print(f"  Total runs: {len(suite.results)}")
        print(f"  Pass@1:  {p1 * 100:.1f}%" if p1 is not None else "  Pass@1:  N/A")
        print(f"  Pass@3:  {p3 * 100:.1f}%" if p3 is not None else "  Pass@3:  N/A")
        print(f"  Pass@8:  {p8 * 100:.1f}%" if p8 is not None else "  Pass@8:  N/A")
        print(f"  Avg Score: {suite.avg_score:.3f}")
        print(f"  Avg Latency: {suite.avg_latency_ms:.0f}ms")
        print(f"  Total Cost: ${suite.total_cost:.4f}")

        # Per-scenario breakdown
        by_scenario: dict[str, list[ScenarioResult]] = {}
        for r in suite.results:
            base_id = r.scenario_id.rsplit("_run", 1)[0]
            by_scenario.setdefault(base_id, []).append(r)

        print(f"\n  Per-scenario breakdown:")
        for scenario_id, runs in by_scenario.items():
            passes = sum(1 for r in runs if r.success)
            avg_s = sum(r.score for r in runs) / len(runs)
            print(f"    {scenario_id}: {passes}/{len(runs)} pass, avg score {avg_s:.2f}")

    total_cost = sum(r.total_cost for r in results.values())
    print(f"\n{'=' * 70}")
    print(f"  Total cost: ${total_cost:.4f}")
    print("=" * 70 + "\n")


# =============================================================================
# Weekly Background Loop (started from server.py)
# =============================================================================


async def start_weekly_benchmark_loop() -> None:
    """Run benchmarks weekly (Sundays at 2 AM UTC).

    Started as a background task from server.py startup.
    Sleeps until next Sunday 2 AM, then runs all suites.
    """
    from datetime import UTC, datetime, timedelta

    from src.config import get_settings

    settings = get_settings()
    runs = settings.benchmark_runs_per_scenario

    while True:
        try:
            # Calculate sleep until next Sunday 2 AM UTC
            now = datetime.now(UTC)
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.hour >= 2:
                days_until_sunday = 7
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(
                days=days_until_sunday
            )
            sleep_seconds = (next_run - now).total_seconds()

            logger.info(
                "benchmark_next_run_scheduled",
                next_run=next_run.isoformat(),
                sleep_hours=round(sleep_seconds / 3600, 1),
            )

            await asyncio.sleep(sleep_seconds)

            logger.info("benchmark_weekly_run_starting", runs_per_scenario=runs)
            results = await run_all_benchmarks(
                runs_per_scenario=runs,
                aggregate=True,
            )
            print_report(results)

        except asyncio.CancelledError:
            logger.info("Weekly benchmark loop cancelled")
            break
        except Exception as e:
            logger.error("benchmark_weekly_run_failed", error=str(e))
            # Sleep 1 hour before retrying on failure
            await asyncio.sleep(3600)


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run agent reliability benchmarks")
    parser.add_argument(
        "--suite", "-s",
        nargs="*",
        choices=list(SUITES.keys()),
        help="Specific suites to run (default: all)",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=8,
        help="Runs per scenario for Pass@k (default: 8)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip benchmark aggregation after run",
    )

    args = parser.parse_args()

    results = await run_all_benchmarks(
        runs_per_scenario=args.runs,
        suites=args.suite,
        aggregate=not args.no_aggregate,
    )

    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
