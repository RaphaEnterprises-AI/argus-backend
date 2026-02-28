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

DATASETS_DIR = Path(__file__).parent / "benchmark_datasets"


def _deep_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass/object trees to plain dicts for scoring.

    Prefers .to_dict() if available (preserves agent-specific serialization),
    otherwise falls back to __dict__ with recursive descent.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _deep_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_to_dict(i) for i in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: _deep_to_dict(v) for k, v in obj.__dict__.items()}
    # Enums, datetimes, etc. — just stringify
    return str(obj)

# Benchmark org/project IDs — sentinel UUID for global benchmarks
# (agent_evaluations.organization_id is UUID NOT NULL)
BENCHMARK_ORG_ID = "00000000-0000-0000-0000-000000000000"
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


def score_testgen(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a TestGen result against ground truth.

    Evaluates:
    - Number of requirements parsed vs expected
    - Number of test scenarios generated vs expected
    - Quality score of generated tests
    - Framework/language correctness
    """
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.data if hasattr(result_data, "data") else {}
    )

    requirements = data.get("requirements", [])
    scenarios = data.get("scenarios", [])
    generated_tests = data.get("generated_tests", [])
    quality_score = data.get("quality_score", 0.0)

    min_requirements = ground_truth.get("min_requirements", 1)
    min_scenarios = ground_truth.get("min_scenarios", 1)
    expected_framework = ground_truth.get("framework")
    min_quality = ground_truth.get("min_quality_score", 0.5)

    reqs_found = len(requirements) if isinstance(requirements, list) else 0
    scenarios_found = len(scenarios) if isinstance(scenarios, list) else 0
    tests_found = len(generated_tests) if isinstance(generated_tests, list) else 0

    # Component scores
    req_score = min(reqs_found / max(min_requirements, 1), 1.0) if min_requirements > 0 else 1.0
    scenario_score = min(scenarios_found / max(min_scenarios, 1), 1.0) if min_scenarios > 0 else 1.0
    quality_met = 1.0 if quality_score >= min_quality else quality_score / max(min_quality, 0.01)
    framework_ok = 1.0  # default
    if expected_framework:
        fw = data.get("config", {}).get("framework", "")
        framework_ok = 1.0 if fw == expected_framework else 0.5

    score = req_score * 0.25 + scenario_score * 0.3 + quality_met * 0.3 + framework_ok * 0.15
    success = reqs_found >= 1 and scenarios_found >= 1 and score >= 0.5

    return success, score, {
        "requirements_found": reqs_found,
        "scenarios_found": scenarios_found,
        "tests_generated": tests_found,
        "quality_score": quality_score,
        "framework_match": framework_ok == 1.0,
    }


def score_healing_cicd(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a TestHealer CICD result against ground truth.

    Evaluates:
    - Failures detected vs expected
    - Fix proposals generated
    - Health score computation
    """
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.data if hasattr(result_data, "data") else {}
    )

    failures_found = len(data.get("failures_analyzed", data.get("fix_proposals", [])))
    fix_proposals = data.get("fix_proposals", [])
    health_score = data.get("health_score", {})
    status = data.get("status", "")

    min_fixes = ground_truth.get("min_fix_proposals", 0)
    must_detect = ground_truth.get("must_detect_failure_types", [])

    # Check failure type detection
    all_text = json.dumps(data).lower()
    detected_types = sum(1 for ft in must_detect if ft.lower() in all_text)
    detection_ratio = detected_types / len(must_detect) if must_detect else 1.0

    fixes_ok = len(fix_proposals) >= min_fixes if min_fixes > 0 else 1.0
    has_health = 1.0 if health_score else 0.5

    score = detection_ratio * 0.4 + (1.0 if fixes_ok else 0.3) * 0.35 + has_health * 0.25
    success = status in ("completed", "done", "") and score >= 0.4

    return success, score, {
        "failures_found": failures_found,
        "fix_proposals_count": len(fix_proposals),
        "detection_ratio": detection_ratio,
        "health_score": health_score,
    }


def score_qa_engineer(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a QA Engineer result against ground truth.

    Evaluates:
    - Coverage gaps identified vs expected
    - Quality score computed
    - Recommendations generated
    - User flows discovered
    """
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.data if hasattr(result_data, "data") else {}
    )

    quality_score = data.get("quality_score", 0.0)
    coverage_gaps = data.get("coverage_gaps", [])
    recommendations = data.get("recommendations", [])
    user_flows = data.get("user_flows", [])
    components = data.get("components_analyzed", [])

    min_gaps = ground_truth.get("min_coverage_gaps", 0)
    min_recommendations = ground_truth.get("min_recommendations", 1)
    min_quality = ground_truth.get("min_quality_score", 20.0)

    gaps_found = len(coverage_gaps) if isinstance(coverage_gaps, list) else 0
    recs_found = len(recommendations) if isinstance(recommendations, list) else 0
    flows_found = len(user_flows) if isinstance(user_flows, list) else 0

    gap_score = min(gaps_found / max(min_gaps, 1), 1.0) if min_gaps > 0 else (1.0 if gaps_found > 0 else 0.5)
    rec_score = min(recs_found / max(min_recommendations, 1), 1.0)
    quality_met = 1.0 if quality_score >= min_quality else quality_score / max(min_quality, 0.01)
    flow_bonus = min(flows_found / 3, 1.0) * 0.1  # bonus for user flow discovery

    score = gap_score * 0.3 + rec_score * 0.25 + quality_met * 0.35 + flow_bonus
    score = min(score, 1.0)
    success = recs_found >= 1 and score >= 0.4

    return success, score, {
        "quality_score": quality_score,
        "gaps_found": gaps_found,
        "recommendations_found": recs_found,
        "user_flows_found": flows_found,
        "components_analyzed": len(components) if isinstance(components, list) else 0,
    }


def score_test_planner(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a TestPlannerAgent result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if hasattr(data, "tests"):
        tests = data.tests or []
    elif isinstance(data, dict):
        tests = data.get("tests", [])
    else:
        tests = []

    min_tests = ground_truth.get("min_tests", 1)
    must_cover = ground_truth.get("must_cover_scenarios", [])
    all_text = json.dumps([t if isinstance(t, dict) else str(t) for t in tests]).lower()

    tests_count = len(tests)
    count_score = min(tests_count / max(min_tests, 1), 1.0)
    covered = sum(1 for s in must_cover if s.lower() in all_text)
    cover_ratio = covered / len(must_cover) if must_cover else 1.0

    score = count_score * 0.4 + cover_ratio * 0.6
    success = tests_count >= min_tests and score >= 0.5

    return success, score, {
        "tests_generated": tests_count,
        "min_expected": min_tests,
        "scenarios_covered": covered,
        "scenarios_expected": len(must_cover),
    }


def score_api_tester(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score an APITesterAgent result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        status = data.get("status", "error")
        requests = data.get("requests", [])
        validations = data.get("schema_validations", [])
    elif hasattr(data, "status"):
        status = data.status
        requests = getattr(data, "requests", []) or []
        validations = getattr(data, "schema_validations", []) or []
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    req_success = sum(1 for r in requests if (r.get("success") if isinstance(r, dict) else getattr(r, "success", False)))
    val_pass = sum(1 for v in validations if (v.get("valid") if isinstance(v, dict) else getattr(v, "valid", False)))

    expect_passed = ground_truth.get("expect_status", "passed") == "passed"
    status_ok = (status == "passed") == expect_passed

    req_ratio = req_success / len(requests) if requests else 0.0
    val_ratio = val_pass / len(validations) if validations else 1.0

    score = (0.3 if status_ok else 0.0) + req_ratio * 0.4 + val_ratio * 0.3
    success = status_ok and score >= 0.5

    return success, score, {
        "status": status,
        "requests_total": len(requests),
        "requests_success": req_success,
        "validations_pass": val_pass,
    }


def score_db_tester(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a DBTesterAgent result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        status = data.get("status", "error")
        queries = data.get("queries", [])
        validations = data.get("validations", [])
    elif hasattr(data, "status"):
        status = data.status
        queries = getattr(data, "queries", []) or []
        validations = getattr(data, "validations", []) or []
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    q_success = sum(1 for q in queries if (q.get("success") if isinstance(q, dict) else getattr(q, "success", False)))
    v_pass = sum(1 for v in validations if (v.get("passed") if isinstance(v, dict) else getattr(v, "passed", False)))

    q_ratio = q_success / len(queries) if queries else 0.0
    v_ratio = v_pass / len(validations) if validations else 1.0

    score = q_ratio * 0.5 + v_ratio * 0.5
    success = status == "passed" and score >= 0.5

    return success, score, {
        "status": status,
        "queries_total": len(queries),
        "queries_success": q_success,
        "validations_pass": v_pass,
    }


def score_ui_tester(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a UITesterAgentV2 result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        status = data.get("status", "error")
        steps = data.get("step_results", [])
        assertions = data.get("assertion_results", [])
        fallback_rate = data.get("fallback_rate", 0.0)
    elif hasattr(data, "status"):
        status = data.status
        steps = getattr(data, "step_results", []) or []
        assertions = getattr(data, "assertion_results", []) or []
        fallback_rate = getattr(data, "fallback_rate", 0.0)
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    step_ok = sum(1 for s in steps if (s.get("success") if isinstance(s, dict) else getattr(s, "success", False)))
    assert_ok = sum(1 for a in assertions if (a.get("passed") if isinstance(a, dict) else getattr(a, "passed", False)))

    step_ratio = step_ok / len(steps) if steps else 0.0
    assert_ratio = assert_ok / len(assertions) if assertions else 1.0
    fallback_penalty = max(0.0, 1.0 - fallback_rate)

    score = step_ratio * 0.35 + assert_ratio * 0.45 + fallback_penalty * 0.2
    success = status == "passed" and score >= 0.5

    return success, score, {
        "status": status,
        "steps_total": len(steps),
        "steps_success": step_ok,
        "assertions_pass": assert_ok,
        "fallback_rate": fallback_rate,
    }


def score_flaky_detection(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a FlakyTestDetector result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    level = str(data.get("flakiness_level", "")).lower()
    quarantine = data.get("should_quarantine", False)
    cause = str(data.get("likely_cause", "")).lower()

    expected_level = ground_truth.get("expected_level", "").lower()
    expected_quarantine = ground_truth.get("should_quarantine", False)
    expected_cause = ground_truth.get("expected_cause", "").lower()

    level_ok = expected_level in level or level in expected_level
    quarantine_ok = quarantine == expected_quarantine
    cause_ok = expected_cause in cause or cause in expected_cause if expected_cause else True

    score = (0.4 if level_ok else 0.0) + (0.3 if quarantine_ok else 0.0) + (0.3 if cause_ok else 0.0)
    success = level_ok and quarantine_ok

    return success, score, {
        "detected_level": level,
        "expected_level": expected_level,
        "quarantine_match": quarantine_ok,
        "cause_match": cause_ok,
    }


def score_visual_comparison(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a VisualAI comparison result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    match = data.get("match", False)
    pct = data.get("match_percentage", 0.0)
    diffs = data.get("differences", [])

    expected_match = ground_truth.get("expected_match", True)
    min_pct = ground_truth.get("min_match_percentage", 0)
    max_pct = ground_truth.get("max_match_percentage", 100)
    expected_diff_types = ground_truth.get("expected_diff_types", [])

    match_ok = match == expected_match
    pct_ok = min_pct <= pct <= max_pct

    diff_text = json.dumps(diffs).lower() if diffs else ""
    type_found = sum(1 for t in expected_diff_types if t.lower() in diff_text)
    type_ratio = type_found / len(expected_diff_types) if expected_diff_types else 1.0

    score = (0.4 if match_ok else 0.0) + (0.3 if pct_ok else 0.1) + type_ratio * 0.3
    success = match_ok and score >= 0.5

    return success, score, {
        "match": match,
        "expected_match": expected_match,
        "match_percentage": pct,
        "diff_types_found": type_found,
    }


def score_test_impact(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a TestImpactAnalyzer result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    affected = set(data.get("affected_tests", []))
    risk = data.get("risk_score", 0.0)
    gaps = data.get("coverage_gaps", [])
    suggestions = data.get("new_tests_suggested", [])

    expected_affected = set(ground_truth.get("expected_affected_tests", []))
    min_risk = ground_truth.get("min_risk_score", 0.0)
    max_risk = ground_truth.get("max_risk_score", 1.0)

    overlap = len(affected & expected_affected)
    precision = overlap / len(affected) if affected else 0.0
    recall = overlap / len(expected_affected) if expected_affected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    risk_ok = min_risk <= risk <= max_risk

    score = f1 * 0.5 + (0.3 if risk_ok else 0.1) + min(len(gaps) / 3, 1.0) * 0.2
    success = recall >= 0.5 and score >= 0.4

    return success, score, {
        "affected_found": len(affected),
        "affected_expected": len(expected_affected),
        "overlap": overlap,
        "f1": round(f1, 3),
        "risk_score": risk,
        "gaps_found": len(gaps),
    }


def score_nlp_test_creator(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score an NLPTestCreator result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    steps = data.get("steps", [])
    assertions = data.get("assertions", [])
    priority = data.get("priority", "")
    all_text = json.dumps(data).lower()

    min_steps = ground_truth.get("min_steps", 1)
    must_contain = ground_truth.get("must_contain_keywords", [])

    step_score = min(len(steps) / max(min_steps, 1), 1.0)
    has_assertions = 1.0 if assertions else 0.3
    keyword_found = sum(1 for kw in must_contain if kw.lower() in all_text)
    kw_ratio = keyword_found / len(must_contain) if must_contain else 1.0

    score = step_score * 0.35 + has_assertions * 0.25 + kw_ratio * 0.4
    success = len(steps) >= min_steps and score >= 0.5

    return success, score, {
        "steps_count": len(steps),
        "assertions_count": len(assertions),
        "keywords_found": keyword_found,
        "priority": priority,
    }


def score_mr_analysis(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score an MRAnalyzerAgent result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        risk = data.get("overall_risk_score", data.get("risk_score", 0.0))
        suggestions = data.get("suggestions", data.get("test_suggestions", []))
        gaps = data.get("coverage_gaps", [])
        all_text = json.dumps(data).lower()
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    expected_risk = ground_truth.get("expected_risk_level", "")
    must_suggest = ground_truth.get("must_suggest_test_types", [])
    min_risk = ground_truth.get("min_risk_score", 0.0)
    max_risk = ground_truth.get("max_risk_score", 1.0)

    risk_ok = min_risk <= risk <= max_risk
    suggested_found = sum(1 for t in must_suggest if t.lower() in all_text)
    suggest_ratio = suggested_found / len(must_suggest) if must_suggest else 1.0

    score = (0.3 if risk_ok else 0.1) + suggest_ratio * 0.4 + min(len(gaps) / 3, 1.0) * 0.3
    success = risk_ok and score >= 0.4

    return success, score, {
        "risk_score": risk,
        "risk_in_range": risk_ok,
        "suggestions_found": suggested_found,
        "suggestions_expected": len(must_suggest),
        "gaps_found": len(gaps),
    }


def score_auto_discovery(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score an AutoDiscovery result against ground truth."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    pages = data.get("pages_discovered", [])
    flows = data.get("flows_discovered", [])
    tests = data.get("suggested_tests", [])
    all_text = json.dumps(data).lower()

    min_flows = ground_truth.get("min_flows", 1)
    must_discover = ground_truth.get("must_discover_flows", [])

    flow_score = min(len(flows) / max(min_flows, 1), 1.0)
    discovered = sum(1 for f in must_discover if f.lower() in all_text)
    discover_ratio = discovered / len(must_discover) if must_discover else 1.0
    has_tests = 1.0 if tests else 0.3

    score = flow_score * 0.3 + discover_ratio * 0.5 + has_tests * 0.2
    success = len(flows) >= 1 and score >= 0.4

    return success, score, {
        "pages_found": len(pages),
        "flows_found": len(flows),
        "flows_discovered": discovered,
        "tests_suggested": len(tests),
    }


def score_test_quality(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a TestQualityIntelligenceAgent trust score result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    trust_score = data.get("score", 0.0)
    min_score = ground_truth.get("min_score", 0)
    max_score = ground_truth.get("max_score", 100)

    in_range = min_score <= trust_score <= max_score
    score = 1.0 if in_range else max(0.0, 1.0 - abs(trust_score - (min_score + max_score) / 2) / 50)
    success = in_range

    return success, score, {
        "trust_score": trust_score,
        "expected_range": f"{min_score}-{max_score}",
        "in_range": in_range,
    }


def score_performance_analysis(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a PerformanceAnalyzerAgent result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        grade = data.get("overall_grade", "").upper()
        perf_score = data.get("score", data.get("performance_score", 0))
        issues = data.get("issues", [])
        recs = data.get("recommendations", [])
    elif hasattr(data, "overall_grade"):
        grade = str(data.overall_grade).upper()
        perf_score = getattr(data, "score", 0)
        issues = getattr(data, "issues", []) or []
        recs = getattr(data, "recommendations", []) or []
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    expected_grade = ground_truth.get("expected_grade", "").upper()
    min_score = ground_truth.get("min_score", 0)
    max_score = ground_truth.get("max_score", 100)

    grade_ok = expected_grade in grade or grade in expected_grade
    score_ok = min_score <= perf_score <= max_score

    score = (0.4 if grade_ok else 0.0) + (0.3 if score_ok else 0.1) + min(len(recs) / 3, 1.0) * 0.3
    success = grade_ok and score >= 0.5

    return success, score, {
        "grade": grade,
        "expected_grade": expected_grade,
        "perf_score": perf_score,
        "issues_count": len(issues),
        "recommendations_count": len(recs),
    }


def score_security_scan(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a SecurityScannerAgent result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        vulns = data.get("vulnerabilities", [])
        risk = data.get("risk_score", 0)
        recs = data.get("recommendations", [])
    elif hasattr(data, "vulnerabilities"):
        vulns = data.vulnerabilities or []
        risk = getattr(data, "risk_score", 0)
        recs = getattr(data, "recommendations", []) or []
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    min_risk = ground_truth.get("min_risk_score", 0)
    max_risk = ground_truth.get("max_risk_score", 100)
    min_critical = ground_truth.get("min_critical_findings", 0)

    critical_count = sum(1 for v in vulns if (v.get("severity") if isinstance(v, dict) else getattr(v, "severity", "")).upper() == "CRITICAL")
    risk_ok = min_risk <= risk <= max_risk
    critical_ok = critical_count >= min_critical

    score = (0.4 if risk_ok else 0.1) + (0.3 if critical_ok else 0.0) + min(len(recs) / 3, 1.0) * 0.3
    success = risk_ok and score >= 0.5

    return success, score, {
        "risk_score": risk,
        "risk_in_range": risk_ok,
        "critical_findings": critical_count,
        "total_vulns": len(vulns),
    }


def score_accessibility_check(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score an AccessibilityCheckerAgent result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        a11y_score = data.get("score", 0)
        issues = data.get("issues", [])
        passes_aa = data.get("passes_wcag_aa", False)
    elif hasattr(data, "score"):
        a11y_score = data.score
        issues = getattr(data, "issues", []) or []
        passes_aa = getattr(data, "passes_wcag_aa", callable(getattr(data, "passes_wcag_aa", None)) and data.passes_wcag_aa())
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    min_score = ground_truth.get("min_score", 0)
    max_score = ground_truth.get("max_score", 100)
    expected_aa = ground_truth.get("expect_passes_aa")

    score_ok = min_score <= a11y_score <= max_score
    aa_ok = (expected_aa is None) or (passes_aa == expected_aa)

    score = (0.5 if score_ok else 0.1) + (0.3 if aa_ok else 0.0) + 0.2
    success = score_ok and aa_ok

    return success, score, {
        "a11y_score": a11y_score,
        "score_in_range": score_ok,
        "passes_aa": passes_aa,
        "issues_count": len(issues),
    }


def score_agent_judge(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score an AgentAsJudge evaluation result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    overall = data.get("overall_score", 0.0)
    acceptable = data.get("is_acceptable", False)
    gt_match = data.get("ground_truth_match")

    min_score = ground_truth.get("min_overall_score", 0.0)
    max_score = ground_truth.get("max_overall_score", 1.0)
    expected_acceptable = ground_truth.get("expect_acceptable")

    score_ok = min_score <= overall <= max_score
    accept_ok = (expected_acceptable is None) or (acceptable == expected_acceptable)

    score = (0.4 if score_ok else 0.1) + (0.3 if accept_ok else 0.0) + min(overall, 1.0) * 0.3
    success = score_ok and accept_ok

    return success, score, {
        "overall_score": overall,
        "is_acceptable": acceptable,
        "ground_truth_match": gt_match,
    }


def score_corrective_rag(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a CorrectiveRAGAgent result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    confidence = data.get("confidence", 0.0)
    relevance = data.get("overall_relevance", 0.0)
    actions = data.get("actions_taken", [])
    web_used = data.get("web_search_used", False)
    rounds = data.get("retrieval_rounds", 1)

    expected_action = ground_truth.get("expected_action", "")
    min_confidence = ground_truth.get("min_confidence", 0.0)
    expect_web = ground_truth.get("expect_web_search")

    action_ok = not expected_action or expected_action.upper() in [a.upper() for a in actions]
    conf_ok = confidence >= min_confidence
    web_ok = (expect_web is None) or (web_used == expect_web)

    score = (0.4 if action_ok else 0.0) + (0.3 if conf_ok else 0.1) + (0.3 if web_ok else 0.1)
    success = action_ok and score >= 0.5

    return success, score, {
        "confidence": confidence,
        "relevance": relevance,
        "actions_taken": actions,
        "web_search_used": web_used,
        "retrieval_rounds": rounds,
    }


def score_reporter(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a ReporterAgent result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        summary = data.get("summary", {})
        failures = data.get("failures", [])
        recs = data.get("recommendations", [])
        has_report = bool(data.get("markdown_report") or data.get("html_report"))
    elif hasattr(data, "summary"):
        summary = data.summary.__dict__ if hasattr(data.summary, "__dict__") else {}
        failures = getattr(data, "failures", []) or []
        recs = getattr(data, "recommendations", []) or []
        has_report = bool(getattr(data, "markdown_report", ""))
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    pass_rate = summary.get("pass_rate", 0) if isinstance(summary, dict) else 0
    expected_rate = ground_truth.get("expected_pass_rate", 0)
    expected_failures = ground_truth.get("expected_failure_count", 0)

    rate_ok = abs(pass_rate - expected_rate) <= 5
    failures_ok = len(failures) == expected_failures or abs(len(failures) - expected_failures) <= 1

    score = (0.3 if rate_ok else 0.1) + (0.3 if failures_ok else 0.1) + (0.2 if has_report else 0.0) + min(len(recs) / 3, 1.0) * 0.2
    success = rate_ok and score >= 0.5

    return success, score, {
        "pass_rate": pass_rate,
        "failure_count": len(failures),
        "has_report": has_report,
        "recommendations_count": len(recs),
    }


def score_recon(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a ReconAgent result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data.data if hasattr(result_data, "data") else result_data
    if isinstance(data, dict):
        ports = data.get("ports", [])
        techs = data.get("technologies", [])
        endpoints = data.get("endpoints", [])
    elif hasattr(data, "ports"):
        ports = data.ports or []
        techs = data.technologies or []
        endpoints = data.endpoints or []
    else:
        return False, 0.0, {"reason": "Unexpected result type"}

    min_ports = ground_truth.get("min_ports", 1)
    min_techs = ground_truth.get("min_technologies", 0)
    min_endpoints = ground_truth.get("min_endpoints", 0)

    port_score = min(len(ports) / max(min_ports, 1), 1.0)
    tech_score = min(len(techs) / max(min_techs, 1), 1.0) if min_techs > 0 else 1.0
    ep_score = min(len(endpoints) / max(min_endpoints, 1), 1.0) if min_endpoints > 0 else 1.0

    score = port_score * 0.4 + tech_score * 0.3 + ep_score * 0.3
    success = len(ports) >= min_ports and score >= 0.5

    return success, score, {
        "ports_found": len(ports),
        "technologies_found": len(techs),
        "endpoints_found": len(endpoints),
    }


def score_session_to_test(result_data: Any, ground_truth: dict) -> tuple[bool, float, dict]:
    """Score a SessionToTestConverter result."""
    if not result_data:
        return False, 0.0, {"reason": "No result data"}

    data = result_data if isinstance(result_data, dict) else (
        result_data.__dict__ if hasattr(result_data, "__dict__") else {}
    )

    test_worthy = data.get("test_worthy", False)
    priority = data.get("test_priority", data.get("priority", ""))
    steps = data.get("steps", [])
    confidence = data.get("confidence", 0.0)

    expected_worthy = ground_truth.get("expect_test_worthy", True)
    expected_priority = ground_truth.get("expected_priority", "")
    min_steps = ground_truth.get("min_steps", 1)

    worthy_ok = test_worthy == expected_worthy
    priority_ok = not expected_priority or expected_priority.lower() in priority.lower()
    steps_ok = len(steps) >= min_steps

    score = (0.3 if worthy_ok else 0.0) + (0.3 if priority_ok else 0.1) + (0.4 if steps_ok else 0.2)
    success = worthy_ok and score >= 0.5

    return success, score, {
        "test_worthy": test_worthy,
        "priority": priority,
        "steps_count": len(steps),
        "confidence": confidence,
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

async def run_testgen_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run TestGenAgent against requirements-to-test scenarios."""
    from src.agents.testgen_agent import TestGenAgent

    dataset = _load_dataset("testgen.json")
    suite = SuiteResult(
        suite_name="testgen",
        agent_type="TestGenAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        for run_idx in range(runs_per_scenario):
            agent = TestGenAgent(
                org_id=BENCHMARK_ORG_ID,
                project_id=BENCHMARK_PROJECT_ID or "00000000-0000-0000-0000-000000000001",
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.execute(
                    source_type="requirements_text",
                    source_data={"text": scenario["requirements_text"]},
                    config=scenario.get("config", {"framework": "playwright", "language": "typescript"}),
                )

                success, score, details = score_testgen(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="TestGenAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="testgen",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


async def run_healing_cicd_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run TestHealerCICDAgent against CI/CD failure scenarios."""
    from src.agents.testhealer_cicd_agent import TestHealerCICDAgent

    dataset = _load_dataset("healing_cicd.json")
    suite = SuiteResult(
        suite_name="healing_cicd",
        agent_type="TestHealerCICDAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        for run_idx in range(runs_per_scenario):
            agent = TestHealerCICDAgent(
                org_id=BENCHMARK_ORG_ID,
                project_id=BENCHMARK_PROJECT_ID or "00000000-0000-0000-0000-000000000001",
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.execute(
                    scan_type=scenario.get("scan_type", "ci"),
                    trigger_context=scenario.get("trigger_context", {}),
                )

                success, score, details = score_healing_cicd(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="TestHealerCICDAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="healing_cicd",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


async def run_qa_engineer_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run QAEngineerAgent against repo quality analysis scenarios."""
    from src.agents.qa_engineer_agent import QAEngineerAgent

    dataset = _load_dataset("qa_engineer.json")
    suite = SuiteResult(
        suite_name="qa_engineer",
        agent_type="QAEngineerAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        for run_idx in range(runs_per_scenario):
            agent = QAEngineerAgent(
                org_id=BENCHMARK_ORG_ID,
                project_id=BENCHMARK_PROJECT_ID or "00000000-0000-0000-0000-000000000001",
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.execute(
                    analysis_type=scenario.get("analysis_type", "full"),
                    config=scenario.get("config", {}),
                )

                success, score, details = score_qa_engineer(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            sr = ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="QAEngineerAgent",
                success=success,
                score=score,
                latency_ms=latency,
                cost_usd=cost,
                details=details,
            )
            suite.results.append(sr)

            logger.info(
                "benchmark_scenario_complete",
                suite="qa_engineer",
                scenario=scenario["id"],
                run=run_idx,
                success=success,
                score=round(score, 3),
                latency_ms=latency,
            )

    return suite


async def run_test_planner_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run TestPlannerAgent against test planning scenarios."""
    from src.agents.test_planner import TestPlannerAgent

    raw = _load_dataset("test_planner.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="test_planner",
        agent_type="TestPlannerAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        surfaces = inp.get("testable_surfaces", [])
        app_url = inp.get("app_url", "https://example.com")
        codebase_summary = inp.get("app_description", "")

        for run_idx in range(runs_per_scenario):
            agent = TestPlannerAgent(
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.execute(
                    testable_surfaces=surfaces,
                    app_url=app_url,
                    codebase_summary=codebase_summary,
                )
                success, score, details = score_test_planner(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="TestPlannerAgent",
                success=success, score=score,
                latency_ms=latency, cost_usd=cost, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="test_planner",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_test_impact_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run TestImpactAnalyzer against impact analysis scenarios."""
    from src.agents.test_impact_analyzer import CodeChange, TestImpactAnalyzer

    raw = _load_dataset("test_impact.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="test_impact",
        agent_type="TestImpactAnalyzer",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        change = CodeChange(
            commit_sha=inp.get("commit_sha", "benchmark"),
            branch=inp.get("branch", "main"),
            changed_files=inp.get("changed_files", []),
        )
        all_tests = inp.get("existing_tests", [])

        for run_idx in range(runs_per_scenario):
            analyzer = TestImpactAnalyzer()

            start = time.time()
            try:
                result = await analyzer.analyze_impact(
                    change=change,
                    all_tests=all_tests,
                )
                # Wrap result for scoring
                result_dict = {
                    "affected_tests": [t.get("id", t) if isinstance(t, dict) else str(t)
                                       for t in (result.affected_tests if hasattr(result, "affected_tests") else [])],
                    "risk_score": getattr(result, "risk_score", 0),
                    "suggested_new_tests": getattr(result, "suggested_new_tests", []),
                    "impact_category": getattr(result, "impact_category", ""),
                }
                success, score, details = score_test_impact(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="TestImpactAnalyzer",
                success=success, score=score,
                latency_ms=latency, cost_usd=0.0, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="test_impact",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_nlp_test_creator_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run NLPTestCreator against natural-language-to-test scenarios."""
    from src.agents.nlp_test_creator import NLPTestCreator

    raw = _load_dataset("nlp_test_creator.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="nlp_test_creator",
        agent_type="NLPTestCreator",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        description = inp.get("natural_language", "")
        app_url = inp.get("app_url", "https://example.com")
        context = json.dumps(inp.get("context", {}))

        for run_idx in range(runs_per_scenario):
            creator = NLPTestCreator(app_url=app_url)

            start = time.time()
            try:
                result = await creator.create(
                    description=description,
                    context=context,
                )
                # Convert GeneratedTest to dict for scoring
                # (steps/assertions are nested dataclasses — __dict__ alone
                #  leaves them as objects which breaks json.dumps in scoring)
                result_dict = _deep_to_dict(result)
                success, score, details = score_nlp_test_creator(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="NLPTestCreator",
                success=success, score=score,
                latency_ms=latency, cost_usd=0.0, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="nlp_test_creator",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_mr_analysis_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run MRAnalyzerAgent against merge request analysis scenarios."""
    from src.agents.mr_analyzer import MRAnalyzerAgent

    raw = _load_dataset("mr_analysis.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="mr_analysis",
        agent_type="MRAnalyzerAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        changes = inp.get("changed_files", [])
        project_id = BENCHMARK_PROJECT_ID or "00000000-0000-0000-0000-000000000001"

        for run_idx in range(runs_per_scenario):
            agent = MRAnalyzerAgent(
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.analyze(
                    changes=changes,
                    project_id=project_id,
                )
                # Convert MRAnalysis to dict for scoring
                # (file_analyses contains nested ChangeAnalysis dataclasses)
                result_dict = _deep_to_dict(result)
                success, score, details = score_mr_analysis(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="MRAnalyzerAgent",
                success=success, score=score,
                latency_ms=latency, cost_usd=cost, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="mr_analysis",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_agent_judge_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run AgentAsJudge against evaluation scenarios."""
    from src.agents.agent_judge import AgentAsJudge

    raw = _load_dataset("agent_judge.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="agent_judge",
        agent_type="AgentAsJudge",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        agent_type = inp.get("agent_type", "UITesterAgent")
        task = inp.get("task", "")
        output = inp.get("output", {})
        context = inp.get("context", {})
        reference = inp.get("reference_answer")

        for run_idx in range(runs_per_scenario):
            agent = AgentAsJudge(
                enable_execution=False,  # No real execution in benchmarks
                enable_ground_truth=reference is not None,
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.evaluate_agent_output(
                    agent_type=agent_type,
                    input_context={"task": task, **context},
                    output=output,
                    ground_truth=reference,
                )
                result_dict = _deep_to_dict(result)
                success, score, details = score_agent_judge(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="AgentAsJudge",
                success=success, score=score,
                latency_ms=latency, cost_usd=cost, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="agent_judge",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_corrective_rag_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run CorrectiveRAGAgent against retrieval scenarios."""
    from src.agents.corrective_rag_agent import CorrectiveRAGAgent

    raw = _load_dataset("corrective_rag.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="corrective_rag",
        agent_type="CorrectiveRAGAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        query = inp.get("query", "")
        docs = inp.get("retrieved_documents", [])
        namespace = inp.get("namespace", "")

        for run_idx in range(runs_per_scenario):
            agent = CorrectiveRAGAgent(
                retriever=None,  # No real retriever in benchmarks
                enable_web_fallback=False,  # Don't do real web searches
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.query(
                    query=query,
                    context={"documents": docs, "namespace": namespace},
                )
                # CRAGResult has nested RetrievedDocument dataclasses
                result_dict = _deep_to_dict(result)
                success, score, details = score_corrective_rag(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="CorrectiveRAGAgent",
                success=success, score=score,
                latency_ms=latency, cost_usd=cost, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="corrective_rag",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_reporter_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run ReporterAgent against report generation scenarios."""
    from src.agents.reporter import ReporterAgent

    raw = _load_dataset("reporter.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="reporter",
        agent_type="ReporterAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        test_results = inp.get("test_results", [])

        for run_idx in range(runs_per_scenario):
            agent = ReporterAgent(
                use_multi_model=_should_use_multi_model(),
            )

            start = time.time()
            try:
                result = await agent.execute(
                    test_results=test_results,
                    execution_metadata={
                        "run_id": inp.get("run_id", "benchmark"),
                        "project_id": inp.get("project_id", "benchmark"),
                    },
                )
                success, score, details = score_reporter(
                    result, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)
            cost = agent._usage.total_cost if hasattr(agent, "_usage") and agent._usage else 0.0

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="ReporterAgent",
                success=success, score=score,
                latency_ms=latency, cost_usd=cost, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="reporter",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_test_quality_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run TestQualityIntelligenceAgent trust score computation.

    This is deterministic (no AI), so runs_per_scenario > 1 just
    verifies consistency.
    """
    from src.agents.test_quality_agent import TestQualityIntelligenceAgent

    raw = _load_dataset("test_quality.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="test_quality",
        agent_type="TestQualityIntelligenceAgent",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]

        for run_idx in range(runs_per_scenario):
            agent = TestQualityIntelligenceAgent()

            start = time.time()
            try:
                result = agent.compute_trust_score(
                    test_id=inp.get("test_id", "benchmark_test"),
                    assertion_strength=inp.get("assertion_strength", "no_assertion"),
                    total_heals=inp.get("total_heals", 0),
                    suspicious_heals=inp.get("suspicious_heals", 0),
                    regression_masking_heals=inp.get("regression_masking_heals", 0),
                    flakiness_rate=inp.get("flakiness_rate", 0.0),
                    days_since_creation=inp.get("days_since_creation", 1),
                    coverage_factor=inp.get("coverage_factor", 0.5),
                )
                result_dict = {"score": result.score} if hasattr(result, "score") else (
                    result.__dict__ if hasattr(result, "__dict__") else {"score": result}
                )
                success, score, details = score_test_quality(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="TestQualityIntelligenceAgent",
                success=success, score=score,
                latency_ms=latency, cost_usd=0.0, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="test_quality",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


async def run_flaky_detection_suite(runs_per_scenario: int = 8) -> SuiteResult:
    """Run FlakyTestDetector against flakiness detection scenarios.

    FlakyTestDetector.analyze_test() reads from internal history,
    so we inject execution records first via record_execution().
    Deterministic — runs_per_scenario > 1 verifies consistency.
    """
    from datetime import datetime as dt

    from src.agents.flaky_detector import FlakyTestDetector, TestRun

    raw = _load_dataset("flaky_detection.json")
    dataset = raw["scenarios"] if isinstance(raw, dict) else raw
    suite = SuiteResult(
        suite_name="flaky_detection",
        agent_type="FlakyTestDetector",
        total_scenarios=len(dataset),
        total_runs=len(dataset) * runs_per_scenario,
    )

    for scenario in dataset:
        inp = scenario["input"]
        test_id = inp.get("test_id", "benchmark_test")
        test_name = inp.get("test_name", test_id)
        history = inp.get("execution_history", [])

        for run_idx in range(runs_per_scenario):
            detector = FlakyTestDetector()

            start = time.time()
            try:
                # Inject execution history via record_run(TestRun)
                for entry in history:
                    ts = entry.get("timestamp", "2026-02-20T10:00:00Z")
                    run = TestRun(
                        test_id=test_id,
                        passed=entry.get("status", "passed") == "passed",
                        duration_ms=entry.get("duration_ms", 1000),
                        timestamp=dt.fromisoformat(ts.replace("Z", "+00:00")),
                        error_message=entry.get("error"),
                        environment=inp.get("environment", "ci"),
                    )
                    detector.record_run(run)

                result = detector.analyze_test(
                    test_id=test_id,
                    test_name=test_name,
                )
                result_dict = result.__dict__ if hasattr(result, "__dict__") else result
                success, score, details = score_flaky_detection(
                    result_dict, scenario["ground_truth"]
                )
            except Exception as e:
                success = False
                score = 0.0
                details = {"error": str(e)}

            latency = int((time.time() - start) * 1000)

            suite.results.append(ScenarioResult(
                scenario_id=f"{scenario['id']}_run{run_idx}",
                agent_type="FlakyTestDetector",
                success=success, score=score,
                latency_ms=latency, cost_usd=0.0, details=details,
            ))
            logger.info("benchmark_scenario_complete", suite="flaky_detection",
                        scenario=scenario["id"], run=run_idx, success=success,
                        score=round(score, 3), latency_ms=latency)

    return suite


# =============================================================================
# Persistence — Write ScenarioResults to agent_evaluations table
# =============================================================================

# Map suite names → task_type values for the evaluations table
SUITE_TASK_TYPES: dict[str, str] = {
    "self_healing": "self_healing",
    "code_analysis": "code_analysis",
    "hallucination_detection": "error_classification",
    "sre_incidents": "root_cause_analysis",
    "testgen": "code_analysis",
    "healing_cicd": "self_healing",
    "qa_engineer": "code_analysis",
    "test_planner": "code_analysis",
    "test_impact": "code_analysis",
    "nlp_test_creator": "code_analysis",
    "mr_analysis": "code_analysis",
    "agent_judge": "error_classification",
    "corrective_rag": "semantic_understanding",
    "reporter": "text_extraction",
    "test_quality": "code_analysis",
    "flaky_detection": "error_classification",
}


async def _persist_suite_results(suite: SuiteResult, suite_name: str) -> int:
    """Persist all ScenarioResults from a suite into agent_evaluations table.

    Returns the number of records successfully inserted.
    """
    try:
        from src.services.supabase_client import get_supabase_client

        supabase = get_supabase_client()
    except Exception as e:
        logger.error("persist_results_no_supabase", error=str(e))
        return 0

    task_type = SUITE_TASK_TYPES.get(suite_name, "general")
    inserted = 0

    for r in suite.results:
        record = {
            "organization_id": BENCHMARK_ORG_ID,
            "project_id": BENCHMARK_PROJECT_ID,
            "agent_type": r.agent_type,
            "task_type": task_type,
            "task_completed": r.success,
            "efficacy_score": r.score,
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "assurance_score": r.score,  # use efficacy as assurance for benchmarks
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_calls_count": 0,
            "self_corrections": 0,
            "human_escalated": False,
            "error_type": "benchmark_failure" if not r.success else None,
            "error_message": (r.error or r.details.get("error", ""))[:500] if not r.success else None,
            "trigger_source": "benchmark",
            "metadata": {
                "benchmark_suite": suite_name,
                "scenario_id": r.scenario_id,
                "details": r.details,
            },
        }
        try:
            await supabase.insert("agent_evaluations", record)
            inserted += 1
        except Exception as e:
            logger.error(
                "persist_evaluation_failed",
                suite=suite_name,
                scenario=r.scenario_id,
                error=str(e),
            )

    logger.info(
        "persist_suite_results",
        suite=suite_name,
        agent=suite.agent_type,
        inserted=inserted,
        total=len(suite.results),
    )
    return inserted


SUITES = {
    # Original suites (7)
    "self_healing": run_self_healing_suite,
    "code_analysis": run_code_analysis_suite,
    "hallucination_detection": run_hallucination_suite,
    "sre_incidents": run_sre_suite,
    "testgen": run_testgen_suite,
    "healing_cicd": run_healing_cicd_suite,
    "qa_engineer": run_qa_engineer_suite,
    # New executable suites — AI-powered (7)
    "test_planner": run_test_planner_suite,
    "test_impact": run_test_impact_suite,
    "nlp_test_creator": run_nlp_test_creator_suite,
    "mr_analysis": run_mr_analysis_suite,
    "agent_judge": run_agent_judge_suite,
    "corrective_rag": run_corrective_rag_suite,
    "reporter": run_reporter_suite,
    # New executable suites — deterministic (2)
    "test_quality": run_test_quality_suite,
    "flaky_detection": run_flaky_detection_suite,
    # Infrastructure-dependent (NOT runnable without browser/DB/network)
    # "api_tester": requires live API endpoint
    # "db_tester": requires database connection
    # "ui_tester": requires browser + running app
    # "visual_comparison": requires screenshots to compare
    # "auto_discovery": requires running app to crawl
    # "performance_analysis": requires running app to measure
    # "security_scan": requires target to scan
    # "accessibility_check": requires rendered HTML pages
    # "recon": requires network target
    # "session_to_test": requires recorded session data
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

            # Persist results to agent_evaluations table
            try:
                persisted = await _persist_suite_results(result, name)
                logger.info("benchmark_results_persisted", suite=name, records=persisted)
            except Exception as pe:
                logger.error("benchmark_persist_failed", suite=name, error=str(pe))

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
