"""
Test Quality Intelligence Agent — Trust Scores and Suite Audit.

Computes per-test trust scores and suite-level quality metrics.
This is the product nobody has built: auditing whether your test suite is
trustworthy, not just whether tests pass.

Trust score formula (deterministic, no LLM):
  score = 100 * (
      assertion_strength * 0.30
    + heal_penalty       * 0.25
    + flaky_penalty      * 0.20
    + age_factor         * 0.10
    + coverage_factor    * 0.15
  )

Designed to run as a daily precomputed job (4 AM UTC) stored in the
intelligence_precomputed table via upsert_precomputed() RPC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.model_router import TaskType
from .base import AgentCapability, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class TrustScore:
    """Per-test trust score with component breakdown."""

    test_id: str
    score: float  # 0-100
    assertion_strength_score: float  # 0-1
    heal_penalty_score: float  # 0-1 (1 = no heals, 0 = heavily healed)
    flaky_penalty_score: float  # 0-1 (1 = stable, 0 = very flaky)
    age_factor_score: float  # 0-1
    coverage_factor_score: float  # 0-1

    # Context
    total_heals: int = 0
    suspicious_heals: int = 0
    regression_masking_heals: int = 0
    flakiness_rate: float = 0.0
    days_since_creation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "score": round(self.score, 1),
            "components": {
                "assertion_strength": round(self.assertion_strength_score, 3),
                "heal_penalty": round(self.heal_penalty_score, 3),
                "flaky_penalty": round(self.flaky_penalty_score, 3),
                "age_factor": round(self.age_factor_score, 3),
                "coverage_factor": round(self.coverage_factor_score, 3),
            },
            "context": {
                "total_heals": self.total_heals,
                "suspicious_heals": self.suspicious_heals,
                "regression_masking_heals": self.regression_masking_heals,
                "flakiness_rate": round(self.flakiness_rate, 3),
                "days_since_creation": self.days_since_creation,
            },
        }


@dataclass
class SuiteAudit:
    """Suite-level quality audit."""

    project_id: str
    total_tests: int
    avg_trust_score: float
    median_trust_score: float

    # Distribution
    high_trust_count: int  # score >= 80
    medium_trust_count: int  # 50 <= score < 80
    low_trust_count: int  # score < 50

    # Heal metrics
    total_heals: int
    tests_with_suspicious_heals: int
    tests_with_regression_masking: int

    # Assertion strength distribution
    exact_match_assertions: int
    contains_assertions: int
    exists_only_assertions: int
    no_assertions: int

    # Flakiness
    flaky_test_count: int
    avg_flakiness_rate: float

    computed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "total_tests": self.total_tests,
            "avg_trust_score": round(self.avg_trust_score, 1),
            "median_trust_score": round(self.median_trust_score, 1),
            "distribution": {
                "high_trust": self.high_trust_count,
                "medium_trust": self.medium_trust_count,
                "low_trust": self.low_trust_count,
            },
            "heal_metrics": {
                "total_heals": self.total_heals,
                "tests_with_suspicious_heals": self.tests_with_suspicious_heals,
                "tests_with_regression_masking": self.tests_with_regression_masking,
            },
            "assertion_strength_distribution": {
                "exact_match": self.exact_match_assertions,
                "contains": self.contains_assertions,
                "exists_only": self.exists_only_assertions,
                "no_assertion": self.no_assertions,
            },
            "flakiness": {
                "flaky_test_count": self.flaky_test_count,
                "avg_flakiness_rate": round(self.avg_flakiness_rate, 3),
            },
            "computed_at": self.computed_at,
        }


# =============================================================================
# Weights for trust score formula
# =============================================================================

WEIGHT_ASSERTION_STRENGTH = 0.30
WEIGHT_HEAL_PENALTY = 0.25
WEIGHT_FLAKY_PENALTY = 0.20
WEIGHT_AGE_FACTOR = 0.10
WEIGHT_COVERAGE_FACTOR = 0.15

# Assertion strength scores
ASSERTION_STRENGTH_MAP = {
    "exact_match": 1.0,
    "contains": 0.7,
    "exists": 0.4,
    "no_assertion": 0.0,
}


class TestQualityIntelligenceAgent(BaseAgent):
    """Computes per-test trust scores and suite-level quality audits.

    This agent extends BaseAgent but primarily uses deterministic computation
    (no LLM needed for trust scores). The suite audit may optionally use LLM
    for "trivially passing" test detection in a future enhancement.
    """

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    CAPABILITIES = [
        AgentCapability.QUALITY_SCORING,
    ]

    def _get_system_prompt(self) -> str:
        return "You are a test quality intelligence agent that computes trust scores."

    def compute_trust_score(
        self,
        test_id: str,
        assertion_strength: str = "no_assertion",
        total_heals: int = 0,
        suspicious_heals: int = 0,
        regression_masking_heals: int = 0,
        flakiness_rate: float = 0.0,
        days_since_creation: int = 0,
        coverage_factor: float = 0.5,  # Default to 0.5 if unknown
    ) -> TrustScore:
        """Compute trust score for a single test. Deterministic, no LLM.

        Args:
            test_id: Test identifier
            assertion_strength: One of exact_match, contains, exists, no_assertion
            total_heals: Total number of times this test has been healed
            suspicious_heals: Number of heals classified as suspicious
            regression_masking_heals: Number of heals classified as regression_masking
            flakiness_rate: Fraction of runs that are flaky (0.0-1.0)
            days_since_creation: Days since test was created
            coverage_factor: Unique code paths covered (0.0-1.0)

        Returns:
            TrustScore with component breakdown
        """
        # 1. Assertion strength (0-1)
        a_score = ASSERTION_STRENGTH_MAP.get(assertion_strength, 0.0)

        # 2. Heal penalty (0-1, exponential decay)
        # Each clean heal costs 8%, suspicious costs 12%, regression costs 20%
        heal_cost = (
            (total_heals - suspicious_heals - regression_masking_heals) * 0.08
            + suspicious_heals * 0.12
            + regression_masking_heals * 0.20
        )
        h_score = max(0.0, 1.0 - heal_cost)

        # 3. Flaky penalty (0-1)
        f_score = max(0.0, 1.0 - flakiness_rate)

        # 4. Age factor (0-1, newer tests get lower trust until proven)
        age_score = min(1.0, days_since_creation / 90.0) if days_since_creation >= 0 else 0.0

        # 5. Coverage factor (0-1, passed through)
        c_score = max(0.0, min(1.0, coverage_factor))

        # Weighted sum
        raw_score = (
            a_score * WEIGHT_ASSERTION_STRENGTH
            + h_score * WEIGHT_HEAL_PENALTY
            + f_score * WEIGHT_FLAKY_PENALTY
            + age_score * WEIGHT_AGE_FACTOR
            + c_score * WEIGHT_COVERAGE_FACTOR
        )

        final_score = round(raw_score * 100, 1)

        return TrustScore(
            test_id=test_id,
            score=final_score,
            assertion_strength_score=a_score,
            heal_penalty_score=h_score,
            flaky_penalty_score=f_score,
            age_factor_score=age_score,
            coverage_factor_score=c_score,
            total_heals=total_heals,
            suspicious_heals=suspicious_heals,
            regression_masking_heals=regression_masking_heals,
            flakiness_rate=flakiness_rate,
            days_since_creation=days_since_creation,
        )

    async def run_trust_score_job(
        self,
        org_id: str,
        project_id: str,
    ) -> list[TrustScore]:
        """Run daily precomputed trust score job for a project.

        Fetches all tests, heal history, and flakiness data, then computes
        scores and stores them in intelligence_precomputed table.

        Args:
            org_id: Organization ID
            project_id: Project ID

        Returns:
            List of computed TrustScores
        """
        from src.services.supabase_client import get_supabase_client

        supabase = get_supabase_client()
        scores: list[TrustScore] = []

        try:
            # 1. Fetch all tests for project
            result = await supabase.select(
                "tests",
                columns="id,created_at,healing_count",
                filters={"project_id": f"eq.{project_id}"},
            )
            tests = result.get("data", [])
            if not tests:
                logger.info("No tests found for project", extra={"project_id": project_id})
                return scores

            test_ids = [t["id"] for t in tests]

            # 2. Fetch heal validations in bulk
            result = await supabase.select(
                "heal_validations",
                columns="test_id,heal_quality",
                filters={
                    "project_id": f"eq.{project_id}",
                    "test_id": f"in.({','.join(test_ids)})",
                },
            )
            heal_data = result.get("data", [])

            # Aggregate heal counts per test
            heal_counts: dict[str, dict[str, int]] = {}
            for h in heal_data:
                tid = h["test_id"]
                if tid not in heal_counts:
                    heal_counts[tid] = {"total": 0, "suspicious": 0, "masking": 0}
                heal_counts[tid]["total"] += 1
                if h["heal_quality"] == "suspicious_heal":
                    heal_counts[tid]["suspicious"] += 1
                elif h["heal_quality"] == "regression_masking":
                    heal_counts[tid]["masking"] += 1

            # 3. Fetch recent test runs for flakiness (last 30 days)
            since = (datetime.now(UTC) - timedelta(days=30)).isoformat().replace("+00:00", "Z")
            result = await supabase.select(
                "test_runs",
                columns="test_id,status",
                filters={
                    "project_id": f"eq.{project_id}",
                    "created_at": f"gte.{since}",
                },
            )
            runs = result.get("data", [])

            # Calculate flakiness per test (transitions between pass/fail)
            run_sequences: dict[str, list[str]] = {}
            for r in runs:
                tid = r.get("test_id")
                if tid:
                    run_sequences.setdefault(tid, []).append(r.get("status", "unknown"))

            flakiness: dict[str, float] = {}
            for tid, statuses in run_sequences.items():
                if len(statuses) < 2:
                    flakiness[tid] = 0.0
                    continue
                transitions = sum(
                    1 for i in range(1, len(statuses))
                    if statuses[i] != statuses[i - 1]
                )
                flakiness[tid] = transitions / (len(statuses) - 1)

            # 4. Compute scores
            now = datetime.now(UTC)
            for test in tests:
                tid = test["id"]
                created_at = test.get("created_at")
                days_old = 0
                if created_at:
                    try:
                        from dateutil.parser import parse as parse_dt
                        created_dt = parse_dt(created_at)
                        days_old = max(0, (now - created_dt).days)
                    except Exception:
                        days_old = 90  # Default to mature

                hc = heal_counts.get(tid, {"total": 0, "suspicious": 0, "masking": 0})

                score = self.compute_trust_score(
                    test_id=tid,
                    assertion_strength="no_assertion",  # Default — could be enhanced with spec analysis
                    total_heals=hc["total"] + test.get("healing_count", 0),
                    suspicious_heals=hc["suspicious"],
                    regression_masking_heals=hc["masking"],
                    flakiness_rate=flakiness.get(tid, 0.0),
                    days_since_creation=days_old,
                )
                scores.append(score)

            # 5. Store all scores as a single JSONB blob in intelligence_precomputed
            # Uses upsert_precomputed() RPC — handles ON CONFLICT (org_id, project_id, computation_type)
            try:
                import json
                scores_blob = {s.test_id: s.to_dict() for s in scores}
                await supabase.rpc("upsert_precomputed", {
                    "p_org_id": org_id,
                    "p_project_id": project_id,
                    "p_type": "test_trust_scores",
                    "p_result": json.dumps(scores_blob),
                    "p_valid_hours": 25,
                })
            except Exception as e:
                logger.warning("Failed to store trust scores", extra={"error": str(e)})

            logger.info(
                "Trust score job completed",
                extra={
                    "project_id": project_id,
                    "tests_scored": len(scores),
                    "avg_score": sum(s.score for s in scores) / len(scores) if scores else 0,
                },
            )

        except Exception as e:
            logger.error("Trust score job failed", extra={"project_id": project_id, "error": str(e)})

        return scores

    async def compute_suite_audit(
        self,
        org_id: str,
        project_id: str,
        trust_scores: list[TrustScore] | None = None,
    ) -> SuiteAudit:
        """Compute suite-level quality audit.

        Args:
            org_id: Organization ID
            project_id: Project ID
            trust_scores: Pre-computed scores (or None to compute fresh)

        Returns:
            SuiteAudit with suite-level metrics
        """
        if trust_scores is None:
            trust_scores = await self.run_trust_score_job(org_id, project_id)

        if not trust_scores:
            return SuiteAudit(
                project_id=project_id,
                total_tests=0,
                avg_trust_score=0.0,
                median_trust_score=0.0,
                high_trust_count=0,
                medium_trust_count=0,
                low_trust_count=0,
                total_heals=0,
                tests_with_suspicious_heals=0,
                tests_with_regression_masking=0,
                exact_match_assertions=0,
                contains_assertions=0,
                exists_only_assertions=0,
                no_assertions=0,
                flaky_test_count=0,
                avg_flakiness_rate=0.0,
                computed_at=datetime.now(UTC).isoformat(),
            )

        all_scores = sorted([s.score for s in trust_scores])
        n = len(all_scores)
        median = all_scores[n // 2] if n > 0 else 0.0

        return SuiteAudit(
            project_id=project_id,
            total_tests=n,
            avg_trust_score=sum(all_scores) / n,
            median_trust_score=median,
            high_trust_count=sum(1 for s in all_scores if s >= 80),
            medium_trust_count=sum(1 for s in all_scores if 50 <= s < 80),
            low_trust_count=sum(1 for s in all_scores if s < 50),
            total_heals=sum(s.total_heals for s in trust_scores),
            tests_with_suspicious_heals=sum(
                1 for s in trust_scores if s.suspicious_heals > 0
            ),
            tests_with_regression_masking=sum(
                1 for s in trust_scores if s.regression_masking_heals > 0
            ),
            # Assertion distribution — requires analysis of test specs (default for now)
            exact_match_assertions=sum(
                1 for s in trust_scores if s.assertion_strength_score == 1.0
            ),
            contains_assertions=sum(
                1 for s in trust_scores if s.assertion_strength_score == 0.7
            ),
            exists_only_assertions=sum(
                1 for s in trust_scores if s.assertion_strength_score == 0.4
            ),
            no_assertions=sum(
                1 for s in trust_scores if s.assertion_strength_score == 0.0
            ),
            flaky_test_count=sum(1 for s in trust_scores if s.flakiness_rate > 0.1),
            avg_flakiness_rate=(
                sum(s.flakiness_rate for s in trust_scores) / n if n > 0 else 0.0
            ),
            computed_at=datetime.now(UTC).isoformat(),
        )

    async def execute(self, state: Any = None) -> AgentResult:
        """BaseAgent execute interface — runs trust score job."""
        # This agent is primarily used via specific methods, not the generic execute
        return AgentResult(
            success=True,
            data={"message": "Use compute_trust_score() or run_trust_score_job() directly"},
        )


# =============================================================================
# Standalone functions (avoid BaseAgent instantiation overhead)
# =============================================================================


def compute_trust_score_static(
    test_id: str,
    assertion_strength: str = "no_assertion",
    total_heals: int = 0,
    suspicious_heals: int = 0,
    regression_masking_heals: int = 0,
    flakiness_rate: float = 0.0,
    days_since_creation: int = 0,
    coverage_factor: float = 0.5,
) -> dict:
    """Compute trust score without instantiating the agent class."""
    a_score = ASSERTION_STRENGTH_MAP.get(assertion_strength, 0.0)

    heal_cost = (
        (total_heals - suspicious_heals - regression_masking_heals) * 0.08
        + suspicious_heals * 0.12
        + regression_masking_heals * 0.20
    )
    h_score = max(0.0, 1.0 - heal_cost)
    f_score = max(0.0, 1.0 - flakiness_rate)
    age_score = min(1.0, days_since_creation / 90.0) if days_since_creation >= 0 else 0.0
    c_score = max(0.0, min(1.0, coverage_factor))

    raw_score = (
        a_score * WEIGHT_ASSERTION_STRENGTH
        + h_score * WEIGHT_HEAL_PENALTY
        + f_score * WEIGHT_FLAKY_PENALTY
        + age_score * WEIGHT_AGE_FACTOR
        + c_score * WEIGHT_COVERAGE_FACTOR
    )
    final_score = round(raw_score * 100, 1)

    return {
        "score": final_score,
        "components": {
            "assertion_strength": a_score,
            "heal_penalty": h_score,
            "flaky_penalty": f_score,
            "age_factor": age_score,
            "coverage_factor": c_score,
        },
        "context": {
            "total_heals": total_heals,
            "suspicious_heals": suspicious_heals,
            "regression_masking_heals": regression_masking_heals,
            "flakiness_rate": flakiness_rate,
            "days_since_creation": days_since_creation,
        },
    }


async def run_trust_score_job_static(org_id: str, project_id: str) -> list[dict]:
    """Run trust score computation without instantiating agent class."""
    from src.services.supabase_client import get_supabase_client

    supabase = get_supabase_client()

    # Fetch all tests for this project
    test_result = await supabase.select(
        "tests", columns="id,created_at", filters={"project_id": f"eq.{project_id}"}
    )
    tests = test_result.get("data", [])
    if not tests:
        return []

    # Fetch heal history in bulk
    heal_result = await supabase.request(
        f"/heal_validations?select=test_id,heal_quality"
        f"&project_id=eq.{project_id}"
    )
    heals = heal_result.get("data", [])

    # Group heals by test_id
    heal_map: dict[str, list[str]] = {}
    for h in heals:
        tid = h.get("test_id", "")
        heal_map.setdefault(tid, []).append(h.get("heal_quality", ""))

    now = datetime.now(UTC)
    scores = []
    scores_blob = {}

    for test in tests:
        tid = test.get("id", "")
        created_at = test.get("created_at", "")
        days = 0
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                days = (now - created).days
            except (ValueError, TypeError):
                days = 0

        test_heals = heal_map.get(tid, [])
        total = len(test_heals)
        suspicious = sum(1 for q in test_heals if q == "suspicious_heal")
        masking = sum(1 for q in test_heals if q == "regression_masking")

        score = compute_trust_score_static(
            test_id=tid,
            total_heals=total,
            suspicious_heals=suspicious,
            regression_masking_heals=masking,
            days_since_creation=days,
        )
        scores.append(score)
        scores_blob[tid] = score

    # Store in intelligence_precomputed (pass dict directly — JSONB handled by PostgREST)
    await supabase.rpc("upsert_precomputed", {
        "p_org_id": org_id,
        "p_project_id": project_id,
        "p_type": "test_trust_scores",
        "p_result": scores_blob,
        "p_valid_hours": 25,
    })

    return scores
