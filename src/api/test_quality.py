"""Test Quality Intelligence API endpoints.

Provides endpoints for:
- Trust scores (precomputed, per-test, on-demand)
- Heal validation records (filterable by quality)
- Suite quality audit (precomputed, daily)
- Trust score refresh (admin trigger)
"""

from __future__ import annotations

import json
import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.middleware.tenant import validate_uuid
from src.api.teams import get_current_user, verify_org_access
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/test-quality", tags=["Test Quality Intelligence"])


# =============================================================================
# Response Models
# =============================================================================


class TrustScoreResponse(BaseModel):
    test_id: str
    score: float
    components: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


class TrustScoresListResponse(BaseModel):
    scores: list[TrustScoreResponse]
    total: int
    project_id: str


class HealValidationResponse(BaseModel):
    id: str
    test_id: str
    heal_quality: str
    blocking: bool
    healing_method: str | None = None
    original_selector: str | None = None
    healed_selector: str | None = None
    addresses_root_cause: bool | None = None
    preserves_test_intent: bool | None = None
    has_side_effects: bool | None = None
    judge_confidence: float | None = None
    assertion_weakening_detected: bool = False
    explanation: str | None = None
    created_at: str | None = None


class HealValidationsListResponse(BaseModel):
    validations: list[HealValidationResponse]
    total: int


class SuiteAuditResponse(BaseModel):
    project_id: str
    total_tests: int
    avg_trust_score: float
    median_trust_score: float
    distribution: dict = Field(default_factory=dict)
    heal_metrics: dict = Field(default_factory=dict)
    assertion_strength_distribution: dict = Field(default_factory=dict)
    flakiness: dict = Field(default_factory=dict)
    computed_at: str = ""


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/{org_id}/{project_id}/trust-scores")
async def get_trust_scores(
    request: Request,
    org_id: str,
    project_id: str,
    min_score: float = Query(0.0, ge=0.0, le=100.0, description="Minimum trust score filter"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get precomputed trust scores for a project's tests."""
    import traceback as tb
    try:
        validate_uuid(org_id, "org_id")
        validate_uuid(project_id, "project_id")
        user = await get_current_user(request)
        await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

        supabase = get_supabase_client()

        result = await supabase.request(
            f"/intelligence_precomputed?select=result,computed_at"
            f"&org_id=eq.{org_id}"
            f"&project_id=eq.{project_id}"
            f"&computation_type=eq.test_trust_scores"
            f"&limit=1"
        )
        rows = result.get("data", [])

        if not rows:
            return TrustScoresListResponse(scores=[], total=0, project_id=project_id)

        raw_result = rows[0].get("result", {})
        scores_blob = json.loads(raw_result) if isinstance(raw_result, str) else raw_result

        scores = []
        for test_id, score_data in scores_blob.items():
            score_val = score_data.get("score", 0.0) if isinstance(score_data, dict) else 0.0
            if score_val >= min_score:
                scores.append(TrustScoreResponse(
                    test_id=test_id,
                    score=score_val,
                    components=score_data.get("components", {}),
                    context=score_data.get("context", {}),
                ))

        scores.sort(key=lambda s: s.score, reverse=True)
        total = len(scores)
        scores = scores[offset : offset + limit]

        return TrustScoresListResponse(
            scores=scores,
            total=total,
            project_id=project_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("trust_scores_error", error=str(e), traceback=tb.format_exc())
        return {"error": str(e), "traceback": tb.format_exc()}


@router.get("/{org_id}/{project_id}/trust-scores/{test_id}")
async def get_trust_score(
    request: Request,
    org_id: str,
    project_id: str,
    test_id: str,
):
    """Get trust score for a single test (on-demand computation if not precomputed)."""
    import traceback as tb
    try:
        validate_uuid(org_id, "org_id")
        validate_uuid(project_id, "project_id")
        user = await get_current_user(request)
        await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

        supabase = get_supabase_client()

        # Try precomputed blob first
        result = await supabase.request(
            f"/intelligence_precomputed?select=result"
            f"&org_id=eq.{org_id}"
            f"&project_id=eq.{project_id}"
            f"&computation_type=eq.test_trust_scores"
            f"&limit=1"
        )
        rows = result.get("data", [])

        if rows:
            raw_result = rows[0].get("result", {})
            scores_blob = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            score_data = scores_blob.get(test_id)
            if score_data and isinstance(score_data, dict):
                return TrustScoreResponse(
                    test_id=test_id,
                    score=score_data.get("score", 0.0),
                    components=score_data.get("components", {}),
                    context=score_data.get("context", {}),
                )

        # On-demand computation: query heal_validations for this test
        try:
            heal_result = await supabase.request(
                f"/heal_validations?select=heal_quality"
                f"&project_id=eq.{project_id}"
                f"&test_id=eq.{test_id}"
            )
            heals = heal_result.get("data", [])
        except Exception:
            heals = []

        total_heals = len(heals)
        suspicious = sum(1 for h in heals if h.get("heal_quality") == "suspicious_heal")
        masking = sum(1 for h in heals if h.get("heal_quality") == "regression_masking")

        # Compute trust score using static method (no agent instantiation needed)
        from src.agents.test_quality_agent import compute_trust_score_static

        score = compute_trust_score_static(
            test_id=test_id,
            total_heals=total_heals,
            suspicious_heals=suspicious,
            regression_masking_heals=masking,
        )

        return TrustScoreResponse(
            test_id=test_id,
            score=score["score"],
            components=score.get("components", {}),
            context=score.get("context", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("trust_score_single_error", error=str(e), traceback=tb.format_exc())
        return {"error": str(e), "traceback": tb.format_exc()}


@router.get("/{org_id}/{project_id}/suite-audit")
async def get_suite_audit(
    request: Request,
    org_id: str,
    project_id: str,
):
    """Get suite-level quality audit (precomputed daily)."""
    import traceback as tb
    try:
        validate_uuid(org_id, "org_id")
        validate_uuid(project_id, "project_id")
        user = await get_current_user(request)
        await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

        supabase = get_supabase_client()

        # Try precomputed
        result = await supabase.request(
            f"/intelligence_precomputed?select=result"
            f"&org_id=eq.{org_id}"
            f"&project_id=eq.{project_id}"
            f"&computation_type=eq.suite_audit"
            f"&limit=1"
        )
        rows = result.get("data", [])

        if rows:
            raw_value = rows[0].get("result", {})
            value = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            return SuiteAuditResponse(**value)

        # Compute on-demand if not precomputed
        # No precomputed data — return empty audit
        return SuiteAuditResponse(
            project_id=project_id,
            total_tests=0,
            avg_trust_score=0.0,
            median_trust_score=0.0,
            computed_at="",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("suite_audit_error", error=str(e), traceback=tb.format_exc())
        return {"error": str(e), "traceback": tb.format_exc()}


@router.get("/{org_id}/{project_id}/heal-validations")
async def get_heal_validations(
    request: Request,
    org_id: str,
    project_id: str,
    quality: str | None = Query(None, description="Filter by heal_quality"),
    test_id: str | None = Query(None, description="Filter by test_id"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> HealValidationsListResponse:
    """Get recent heal validation records, filterable by quality and test."""
    validate_uuid(org_id, "org_id")
    validate_uuid(project_id, "project_id")
    user = await get_current_user(request)
    await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

    supabase = get_supabase_client()

    # Build PostgREST query path for ordering + pagination
    path = (
        f"/heal_validations?select=*"
        f"&organization_id=eq.{org_id}"
        f"&project_id=eq.{project_id}"
        f"&order=created_at.desc"
        f"&limit={limit}"
        f"&offset={offset}"
    )

    if quality:
        valid_qualities = {"valid_heal", "suspicious_heal", "regression_masking", "needs_human_review"}
        if quality not in valid_qualities:
            raise HTTPException(400, f"Invalid quality filter. Must be one of: {valid_qualities}")
        path += f"&heal_quality=eq.{quality}"
    if test_id:
        path += f"&test_id=eq.{test_id}"

    result = await supabase.request(path)
    rows = result.get("data", [])

    validations = [
        HealValidationResponse(
            id=str(r.get("id", "")),
            test_id=r.get("test_id", ""),
            heal_quality=r.get("heal_quality", ""),
            blocking=r.get("blocking", False),
            healing_method=r.get("healing_method"),
            original_selector=r.get("original_selector"),
            healed_selector=r.get("healed_selector"),
            addresses_root_cause=r.get("addresses_root_cause"),
            preserves_test_intent=r.get("preserves_test_intent"),
            has_side_effects=r.get("has_side_effects"),
            judge_confidence=r.get("judge_confidence"),
            assertion_weakening_detected=r.get("assertion_weakening_detected", False),
            explanation=r.get("explanation"),
            created_at=r.get("created_at"),
        )
        for r in rows
    ]

    return HealValidationsListResponse(
        validations=validations,
        total=len(validations),
    )


@router.post("/{org_id}/{project_id}/trust-scores/refresh")
async def refresh_trust_scores(
    request: Request,
    org_id: str,
    project_id: str,
) -> dict:
    """Trigger recomputation of trust scores (admin action)."""
    import traceback as tb
    try:
        validate_uuid(org_id, "org_id")
        validate_uuid(project_id, "project_id")
        user = await get_current_user(request)
        await verify_org_access(org_id, user["user_id"], user_email=user.get("email"), request=request)

        from src.agents.test_quality_agent import run_trust_score_job_static

        scores = await run_trust_score_job_static(org_id, project_id)

        return {
            "status": "completed",
            "tests_scored": len(scores),
            "avg_score": round(sum(s["score"] for s in scores) / len(scores), 1) if scores else 0.0,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("refresh_trust_scores_error", error=str(e), traceback=tb.format_exc())
        return {"error": str(e), "traceback": tb.format_exc()}
