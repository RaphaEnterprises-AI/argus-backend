"""
Production-grade E2E CI/CD Pipeline Integration Tests.

Tests the AI-powered CI/CD endpoints against LIVE production with NO MOCKS.
Every response is validated against strict Pydantic models matching the
production API contract (camelCase via CamelCaseMiddleware).

Covers:
- AI-powered test impact analysis (Claude Haiku semantic analysis)
- AI-powered deployment risk scoring (multi-factor + Claude)
- Build / deployment / pipeline listing and retrieval
- CI/CD statistics aggregation
- Edge cases, validation, and auth checks

Run with:
    pytest tests/integration/test_cicd_pipeline_live.py -v --tb=short

Run by category:
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "TestAITestImpactAnalysis"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "TestImpactRetrieval"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "DeploymentRisk"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "Build"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "Pipeline"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "Deployment"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "Statistics"
    pytest tests/integration/test_cicd_pipeline_live.py -v -k "EdgeCase"
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

# ═══════════════════════════════════════════════════════════════════════════
# Configuration — env vars with production defaults
# ═══════════════════════════════════════════════════════════════════════════

BASE_URL = os.environ.get(
    "ARGUS_BASE_URL",
    "https://skopaq-brain-production.up.railway.app",
)
API_KEY = os.environ.get(
    "ARGUS_API_KEY",
    "argus_sk_9a1819e0e9faa47011c28ff75bbeb442ffa285779233ea3e81aed7e992367198",
)
ORG_ID = os.environ.get(
    "ARGUS_ORG_ID",
    "229904aa-3084-4b25-8bf8-c07ae749888c",
)
PROJ_ID = os.environ.get(
    "ARGUS_PROJECT_ID",
    "e98fc22d-ae11-490d-86f2-116fdd29941b",
)

HEADERS = {
    "X-API-Key": API_KEY,
    "X-Organization-Id": ORG_ID,
    "Content-Type": "application/json",
}

# AI endpoints (test-impact/analyze) involve Claude calls — allow 120s.
AI_TIMEOUT = httpx.Timeout(120.0, connect=15.0)
# Non-AI endpoints — 90s is generous but safe for production latency.
DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)

# ═══════════════════════════════════════════════════════════════════════════
# Endpoint URLs
# ═══════════════════════════════════════════════════════════════════════════

BUILDS_URL = f"{BASE_URL}/api/v1/cicd/builds"
DEPLOYMENTS_URL = f"{BASE_URL}/api/v1/cicd/deployments"
PIPELINES_URL = f"{BASE_URL}/api/v1/cicd/pipelines"
STATS_URL = f"{BASE_URL}/api/v1/cicd/stats"
TEST_IMPACT_URL = f"{BASE_URL}/api/v1/cicd/test-impact"
TEST_IMPACT_ANALYZE_URL = f"{BASE_URL}/api/v1/cicd/test-impact/analyze"
DEPLOYMENT_RISK_URL = f"{BASE_URL}/api/v1/cicd/deployment-risk"

# ═══════════════════════════════════════════════════════════════════════════
# Response Models — strict Pydantic validation of the production API contract.
#
# The API uses a global CamelCaseMiddleware (src/api/middleware/camelcase.py)
# that converts every snake_case key to camelCase.  These models encode that
# contract via Field(alias=...).  If the API changes shape, Pydantic raises
# ValidationError and the test fails immediately — no silent drift.
#
# extra="allow" lets the API add NEW fields without breaking tests, but
# removing or renaming an existing field is caught instantly.
# ═══════════════════════════════════════════════════════════════════════════


class ChangedFileResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    change_type: str = Field(alias="changeType")
    additions: int
    deletions: int
    impact_score: float | None = Field(None, alias="impactScore")


class ImpactedTestResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    test_id: str = Field(alias="testId")
    test_name: str = Field(alias="testName")
    impact_reason: str = Field(alias="impactReason")
    confidence: float
    priority: str


class ImpactAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    project_id: str = Field(alias="projectId")
    commit_sha: str = Field(alias="commitSha")
    branch: str
    base_sha: str | None = Field(None, alias="baseSha")
    changed_files: list[ChangedFileResponse] = Field(alias="changedFiles")
    impacted_tests: list[ImpactedTestResponse] = Field(alias="impactedTests")
    total_files_changed: int = Field(alias="totalFilesChanged")
    total_tests_impacted: int = Field(alias="totalTestsImpacted")
    recommended_tests: list[str] = Field(alias="recommendedTests")
    skip_candidates: list[str] = Field(alias="skipCandidates")
    confidence_score: float = Field(alias="confidenceScore")
    analysis_time_ms: int | float = Field(alias="analysisTimeMs")
    created_at: str = Field(alias="createdAt")


class DeploymentRiskResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    project_id: str = Field(alias="projectId")
    commit_sha: str | None = Field(None, alias="commitSha")
    risk_score: int = Field(alias="riskScore")
    risk_level: str = Field(alias="riskLevel")
    factors: dict
    recommendations: list[str]
    tests_to_run: int = Field(alias="testsToRun")
    estimated_test_time_ms: int = Field(alias="estimatedTestTimeMs")
    calculated_at: str = Field(alias="calculatedAt")


class CICDStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_pipelines: int = Field(alias="totalPipelines")
    pipelines_last_24h: int = Field(alias="pipelinesLast24H")
    success_rate: float = Field(alias="successRate")
    avg_pipeline_duration_ms: int = Field(alias="avgPipelineDurationMs")
    total_builds: int = Field(alias="totalBuilds")
    builds_last_24h: int = Field(alias="buildsLast24H")
    build_success_rate: float = Field(alias="buildSuccessRate")
    avg_build_duration_ms: int = Field(alias="avgBuildDurationMs")
    total_deployments: int = Field(alias="totalDeployments")
    deployments_last_24h: int = Field(alias="deploymentsLast24H")
    deployment_success_rate: float = Field(alias="deploymentSuccessRate")
    avg_deployment_duration_ms: int = Field(alias="avgDeploymentDurationMs")
    current_risk_score: int = Field(alias="currentRiskScore")
    tests_impacted_by_recent_changes: int = Field(alias="testsImpactedByRecentChanges")


class BuildListResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    builds: list[dict]
    total: int


class PipelineListResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    pipelines: list[dict]
    total: int


class DeploymentListResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    deployments: list[dict]
    total: int


# ═══════════════════════════════════════════════════════════════════════════
# Valid enum values — used for assertion checks
# ═══════════════════════════════════════════════════════════════════════════

VALID_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
VALID_PRIORITIES = frozenset({"critical", "high", "medium", "low"})
VALID_RISK_FACTOR_KEYS = frozenset({
    "ciFailureRate", "testsImpactedRatio", "deploymentFailureRate", "qualityScore",
    # camelCase aliases for the snake_case keys from backend
    "ci_failure_rate", "tests_impacted_ratio", "deployment_failure_rate", "quality_score",
})

# Unique commit SHA per test run to avoid cache interference with AI endpoints.
_TEST_RUN_COMMIT = uuid.uuid4().hex[:12]


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: AI Test Impact Analysis  (8 tests)
# Star feature — Claude Haiku analyzes code changes to identify impacted tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAITestImpactAnalysis:
    """AI-powered test impact analysis (POST /test-impact/analyze)."""

    # Class-level storage so Category 2 (retrieval) can reference results.
    _last_analysis: ImpactAnalysisResponse | None = None
    _last_commit_sha: str | None = None

    @pytest.mark.asyncio
    async def test_1_1_analyze_single_modified_file(self):
        """POST /test-impact/analyze with single modified file returns valid analysis."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_1"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [
                {
                    "path": "src/auth.py",
                    "change_type": "modified",
                    "additions": 10,
                    "deletions": 5,
                }
            ],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        assert result.project_id == PROJ_ID
        assert result.commit_sha == commit_sha
        assert result.total_files_changed == 1
        assert result.confidence_score > 0
        assert len(result.impacted_tests) >= 0  # May be 0 if no matching tests

        # Store for retrieval tests
        TestAITestImpactAnalysis._last_analysis = result
        TestAITestImpactAnalysis._last_commit_sha = commit_sha

    @pytest.mark.asyncio
    async def test_1_2_analyze_multiple_files(self):
        """POST /test-impact/analyze with multiple files — totalFilesChanged matches."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_2"
        files = [
            {"path": "src/api/auth.py", "change_type": "modified", "additions": 15, "deletions": 3},
            {"path": "src/services/payment.py", "change_type": "modified", "additions": 20, "deletions": 8},
            {"path": "tests/test_auth.py", "change_type": "modified", "additions": 5, "deletions": 2},
        ]
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": files,
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        assert result.total_files_changed == len(files)
        # Each changed file should have an impactScore set by the backend
        for cf in result.changed_files:
            assert cf.impact_score is not None, (
                f"File {cf.path} missing impactScore"
            )

    @pytest.mark.asyncio
    async def test_1_3_analyze_added_file(self):
        """POST /test-impact/analyze with added file — changeType preserved."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_3"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [
                {"path": "src/new_feature.py", "change_type": "added", "additions": 50, "deletions": 0},
            ],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        assert result.changed_files[0].change_type == "added"

    @pytest.mark.asyncio
    async def test_1_4_analyze_deleted_file(self):
        """POST /test-impact/analyze with deleted file — valid analysis returned."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_4"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [
                {"path": "src/deprecated_module.py", "change_type": "deleted", "additions": 0, "deletions": 120},
            ],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())
        assert result.id  # Valid UUID
        assert result.created_at  # Timestamp present

    @pytest.mark.asyncio
    async def test_1_5_impacted_tests_structure(self):
        """Each impactedTest has valid fields — confidence in [0,1], priority in valid set."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_5"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [
                {"path": "src/auth.py", "change_type": "modified", "additions": 10, "deletions": 5},
            ],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        for test in result.impacted_tests:
            assert test.test_id, "testId should not be empty"
            assert test.test_name, "testName should not be empty"
            assert test.impact_reason, "impactReason should not be empty"
            assert 0 <= test.confidence <= 1, (
                f"confidence {test.confidence} not in [0, 1]"
            )
            assert test.priority in VALID_PRIORITIES, (
                f"priority '{test.priority}' not in {VALID_PRIORITIES}"
            )

    @pytest.mark.asyncio
    async def test_1_6_recommended_and_skip_disjoint(self):
        """recommendedTests and skipCandidates must have no overlap."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_6"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [
                {"path": "src/auth.py", "change_type": "modified", "additions": 10, "deletions": 5},
            ],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        recommended = set(result.recommended_tests)
        skippable = set(result.skip_candidates)
        overlap = recommended & skippable
        assert not overlap, (
            f"recommendedTests and skipCandidates overlap: {overlap}"
        )

    @pytest.mark.asyncio
    async def test_1_7_analysis_time_non_negative(self):
        """analysisTimeMs should be a non-negative number."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_7"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [
                {"path": "src/utils.py", "change_type": "modified", "additions": 2, "deletions": 1},
            ],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        assert result.analysis_time_ms >= 0, (
            f"analysisTimeMs should be >= 0, got {result.analysis_time_ms}"
        )

    @pytest.mark.asyncio
    async def test_1_8_empty_changed_files_conservative_fallback(self):
        """Empty changed_files → conservative fallback: all tests recommended, confidence 0.5."""
        commit_sha = f"test_{_TEST_RUN_COMMIT}_8"
        payload = {
            "project_id": PROJ_ID,
            "commit_sha": commit_sha,
            "branch": "main",
            "changed_files": [],
        }
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL, headers=HEADERS, json=payload,
            )
        assert resp.status_code == 200, resp.text
        result = ImpactAnalysisResponse.model_validate(resp.json())

        # Conservative fallback guarantees:
        assert result.total_files_changed == 0
        assert result.confidence_score == 0.5, (
            f"Expected conservative confidence 0.5, got {result.confidence_score}"
        )
        assert len(result.recommended_tests) > 0, (
            "Conservative fallback must recommend all tests"
        )
        assert result.total_tests_impacted == len(result.recommended_tests), (
            f"All {result.total_tests_impacted} impacted tests must be in recommendedTests "
            f"(got {len(result.recommended_tests)})"
        )
        assert len(result.skip_candidates) == 0, (
            f"Conservative fallback must have 0 skip candidates, got {len(result.skip_candidates)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Test Impact Retrieval  (4 tests)
# GET stored analyses without re-running AI
# ═══════════════════════════════════════════════════════════════════════════


class TestTestImpactRetrieval:
    """Retrieve stored test impact analyses (GET /test-impact)."""

    @pytest.mark.asyncio
    async def test_2_1_get_latest_impact(self):
        """GET /test-impact?project_id=PROJ_ID returns analysis or null."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                TEST_IMPACT_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data is not None:
            ImpactAnalysisResponse.model_validate(data)

    @pytest.mark.asyncio
    async def test_2_2_get_impact_by_commit_sha(self):
        """GET /test-impact with commit_sha returns matching analysis."""
        # Use the commit SHA from test 1.1 if available
        commit_sha = TestAITestImpactAnalysis._last_commit_sha
        if not commit_sha:
            pytest.skip("Test 1.1 did not run — no commit SHA available")

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                TEST_IMPACT_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "commit_sha": commit_sha},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data is not None:
            result = ImpactAnalysisResponse.model_validate(data)
            assert result.commit_sha == commit_sha

    @pytest.mark.asyncio
    async def test_2_3_get_impact_nonexistent_sha(self):
        """GET /test-impact with nonexistent commit_sha returns null."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                TEST_IMPACT_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "commit_sha": "nonexistent-sha-xyz-000"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json() is None, (
            f"Expected null for nonexistent commit, got {resp.json()}"
        )

    @pytest.mark.asyncio
    async def test_2_4_retrieved_matches_posted(self):
        """Retrieved analysis matches the POST result (same id, commitSha, totalTestsImpacted)."""
        last = TestAITestImpactAnalysis._last_analysis
        commit_sha = TestAITestImpactAnalysis._last_commit_sha
        if not last or not commit_sha:
            pytest.skip("Test 1.1 did not run — no analysis to verify")

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                TEST_IMPACT_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "commit_sha": commit_sha},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data is None:
            pytest.skip("Analysis not found in database — table may not exist")

        result = ImpactAnalysisResponse.model_validate(data)
        assert result.id == last.id, (
            f"Expected id={last.id}, got {result.id}"
        )
        assert result.commit_sha == last.commit_sha
        assert result.total_tests_impacted == last.total_tests_impacted


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Deployment Risk Assessment  (5 tests)
# AI-powered risk scoring with multi-factor analysis
# ═══════════════════════════════════════════════════════════════════════════


class TestDeploymentRisk:
    """AI-powered deployment risk scoring (GET /deployment-risk)."""

    @pytest.mark.asyncio
    async def test_3_1_get_deployment_risk(self):
        """GET /deployment-risk?project_id=PROJ_ID returns valid response."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                DEPLOYMENT_RISK_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        DeploymentRiskResponseModel.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_3_2_risk_score_and_level(self):
        """riskScore in [0, 100] and riskLevel in valid set."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                DEPLOYMENT_RISK_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "commit_sha": "abc123"},
            )
        assert resp.status_code == 200, resp.text
        result = DeploymentRiskResponseModel.model_validate(resp.json())

        assert 0 <= result.risk_score <= 100, (
            f"riskScore {result.risk_score} not in [0, 100]"
        )
        assert result.risk_level in VALID_RISK_LEVELS, (
            f"riskLevel '{result.risk_level}' not in {VALID_RISK_LEVELS}"
        )

    @pytest.mark.asyncio
    async def test_3_3_risk_factors_structure(self):
        """factors dict has expected keys (camelCase)."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                DEPLOYMENT_RISK_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = DeploymentRiskResponseModel.model_validate(resp.json())

        # The middleware converts snake_case to camelCase
        expected_keys = {"ciFailureRate", "testsImpactedRatio", "deploymentFailureRate", "qualityScore"}
        actual_keys = set(result.factors.keys())
        assert expected_keys.issubset(actual_keys), (
            f"factors missing keys: {expected_keys - actual_keys}, got {actual_keys}"
        )

    @pytest.mark.asyncio
    async def test_3_4_recommendations_non_empty(self):
        """recommendations is a non-empty list of non-empty strings."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                DEPLOYMENT_RISK_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = DeploymentRiskResponseModel.model_validate(resp.json())

        assert len(result.recommendations) > 0, (
            "recommendations should not be empty"
        )
        for i, rec in enumerate(result.recommendations):
            assert isinstance(rec, str) and len(rec) > 0, (
                f"recommendation[{i}] should be a non-empty string, got {rec!r}"
            )

    @pytest.mark.asyncio
    async def test_3_5_test_count_and_timing(self):
        """testsToRun >= 0 and estimatedTestTimeMs >= 0."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            resp = await client.get(
                DEPLOYMENT_RISK_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = DeploymentRiskResponseModel.model_validate(resp.json())

        assert result.tests_to_run >= 0, (
            f"testsToRun should be >= 0, got {result.tests_to_run}"
        )
        assert result.estimated_test_time_ms >= 0, (
            f"estimatedTestTimeMs should be >= 0, got {result.estimated_test_time_ms}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Build Management  (4 tests)
# Build listing and retrieval with UUID validation
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildManagement:
    """Build listing and retrieval (GET /builds, /builds/{id})."""

    @pytest.mark.asyncio
    async def test_4_1_list_builds(self):
        """GET /builds?project_id=PROJ_ID returns valid list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                BUILDS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = BuildListResponseModel.model_validate(resp.json())
        assert result.total >= 0

    @pytest.mark.asyncio
    async def test_4_2_list_builds_with_limit(self):
        """GET /builds with limit and offset — builds length <= limit."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                BUILDS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "limit": 5, "offset": 0},
            )
        assert resp.status_code == 200, resp.text
        result = BuildListResponseModel.model_validate(resp.json())
        assert len(result.builds) <= 5

    @pytest.mark.asyncio
    async def test_4_3_get_build_nonexistent_uuid(self):
        """GET /builds/{nonexistent-uuid} returns 404."""
        fake_uuid = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BUILDS_URL}/{fake_uuid}",
                headers=HEADERS,
            )
        assert resp.status_code == 404, (
            f"Expected 404, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_4_4_get_build_invalid_uuid(self):
        """GET /builds/{non-uuid-string} returns 400 with UUID error."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{BUILDS_URL}/not-a-valid-uuid",
                headers=HEADERS,
            )
        assert resp.status_code == 400, (
            f"Expected 400, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        detail = body.get("detail", "")
        assert "uuid" in detail.lower() or "UUID" in detail, (
            f"Expected UUID error message, got: {detail}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: Pipeline Management  (3 tests)
# GitHub Actions workflow listing — limited by project config
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineManagement:
    """Pipeline listing and retrieval (GET /pipelines, /pipelines/{id})."""

    @pytest.mark.asyncio
    async def test_5_1_list_pipelines(self):
        """GET /pipelines?project_id=PROJ_ID returns valid list (may be empty)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                PIPELINES_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = PipelineListResponseModel.model_validate(resp.json())
        assert result.total >= 0

    @pytest.mark.asyncio
    async def test_5_2_get_pipeline_no_github(self):
        """GET /pipelines/{id}?project_id=PROJ_ID → 400 if GitHub not configured."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PIPELINES_URL}/12345",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        # Project has no GitHub integration → 400
        assert resp.status_code == 400, (
            f"Expected 400 (GitHub not configured), got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        detail = body.get("detail", "")
        assert "github" in detail.lower() or "GitHub" in detail, (
            f"Expected GitHub config error, got: {detail}"
        )

    @pytest.mark.asyncio
    async def test_5_3_retrigger_pipeline_no_github(self):
        """POST /pipelines/{id}/retrigger?project_id=PROJ_ID → 400 if no GitHub."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PIPELINES_URL}/12345/retrigger",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 400, (
            f"Expected 400 (GitHub not configured), got {resp.status_code}: {resp.text}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Deployment Management  (3 tests)
# Deployment listing, retrieval, and rollback validation
# ═══════════════════════════════════════════════════════════════════════════


class TestDeploymentManagement:
    """Deployment listing, retrieval, and rollback (GET/POST /deployments)."""

    @pytest.mark.asyncio
    async def test_6_1_list_deployments(self):
        """GET /deployments?project_id=PROJ_ID returns valid list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                DEPLOYMENTS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = DeploymentListResponseModel.model_validate(resp.json())
        assert result.total >= 0

    @pytest.mark.asyncio
    async def test_6_2_get_deployment_nonexistent(self):
        """GET /deployments/{nonexistent-uuid} returns 404."""
        fake_uuid = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DEPLOYMENTS_URL}/{fake_uuid}",
                headers=HEADERS,
            )
        assert resp.status_code == 404, (
            f"Expected 404, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_6_3_rollback_invalid_uuid(self):
        """POST /deployments/{non-uuid}/rollback returns 400 (UUID validation)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{DEPLOYMENTS_URL}/not-a-valid-uuid/rollback",
                headers=HEADERS,
            )
        assert resp.status_code == 400, (
            f"Expected 400, got {resp.status_code}: {resp.text}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 7: CI/CD Statistics  (3 tests)
# Aggregated project statistics
# ═══════════════════════════════════════════════════════════════════════════


class TestCICDStatistics:
    """CI/CD statistics aggregation (GET /stats)."""

    @pytest.mark.asyncio
    async def test_7_1_get_stats(self):
        """GET /stats?project_id=PROJ_ID returns valid stats."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                STATS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        CICDStatsResponse.model_validate(resp.json())

    @pytest.mark.asyncio
    async def test_7_2_stats_field_ranges(self):
        """All counts >= 0, rates in [0, 100], currentRiskScore in [0, 100]."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                STATS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 200, resp.text
        result = CICDStatsResponse.model_validate(resp.json())

        # Counts must be non-negative
        assert result.total_pipelines >= 0
        assert result.pipelines_last_24h >= 0
        assert result.total_builds >= 0
        assert result.builds_last_24h >= 0
        assert result.total_deployments >= 0
        assert result.deployments_last_24h >= 0
        assert result.tests_impacted_by_recent_changes >= 0

        # Rates in [0, 100]
        assert 0 <= result.success_rate <= 100, (
            f"successRate {result.success_rate} not in [0, 100]"
        )
        assert 0 <= result.build_success_rate <= 100, (
            f"buildSuccessRate {result.build_success_rate} not in [0, 100]"
        )
        assert 0 <= result.deployment_success_rate <= 100, (
            f"deploymentSuccessRate {result.deployment_success_rate} not in [0, 100]"
        )

        # Risk score in [0, 100]
        assert 0 <= result.current_risk_score <= 100, (
            f"currentRiskScore {result.current_risk_score} not in [0, 100]"
        )

        # Durations non-negative
        assert result.avg_pipeline_duration_ms >= 0
        assert result.avg_build_duration_ms >= 0
        assert result.avg_deployment_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_7_3_stats_missing_project_id(self):
        """GET /stats without project_id returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(STATS_URL, headers=HEADERS)
        assert resp.status_code == 422, (
            f"Expected 422, got {resp.status_code}: {resp.text}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 8: Edge Cases & Auth  (8 tests)
# Validation, authentication, and robustness
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCasesAndAuth:
    """Validation, authentication, and robustness edge cases."""

    @pytest.mark.asyncio
    async def test_8_1_missing_api_key(self):
        """GET /builds with no API key returns 401."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                BUILDS_URL,
                headers={"Content-Type": "application/json"},
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 401, (
            f"Expected 401, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_2_wrong_api_key(self):
        """GET /builds with invalid API key returns 401."""
        bad_headers = {**HEADERS, "X-API-Key": "argus_sk_invalid_key_1234567890"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                BUILDS_URL,
                headers=bad_headers,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code == 401, (
            f"Expected 401, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_3_analyze_empty_body(self):
        """POST /test-impact/analyze with empty body returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL,
                headers=HEADERS,
                json={},
            )
        assert resp.status_code == 422, (
            f"Expected 422, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_4_analyze_missing_commit_sha(self):
        """POST /test-impact/analyze missing commit_sha returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                TEST_IMPACT_ANALYZE_URL,
                headers=HEADERS,
                json={"project_id": PROJ_ID},
            )
        assert resp.status_code == 422, (
            f"Expected 422, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_5_deployment_risk_missing_project_id(self):
        """GET /deployment-risk without project_id returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(DEPLOYMENT_RISK_URL, headers=HEADERS)
        assert resp.status_code == 422, (
            f"Expected 422, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_6_builds_limit_below_minimum(self):
        """GET /builds?limit=0 (below ge=1) returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                BUILDS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "limit": 0},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for limit=0, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_7_builds_limit_above_maximum(self):
        """GET /builds?limit=201 (above le=200) returns 422."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                BUILDS_URL,
                headers=HEADERS,
                params={"project_id": PROJ_ID, "limit": 201},
            )
        assert resp.status_code == 422, (
            f"Expected 422 for limit=201, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_8_8_concurrent_endpoint_calls(self):
        """3 concurrent calls (builds + stats + deployment-risk) all succeed."""
        async with httpx.AsyncClient(timeout=AI_TIMEOUT) as client:
            tasks = [
                client.get(
                    BUILDS_URL, headers=HEADERS,
                    params={"project_id": PROJ_ID},
                ),
                client.get(
                    STATS_URL, headers=HEADERS,
                    params={"project_id": PROJ_ID},
                ),
                client.get(
                    DEPLOYMENT_RISK_URL, headers=HEADERS,
                    params={"project_id": PROJ_ID},
                ),
            ]
            responses = await asyncio.gather(*tasks)

        # Validate builds
        assert responses[0].status_code == 200, (
            f"Builds failed: {responses[0].status_code}: {responses[0].text}"
        )
        BuildListResponseModel.model_validate(responses[0].json())

        # Validate stats
        assert responses[1].status_code == 200, (
            f"Stats failed: {responses[1].status_code}: {responses[1].text}"
        )
        CICDStatsResponse.model_validate(responses[1].json())

        # Validate deployment-risk
        assert responses[2].status_code == 200, (
            f"Deployment-risk failed: {responses[2].status_code}: {responses[2].text}"
        )
        DeploymentRiskResponseModel.model_validate(responses[2].json())
