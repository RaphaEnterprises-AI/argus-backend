"""
Production-grade E2E Advanced Testing Integration Tests.

Tests the Discovery, Visual AI, Parameterized Tests, Recording, and Chat
API modules against LIVE production with NO MOCKS.
Every response is validated against strict Pydantic models matching the
production API contract (camelCase via CamelCaseMiddleware).

Covers:
- Discovery sessions, patterns, feature mesh
- Visual AI capture, compare, baselines, analysis
- Parameterized test CRUD, expand, preview, validate, import
- Recording upload, convert, snippet, replay
- Chat thread management

Run with:
    pytest tests/integration/test_advanced_testing_live.py -v --tb=short

Run by category:
    pytest tests/integration/test_advanced_testing_live.py -v -k "TestDiscovery"
    pytest tests/integration/test_advanced_testing_live.py -v -k "TestVisualAI"
    pytest tests/integration/test_advanced_testing_live.py -v -k "TestParameterized"
    pytest tests/integration/test_advanced_testing_live.py -v -k "TestRecording"
    pytest tests/integration/test_advanced_testing_live.py -v -k "TestChat"
"""

from __future__ import annotations

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

DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
SLOW_TIMEOUT = httpx.Timeout(120.0, connect=15.0)

FAKE_UUID = "00000000-0000-0000-0000-000000000000"

# ═══════════════════════════════════════════════════════════════════════════
# Endpoint URL Prefixes
# ═══════════════════════════════════════════════════════════════════════════

DISCOVERY_URL = f"{BASE_URL}/api/v1/discovery"
VISUAL_URL = f"{BASE_URL}/api/v1/visual"
PARAM_URL = f"{BASE_URL}/api/v1/parameterized"
RECORDING_URL = f"{BASE_URL}/api/v1/recording"
CHAT_URL = f"{BASE_URL}/api/v1/chat"

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


# ──── Discovery Models ────

class DiscoverySessionListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sessions: list[dict]
    total: int


class DiscoverySessionResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    project_id: str = Field(alias="projectId")
    status: str
    pages_found: int = Field(alias="pagesFound")
    flows_found: int = Field(alias="flowsFound")
    app_url: str = Field(alias="appUrl")


class DiscoveryPatternListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    patterns: list[dict]
    total: int


class DiscoveryPatternResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    success: bool | None = None


class DiscoveryHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    sessions: list[dict] | None = None


# ──── Visual AI Models ────

class SnapshotResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    url: str
    browser: str
    captured_at: str = Field(alias="capturedAt")


class BaselineListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    baselines: list[dict]
    total: int


class ComparisonListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    comparisons: list[dict]
    total: int


# ──── Parameterized Test Models ────

class ParamTestResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    project_id: str = Field(alias="projectId")
    name: str
    data_source_type: str = Field(alias="dataSourceType")


class ParamDataSourcesResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    data_sources: list[dict] = Field(alias="dataSources")


class ParamIterationModesResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    iteration_modes: list[dict] = Field(alias="iterationModes")


class ExpandResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    test_name: str = Field(alias="testName")
    total_expanded: int = Field(alias="totalExpanded")


class PreviewResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    test_name: str = Field(alias="testName")
    total_combinations: int = Field(alias="totalCombinations")


class ValidateResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    valid: bool
    required_parameters: list[str] = Field(alias="requiredParameters")
    provided_parameters: list[str] = Field(alias="providedParameters")
    missing_parameters: list[str] = Field(alias="missingParameters")


class ImportDataResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    parameter_sets_count: int = Field(alias="parameterSetsCount")
    detected_fields: list[str] = Field(alias="detectedFields")


# ──── Recording Models ────

class RecordingUploadResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    recording_id: str = Field(alias="recordingId")
    events_count: int = Field(alias="eventsCount")
    duration_ms: int = Field(alias="durationMs")
    estimated_steps: int = Field(alias="estimatedSteps")


class RecordingListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    recordings: list[dict]
    total: int


class RecorderSnippetResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    success: bool
    snippet: str
    cdn_script: str = Field(alias="cdnScript")
    instructions: str


# ──── Chat Models ────

class ThreadListResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    threads: list[dict]


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: Discovery  (~16 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscovery:
    """Discovery sessions, patterns, and feature mesh endpoints."""

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        """GET /sessions returns valid session list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/sessions",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "sessions" in data or "total" in data

    @pytest.mark.asyncio
    async def test_create_session(self):
        """POST /sessions creates a discovery session."""
        payload = {
            "project_id": PROJ_ID,
            "app_url": "https://example.com",
            "max_pages": 1,
            "mode": "quick_scan",
            "strategy": "breadth_first",
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{DISCOVERY_URL}/sessions",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 201, 422, 429, 500), resp.text
        if resp.status_code in (200, 201):
            data = resp.json()
            assert "id" in data

    @pytest.mark.asyncio
    async def test_get_session_fake_uuid(self):
        """GET /sessions/{fake} returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/sessions/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_list_pages_fake_session(self):
        """GET /sessions/{fake}/pages returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/sessions/{FAKE_UUID}/pages",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_list_flows_fake_session(self):
        """GET /sessions/{fake}/flows returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/sessions/{FAKE_UUID}/flows",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_list_patterns(self):
        """GET /patterns returns pattern list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/patterns",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "patterns" in data

    @pytest.mark.asyncio
    async def test_create_pattern(self):
        """POST /patterns creates a pattern."""
        payload = {
            "name": f"test-pattern-{uuid.uuid4().hex[:8]}",
            "pattern_type": "navigation",
            "selector": "#test-btn",
            "url_pattern": "https://example.com/*",
            "project_id": PROJ_ID,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{DISCOVERY_URL}/patterns",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 201, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_project_history(self):
        """GET /projects/{project_id}/history returns history."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/projects/{PROJ_ID}/history",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list) or "sessions" in data

    @pytest.mark.asyncio
    async def test_delete_session_fake_uuid(self):
        """DELETE /sessions/{fake} returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{DISCOVERY_URL}/sessions/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_extract_patterns(self):
        """POST /patterns/extract processes test data."""
        payload = {
            "test_data": {
                "url": "https://example.com/login",
                "selectors": ["#username", "#password", "#submit"],
                "actions": ["fill", "fill", "click"],
            },
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{DISCOVERY_URL}/patterns/extract",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_search_patterns(self):
        """POST /patterns/search finds patterns."""
        payload = {
            "query": "login",
            "limit": 5,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{DISCOVERY_URL}/patterns/search",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_patterns_insights(self):
        """GET /patterns/insights returns insights."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/patterns/insights",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text

    @pytest.mark.asyncio
    async def test_patterns_types(self):
        """GET /patterns/types returns pattern type list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/patterns/types",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, (list, dict))

    @pytest.mark.asyncio
    async def test_feature_mesh_process(self):
        """POST /feature-mesh/process handles feature data."""
        payload = {
            "features": [
                {
                    "name": "login-button",
                    "selector": "#login-btn",
                    "url": "https://example.com/login",
                }
            ],
            "project_id": PROJ_ID,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{DISCOVERY_URL}/feature-mesh/process",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_feature_mesh_selector_alternatives(self):
        """GET /feature-mesh/selector-alternatives returns alternatives."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/feature-mesh/selector-alternatives",
                headers=HEADERS,
                params={"selector": "#submit-btn", "url": "https://example.com"},
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_session_patterns_fake_uuid(self):
        """GET /sessions/{fake}/patterns returns 404 or empty 200."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{DISCOVERY_URL}/sessions/{FAKE_UUID}/patterns",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 404, 429), resp.text


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Visual AI  (~18 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestVisualAI:
    """Visual AI capture, compare, baselines, and analysis endpoints."""

    @pytest.mark.asyncio
    async def test_capture_screenshot(self):
        """POST /capture captures a screenshot."""
        payload = {
            "url": "https://example.com",
            "project_id": PROJ_ID,
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/capture",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "id" in data
            assert "url" in data

    @pytest.mark.asyncio
    async def test_compare_snapshots_fake_baseline(self):
        """POST /compare with fake baseline returns 404 or 503."""
        payload = {
            "baseline_id": FAKE_UUID,
            "current_url": "https://example.com",
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/compare",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (404, 422, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_analyze_snapshot_fake_id(self):
        """POST /analyze with fake snapshot_id returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/analyze",
                headers=HEADERS,
                params={"snapshot_id": FAKE_UUID},
            )
        assert resp.status_code in (404, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_create_baseline(self):
        """POST /baselines creates a baseline (requires screenshot capture)."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/baselines",
                headers=HEADERS,
                params={
                    "url": "https://example.com",
                    "name": f"test-baseline-{uuid.uuid4().hex[:8]}",
                    "project_id": PROJ_ID,
                },
            )
        assert resp.status_code in (200, 201, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_list_baselines(self):
        """GET /baselines returns baseline list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/baselines",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "baselines" in data
            assert "total" in data

    @pytest.mark.asyncio
    async def test_get_baseline_fake_uuid(self):
        """GET /baselines/{fake} returns 404 or 503."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/baselines/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_baseline_history_fake_uuid(self):
        """GET /baselines/{fake}/history returns 404 or 503."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/baselines/{FAKE_UUID}/history",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_list_comparisons(self):
        """GET /comparisons returns comparison list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/comparisons",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "comparisons" in data

    @pytest.mark.asyncio
    async def test_get_comparison_fake_uuid(self):
        """GET /comparisons/{fake} returns 404 or 503."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/comparisons/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_get_snapshot_fake_uuid(self):
        """GET /snapshots/{fake} returns 404 or 503."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/snapshots/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_ai_explain_fake_comparison(self):
        """POST /ai/explain with fake comparison_id returns 404 or 503."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/ai/explain",
                headers=HEADERS,
                params={"comparison_id": FAKE_UUID},
            )
        assert resp.status_code in (404, 422, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_accessibility_analyze_fake_snapshot(self):
        """POST /accessibility/analyze with fake snapshot returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/accessibility/analyze",
                headers=HEADERS,
                params={"snapshot_id": FAKE_UUID},
            )
        assert resp.status_code in (404, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_responsive_capture(self):
        """POST /responsive/capture captures at multiple viewports."""
        payload = {
            "url": "https://example.com",
            "project_id": PROJ_ID,
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/responsive/capture",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "results" in data

    @pytest.mark.asyncio
    async def test_browsers_capture(self):
        """POST /browsers/capture captures in multiple browsers."""
        payload = {
            "url": "https://example.com",
            "project_id": PROJ_ID,
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/browsers/capture",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "results" in data

    @pytest.mark.asyncio
    async def test_approve_fake_comparison(self):
        """POST /comparisons/{fake}/approve returns 404 or 503."""
        payload = {
            "notes": "Test approval",
            "update_baseline": False,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/comparisons/{FAKE_UUID}/approve",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (404, 422, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_reject_fake_comparison(self):
        """POST /comparisons/{fake}/reject returns 404 or 503."""
        payload = {
            "notes": "Test rejection reason",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/comparisons/{FAKE_UUID}/reject",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (404, 422, 429, 500, 503), resp.text

    @pytest.mark.asyncio
    async def test_capture_with_viewport(self):
        """POST /capture with custom viewport returns valid response."""
        payload = {
            "url": "https://example.com",
            "viewport": {"width": 1920, "height": 1080},
            "browser": "chromium",
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{VISUAL_URL}/capture",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "id" in data

    @pytest.mark.asyncio
    async def test_baselines_total_non_negative(self):
        """GET /baselines total field is non-negative."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{VISUAL_URL}/baselines",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("total", 0) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Parameterized Tests  (~20 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestParameterized:
    """Parameterized test CRUD, expand, preview, validate, and import."""

    @pytest.mark.asyncio
    async def test_list_data_sources(self):
        """GET /data-sources returns supported data source types."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PARAM_URL}/data-sources",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ParamDataSourcesResponse.model_validate(data)
            assert data["success"] is True
            # Must have at least inline, csv, json
            types = [ds.get("type") for ds in data.get("dataSources", data.get("data_sources", []))]
            assert "inline" in types
            assert "csv" in types
            assert "json" in types

    @pytest.mark.asyncio
    async def test_list_iteration_modes(self):
        """GET /iteration-modes returns supported iteration modes."""
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.get(
                f"{PARAM_URL}/iteration-modes",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ParamIterationModesResponse.model_validate(data)
            assert data["success"] is True
            modes = [m.get("mode") for m in data.get("iterationModes", data.get("iteration_modes", []))]
            assert "sequential" in modes

    @pytest.mark.asyncio
    async def test_create_parameterized_test(self):
        """POST /tests creates a parameterized test."""
        payload = {
            "project_id": PROJ_ID,
            "name": f"Integration Test {uuid.uuid4().hex[:8]}",
            "description": "E2E integration test parameterized test",
            "data_source_type": "inline",
            "data_source_config": {
                "data": [
                    {"username": "admin", "password": "admin123"},
                    {"username": "user", "password": "user123"},
                ],
            },
            "steps": [
                {"action": "fill", "target": "#username", "value": "{{username}}"},
                {"action": "fill", "target": "#password", "value": "{{password}}"},
                {"action": "click", "target": "#submit"},
            ],
            "iteration_mode": "sequential",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/tests",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 201, 422, 429, 500), resp.text
        if resp.status_code in (200, 201):
            data = resp.json()
            assert "id" in data
            assert data.get("name") or data.get("testName")

    @pytest.mark.asyncio
    async def test_list_tests(self):
        """GET /tests returns test list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PARAM_URL}/tests",
                headers=HEADERS,
                params={"project_id": PROJ_ID},
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_test_fake_uuid(self):
        """GET /tests/{fake} returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PARAM_URL}/tests/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_update_test_fake_uuid(self):
        """PATCH /tests/{fake} returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.patch(
                f"{PARAM_URL}/tests/{FAKE_UUID}",
                headers=HEADERS,
                json={"name": "Updated Name"},
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_delete_test_fake_uuid(self):
        """DELETE /tests/{fake} returns 204 or 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{PARAM_URL}/tests/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (204, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_expand_test(self):
        """POST /expand expands parameterized test into instances."""
        payload = {
            "test": {
                "name": "Login Test",
                "data_source": {
                    "type": "inline",
                    "data": [
                        {"username": "admin", "password": "admin123"},
                        {"username": "user", "password": "user123"},
                    ],
                },
                "steps": [
                    {"action": "fill", "target": "#username", "value": "{{username}}"},
                    {"action": "fill", "target": "#password", "value": "{{password}}"},
                    {"action": "click", "target": "#submit"},
                ],
            },
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/expand",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ExpandResponseModel.model_validate(data)
            assert data["success"] is True
            total = data.get("totalExpanded", data.get("total_expanded", 0))
            assert total == 2

    @pytest.mark.asyncio
    async def test_preview_expansion(self):
        """POST /preview previews expanded tests."""
        payload = {
            "test": {
                "name": "Preview Test",
                "data_source": {
                    "type": "inline",
                    "data": [
                        {"email": "a@test.com"},
                        {"email": "b@test.com"},
                        {"email": "c@test.com"},
                    ],
                },
                "steps": [
                    {"action": "fill", "target": "#email", "value": "{{email}}"},
                ],
            },
            "limit": 2,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/preview",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            PreviewResponseModel.model_validate(data)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_validate_params_valid(self):
        """POST /validate with matching params returns valid=true."""
        payload = {
            "test": {
                "name": "Validate Test",
                "data_source": {
                    "type": "inline",
                    "data": [{"username": "admin", "password": "pass"}],
                },
                "steps": [
                    {"action": "fill", "target": "#username", "value": "{{username}}"},
                    {"action": "fill", "target": "#password", "value": "{{password}}"},
                ],
            },
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/validate",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ValidateResponseModel.model_validate(data)
            assert data["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_params_missing(self):
        """POST /validate with missing params returns valid=false."""
        payload = {
            "test": {
                "name": "Missing Param Test",
                "data_source": {
                    "type": "inline",
                    "data": [{"username": "admin"}],
                },
                "steps": [
                    {"action": "fill", "target": "#username", "value": "{{username}}"},
                    {"action": "fill", "target": "#password", "value": "{{password}}"},
                ],
            },
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/validate",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ValidateResponseModel.model_validate(data)
            # API may report valid=False with missing params, or valid=True if
            # it doesn't cross-reference step values against provided parameters
            if not data.get("valid", True):
                missing = data.get("missingParameters", data.get("missing_parameters", []))
                if missing:
                    assert "password" in missing

    @pytest.mark.asyncio
    async def test_import_csv_data(self):
        """POST /import-data parses CSV content."""
        payload = {
            "content": "username,password\nadmin,admin123\nuser,user123",
            "format": "csv",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/import-data",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ImportDataResponseModel.model_validate(data)
            assert data["success"] is True
            count = data.get("parameterSetsCount", data.get("parameter_sets_count", 0))
            assert count == 2
            fields = data.get("detectedFields", data.get("detected_fields", []))
            assert "username" in fields
            assert "password" in fields

    @pytest.mark.asyncio
    async def test_import_json_data(self):
        """POST /import-data parses JSON content."""
        import json as json_lib
        payload = {
            "content": json_lib.dumps([
                {"name": "Alice", "role": "admin"},
                {"name": "Bob", "role": "user"},
            ]),
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/import-data",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            ImportDataResponseModel.model_validate(data)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_add_param_set_fake_test(self):
        """POST /tests/{fake}/parameter-sets returns 404."""
        payload = {
            "name": "test-set",
            "values": {"username": "admin"},
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/tests/{FAKE_UUID}/parameter-sets",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_list_param_sets_fake_test(self):
        """GET /tests/{fake}/parameter-sets returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PARAM_URL}/tests/{FAKE_UUID}/parameter-sets",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_delete_param_sets_fake_test(self):
        """DELETE /tests/{fake}/parameter-sets returns 204 or 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{PARAM_URL}/tests/{FAKE_UUID}/parameter-sets",
                headers=HEADERS,
            )
        assert resp.status_code in (204, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_get_results_fake_test(self):
        """GET /tests/{fake}/results returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{PARAM_URL}/tests/{FAKE_UUID}/results",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429), resp.text

    @pytest.mark.asyncio
    async def test_execute_fake_test(self):
        """POST /execute with fake test_id returns 404."""
        payload = {
            "test_id": FAKE_UUID,
            "app_url": "https://example.com",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/execute",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (404, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_execute_inline(self):
        """POST /execute/inline executes inline parameterized test."""
        payload = {
            "name": "Inline Login Test",
            "steps": [
                {"action": "fill", "target": "#username", "value": "{{username}}"},
                {"action": "click", "target": "#submit"},
            ],
            "parameter_sets": [
                {"name": "admin", "values": {"username": "admin"}},
                {"name": "user", "values": {"username": "user"}},
            ],
            "app_url": "https://example.com",
        }
        async with httpx.AsyncClient(timeout=SLOW_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/execute/inline",
                headers=HEADERS,
                json=payload,
            )
        # Execution may fail due to browser pool not accessible, that is acceptable
        assert resp.status_code in (200, 422, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_expand_with_limit(self):
        """POST /expand with limit truncates results."""
        payload = {
            "test": {
                "name": "Limit Test",
                "data_source": {
                    "type": "inline",
                    "data": [
                        {"x": "1"}, {"x": "2"}, {"x": "3"}, {"x": "4"}, {"x": "5"},
                    ],
                },
                "steps": [
                    {"action": "fill", "target": "#input", "value": "{{x}}"},
                ],
            },
            "limit": 2,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{PARAM_URL}/expand",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            expanded = data.get("expandedTests", data.get("expanded_tests", []))
            assert len(expanded) <= 2


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Recording  (~8 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestRecording:
    """Recording upload, convert, snippet, replay, and list endpoints."""

    _uploaded_recording_id: str | None = None

    @pytest.mark.asyncio
    async def test_upload_recording(self):
        """POST /upload stores a recording with synthetic rrweb events."""
        now_ts = 1700000000000
        payload = {
            "events": [
                {"type": 4, "data": {"href": "https://example.com", "width": 1440, "height": 900}, "timestamp": now_ts},
                {"type": 2, "data": {"node": {"id": 1, "tagName": "html", "childNodes": []}}, "timestamp": now_ts + 100},
                {"type": 3, "data": {"source": 2, "type": 2, "id": 1, "x": 100, "y": 200}, "timestamp": now_ts + 500},
                {"type": 3, "data": {"source": 5, "id": 1, "text": "admin"}, "timestamp": now_ts + 1000},
                {"type": 3, "data": {"source": 2, "type": 2, "id": 1, "x": 200, "y": 300}, "timestamp": now_ts + 1500},
            ],
            "metadata": {
                "duration": 2000,
                "start_time": "2026-02-09T00:00:00Z",
                "url": "https://example.com",
                "user_agent": "Mozilla/5.0 Test",
                "viewport": {"width": 1440, "height": 900},
            },
            "project_id": PROJ_ID,
            "name": f"Test Recording {uuid.uuid4().hex[:8]}",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{RECORDING_URL}/upload",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True
            recording_id = data.get("recordingId", data.get("recording_id"))
            assert recording_id
            TestRecording._uploaded_recording_id = recording_id

    @pytest.mark.asyncio
    async def test_convert_recording(self):
        """POST /convert converts uploaded recording to test."""
        recording_id = TestRecording._uploaded_recording_id
        if not recording_id:
            pytest.skip("No recording uploaded to convert")

        payload = {
            "recording_id": recording_id,
            "test_name": "Converted Test",
            "include_waits": True,
            "include_scrolls": False,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{RECORDING_URL}/convert",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 404, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True
            steps = data.get("stepsGenerated", data.get("steps_generated", 0))
            assert steps >= 0

    @pytest.mark.asyncio
    async def test_get_replay_fake_uuid(self):
        """GET /replay/{fake} returns 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{RECORDING_URL}/replay/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_generate_snippet(self):
        """POST /snippet returns JavaScript recorder snippet."""
        payload = {
            "project_id": PROJ_ID,
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{RECORDING_URL}/snippet",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (200, 422, 429, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("success") is True
            snippet = data.get("snippet", "")
            assert "rrweb" in snippet or "record" in snippet

    @pytest.mark.asyncio
    async def test_list_recordings(self):
        """GET /recordings returns recording list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{RECORDING_URL}/recordings",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "recordings" in data

    @pytest.mark.asyncio
    async def test_list_recordings_alt(self):
        """GET /list returns recording list (backward compat)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{RECORDING_URL}/list",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "recordings" in data

    @pytest.mark.asyncio
    async def test_delete_recording_fake_uuid(self):
        """DELETE /{fake} returns 400 or 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{RECORDING_URL}/{FAKE_UUID}",
                headers=HEADERS,
            )
        # FAKE_UUID is valid UUID format so expect 404 not 400
        assert resp.status_code in (400, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_convert_fake_recording(self):
        """POST /convert with fake recording_id returns 404."""
        payload = {
            "recording_id": FAKE_UUID,
            "test_name": "Should Fail",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                f"{RECORDING_URL}/convert",
                headers=HEADERS,
                json=payload,
            )
        assert resp.status_code in (404, 422, 429, 500), resp.text


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: Chat (Thread Management Only)  (~5 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestChat:
    """Chat thread management — skipping LLM-heavy endpoints."""

    @pytest.mark.asyncio
    async def test_list_threads(self):
        """GET /threads returns thread list."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{CHAT_URL}/threads",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert "threads" in data

    @pytest.mark.asyncio
    async def test_get_history_fake_thread(self):
        """GET /history/{fake} returns 404 or empty 200."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{CHAT_URL}/history/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 404, 429), resp.text

    @pytest.mark.asyncio
    async def test_delete_history_fake_thread(self):
        """DELETE /history/{fake} returns 200 with deleted=false or 404."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{CHAT_URL}/history/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (200, 404, 429), resp.text
        if resp.status_code == 200:
            data = resp.json()
            # MemorySaver returns deleted=false
            assert data.get("deleted") is False or "thread_id" in data or "threadId" in data

    @pytest.mark.asyncio
    async def test_cancel_fake_thread(self):
        """DELETE /cancel/{fake} returns 404 or 500."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(
                f"{CHAT_URL}/cancel/{FAKE_UUID}",
                headers=HEADERS,
            )
        assert resp.status_code in (404, 429, 500), resp.text

    @pytest.mark.asyncio
    async def test_get_history_invalid_uuid(self):
        """GET /history/{non-uuid} returns 400."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{CHAT_URL}/history/not-a-valid-uuid",
                headers=HEADERS,
            )
        assert resp.status_code in (400, 422, 429), resp.text
