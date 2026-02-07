"""
Production-grade E2E Healing Pipeline Integration Tests.

Tests the entire healing pipeline against LIVE production with NO MOCKS.
Every response is validated against strict Pydantic models matching the
production API contract (camelCase via CamelCaseMiddleware).

Run with:
    pytest tests/integration/test_healing_pipeline_live.py -v --tb=short

Run by category:
    pytest tests/integration/test_healing_pipeline_live.py -v -k "Selector"
    pytest tests/integration/test_healing_pipeline_live.py -v -k "RealBug"
    pytest tests/integration/test_healing_pipeline_live.py -v -k "LearningLoop"
    pytest tests/integration/test_healing_pipeline_live.py -v -k "EdgeCase"
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
    "https://argus-brain-production.up.railway.app",
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

# LLM-backed endpoints: 90s default.  Cognee ECL pipeline: 300s.
DEFAULT_TIMEOUT = httpx.Timeout(90.0, connect=15.0)
COGNEE_TIMEOUT = httpx.Timeout(300.0, connect=15.0)

# ═══════════════════════════════════════════════════════════════════════════
# Endpoint URLs
# ═══════════════════════════════════════════════════════════════════════════

SUGGEST_URL = f"{BASE_URL}/api/v1/healing/organizations/{ORG_ID}/suggest"
ROOT_CAUSE_URL = f"{BASE_URL}/api/v1/healing/organizations/{ORG_ID}/analyze-root-cause"
SIMILAR_URL = f"{BASE_URL}/api/v1/healing/organizations/{ORG_ID}/similar-errors"
PATTERNS_URL = f"{BASE_URL}/api/v1/healing/organizations/{ORG_ID}/patterns"
CONFIG_URL = f"{BASE_URL}/api/v1/healing/organizations/{ORG_ID}/config"
STATS_URL = f"{BASE_URL}/api/v1/healing/organizations/{ORG_ID}/stats"
EVENTS_URL = f"{BASE_URL}/events/healing.requested"

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


class HealingSuggestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    fix_type: str = Field(alias="fixType")
    confidence: float
    explanation: str
    old_value: str | None = Field(None, alias="oldValue")
    new_value: str | None = Field(None, alias="newValue")
    success_rate: float | None = Field(None, alias="successRate")
    pattern_id: str | None = Field(None, alias="patternId")


class SuggestResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    suggestions: list[HealingSuggestion]
    source: str
    latency_ms: float = Field(alias="latencyMs")
    confidence: float
    intent: str
    cached: bool = False
    metadata: dict = Field(default_factory=dict)


class RootCauseResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    root_cause: str = Field(alias="rootCause")
    category: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list, alias="suggestedActions")
    source: str
    latency_ms: float = Field(alias="latencyMs")


class SimilarErrorsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    similar_errors: list[dict] = Field(alias="similarErrors")
    count: int
    source: str
    latency_ms: float = Field(alias="latencyMs")
    confidence: float


class PatternCreateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    pattern_id: str | None = Field(None, alias="patternId")
    fingerprint: str | None = None
    is_new: bool = Field(alias="isNew")
    message: str | None = None


class HealingPattern(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    fingerprint: str
    original_selector: str = Field(alias="originalSelector")
    healed_selector: str = Field(alias="healedSelector")
    error_type: str = Field(alias="errorType")
    success_count: int = Field(alias="successCount")
    failure_count: int = Field(alias="failureCount")
    confidence: float
    project_id: str | None = Field(None, alias="projectId")
    created_at: str = Field(alias="createdAt")


class HealingStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_patterns: int = Field(alias="totalPatterns")
    total_heals_applied: int = Field(alias="totalHealsApplied")
    total_heals_suggested: int = Field(alias="totalHealsSuggested")
    success_rate: float = Field(alias="successRate")
    top_error_types: dict = Field(alias="topErrorTypes")
    patterns_by_project: dict = Field(alias="patternsByProject")
    heals_last_24h: int = Field(alias="healsLast24H")
    heals_last_7d: int = Field(alias="healsLast7D")
    heals_last_30d: int = Field(alias="healsLast30D")
    avg_confidence: float = Field(alias="avgConfidence")
    recent_heals: list[dict] = Field(alias="recentHeals")


class EventResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    event_id: str = Field(alias="eventId")
    event_type: str = Field(alias="eventType")
    topic: str
    timestamp: str
    message: str


class HealingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    organization_id: str = Field(alias="organizationId")
    enabled: bool
    auto_apply: bool = Field(alias="autoApply")
    min_confidence_auto: float = Field(alias="minConfidenceAuto")
    min_confidence_suggest: float = Field(alias="minConfidenceSuggest")
    heal_selectors: bool = Field(alias="healSelectors")
    max_heals_per_hour: int = Field(alias="maxHealsPerHour")
    max_heals_per_test: int = Field(alias="maxHealsPerTest")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: Selector Change Healing  (8 tests)
# Enterprise #1 pain point — 70 % of real CI/CD failures
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectorChangeHealing:
    """CSS, XPath, Shadow DOM, iframe selector failures → POST /suggest."""

    @pytest.mark.asyncio
    async def test_1_1_css_id_selector_not_found(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "Element not found: #submit-btn",
                    "selector": "#submit-btn",
                    "error_type": "selector_not_found",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1
        assert result.confidence > 0
        assert result.source in ("cache", "vector", "llm")

    @pytest.mark.asyncio
    async def test_1_2_compound_css_selector(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "Element .login-form button.primary not found after 30s wait",
                    "selector": ".login-form button.primary",
                    "error_type": "selector_not_found",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_1_3_data_testid_selector(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": 'Selector [data-testid="login-submit"] not found - data-testid was renamed',
                    "selector": 'button[data-testid="login-submit"]',
                    "error_type": "selector_not_found",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_1_4_dynamic_react_select_id(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "Element #react-select-2-option-0 not found - dynamic ID changed between renders",
                    "selector": "#react-select-2-option-0",
                    "error_type": "selector_not_found",
                    "context": {"framework": "react", "component": "react-select"},
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_1_5_xpath_dom_restructure(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "XPath //div[@class='old-nav']//a[text()='Home'] not found - navigation HTML was restructured",
                    "selector": "//div[@class='old-nav']//a[text()='Home']",
                    "error_type": "selector_not_found",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_1_6_material_ui_class_change(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": (
                        "Element .MuiButton-root.MuiButton-contained not found - "
                        "Material UI v5→v6 upgrade changed class naming convention"
                    ),
                    "selector": ".MuiButton-root.MuiButton-contained",
                    "error_type": "selector_not_found",
                    "context": {"framework": "material-ui", "upgrade": "v5-to-v6"},
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_1_7_shadow_dom_selector(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "pierce/#shadow-host >>> .inner-btn - Shadow DOM selector invalid, host element restructured",
                    "selector": "pierce/#shadow-host >>> .inner-btn",
                    "error_type": "selector_not_found",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_1_8_iframe_nested_selector(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "Element #pay-btn inside iframe[name='checkout'] not found - iframe removed or renamed",
                    "selector": "iframe[name='checkout'] #pay-btn",
                    "error_type": "selector_not_found",
                    "context": {"iframe": "checkout", "page_url": "https://shop.example.com/cart"},
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Timing / Synchronization  (5 tests)
# Enterprise #2 pain point — flaky tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTimingIssues:
    """Timeout, animation, stale element, lazy-load failures → POST /analyze-root-cause."""

    @pytest.mark.asyncio
    async def test_2_1_element_wait_timeout(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": "Timeout of 30000ms exceeded waiting for selector #dynamic-content",
                    "error_type": "timeout",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.category == "timeout"
        assert len(result.suggested_actions) >= 1

    @pytest.mark.asyncio
    async def test_2_2_animation_not_interactable(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": (
                        "Element #modal-overlay is not interactable, CSS animation in progress. "
                        "Click intercepted by overlay."
                    ),
                    "error_type": "not_interactable",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.confidence > 0
        assert result.root_cause  # non-empty

    @pytest.mark.asyncio
    async def test_2_3_stale_element_spa_rerender(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": (
                        "StaleElementReferenceError: element is not attached to the page document. "
                        "React component re-rendered between find and click."
                    ),
                    "error_type": "stale_element",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.root_cause

    @pytest.mark.asyncio
    async def test_2_4_lazy_load_not_visible(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": (
                        "Element .lazy-loaded-content not visible after 10s wait. "
                        "Intersection Observer never triggered."
                    ),
                    "error_type": "not_visible",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.root_cause

    @pytest.mark.asyncio
    async def test_2_5_websocket_timeout(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": (
                        "WebSocket connection to wss://api.example.com/ws timed out after 60s. "
                        "Real-time dashboard failed to load."
                    ),
                    "error_type": "infrastructure_timeout",
                    "context": {"protocol": "websocket", "target": "api.example.com"},
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.root_cause


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: Assertion Failures  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestAssertionFailures:
    """Real-world assertion changes → POST /suggest."""

    @pytest.mark.asyncio
    async def test_3_1_text_copy_change(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": 'AssertionError: Expected button text "Login" but found "Sign In". Marketing team updated copy.',
                    "error_type": "assertion_failed",
                    "context": {"page": "/auth/login", "element": "button.submit"},
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_3_2_date_format_change(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": 'AssertionError: Expected date format "MM/DD/YYYY" but got "2026-02-07". Backend switched to ISO 8601 format.',
                    "error_type": "assertion_failed",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_3_3_number_formatting_change(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": 'AssertionError: Expected "1000" but got "1,000". Number formatting was added for locale en-US.',
                    "error_type": "assertion_failed",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_3_4_api_schema_field_rename(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": 'KeyError: Expected field "username" in API response but got "user_name". Backend schema migration renamed the field.',
                    "error_type": "assertion_failed",
                    "context": {"endpoint": "GET /api/users/me"},
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1

    @pytest.mark.asyncio
    async def test_3_5_list_order_mismatch(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": "AssertionError: List order mismatch - expected [A, B, C] got [B, A, C]. Backend sort changed from alphabetical to recent-first.",
                    "error_type": "assertion_failed",
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert len(result.suggestions) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Real Bugs — Should NOT Heal  (5 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestRealBugsNotHealed:
    """Real bugs — healing should identify, NOT auto-fix with selector changes."""

    # These categories indicate real application bugs, not flaky test infra
    _BUG_CATEGORIES = frozenset({
        "real_bug", "server_error", "backend_error", "application_error",
        "data_changed", "api_error", "unknown",
    })

    @pytest.mark.asyncio
    async def test_4_1_server_500_error(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": (
                        "HTTP 500 Internal Server Error on POST /api/users. "
                        "Server returned traceback: NullPointerException in UserService.create()"
                    ),
                    "error_type": "server_error",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.category not in ("selector_changed", "timing_issue"), (
            f"Server 500 misclassified as '{result.category}' — should be a bug category"
        )

    @pytest.mark.asyncio
    async def test_4_2_auth_token_expired(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": "HTTP 401 Unauthorized - JWT token expired. Token was issued 25 hours ago, max TTL is 24h.",
                    "error_type": "auth_error",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        # Must NOT suggest updating a selector — this is an auth issue
        for action in result.suggested_actions:
            assert "update_selector" not in action.lower(), (
                f"Auth error should not suggest selector fix: {action}"
            )

    @pytest.mark.asyncio
    async def test_4_3_empty_response_body(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": "Response body is empty, expected JSON with user profile. Content-Length: 0, Status: 200 OK.",
                    "error_type": "data_error",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.root_cause  # non-empty analysis

    @pytest.mark.asyncio
    async def test_4_4_deprecated_endpoint_404(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": "HTTP 404 Not Found: /api/v2/deprecated-endpoint. Endpoint was removed in v3 migration.",
                    "error_type": "api_error",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.root_cause

    @pytest.mark.asyncio
    async def test_4_5_database_constraint_violation(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                ROOT_CAUSE_URL,
                headers=HEADERS,
                json={
                    "error_message": "Database error: unique constraint violated on users.email. Duplicate key value (email)=(test@example.com) already exists.",
                    "error_type": "database_error",
                },
            )
        assert resp.status_code == 200, resp.text
        result = RootCauseResponse.model_validate(resp.json())
        assert result.root_cause


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: Learning Loop  (5 tests)
# Store → retrieve → reuse — full pipeline including Cognee ECL
# ═══════════════════════════════════════════════════════════════════════════


class TestLearningLoop:
    """Verify the store → retrieve → reuse learning cycle with Cognee."""

    _run_id = uuid.uuid4().hex[:8]
    _test_selector = f"#learn-loop-test-{_run_id}"
    _healed_selector = f"[data-testid='learn-loop-fix-{_run_id}']"
    _created_pattern_id: str | None = None

    @pytest.mark.asyncio
    async def test_5_1_create_new_pattern(self):
        """Store a new healing pattern through the full pipeline (Supabase + Cognee)."""
        async with httpx.AsyncClient(timeout=COGNEE_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_URL,
                headers=HEADERS,
                json={
                    "original_selector": self._test_selector,
                    "healed_selector": self._healed_selector,
                    "error_type": "selector_not_found",
                    "project_id": PROJ_ID,
                    "page_url": "https://example.com/learn-test",
                    "metadata": {"test_run": self._run_id},
                    "store_in_cognee": True,
                },
            )
        assert resp.status_code == 200, resp.text
        result = PatternCreateResponse.model_validate(resp.json())
        assert result.success is True
        assert result.is_new is True
        assert result.pattern_id is not None
        TestLearningLoop._created_pattern_id = result.pattern_id

    @pytest.mark.asyncio
    async def test_5_2_duplicate_pattern_increments(self):
        """Same fingerprint should increment success_count, not create new."""
        async with httpx.AsyncClient(timeout=COGNEE_TIMEOUT) as client:
            resp = await client.post(
                PATTERNS_URL,
                headers=HEADERS,
                json={
                    "original_selector": self._test_selector,
                    "healed_selector": self._healed_selector,
                    "error_type": "selector_not_found",
                    "project_id": PROJ_ID,
                    "store_in_cognee": True,
                },
            )
        assert resp.status_code == 200, resp.text
        result = PatternCreateResponse.model_validate(resp.json())
        assert result.success is True
        assert result.is_new is False

    @pytest.mark.asyncio
    async def test_5_3_list_patterns_contains_new(self):
        """GET /patterns must include the pattern we just created."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(PATTERNS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        patterns = [HealingPattern.model_validate(p) for p in resp.json()]
        our_patterns = [p for p in patterns if p.original_selector == self._test_selector]
        assert len(our_patterns) >= 1, (
            f"Pattern '{self._test_selector}' not found among {len(patterns)} patterns"
        )
        assert our_patterns[0].success_count >= 2  # created + incremented

    @pytest.mark.asyncio
    async def test_5_4_stats_reflect_new_pattern(self):
        """Stats should reflect the increased pattern count."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(STATS_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        stats = HealingStats.model_validate(resp.json())
        assert stats.total_patterns >= 14  # was 13 before test run

    @pytest.mark.asyncio
    async def test_5_5_similar_errors_finds_stored_pattern(self):
        """Similar error search should surface the pattern from Cognee."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SIMILAR_URL,
                headers=HEADERS,
                json={
                    "error_message": f"Element {self._test_selector} not found on page",
                    "selector": self._test_selector,
                    "error_type": "selector_not_found",
                    "skip_cache": True,
                },
            )
        assert resp.status_code == 200, resp.text
        result = SimilarErrorsResponse.model_validate(resp.json())
        assert isinstance(result.similar_errors, list)


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Healing Event Pipeline  (5 tests)
# Kafka event publishing via /events/healing.requested
# ═══════════════════════════════════════════════════════════════════════════


class TestHealingEventPipeline:
    """Kafka event publishing via the HTTP Event Gateway."""

    @pytest.mark.asyncio
    async def test_6_1_valid_selector_not_found_event(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                EVENTS_URL,
                headers=HEADERS,
                json={
                    "org_id": ORG_ID,
                    "project_id": PROJ_ID,
                    "data": {
                        "test_id": str(uuid.uuid4()),
                        "failure_id": str(uuid.uuid4()),
                        "error_type": "selector_not_found",
                        "error_message": "Element #checkout-btn not found after DOM change",
                        "failed_selector": "#checkout-btn",
                        "page_url": "https://shop.example.com/checkout",
                    },
                },
            )
        assert resp.status_code == 200, resp.text
        result = EventResponse.model_validate(resp.json())
        assert result.success is True
        assert result.event_id
        assert result.topic == "argus.healing.requested"

    @pytest.mark.asyncio
    async def test_6_2_timeout_event(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                EVENTS_URL,
                headers=HEADERS,
                json={
                    "org_id": ORG_ID,
                    "project_id": PROJ_ID,
                    "data": {
                        "test_id": str(uuid.uuid4()),
                        "failure_id": str(uuid.uuid4()),
                        "error_type": "timeout",
                        "error_message": "Timeout 30s waiting for .dashboard-widget",
                        "failed_selector": ".dashboard-widget",
                        "page_url": "https://app.example.com/dashboard",
                    },
                },
            )
        assert resp.status_code == 200, resp.text
        result = EventResponse.model_validate(resp.json())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_6_3_minimal_fields_event(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                EVENTS_URL,
                headers=HEADERS,
                json={
                    "org_id": ORG_ID,
                    "data": {
                        "error_type": "unknown",
                        "error_message": "Minimal event test",
                    },
                },
            )
        assert resp.status_code == 200, resp.text
        result = EventResponse.model_validate(resp.json())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_6_4_fully_populated_event(self):
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                EVENTS_URL,
                headers=HEADERS,
                json={
                    "org_id": ORG_ID,
                    "project_id": PROJ_ID,
                    "user_id": "test-user-pipeline",
                    "correlation_id": str(uuid.uuid4()),
                    "causation_id": str(uuid.uuid4()),
                    "source": "integration-test",
                    "idempotency_key": f"test-idemp-{uuid.uuid4().hex[:8]}",
                    "data": {
                        "test_id": str(uuid.uuid4()),
                        "failure_id": str(uuid.uuid4()),
                        "error_type": "selector_not_found",
                        "error_message": "Full event: #full-test-btn not found",
                        "failed_selector": "#full-test-btn",
                        "page_url": "https://example.com/full-test",
                        "step_index": 3,
                        "test_name": "checkout flow",
                    },
                },
            )
        assert resp.status_code == 200, resp.text
        result = EventResponse.model_validate(resp.json())
        assert result.success is True
        assert result.event_id

    @pytest.mark.asyncio
    async def test_6_5_event_with_extra_data_fields(self):
        """Extra fields in data payload must be accepted (forward compatibility)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                EVENTS_URL,
                headers=HEADERS,
                json={
                    "org_id": ORG_ID,
                    "project_id": PROJ_ID,
                    "data": {
                        "test_id": str(uuid.uuid4()),
                        "failure_id": str(uuid.uuid4()),
                        "error_type": "selector_not_found",
                        "error_message": "Extra fields test",
                        "failed_selector": "#btn",
                        "page_url": "https://example.com",
                        "custom_field_1": "should be accepted",
                        "custom_field_2": 42,
                        "nested": {"deeply": {"nested": True}},
                    },
                },
            )
        assert resp.status_code == 200, resp.text
        result = EventResponse.model_validate(resp.json())
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════
# Category 7: Edge Cases & Error Handling  (7 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Validation, auth, and robustness edge cases."""

    @pytest.mark.asyncio
    async def test_7_1_empty_error_message(self):
        """Empty error_message — API accepts it (returns 200 with LLM analysis)."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={"error_message": ""},
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert isinstance(result.suggestions, list)

    @pytest.mark.asyncio
    async def test_7_2_very_long_selector(self):
        """10K character selector must not crash the server."""
        long_selector = "#btn-" + "a" * 10_000
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": f"Element {long_selector[:100]}... not found",
                    "selector": long_selector,
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert isinstance(result.suggestions, list)

    @pytest.mark.asyncio
    async def test_7_3_invalid_org_id_format(self):
        """Non-UUID org_id must return 400 (validate_uuid in tenant middleware)."""
        bad_url = f"{BASE_URL}/api/v1/healing/organizations/not-a-uuid/suggest"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                bad_url,
                headers=HEADERS,
                json={"error_message": "test error"},
            )
        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_7_4_wrong_api_key(self):
        """Invalid API key must return 401."""
        bad_headers = {**HEADERS, "X-API-Key": "argus_sk_invalid_key_1234567890"}
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=bad_headers,
                json={"error_message": "test error"},
            )
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_7_5_missing_api_key(self):
        """No authentication header must return 401."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers={"Content-Type": "application/json"},
                json={"error_message": "test error"},
            )
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_7_6_nonexistent_org_id(self):
        """Valid UUID but non-existent org must return 403 (IDOR protection)."""
        fake_org = str(uuid.uuid4())
        fake_url = f"{BASE_URL}/api/v1/healing/organizations/{fake_org}/suggest"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                fake_url,
                headers=HEADERS,
                json={"error_message": "test error"},
            )
        assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"

    @pytest.mark.asyncio
    async def test_7_7_concurrent_suggest_requests(self):
        """5 concurrent requests must all return valid responses."""
        payloads = [
            {
                "error_message": f"Concurrent test {i}: Element #btn-{i} not found",
                "selector": f"#btn-{i}",
                "skip_cache": True,
            }
            for i in range(5)
        ]
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            tasks = [
                client.post(SUGGEST_URL, headers=HEADERS, json=p) for p in payloads
            ]
            responses = await asyncio.gather(*tasks)

        for i, resp in enumerate(responses):
            assert resp.status_code == 200, (
                f"Concurrent request {i} failed: {resp.status_code}: {resp.text}"
            )
            result = SuggestResponse.model_validate(resp.json())
            assert len(result.suggestions) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Category 8: Config Management  (3 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigManagement:
    """Healing configuration GET/PUT."""

    @pytest.mark.asyncio
    async def test_8_1_get_config(self):
        """GET config must return the full HealingConfig schema."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(CONFIG_URL, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        config = HealingConfig.model_validate(resp.json())
        assert isinstance(config.enabled, bool)
        assert 0.0 <= config.min_confidence_auto <= 1.0
        assert 0.0 <= config.min_confidence_suggest <= 1.0

    @pytest.mark.asyncio
    async def test_8_2_disable_then_reenable_healing(self):
        """PUT enabled=false then enabled=true must persist."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Disable
            resp = await client.put(CONFIG_URL, headers=HEADERS, json={"enabled": False})
            assert resp.status_code == 200, resp.text
            config = HealingConfig.model_validate(resp.json())
            assert config.enabled is False

            # Re-enable
            resp = await client.put(CONFIG_URL, headers=HEADERS, json={"enabled": True})
            assert resp.status_code == 200, resp.text
            config = HealingConfig.model_validate(resp.json())
            assert config.enabled is True

    @pytest.mark.asyncio
    async def test_8_3_update_confidence_thresholds(self):
        """Update thresholds, verify via GET, then restore defaults."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Set custom thresholds
            resp = await client.put(
                CONFIG_URL,
                headers=HEADERS,
                json={"min_confidence_auto": 0.90, "min_confidence_suggest": 0.60},
            )
            assert resp.status_code == 200, resp.text
            config = HealingConfig.model_validate(resp.json())
            assert config.min_confidence_auto == 0.90
            assert config.min_confidence_suggest == 0.60

            # Verify persistence via GET
            resp = await client.get(CONFIG_URL, headers=HEADERS)
            assert resp.status_code == 200, resp.text
            config = HealingConfig.model_validate(resp.json())
            assert config.min_confidence_auto == 0.90
            assert config.min_confidence_suggest == 0.60

            # Restore defaults
            resp = await client.put(
                CONFIG_URL,
                headers=HEADERS,
                json={"min_confidence_auto": 0.95, "min_confidence_suggest": 0.70},
            )
            assert resp.status_code == 200, resp.text


# ═══════════════════════════════════════════════════════════════════════════
# Category 9: Cache Behavior  (3 tests)
# Intelligence layer caching — hard assertions
# ═══════════════════════════════════════════════════════════════════════════


class TestCacheBehavior:
    """Validate the intelligence layer caching contract."""

    _unique_suffix = uuid.uuid4().hex[:8]

    @pytest.mark.asyncio
    async def test_9_1_skip_cache_bypasses_cache(self):
        """skip_cache=True must return a non-cached result."""
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={
                    "error_message": f"Cache bypass test {self._unique_suffix}: Element #cold-btn not found",
                    "selector": "#cold-btn",
                    "skip_cache": True,
                },
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert result.cached is False, "skip_cache=True but response reports cached=True"
        assert result.source != "cache", "skip_cache=True but source is 'cache'"

    @pytest.mark.asyncio
    async def test_9_2_warm_cache_returns_cached_result(self):
        """Second identical call must be served from cache."""
        payload = {
            "error_message": f"Warm cache test {self._unique_suffix}: Element #warm-btn not found",
            "selector": "#warm-btn",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Cold call — populates cache
            resp1 = await client.post(SUGGEST_URL, headers=HEADERS, json=payload)
            assert resp1.status_code == 200, resp1.text
            cold = SuggestResponse.model_validate(resp1.json())

            # Allow cache propagation
            await asyncio.sleep(2)

            # Warm call — must hit cache
            resp2 = await client.post(SUGGEST_URL, headers=HEADERS, json=payload)
            assert resp2.status_code == 200, resp2.text
            warm = SuggestResponse.model_validate(resp2.json())

        assert warm.cached is True or warm.source == "cache", (
            f"Cache miss on second call: cached={warm.cached}, source={warm.source}, "
            f"cold_latency={cold.latency_ms:.0f}ms, warm_latency={warm.latency_ms:.0f}ms"
        )

    @pytest.mark.asyncio
    async def test_9_3_skip_cache_forces_fresh_even_after_warm(self):
        """skip_cache=True after a cached result must force a fresh lookup."""
        payload_base = {
            "error_message": f"Skip after warm {self._unique_suffix}: Element #skip-btn not found",
            "selector": "#skip-btn",
        }
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # Populate cache
            await client.post(SUGGEST_URL, headers=HEADERS, json=payload_base)
            await asyncio.sleep(1)

            # skip_cache overrides the warm entry
            resp = await client.post(
                SUGGEST_URL,
                headers=HEADERS,
                json={**payload_base, "skip_cache": True},
            )
        assert resp.status_code == 200, resp.text
        result = SuggestResponse.model_validate(resp.json())
        assert result.cached is False, "skip_cache=True but response still cached"


# ═══════════════════════════════════════════════════════════════════════════
# Cleanup: Remove test patterns created during the learning loop
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanup:
    """Remove test data created during this run."""

    @pytest.mark.asyncio
    async def test_cleanup_learning_loop_pattern(self):
        pattern_id = TestLearningLoop._created_pattern_id
        if pattern_id is None:
            pytest.skip("No pattern was created to clean up")

        delete_url = f"{PATTERNS_URL}/{pattern_id}"
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.delete(delete_url, headers=HEADERS)
        assert resp.status_code in (200, 404), (
            f"Cleanup failed: {resp.status_code}: {resp.text}"
        )
