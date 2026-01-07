"""Self-Healer Agent - Analyzes test failures and suggests fixes.

This agent:
- Analyzes failure patterns (selector changes, timing issues, etc.)
- Uses Claude vision to understand visual changes
- Generates fix suggestions with confidence scores
- Can auto-apply fixes above confidence threshold
- Learns from successful healings via Cloudflare KV cache
"""

import base64
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .base import BaseAgent, AgentResult
from .prompts import get_enhanced_prompt
from ..config import ModelName
from ..core.model_router import TaskType, TaskComplexity
from ..services.cache import cache_healing_pattern, get_cached, set_cached


class FailureType(Enum):
    """Types of test failures that can be healed."""

    SELECTOR_CHANGED = "selector_changed"
    TIMING_ISSUE = "timing_issue"
    UI_CHANGED = "ui_changed"
    DATA_CHANGED = "data_changed"
    REAL_BUG = "real_bug"
    UNKNOWN = "unknown"


class FixType(Enum):
    """Types of fixes that can be applied."""

    UPDATE_SELECTOR = "update_selector"
    ADD_WAIT = "add_wait"
    INCREASE_TIMEOUT = "increase_timeout"
    UPDATE_ASSERTION = "update_assertion"
    UPDATE_TEST_DATA = "update_test_data"
    NONE = "none"


@dataclass
class FailureDiagnosis:
    """Diagnosis of a test failure."""

    failure_type: FailureType
    confidence: float  # 0.0 to 1.0
    explanation: str
    affected_step: Optional[int] = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class FixSuggestion:
    """A suggested fix for a failure."""

    fix_type: FixType
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    confidence: float = 0.0
    explanation: str = ""
    requires_review: bool = True

    def to_dict(self) -> dict:
        return {
            "fix_type": self.fix_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "requires_review": self.requires_review,
        }


@dataclass
class HealingResult:
    """Result from attempting to heal a test failure."""

    test_id: str
    diagnosis: FailureDiagnosis
    suggested_fixes: list[FixSuggestion]
    auto_healed: bool = False
    healed_test_spec: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "diagnosis": {
                "type": self.diagnosis.failure_type.value,
                "confidence": self.diagnosis.confidence,
                "explanation": self.diagnosis.explanation,
                "affected_step": self.diagnosis.affected_step,
            },
            "suggested_fixes": [f.to_dict() for f in self.suggested_fixes],
            "auto_healed": self.auto_healed,
        }


class SelfHealerAgent(BaseAgent):
    """Agent that analyzes test failures and suggests/applies fixes.

    Uses:
    - Pattern matching for common failure types
    - Multi-model AI for visual analysis (GPT-4V, Gemini, Claude)
    - Historical failure data for learning
    - Cached healing patterns from previous successful fixes

    Healing strategies:
    - Selector updates when elements move
    - Timing adjustments for flaky tests
    - Assertion updates for intentional changes
    - Flagging real bugs for human review
    """

    # Self-healing requires high-quality reasoning models
    DEFAULT_TASK_TYPE = TaskType.SELF_HEALING

    def __init__(
        self,
        auto_heal_threshold: float = 0.9,
        **kwargs,
    ):
        """Initialize healer with configuration.

        Args:
            auto_heal_threshold: Minimum confidence to auto-apply fixes
        """
        super().__init__(**kwargs)
        self.auto_heal_threshold = auto_heal_threshold

        # Override to use vision-capable model (only used in single-model mode)
        self.model = ModelName.SONNET

    def _generate_healing_fingerprint(
        self,
        original_selector: str,
        error_type: str,
    ) -> str:
        """Generate a unique fingerprint for a healing pattern."""
        key = f"{original_selector}:{error_type}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def _lookup_cached_healing(
        self,
        original_selector: str,
        error_type: str,
    ) -> Optional[FixSuggestion]:
        """Look up a cached healing pattern from previous successful fixes."""
        fingerprint = self._generate_healing_fingerprint(original_selector, error_type)
        cache_key = f"heal:{fingerprint}"

        cached = await get_cached(cache_key)
        if cached:
            self.log.info(
                "Found cached healing pattern",
                original_selector=original_selector,
                healed_selector=cached.get("healed_selector"),
                confidence=cached.get("confidence", 0),
            )
            return FixSuggestion(
                fix_type=FixType.UPDATE_SELECTOR,
                old_value=original_selector,
                new_value=cached.get("healed_selector"),
                confidence=cached.get("confidence", 0.9),
                explanation=f"Using cached healing pattern (success rate: {cached.get('success_count', 0)} successes)",
                requires_review=False,
            )
        return None

    async def _store_healing_pattern(
        self,
        original_selector: str,
        healed_selector: str,
        error_type: str,
        success: bool = True,
    ) -> None:
        """Store a healing pattern in cache for future use."""
        fingerprint = self._generate_healing_fingerprint(original_selector, error_type)
        cache_key = f"heal:{fingerprint}"

        # Get existing pattern or create new one
        existing = await get_cached(cache_key)
        if existing:
            if success:
                existing["success_count"] = existing.get("success_count", 0) + 1
            else:
                existing["failure_count"] = existing.get("failure_count", 0) + 1

            # Recalculate confidence
            total = existing["success_count"] + existing["failure_count"]
            existing["confidence"] = existing["success_count"] / total if total > 0 else 0

            await set_cached(cache_key, existing, ttl_seconds=604800)  # 7 days
        elif success:
            # Create new pattern
            pattern = {
                "fingerprint": fingerprint,
                "original_selector": original_selector,
                "healed_selector": healed_selector,
                "error_type": error_type,
                "success_count": 1,
                "failure_count": 0,
                "confidence": 1.0,
            }
            await set_cached(cache_key, pattern, ttl_seconds=604800)  # 7 days
            self.log.info(
                "Stored new healing pattern",
                fingerprint=fingerprint,
                original_selector=original_selector,
                healed_selector=healed_selector,
            )

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for self-healing."""
        enhanced = get_enhanced_prompt("self_healer")
        if enhanced:
            return enhanced

        return """You are an expert test failure analyzer and healer.

Your task is to diagnose test failures and suggest fixes:

1. SELECTOR_CHANGED: Element moved or was renamed
   - Fix: Update selector to match new location
   - Evidence: Element not found, but similar element exists

2. TIMING_ISSUE: Element not ready in time
   - Fix: Add wait or increase timeout
   - Evidence: Intermittent failures, element appears later

3. UI_CHANGED: Intentional UI change
   - Fix: Update assertion or expected value
   - Evidence: Consistent failure, visual shows valid new state

4. DATA_CHANGED: Test data no longer valid
   - Fix: Update test data
   - Evidence: Data-related assertion failures

5. REAL_BUG: Actual application bug
   - Fix: None (report to developers)
   - Evidence: Unexpected behavior, errors in console

Analyze carefully and provide:
- Failure type with confidence score
- Specific fix suggestion with old and new values
- Clear explanation of your reasoning

Output must be valid JSON."""

    async def execute(
        self,
        test_spec: dict,
        failure_details: dict,
        screenshot: Optional[bytes] = None,
        error_logs: Optional[str] = None,
    ) -> AgentResult[HealingResult]:
        """Analyze a test failure and suggest fixes.

        Args:
            test_spec: The failing test specification
            failure_details: Details about the failure
            screenshot: Screenshot at failure point
            error_logs: Any error logs from the test

        Returns:
            AgentResult containing HealingResult
        """
        test_id = test_spec.get("id", "unknown")

        self.log.info(
            "Analyzing test failure",
            test_id=test_id,
            failure_type=failure_details.get("type"),
        )

        # Check for cached healing pattern first (before LLM call)
        original_selector = failure_details.get("selector") or failure_details.get("target")
        error_type = failure_details.get("type", "unknown")

        if original_selector and error_type == "selector_changed":
            cached_fix = await self._lookup_cached_healing(original_selector, error_type)
            if cached_fix and cached_fix.confidence >= self.auto_heal_threshold:
                # Use cached healing pattern - skip LLM call entirely
                diagnosis = FailureDiagnosis(
                    failure_type=FailureType.SELECTOR_CHANGED,
                    confidence=cached_fix.confidence,
                    explanation="Using cached healing pattern from previous successful fix",
                    evidence=["Cached pattern matched"],
                )

                healed_spec = self._apply_fix(test_spec, cached_fix)

                self.log.info(
                    "Applied cached healing pattern",
                    test_id=test_id,
                    original_selector=original_selector,
                    healed_selector=cached_fix.new_value,
                )

                return AgentResult(
                    success=True,
                    data=HealingResult(
                        test_id=test_id,
                        diagnosis=diagnosis,
                        suggested_fixes=[cached_fix],
                        auto_healed=True,
                        healed_test_spec=healed_spec,
                    ),
                    input_tokens=0,  # No LLM call made
                    output_tokens=0,
                )

        if not self._check_cost_limit():
            return AgentResult(
                success=False,
                error="Cost limit exceeded",
            )

        # Build analysis prompt
        prompt_content = self._build_analysis_prompt(
            test_spec, failure_details, error_logs
        )

        messages = [{"role": "user", "content": prompt_content}]

        # Add screenshot if available
        if screenshot:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(screenshot).decode(),
                            },
                        },
                        {"type": "text", "text": prompt_content},
                    ],
                }
            ]

        try:
            response = self._call_claude(
                messages=messages,
                max_tokens=2048,
            )

            content = self._extract_text_response(response)
            result_data = self._parse_json_response(content)

            if not result_data:
                return AgentResult(
                    success=False,
                    error="Failed to parse healing analysis",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            # Parse diagnosis
            diagnosis = self._parse_diagnosis(result_data)

            # Parse fix suggestions
            fixes = self._parse_fixes(result_data)

            # Attempt auto-healing if confidence is high enough
            healed_spec = None
            auto_healed = False

            if fixes and fixes[0].confidence >= self.auto_heal_threshold:
                healed_spec = self._apply_fix(test_spec, fixes[0])
                auto_healed = True
                self.log.info(
                    "Auto-healed test",
                    test_id=test_id,
                    fix_type=fixes[0].fix_type.value,
                    confidence=fixes[0].confidence,
                )

                # Store successful healing pattern for future use
                if fixes[0].fix_type == FixType.UPDATE_SELECTOR and fixes[0].old_value and fixes[0].new_value:
                    await self._store_healing_pattern(
                        original_selector=fixes[0].old_value,
                        healed_selector=fixes[0].new_value,
                        error_type=diagnosis.failure_type.value,
                        success=True,
                    )

            result = HealingResult(
                test_id=test_id,
                diagnosis=diagnosis,
                suggested_fixes=fixes,
                auto_healed=auto_healed,
                healed_test_spec=healed_spec,
            )

            return AgentResult(
                success=True,
                data=result,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        except Exception as e:
            self.log.error("Healing analysis failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
            )

    def _build_analysis_prompt(
        self,
        test_spec: dict,
        failure_details: dict,
        error_logs: Optional[str],
    ) -> str:
        """Build the analysis prompt."""
        import json

        prompt_parts = [
            "Analyze this test failure and suggest how to fix it.",
            "",
            "TEST SPECIFICATION:",
            json.dumps(test_spec, indent=2),
            "",
            "FAILURE DETAILS:",
            json.dumps(failure_details, indent=2),
        ]

        if error_logs:
            prompt_parts.extend([
                "",
                "ERROR LOGS:",
                error_logs[:2000],  # Truncate long logs
            ])

        prompt_parts.extend([
            "",
            "Analyze and respond with JSON:",
            """{
    "diagnosis": {
        "failure_type": "selector_changed|timing_issue|ui_changed|data_changed|real_bug|unknown",
        "confidence": 0.0-1.0,
        "explanation": "Why you think this is the cause",
        "affected_step": 0,
        "evidence": ["evidence 1", "evidence 2"]
    },
    "fixes": [
        {
            "fix_type": "update_selector|add_wait|increase_timeout|update_assertion|update_test_data|none",
            "old_value": "current value",
            "new_value": "suggested new value",
            "confidence": 0.0-1.0,
            "explanation": "Why this fix should work"
        }
    ]
}""",
        ])

        return "\n".join(prompt_parts)

    def _parse_diagnosis(self, data: dict) -> FailureDiagnosis:
        """Parse diagnosis from response data."""
        diag_data = data.get("diagnosis", {})

        failure_type_str = diag_data.get("failure_type", "unknown")
        try:
            failure_type = FailureType(failure_type_str)
        except ValueError:
            failure_type = FailureType.UNKNOWN

        return FailureDiagnosis(
            failure_type=failure_type,
            confidence=float(diag_data.get("confidence", 0.5)),
            explanation=diag_data.get("explanation", ""),
            affected_step=diag_data.get("affected_step"),
            evidence=diag_data.get("evidence", []),
        )

    def _parse_fixes(self, data: dict) -> list[FixSuggestion]:
        """Parse fix suggestions from response data."""
        fixes = []

        for fix_data in data.get("fixes", []):
            fix_type_str = fix_data.get("fix_type", "none")
            try:
                fix_type = FixType(fix_type_str)
            except ValueError:
                fix_type = FixType.NONE

            confidence = float(fix_data.get("confidence", 0.5))

            fixes.append(
                FixSuggestion(
                    fix_type=fix_type,
                    old_value=fix_data.get("old_value"),
                    new_value=fix_data.get("new_value"),
                    confidence=confidence,
                    explanation=fix_data.get("explanation", ""),
                    requires_review=confidence < self.auto_heal_threshold,
                )
            )

        # Sort by confidence
        fixes.sort(key=lambda f: f.confidence, reverse=True)
        return fixes

    def _apply_fix(self, test_spec: dict, fix: FixSuggestion) -> dict:
        """Apply a fix to a test specification."""
        import copy

        healed_spec = copy.deepcopy(test_spec)

        if fix.fix_type == FixType.UPDATE_SELECTOR:
            # Find and update the selector in steps
            for step in healed_spec.get("steps", []):
                if step.get("target") == fix.old_value:
                    step["target"] = fix.new_value
                    break

        elif fix.fix_type == FixType.ADD_WAIT:
            # Add a wait step before the failing step
            if fix.new_value:
                wait_step = {
                    "action": "wait",
                    "target": fix.new_value,
                    "timeout": 5000,
                }
                # Insert before failing step (would need step index)
                healed_spec.setdefault("steps", []).insert(0, wait_step)

        elif fix.fix_type == FixType.INCREASE_TIMEOUT:
            # Increase timeout for the failing step
            for step in healed_spec.get("steps", []):
                if step.get("target") == fix.old_value or step.get("timeout"):
                    step["timeout"] = int(fix.new_value or 10000)

        elif fix.fix_type == FixType.UPDATE_ASSERTION:
            # Update assertion expected value
            for assertion in healed_spec.get("assertions", []):
                if assertion.get("expected") == fix.old_value:
                    assertion["expected"] = fix.new_value
                    break

        elif fix.fix_type == FixType.UPDATE_TEST_DATA:
            # Update test data in steps
            for step in healed_spec.get("steps", []):
                if step.get("value") == fix.old_value:
                    step["value"] = fix.new_value
                    break

        # Mark as healed
        healed_spec["_healed"] = True
        healed_spec["_fix_applied"] = fix.to_dict()

        return healed_spec

    async def batch_analyze(
        self,
        failures: list[tuple[dict, dict, Optional[bytes]]],
    ) -> list[AgentResult[HealingResult]]:
        """Analyze multiple failures.

        Args:
            failures: List of (test_spec, failure_details, screenshot) tuples

        Returns:
            List of AgentResult with HealingResult for each failure
        """
        results = []

        for test_spec, failure_details, screenshot in failures:
            result = await self.execute(
                test_spec=test_spec,
                failure_details=failure_details,
                screenshot=screenshot,
            )
            results.append(result)

        return results
