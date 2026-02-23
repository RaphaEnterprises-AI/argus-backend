"""
Heal Validation Pipeline — validates whether a self-heal is trustworthy.

Two classes:
- AssertionAnalyzer: Rule-based, zero LLM cost assertion strength analysis
- HealValidationService: Orchestrates the validation gate with AgentAsJudge

This is the core of the "Is this self-heal trustworthy?" feature that
no competitor provides. It catches:
- Assertion weakening (heal silently drops assertion specificity)
- Regression masking (heal fixes test but masks a real application bug)
- Low-confidence heals that need human review
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Types
# =============================================================================


class HealQuality(str, Enum):
    """Quality classification of a heal attempt."""

    VALID_HEAL = "valid_heal"
    SUSPICIOUS_HEAL = "suspicious_heal"
    REGRESSION_MASKING = "regression_masking"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class AssertionStrength(str, Enum):
    """Strength of a test assertion, from strongest to weakest."""

    EXACT_MATCH = "exact_match"        # assertEquals, toBe, toEqual
    CONTAINS = "contains"              # assertContains, toContain
    EXISTS = "exists"                  # assertTrue(exists), toBeVisible
    NO_ASSERTION = "no_assertion"      # No meaningful assertion

    @property
    def score(self) -> float:
        """Numeric score for comparison (higher = stronger)."""
        return {
            AssertionStrength.EXACT_MATCH: 1.0,
            AssertionStrength.CONTAINS: 0.7,
            AssertionStrength.EXISTS: 0.4,
            AssertionStrength.NO_ASSERTION: 0.0,
        }[self]


@dataclass
class AssertionAnalysis:
    """Result of analyzing assertion strength in a test spec."""

    strength: AssertionStrength
    assertion_count: int = 0
    operators_found: list[str] = field(default_factory=list)


@dataclass
class HealValidationResult:
    """Complete result of validating a heal attempt."""

    heal_quality: HealQuality
    blocking: bool  # If True, heal should NOT be auto-applied
    auto_apply_allowed: bool

    # Assertion analysis (rule-based, free)
    assertion_weakening_detected: bool = False
    original_assertion_strength: AssertionStrength | None = None
    healed_assertion_strength: AssertionStrength | None = None

    # AgentAsJudge signals (LLM-based)
    addresses_root_cause: bool | None = None
    preserves_test_intent: bool | None = None
    has_side_effects: bool | None = None
    judge_confidence: float = 0.0
    explanation: str = ""
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heal_quality": self.heal_quality.value,
            "blocking": self.blocking,
            "auto_apply_allowed": self.auto_apply_allowed,
            "assertion_weakening_detected": self.assertion_weakening_detected,
            "original_assertion_strength": (
                self.original_assertion_strength.value
                if self.original_assertion_strength
                else None
            ),
            "healed_assertion_strength": (
                self.healed_assertion_strength.value
                if self.healed_assertion_strength
                else None
            ),
            "addresses_root_cause": self.addresses_root_cause,
            "preserves_test_intent": self.preserves_test_intent,
            "has_side_effects": self.has_side_effects,
            "judge_confidence": self.judge_confidence,
            "explanation": self.explanation,
            "alternatives": self.alternatives,
        }


# =============================================================================
# AssertionAnalyzer — Rule-based, zero LLM cost
# =============================================================================

# Operators that indicate assertion strength
_EXACT_MATCH_OPERATORS = {
    "equals", "assertEqual", "toBe", "toEqual", "toStrictEqual",
    "eq", "exact_match", "===", "==",
}
_CONTAINS_OPERATORS = {
    "contains", "assertContains", "toContain", "toMatch",
    "includes", "has_text", "containsText", "matches",
}
_EXISTS_OPERATORS = {
    "exists", "visible", "toBeVisible", "isDisplayed",
    "toExist", "toBeTruthy", "present", "displayed",
}


class AssertionAnalyzer:
    """Rule-based assertion strength classifier. Zero LLM cost.

    Inspects test spec assertions to classify their strength:
    - EXACT_MATCH: Verifies exact values (strongest)
    - CONTAINS: Verifies partial content
    - EXISTS: Only checks element presence (weakest meaningful)
    - NO_ASSERTION: No meaningful verification
    """

    def analyze_assertion_strength(self, test_spec: dict) -> AssertionAnalysis:
        """Classify the overall assertion strength of a test spec.

        Args:
            test_spec: Test specification dict with 'assertions' and/or 'steps'

        Returns:
            AssertionAnalysis with strength classification
        """
        assertions = test_spec.get("assertions", [])
        steps = test_spec.get("steps", [])
        operators_found: list[str] = []
        max_strength = AssertionStrength.NO_ASSERTION

        # Check explicit assertions
        for assertion in assertions:
            operator = assertion.get("operator", "").lower()
            action = assertion.get("action", "").lower()
            key = operator or action

            if not key:
                continue

            operators_found.append(key)
            strength = self._classify_operator(key)
            if strength.score > max_strength.score:
                max_strength = strength

        # Check assertion-like steps (e.g., verify, assert, check)
        for step in steps:
            action = step.get("action", "").lower()
            if action in ("assert", "verify", "check", "validate", "expect"):
                operator = step.get("operator", "").lower()
                key = operator or action
                operators_found.append(key)
                strength = self._classify_operator(key)
                if strength.score > max_strength.score:
                    max_strength = strength

            # Steps with expected values imply assertions
            if step.get("expected"):
                operators_found.append("implicit_expected")
                if max_strength.score < AssertionStrength.CONTAINS.score:
                    max_strength = AssertionStrength.CONTAINS

        return AssertionAnalysis(
            strength=max_strength,
            assertion_count=len(assertions) + sum(
                1 for s in steps
                if s.get("action", "").lower()
                in ("assert", "verify", "check", "validate", "expect")
            ),
            operators_found=operators_found,
        )

    def detect_weakening(
        self,
        original: AssertionAnalysis,
        healed: AssertionAnalysis,
    ) -> bool:
        """Detect if healing weakened assertion strength.

        Args:
            original: Analysis of original test spec
            healed: Analysis of healed test spec

        Returns:
            True if healed assertion is weaker than original
        """
        return healed.strength.score < original.strength.score

    def _classify_operator(self, operator: str) -> AssertionStrength:
        """Classify a single operator/action into assertion strength."""
        op = operator.lower().replace("-", "_").replace(" ", "_")

        if op in _EXACT_MATCH_OPERATORS or any(em in op for em in ("equal", "exact", "tobe", "strict")):
            return AssertionStrength.EXACT_MATCH
        if op in _CONTAINS_OPERATORS or any(c in op for c in ("contain", "match", "include", "has_text")):
            return AssertionStrength.CONTAINS
        if op in _EXISTS_OPERATORS or any(e in op for e in ("exist", "visible", "present", "display")):
            return AssertionStrength.EXISTS

        return AssertionStrength.NO_ASSERTION


# =============================================================================
# HealValidationService — Orchestrates the validation gate
# =============================================================================


class HealValidationService:
    """Orchestrates heal validation using rule-based analysis + AgentAsJudge.

    Validation pipeline:
    1. Assertion analysis (rule-based, free)
    2. REAL_BUG fast-path (free)
    3. AgentAsJudge verification (LLM, ~150-400 tokens)
    4. Classify HealQuality from combined signals
    5. Persist to heal_validations table (fire-and-forget)

    This is a service (not a BaseAgent) because it must be called synchronously
    inside SelfHealerAgent.execute() — making it an agent would add A2A/Kafka
    round-trip latency.
    """

    def __init__(
        self,
        enable_judge: bool = True,
        judge_confidence_threshold: float = 0.6,
    ):
        self._assertion_analyzer = AssertionAnalyzer()
        self._enable_judge = enable_judge
        self._judge_confidence_threshold = judge_confidence_threshold

    async def validate(
        self,
        test_id: str,
        org_id: str,
        project_id: str,
        original_test_spec: dict,
        healed_test_spec: dict | None,
        diagnosis_type: str,
        healing_method: str,
        original_selector: str | None = None,
        healed_selector: str | None = None,
        error_message: str | None = None,
        failure_id: str | None = None,
        healing_request_id: str | None = None,
    ) -> HealValidationResult:
        """Validate a heal attempt and classify its quality.

        Args:
            test_id: ID of the test being healed
            org_id: Organization ID
            project_id: Project ID
            original_test_spec: Original test specification
            healed_test_spec: Healed test specification (may be None if heal failed)
            diagnosis_type: Type of failure diagnosis (e.g., 'selector_changed', 'real_bug')
            healing_method: Method used (cache, cognee, code_aware, llm_analysis)
            original_selector: Original failed selector
            healed_selector: New selector after healing
            error_message: Original error message
            failure_id: Failure record ID
            healing_request_id: Healing request event ID

        Returns:
            HealValidationResult with quality classification
        """

        # --- Step 1: Assertion analysis (rule-based, free) ---
        original_analysis = self._assertion_analyzer.analyze_assertion_strength(
            original_test_spec
        )
        healed_analysis = (
            self._assertion_analyzer.analyze_assertion_strength(healed_test_spec)
            if healed_test_spec
            else original_analysis
        )
        assertion_weakening = self._assertion_analyzer.detect_weakening(
            original_analysis, healed_analysis
        )

        # --- Step 2: REAL_BUG fast-path (free) ---
        if diagnosis_type == "real_bug":
            result = HealValidationResult(
                heal_quality=HealQuality.REGRESSION_MASKING,
                blocking=True,
                auto_apply_allowed=False,
                assertion_weakening_detected=assertion_weakening,
                original_assertion_strength=original_analysis.strength,
                healed_assertion_strength=healed_analysis.strength,
                explanation="Diagnosis indicates a real application bug — healing would mask a regression",
                judge_confidence=1.0,
            )
            await self._persist(result, test_id, org_id, project_id, failure_id,
                                healing_request_id, healing_method, original_selector,
                                healed_selector, diagnosis_type)
            return result

        # --- Step 3: AgentAsJudge verification (LLM) ---
        judge_result = None
        if self._enable_judge and healed_test_spec:
            judge_result = await self._run_judge(
                original_test_spec=original_test_spec,
                healed_test_spec=healed_test_spec,
                error_message=error_message or "",
                healing_method=healing_method,
                original_selector=original_selector,
                healed_selector=healed_selector,
            )

        # --- Step 4: Classify HealQuality ---
        result = self._classify_quality(
            assertion_weakening=assertion_weakening,
            original_analysis=original_analysis,
            healed_analysis=healed_analysis,
            diagnosis_type=diagnosis_type,
            judge_result=judge_result,
        )

        # --- Step 5: Persist (fire-and-forget) ---
        await self._persist(result, test_id, org_id, project_id, failure_id,
                            healing_request_id, healing_method, original_selector,
                            healed_selector, diagnosis_type)

        return result

    async def _run_judge(
        self,
        original_test_spec: dict,
        healed_test_spec: dict,
        error_message: str,
        healing_method: str,
        original_selector: str | None,
        healed_selector: str | None,
    ) -> dict | None:
        """Run AgentAsJudge.verify_healing_suggestion().

        Returns dict with is_valid, addresses_root_cause, preserves_test_intent,
        has_side_effects, confidence, alternative_suggestions, reasoning.
        """
        try:
            from src.agents.agent_judge import AgentAsJudge

            judge = AgentAsJudge()
            validation = await judge.verify_healing_suggestion(
                original_error=error_message,
                healing_suggestion={
                    "healing_method": healing_method,
                    "original_selector": original_selector,
                    "healed_selector": healed_selector,
                    "healed_spec_preview": {
                        k: v
                        for k, v in (healed_test_spec or {}).items()
                        if k in ("steps", "assertions", "name", "id")
                    },
                },
                test_context={
                    "original_spec_preview": {
                        k: v
                        for k, v in original_test_spec.items()
                        if k in ("steps", "assertions", "name", "id")
                    },
                },
            )
            return validation.to_dict()

        except Exception as e:
            logger.warning("AgentAsJudge verification failed, proceeding without", error=str(e))
            return None

    def _classify_quality(
        self,
        assertion_weakening: bool,
        original_analysis: AssertionAnalysis,
        healed_analysis: AssertionAnalysis,
        diagnosis_type: str,
        judge_result: dict | None,
    ) -> HealValidationResult:
        """Classify HealQuality from combined signals."""

        addresses_root_cause = None
        preserves_test_intent = None
        has_side_effects = None
        judge_confidence = 0.0
        explanation_parts: list[str] = []
        alternatives: list[dict] = []

        if judge_result:
            addresses_root_cause = judge_result.get("addresses_root_cause")
            preserves_test_intent = judge_result.get("preserves_test_intent")
            has_side_effects = judge_result.get("has_side_effects")
            judge_confidence = judge_result.get("confidence", 0.0)
            alternatives = judge_result.get("alternative_suggestions", [])

        # Classification logic
        heal_quality = HealQuality.VALID_HEAL
        blocking = False

        # REGRESSION_MASKING: judge says doesn't address root cause AND
        # failure isn't a simple selector change
        if (
            judge_result
            and not addresses_root_cause
            and diagnosis_type not in ("selector_changed", "selector_renamed", "timing_issue")
        ):
            heal_quality = HealQuality.REGRESSION_MASKING
            blocking = True
            explanation_parts.append(
                "Judge determined heal does not address root cause for a non-trivial failure type"
            )

        # SUSPICIOUS_HEAL: assertion weakening detected
        elif assertion_weakening:
            heal_quality = HealQuality.SUSPICIOUS_HEAL
            blocking = True
            explanation_parts.append(
                f"Assertion strength weakened from {original_analysis.strength.value} "
                f"to {healed_analysis.strength.value}"
            )

        # NEEDS_HUMAN_REVIEW: judge says invalid OR low confidence
        elif judge_result and (
            not judge_result.get("is_valid") or judge_confidence < self._judge_confidence_threshold
        ):
            heal_quality = HealQuality.NEEDS_HUMAN_REVIEW
            blocking = True
            if not judge_result.get("is_valid"):
                explanation_parts.append("Judge marked heal as invalid")
            if judge_confidence < self._judge_confidence_threshold:
                explanation_parts.append(
                    f"Judge confidence {judge_confidence:.0%} below threshold "
                    f"{self._judge_confidence_threshold:.0%}"
                )

        # VALID_HEAL: all checks pass
        else:
            explanation_parts.append("All validation checks passed")

        # Add judge reasoning if available
        if judge_result and judge_result.get("reasoning"):
            explanation_parts.append(f"Judge reasoning: {judge_result['reasoning']}")

        return HealValidationResult(
            heal_quality=heal_quality,
            blocking=blocking,
            auto_apply_allowed=not blocking,
            assertion_weakening_detected=assertion_weakening,
            original_assertion_strength=original_analysis.strength,
            healed_assertion_strength=healed_analysis.strength,
            addresses_root_cause=addresses_root_cause,
            preserves_test_intent=preserves_test_intent,
            has_side_effects=has_side_effects,
            judge_confidence=judge_confidence,
            explanation="; ".join(explanation_parts),
            alternatives=alternatives,
        )

    async def _persist(
        self,
        result: HealValidationResult,
        test_id: str,
        org_id: str,
        project_id: str,
        failure_id: str | None,
        healing_request_id: str | None,
        healing_method: str,
        original_selector: str | None,
        healed_selector: str | None,
        diagnosis_type: str,
    ) -> None:
        """Persist validation result to heal_validations table (fire-and-forget)."""
        try:
            from src.services.supabase_client import get_supabase_client

            supabase = get_supabase_client()
            await supabase.insert("heal_validations", {
                "organization_id": org_id,
                "project_id": project_id,
                "test_id": test_id,
                "failure_id": failure_id,
                "healing_request_id": healing_request_id,
                "healing_method": healing_method,
                "original_selector": original_selector,
                "healed_selector": healed_selector,
                "diagnosis_type": diagnosis_type,
                "heal_quality": result.heal_quality.value,
                "blocking": result.blocking,
                "auto_apply_allowed": result.auto_apply_allowed,
                "addresses_root_cause": result.addresses_root_cause,
                "preserves_test_intent": result.preserves_test_intent,
                "has_side_effects": result.has_side_effects,
                "judge_confidence": result.judge_confidence,
                "explanation": result.explanation,
                "alternatives": result.alternatives,
                "assertion_weakening_detected": result.assertion_weakening_detected,
                "original_assertion_strength": (
                    result.original_assertion_strength.value
                    if result.original_assertion_strength
                    else None
                ),
                "healed_assertion_strength": (
                    result.healed_assertion_strength.value
                    if result.healed_assertion_strength
                    else None
                ),
            })
            logger.info(
                "Persisted heal validation",
                test_id=test_id,
                heal_quality=result.heal_quality.value,
                blocking=result.blocking,
            )
        except Exception as e:
            logger.warning("Failed to persist heal validation (non-fatal)", error=str(e))
