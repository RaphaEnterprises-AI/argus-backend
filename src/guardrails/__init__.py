"""
Enterprise Guardrails for Multi-Agent Systems.

Implements a 5-layer guardrail architecture:
L1: Identity & Access Control
L2: Input Validation (prompt injection, sanitization)
L3: Execution Boundaries (rate limits, quotas, sandboxing)
L4: Output Filtering (content policy, PII, hallucination)
L5: Human-in-the-Loop (risk-tiered approvals)
"""

from .stack import (
    GuardrailResult,
    GuardrailStack,
    RiskTier,
    ActionClassifier,
)
from .validators import (
    PromptInjectionDetector,
    InputSanitizer,
    OutputFilter,
)

__all__ = [
    "GuardrailResult",
    "GuardrailStack",
    "RiskTier",
    "ActionClassifier",
    "PromptInjectionDetector",
    "InputSanitizer",
    "OutputFilter",
]
