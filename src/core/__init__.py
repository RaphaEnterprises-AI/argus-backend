"""Core cognitive engine for intelligent testing and quality analysis."""

from .cognitive_engine import (
    ApplicationModel,
    CognitiveInsight,
    CognitiveTestingEngine,
    UnderstandingLevel,
)
from .correlator import (
    CodeLocation,
    ConfidenceLevel,
    Correlation,
    CorrelationType,
    ErrorCorrelator,
    ErrorPattern,
)
from .coverage import (
    CoverageAnalyzer,
    CoverageGap,
    CoverageLevel,
    CoverageSummary,
    FileCoverage,
    GapPriority,
)
from .model_router import (
    TASK_MODEL_MAPPING,
    BudgetExceededError,
    ModelRouter,
    TaskComplexity,
    TaskType,
)
from .normalizer import (
    EventNormalizer,
    EventSource,
    EventType,
    NormalizedEvent,
    Severity,
    StackFrame,
)
from .risk import (
    FileRisk,
    ProjectRisk,
    RiskFactor,
    RiskLevel,
    RiskScorer,
)
from .tenant import (
    TenantContext,
    get_current_tenant,
    set_current_tenant,
    require_tenant,
)

__all__ = [
    # Cognitive Engine
    "CognitiveTestingEngine",
    "ApplicationModel",
    "CognitiveInsight",
    "UnderstandingLevel",
    # Model Router
    "ModelRouter",
    "TaskType",
    "TaskComplexity",
    "TASK_MODEL_MAPPING",
    "BudgetExceededError",
    # Event Normalizer
    "EventNormalizer",
    "NormalizedEvent",
    "StackFrame",
    "EventSource",
    "EventType",
    "Severity",
    # Error Correlator
    "ErrorCorrelator",
    "Correlation",
    "CorrelationType",
    "ConfidenceLevel",
    "CodeLocation",
    "ErrorPattern",
    # Coverage Analyzer
    "CoverageAnalyzer",
    "CoverageSummary",
    "FileCoverage",
    "CoverageGap",
    "CoverageLevel",
    "GapPriority",
    # Risk Scorer
    "RiskScorer",
    "FileRisk",
    "ProjectRisk",
    "RiskFactor",
    "RiskLevel",
    # Tenant Context
    "TenantContext",
    "get_current_tenant",
    "set_current_tenant",
    "require_tenant",
]
