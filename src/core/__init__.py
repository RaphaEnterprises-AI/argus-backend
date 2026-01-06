"""Core cognitive engine for intelligent testing and quality analysis."""

from .cognitive_engine import (
    CognitiveTestingEngine,
    ApplicationModel,
    CognitiveInsight,
    UnderstandingLevel,
)
from .model_router import (
    ModelRouter,
    TaskType,
    TaskComplexity,
    TASK_MODEL_MAPPING,
    BudgetExceededError,
)
from .normalizer import (
    EventNormalizer,
    NormalizedEvent,
    StackFrame,
    EventSource,
    EventType,
    Severity,
)
from .correlator import (
    ErrorCorrelator,
    Correlation,
    CorrelationType,
    ConfidenceLevel,
    CodeLocation,
    ErrorPattern,
)
from .coverage import (
    CoverageAnalyzer,
    CoverageSummary,
    FileCoverage,
    CoverageGap,
    CoverageLevel,
    GapPriority,
)
from .risk import (
    RiskScorer,
    FileRisk,
    ProjectRisk,
    RiskFactor,
    RiskLevel,
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
]
