"""QA Engineer meta-orchestrator package.

Provides the QAEngineerAgent that composes the full Argus agent fleet
to deliver end-to-end quality analysis, coverage assessment, gap
identification, and automatic test generation.
"""

from .qa_engineer_agent import (
    AnalysisType,
    CoverageAssessment,
    CoverageGap,
    EffortSize,
    QAAnalysisResult,
    QAEngineerAgent,
    QualityScoreBreakdown,
    Recommendation,
    RecommendationType,
    RiskLevel,
    UserFlow,
)

from .ci_config_generator import (
    CIConfig,
    CIConfigGenerator,
    GeneratedCIConfig,
)

from .coverage_assessor import (
    CoverageAssessor,
    CoverageAssessment as AssessorCoverageAssessment,
    CoverageGap as AssessorCoverageGap,
    CoverageMetrics,
    RiskLevel as AssessorRiskLevel,
)

__all__ = [
    # QA Engineer Agent
    "AnalysisType",
    "CoverageAssessment",
    "CoverageGap",
    "EffortSize",
    "QAAnalysisResult",
    "QAEngineerAgent",
    "QualityScoreBreakdown",
    "Recommendation",
    "RecommendationType",
    "RiskLevel",
    "UserFlow",
    # CI Config Generator
    "CIConfig",
    "CIConfigGenerator",
    "GeneratedCIConfig",
    # Coverage Assessor
    "CoverageAssessor",
    "AssessorCoverageAssessment",
    "AssessorCoverageGap",
    "CoverageMetrics",
    "AssessorRiskLevel",
]
