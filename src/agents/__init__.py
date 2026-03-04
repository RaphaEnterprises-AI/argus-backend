"""Agent implementations for E2E testing.

This module provides specialized agents that work together to analyze codebases,
plan tests, execute them, and self-heal failures.

Architecture:
    BaseAgent -> Specialized Agents -> Orchestrator Node Integration

Competitive Features:
    - Visual AI: Screenshot comparison using Claude Vision (like Applitools)
    - NLP Test Creator: Plain English to test conversion (like testRigor)
    - Auto-Discovery: Autonomous app crawling and test generation (like Octomind)
    - Performance Analyzer: Core Web Vitals & Lighthouse-style metrics
    - Security Scanner: OWASP Top 10 vulnerability detection
    - Accessibility Checker: WCAG 2.1 compliance testing
"""

from .accessibility_checker import (
    AccessibilityCheckerAgent,
    AccessibilityCheckResult,
    AccessibilityIssue,
    IssueImpact,
    WCAGLevel,
)
from .api_tester import APITesterAgent
from .auto_discovery import AutoDiscovery, DiscoveryResult, QuickDiscover
from .base import AgentCapability, AgentResult, AICapability, AIResponse, BaseAgent, UsageStats

# Agent-as-Judge + MetaJudge (RAP-255)
from .agent_judge import (
    AgentAsJudge,
    AgentEvaluation,
    create_agent_judge,
    create_meta_judge,
    DebateRound,
    EvaluationCategory,
    EvaluationIssue,
    evaluate_agent_output_simple,
    ExecutionEvaluation,
    HealingValidation,
    IssueSeverity,
    JudgeCapability,
    MetaEvaluation,
    MetaJudge,
    TestValidation,
    VerificationMethod,
)

from .code_analyzer import CodeAnalyzerAgent

# Corrective RAG (CRAG) - Self-correcting retrieval (RAP-254)
from .corrective_rag_agent import (
    AdaptiveRAGRouter,
    AdaptiveRAGStrategy,
    CorrectiveRAGAgent,
    CRAGResult,
    RelevanceLevel,
    RetrievalAction,
    RetrievedDocument,
)

# Hallucination Detection - SelfCheckGPT pattern (RAP-330)
from .hallucination_detector import (
    create_hallucination_detector,
    create_hallucination_middleware,
    GroundingCheckResult,
    HallucinationCheckMiddleware,
    HallucinationDetectorAgent,
    HallucinationResult,
    HallucinationSeverity,
    HallucinationType,
    quick_hallucination_check,
    SentenceScore,
)
from .db_tester import DBTesterAgent

# Flaky Test Detection
from .flaky_detector import (
    FlakinessLevel,
    FlakinessReport,
    FlakyCause,
    FlakyTestDetector,
    QuarantineConfig,
    SmartRetryStrategy,
    TestRun,
)

# MR/PR Analyzer (RAP-234)
from .mr_analyzer import (
    ChangeAnalysis,
    ChangeType,
    MRAnalysis,
    MRAnalyzerAgent,
    TestPriority,
    TestSuggestion,
    TestType,
    create_mr_analyzer,
)
from .nlp_test_creator import ConversationalTestBuilder, GeneratedTest, NLPTestCreator

# Tool Discovery and Creation (ToolLLM + CREATOR patterns)
from .tool_discovery_agent import (
    AuthType,
    CapabilityGap,
    DFSDTToolSelector,
    DiscoveredTool,
    GeneratedTool,
    IntegrationResult,
    ParameterType,
    ToolCapability,
    ToolCreatorAgent,
    ToolDesign,
    ToolDiscoveryAgent,
    ToolLearning,
    ToolParameter,
    ToolRegistry,
    ToolReturn,
    ToolType,
    ToolVerification,
    create_tool_creator_agent,
    create_tool_discovery_agent,
    get_tool_registry,
)

# New Advanced Agents
from .performance_analyzer import (
    CoreWebVitals,
    PerformanceAnalysisResult,
    PerformanceAnalyzerAgent,
    PerformanceGrade,
    PerformanceMetrics,
)

# Enhanced System Prompts
from .prompts import ENHANCED_PROMPTS, get_enhanced_prompt
from .reporter import ReporterAgent

# Intelligent Model Routing
from .router_agent import (
    RouterAgent,
    RouterAgentConfig,
    RoutingDecision,
    TaskContext,
    route_with_agent,
)
from .security_scanner import (
    SecurityScannerAgent,
    SecurityScanResult,
    Vulnerability,
    VulnerabilityCategory,
    VulnerabilitySeverity,
)

# Penetration Testing Agents
from .recon_agent import ReconAgent
from .vuln_scanner_agent import VulnScannerAgent
from .exploitation_agent import ExploitationAgent

from .self_healer import SelfHealerAgent

# Test Quality Intelligence
from .test_quality_agent import (
    SuiteAudit,
    TestQualityIntelligenceAgent,
    TrustScore,
)

# QA Intelligence Agents
from .testgen_agent import TestGenAgent
from .testhealer_cicd_agent import TestHealerCICDAgent
from .qa_engineer import QAEngineerAgent

# SRE Agent - Unified incident correlation, remediation, optimization (RAP-253)
from .sre_agent import (
    CorrelatedIncident,
    IncidentSeverity,
    RemediationStatus,
    Runbook,
    SREAgent,
    SREMetrics,
)

# Test Impact Analysis
from .test_impact_analyzer import (
    ChangeRisk,
    CodeChange,
    ImpactAnalysis,
    SmartTestSelector,
    TestImpactAnalyzer,
)
from .test_planner import TestPlannerAgent

from .ui_tester_v2 import AssertionResult, StepResult, UITesterAgent, UITesterAgentV2, UITestResult

# Competitive Features
from .visual_ai import VisualAI, VisualComparisonResult, VisualRegressionManager

__all__ = [
    # Base
    "BaseAgent",
    "AgentCapability",
    "AgentResult",
    "AICapability",
    "AIResponse",
    "UsageStats",
    # Specialized Agents
    "CodeAnalyzerAgent",
    "TestPlannerAgent",
    "UITesterAgent",
    "UITesterAgentV2",
    "UITestResult",
    "StepResult",
    "AssertionResult",
    "APITesterAgent",
    "DBTesterAgent",
    "SelfHealerAgent",
    "ReporterAgent",
    # Competitive Features - Visual AI (like Applitools)
    "VisualAI",
    "VisualRegressionManager",
    "VisualComparisonResult",
    # Competitive Features - NLP Test Creator (like testRigor)
    "NLPTestCreator",
    "ConversationalTestBuilder",
    "GeneratedTest",
    # Competitive Features - Auto-Discovery (like Octomind)
    "AutoDiscovery",
    "QuickDiscover",
    "DiscoveryResult",
    # Performance Analyzer (like Lighthouse)
    "PerformanceAnalyzerAgent",
    "PerformanceAnalysisResult",
    "PerformanceMetrics",
    "CoreWebVitals",
    "PerformanceGrade",
    # Security Scanner (like OWASP ZAP)
    "SecurityScannerAgent",
    "SecurityScanResult",
    "Vulnerability",
    "VulnerabilitySeverity",
    "VulnerabilityCategory",
    # Accessibility Checker (like axe-core)
    "AccessibilityCheckerAgent",
    "AccessibilityCheckResult",
    "AccessibilityIssue",
    "WCAGLevel",
    "IssueImpact",
    # Enhanced Prompts
    "ENHANCED_PROMPTS",
    "get_enhanced_prompt",
    # Intelligent Model Routing
    "RouterAgent",
    "RouterAgentConfig",
    "RoutingDecision",
    "TaskContext",
    "route_with_agent",
    # Flaky Test Detection
    "FlakyTestDetector",
    "FlakinessReport",
    "FlakinessLevel",
    "FlakyCause",
    "QuarantineConfig",
    "SmartRetryStrategy",
    "TestRun",
    # Test Impact Analysis
    "TestImpactAnalyzer",
    "ImpactAnalysis",
    "CodeChange",
    "ChangeRisk",
    "SmartTestSelector",
    # MR/PR Analyzer (RAP-234)
    "MRAnalyzerAgent",
    "MRAnalysis",
    "TestSuggestion",
    "ChangeAnalysis",
    "ChangeType",
    "TestPriority",
    "TestType",
    "create_mr_analyzer",
    # Tool Discovery and Creation (ToolLLM + CREATOR patterns)
    "ToolDiscoveryAgent",
    "ToolCreatorAgent",
    "ToolRegistry",
    "DFSDTToolSelector",
    "ToolCapability",
    "ToolType",
    "ParameterType",
    "AuthType",
    "ToolParameter",
    "ToolReturn",
    "DiscoveredTool",
    "GeneratedTool",
    "ToolVerification",
    "IntegrationResult",
    "ToolLearning",
    "CapabilityGap",
    "ToolDesign",
    "create_tool_discovery_agent",
    "create_tool_creator_agent",
    "get_tool_registry",
    # SRE Agent (RAP-253) - Unified incident response
    "SREAgent",
    "CorrelatedIncident",
    "IncidentSeverity",
    "RemediationStatus",
    "Runbook",
    "SREMetrics",
    # Corrective RAG Agent (RAP-254) - Self-correcting retrieval
    "CorrectiveRAGAgent",
    "CRAGResult",
    "RetrievedDocument",
    "RelevanceLevel",
    "RetrievalAction",
    "AdaptiveRAGRouter",
    "AdaptiveRAGStrategy",
    # Penetration Testing Agents
    "ReconAgent",
    "VulnScannerAgent",
    "ExploitationAgent",
    # Test Quality Intelligence
    "TestQualityIntelligenceAgent",
    "TrustScore",
    "SuiteAudit",
    # Hallucination Detection (RAP-330) - SelfCheckGPT pattern
    "HallucinationDetectorAgent",
    "HallucinationResult",
    "HallucinationSeverity",
    "HallucinationType",
    "SentenceScore",
    "GroundingCheckResult",
    "HallucinationCheckMiddleware",
    "create_hallucination_detector",
    "create_hallucination_middleware",
    "quick_hallucination_check",
    # QA Intelligence Agents
    "TestGenAgent",
    "TestHealerCICDAgent",
    "QAEngineerAgent",
]
