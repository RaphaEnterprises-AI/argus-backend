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
from .base import AgentResult, BaseAgent
from .code_analyzer import CodeAnalyzerAgent
from .db_tester import DBTesterAgent
from .nlp_test_creator import ConversationalTestBuilder, GeneratedTest, NLPTestCreator

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
from .self_healer import SelfHealerAgent
from .test_planner import TestPlannerAgent

# Legacy UI Tester - Available as UITesterAgentLegacy for migration
from .ui_tester import UITesterAgent as UITesterAgentLegacy

# UI Tester V2 - Uses unified BrowserPoolClient (Vultr K8s)
from .ui_tester_v2 import AssertionResult, StepResult, UITesterAgent, UITesterAgentV2, UITestResult

# Competitive Features
from .visual_ai import VisualAI, VisualComparisonResult, VisualRegressionManager

# Flaky Test Detection
from .flaky_detector import (
    FlakyTestDetector,
    FlakinessReport,
    FlakinessLevel,
    FlakyCause,
    QuarantineConfig,
    SmartRetryStrategy,
    TestRun,
)

# Test Impact Analysis
from .test_impact_analyzer import (
    TestImpactAnalyzer,
    ImpactAnalysis,
    CodeChange,
    ChangeRisk,
    SmartTestSelector,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    # Specialized Agents
    "CodeAnalyzerAgent",
    "TestPlannerAgent",
    "UITesterAgent",  # V2 (uses BrowserPoolClient)
    "UITesterAgentV2",  # Explicit V2 reference
    "UITesterAgentLegacy",  # Legacy (uses E2EBrowserClient)
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
]
