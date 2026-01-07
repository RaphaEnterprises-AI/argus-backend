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

from .base import BaseAgent, AgentResult
from .code_analyzer import CodeAnalyzerAgent
from .test_planner import TestPlannerAgent
from .ui_tester import UITesterAgent
from .api_tester import APITesterAgent
from .db_tester import DBTesterAgent
from .self_healer import SelfHealerAgent
from .reporter import ReporterAgent

# Competitive Features
from .visual_ai import VisualAI, VisualRegressionManager, VisualComparisonResult
from .nlp_test_creator import NLPTestCreator, ConversationalTestBuilder, GeneratedTest
from .auto_discovery import AutoDiscovery, QuickDiscover, DiscoveryResult

# New Advanced Agents
from .performance_analyzer import (
    PerformanceAnalyzerAgent,
    PerformanceAnalysisResult,
    PerformanceMetrics,
    CoreWebVitals,
    PerformanceGrade,
)
from .security_scanner import (
    SecurityScannerAgent,
    SecurityScanResult,
    Vulnerability,
    VulnerabilitySeverity,
    VulnerabilityCategory,
)
from .accessibility_checker import (
    AccessibilityCheckerAgent,
    AccessibilityCheckResult,
    AccessibilityIssue,
    WCAGLevel,
    IssueImpact,
)

# Enhanced System Prompts
from .prompts import ENHANCED_PROMPTS, get_enhanced_prompt

# Intelligent Model Routing
from .router_agent import (
    RouterAgent,
    RouterAgentConfig,
    RoutingDecision,
    TaskContext,
    route_with_agent,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    # Specialized Agents
    "CodeAnalyzerAgent",
    "TestPlannerAgent",
    "UITesterAgent",
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
]
