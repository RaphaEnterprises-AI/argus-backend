"""Integrations module for external services.

Provides:
- GitHub PR comments
- Slack notifications (basic and enhanced)
- Report generation and export
- Observability platform integrations (Datadog, Sentry, New Relic, etc.)
- AI synthesis layer for production intelligence
"""

from .ai_synthesis import (
    ActionPriority,
    AISynthesizer,
    CoverageGap,
    ErrorInsight,
    FailurePrediction,
    InsightType,
    SynthesisReport,
    TestSuggestion,
    create_ai_synthesizer,
)
from .github_integration import GitHubIntegration, create_github_integration
from .observability_hub import (
    DatadogProvider,
    FullStoryProvider,
    NewRelicProvider,
    ObservabilityHub,
    ObservabilityProvider,
    PerformanceAnomaly,
    Platform,
    PostHogProvider,
    ProductionError,
    RealUserSession,
    SentryProvider,
    UserJourneyPattern,
)
from .reporter import ReportGenerator, create_reporter
from .slack import (
    FailureDetails,
    NotificationStatus,
    QualityReport,
    ScheduleInfo,
    SlackConfig,
    SlackNotifier,
    TestResult,
    create_slack_notifier,
)
from .slack_integration import SlackIntegration, create_slack_integration

__all__ = [
    # Existing integrations
    "GitHubIntegration",
    "create_github_integration",
    "SlackIntegration",
    "create_slack_integration",
    # Enhanced Slack notifications
    "SlackNotifier",
    "SlackConfig",
    "TestResult",
    "FailureDetails",
    "ScheduleInfo",
    "QualityReport",
    "NotificationStatus",
    "create_slack_notifier",
    # Reporting
    "ReportGenerator",
    "create_reporter",
    # Observability integrations
    "ObservabilityHub",
    "ObservabilityProvider",
    "Platform",
    "RealUserSession",
    "ProductionError",
    "PerformanceAnomaly",
    "UserJourneyPattern",
    "DatadogProvider",
    "SentryProvider",
    "NewRelicProvider",
    "FullStoryProvider",
    "PostHogProvider",
    # AI Synthesis
    "AISynthesizer",
    "SynthesisReport",
    "TestSuggestion",
    "ErrorInsight",
    "FailurePrediction",
    "CoverageGap",
    "InsightType",
    "ActionPriority",
    "create_ai_synthesizer",
]
