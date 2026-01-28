"""Integrations module for external services.

Provides:
- GitHub PR comments
- Slack notifications (basic and enhanced)
- Report generation and export
- Observability platform integrations (Datadog, Sentry, New Relic, etc.)
- AI synthesis layer for production intelligence
- Jira integration for requirements and issue tracking
- LaunchDarkly feature flag awareness
- Amplitude product analytics for test prioritization
- PagerDuty incident tracking and correlation
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
from .amplitude_integration import (
    AmplitudeIntegration,
    EventMetrics,
    EventProperty,
    FunnelStep,
    RetentionData,
    UserPath,
    create_amplitude_integration,
)
from .github_integration import GitHubIntegration, create_github_integration
from .jira_integration import (
    AcceptanceCriteriaExtractor,
    JiraIntegration,
    JiraIssue,
    create_jira_integration,
)
from .launchdarkly_integration import (
    FeatureFlag,
    FlagChange,
    FlagEvaluationContext,
    LaunchDarklyIntegration,
    create_launchdarkly_integration,
)
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
from .pagerduty_integration import (
    IncidentAnalysis,
    IncidentStatus,
    IncidentUrgency,
    PagerDutyIncident,
    PagerDutyIntegration,
    PagerDutyService,
    TimelineEntry,
    create_pagerduty_integration,
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
from .vercel_integration import (
    DeploymentState,
    VercelDeployment,
    VercelIntegration,
    VercelProject,
    create_vercel_integration,
)
from .comment_poster import (
    CommentPoster,
    CommentResult,
    create_comment_poster,
)

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
    # Vercel integration
    "VercelIntegration",
    "VercelDeployment",
    "VercelProject",
    "DeploymentState",
    "create_vercel_integration",
    # Jira integration (Tier 1)
    "JiraIntegration",
    "JiraIssue",
    "AcceptanceCriteriaExtractor",
    "create_jira_integration",
    # LaunchDarkly integration (Tier 2)
    "LaunchDarklyIntegration",
    "FeatureFlag",
    "FlagChange",
    "FlagEvaluationContext",
    "create_launchdarkly_integration",
    # Amplitude integration (Tier 2)
    "AmplitudeIntegration",
    "EventMetrics",
    "EventProperty",
    "FunnelStep",
    "RetentionData",
    "UserPath",
    "create_amplitude_integration",
    # PagerDuty integration (Tier 2)
    "PagerDutyIntegration",
    "PagerDutyIncident",
    "PagerDutyService",
    "IncidentStatus",
    "IncidentUrgency",
    "IncidentAnalysis",
    "TimelineEntry",
    "create_pagerduty_integration",
    # Comment Poster (RAP-234)
    "CommentPoster",
    "CommentResult",
    "create_comment_poster",
]
