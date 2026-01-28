"""Services module for external integrations."""

from src.services.ai_cost_tracker import (
    MODEL_PRICING,
    AICostTracker,
    BudgetStatus,
    TaskType,
    UsageRecord,
    calculate_ai_cost,
    check_ai_budget,
    get_cost_tracker,
    record_ai_usage,
)
from src.services.cicd_monitor import (
    CICDMonitor,
    FileChange,
    MonitoredProject,
    MRInfo,
    PRInfo,
    PRState,
    VCSPlatform,
    get_cicd_monitor,
    start_cicd_monitoring,
    stop_cicd_monitoring,
)
from src.services.dependency_analyzer import (
    CodeModule,
    ComponentNode,
    DependencyAnalyzer,
    ImpactResult,
    RouteNode,
    get_dependency_analyzer,
)
from src.services.email_service import (
    ConsoleEmailProvider,
    EmailProvider,
    EmailService,
    ResendEmailProvider,
    SendGridEmailProvider,
    SMTPEmailProvider,
    get_email_service,
)
from src.services.event_gateway import (
    ArgusEvent,
    EventGateway,
    EventType,
    emit_codebase_ingested,
    emit_healing_completed,
    emit_healing_requested,
    emit_test_executed,
    emit_test_failed,
    get_event_gateway,
)
from src.services.git_analyzer import (
    CodeChange,
    GitAnalyzer,
    GitBlameResult,
    GitCommit,
    SelectorChange,
    get_git_analyzer,
)
from src.services.infra_optimizer import (
    AIInfraOptimizer,
    Anomaly,
    CostReport,
    DemandForecast,
    InfraRecommendation,
    RecommendationPriority,
    RecommendationType,
    create_infra_optimizer,
)
from src.services.prometheus_collector import (
    BrowserNodeMetrics,
    InfrastructureSnapshot,
    PrometheusCollector,
    SeleniumMetrics,
    create_prometheus_collector,
)
from src.services.source_analyzer import (
    ComponentInfo,
    ExtractedSelector,
    SelectorMapping,
    SourceAnalyzer,
    get_source_analyzer,
)
from src.services.supabase_client import SupabaseClient, get_supabase_client

# Cloudflare Storage (RAP-248 Enhanced)
from src.services.cloudflare_storage import (
    CloudflareClient,
    CloudflareConfig,
    VectorizeIndexType,
    VectorizeMemory,
    VectorSearchResult,
    get_cloudflare_client,
    is_cloudflare_configured,
)

__all__ = [
    # Supabase
    "SupabaseClient",
    "get_supabase_client",
    # AI Cost Tracking
    "AICostTracker",
    "get_cost_tracker",
    "record_ai_usage",
    "check_ai_budget",
    "calculate_ai_cost",
    "TaskType",
    "UsageRecord",
    "BudgetStatus",
    "MODEL_PRICING",
    # Email Service
    "EmailService",
    "EmailProvider",
    "ConsoleEmailProvider",
    "ResendEmailProvider",
    "SendGridEmailProvider",
    "SMTPEmailProvider",
    "get_email_service",
    # Prometheus Metrics
    "PrometheusCollector",
    "create_prometheus_collector",
    "InfrastructureSnapshot",
    "SeleniumMetrics",
    "BrowserNodeMetrics",
    # Infrastructure Optimizer
    "AIInfraOptimizer",
    "create_infra_optimizer",
    "InfraRecommendation",
    "CostReport",
    "DemandForecast",
    "Anomaly",
    "RecommendationType",
    "RecommendationPriority",
    # Event Gateway
    "EventGateway",
    "EventType",
    "ArgusEvent",
    "get_event_gateway",
    "emit_codebase_ingested",
    "emit_test_executed",
    "emit_test_failed",
    "emit_healing_requested",
    "emit_healing_completed",
    # Dependency Analyzer
    "DependencyAnalyzer",
    "get_dependency_analyzer",
    "CodeModule",
    "ComponentNode",
    "RouteNode",
    "ImpactResult",
    # Git Analyzer
    "GitAnalyzer",
    "get_git_analyzer",
    "GitCommit",
    "GitBlameResult",
    "SelectorChange",
    "CodeChange",
    # Source Analyzer
    "SourceAnalyzer",
    "get_source_analyzer",
    "ExtractedSelector",
    "ComponentInfo",
    "SelectorMapping",
    # CI/CD Monitor
    "CICDMonitor",
    "get_cicd_monitor",
    "start_cicd_monitoring",
    "stop_cicd_monitoring",
    "PRInfo",
    "MRInfo",
    "FileChange",
    "MonitoredProject",
    "VCSPlatform",
    "PRState",
    # Cloudflare Storage (RAP-248 Enhanced)
    "CloudflareClient",
    "CloudflareConfig",
    "VectorizeIndexType",
    "VectorizeMemory",
    "VectorSearchResult",
    "get_cloudflare_client",
    "is_cloudflare_configured",
]
