"""Services module for external integrations."""

from src.services.supabase_client import SupabaseClient, get_supabase_client
from src.services.ai_cost_tracker import (
    AICostTracker,
    get_cost_tracker,
    record_ai_usage,
    check_ai_budget,
    calculate_ai_cost,
    TaskType,
    UsageRecord,
    BudgetStatus,
    MODEL_PRICING,
)
from src.services.email_service import (
    EmailService,
    EmailProvider,
    ConsoleEmailProvider,
    ResendEmailProvider,
    SendGridEmailProvider,
    SMTPEmailProvider,
    get_email_service,
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
]
