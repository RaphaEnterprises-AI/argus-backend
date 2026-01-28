"""Webhook handlers for VCS platform integrations.

This package provides webhook handlers for:
- GitHub: PR, push, check_run events
- GitLab: merge_request, push events

Each handler:
1. Verifies webhook signature
2. Parses event payload
3. Emits appropriate internal events
4. Triggers test impact analysis for PRs/MRs
"""

from src.api.webhooks.github import router as github_webhook_router
from src.api.webhooks.gitlab import router as gitlab_webhook_router

__all__ = ["github_webhook_router", "gitlab_webhook_router"]
