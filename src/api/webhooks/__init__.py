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

from fastapi import APIRouter

from src.api.webhooks.github import router as github_webhook_router
from src.api.webhooks.gitlab import router as gitlab_webhook_router

# Combined router that includes both GitHub and GitLab webhook handlers
router = APIRouter(tags=["VCS Webhooks"])
router.include_router(github_webhook_router)
router.include_router(gitlab_webhook_router)

__all__ = ["router", "github_webhook_router", "gitlab_webhook_router"]
