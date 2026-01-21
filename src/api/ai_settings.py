"""AI Settings Management API endpoints.

Provides endpoints for:
- Managing AI preferences (default model, cost limits)
- BYOK (Bring Your Own Key) provider key management
- AI usage tracking and billing summary
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.teams import get_current_user
from src.api.users import get_or_create_profile
from src.core.model_registry import ModelRegistry, Provider, get_model_registry
from src.services.cloudflare_key_vault import (
    decrypt_api_key_secure,
    encrypt_api_key_secure,
    get_key_vault_client,
    is_key_vault_available,
)
from src.services.pricing_service import get_pricing_service
from src.services.provider_router import Provider as ProviderEnum
from src.services.provider_router import get_provider_router
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/users/me", tags=["AI Settings"])


# ============================================================================
# Request/Response Models
# ============================================================================


class AIPreferences(BaseModel):
    """User AI preferences."""

    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-5"
    cost_limit_per_day: float = Field(default=10.0, ge=0, le=1000)
    cost_limit_per_message: float = Field(default=1.0, ge=0, le=100)
    use_platform_key_fallback: bool = True
    show_token_costs: bool = True
    show_model_in_chat: bool = True
    preferred_models_by_task: dict[str, str] = Field(default_factory=dict)


class UpdateAIPreferencesRequest(BaseModel):
    """Request to update AI preferences."""

    default_provider: str | None = None
    default_model: str | None = None
    cost_limit_per_day: float | None = Field(None, ge=0, le=1000)
    cost_limit_per_message: float | None = Field(None, ge=0, le=100)
    use_platform_key_fallback: bool | None = None
    show_token_costs: bool | None = None
    show_model_in_chat: bool | None = None
    preferred_models_by_task: dict[str, str] | None = None


class ProviderKeyInfo(BaseModel):
    """Provider key information (masked for display)."""

    id: str
    provider: str
    key_display: str  # "sk-ant-...xyz9"
    is_valid: bool
    last_validated_at: str | None
    validation_error: str | None
    created_at: str


class AddProviderKeyRequest(BaseModel):
    """Request to add a provider API key."""

    provider: str = Field(..., pattern="^(anthropic|openai|google|groq|together)$")
    api_key: str = Field(..., min_length=10, max_length=500)
    display_name: str | None = Field(None, max_length=100)


class ValidateKeyResponse(BaseModel):
    """Response from key validation."""

    is_valid: bool
    error: str | None = None
    provider: str


class UsageRecord(BaseModel):
    """Single AI usage record."""

    id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    key_source: str
    thread_id: str | None
    created_at: str


class UsageSummary(BaseModel):
    """AI usage summary."""

    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    platform_key_cost: float
    byok_cost: float
    usage_by_model: dict[str, dict[str, Any]]
    usage_by_provider: dict[str, dict[str, Any]]


class DailyUsage(BaseModel):
    """Daily usage breakdown."""

    date: str
    total_requests: int
    total_tokens: int
    total_cost_usd: float


class UsageResponse(BaseModel):
    """Full usage response."""

    summary: UsageSummary
    daily: list[DailyUsage]
    records: list[UsageRecord]


class BudgetStatus(BaseModel):
    """User's current budget status."""

    has_budget: bool
    daily_limit: float
    daily_spent: float
    daily_remaining: float
    message_limit: float


class ModelInfo(BaseModel):
    """Model information for selection UI."""

    model_id: str
    display_name: str
    provider: str
    input_price: float  # Per 1M tokens
    output_price: float
    capabilities: list[str]
    is_available: bool  # Based on user's configured keys


class AvailableModelsResponse(BaseModel):
    """Available models grouped by provider."""

    models: list[ModelInfo]
    user_providers: list[str]  # Providers user has keys for


# ============================================================================
# Helper Functions
# ============================================================================


def get_default_ai_preferences() -> dict:
    """Get default AI preferences."""
    return {
        "default_provider": "anthropic",
        "default_model": "claude-sonnet-4-5",
        "cost_limit_per_day": 10.0,
        "cost_limit_per_message": 1.0,
        "use_platform_key_fallback": True,
        "show_token_costs": True,
        "show_model_in_chat": True,
        "preferred_models_by_task": {},
    }


def mask_key(prefix: str, suffix: str | None) -> str:
    """Create masked key display string."""
    if suffix:
        return f"{prefix}...{suffix}"
    return f"{prefix}..."


# ============================================================================
# AI Preferences Endpoints
# ============================================================================


@router.get("/ai-preferences", response_model=AIPreferences)
async def get_ai_preferences(request: Request):
    """Get the current user's AI preferences."""
    user = await get_current_user(request)
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    prefs = profile.get("ai_preferences") or get_default_ai_preferences()

    return AIPreferences(**prefs)


@router.put("/ai-preferences", response_model=AIPreferences)
async def update_ai_preferences(
    body: UpdateAIPreferencesRequest, request: Request
):
    """Update the current user's AI preferences."""
    user = await get_current_user(request)
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    supabase = get_supabase_client()

    # Get current preferences
    current_prefs = profile.get("ai_preferences") or get_default_ai_preferences()

    # Merge updates
    if body.default_provider is not None:
        current_prefs["default_provider"] = body.default_provider
    if body.default_model is not None:
        current_prefs["default_model"] = body.default_model
    if body.cost_limit_per_day is not None:
        current_prefs["cost_limit_per_day"] = body.cost_limit_per_day
    if body.cost_limit_per_message is not None:
        current_prefs["cost_limit_per_message"] = body.cost_limit_per_message
    if body.use_platform_key_fallback is not None:
        current_prefs["use_platform_key_fallback"] = body.use_platform_key_fallback
    if body.show_token_costs is not None:
        current_prefs["show_token_costs"] = body.show_token_costs
    if body.show_model_in_chat is not None:
        current_prefs["show_model_in_chat"] = body.show_model_in_chat
    if body.preferred_models_by_task is not None:
        current_prefs["preferred_models_by_task"] = body.preferred_models_by_task

    result = await supabase.update(
        "user_profiles",
        {"user_id": f"eq.{user['user_id']}"},
        {
            "ai_preferences": current_prefs,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )

    if result.get("error"):
        logger.error(
            "Failed to update AI preferences",
            user_id=user["user_id"],
            error=result.get("error"),
        )
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    logger.info("AI preferences updated", user_id=user["user_id"])

    return AIPreferences(**current_prefs)


# ============================================================================
# Provider Keys (BYOK) Endpoints
# ============================================================================


@router.get("/provider-keys", response_model=list[ProviderKeyInfo])
async def list_provider_keys(request: Request):
    """List all provider keys configured by the user (masked)."""
    user = await get_current_user(request)

    supabase = get_supabase_client()

    result = await supabase.select(
        "user_provider_keys",
        columns="id,provider,key_prefix,key_suffix,is_valid,last_validated_at,validation_error,created_at",
        filters={"user_id": f"eq.{user['user_id']}"},
    )

    if result.get("error"):
        logger.error(
            "Failed to fetch provider keys",
            user_id=user["user_id"],
            error=result.get("error"),
        )
        raise HTTPException(status_code=500, detail="Failed to fetch provider keys")

    keys = []
    for key in result.get("data", []):
        keys.append(
            ProviderKeyInfo(
                id=key["id"],
                provider=key["provider"],
                key_display=mask_key(key["key_prefix"], key.get("key_suffix")),
                is_valid=key.get("is_valid", True),
                last_validated_at=key.get("last_validated_at"),
                validation_error=key.get("validation_error"),
                created_at=key["created_at"],
            )
        )

    return keys


@router.post("/provider-keys", response_model=ProviderKeyInfo)
async def add_provider_key(body: AddProviderKeyRequest, request: Request):
    """Add or update a provider API key."""
    try:
        user = await get_current_user(request)
    except Exception as e:
        logger.error("Failed to get current user", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")

    # Validate the key first
    try:
        router_service = get_provider_router()
        provider = ProviderEnum(body.provider)
        is_valid, error = await router_service.validate_key(provider, body.api_key)
    except Exception as e:
        logger.error(
            "Failed to validate API key",
            provider=body.provider,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to validate API key. Please try again.",
        )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid API key: {error}",
        )

    supabase = get_supabase_client()

    # CLOUDFLARE-ONLY: All encryption happens via Cloudflare Key Vault (zero-knowledge)
    # No local encryption fallback - keys are never seen by the backend in plaintext
    if not is_key_vault_available():
        logger.error(
            "Cloudflare Key Vault not configured - cannot store API keys",
            user_id=user["user_id"],
            provider=body.provider,
        )
        raise HTTPException(
            status_code=503,
            detail="Key storage service unavailable. Please contact support.",
        )

    try:
        logger.info(
            "Encrypting API key via Cloudflare Key Vault",
            user_id=user["user_id"],
            provider=body.provider,
        )
        encrypted_bundle = await encrypt_api_key_secure(
            user_id=user["user_id"],
            provider=body.provider,
            api_key=body.api_key,
        )
    except Exception as e:
        logger.error(
            "Cloudflare Key Vault encryption failed",
            user_id=user["user_id"],
            provider=body.provider,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to encrypt API key. Please try again.",
        )

    key_data = {
        "user_id": user["user_id"],
        "provider": body.provider,
        "encrypted_key": encrypted_bundle.encrypted_key,
        "key_prefix": encrypted_bundle.key_prefix,
        "key_suffix": encrypted_bundle.key_suffix,
        "is_valid": True,
        "last_validated_at": datetime.now(UTC).isoformat(),
        "validation_error": None,
        "display_name": body.display_name,
        "updated_at": datetime.now(UTC).isoformat(),
        # Cloudflare Key Vault fields
        "dek_reference": encrypted_bundle.dek_reference,
        "dek_version": encrypted_bundle.dek_version,
        "encryption_method": "cloudflare",
        "encrypted_at": encrypted_bundle.encrypted_at.isoformat(),
    }

    # Check if key exists
    existing = await supabase.select(
        "user_provider_keys",
        columns="id",
        filters={
            "user_id": f"eq.{user['user_id']}",
            "provider": f"eq.{body.provider}",
        },
    )

    if existing.get("data") and len(existing["data"]) > 0:
        # Update existing
        result = await supabase.update(
            "user_provider_keys",
            {
                "user_id": f"eq.{user['user_id']}",
                "provider": f"eq.{body.provider}",
            },
            key_data,
        )
    else:
        # Insert new
        result = await supabase.insert("user_provider_keys", key_data)

    if result.get("error"):
        logger.error(
            "Failed to save provider key",
            user_id=user["user_id"],
            provider=body.provider,
            error=result.get("error"),
        )
        raise HTTPException(status_code=500, detail="Failed to save provider key")

    # Get the saved key
    try:
        saved = await supabase.select(
            "user_provider_keys",
            columns="*",
            filters={
                "user_id": f"eq.{user['user_id']}",
                "provider": f"eq.{body.provider}",
            },
        )

        if not saved.get("data") or len(saved["data"]) == 0:
            logger.error(
                "Provider key was saved but couldn't be retrieved",
                user_id=user["user_id"],
                provider=body.provider,
            )
            raise HTTPException(
                status_code=500,
                detail="Key was saved but couldn't be retrieved. Please refresh.",
            )

        key = saved["data"][0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to retrieve saved provider key",
            user_id=user["user_id"],
            provider=body.provider,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve saved key. Please refresh.",
        )

    logger.info(
        "Provider key saved",
        user_id=user["user_id"],
        provider=body.provider,
        key_prefix=key_data["key_prefix"],
    )

    return ProviderKeyInfo(
        id=key["id"],
        provider=key["provider"],
        key_display=mask_key(key.get("key_prefix", "***"), key.get("key_suffix")),
        is_valid=key.get("is_valid", True),
        last_validated_at=key.get("last_validated_at"),
        validation_error=key.get("validation_error"),
        created_at=key.get("created_at", datetime.now(UTC).isoformat()),
    )


@router.delete("/provider-keys/{provider}")
async def delete_provider_key(provider: str, request: Request):
    """Remove a provider API key."""
    user = await get_current_user(request)

    if provider not in ["anthropic", "openai", "google", "groq", "together"]:
        raise HTTPException(status_code=400, detail="Invalid provider")

    supabase = get_supabase_client()

    # First, get the key to check if it uses Cloudflare Key Vault
    existing = await supabase.select(
        "user_provider_keys",
        columns="dek_reference,encryption_method",
        filters={
            "user_id": f"eq.{user['user_id']}",
            "provider": f"eq.{provider}",
        },
    )

    # Delete DEK from Cloudflare if using Key Vault
    if existing.get("data") and len(existing["data"]) > 0:
        key_info = existing["data"][0]
        if key_info.get("encryption_method") == "cloudflare" and key_info.get("dek_reference"):
            try:
                vault = get_key_vault_client()
                await vault.delete_dek(key_info["dek_reference"])
                logger.info(
                    "Deleted DEK from Cloudflare Key Vault",
                    user_id=user["user_id"],
                    provider=provider,
                    dek_reference=key_info["dek_reference"],
                )
            except Exception as e:
                # Log but don't fail - DEK will be orphaned but not usable
                logger.warning(
                    "Failed to delete DEK from Cloudflare",
                    user_id=user["user_id"],
                    provider=provider,
                    error=str(e),
                )

    # Delete from database
    result = await supabase.delete(
        "user_provider_keys",
        filters={
            "user_id": f"eq.{user['user_id']}",
            "provider": f"eq.{provider}",
        },
    )

    if result.get("error"):
        logger.error(
            "Failed to delete provider key",
            user_id=user["user_id"],
            provider=provider,
            error=result.get("error"),
        )
        raise HTTPException(status_code=500, detail="Failed to delete provider key")

    logger.info(
        "Provider key deleted",
        user_id=user["user_id"],
        provider=provider,
    )

    return {"success": True, "message": f"Removed {provider} API key"}


@router.post("/provider-keys/{provider}/validate", response_model=ValidateKeyResponse)
async def validate_provider_key(provider: str, request: Request):
    """Validate a stored provider key still works."""
    user = await get_current_user(request)

    if provider not in ["anthropic", "openai", "google", "groq", "together"]:
        raise HTTPException(status_code=400, detail="Invalid provider")

    supabase = get_supabase_client()

    # Get the encrypted key with encryption method info
    result = await supabase.select(
        "user_provider_keys",
        columns="encrypted_key,encryption_method,dek_reference",
        filters={
            "user_id": f"eq.{user['user_id']}",
            "provider": f"eq.{provider}",
        },
    )

    if not result.get("data") or len(result["data"]) == 0:
        raise HTTPException(
            status_code=404, detail=f"No {provider} key configured"
        )

    key_data = result["data"][0]

    # CLOUDFLARE-ONLY: Decrypt using Cloudflare Key Vault
    if not key_data.get("dek_reference"):
        logger.error(
            "Key has no DEK reference - cannot decrypt",
            user_id=user["user_id"],
            provider=provider,
        )
        raise HTTPException(
            status_code=500,
            detail="Key validation failed. Key may be corrupted.",
        )

    try:
        decrypted_key = await decrypt_api_key_secure(
            key_data["encrypted_key"],
            key_data["dek_reference"],
        )
        logger.debug(
            "Decrypted key via Cloudflare for validation",
            user_id=user["user_id"],
            provider=provider,
        )
    except Exception as e:
        logger.error(
            "Failed to decrypt provider key",
            user_id=user["user_id"],
            provider=provider,
            error=str(e),
        )
        # Mark as invalid
        await supabase.update(
            "user_provider_keys",
            {
                "user_id": f"eq.{user['user_id']}",
                "provider": f"eq.{provider}",
            },
            {
                "is_valid": False,
                "validation_error": "Decryption failed",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        return ValidateKeyResponse(
            is_valid=False, error="Decryption failed", provider=provider
        )

    # Validate with provider
    try:
        router_service = get_provider_router()
        provider_enum = ProviderEnum(provider)
        is_valid, error = await router_service.validate_key(provider_enum, decrypted_key)
    except Exception as e:
        logger.error(
            "Failed to validate provider key",
            user_id=user["user_id"],
            provider=provider,
            error=str(e),
        )
        is_valid = False
        error = f"Validation error: {str(e)}"

    # Update validation status
    try:
        await supabase.update(
            "user_provider_keys",
            {
                "user_id": f"eq.{user['user_id']}",
                "provider": f"eq.{provider}",
            },
            {
                "is_valid": is_valid,
                "last_validated_at": datetime.now(UTC).isoformat(),
                "validation_error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    except Exception as e:
        logger.warning(
            "Failed to update validation status (non-critical)",
            user_id=user["user_id"],
            provider=provider,
            error=str(e),
        )
        # Don't fail the request, just log it

    logger.info(
        "Provider key validated",
        user_id=user["user_id"],
        provider=provider,
        is_valid=is_valid,
    )

    return ValidateKeyResponse(is_valid=is_valid, error=error, provider=provider)


# ============================================================================
# Usage & Billing Endpoints
# ============================================================================


@router.get("/ai-usage", response_model=UsageResponse)
async def get_ai_usage(
    request: Request,
    days: int = 30,
    limit: int = 100,
):
    """Get detailed AI usage history."""
    try:
        user = await get_current_user(request)
    except Exception as e:
        logger.error("Failed to get current user for usage", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")

    try:
        supabase = get_supabase_client()
    except Exception as e:
        logger.error("Failed to get Supabase client", error=str(e))
        raise HTTPException(status_code=500, detail="Database connection failed")

    start_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    # Get individual usage records
    try:
        records_result = await supabase.select(
            "user_ai_usage",
            columns="*",
            filters={
                "user_id": f"eq.{user['user_id']}",
                "created_at": f"gte.{start_date}",
            },
            order="created_at.desc",
            limit=limit,
        )
    except Exception as e:
        logger.warning("Failed to fetch usage records", user_id=user["user_id"], error=str(e))
        records_result = {"data": []}  # Return empty on failure

    records = []
    for r in records_result.get("data", []):
        records.append(
            UsageRecord(
                id=r["id"],
                provider=r["provider"],
                model=r["model"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                cost_usd=float(r["cost_usd"]),
                key_source=r["key_source"],
                thread_id=r.get("thread_id"),
                created_at=r["created_at"],
            )
        )

    # Get daily summaries
    try:
        daily_result = await supabase.select(
            "user_ai_usage_daily",
            columns="*",
            filters={
                "user_id": f"eq.{user['user_id']}",
                "date": f"gte.{start_date[:10]}",  # Date only
            },
            order="date.desc",
        )
    except Exception as e:
        logger.warning("Failed to fetch daily summaries", user_id=user["user_id"], error=str(e))
        daily_result = {"data": []}  # Return empty on failure

    daily = []
    total_requests = 0
    total_input = 0
    total_output = 0
    total_cost = Decimal("0")
    platform_cost = Decimal("0")
    byok_cost = Decimal("0")
    usage_by_model: dict[str, dict] = {}
    usage_by_provider: dict[str, dict] = {}

    for d in daily_result.get("data", []):
        daily.append(
            DailyUsage(
                date=d["date"],
                total_requests=d["total_requests"],
                total_tokens=d["total_input_tokens"] + d["total_output_tokens"],
                total_cost_usd=float(d["total_cost_usd"]),
            )
        )

        # Aggregate summary
        total_requests += d["total_requests"]
        total_input += d["total_input_tokens"]
        total_output += d["total_output_tokens"]
        total_cost += Decimal(str(d["total_cost_usd"]))
        platform_cost += Decimal(str(d.get("platform_key_cost", 0)))
        byok_cost += Decimal(str(d.get("byok_cost", 0)))

        # Merge model usage
        for model, stats in (d.get("usage_by_model") or {}).items():
            if model not in usage_by_model:
                usage_by_model[model] = {"requests": 0, "tokens": 0, "cost": 0.0}
            usage_by_model[model]["requests"] += stats.get("requests", 0)
            usage_by_model[model]["tokens"] += stats.get("tokens", 0)
            usage_by_model[model]["cost"] += stats.get("cost", 0)

        # Merge provider usage
        for prov, stats in (d.get("usage_by_provider") or {}).items():
            if prov not in usage_by_provider:
                usage_by_provider[prov] = {"requests": 0, "cost": 0.0}
            usage_by_provider[prov]["requests"] += stats.get("requests", 0)
            usage_by_provider[prov]["cost"] += stats.get("cost", 0)

    summary = UsageSummary(
        total_requests=total_requests,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=float(total_cost),
        platform_key_cost=float(platform_cost),
        byok_cost=float(byok_cost),
        usage_by_model=usage_by_model,
        usage_by_provider=usage_by_provider,
    )

    return UsageResponse(summary=summary, daily=daily, records=records)


@router.get("/ai-usage/summary", response_model=UsageSummary)
async def get_ai_usage_summary(
    request: Request,
    period: str = "month",  # "day", "week", "month"
):
    """Get aggregated AI usage summary."""
    user = await get_current_user(request)

    days = {"day": 1, "week": 7, "month": 30}.get(period, 30)

    supabase = get_supabase_client()

    start_date = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")

    result = await supabase.select(
        "user_ai_usage_daily",
        columns="*",
        filters={
            "user_id": f"eq.{user['user_id']}",
            "date": f"gte.{start_date}",
        },
    )

    total_requests = 0
    total_input = 0
    total_output = 0
    total_cost = Decimal("0")
    platform_cost = Decimal("0")
    byok_cost = Decimal("0")
    usage_by_model: dict[str, dict] = {}
    usage_by_provider: dict[str, dict] = {}

    for d in result.get("data", []):
        total_requests += d["total_requests"]
        total_input += d["total_input_tokens"]
        total_output += d["total_output_tokens"]
        total_cost += Decimal(str(d["total_cost_usd"]))
        platform_cost += Decimal(str(d.get("platform_key_cost", 0)))
        byok_cost += Decimal(str(d.get("byok_cost", 0)))

        for model, stats in (d.get("usage_by_model") or {}).items():
            if model not in usage_by_model:
                usage_by_model[model] = {"requests": 0, "tokens": 0, "cost": 0.0}
            usage_by_model[model]["requests"] += stats.get("requests", 0)
            usage_by_model[model]["tokens"] += stats.get("tokens", 0)
            usage_by_model[model]["cost"] += stats.get("cost", 0)

        for prov, stats in (d.get("usage_by_provider") or {}).items():
            if prov not in usage_by_provider:
                usage_by_provider[prov] = {"requests": 0, "cost": 0.0}
            usage_by_provider[prov]["requests"] += stats.get("requests", 0)
            usage_by_provider[prov]["cost"] += stats.get("cost", 0)

    return UsageSummary(
        total_requests=total_requests,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=float(total_cost),
        platform_key_cost=float(platform_cost),
        byok_cost=float(byok_cost),
        usage_by_model=usage_by_model,
        usage_by_provider=usage_by_provider,
    )


@router.get("/ai-budget", response_model=BudgetStatus)
async def get_ai_budget_status(request: Request):
    """Get user's current AI budget status."""
    user = await get_current_user(request)
    profile = await get_or_create_profile(user["user_id"], user.get("email"))

    prefs = profile.get("ai_preferences") or get_default_ai_preferences()

    supabase = get_supabase_client()

    # Get today's spend
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    result = await supabase.select(
        "user_ai_usage_daily",
        columns="total_cost_usd",
        filters={
            "user_id": f"eq.{user['user_id']}",
            "date": f"eq.{today}",
        },
    )

    daily_spent = 0.0
    if result.get("data") and len(result["data"]) > 0:
        daily_spent = float(result["data"][0]["total_cost_usd"])

    daily_limit = float(prefs.get("cost_limit_per_day", 10.0))
    daily_remaining = max(0, daily_limit - daily_spent)

    return BudgetStatus(
        has_budget=daily_remaining > 0,
        daily_limit=daily_limit,
        daily_spent=daily_spent,
        daily_remaining=daily_remaining,
        message_limit=float(prefs.get("cost_limit_per_message", 1.0)),
    )


# ============================================================================
# Available Models Endpoint
# ============================================================================


@router.get("/available-models", response_model=AvailableModelsResponse)
async def get_available_models(request: Request):
    """Get list of available models for the user.

    Returns all models with availability based on user's configured keys.
    """
    try:
        user = await get_current_user(request)
    except Exception as e:
        logger.error("Failed to get current user for models", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")

    # Get user's configured providers
    try:
        supabase = get_supabase_client()
        keys_result = await supabase.select(
            "user_provider_keys",
            columns="provider,is_valid",
            filters={"user_id": f"eq.{user['user_id']}"},
        )
    except Exception as e:
        logger.warning("Failed to fetch user provider keys", user_id=user["user_id"], error=str(e))
        keys_result = {"data": []}  # Return empty on failure

    user_providers = set()
    for key in keys_result.get("data", []):
        if key.get("is_valid", True):
            user_providers.add(key["provider"])

    # Check which platform keys are available
    import os

    platform_providers = set()
    if os.getenv("ANTHROPIC_API_KEY"):
        platform_providers.add("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        platform_providers.add("openai")
    if os.getenv("GOOGLE_API_KEY"):
        platform_providers.add("google")
    if os.getenv("GROQ_API_KEY"):
        platform_providers.add("groq")
    if os.getenv("TOGETHER_API_KEY"):
        platform_providers.add("together")

    # Get user's fallback preference
    profile = await get_or_create_profile(user["user_id"], user.get("email"))
    prefs = profile.get("ai_preferences") or get_default_ai_preferences()
    use_platform_fallback = prefs.get("use_platform_key_fallback", True)

    # Available providers = user keys + (platform keys if fallback enabled)
    available_providers = user_providers.copy()
    if use_platform_fallback:
        available_providers.update(platform_providers)

    # Get models from registry
    try:
        registry = get_model_registry()
        all_models = registry.list_all_models()
    except Exception as e:
        logger.error("Failed to get model registry", error=str(e))
        # Return empty list instead of crashing
        return AvailableModelsResponse(models=[], user_providers=list(user_providers))

    models = []

    for model in all_models:
        try:
            provider = model.provider.value if hasattr(model.provider, "value") else str(model.provider)

            capabilities = []
            for cap in model.capabilities:
                cap_name = cap.value if hasattr(cap, "value") else str(cap)
                capabilities.append(cap_name)

            models.append(
                ModelInfo(
                    model_id=model.model_id,
                    display_name=model.display_name,
                    provider=provider,
                    input_price=getattr(model, "input_price", 0.0),
                    output_price=getattr(model, "output_price", 0.0),
                    capabilities=capabilities,
                    is_available=provider in available_providers,
                )
            )
        except Exception as e:
            logger.warning("Failed to process model", model=str(model), error=str(e))
            continue  # Skip this model but don't crash

    # Sort: available first, then by provider, then by name
    models.sort(key=lambda m: (not m.is_available, m.provider, m.display_name))

    return AvailableModelsResponse(
        models=models,
        user_providers=list(user_providers),
    )
