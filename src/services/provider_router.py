"""Provider Router for BYOK multi-provider AI routing.

This service routes AI requests to the appropriate provider based on:
1. User's BYOK (Bring Your Own Key) configuration
2. Model requirements (which provider hosts the model)

The router handles:
- Decrypting user's API keys on demand via Cloudflare Key Vault (zero-knowledge)
- Validating keys before use
- Tracking usage for billing

Security Architecture:
- CLOUDFLARE-ONLY: All encryption/decryption via Cloudflare Key Vault
- Keys are NEVER decrypted on the backend - only at Cloudflare edge
- Zero-knowledge architecture ensures backend never sees plaintext keys after initial storage
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import httpx
import structlog

from src.services.cloudflare_key_vault import (
    is_key_vault_available,
    decrypt_api_key_secure,
)
from src.services.supabase_client import get_supabase_client
from src.services.ai_cost_tracker import MODEL_PRICING, get_cost_tracker

logger = structlog.get_logger()


class Provider(str, Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"


@dataclass
class AIConfig:
    """Configuration for an AI request."""

    model: str
    provider: Provider
    api_key: str
    key_source: str  # "platform" or "byok"
    user_id: str | None = None


@dataclass
class UserProviderKey:
    """A user's stored provider key."""

    id: str
    user_id: str
    provider: Provider
    encrypted_key: str
    key_prefix: str
    key_suffix: str | None
    is_valid: bool
    last_validated_at: datetime | None
    validation_error: str | None
    # Cloudflare Key Vault fields (for envelope encryption)
    dek_reference: str | None = None
    dek_version: int = 1
    encryption_method: str = "local"  # "local" or "cloudflare"


class ProviderRouter:
    """Routes AI requests to appropriate provider with BYOK support."""

    # Map model prefixes to providers
    MODEL_PROVIDER_MAP = {
        "claude": Provider.ANTHROPIC,
        "gpt": Provider.OPENAI,
        "o1": Provider.OPENAI,
        "gemini": Provider.GOOGLE,
        "llama": Provider.GROQ,
        "deepseek": Provider.TOGETHER,
    }

    # Platform API key env var names
    PLATFORM_KEY_VARS = {
        Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
        Provider.OPENAI: "OPENAI_API_KEY",
        Provider.GOOGLE: "GOOGLE_API_KEY",
        Provider.GROQ: "GROQ_API_KEY",
        Provider.TOGETHER: "TOGETHER_API_KEY",
    }

    # API endpoints for key validation
    VALIDATION_ENDPOINTS = {
        Provider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
        Provider.OPENAI: "https://api.openai.com/v1/models",
        Provider.GOOGLE: "https://generativelanguage.googleapis.com/v1/models",
        Provider.GROQ: "https://api.groq.com/openai/v1/models",
        Provider.TOGETHER: "https://api.together.xyz/v1/models",
    }

    def __init__(self):
        self._supabase = get_supabase_client()
        self._cost_tracker = get_cost_tracker()
        # Cache user keys briefly to avoid repeated DB lookups
        self._key_cache: dict[str, dict[str, UserProviderKey]] = {}

    def get_provider_for_model(self, model: str) -> Provider:
        """Determine which provider hosts a model.

        Args:
            model: Model ID (e.g., "claude-sonnet-4-5")

        Returns:
            Provider enum value
        """
        model_lower = model.lower()

        for prefix, provider in self.MODEL_PROVIDER_MAP.items():
            if prefix in model_lower:
                return provider

        # Check MODEL_PRICING for provider info
        if model in MODEL_PRICING:
            provider_str = MODEL_PRICING[model].provider
            try:
                return Provider(provider_str)
            except ValueError:
                pass

        # Default to Anthropic for unknown models
        return Provider.ANTHROPIC

    async def get_ai_config(
        self,
        user_id: str,
        model: str,
        allow_platform_fallback: bool = True,
    ) -> AIConfig:
        """Get AI configuration for a request.

        This determines:
        1. Which provider to use
        2. Which API key to use (user's BYOK or platform key)

        Args:
            user_id: The user making the request
            model: Model to use
            allow_platform_fallback: Whether to fall back to platform keys

        Returns:
            AIConfig with provider and key information

        Raises:
            ValueError: If no valid key is available
        """
        provider = self.get_provider_for_model(model)

        # Try to get user's BYOK key
        user_key = await self._get_user_key(user_id, provider)

        if user_key and user_key.is_valid:
            # CLOUDFLARE-ONLY: Decrypt via Cloudflare Key Vault
            if not user_key.dek_reference:
                logger.error(
                    "Key has no DEK reference - cannot decrypt",
                    user_id=user_id,
                    provider=provider.value,
                )
                await self._mark_key_invalid(user_id, provider, "No DEK reference")
            else:
                try:
                    decrypted_key = await decrypt_api_key_secure(
                        user_key.encrypted_key,
                        user_key.dek_reference,
                    )
                    logger.debug(
                        "Decrypted BYOK key via Cloudflare Key Vault",
                        user_id=user_id,
                        provider=provider.value,
                        key_prefix=user_key.key_prefix,
                        dek_version=user_key.dek_version,
                    )

                    return AIConfig(
                        model=model,
                        provider=provider,
                        api_key=decrypted_key,
                        key_source="byok",
                        user_id=user_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to decrypt BYOK key via Cloudflare, marking invalid",
                        user_id=user_id,
                        provider=provider.value,
                        dek_reference=user_key.dek_reference,
                        error=str(e),
                    )
                    await self._mark_key_invalid(user_id, provider, str(e))

        # No BYOK key available - BYOK-only mode, no platform fallback
        raise ValueError(
            f"No API key configured for {provider.value}. "
            f"Please add your {provider.value.title()} API key in Settings → AI Configuration."
        )

    async def validate_key(
        self, provider: Provider, api_key: str
    ) -> tuple[bool, str | None]:
        """Validate an API key works with the provider.

        Args:
            provider: The provider to validate against
            api_key: The API key to test

        Returns:
            Tuple of (is_valid, error_message)
        """
        endpoint = self.VALIDATION_ENDPOINTS.get(provider)
        if not endpoint:
            return True, None  # Can't validate, assume valid

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = self._get_auth_headers(provider, api_key)

                # For Anthropic, we need to make a minimal API call
                if provider == Provider.ANTHROPIC:
                    # Use a minimal messages request to validate
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": "claude-haiku-4-5",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "Hi"}],
                        },
                    )
                else:
                    # Other providers have models list endpoint
                    response = await client.get(endpoint, headers=headers)

                if response.status_code == 200:
                    return True, None
                elif response.status_code == 401:
                    return False, "Invalid API key"
                elif response.status_code == 403:
                    return False, "API key lacks required permissions"
                else:
                    return False, f"Validation failed: {response.status_code}"

        except httpx.TimeoutException:
            return False, "Validation timed out"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _get_auth_headers(self, provider: Provider, api_key: str) -> dict[str, str]:
        """Get authorization headers for a provider."""
        if provider == Provider.ANTHROPIC:
            return {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        elif provider == Provider.GOOGLE:
            return {"x-goog-api-key": api_key}
        else:
            # OpenAI-compatible (OpenAI, Groq, Together)
            return {"Authorization": f"Bearer {api_key}"}

    async def track_usage(
        self,
        config: AIConfig,
        input_tokens: int,
        output_tokens: int,
        organization_id: str | None = None,
        thread_id: str | None = None,
        message_id: str | None = None,
        task_type: str | None = None,
    ) -> None:
        """Track AI usage for billing.

        Records usage to appropriate tables based on key source.

        Args:
            config: AI configuration used for the request
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            organization_id: Optional organization for attribution
            thread_id: Optional thread ID for context
            message_id: Optional message ID
            task_type: Type of AI task
        """
        # Calculate cost
        cost = self._cost_tracker.calculate_cost(
            config.model, input_tokens, output_tokens
        )

        request_id = str(uuid.uuid4())

        # Record to user-level tracking
        if config.user_id and self._supabase.is_configured:
            await self._record_user_usage(
                user_id=config.user_id,
                organization_id=organization_id,
                provider=config.provider.value,
                model=config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                key_source=config.key_source,
                thread_id=thread_id,
                message_id=message_id,
                request_id=request_id,
                task_type=task_type,
            )

        logger.info(
            "Tracked AI usage",
            user_id=config.user_id,
            provider=config.provider.value,
            model=config.model,
            key_source=config.key_source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=float(cost),
        )

    async def _get_user_key(
        self, user_id: str, provider: Provider
    ) -> UserProviderKey | None:
        """Get user's key for a provider from DB."""
        if not self._supabase.is_configured:
            return None

        result = await self._supabase.select(
            "user_provider_keys",
            columns="*",
            filters={
                "user_id": f"eq.{user_id}",
                "provider": f"eq.{provider.value}",
            },
        )

        if result.get("error") or not result.get("data"):
            return None

        data = result["data"][0]
        return UserProviderKey(
            id=data["id"],
            user_id=data["user_id"],
            provider=Provider(data["provider"]),
            encrypted_key=data["encrypted_key"],
            key_prefix=data["key_prefix"],
            key_suffix=data.get("key_suffix"),
            is_valid=data.get("is_valid", True),
            last_validated_at=data.get("last_validated_at"),
            validation_error=data.get("validation_error"),
            # Cloudflare Key Vault fields
            dek_reference=data.get("dek_reference"),
            dek_version=data.get("dek_version", 1),
            encryption_method=data.get("encryption_method", "local"),
        )

    async def _mark_key_invalid(
        self, user_id: str, provider: Provider, error: str
    ) -> None:
        """Mark a user's key as invalid in the database."""
        if not self._supabase.is_configured:
            return

        await self._supabase.update(
            "user_provider_keys",
            {
                "is_valid": False,
                "validation_error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            filters={
                "user_id": f"eq.{user_id}",
                "provider": f"eq.{provider.value}",
            },
        )

    async def _record_user_usage(
        self,
        user_id: str,
        organization_id: str | None,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
        key_source: str,
        thread_id: str | None,
        message_id: str | None,
        request_id: str,
        task_type: str | None,
    ) -> None:
        """Record usage to user_ai_usage table."""
        # Use the database function for atomic insert + daily summary update
        await self._supabase.rpc(
            "record_user_ai_usage",
            {
                "p_user_id": user_id,
                "p_organization_id": organization_id,
                "p_provider": provider,
                "p_model": model,
                "p_input_tokens": input_tokens,
                "p_output_tokens": output_tokens,
                "p_cost_usd": float(cost_usd),
                "p_key_source": key_source,
                "p_thread_id": thread_id,
                "p_message_id": message_id,
                "p_request_id": request_id,
                "p_task_type": task_type,
            },
        )


# Singleton instance
_provider_router: ProviderRouter | None = None


def get_provider_router() -> ProviderRouter:
    """Get or create global provider router."""
    global _provider_router
    if _provider_router is None:
        _provider_router = ProviderRouter()
    return _provider_router


async def get_ai_config(
    user_id: str, model: str, allow_platform_fallback: bool = True
) -> AIConfig:
    """Get AI configuration (convenience function)."""
    return await get_provider_router().get_ai_config(
        user_id, model, allow_platform_fallback
    )


async def track_ai_usage(
    config: AIConfig,
    input_tokens: int,
    output_tokens: int,
    **kwargs,
) -> None:
    """Track AI usage (convenience function)."""
    await get_provider_router().track_usage(
        config, input_tokens, output_tokens, **kwargs
    )
