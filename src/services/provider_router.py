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

from src.services.ai_cost_tracker import MODEL_PRICING, get_cost_tracker
from src.services.cloudflare_key_vault import (
    decrypt_api_key_secure,
    is_key_vault_available,
)
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


class Provider(str, Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    # Multi-model router (OpenAI-compatible, 400+ models)
    OPENROUTER = "openrouter"
    # Additional providers
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    COHERE = "cohere"
    XAI = "xai"
    CEREBRAS = "cerebras"
    # Enterprise providers
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    GOOGLE_VERTEX = "google_vertex"


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
        # Direct provider models
        "claude": Provider.ANTHROPIC,
        "gpt": Provider.OPENAI,
        "o1": Provider.OPENAI,
        "o3": Provider.OPENAI,
        "gemini": Provider.GOOGLE,
        "llama": Provider.GROQ,
        "mixtral": Provider.GROQ,
        "deepseek": Provider.DEEPSEEK,
        "mistral": Provider.MISTRAL,
        "codestral": Provider.MISTRAL,
        "command": Provider.COHERE,
        "grok": Provider.XAI,
        "pplx": Provider.PERPLEXITY,
        "sonar": Provider.PERPLEXITY,
    }

    # OpenRouter model prefix patterns (e.g., "anthropic/claude-sonnet-4-5")
    OPENROUTER_PREFIXES = [
        "anthropic/", "openai/", "google/", "meta-llama/", "mistralai/",
        "deepseek/", "cohere/", "x-ai/", "perplexity/", "qwen/", "nvidia/",
    ]

    # Platform API key env var names
    PLATFORM_KEY_VARS = {
        Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
        Provider.OPENAI: "OPENAI_API_KEY",
        Provider.GOOGLE: "GOOGLE_API_KEY",
        Provider.GROQ: "GROQ_API_KEY",
        Provider.TOGETHER: "TOGETHER_API_KEY",
        Provider.OPENROUTER: "OPENROUTER_API_KEY",
        Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
        Provider.MISTRAL: "MISTRAL_API_KEY",
        Provider.FIREWORKS: "FIREWORKS_API_KEY",
        Provider.PERPLEXITY: "PERPLEXITY_API_KEY",
        Provider.COHERE: "COHERE_API_KEY",
        Provider.XAI: "XAI_API_KEY",
        Provider.CEREBRAS: "CEREBRAS_API_KEY",
        Provider.AZURE_OPENAI: "AZURE_OPENAI_API_KEY",
    }

    # API endpoints for key validation (OpenAI-compatible use /models)
    VALIDATION_ENDPOINTS = {
        Provider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
        Provider.OPENAI: "https://api.openai.com/v1/models",
        Provider.GOOGLE: "https://generativelanguage.googleapis.com/v1/models",
        Provider.GROQ: "https://api.groq.com/openai/v1/models",
        Provider.TOGETHER: "https://api.together.xyz/v1/models",
        Provider.OPENROUTER: "https://openrouter.ai/api/v1/models",
        Provider.DEEPSEEK: "https://api.deepseek.com/v1/models",
        Provider.MISTRAL: "https://api.mistral.ai/v1/models",
        Provider.FIREWORKS: "https://api.fireworks.ai/inference/v1/models",
        Provider.PERPLEXITY: "https://api.perplexity.ai/chat/completions",
        Provider.COHERE: "https://api.cohere.ai/v1/models",
        Provider.XAI: "https://api.x.ai/v1/models",
        Provider.CEREBRAS: "https://api.cerebras.ai/v1/models",
    }

    def __init__(self):
        self._supabase = get_supabase_client()
        self._cost_tracker = get_cost_tracker()
        # Cache user keys briefly to avoid repeated DB lookups
        self._key_cache: dict[str, dict[str, UserProviderKey]] = {}

    def get_provider_for_model(self, model: str) -> Provider:
        """Determine which provider hosts a model.

        Args:
            model: Model ID (e.g., "claude-sonnet-4-5" or "anthropic/claude-sonnet-4-5")

        Returns:
            Provider enum value
        """
        model_lower = model.lower()

        # Check if this is an OpenRouter-style model ID (provider/model format)
        for prefix in self.OPENROUTER_PREFIXES:
            if model_lower.startswith(prefix):
                return Provider.OPENROUTER

        # Check standard model prefixes
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

        # No BYOK key available - try platform fallback if allowed
        if allow_platform_fallback:
            import os
            platform_key_var = self.PLATFORM_KEY_VARS.get(provider)
            if platform_key_var:
                platform_key = os.environ.get(platform_key_var)
                if platform_key:
                    logger.info(
                        "Using platform API key (BYOK not configured)",
                        user_id=user_id,
                        provider=provider.value,
                        model=model,
                    )
                    return AIConfig(
                        model=model,
                        provider=provider,
                        api_key=platform_key,
                        key_source="platform",
                        user_id=user_id,
                    )

        # No key available at all
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
            # For providers without validation endpoints (enterprise), assume valid
            logger.info(f"No validation endpoint for {provider.value}, assuming valid")
            return True, None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                headers = self._get_auth_headers(provider, api_key)

                # Anthropic requires a minimal messages request
                if provider == Provider.ANTHROPIC:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": "claude-haiku-4-5",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "Hi"}],
                        },
                    )
                # Perplexity doesn't have /models, use chat completions
                elif provider == Provider.PERPLEXITY:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json={
                            "model": "sonar",
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 1,
                        },
                    )
                # Cohere uses different endpoint structure
                elif provider == Provider.COHERE:
                    response = await client.get(
                        "https://api.cohere.ai/v1/models",
                        headers=headers,
                    )
                else:
                    # Most providers have OpenAI-compatible /models endpoint
                    response = await client.get(endpoint, headers=headers)

                if response.status_code == 200:
                    return True, None
                elif response.status_code == 401:
                    return False, "Invalid API key"
                elif response.status_code == 403:
                    return False, "API key lacks required permissions"
                elif response.status_code == 429:
                    # Rate limited but key is valid
                    return True, None
                else:
                    error_detail = ""
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("error", {}).get("message", "")
                    except Exception:
                        pass
                    return False, f"Validation failed: {response.status_code} {error_detail}"

        except httpx.TimeoutException:
            return False, "Validation timed out - please try again"
        except Exception as e:
            logger.warning(f"Key validation error for {provider.value}: {e}")
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
        elif provider == Provider.OPENROUTER:
            # OpenRouter uses OpenAI-compatible auth with optional site headers
            return {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://argus.app",  # For OpenRouter analytics
                "X-Title": "Argus E2E Testing",
            }
        elif provider == Provider.COHERE:
            return {"Authorization": f"Bearer {api_key}"}
        else:
            # OpenAI-compatible providers (OpenAI, Groq, Together, DeepSeek,
            # Mistral, Fireworks, Perplexity, xAI, Cerebras)
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
