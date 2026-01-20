"""Cloudflare Key Vault Client - Zero-Knowledge BYOK Encryption.

This service provides a client for the Cloudflare Worker Key Vault,
enabling secure API key storage with envelope encryption:

1. Keys are NEVER seen by the Python backend in plaintext
2. All encryption/decryption happens at Cloudflare edge
3. Per-user DEKs limit blast radius if compromised
4. HSM-backed on Cloudflare Enterprise tier

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    ZERO-KNOWLEDGE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Python Backend (this file)                                        │
│        │                                                            │
│        │ HTTPS request with API key                                 │
│        ▼                                                            │
│   ┌─────────────────────┐                                          │
│   │ Cloudflare Worker   │ ← Encryption happens here               │
│   │ (key-vault.ts)      │                                          │
│   └──────────┬──────────┘                                          │
│              │                                                      │
│              │ Returns: encrypted_key + dek_reference               │
│              ▼                                                      │
│   ┌─────────────────────┐                                          │
│   │ Supabase            │ ← Stores encrypted_key + dek_reference   │
│   │ (user_provider_keys)│                                          │
│   └─────────────────────┘                                          │
│                                                                      │
│   Backend NEVER sees plaintext API key after initial request!       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Usage:
    from src.services.cloudflare_key_vault import get_key_vault_client

    vault = get_key_vault_client()

    # Encrypt a new API key
    result = await vault.encrypt_key(user_id, "anthropic", "sk-ant-...")
    # Returns: EncryptedKeyBundle with encrypted_key and dek_reference

    # Decrypt when needed (for API calls)
    api_key = await vault.decrypt_key(result.encrypted_key, result.dek_reference)

    # Delete when user removes their key
    await vault.delete_dek(result.dek_reference)
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class EncryptedKeyBundle:
    """Result from encrypting an API key via Cloudflare Key Vault."""

    encrypted_key: str
    """Base64-encoded encrypted API key (encrypted with DEK)."""

    dek_reference: str
    """Reference to DEK in Cloudflare KV (e.g., 'dek:user_123:anthropic:v1')."""

    key_prefix: str
    """First 8 chars for UI display (e.g., 'sk-ant-a')."""

    key_suffix: str
    """Last 4 chars for identification (e.g., 'xyz9')."""

    dek_version: int
    """Version number for rotation tracking."""

    encrypted_at: datetime
    """When the encryption was performed."""


class CloudflareKeyVaultClient:
    """Client for Cloudflare Worker Key Vault service.

    Provides zero-knowledge encryption for user API keys:
    - Backend sends plaintext key once to Cloudflare
    - Cloudflare encrypts and returns encrypted blob + DEK reference
    - Backend stores encrypted blob in Supabase
    - Decryption requires calling Cloudflare again
    """

    def __init__(
        self,
        worker_url: str,
        api_token: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Key Vault client.

        Args:
            worker_url: URL to Cloudflare Worker (e.g., https://argus-api.workers.dev)
            api_token: API token for authenticating with worker
            timeout: Request timeout in seconds
        """
        self.worker_url = worker_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            self._client = httpx.AsyncClient(
                base_url=self.worker_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> dict[str, Any]:
        """Check Key Vault health.

        Returns:
            Health status including KV and KEK availability
        """
        client = await self._get_client()
        response = await client.get("/key-vault/health")
        response.raise_for_status()
        return response.json()

    async def encrypt_key(
        self,
        user_id: str,
        provider: str,
        api_key: str,
    ) -> EncryptedKeyBundle:
        """Encrypt an API key using Cloudflare Key Vault.

        This sends the plaintext API key to Cloudflare ONCE for encryption.
        After this, the backend only stores the encrypted blob.

        Args:
            user_id: User's ID
            provider: AI provider (anthropic, openai, etc.)
            api_key: Plaintext API key to encrypt

        Returns:
            EncryptedKeyBundle with encrypted data and DEK reference

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is invalid
        """
        client = await self._get_client()

        response = await client.post(
            "/key-vault/encrypt",
            json={
                "user_id": user_id,
                "provider": provider,
                "api_key": api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

        logger.info(
            "Encrypted API key via Cloudflare Key Vault",
            user_id=user_id,
            provider=provider,
            dek_reference=data["dek_reference"],
            key_prefix=data["key_prefix"],
        )

        return EncryptedKeyBundle(
            encrypted_key=data["encrypted_key"],
            dek_reference=data["dek_reference"],
            key_prefix=data["key_prefix"],
            key_suffix=data["key_suffix"],
            dek_version=data["dek_version"],
            encrypted_at=datetime.fromisoformat(
                data["encrypted_at"].replace("Z", "+00:00")
            ),
        )

    async def decrypt_key(
        self,
        encrypted_key: str,
        dek_reference: str,
    ) -> str:
        """Decrypt an API key using Cloudflare Key Vault.

        This sends the encrypted blob to Cloudflare for decryption.
        The decryption happens at the edge, never in our backend.

        Args:
            encrypted_key: Base64-encoded encrypted API key
            dek_reference: Reference to DEK in Cloudflare KV

        Returns:
            Plaintext API key

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If DEK not found or decryption fails
        """
        client = await self._get_client()

        response = await client.post(
            "/key-vault/decrypt",
            json={
                "encrypted_key": encrypted_key,
                "dek_reference": dek_reference,
            },
        )

        if response.status_code == 404:
            raise ValueError(f"DEK not found: {dek_reference}")

        response.raise_for_status()
        data = response.json()

        logger.debug(
            "Decrypted API key via Cloudflare Key Vault",
            dek_reference=dek_reference,
        )

        return data["api_key"]

    async def delete_dek(self, dek_reference: str) -> None:
        """Delete a DEK from Cloudflare KV.

        Call this when a user removes their API key.

        Args:
            dek_reference: Reference to DEK to delete
        """
        client = await self._get_client()

        # URL-encode the reference since it contains colons
        encoded_ref = dek_reference.replace(":", "%3A")
        response = await client.delete(f"/key-vault/dek/{encoded_ref}")
        response.raise_for_status()

        logger.info(
            "Deleted DEK from Cloudflare Key Vault",
            dek_reference=dek_reference,
        )

    async def rotate_key(
        self,
        user_id: str,
        provider: str,
        encrypted_key: str,
        dek_reference: str,
    ) -> EncryptedKeyBundle:
        """Rotate the DEK for a user's API key.

        This re-encrypts the key with a new DEK without exposing plaintext.

        Args:
            user_id: User's ID
            provider: AI provider
            encrypted_key: Currently encrypted API key
            dek_reference: Current DEK reference

        Returns:
            New EncryptedKeyBundle with new DEK
        """
        client = await self._get_client()

        response = await client.post(
            "/key-vault/rotate",
            json={
                "user_id": user_id,
                "provider": provider,
                "encrypted_key": encrypted_key,
                "dek_reference": dek_reference,
            },
        )
        response.raise_for_status()
        data = response.json()

        logger.info(
            "Rotated DEK via Cloudflare Key Vault",
            user_id=user_id,
            provider=provider,
            old_dek=dek_reference,
            new_dek=data["dek_reference"],
        )

        return EncryptedKeyBundle(
            encrypted_key=data["encrypted_key"],
            dek_reference=data["dek_reference"],
            key_prefix=data["key_prefix"],
            key_suffix=data["key_suffix"],
            dek_version=data["dek_version"],
            encrypted_at=datetime.fromisoformat(
                data["encrypted_at"].replace("Z", "+00:00")
            ),
        )


# ============================================================================
# Singleton Instance
# ============================================================================

_key_vault_client: CloudflareKeyVaultClient | None = None


def get_key_vault_client() -> CloudflareKeyVaultClient:
    """Get or create global Key Vault client.

    Configuration from environment variables:
    - CLOUDFLARE_WORKER_URL: URL to Cloudflare Worker
    - CLOUDFLARE_API_TOKEN: API token for authentication

    Returns:
        Configured CloudflareKeyVaultClient instance
    """
    global _key_vault_client

    if _key_vault_client is None:
        worker_url = os.getenv(
            "CLOUDFLARE_WORKER_URL",
            "https://argus-api.heyargus.workers.dev",
        )
        api_token = os.getenv("CLOUDFLARE_API_TOKEN")

        _key_vault_client = CloudflareKeyVaultClient(
            worker_url=worker_url,
            api_token=api_token,
        )

    return _key_vault_client


def is_key_vault_available() -> bool:
    """Check if Key Vault is configured.

    Returns True if CLOUDFLARE_WORKER_URL is set.
    """
    return bool(os.getenv("CLOUDFLARE_WORKER_URL"))


async def encrypt_api_key_secure(
    user_id: str,
    provider: str,
    api_key: str,
) -> EncryptedKeyBundle:
    """Encrypt an API key using Cloudflare Key Vault (convenience function).

    Args:
        user_id: User's ID
        provider: AI provider
        api_key: Plaintext API key

    Returns:
        EncryptedKeyBundle with encrypted data
    """
    vault = get_key_vault_client()
    return await vault.encrypt_key(user_id, provider, api_key)


async def decrypt_api_key_secure(
    encrypted_key: str,
    dek_reference: str,
) -> str:
    """Decrypt an API key using Cloudflare Key Vault (convenience function).

    Args:
        encrypted_key: Encrypted API key blob
        dek_reference: DEK reference in KV

    Returns:
        Plaintext API key
    """
    vault = get_key_vault_client()
    return await vault.decrypt_key(encrypted_key, dek_reference)
