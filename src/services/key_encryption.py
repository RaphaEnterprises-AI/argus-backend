"""Key Encryption Service for secure BYOK (Bring Your Own Key) storage.

This service provides AES-256-GCM encryption for user API keys, ensuring:
- Keys are encrypted at rest using a master encryption key
- Each key has a unique nonce (IV) for defense against IV reuse attacks
- Decryption is only possible with the master key
- Key prefixes/suffixes are stored separately for UI display

Security Model:
- Master key stored in ENCRYPTION_KEY env var (32 bytes, base64 encoded)
- Per-key nonces stored alongside encrypted data
- GCM provides both encryption and authentication
"""

import base64
import os
import secrets
from dataclasses import dataclass

import structlog
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = structlog.get_logger()


@dataclass
class EncryptedKey:
    """Encrypted API key with metadata for storage."""

    encrypted_data: str  # Base64 encoded: nonce + ciphertext + tag
    key_prefix: str  # First 8 chars for display (e.g., "sk-ant-a")
    key_suffix: str  # Last 4 chars for identification (e.g., "xyz9")


class KeyEncryptionService:
    """Service for encrypting and decrypting user API keys."""

    # Nonce size for AES-GCM (96 bits is recommended)
    NONCE_SIZE = 12

    def __init__(self, encryption_key: str | None = None):
        """Initialize with master encryption key.

        Args:
            encryption_key: Base64-encoded 32-byte key.
                           Falls back to ENCRYPTION_KEY env var.
        """
        key_b64 = encryption_key or os.getenv("ENCRYPTION_KEY")

        if not key_b64:
            logger.warning(
                "No ENCRYPTION_KEY configured - using fallback (NOT FOR PRODUCTION)"
            )
            # Generate a deterministic fallback for development only
            # In production, ENCRYPTION_KEY must be set
            key_b64 = base64.b64encode(b"dev-key-32-bytes-1234567890ab").decode()

        try:
            self._key = base64.b64decode(key_b64)
            if len(self._key) != 32:
                raise ValueError(f"Key must be 32 bytes, got {len(self._key)}")
            self._cipher = AESGCM(self._key)
        except Exception as e:
            logger.error("Failed to initialize encryption key", error=str(e))
            raise ValueError(
                "Invalid ENCRYPTION_KEY. Must be base64-encoded 32-byte key."
            ) from e

    def encrypt(self, plaintext: str) -> EncryptedKey:
        """Encrypt an API key.

        Args:
            plaintext: The API key to encrypt

        Returns:
            EncryptedKey with encrypted data and display metadata
        """
        if not plaintext:
            raise ValueError("Cannot encrypt empty key")

        # Generate random nonce
        nonce = secrets.token_bytes(self.NONCE_SIZE)

        # Encrypt with AES-256-GCM
        ciphertext = self._cipher.encrypt(
            nonce, plaintext.encode("utf-8"), associated_data=None
        )

        # Combine nonce + ciphertext for storage
        encrypted_blob = nonce + ciphertext
        encrypted_b64 = base64.b64encode(encrypted_blob).decode("utf-8")

        # Extract prefix and suffix for display
        key_prefix = plaintext[:8] if len(plaintext) >= 8 else plaintext
        key_suffix = plaintext[-4:] if len(plaintext) >= 4 else ""

        logger.debug(
            "Encrypted API key",
            key_prefix=key_prefix,
            encrypted_length=len(encrypted_b64),
        )

        return EncryptedKey(
            encrypted_data=encrypted_b64,
            key_prefix=key_prefix,
            key_suffix=key_suffix,
        )

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt an API key.

        Args:
            encrypted_data: Base64-encoded encrypted blob (nonce + ciphertext)

        Returns:
            Decrypted API key as string
        """
        if not encrypted_data:
            raise ValueError("Cannot decrypt empty data")

        try:
            # Decode base64
            encrypted_blob = base64.b64decode(encrypted_data)

            # Extract nonce and ciphertext
            nonce = encrypted_blob[: self.NONCE_SIZE]
            ciphertext = encrypted_blob[self.NONCE_SIZE :]

            # Decrypt
            plaintext = self._cipher.decrypt(nonce, ciphertext, associated_data=None)

            return plaintext.decode("utf-8")

        except Exception as e:
            logger.error("Failed to decrypt API key", error=str(e))
            raise ValueError("Decryption failed - key may be corrupted") from e

    def rotate_key(self, old_encrypted: str, new_key: bytes) -> EncryptedKey:
        """Re-encrypt data with a new master key.

        Used during key rotation to update all stored keys.

        Args:
            old_encrypted: Data encrypted with current key
            new_key: New 32-byte encryption key

        Returns:
            EncryptedKey encrypted with new key
        """
        # Decrypt with current key
        plaintext = self.decrypt(old_encrypted)

        # Create new cipher with new key
        new_cipher = AESGCM(new_key)
        nonce = secrets.token_bytes(self.NONCE_SIZE)

        # Re-encrypt
        ciphertext = new_cipher.encrypt(
            nonce, plaintext.encode("utf-8"), associated_data=None
        )

        encrypted_blob = nonce + ciphertext
        encrypted_b64 = base64.b64encode(encrypted_blob).decode("utf-8")

        return EncryptedKey(
            encrypted_data=encrypted_b64,
            key_prefix=plaintext[:8] if len(plaintext) >= 8 else plaintext,
            key_suffix=plaintext[-4:] if len(plaintext) >= 4 else "",
        )


# Singleton instance
_encryption_service: KeyEncryptionService | None = None


def get_encryption_service() -> KeyEncryptionService:
    """Get or create global key encryption service."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = KeyEncryptionService()
    return _encryption_service


def encrypt_api_key(api_key: str) -> EncryptedKey:
    """Encrypt an API key (convenience function)."""
    return get_encryption_service().encrypt(api_key)


def decrypt_api_key(encrypted_data: str) -> str:
    """Decrypt an API key (convenience function)."""
    return get_encryption_service().decrypt(encrypted_data)
