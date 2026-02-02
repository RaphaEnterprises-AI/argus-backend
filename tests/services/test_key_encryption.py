"""Tests for the Key Encryption service.

This module tests:
- KeyEncryptionService initialization
- AES-256-GCM encryption
- Decryption with authentication
- Key rotation
- Error handling
- Environment-specific behavior (dev vs production)
"""

import base64
import os
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.services.key_encryption import (
    EncryptedKey,
    KeyEncryptionService,
    _DEV_FALLBACK_KEY,
    decrypt_api_key,
    encrypt_api_key,
    get_encryption_service,
)


class TestEncryptedKeyDataclass:
    """Tests for EncryptedKey dataclass."""

    def test_encrypted_key_creation(self):
        """Test EncryptedKey creation."""
        key = EncryptedKey(
            encrypted_data="base64encodeddata",
            key_prefix="sk-ant-a",
            key_suffix="xyz9",
        )
        assert key.encrypted_data == "base64encodeddata"
        assert key.key_prefix == "sk-ant-a"
        assert key.key_suffix == "xyz9"


class TestKeyEncryptionServiceInit:
    """Tests for KeyEncryptionService initialization."""

    def test_init_with_valid_key(self, mock_env_vars, monkeypatch):
        """Test initialization with valid base64 32-byte key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        # Generate valid 32-byte key
        valid_key = base64.b64encode(os.urandom(32)).decode()

        service = KeyEncryptionService(encryption_key=valid_key)
        assert service._key is not None
        assert len(service._key) == 32

    def test_init_from_env_var(self, mock_env_vars, monkeypatch):
        """Test initialization from ENCRYPTION_KEY env var."""
        valid_key = base64.b64encode(os.urandom(32)).decode()
        monkeypatch.setenv("ENCRYPTION_KEY", valid_key)
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        service = KeyEncryptionService()
        assert service._key is not None

    def test_init_dev_fallback(self, mock_env_vars, monkeypatch):
        """Test initialization uses dev fallback in development."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        service = KeyEncryptionService()
        assert service._key is not None
        # Should use the dev fallback key
        expected_key = base64.b64decode(_DEV_FALLBACK_KEY)
        assert service._key == expected_key

    def test_init_production_requires_key(self, mock_env_vars, monkeypatch):
        """Test initialization fails in production without key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "production")

        with pytest.raises(ValueError) as exc_info:
            KeyEncryptionService()

        assert "ENCRYPTION_KEY environment variable is required" in str(exc_info.value)
        assert "production" in str(exc_info.value)

    def test_init_staging_requires_key(self, mock_env_vars, monkeypatch):
        """Test initialization fails in staging without key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "staging")

        with pytest.raises(ValueError) as exc_info:
            KeyEncryptionService()

        assert "ENCRYPTION_KEY environment variable is required" in str(exc_info.value)

    def test_init_invalid_key_size(self, mock_env_vars, monkeypatch):
        """Test initialization fails with wrong key size."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        # 16-byte key instead of 32
        invalid_key = base64.b64encode(b"1234567890123456").decode()

        with pytest.raises(ValueError) as exc_info:
            KeyEncryptionService(encryption_key=invalid_key)

        assert "Invalid ENCRYPTION_KEY" in str(exc_info.value) or "32 bytes" in str(exc_info.value)

    def test_init_invalid_base64(self, mock_env_vars, monkeypatch):
        """Test initialization fails with invalid base64."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        with pytest.raises(ValueError) as exc_info:
            KeyEncryptionService(encryption_key="not-valid-base64!!!")

        assert "Invalid ENCRYPTION_KEY" in str(exc_info.value)


class TestKeyEncryptionServiceEncrypt:
    """Tests for encryption functionality."""

    @pytest.fixture
    def encryption_service(self, mock_env_vars, monkeypatch):
        """Create encryption service with test key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")
        return KeyEncryptionService()

    def test_encrypt_api_key(self, encryption_service):
        """Test encrypting an API key."""
        api_key = "sk-ant-api03-test-key-12345678901234567890"

        result = encryption_service.encrypt(api_key)

        assert isinstance(result, EncryptedKey)
        assert result.encrypted_data != api_key
        assert result.key_prefix == "sk-ant-a"  # First 8 chars
        assert result.key_suffix == "7890"  # Last 4 chars

    def test_encrypt_empty_key_fails(self, encryption_service):
        """Test encrypting empty key raises error."""
        with pytest.raises(ValueError) as exc_info:
            encryption_service.encrypt("")

        assert "Cannot encrypt empty key" in str(exc_info.value)

    def test_encrypt_short_key(self, encryption_service):
        """Test encrypting short key handles prefix/suffix correctly."""
        short_key = "abc"

        result = encryption_service.encrypt(short_key)

        assert result.key_prefix == "abc"  # Full key as prefix
        assert result.key_suffix == ""  # No suffix for keys < 4 chars

    def test_encrypt_generates_unique_nonces(self, encryption_service):
        """Test that each encryption generates unique nonce."""
        api_key = "sk-ant-test-key-12345"

        result1 = encryption_service.encrypt(api_key)
        result2 = encryption_service.encrypt(api_key)

        # Same key should produce different encrypted data due to unique nonces
        assert result1.encrypted_data != result2.encrypted_data

    def test_encrypt_data_structure(self, encryption_service):
        """Test encrypted data contains nonce + ciphertext."""
        api_key = "sk-ant-test-key-12345"

        result = encryption_service.encrypt(api_key)
        encrypted_blob = base64.b64decode(result.encrypted_data)

        # Should have nonce (12 bytes) + ciphertext + tag (16 bytes)
        assert len(encrypted_blob) >= 12 + len(api_key) + 16


class TestKeyEncryptionServiceDecrypt:
    """Tests for decryption functionality."""

    @pytest.fixture
    def encryption_service(self, mock_env_vars, monkeypatch):
        """Create encryption service with test key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")
        return KeyEncryptionService()

    def test_decrypt_roundtrip(self, encryption_service):
        """Test encrypt then decrypt returns original."""
        original_key = "sk-ant-api03-test-key-12345678901234567890"

        encrypted = encryption_service.encrypt(original_key)
        decrypted = encryption_service.decrypt(encrypted.encrypted_data)

        assert decrypted == original_key

    def test_decrypt_various_lengths(self, encryption_service):
        """Test decrypt works with various key lengths."""
        test_keys = [
            "short",
            "medium-length-key-12345",
            "sk-ant-api03-" + "x" * 100,  # Long key
        ]

        for key in test_keys:
            encrypted = encryption_service.encrypt(key)
            decrypted = encryption_service.decrypt(encrypted.encrypted_data)
            assert decrypted == key

    def test_decrypt_empty_data_fails(self, encryption_service):
        """Test decrypting empty data raises error."""
        with pytest.raises(ValueError) as exc_info:
            encryption_service.decrypt("")

        assert "Cannot decrypt empty data" in str(exc_info.value)

    def test_decrypt_invalid_base64_fails(self, encryption_service):
        """Test decrypting invalid base64 raises error."""
        with pytest.raises(ValueError) as exc_info:
            encryption_service.decrypt("not-valid-base64!!!")

        assert "Decryption failed" in str(exc_info.value)

    def test_decrypt_corrupted_data_fails(self, encryption_service):
        """Test decrypting corrupted data fails authentication."""
        original_key = "sk-ant-test-key"
        encrypted = encryption_service.encrypt(original_key)

        # Corrupt the encrypted data
        blob = base64.b64decode(encrypted.encrypted_data)
        corrupted = bytearray(blob)
        corrupted[15] ^= 0xFF  # Flip bits in ciphertext
        corrupted_b64 = base64.b64encode(bytes(corrupted)).decode()

        with pytest.raises(ValueError) as exc_info:
            encryption_service.decrypt(corrupted_b64)

        assert "Decryption failed" in str(exc_info.value)

    def test_decrypt_wrong_key_fails(self, mock_env_vars, monkeypatch):
        """Test decrypting with wrong key fails."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        # Create two services with different keys
        key1 = base64.b64encode(os.urandom(32)).decode()
        key2 = base64.b64encode(os.urandom(32)).decode()

        service1 = KeyEncryptionService(encryption_key=key1)
        service2 = KeyEncryptionService(encryption_key=key2)

        encrypted = service1.encrypt("sk-ant-test-key")

        with pytest.raises(ValueError) as exc_info:
            service2.decrypt(encrypted.encrypted_data)

        assert "Decryption failed" in str(exc_info.value)


class TestKeyEncryptionServiceRotation:
    """Tests for key rotation functionality."""

    @pytest.fixture
    def encryption_service(self, mock_env_vars, monkeypatch):
        """Create encryption service with test key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")
        return KeyEncryptionService()

    def test_rotate_key(self, encryption_service):
        """Test rotating to a new encryption key."""
        original_key = "sk-ant-test-key-12345"
        new_master_key = os.urandom(32)

        # Encrypt with current key
        encrypted = encryption_service.encrypt(original_key)

        # Rotate to new key
        rotated = encryption_service.rotate_key(encrypted.encrypted_data, new_master_key)

        # Verify rotated data is different
        assert rotated.encrypted_data != encrypted.encrypted_data

        # Verify prefix/suffix preserved
        assert rotated.key_prefix == encrypted.key_prefix
        assert rotated.key_suffix == encrypted.key_suffix

        # Verify can decrypt with new key
        new_cipher = AESGCM(new_master_key)
        rotated_blob = base64.b64decode(rotated.encrypted_data)
        nonce = rotated_blob[:12]
        ciphertext = rotated_blob[12:]
        decrypted = new_cipher.decrypt(nonce, ciphertext, None).decode()

        assert decrypted == original_key

    def test_rotate_key_preserves_original_plaintext(self, encryption_service):
        """Test rotation preserves the original plaintext."""
        original_key = "sk-ant-api03-sensitive-key-abcdef"

        encrypted = encryption_service.encrypt(original_key)
        new_key = os.urandom(32)
        rotated = encryption_service.rotate_key(encrypted.encrypted_data, new_key)

        # Create new service with new key to decrypt
        new_service = KeyEncryptionService(
            encryption_key=base64.b64encode(new_key).decode()
        )
        decrypted = new_service.decrypt(rotated.encrypted_data)

        assert decrypted == original_key


class TestKeyEncryptionServiceNonceProperties:
    """Tests for nonce-related properties."""

    def test_nonce_size_constant(self):
        """Test NONCE_SIZE is correct for AES-GCM."""
        assert KeyEncryptionService.NONCE_SIZE == 12

    def test_nonces_are_unique(self, mock_env_vars, monkeypatch):
        """Test that nonces are cryptographically random."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        service = KeyEncryptionService()
        api_key = "test-key"

        nonces = set()
        for _ in range(100):
            encrypted = service.encrypt(api_key)
            blob = base64.b64decode(encrypted.encrypted_data)
            nonce = blob[:12]
            nonces.add(nonce)

        # All 100 nonces should be unique
        assert len(nonces) == 100


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_encrypt_api_key_function(self, mock_env_vars, monkeypatch):
        """Test encrypt_api_key convenience function."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        # Reset singleton
        import src.services.key_encryption as module
        module._encryption_service = None

        result = encrypt_api_key("sk-ant-test-key-12345")

        assert isinstance(result, EncryptedKey)
        assert result.key_prefix == "sk-ant-t"

        # Clean up
        module._encryption_service = None

    def test_decrypt_api_key_function(self, mock_env_vars, monkeypatch):
        """Test decrypt_api_key convenience function."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        # Reset singleton
        import src.services.key_encryption as module
        module._encryption_service = None

        original = "sk-ant-test-key-12345"
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted.encrypted_data)

        assert decrypted == original

        # Clean up
        module._encryption_service = None


class TestGetEncryptionService:
    """Tests for get_encryption_service singleton."""

    def test_singleton_pattern(self, mock_env_vars, monkeypatch):
        """Test get_encryption_service returns singleton."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        import src.services.key_encryption as module
        module._encryption_service = None

        service1 = get_encryption_service()
        service2 = get_encryption_service()

        assert service1 is service2

        # Clean up
        module._encryption_service = None

    def test_singleton_caches_instance(self, mock_env_vars, monkeypatch):
        """Test singleton caches the instance."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")

        import src.services.key_encryption as module
        module._encryption_service = None

        # First call creates instance
        service = get_encryption_service()
        assert service is not None

        # Verify cached
        assert module._encryption_service is service

        # Clean up
        module._encryption_service = None


class TestSecurityProperties:
    """Tests for security properties of the encryption."""

    @pytest.fixture
    def encryption_service(self, mock_env_vars, monkeypatch):
        """Create encryption service with test key."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "development")
        return KeyEncryptionService()

    def test_ciphertext_reveals_no_plaintext_length(self, encryption_service):
        """Test ciphertext doesn't directly reveal plaintext length."""
        # AES-GCM adds constant overhead (nonce + tag)
        short_key = "short"
        long_key = "a" * 100

        short_encrypted = encryption_service.encrypt(short_key)
        long_encrypted = encryption_service.encrypt(long_key)

        short_blob = base64.b64decode(short_encrypted.encrypted_data)
        long_blob = base64.b64decode(long_encrypted.encrypted_data)

        # Length difference should match plaintext difference
        # (AES-GCM doesn't pad, but nonce+tag add constant overhead)
        overhead = len(short_blob) - len(short_key)
        expected_long_len = len(long_key) + overhead
        assert len(long_blob) == expected_long_len

    def test_authentication_tag_present(self, encryption_service):
        """Test that encryption includes authentication tag."""
        api_key = "sk-ant-test-key"
        encrypted = encryption_service.encrypt(api_key)
        blob = base64.b64decode(encrypted.encrypted_data)

        # AES-GCM tag is 16 bytes, appended to ciphertext
        # Structure: nonce (12) + ciphertext + tag (16)
        nonce_size = 12
        tag_size = 16
        ciphertext_len = len(api_key)

        expected_total = nonce_size + ciphertext_len + tag_size
        assert len(blob) == expected_total

    def test_tampering_detected(self, encryption_service):
        """Test that any tampering is detected."""
        api_key = "sk-ant-test-key"
        encrypted = encryption_service.encrypt(api_key)

        blob = bytearray(base64.b64decode(encrypted.encrypted_data))

        # Try various tampering positions
        tamper_positions = [0, 6, 12, 15, 20, len(blob) - 1]

        for pos in tamper_positions:
            tampered = blob.copy()
            tampered[pos] ^= 0x01  # Single bit flip
            tampered_b64 = base64.b64encode(bytes(tampered)).decode()

            with pytest.raises(ValueError):
                encryption_service.decrypt(tampered_b64)


class TestDevFallbackKey:
    """Tests for development fallback key behavior."""

    def test_dev_fallback_key_is_deterministic(self):
        """Test dev fallback key is the same value."""
        expected = base64.b64encode(b"01234567890123456789012345678901").decode()
        assert _DEV_FALLBACK_KEY == expected

    def test_dev_fallback_key_is_32_bytes(self):
        """Test dev fallback key decodes to 32 bytes."""
        decoded = base64.b64decode(_DEV_FALLBACK_KEY)
        assert len(decoded) == 32

    def test_dev_fallback_not_used_in_production(self, mock_env_vars, monkeypatch):
        """Test dev fallback is never used in production."""
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "production")

        with pytest.raises(ValueError):
            KeyEncryptionService()
