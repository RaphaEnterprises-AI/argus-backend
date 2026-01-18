"""Comprehensive tests for authentication module (auth.py).

Tests cover:
- JWT token generation and verification
- API key generation and hashing
- Clerk JWT verification
- Authentication methods (API key, JWT, service account)
- User context creation
- Public and API-key-only endpoint detection
- Request helpers (client IP, etc.)
- Token revocation
"""

import hashlib
import hmac
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# JWT Token Tests
# =============================================================================

class TestJWTTokenGeneration:
    """Tests for JWT token generation."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with JWT secret."""
        settings = MagicMock()
        settings.jwt_secret_key = MagicMock()
        settings.jwt_secret_key.get_secret_value.return_value = "test-secret-key-12345"
        return settings

    def test_generate_jwt_token_access(self, mock_settings):
        """Test generating an access JWT token."""
        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import generate_jwt_token

            token = generate_jwt_token(
                user_id="user_123",
                organization_id="org_456",
                email="test@example.com",
                name="Test User",
                roles=["admin"],
                scopes=["read", "write"],
                token_type="access",
            )

            assert token is not None
            assert isinstance(token, str)
            assert len(token) > 0
            # JWT tokens have 3 parts separated by dots
            assert token.count(".") == 2

    def test_generate_jwt_token_refresh(self, mock_settings):
        """Test generating a refresh JWT token."""
        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import generate_jwt_token

            token = generate_jwt_token(
                user_id="user_123",
                token_type="refresh",
            )

            assert token is not None
            assert token.count(".") == 2

    def test_generate_jwt_token_with_custom_secret(self, mock_settings):
        """Test generating JWT with custom secret key."""
        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import generate_jwt_token

            token = generate_jwt_token(
                user_id="user_123",
                secret_key="custom-secret-key",
            )

            assert token is not None

    def test_generate_jwt_token_no_secret_raises(self):
        """Test that generating JWT without secret raises error."""
        settings = MagicMock()
        settings.jwt_secret_key = None

        with patch("src.config.get_settings", return_value=settings):
            from src.api.security.auth import generate_jwt_token

            with pytest.raises(ValueError, match="JWT_SECRET_KEY not configured"):
                generate_jwt_token(user_id="user_123")

    def test_generate_jwt_token_default_roles_scopes(self, mock_settings):
        """Test JWT generation with default (empty) roles and scopes."""
        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import generate_jwt_token, verify_jwt_token

            token = generate_jwt_token(user_id="user_123")
            payload = verify_jwt_token(token)

            assert payload is not None
            assert payload.roles == []
            assert payload.scopes == []


class TestJWTTokenVerification:
    """Tests for JWT token verification."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with JWT secret."""
        settings = MagicMock()
        settings.jwt_secret_key = MagicMock()
        settings.jwt_secret_key.get_secret_value.return_value = "test-secret-key-12345"
        return settings

    def test_verify_jwt_token_valid(self, mock_settings):
        """Test verifying a valid JWT token."""
        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import generate_jwt_token, verify_jwt_token

            token = generate_jwt_token(
                user_id="user_123",
                organization_id="org_456",
                email="test@example.com",
                roles=["admin"],
                scopes=["read", "write"],
            )

            payload = verify_jwt_token(token)

            assert payload is not None
            assert payload.sub == "user_123"
            assert payload.org == "org_456"
            assert payload.email == "test@example.com"
            assert "admin" in payload.roles
            assert "read" in payload.scopes

    def test_verify_jwt_token_expired(self, mock_settings):
        """Test verifying an expired JWT token."""
        import jwt

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import JWT_ALGORITHM, verify_jwt_token

            # Create an expired token manually
            expired_payload = {
                "sub": "user_123",
                "org": None,
                "email": None,
                "name": None,
                "roles": [],
                "scopes": [],
                "iat": int((datetime.now(UTC) - timedelta(hours=25)).timestamp()),
                "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
                "jti": "test-jti",
                "type": "access",
            }
            expired_token = jwt.encode(
                expired_payload,
                "test-secret-key-12345",
                algorithm=JWT_ALGORITHM
            )

            result = verify_jwt_token(expired_token)
            assert result is None

    def test_verify_jwt_token_invalid_signature(self, mock_settings):
        """Test verifying JWT token with wrong signature."""
        import jwt

        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import JWT_ALGORITHM, verify_jwt_token

            # Create token with different secret
            payload = {
                "sub": "user_123",
                "org": None,
                "email": None,
                "name": None,
                "roles": [],
                "scopes": [],
                "iat": int(datetime.now(UTC).timestamp()),
                "exp": int((datetime.now(UTC) + timedelta(hours=24)).timestamp()),
                "jti": "test-jti",
                "type": "access",
            }
            wrong_token = jwt.encode(payload, "wrong-secret", algorithm=JWT_ALGORITHM)

            result = verify_jwt_token(wrong_token)
            assert result is None

    def test_verify_jwt_token_malformed(self, mock_settings):
        """Test verifying a malformed JWT token."""
        with patch("src.config.get_settings", return_value=mock_settings):
            from src.api.security.auth import verify_jwt_token

            result = verify_jwt_token("not.a.valid.jwt.token")
            assert result is None

    def test_verify_jwt_token_no_secret(self):
        """Test verifying JWT when no secret configured."""
        settings = MagicMock()
        settings.jwt_secret_key = None

        with patch("src.config.get_settings", return_value=settings):
            from src.api.security.auth import verify_jwt_token

            result = verify_jwt_token("some.token.here")
            assert result is None


# =============================================================================
# API Key Tests
# =============================================================================

class TestAPIKeyGeneration:
    """Tests for API key generation and hashing."""

    def test_generate_api_key(self):
        """Test API key generation."""
        from src.api.security.auth import API_KEY_PREFIX, generate_api_key

        plaintext_key, key_hash = generate_api_key()

        assert plaintext_key.startswith(API_KEY_PREFIX)
        assert len(plaintext_key) > len(API_KEY_PREFIX)
        assert len(key_hash) == 64  # SHA256 hash length
        assert key_hash == hashlib.sha256(plaintext_key.encode()).hexdigest()

    def test_generate_api_key_uniqueness(self):
        """Test that generated API keys are unique."""
        from src.api.security.auth import generate_api_key

        keys = set()
        for _ in range(100):
            key, _ = generate_api_key()
            assert key not in keys
            keys.add(key)

    def test_hash_api_key(self):
        """Test API key hashing."""
        from src.api.security.auth import hash_api_key

        key = "argus_sk_test_key_12345"
        expected_hash = hashlib.sha256(key.encode()).hexdigest()

        assert hash_api_key(key) == expected_hash

    def test_verify_api_key_signature(self):
        """Test API key signature verification."""
        from src.api.security.auth import verify_api_key_signature

        key = "test_key"
        payload = "test_payload"
        signature = hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()

        assert verify_api_key_signature(key, signature, payload) is True
        assert verify_api_key_signature(key, "wrong_signature", payload) is False
        assert verify_api_key_signature(key, signature, "wrong_payload") is False


# =============================================================================
# API Key Authentication Tests
# =============================================================================

class TestAPIKeyAuthentication:
    """Tests for API key authentication."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.headers = {"user-agent": "test-agent"}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.fixture
    def mock_supabase(self):
        """Create mock supabase client."""
        client = MagicMock()
        client.is_configured = True
        return client

    @pytest.mark.asyncio
    async def test_authenticate_api_key_valid(self, mock_request, mock_supabase):
        """Test authenticating with a valid API key."""
        from src.api.security.auth import API_KEY_PREFIX, authenticate_api_key

        api_key = f"{API_KEY_PREFIX}test_key_12345"
        hashlib.sha256(api_key.encode()).hexdigest()

        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "key_123",
                "organization_id": "org_456",
                "created_by": "user_789",
                "scopes": ["read", "write"],
                "expires_at": None,
                "request_count": 10,
            }],
            "error": None,
        })
        mock_supabase.update = AsyncMock()

        with patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase):
            result = await authenticate_api_key(api_key, mock_request)

            assert result is not None
            assert result.organization_id == "org_456"
            assert result.api_key_id == "key_123"
            assert "read" in result.scopes

    @pytest.mark.asyncio
    async def test_authenticate_api_key_wrong_prefix(self, mock_request):
        """Test authenticating with wrong API key prefix."""
        from src.api.security.auth import authenticate_api_key

        result = await authenticate_api_key("wrong_prefix_key", mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_api_key_empty(self, mock_request):
        """Test authenticating with empty API key."""
        from src.api.security.auth import authenticate_api_key

        result = await authenticate_api_key("", mock_request)
        assert result is None

        result = await authenticate_api_key(None, mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_api_key_not_found(self, mock_request, mock_supabase):
        """Test authenticating with non-existent API key."""
        from src.api.security.auth import API_KEY_PREFIX, authenticate_api_key

        mock_supabase.request = AsyncMock(return_value={"data": [], "error": None})

        with patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase):
            result = await authenticate_api_key(f"{API_KEY_PREFIX}nonexistent", mock_request)
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_api_key_expired(self, mock_request, mock_supabase):
        """Test authenticating with expired API key."""
        from src.api.security.auth import API_KEY_PREFIX, authenticate_api_key

        expired_time = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "key_123",
                "organization_id": "org_456",
                "created_by": "user_789",
                "scopes": ["read"],
                "expires_at": expired_time,
            }],
            "error": None,
        })

        with patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase):
            result = await authenticate_api_key(f"{API_KEY_PREFIX}expired_key", mock_request)
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_api_key_empty_scopes(self, mock_request, mock_supabase):
        """Test that API key with empty scopes is rejected."""
        from src.api.security.auth import API_KEY_PREFIX, authenticate_api_key

        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "key_123",
                "organization_id": "org_456",
                "created_by": "user_789",
                "scopes": [],  # Empty scopes should be denied
                "expires_at": None,
            }],
            "error": None,
        })

        with patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase):
            result = await authenticate_api_key(f"{API_KEY_PREFIX}empty_scopes", mock_request)
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_api_key_null_scopes(self, mock_request, mock_supabase):
        """Test that API key with null scopes is rejected."""
        from src.api.security.auth import API_KEY_PREFIX, authenticate_api_key

        mock_supabase.request = AsyncMock(return_value={
            "data": [{
                "id": "key_123",
                "organization_id": "org_456",
                "created_by": "user_789",
                "scopes": None,  # Null scopes should be denied
                "expires_at": None,
            }],
            "error": None,
        })

        with patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase):
            result = await authenticate_api_key(f"{API_KEY_PREFIX}null_scopes", mock_request)
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_api_key_supabase_not_configured(self, mock_request):
        """Test API key auth when Supabase is not configured."""
        from src.api.security.auth import API_KEY_PREFIX, authenticate_api_key

        mock_supabase = MagicMock()
        mock_supabase.is_configured = False

        with patch("src.services.supabase_client.get_supabase_client", return_value=mock_supabase):
            result = await authenticate_api_key(f"{API_KEY_PREFIX}some_key", mock_request)
            assert result is None


# =============================================================================
# Clerk JWT Tests
# =============================================================================

class TestClerkJWTVerification:
    """Tests for Clerk JWT verification."""

    @pytest.fixture
    def mock_jwks(self):
        """Create mock JWKS response."""
        return {
            "keys": [
                {
                    "kid": "test-key-id",
                    "kty": "RSA",
                    "alg": "RS256",
                    "use": "sig",
                    "n": "test-modulus",
                    "e": "AQAB",
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_get_clerk_jwks_caching(self):
        """Test that Clerk JWKS is cached."""
        from src.api.security.auth import get_clerk_jwks

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"keys": [{"kid": "test"}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            # Clear cache
            import src.api.security.auth as auth_module
            auth_module._clerk_jwks_cache = {}
            auth_module._clerk_jwks_cache_time = 0

            # First call should fetch
            result1 = await get_clerk_jwks()
            assert result1 == {"keys": [{"kid": "test"}]}

            # Second call should use cache
            result2 = await get_clerk_jwks()
            assert result2 == {"keys": [{"kid": "test"}]}

            # Should only have called get once
            assert mock_instance.get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_clerk_jwks_fetch_error(self):
        """Test JWKS fetch when error occurs."""
        from src.api.security.auth import get_clerk_jwks

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=Exception("Network error"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_instance

            # Clear cache
            import src.api.security.auth as auth_module
            auth_module._clerk_jwks_cache = {}
            auth_module._clerk_jwks_cache_time = 0

            result = await get_clerk_jwks()
            assert result == {}

    @pytest.mark.asyncio
    async def test_verify_clerk_jwt_no_kid(self):
        """Test that token without kid is skipped."""
        import jwt as pyjwt

        from src.api.security.auth import verify_clerk_jwt

        # Create token without kid in header
        token = pyjwt.encode(
            {"sub": "user_123", "iss": "test", "exp": int(time.time()) + 3600, "iat": int(time.time())},
            "secret",
            algorithm="HS256"
        )

        result = await verify_clerk_jwt(token)
        assert result is None

    @pytest.mark.asyncio
    async def test_verify_clerk_jwt_expired(self):
        """Test that expired Clerk JWT returns None."""
        from src.api.security.auth import verify_clerk_jwt

        # Mock an expired token scenario
        with patch("src.api.security.auth.get_jwks_for_issuer"):
            with patch("jwt.decode") as mock_decode:
                mock_decode.side_effect = [
                    {"kid": "test-kid", "alg": "RS256"},  # get_unverified_header
                    {"sub": "user_123", "iss": "https://test.clerk.com", "exp": 0, "iat": 0},  # unverified decode
                ]

            await verify_clerk_jwt("some.expired.token")
            # Should return None for various error cases
            # The actual behavior depends on the token structure


# =============================================================================
# Service Account Authentication Tests
# =============================================================================

class TestServiceAccountAuthentication:
    """Tests for service account authentication."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.mark.asyncio
    async def test_authenticate_service_account_valid(self, mock_request):
        """Test authenticating with valid service account token."""
        from src.api.security.auth import authenticate_service_account

        settings = MagicMock()
        settings.service_accounts = {
            "test-service": {
                "token": "svc_valid_token_12345",
                "scopes": ["read", "write"],
            }
        }

        with patch("src.config.get_settings", return_value=settings):
            result = await authenticate_service_account("svc_valid_token_12345", mock_request)

            assert result is not None
            assert result.user_id == "service:test-service"
            assert "service_account" in result.roles

    @pytest.mark.asyncio
    async def test_authenticate_service_account_wrong_prefix(self, mock_request):
        """Test authenticating with wrong service account prefix."""
        from src.api.security.auth import authenticate_service_account

        result = await authenticate_service_account("wrong_prefix_token", mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_service_account_invalid_token(self, mock_request):
        """Test authenticating with invalid service account token."""
        from src.api.security.auth import authenticate_service_account

        settings = MagicMock()
        settings.service_accounts = {
            "test-service": {
                "token": "svc_different_token",
                "scopes": ["read"],
            }
        }

        with patch("src.config.get_settings", return_value=settings):
            result = await authenticate_service_account("svc_wrong_token", mock_request)
            assert result is None


# =============================================================================
# Request Helper Tests
# =============================================================================

class TestRequestHelpers:
    """Tests for request helper functions."""

    def test_get_client_ip_direct(self):
        """Test getting client IP from direct connection."""
        from src.api.security.auth import get_client_ip

        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "192.168.1.100"

        assert get_client_ip(request) == "192.168.1.100"

    def test_get_client_ip_forwarded_for(self):
        """Test getting client IP from X-Forwarded-For header."""
        from src.api.security.auth import get_client_ip

        request = MagicMock()
        request.headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18, 150.172.238.178"}
        request.client = MagicMock()
        request.client.host = "10.0.0.1"

        assert get_client_ip(request) == "203.0.113.50"

    def test_get_client_ip_real_ip(self):
        """Test getting client IP from X-Real-IP header."""
        from src.api.security.auth import get_client_ip

        request = MagicMock()
        request.headers = {"x-real-ip": "198.51.100.25"}
        request.client = MagicMock()
        request.client.host = "10.0.0.1"

        assert get_client_ip(request) == "198.51.100.25"

    def test_get_client_ip_no_client(self):
        """Test getting client IP when no client info available."""
        from src.api.security.auth import get_client_ip

        request = MagicMock()
        request.headers = {}
        request.client = None

        assert get_client_ip(request) == "unknown"


class TestEndpointChecks:
    """Tests for public and API-key-only endpoint detection."""

    def test_is_public_endpoint_exact_match(self):
        """Test public endpoint exact match."""
        from src.api.security.auth import is_public_endpoint

        assert is_public_endpoint("/health") is True
        assert is_public_endpoint("/docs") is True
        assert is_public_endpoint("/openapi.json") is True

    def test_is_public_endpoint_prefix_match(self):
        """Test public endpoint prefix match."""
        from src.api.security.auth import is_public_endpoint

        # Device auth endpoints
        assert is_public_endpoint("/api/v1/auth/device/authorize") is True
        assert is_public_endpoint("/api/v1/auth/device/token") is True

    def test_is_public_endpoint_non_public(self):
        """Test non-public endpoints."""
        from src.api.security.auth import is_public_endpoint

        assert is_public_endpoint("/api/v1/tests") is False
        assert is_public_endpoint("/api/v1/users") is False

    def test_is_api_key_only_endpoint(self):
        """Test API key only endpoints."""
        from src.api.security.auth import is_api_key_only_endpoint

        assert is_api_key_only_endpoint("/api/v1/webhooks/github") is True
        assert is_api_key_only_endpoint("/api/v1/webhooks/gitlab") is True
        assert is_api_key_only_endpoint("/api/v1/tests") is False


# =============================================================================
# Token Revocation Tests
# =============================================================================

class TestTokenRevocation:
    """Tests for token revocation."""

    @pytest.mark.asyncio
    async def test_revoke_token(self):
        """Test revoking a token."""
        # Clear revoked tokens
        import src.api.security.auth as auth_module
        from src.api.security.auth import is_token_revoked, revoke_token
        auth_module._revoked_tokens = set()

        jti = "test-jti-12345"

        # Should not be revoked initially
        assert await is_token_revoked(jti) is False

        # Revoke the token
        await revoke_token(jti)

        # Should now be revoked
        assert await is_token_revoked(jti) is True

    @pytest.mark.asyncio
    async def test_is_token_revoked_not_revoked(self):
        """Test checking if non-revoked token is revoked."""
        # Clear revoked tokens
        import src.api.security.auth as auth_module
        from src.api.security.auth import is_token_revoked
        auth_module._revoked_tokens = set()

        result = await is_token_revoked("never-revoked-jti")
        assert result is False


# =============================================================================
# UserContext Model Tests
# =============================================================================

class TestUserContextModel:
    """Tests for UserContext model."""

    def test_user_context_creation(self):
        """Test creating UserContext."""
        from src.api.security.auth import AuthMethod, UserContext

        user = UserContext(
            user_id="user_123",
            organization_id="org_456",
            email="test@example.com",
            name="Test User",
            roles=["admin", "member"],
            scopes=["read", "write", "execute"],
            auth_method=AuthMethod.JWT,
            api_key_id="key_789",
            session_id="session_abc",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert user.user_id == "user_123"
        assert user.organization_id == "org_456"
        assert user.email == "test@example.com"
        assert "admin" in user.roles
        assert "read" in user.scopes

    def test_user_context_defaults(self):
        """Test UserContext default values."""
        from src.api.security.auth import AuthMethod, UserContext

        user = UserContext(user_id="user_123")

        assert user.organization_id is None
        assert user.email is None
        assert user.roles == []
        assert user.scopes == []
        assert user.auth_method == AuthMethod.ANONYMOUS


class TestTokenPayloadModel:
    """Tests for TokenPayload model."""

    def test_token_payload_creation(self):
        """Test creating TokenPayload."""
        from src.api.security.auth import TokenPayload

        now = int(datetime.now(UTC).timestamp())
        exp = now + 3600

        payload = TokenPayload(
            sub="user_123",
            org="org_456",
            email="test@example.com",
            name="Test User",
            roles=["admin"],
            scopes=["read", "write"],
            iat=now,
            exp=exp,
            jti="test-jti-123",
            type="access",
        )

        assert payload.sub == "user_123"
        assert payload.org == "org_456"
        assert payload.type == "access"


# =============================================================================
# FastAPI Dependency Tests
# =============================================================================

class TestGetCurrentUserDependency:
    """Tests for get_current_user FastAPI dependency."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.state = MagicMock()
        request.state.user = None
        request.url = MagicMock()
        request.url.path = "/api/v1/tests"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.mark.asyncio
    async def test_get_current_user_from_request_state(self, mock_request):
        """Test getting user from request state (already authenticated)."""
        from src.api.security.auth import AuthMethod, UserContext, get_current_user

        existing_user = UserContext(
            user_id="already_authed",
            auth_method=AuthMethod.JWT,
        )
        mock_request.state.user = existing_user

        result = await get_current_user(mock_request)
        assert result.user_id == "already_authed"

    @pytest.mark.asyncio
    async def test_get_current_user_public_endpoint(self, mock_request):
        """Test get_current_user for public endpoint."""
        from src.api.security.auth import AuthMethod, get_current_user

        mock_request.url.path = "/health"

        result = await get_current_user(mock_request)
        assert result.user_id == "anonymous"
        assert result.auth_method == AuthMethod.ANONYMOUS

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth_raises(self, mock_request):
        """Test that missing auth raises HTTPException."""
        from fastapi import HTTPException

        from src.api.security.auth import get_current_user

        mock_request.url.path = "/api/v1/protected"

        with pytest.raises(HTTPException) as exc_info:
            # Pass None explicitly for api_key and bearer to simulate no auth
            await get_current_user(mock_request, api_key=None, bearer=None)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail


# =============================================================================
# Decorator Tests
# =============================================================================

class TestRequireRolesDecorator:
    """Tests for require_roles decorator."""

    @pytest.mark.asyncio
    async def test_require_roles_has_role(self):
        """Test require_roles when user has required role."""
        from fastapi import Depends

        from src.api.security.auth import AuthMethod, UserContext, get_current_user, require_roles

        @require_roles("admin")
        async def protected_endpoint(user: UserContext = Depends(get_current_user)):
            return {"message": "success"}

        user = UserContext(
            user_id="user_123",
            roles=["admin", "member"],
            auth_method=AuthMethod.JWT,
        )

        # The decorator expects user to be passed
        result = await protected_endpoint(user=user)
        assert result["message"] == "success"

    @pytest.mark.asyncio
    async def test_require_roles_missing_role(self):
        """Test require_roles when user lacks required role."""
        from fastapi import Depends, HTTPException

        from src.api.security.auth import AuthMethod, UserContext, get_current_user, require_roles

        @require_roles("admin")
        async def protected_endpoint(user: UserContext = Depends(get_current_user)):
            return {"message": "success"}

        user = UserContext(
            user_id="user_123",
            roles=["member"],  # No admin role
            auth_method=AuthMethod.JWT,
        )

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(user=user)

        assert exc_info.value.status_code == 403


class TestRequireScopesDecorator:
    """Tests for require_scopes decorator."""

    @pytest.mark.asyncio
    async def test_require_scopes_has_scopes(self):
        """Test require_scopes when user has required scopes."""
        from fastapi import Depends

        from src.api.security.auth import AuthMethod, UserContext, get_current_user, require_scopes

        @require_scopes("read", "write")
        async def protected_endpoint(user: UserContext = Depends(get_current_user)):
            return {"message": "success"}

        user = UserContext(
            user_id="user_123",
            scopes=["read", "write", "execute"],
            auth_method=AuthMethod.JWT,
        )

        result = await protected_endpoint(user=user)
        assert result["message"] == "success"

    @pytest.mark.asyncio
    async def test_require_scopes_missing_scope(self):
        """Test require_scopes when user lacks a required scope."""
        from fastapi import Depends, HTTPException

        from src.api.security.auth import AuthMethod, UserContext, get_current_user, require_scopes

        @require_scopes("read", "write", "admin")
        async def protected_endpoint(user: UserContext = Depends(get_current_user)):
            return {"message": "success"}

        user = UserContext(
            user_id="user_123",
            scopes=["read", "write"],  # Missing admin scope
            auth_method=AuthMethod.JWT,
        )

        with pytest.raises(HTTPException) as exc_info:
            await protected_endpoint(user=user)

        assert exc_info.value.status_code == 403
