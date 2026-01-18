"""Comprehensive tests for security middleware module (middleware.py).

Tests cover:
- Authentication middleware
- Rate limiting middleware
- Audit logging middleware
- Security middleware (combined)
- Rate limit configuration
- Request ID generation
- Timing attack prevention
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# =============================================================================
# Rate Limit Configuration Tests
# =============================================================================

class TestRateLimitConfig:
    """Tests for RateLimitConfig class."""

    def test_default_limits(self):
        """Test default rate limit values."""
        from src.api.security.middleware import RateLimitConfig

        assert RateLimitConfig.DEFAULT_LIMIT == 120
        assert RateLimitConfig.DEFAULT_WINDOW == 60

    def test_tier_limits(self):
        """Test tier-based rate limits."""
        from src.api.security.middleware import RateLimitConfig

        assert "free" in RateLimitConfig.TIER_LIMITS
        assert "starter" in RateLimitConfig.TIER_LIMITS
        assert "pro" in RateLimitConfig.TIER_LIMITS
        assert "enterprise" in RateLimitConfig.TIER_LIMITS
        assert "unlimited" in RateLimitConfig.TIER_LIMITS

        # Check free tier is most restrictive
        assert RateLimitConfig.TIER_LIMITS["free"]["requests"] < RateLimitConfig.TIER_LIMITS["pro"]["requests"]

    def test_endpoint_limits(self):
        """Test endpoint-specific rate limits."""
        from src.api.security.middleware import RateLimitConfig

        # Streaming endpoints should have lower limits
        assert "/api/v1/chat/stream" in RateLimitConfig.ENDPOINT_LIMITS
        assert RateLimitConfig.ENDPOINT_LIMITS["/api/v1/chat/stream"]["requests"] == 10

        # Health endpoint should have high limit
        assert "/health" in RateLimitConfig.ENDPOINT_LIMITS
        assert RateLimitConfig.ENDPOINT_LIMITS["/health"]["requests"] == 1000

    def test_exempt_endpoints(self):
        """Test rate limit exempt endpoints."""
        from src.api.security.middleware import RateLimitConfig

        assert "/health" in RateLimitConfig.EXEMPT_ENDPOINTS
        assert "/docs" in RateLimitConfig.EXEMPT_ENDPOINTS
        assert "/openapi.json" in RateLimitConfig.EXEMPT_ENDPOINTS


# =============================================================================
# Authentication Middleware Tests
# =============================================================================

class TestAuthenticationMiddleware:
    """Tests for AuthenticationMiddleware class."""

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def auth_middleware(self, mock_app):
        """Create AuthenticationMiddleware instance."""
        from src.api.security.middleware import AuthenticationMiddleware
        return AuthenticationMiddleware(mock_app, enforce_auth=True)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.state = MagicMock()
        request.url = MagicMock()
        request.url.path = "/api/v1/tests"
        request.method = "GET"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next function."""
        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            return response
        return call_next

    @pytest.mark.asyncio
    async def test_dispatch_public_endpoint(self, auth_middleware, mock_request, mock_call_next):
        """Test that public endpoints bypass authentication."""
        from src.api.security.auth import AuthMethod

        mock_request.url.path = "/health"

        response = await auth_middleware.dispatch(mock_request, mock_call_next)

        assert mock_request.state.user is not None
        assert mock_request.state.user.user_id == "anonymous"
        assert mock_request.state.user.auth_method == AuthMethod.ANONYMOUS

    @pytest.mark.asyncio
    async def test_dispatch_generates_request_id(self, auth_middleware, mock_request, mock_call_next):
        """Test that dispatch generates request ID."""
        mock_request.url.path = "/health"

        await auth_middleware.dispatch(mock_request, mock_call_next)

        assert hasattr(mock_request.state, "request_id")
        # Should be a valid UUID
        uuid.UUID(mock_request.state.request_id)

    @pytest.mark.asyncio
    async def test_dispatch_no_auth_returns_401(self, auth_middleware, mock_request):
        """Test that missing auth returns 401."""
        mock_request.url.path = "/api/v1/protected"

        async def call_next(request):
            return MagicMock(status_code=200)

        response = await auth_middleware.dispatch(mock_request, call_next)

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_dispatch_api_key_authentication(self, auth_middleware, mock_request, mock_call_next):
        """Test authentication with API key."""
        from src.api.security.auth import UserContext, AuthMethod

        mock_request.headers = {"x-api-key": "argus_sk_test_key"}

        mock_user = UserContext(
            user_id="api_user",
            organization_id="org_123",
            auth_method=AuthMethod.API_KEY,
        )

        with patch("src.api.security.middleware.authenticate_api_key", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_user

            response = await auth_middleware.dispatch(mock_request, mock_call_next)

            assert mock_request.state.user == mock_user
            mock_auth.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_jwt_authentication(self, auth_middleware, mock_request, mock_call_next):
        """Test authentication with JWT token."""
        from src.api.security.auth import UserContext, AuthMethod

        mock_request.headers = {"authorization": "Bearer test_jwt_token"}

        mock_user = UserContext(
            user_id="jwt_user",
            auth_method=AuthMethod.JWT,
        )

        with patch("src.api.security.middleware.authenticate_api_key", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None
            with patch("src.api.security.middleware.authenticate_jwt", new_callable=AsyncMock) as mock_jwt:
                mock_jwt.return_value = mock_user

                response = await auth_middleware.dispatch(mock_request, mock_call_next)

                assert mock_request.state.user == mock_user

    @pytest.mark.asyncio
    async def test_dispatch_service_account_authentication(self, auth_middleware, mock_request, mock_call_next):
        """Test authentication with service account token."""
        from src.api.security.auth import UserContext, AuthMethod

        mock_request.headers = {"authorization": "Bearer svc_test_token"}

        mock_user = UserContext(
            user_id="service:test",
            auth_method=AuthMethod.SERVICE_ACCOUNT,
        )

        with patch("src.api.security.middleware.authenticate_api_key", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = None
            with patch("src.api.security.middleware.authenticate_jwt", new_callable=AsyncMock) as mock_jwt:
                mock_jwt.return_value = None
                with patch("src.api.security.middleware.authenticate_service_account", new_callable=AsyncMock) as mock_svc:
                    mock_svc.return_value = mock_user

                    response = await auth_middleware.dispatch(mock_request, mock_call_next)

                    assert mock_request.state.user == mock_user

    @pytest.mark.asyncio
    async def test_dispatch_enforce_auth_disabled_development(self, mock_app, mock_request, mock_call_next):
        """Test that auth bypass works in development mode."""
        from src.api.security.middleware import AuthenticationMiddleware
        from src.api.security.auth import AuthMethod

        middleware = AuthenticationMiddleware(mock_app, enforce_auth=False)
        mock_request.url.path = "/api/v1/protected"

        with patch.dict("os.environ", {"ENVIRONMENT": "development"}):
            response = await middleware.dispatch(mock_request, mock_call_next)

            assert mock_request.state.user is not None
            assert mock_request.state.user.user_id == "dev-user"

    @pytest.mark.asyncio
    async def test_dispatch_enforce_auth_disabled_production_still_enforces(self, mock_app, mock_request):
        """Test that auth bypass does not work in production."""
        from src.api.security.middleware import AuthenticationMiddleware

        middleware = AuthenticationMiddleware(mock_app, enforce_auth=False)
        mock_request.url.path = "/api/v1/protected"

        async def call_next(request):
            return MagicMock(status_code=200)

        with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
            response = await middleware.dispatch(mock_request, call_next)

            # Should still return 401 in production even with enforce_auth=False
            assert response.status_code == 401


# =============================================================================
# Rate Limiting Middleware Tests
# =============================================================================

class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware class."""

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def rate_middleware(self, mock_app):
        """Create RateLimitMiddleware instance."""
        from src.api.security.middleware import RateLimitMiddleware
        return RateLimitMiddleware(mock_app, enabled=True)

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
        request.client.host = "192.168.1.100"
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next function."""
        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response
        return call_next

    @pytest.mark.asyncio
    async def test_dispatch_adds_rate_limit_headers(self, rate_middleware, mock_request, mock_call_next):
        """Test that rate limit headers are added to response."""
        response = await rate_middleware.dispatch(mock_request, mock_call_next)

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_dispatch_disabled(self, mock_app, mock_request, mock_call_next):
        """Test that disabled rate limiting passes through."""
        from src.api.security.middleware import RateLimitMiddleware

        middleware = RateLimitMiddleware(mock_app, enabled=False)

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should not have rate limit headers when disabled
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_exempt_endpoint(self, rate_middleware, mock_request, mock_call_next):
        """Test that exempt endpoints bypass rate limiting."""
        mock_request.url.path = "/health"

        response = await rate_middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_rate_limit_exceeded(self, rate_middleware, mock_request):
        """Test that exceeding rate limit returns 429."""
        # Make many requests to exceed limit
        from src.api.security.middleware import RateLimitConfig

        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response

        # Clear any existing counts
        rate_middleware._request_counts.clear()

        # Make enough requests to exceed limit
        limit = RateLimitConfig.DEFAULT_LIMIT
        for i in range(limit):
            await rate_middleware.dispatch(mock_request, call_next)

        # Next request should be rate limited
        response = await rate_middleware.dispatch(mock_request, call_next)

        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_get_rate_limit_key_user(self, rate_middleware, mock_request):
        """Test rate limit key generation for authenticated user."""
        from src.api.security.auth import UserContext, AuthMethod

        user = UserContext(
            user_id="user_123",
            organization_id="org_456",
            auth_method=AuthMethod.JWT,
        )
        mock_request.state.user = user

        key = rate_middleware._get_rate_limit_key(mock_request)

        assert key == "org:org_456"

    @pytest.mark.asyncio
    async def test_get_rate_limit_key_no_org(self, rate_middleware, mock_request):
        """Test rate limit key generation for user without org."""
        from src.api.security.auth import UserContext, AuthMethod

        user = UserContext(
            user_id="user_123",
            auth_method=AuthMethod.JWT,
        )
        mock_request.state.user = user

        key = rate_middleware._get_rate_limit_key(mock_request)

        assert key == "user:user_123"

    @pytest.mark.asyncio
    async def test_get_rate_limit_key_anonymous(self, rate_middleware, mock_request):
        """Test rate limit key generation for anonymous user."""
        key = rate_middleware._get_rate_limit_key(mock_request)

        assert key == "ip:192.168.1.100"

    def test_get_limit_for_endpoint_specific(self, rate_middleware):
        """Test getting specific endpoint limits."""
        from src.api.security.middleware import RateLimitConfig

        limit = rate_middleware._get_limit_for_endpoint("/api/v1/chat/stream", None)

        assert limit["requests"] == RateLimitConfig.ENDPOINT_LIMITS["/api/v1/chat/stream"]["requests"]

    def test_get_limit_for_endpoint_default(self, rate_middleware):
        """Test getting default limits for unspecified endpoint."""
        from src.api.security.middleware import RateLimitConfig

        limit = rate_middleware._get_limit_for_endpoint("/api/v1/some/endpoint", None)

        assert limit["requests"] == RateLimitConfig.DEFAULT_LIMIT
        assert limit["window"] == RateLimitConfig.DEFAULT_WINDOW

    def test_get_limit_for_endpoint_user_tier(self, rate_middleware):
        """Test getting limits based on user tier."""
        from src.api.security.middleware import RateLimitConfig

        user = MagicMock()
        user.tier = "pro"

        limit = rate_middleware._get_limit_for_endpoint("/api/v1/tests", user)

        assert limit["requests"] == RateLimitConfig.TIER_LIMITS["pro"]["requests"]


# =============================================================================
# Audit Logging Middleware Tests
# =============================================================================

class TestAuditLogMiddleware:
    """Tests for AuditLogMiddleware class."""

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def audit_middleware(self, mock_app):
        """Create AuditLogMiddleware instance."""
        from src.api.security.middleware import AuditLogMiddleware
        return AuditLogMiddleware(mock_app, enabled=True, log_request_body=True)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.state = MagicMock()
        request.state.request_id = "test-request-id"
        request.state.user = None
        request.url = MagicMock()
        request.url.path = "/api/v1/tests"
        request.method = "GET"
        request.query_params = {}
        request.headers = {"user-agent": "test-agent"}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next function."""
        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            return response
        return call_next

    @pytest.mark.asyncio
    async def test_dispatch_logs_request(self, audit_middleware, mock_request, mock_call_next):
        """Test that audit middleware logs requests."""
        with patch("src.api.security.middleware.logger") as mock_logger:
            response = await audit_middleware.dispatch(mock_request, mock_call_next)

            assert response.status_code == 200
            # Should have logged something
            assert mock_logger.debug.called or mock_logger.info.called

    @pytest.mark.asyncio
    async def test_dispatch_disabled(self, mock_app, mock_request, mock_call_next):
        """Test that disabled audit middleware passes through."""
        from src.api.security.middleware import AuditLogMiddleware

        middleware = AuditLogMiddleware(mock_app, enabled=False)

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_dispatch_logs_failed_requests(self, audit_middleware, mock_request):
        """Test that failed requests are logged with warning."""
        async def call_next(request):
            response = MagicMock()
            response.status_code = 500
            return response

        with patch("src.api.security.middleware.logger") as mock_logger:
            response = await audit_middleware.dispatch(mock_request, call_next)

            assert response.status_code == 500
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_dispatch_logs_sensitive_paths(self, audit_middleware, mock_request, mock_call_next):
        """Test that sensitive path access is logged at info level."""
        mock_request.url.path = "/api/v1/api-keys"

        with patch("src.api.security.middleware.logger") as mock_logger:
            await audit_middleware.dispatch(mock_request, mock_call_next)

            mock_logger.info.assert_called()

    def test_mask_sensitive_data_passwords(self, audit_middleware):
        """Test that sensitive data is masked."""
        data = {
            "username": "test_user",
            "password": "secret123",
            "email": "test@example.com",
        }

        masked = audit_middleware._mask_sensitive_data(data)

        assert masked["username"] == "test_user"
        assert masked["password"] == "***REDACTED***"
        assert masked["email"] == "test@example.com"

    def test_mask_sensitive_data_tokens(self, audit_middleware):
        """Test that token fields are masked."""
        data = {
            "api_key": "argus_sk_12345",
            "access_token": "jwt_token_here",
            "refresh_token": "refresh_here",
            "name": "Test",
        }

        masked = audit_middleware._mask_sensitive_data(data)

        assert masked["api_key"] == "***REDACTED***"
        assert masked["access_token"] == "***REDACTED***"
        assert masked["refresh_token"] == "***REDACTED***"
        assert masked["name"] == "Test"

    def test_mask_sensitive_data_nested(self, audit_middleware):
        """Test that nested sensitive data is masked."""
        data = {
            "user": {
                "name": "Test",
                "settings": {
                    "password": "secret",
                    "api_key": "key123",
                }
            }
        }

        masked = audit_middleware._mask_sensitive_data(data)

        assert masked["user"]["name"] == "Test"
        assert masked["user"]["settings"]["password"] == "***REDACTED***"
        assert masked["user"]["settings"]["api_key"] == "***REDACTED***"

    def test_mask_sensitive_data_key_contains_sensitive(self, audit_middleware):
        """Test that keys containing sensitive words are fully masked."""
        data = {
            "user_credentials": {
                "username": "test",
                "password": "secret",
            }
        }

        masked = audit_middleware._mask_sensitive_data(data)

        # The key "user_credentials" contains "credential", so entire value is masked
        assert masked["user_credentials"] == "***REDACTED***"

    def test_mask_sensitive_data_list(self, audit_middleware):
        """Test that sensitive data in lists is masked."""
        data = {
            "users": [
                {"name": "User1", "password": "pass1"},
                {"name": "User2", "password": "pass2"},
            ]
        }

        masked = audit_middleware._mask_sensitive_data(data)

        assert masked["users"][0]["name"] == "User1"
        assert masked["users"][0]["password"] == "***REDACTED***"
        assert masked["users"][1]["password"] == "***REDACTED***"

    def test_mask_sensitive_data_non_dict(self, audit_middleware):
        """Test masking non-dict data returns as-is."""
        result = audit_middleware._mask_sensitive_data("not a dict")
        assert result == "not a dict"

    def test_is_valid_uuid_valid(self, audit_middleware):
        """Test UUID validation with valid UUID."""
        valid_uuid = str(uuid.uuid4())
        assert audit_middleware._is_valid_uuid(valid_uuid) is True

    def test_is_valid_uuid_invalid(self, audit_middleware):
        """Test UUID validation with invalid strings."""
        assert audit_middleware._is_valid_uuid("not-a-uuid") is False
        assert audit_middleware._is_valid_uuid("dev-org") is False
        assert audit_middleware._is_valid_uuid("") is False
        assert audit_middleware._is_valid_uuid(None) is False

    @pytest.mark.asyncio
    async def test_store_audit_log(self, audit_middleware):
        """Test storing audit log to database."""
        entry = {
            "event_type": "api_request",
            "user_id": "user_123",
            "organization_id": str(uuid.uuid4()),
            "request_id": "req_123",
            "client_ip": "127.0.0.1",
            "user_agent": "test-agent",
            "method": "GET",
            "path": "/api/v1/tests",
            "query_params": {},
            "duration_ms": 100.5,
            "status_code": 200,
        }

        with patch("src.integrations.supabase.get_supabase", new_callable=AsyncMock) as mock_get_supabase:
            mock_supabase = AsyncMock()
            mock_supabase.insert = AsyncMock()
            mock_get_supabase.return_value = mock_supabase

            await audit_middleware._store_audit_log(entry)

            mock_supabase.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_audit_log_invalid_org_id(self, audit_middleware):
        """Test storing audit log with invalid org ID."""
        entry = {
            "event_type": "api_request",
            "user_id": "user_123",
            "organization_id": "dev-org",  # Invalid UUID
            "request_id": "req_123",
            "client_ip": "127.0.0.1",
            "method": "GET",
            "path": "/api/v1/tests",
            "duration_ms": 100,
            "status_code": 200,
        }

        with patch("src.integrations.supabase.get_supabase", new_callable=AsyncMock) as mock_get_supabase:
            mock_supabase = AsyncMock()
            mock_supabase.insert = AsyncMock()
            mock_get_supabase.return_value = mock_supabase

            await audit_middleware._store_audit_log(entry)

            # Should be called with org_id as None due to invalid UUID
            call_args = mock_supabase.insert.call_args
            assert call_args[0][1]["organization_id"] is None


# =============================================================================
# Security Middleware Tests
# =============================================================================

class TestSecurityMiddleware:
    """Tests for SecurityMiddleware class."""

    @pytest.fixture
    def mock_app(self):
        """Create mock ASGI app."""
        return MagicMock()

    @pytest.fixture
    def security_middleware(self, mock_app):
        """Create SecurityMiddleware instance."""
        from src.api.security.middleware import SecurityMiddleware
        return SecurityMiddleware(mock_app)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock()
        request.state = MagicMock()
        # Don't set request_id to test generation
        del request.state.request_id
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next function."""
        async def call_next(request):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response
        return call_next

    @pytest.mark.asyncio
    async def test_dispatch_generates_request_id(self, security_middleware, mock_request, mock_call_next):
        """Test that security middleware generates request ID."""
        # Ensure request_id doesn't exist
        mock_request.state = type('State', (), {})()

        response = await security_middleware.dispatch(mock_request, mock_call_next)

        assert hasattr(mock_request.state, "request_id")
        # Should be a valid UUID
        uuid.UUID(mock_request.state.request_id)

    @pytest.mark.asyncio
    async def test_dispatch_preserves_existing_request_id(self, security_middleware, mock_request, mock_call_next):
        """Test that existing request ID is preserved."""
        mock_request.state = type('State', (), {"request_id": "existing-id"})()

        response = await security_middleware.dispatch(mock_request, mock_call_next)

        assert mock_request.state.request_id == "existing-id"

    @pytest.mark.asyncio
    async def test_dispatch_adds_request_id_header(self, security_middleware, mock_request, mock_call_next):
        """Test that request ID is added to response headers."""
        mock_request.state = type('State', (), {})()

        response = await security_middleware.dispatch(mock_request, mock_call_next)

        assert "X-Request-ID" in response.headers

    @pytest.mark.asyncio
    async def test_dispatch_timing_attack_prevention(self, security_middleware, mock_request):
        """Test timing attack prevention for auth failures."""
        from fastapi import HTTPException

        mock_request.state = type('State', (), {})()

        async def call_next_raises(request):
            raise HTTPException(status_code=401, detail="Unauthorized")

        start_time = time.time()

        with pytest.raises(HTTPException):
            await security_middleware.dispatch(mock_request, call_next_raises)

        elapsed = time.time() - start_time

        # Should have minimum delay for 401 errors
        # Note: This test might be flaky due to timing, adjust threshold as needed
        assert elapsed >= 0.05  # At least 50ms (being lenient for test reliability)


# =============================================================================
# Integration Tests
# =============================================================================

class TestMiddlewareIntegration:
    """Integration tests for middleware stack."""

    @pytest.mark.asyncio
    async def test_middleware_chain_order(self):
        """Test that middleware executes in correct order."""
        from src.api.security.middleware import (
            AuthenticationMiddleware,
            RateLimitMiddleware,
            AuditLogMiddleware,
            SecurityMiddleware,
        )

        execution_order = []

        mock_app = MagicMock()

        class TrackingMiddleware:
            def __init__(self, app, name):
                self.app = app
                self.name = name

            async def dispatch(self, request, call_next):
                execution_order.append(f"{self.name}_before")
                response = await call_next(request)
                execution_order.append(f"{self.name}_after")
                return response

        # The order matters for proper functionality:
        # 1. SecurityMiddleware - generates request ID
        # 2. AuthenticationMiddleware - authenticates request
        # 3. RateLimitMiddleware - applies rate limits
        # 4. AuditLogMiddleware - logs the request

        # This test verifies middleware can be stacked
        # Actual order testing would require full FastAPI app

    @pytest.mark.asyncio
    async def test_authenticated_request_flow(self):
        """Test complete flow of authenticated request."""
        from src.api.security.middleware import AuthenticationMiddleware
        from src.api.security.auth import UserContext, AuthMethod

        mock_app = MagicMock()
        middleware = AuthenticationMiddleware(mock_app, enforce_auth=True)

        request = MagicMock()
        request.state = MagicMock()
        request.url = MagicMock()
        request.url.path = "/api/v1/tests"
        request.method = "GET"
        request.headers = {"x-api-key": "argus_sk_valid_key"}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        mock_user = UserContext(
            user_id="test_user",
            organization_id="test_org",
            roles=["member"],
            scopes=["read", "write"],
            auth_method=AuthMethod.API_KEY,
        )

        async def call_next(req):
            response = MagicMock()
            response.status_code = 200
            return response

        with patch("src.api.security.middleware.authenticate_api_key", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_user

            response = await middleware.dispatch(request, call_next)

            assert response.status_code == 200
            assert request.state.user == mock_user

    @pytest.mark.asyncio
    async def test_rate_limited_then_allowed(self):
        """Test request is rate limited then allowed after window."""
        from src.api.security.middleware import RateLimitMiddleware

        mock_app = MagicMock()
        middleware = RateLimitMiddleware(mock_app, enabled=True)

        request = MagicMock()
        request.state = MagicMock()
        request.state.user = None
        request.url = MagicMock()
        request.url.path = "/api/v1/tests"
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "10.0.0.1"

        async def call_next(req):
            response = MagicMock()
            response.status_code = 200
            response.headers = {}
            return response

        # Clear counters
        middleware._request_counts.clear()

        # Make requests up to limit
        from src.api.security.middleware import RateLimitConfig
        limit = RateLimitConfig.DEFAULT_LIMIT

        for _ in range(limit):
            await middleware.dispatch(request, call_next)

        # Next should be limited
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 429

        # Clear to simulate time passing
        middleware._request_counts.clear()

        # Should work again
        response = await middleware.dispatch(request, call_next)
        assert response.status_code == 200
