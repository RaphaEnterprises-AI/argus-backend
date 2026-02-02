"""
Comprehensive tests for the distributed rate limiting service.

Tests cover:
- Sliding window algorithm correctness
- Redis fallback chain (Valkey -> Upstash -> Memory)
- Edge cases and boundary conditions
- Key generation utilities
- Health check functionality
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# RateLimitResult Tests
# =============================================================================


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_result_creation(self):
        """Test creating a rate limit result."""
        from src.services.rate_limiter import RateLimitResult

        result = RateLimitResult(
            allowed=True,
            remaining=50,
            limit=60,
            reset_at=1234567890,
            retry_after=0,
            backend="valkey",
        )

        assert result.allowed is True
        assert result.remaining == 50
        assert result.limit == 60
        assert result.reset_at == 1234567890
        assert result.retry_after == 0
        assert result.backend == "valkey"

    def test_result_denied(self):
        """Test result when rate limit exceeded."""
        from src.services.rate_limiter import RateLimitResult

        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=60,
            reset_at=1234567890,
            retry_after=30,
            backend="upstash",
        )

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30


# =============================================================================
# SlidingWindowRateLimiter Tests
# =============================================================================


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter class."""

    @pytest.fixture
    def limiter(self):
        """Create a fresh limiter for each test."""
        from src.services.rate_limiter import SlidingWindowRateLimiter

        return SlidingWindowRateLimiter(
            key_prefix="test_ratelimit",
            fallback_to_memory=True,
        )

    @pytest.fixture
    def mock_valkey_client(self):
        """Create mock Valkey client."""
        client = AsyncMock()
        client.ping = AsyncMock(return_value=True)

        # Mock the internal Redis client for pipeline operations
        redis_client = AsyncMock()
        pipe = AsyncMock()

        # Setup pipeline mock - returns [prev_count, current_count, True]
        pipe.execute = AsyncMock(return_value=[None, 1, True])
        redis_client.pipeline = MagicMock(return_value=pipe)

        client._get_client = AsyncMock(return_value=redis_client)
        return client

    @pytest.fixture
    def mock_upstash_client(self):
        """Create mock Upstash client."""
        client = AsyncMock()
        client.ping = AsyncMock(return_value=True)
        client.get = AsyncMock(return_value=None)
        client.incr = AsyncMock(return_value=1)
        client.expire = AsyncMock(return_value=True)
        return client

    # -------------------------------------------------------------------------
    # Window ID Tests
    # -------------------------------------------------------------------------

    def test_get_window_id(self, limiter):
        """Test window ID calculation."""
        window_seconds = 60

        with patch("time.time", return_value=1000.0):
            current, prev = limiter._get_window_id(window_seconds)

            assert current == 16  # 1000 // 60 = 16
            assert prev == 15  # 16 - 1

    def test_get_window_id_at_boundary(self, limiter):
        """Test window ID at exact boundary."""
        window_seconds = 60

        # At exactly 960 seconds (16 * 60)
        with patch("time.time", return_value=960.0):
            current, prev = limiter._get_window_id(window_seconds)

            assert current == 16
            assert prev == 15

    # -------------------------------------------------------------------------
    # Key Generation Tests
    # -------------------------------------------------------------------------

    def test_get_key(self, limiter):
        """Test Redis key generation."""
        key = limiter._get_key("org:acme", 123)

        assert key == "test_ratelimit:org:acme:123"

    def test_get_key_with_special_chars(self, limiter):
        """Test key generation with special characters."""
        key = limiter._get_key("ip:192.168.1.1", 456)

        assert key == "test_ratelimit:ip:192.168.1.1:456"

    # -------------------------------------------------------------------------
    # Memory Fallback Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_memory_fallback_allows_requests(self, limiter):
        """Test that memory fallback allows requests under limit."""
        # Disable Redis backends
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        result = await limiter.is_allowed(
            key="test_key",
            limit=10,
            window_seconds=60,
        )

        assert result.allowed is True
        assert result.remaining == 9
        assert result.backend == "memory"

    @pytest.mark.asyncio
    async def test_memory_fallback_rate_limits(self, limiter):
        """Test that memory fallback enforces rate limits."""
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        # Make requests up to the limit
        for i in range(5):
            result = await limiter.is_allowed(
                key="memory_test",
                limit=5,
                window_seconds=60,
            )
            if i < 5:
                assert result.allowed is True

        # Next request should be denied
        result = await limiter.is_allowed(
            key="memory_test",
            limit=5,
            window_seconds=60,
        )

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after > 0
        assert result.backend == "memory"

    @pytest.mark.asyncio
    async def test_memory_window_expires(self, limiter):
        """Test that memory window properly expires old requests."""
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        # Add some requests
        limiter._memory_storage["expire_test"] = [
            time.time() - 120,  # 2 minutes ago (should expire)
            time.time() - 90,  # 1.5 minutes ago (should expire)
            time.time() - 30,  # 30 seconds ago (should remain)
        ]

        result = await limiter.is_allowed(
            key="expire_test",
            limit=5,
            window_seconds=60,
        )

        assert result.allowed is True
        # Only recent request + new one should count
        assert result.remaining == 3  # 5 - 2 (kept request + new)

    # -------------------------------------------------------------------------
    # Valkey Integration Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_valkey_check(self, limiter, mock_valkey_client):
        """Test rate limit check via Valkey."""
        with patch(
            "src.services.cache.get_valkey_client",
            return_value=mock_valkey_client,
        ):
            limiter._valkey_client = None  # Force re-fetch
            result = await limiter._check_with_valkey(
                key="valkey_test",
                limit=60,
                window_seconds=60,
            )

            assert result is not None
            assert result.allowed is True
            assert result.backend == "valkey"

    @pytest.mark.asyncio
    async def test_valkey_unhealthy_returns_none(self, limiter):
        """Test that unhealthy Valkey returns None."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=False)

        with patch(
            "src.services.cache.get_valkey_client",
            return_value=mock_client,
        ):
            limiter._valkey_client = None
            result = await limiter._check_with_valkey(
                key="test",
                limit=60,
                window_seconds=60,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_valkey_exception_returns_none(self, limiter):
        """Test that Valkey exception returns None and marks unhealthy."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "src.services.cache.get_valkey_client",
            return_value=mock_client,
        ):
            limiter._valkey_client = None
            result = await limiter._check_with_valkey(
                key="test",
                limit=60,
                window_seconds=60,
            )

            assert result is None
            assert limiter._valkey_healthy is False

    # -------------------------------------------------------------------------
    # Upstash Integration Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upstash_check(self, limiter, mock_upstash_client):
        """Test rate limit check via Upstash."""
        with patch(
            "src.services.cache.get_upstash_client",
            return_value=mock_upstash_client,
        ):
            limiter._upstash_client = None
            result = await limiter._check_with_upstash(
                key="upstash_test",
                limit=60,
                window_seconds=60,
            )

            assert result is not None
            assert result.allowed is True
            assert result.backend == "upstash"

    @pytest.mark.asyncio
    async def test_upstash_increments_counter(self, limiter, mock_upstash_client):
        """Test that Upstash increments counter atomically."""
        mock_upstash_client.incr = AsyncMock(return_value=5)

        with patch(
            "src.services.cache.get_upstash_client",
            return_value=mock_upstash_client,
        ):
            limiter._upstash_client = None
            result = await limiter._check_with_upstash(
                key="incr_test",
                limit=60,
                window_seconds=60,
            )

            mock_upstash_client.incr.assert_called_once()
            mock_upstash_client.expire.assert_called_once()
            assert result.allowed is True

    # -------------------------------------------------------------------------
    # Fallback Chain Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fallback_chain_valkey_first(self, limiter, mock_valkey_client):
        """Test that Valkey is tried first in fallback chain."""
        with patch(
            "src.services.cache.get_valkey_client",
            return_value=mock_valkey_client,
        ):
            result = await limiter.is_allowed(
                key="chain_test",
                limit=60,
                window_seconds=60,
            )

            assert result.backend == "valkey"

    @pytest.mark.asyncio
    async def test_fallback_chain_upstash_second(
        self, limiter, mock_upstash_client
    ):
        """Test fallback to Upstash when Valkey unavailable."""
        limiter._valkey_healthy = False

        with patch(
            "src.services.cache.get_upstash_client",
            return_value=mock_upstash_client,
        ):
            result = await limiter.is_allowed(
                key="chain_test",
                limit=60,
                window_seconds=60,
            )

            assert result.backend == "upstash"

    @pytest.mark.asyncio
    async def test_fallback_chain_memory_last(self, limiter):
        """Test fallback to memory when Redis unavailable."""
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        result = await limiter.is_allowed(
            key="chain_test",
            limit=60,
            window_seconds=60,
        )

        assert result.backend == "memory"

    @pytest.mark.asyncio
    async def test_no_fallback_allows_request(self):
        """Test fail-open when no fallback enabled."""
        from src.services.rate_limiter import SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter(
            key_prefix="test",
            fallback_to_memory=False,
        )
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        with patch(
            "src.services.cache.get_valkey_client",
            return_value=None,
        ):
            with patch(
                "src.services.cache.get_upstash_client",
                return_value=None,
            ):
                result = await limiter.is_allowed(
                    key="no_fallback",
                    limit=60,
                    window_seconds=60,
                )

                # Should fail-open
                assert result.allowed is True
                assert result.backend == "none"

    # -------------------------------------------------------------------------
    # Sliding Window Algorithm Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_sliding_window_weights_previous(self, limiter, mock_valkey_client):
        """Test that sliding window properly weights previous window."""
        # Mock pipeline to return previous count of 30
        mock_redis = AsyncMock()
        pipe = AsyncMock()
        pipe.execute = AsyncMock(return_value=[30, 5, True])  # prev=30, current=5
        mock_redis.pipeline = MagicMock(return_value=pipe)
        mock_valkey_client._get_client = AsyncMock(return_value=mock_redis)

        # Mock time to be 30 seconds into the window (50% elapsed)
        # Window 17 starts at 1020 (17*60), so 1030 is 10 seconds in (16.67% elapsed)
        # elapsed_ratio = (1030 - 1020) / 60 = 0.167
        # effective_count = 30 * (1 - 0.167) + 5 = 30 * 0.833 + 5 = 25 + 5 = 30
        # remaining = 60 - 30 = 30
        with patch(
            "src.services.cache.get_valkey_client",
            return_value=mock_valkey_client,
        ):
            with patch("time.time", return_value=1030.0):
                limiter._valkey_client = None

                result = await limiter._check_with_valkey(
                    key="sliding_test",
                    limit=60,
                    window_seconds=60,
                )

                assert result is not None
                assert result.allowed is True
                # The exact value depends on the time mock - verify it's reasonable
                # prev=30, current=5, ~10s into window -> ~30 remaining
                assert result.remaining == 30

    # -------------------------------------------------------------------------
    # Get Remaining Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_remaining_without_increment(self, limiter):
        """Test getting remaining count without incrementing."""
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        # Add some requests to memory
        limiter._memory_storage["remaining_test"] = [
            time.time() - 10,
            time.time() - 5,
        ]

        remaining = await limiter.get_remaining(
            key="remaining_test",
            limit=10,
            window_seconds=60,
        )

        # Should have 8 remaining (10 - 2)
        assert remaining == 8

        # Memory storage should not have been modified
        assert len(limiter._memory_storage["remaining_test"]) == 2

    # -------------------------------------------------------------------------
    # Reset Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reset_clears_memory(self, limiter):
        """Test that reset clears memory storage."""
        limiter._memory_storage["reset_test"] = [
            time.time() - 10,
            time.time() - 5,
        ]

        success = await limiter.reset("reset_test")

        assert success is True
        assert "reset_test" not in limiter._memory_storage

    # -------------------------------------------------------------------------
    # Health Check Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, limiter, mock_valkey_client, mock_upstash_client):
        """Test health check with all backends healthy."""
        with patch(
            "src.services.cache.get_valkey_client",
            return_value=mock_valkey_client,
        ):
            with patch(
                "src.services.cache.get_upstash_client",
                return_value=mock_upstash_client,
            ):
                health = await limiter.health_check()

                assert health["healthy"] is True
                assert health["valkey_available"] is True
                assert health["upstash_available"] is True
                assert health["active_backend"] == "valkey"

    @pytest.mark.asyncio
    async def test_health_check_memory_only(self, limiter):
        """Test health check with only memory available."""
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        with patch(
            "src.services.cache.get_valkey_client",
            return_value=None,
        ):
            with patch(
                "src.services.cache.get_upstash_client",
                return_value=None,
            ):
                health = await limiter.health_check()

                assert health["healthy"] is True
                assert health["valkey_available"] is False
                assert health["upstash_available"] is False
                assert health["active_backend"] == "memory"

    # -------------------------------------------------------------------------
    # Reset Health Cache Tests
    # -------------------------------------------------------------------------

    def test_reset_health_cache(self, limiter):
        """Test resetting health cache."""
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        limiter.reset_health_cache()

        assert limiter._valkey_healthy is None
        assert limiter._upstash_healthy is None


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestRateLimitUtilities:
    """Tests for rate limit utility functions."""

    def test_generate_rate_limit_key_org(self):
        """Test key generation with org ID."""
        from src.services.rate_limiter import generate_rate_limit_key

        key = generate_rate_limit_key(org_id="acme_corp")
        assert key == "org:acme_corp"

    def test_generate_rate_limit_key_user(self):
        """Test key generation with user ID."""
        from src.services.rate_limiter import generate_rate_limit_key

        key = generate_rate_limit_key(user_id="user_123")
        assert key == "user:user_123"

    def test_generate_rate_limit_key_ip(self):
        """Test key generation with IP address."""
        from src.services.rate_limiter import generate_rate_limit_key

        key = generate_rate_limit_key(ip_address="192.168.1.100")
        assert key == "ip:192.168.1.100"

    def test_generate_rate_limit_key_priority(self):
        """Test that org takes priority over user and IP."""
        from src.services.rate_limiter import generate_rate_limit_key

        key = generate_rate_limit_key(
            org_id="acme",
            user_id="user_123",
            ip_address="192.168.1.100",
        )
        assert key == "org:acme"

    def test_generate_rate_limit_key_with_endpoint(self):
        """Test key generation with endpoint hash."""
        from src.services.rate_limiter import generate_rate_limit_key

        key = generate_rate_limit_key(
            org_id="acme",
            endpoint_hash="abc123",
        )
        assert key == "org:acme:abc123"

    def test_generate_rate_limit_key_global(self):
        """Test global key when no identifiers provided."""
        from src.services.rate_limiter import generate_rate_limit_key

        key = generate_rate_limit_key()
        assert key == "global"

    def test_hash_endpoint(self):
        """Test endpoint hashing."""
        from src.services.rate_limiter import hash_endpoint

        hash1 = hash_endpoint("/api/v1/chat/stream")
        hash2 = hash_endpoint("/api/v1/chat/stream")
        hash3 = hash_endpoint("/api/v1/tests")

        # Same endpoint should produce same hash
        assert hash1 == hash2
        # Different endpoints should produce different hashes
        assert hash1 != hash3
        # Hash should be 8 characters
        assert len(hash1) == 8


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGlobalInstance:
    """Tests for global rate limiter instance management."""

    def test_get_rate_limiter_singleton(self):
        """Test that get_rate_limiter returns same instance."""
        from src.services.rate_limiter import get_rate_limiter, reset_rate_limiter

        reset_rate_limiter()

        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2

        reset_rate_limiter()

    def test_reset_rate_limiter(self):
        """Test that reset creates new instance."""
        from src.services.rate_limiter import get_rate_limiter, reset_rate_limiter

        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is not limiter2

        reset_rate_limiter()


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent rate limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_memory(self):
        """Test concurrent requests with memory backend."""
        from src.services.rate_limiter import SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter(
            key_prefix="concurrent_test",
            fallback_to_memory=True,
        )
        limiter._valkey_healthy = False
        limiter._upstash_healthy = False

        # Make 10 concurrent requests with limit of 5
        tasks = [
            limiter.is_allowed(key="concurrent", limit=5, window_seconds=60)
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # Exactly 5 should be allowed
        allowed_count = sum(1 for r in results if r.allowed)
        denied_count = sum(1 for r in results if not r.allowed)

        assert allowed_count == 5
        assert denied_count == 5
