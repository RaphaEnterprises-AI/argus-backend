"""Tests for provider rate limit handling and fallback."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestProviderRateLimitInfo:
    """Tests for ProviderRateLimitInfo dataclass."""

    def test_creation(self):
        """Test creating rate limit info."""
        from src.services.provider_rate_limits import ProviderRateLimitInfo

        info = ProviderRateLimitInfo(
            provider="anthropic",
            limit_type="rpm",
            retry_after=60.0,
            message="Rate limit exceeded",
        )

        assert info.provider == "anthropic"
        assert info.limit_type == "rpm"
        assert info.retry_after == 60.0
        assert info.cooling_until > time.time()

    def test_cooling_until(self):
        """Test cooling_until calculation."""
        from src.services.provider_rate_limits import ProviderRateLimitInfo

        now = time.time()
        info = ProviderRateLimitInfo(
            provider="openai",
            limit_type="tpm",
            retry_after=30.0,
            message="Token limit",
            timestamp=now,
        )

        assert info.cooling_until == now + 30.0

    def test_to_dict(self):
        """Test serialization."""
        from src.services.provider_rate_limits import ProviderRateLimitInfo

        info = ProviderRateLimitInfo(
            provider="anthropic",
            limit_type="rpm",
            retry_after=60.0,
            message="Rate limit",
        )

        d = info.to_dict()
        assert d["provider"] == "anthropic"
        assert d["limit_type"] == "rpm"
        assert d["retry_after"] == 60.0


class TestProviderRateLimitError:
    """Tests for ProviderRateLimitError exception."""

    def test_creation(self):
        """Test creating the exception."""
        from src.services.provider_rate_limits import ProviderRateLimitError

        err = ProviderRateLimitError(
            message="Too many requests",
            provider="anthropic",
            retry_after=60.0,
            limit_type="rpm",
            is_byok=True,
        )

        assert str(err) == "Too many requests"
        assert err.provider == "anthropic"
        assert err.retry_after == 60.0
        assert err.is_byok is True

    def test_to_response_dict_byok(self):
        """Test response dict for BYOK user."""
        from src.services.provider_rate_limits import ProviderRateLimitError

        err = ProviderRateLimitError(
            message="Rate limit",
            provider="openai",
            retry_after=30.0,
            is_byok=True,
        )

        response = err.to_response_dict()
        assert response["error"] == "provider_rate_limit"
        assert response["provider"] == "openai"
        assert response["is_byok"] is True
        assert "Your API key" in response["suggestion"]

    def test_to_response_dict_platform(self):
        """Test response dict for platform user."""
        from src.services.provider_rate_limits import ProviderRateLimitError

        err = ProviderRateLimitError(
            message="Rate limit",
            provider="anthropic",
            retry_after=60.0,
            is_byok=False,
        )

        response = err.to_response_dict()
        assert response["is_byok"] is False
        assert "AI provider" in response["suggestion"]


class TestProviderRateLimitTracker:
    """Tests for ProviderRateLimitTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a fresh tracker for each test."""
        from src.services.provider_rate_limits import ProviderRateLimitTracker

        return ProviderRateLimitTracker(key_prefix="test_provider")

    @pytest.mark.asyncio
    async def test_mark_rate_limited(self, tracker):
        """Test marking a provider as rate limited."""
        await tracker.mark_rate_limited(
            provider="anthropic",
            org_id="org_123",
            retry_after=60.0,
            limit_type="rpm",
            message="Too many requests",
        )

        info = await tracker.is_cooling("anthropic", "org_123")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.limit_type == "rpm"

    @pytest.mark.asyncio
    async def test_is_cooling_not_limited(self, tracker):
        """Test checking non-limited provider."""
        info = await tracker.is_cooling("anthropic", "org_456")
        assert info is None

    @pytest.mark.asyncio
    async def test_cooling_expires(self, tracker):
        """Test that cooling expires after retry_after."""
        await tracker.mark_rate_limited(
            provider="anthropic",
            org_id="org_789",
            retry_after=0.1,  # Very short
            limit_type="rpm",
        )

        # Should be cooling
        info = await tracker.is_cooling("anthropic", "org_789")
        assert info is not None

        # Wait for expiry
        await asyncio.sleep(0.15)

        # Should no longer be cooling
        info = await tracker.is_cooling("anthropic", "org_789")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_available_providers(self, tracker):
        """Test getting available providers."""
        # Mark one as rate limited
        await tracker.mark_rate_limited(
            provider="anthropic",
            org_id="org_test",
            retry_after=60.0,
        )

        available = await tracker.get_available_providers("org_test")
        assert "anthropic" not in available
        assert "openai" in available

    @pytest.mark.asyncio
    async def test_get_available_providers_preferred_first(self, tracker):
        """Test that preferred provider is listed first."""
        available = await tracker.get_available_providers(
            "org_test2",
            preferred="groq",
        )
        assert available[0] == "groq"

    @pytest.mark.asyncio
    async def test_get_fallback_providers(self, tracker):
        """Test getting fallback providers."""
        fallbacks = await tracker.get_fallback_providers(
            failed_provider="anthropic",
            org_id="org_fallback",
        )

        # Should get OpenRouter, OpenAI, Google as fallbacks
        assert "openrouter" in fallbacks
        assert "anthropic" not in fallbacks  # Failed provider excluded

    @pytest.mark.asyncio
    async def test_get_fallback_providers_excludes_cooling(self, tracker):
        """Test that fallback excludes cooling providers."""
        # Mark OpenRouter as cooling too
        await tracker.mark_rate_limited(
            provider="openrouter",
            org_id="org_fallback2",
            retry_after=60.0,
        )

        fallbacks = await tracker.get_fallback_providers(
            failed_provider="anthropic",
            org_id="org_fallback2",
        )

        assert "openrouter" not in fallbacks

    @pytest.mark.asyncio
    async def test_get_earliest_retry(self, tracker):
        """Test getting earliest retry time."""
        # Mark two providers with different retry times
        await tracker.mark_rate_limited(
            provider="anthropic",
            org_id="org_retry",
            retry_after=60.0,
        )
        await tracker.mark_rate_limited(
            provider="openai",
            org_id="org_retry",
            retry_after=30.0,
        )

        earliest = await tracker.get_earliest_retry("org_retry")
        assert earliest is not None
        assert earliest < 35  # Should be close to 30 seconds

    @pytest.mark.asyncio
    async def test_get_earliest_retry_none_cooling(self, tracker):
        """Test earliest retry when nothing is cooling."""
        earliest = await tracker.get_earliest_retry("org_no_cooling")
        assert earliest is None

    @pytest.mark.asyncio
    async def test_clear_cooling(self, tracker):
        """Test clearing cooling state."""
        await tracker.mark_rate_limited(
            provider="anthropic",
            org_id="org_clear",
            retry_after=60.0,
        )

        # Should be cooling
        assert await tracker.is_cooling("anthropic", "org_clear") is not None

        # Clear it
        await tracker.clear_cooling("anthropic", "org_clear")

        # Should no longer be cooling
        assert await tracker.is_cooling("anthropic", "org_clear") is None

    @pytest.mark.asyncio
    async def test_different_orgs_isolated(self, tracker):
        """Test that different orgs have isolated limits."""
        await tracker.mark_rate_limited(
            provider="anthropic",
            org_id="org_a",
            retry_after=60.0,
        )

        # Org A should be cooling
        assert await tracker.is_cooling("anthropic", "org_a") is not None

        # Org B should not be affected
        assert await tracker.is_cooling("anthropic", "org_b") is None


class TestParseProviderRateLimitError:
    """Tests for parse_provider_rate_limit_error function."""

    def test_parse_rate_limit_429(self):
        """Test parsing 429 status code exception."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        class MockException(Exception):
            status_code = 429

        err = MockException("Rate limit exceeded")
        info = parse_provider_rate_limit_error(err, "anthropic")

        assert info is not None
        assert info.provider == "anthropic"
        assert info.retry_after == 60.0  # Default fallback

    def test_parse_rate_limit_message(self):
        """Test parsing rate limit from message."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        err = Exception("Too many requests, please slow down")
        info = parse_provider_rate_limit_error(err, "openai")

        assert info is not None
        assert info.provider == "openai"

    def test_parse_with_retry_after_attribute(self):
        """Test parsing with retry_after attribute."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        class MockException(Exception):
            retry_after = 120.0
            status_code = 429

        err = MockException("Rate limited")
        info = parse_provider_rate_limit_error(err, "groq")

        assert info is not None
        assert info.retry_after == 120.0

    def test_parse_with_retry_after_header(self):
        """Test parsing with Retry-After header."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        class MockException(Exception):
            status_code = 429
            headers = {"retry-after": "45"}

        err = MockException("Rate limited")
        info = parse_provider_rate_limit_error(err, "together")

        assert info is not None
        assert info.retry_after == 45.0

    def test_parse_token_limit(self):
        """Test parsing token limit error."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        err = Exception("Rate limit exceeded: tokens per minute (TPM)")
        info = parse_provider_rate_limit_error(err, "openai")

        assert info is not None
        assert info.limit_type == "tpm"

    def test_parse_request_limit(self):
        """Test parsing request limit error."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        err = Exception("Rate limit: requests per minute exceeded")
        info = parse_provider_rate_limit_error(err, "anthropic")

        assert info is not None
        assert info.limit_type == "rpm"

    def test_parse_non_rate_limit(self):
        """Test that non-rate-limit errors return None."""
        from src.services.provider_rate_limits import parse_provider_rate_limit_error

        err = Exception("Connection timeout")
        info = parse_provider_rate_limit_error(err, "anthropic")

        assert info is None


class TestGlobalTracker:
    """Tests for global tracker singleton."""

    def test_singleton(self):
        """Test that get_provider_rate_tracker returns singleton."""
        from src.services.provider_rate_limits import (
            get_provider_rate_tracker,
            reset_provider_rate_tracker,
        )

        reset_provider_rate_tracker()

        tracker1 = get_provider_rate_tracker()
        tracker2 = get_provider_rate_tracker()

        assert tracker1 is tracker2

        reset_provider_rate_tracker()

    def test_reset(self):
        """Test that reset creates new instance."""
        from src.services.provider_rate_limits import (
            get_provider_rate_tracker,
            reset_provider_rate_tracker,
        )

        tracker1 = get_provider_rate_tracker()
        reset_provider_rate_tracker()
        tracker2 = get_provider_rate_tracker()

        assert tracker1 is not tracker2

        reset_provider_rate_tracker()
