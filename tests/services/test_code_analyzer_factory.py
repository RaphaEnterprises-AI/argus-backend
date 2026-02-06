"""End-to-end tests for CodeAnalyzerFactory."""
import pytest
import asyncio
import inspect
import time
from unittest.mock import AsyncMock, patch, MagicMock

def test_imports():
    """Test all imports work."""
    from src.services.code_analyzer_factory import (
        CodeAnalyzerFactory,
        AnalyzerType,
        get_code_analyzer_factory,
    )
    assert CodeAnalyzerFactory is not None
    assert AnalyzerType is not None

def test_analyzer_type_enum():
    """Test AnalyzerType enum values."""
    from src.services.code_analyzer_factory import AnalyzerType

    assert AnalyzerType.LOCAL_GIT.value == 'local_git'
    assert AnalyzerType.GITHUB_API.value == 'github_api'
    assert AnalyzerType.GITLAB_API.value == 'gitlab_api'
    assert AnalyzerType.NONE.value == 'none'
    # Also test additional types
    assert AnalyzerType.BITBUCKET_API.value == 'bitbucket_api'
    assert AnalyzerType.AZURE_DEVOPS.value == 'azure_devops'
    assert AnalyzerType.CACHED.value == 'cached'

def test_factory_instantiation():
    """Test factory can be instantiated."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()
    assert factory is not None

def test_factory_has_get_analyzer():
    """Test factory has get_analyzer method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()
    assert hasattr(factory, 'get_analyzer')
    assert inspect.iscoroutinefunction(factory.get_analyzer)

def test_factory_has_cache_methods():
    """Test factory has cache methods."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()
    assert hasattr(factory, 'clear_analyzer_cache')
    assert hasattr(factory, 'clear_all_caches')
    assert hasattr(factory, 'get_analyzer_cache_stats')
    assert hasattr(factory, 'clear_cache')
    assert hasattr(factory, 'get_rate_limit_status')

def test_get_code_analyzer_factory():
    """Test get_code_analyzer_factory singleton."""
    from src.services.code_analyzer_factory import get_code_analyzer_factory

    factory1 = get_code_analyzer_factory()
    factory2 = get_code_analyzer_factory()
    assert factory1 is factory2  # Should be singleton

def test_platform_priority_defined():
    """Test PLATFORM_PRIORITY is defined."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType

    assert hasattr(CodeAnalyzerFactory, 'PLATFORM_PRIORITY')
    # Check priority order
    priority = CodeAnalyzerFactory.PLATFORM_PRIORITY
    assert AnalyzerType.LOCAL_GIT in priority
    assert AnalyzerType.GITHUB_API in priority
    # Local should be highest priority (first)
    assert priority.index(AnalyzerType.LOCAL_GIT) < priority.index(AnalyzerType.GITHUB_API)

def test_cache_ttl_defined():
    """Test cache TTL is defined."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    assert hasattr(CodeAnalyzerFactory, 'ANALYZER_CACHE_TTL_SECONDS')
    assert CodeAnalyzerFactory.ANALYZER_CACHE_TTL_SECONDS > 0

def test_platform_patterns_defined():
    """Test PLATFORM_PATTERNS is defined."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType

    assert hasattr(CodeAnalyzerFactory, 'PLATFORM_PATTERNS')
    patterns = CodeAnalyzerFactory.PLATFORM_PATTERNS
    assert "github.com" in patterns
    assert patterns["github.com"] == AnalyzerType.GITHUB_API
    assert "gitlab.com" in patterns
    assert patterns["gitlab.com"] == AnalyzerType.GITLAB_API

def test_platform_rate_limits_defined():
    """Test PLATFORM_RATE_LIMITS is defined."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType

    assert hasattr(CodeAnalyzerFactory, 'PLATFORM_RATE_LIMITS')
    limits = CodeAnalyzerFactory.PLATFORM_RATE_LIMITS
    assert AnalyzerType.GITHUB_API in limits
    assert limits[AnalyzerType.GITHUB_API] == 5000

def test_rate_limit_state_can_make_call():
    """Test RateLimitState.can_make_call() logic."""
    from src.services.code_analyzer_factory import RateLimitState

    state = RateLimitState(calls_made=0, limit_per_hour=100)
    assert state.can_make_call() is True

    state.calls_made = 100
    assert state.can_make_call() is False

def test_rate_limit_state_record_call():
    """Test RateLimitState.record_call() logic."""
    from src.services.code_analyzer_factory import RateLimitState

    state = RateLimitState(calls_made=0, limit_per_hour=100)
    state.record_call()
    assert state.calls_made == 1

    state.record_call(remaining=50)
    assert state.calls_made == 50  # Updated from remaining

def test_rate_limit_state_retry_after():
    """Test RateLimitState retry_after logic."""
    from src.services.code_analyzer_factory import RateLimitState

    state = RateLimitState(calls_made=0, limit_per_hour=100)
    state.record_rate_limited(retry_after_seconds=60)

    assert state.retry_after is not None
    assert state.can_make_call() is False

def test_detect_platform():
    """Test _detect_platform method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType

    factory = CodeAnalyzerFactory()

    # Test GitHub
    assert factory._detect_platform("https://github.com/org/repo") == AnalyzerType.GITHUB_API

    # Test GitLab
    assert factory._detect_platform("https://gitlab.com/org/repo") == AnalyzerType.GITLAB_API

    # Test Bitbucket
    assert factory._detect_platform("https://bitbucket.org/org/repo") == AnalyzerType.BITBUCKET_API

    # Test Azure DevOps
    assert factory._detect_platform("https://dev.azure.com/org/repo") == AnalyzerType.AZURE_DEVOPS

    # Test self-hosted GitLab
    assert factory._detect_platform("https://gitlab.mycompany.com/org/repo") == AnalyzerType.GITLAB_API

    # Test unknown platform
    assert factory._detect_platform("https://unknown.com/org/repo") is None

    # Test invalid URL
    assert factory._detect_platform("not-a-url") is None

def test_check_local_repo():
    """Test _check_local_repo method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()

    # None path
    assert factory._check_local_repo(None) is False

    # Non-existent path
    assert factory._check_local_repo("/nonexistent/path") is False

    # Current directory (should be a git repo based on git status)
    # Note: This test works because we're in a git repo
    import os
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, ".git")):
        assert factory._check_local_repo(cwd) is True

def test_get_cache_key():
    """Test _get_cache_key method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()

    key1 = factory._get_cache_key("#btn", "https://github.com/org/repo", None)
    key2 = factory._get_cache_key("#btn", "https://github.com/org/repo", None)
    key3 = factory._get_cache_key("#btn", "https://github.com/org/repo", "file.py")

    # Same inputs should produce same key
    assert key1 == key2
    # Different inputs should produce different key
    assert key1 != key3

def test_get_analyzer_cache_key():
    """Test _get_analyzer_cache_key method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()

    key = factory._get_analyzer_cache_key("proj_123", "org_456")
    assert key == "org_456:proj_123"

def test_analyzer_cache_stats():
    """Test get_analyzer_cache_stats method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType, CachedAnalyzer

    factory = CodeAnalyzerFactory()

    # Clear any existing cache
    factory._analyzer_cache.clear()

    # Initially empty
    stats = factory.get_analyzer_cache_stats()
    assert stats["total_entries"] == 0
    assert stats["active_entries"] == 0

    # Add an entry
    factory._analyzer_cache["org:proj"] = CachedAnalyzer(
        analyzer=None,
        analyzer_type=AnalyzerType.NONE,
        created_at=time.time(),
    )

    stats = factory.get_analyzer_cache_stats()
    assert stats["total_entries"] == 1
    assert stats["active_entries"] == 1
    assert "none" in stats["by_analyzer_type"]

def test_clear_analyzer_cache():
    """Test clear_analyzer_cache method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType, CachedAnalyzer

    factory = CodeAnalyzerFactory()

    # Add some entries
    factory._analyzer_cache["org1:proj1"] = CachedAnalyzer(
        analyzer=None, analyzer_type=AnalyzerType.NONE, created_at=time.time()
    )
    factory._analyzer_cache["org1:proj2"] = CachedAnalyzer(
        analyzer=None, analyzer_type=AnalyzerType.NONE, created_at=time.time()
    )
    factory._analyzer_cache["org2:proj1"] = CachedAnalyzer(
        analyzer=None, analyzer_type=AnalyzerType.NONE, created_at=time.time()
    )

    # Clear specific project
    cleared = factory.clear_analyzer_cache(project_id="proj1", org_id="org1")
    assert cleared == 1
    assert "org1:proj1" not in factory._analyzer_cache

    # Clear all for org
    cleared = factory.clear_analyzer_cache(org_id="org1")
    assert cleared == 1  # Only proj2 left

    # Clear all
    factory._analyzer_cache["org3:proj1"] = CachedAnalyzer(
        analyzer=None, analyzer_type=AnalyzerType.NONE, created_at=time.time()
    )
    cleared = factory.clear_analyzer_cache()
    assert cleared >= 1

def test_clear_all_caches():
    """Test clear_all_caches method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType, CachedAnalyzer, CachedAnalysis
    from datetime import datetime, timedelta

    factory = CodeAnalyzerFactory()

    # Add entries to both caches
    factory._analyzer_cache["org:proj"] = CachedAnalyzer(
        analyzer=None, analyzer_type=AnalyzerType.NONE, created_at=time.time()
    )
    factory._cache["cache_key"] = CachedAnalysis(
        selector="#btn",
        result=None,
        analyzer_type=AnalyzerType.NONE,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=24),
        repo_url="https://github.com/org/repo",
    )

    result = factory.clear_all_caches()
    assert "analysis_cache" in result
    assert "analyzer_cache" in result
    assert result["analyzer_cache"] >= 1

def test_rate_limit_status():
    """Test get_rate_limit_status method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType, RateLimitState

    factory = CodeAnalyzerFactory()

    # Add rate limit state
    factory._rate_limits["org_123:github_api"] = RateLimitState(
        calls_made=10, limit_per_hour=5000
    )

    status = factory.get_rate_limit_status("org_123")
    assert "github_api" in status
    assert status["github_api"]["calls_made"] == 10
    assert status["github_api"]["remaining"] == 4990

@pytest.mark.asyncio
async def test_get_analyzer_with_invalid_ids():
    """Test get_analyzer with invalid project/org IDs."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory

    factory = CodeAnalyzerFactory()

    # Mock supabase to return empty results
    with patch('src.services.code_analyzer_factory.CodeAnalyzerFactory._get_project_repo_url') as mock_get_url:
        mock_get_url.return_value = None

        result = await factory.get_analyzer(
            project_id="nonexistent-project",
            org_id="nonexistent-org",
            skip_cache=True,  # Skip cache to test the lookup
        )
        assert result is None

@pytest.mark.asyncio
async def test_get_analyzer_respects_airgapped_mode():
    """Test get_analyzer respects air-gapped mode."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory
    from src.config import DeploymentMode

    factory = CodeAnalyzerFactory()

    # Mock settings to be air-gapped and mock the internal methods
    mock_settings = MagicMock()
    mock_settings.deployment_mode = DeploymentMode.AIRGAPPED

    with patch('src.config.get_settings', return_value=mock_settings):
        result = await factory.get_analyzer(
            project_id="test-project",
            org_id="test-org",
            skip_cache=True,  # Skip cache to test air-gapped mode
        )
        # In air-gapped mode without local repo, should return None
        assert result is None

@pytest.mark.asyncio
async def test_get_analyzer_uses_local_repo():
    """Test get_analyzer prefers local repo."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory
    import os

    factory = CodeAnalyzerFactory()

    # Use current directory which should be a git repo
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, ".git")):
        with patch.object(factory, '_get_local_analyzer') as mock_local:
            mock_analyzer = MagicMock()
            mock_local.return_value = mock_analyzer

            result = await factory.get_analyzer(
                project_id="test-project",
                org_id="test-org",
                local_repo_path=cwd,
            )

            mock_local.assert_called_once_with(cwd)
            assert result == mock_analyzer

@pytest.mark.asyncio
async def test_get_analyzer_caches_result():
    """Test get_analyzer caches the result (RAP-352)."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType
    from src.config import DeploymentMode

    factory = CodeAnalyzerFactory()
    factory._analyzer_cache.clear()

    mock_settings = MagicMock()
    mock_settings.deployment_mode = DeploymentMode.CLOUD

    # Mock to simulate no repo URL found
    with patch('src.config.get_settings', return_value=mock_settings):
        with patch.object(factory, '_get_project_repo_url') as mock_get_url:
            mock_get_url.return_value = None

            # First call - should look up and cache
            result1 = await factory.get_analyzer(
                project_id="test-project",
                org_id="test-org",
                skip_cache=False,
            )

            # Second call - should use cache
            result2 = await factory.get_analyzer(
                project_id="test-project",
                org_id="test-org",
                skip_cache=False,
            )

            # Should only have called _get_project_repo_url once (first call)
            # Second call should use cache
            assert mock_get_url.call_count == 1

            # Both results should be None (no repo URL)
            assert result1 is None
            assert result2 is None

@pytest.mark.asyncio
async def test_get_analyzer_github_integration():
    """Test get_analyzer with GitHub integration."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType
    from src.config import DeploymentMode

    factory = CodeAnalyzerFactory()
    factory._analyzer_cache.clear()

    mock_settings = MagicMock()
    mock_settings.deployment_mode = DeploymentMode.CLOUD

    mock_analyzer = MagicMock()

    with patch('src.config.get_settings', return_value=mock_settings):
        with patch.object(factory, '_get_project_repo_url') as mock_get_url:
            mock_get_url.return_value = "https://github.com/org/repo"

            with patch.object(factory, '_get_platform_credentials') as mock_creds:
                mock_creds.return_value = "ghp_test_token"

                with patch.object(factory, '_get_remote_analyzer') as mock_remote:
                    mock_remote.return_value = mock_analyzer

                    result = await factory.get_analyzer(
                        project_id="test-project",
                        org_id="test-org",
                        skip_cache=True,
                    )

                    mock_remote.assert_called_once()
                    assert result == mock_analyzer

@pytest.mark.asyncio
async def test_get_analyzer_rate_limited():
    """Test get_analyzer handles rate limiting."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType, RateLimitState
    from src.config import DeploymentMode

    factory = CodeAnalyzerFactory()
    factory._analyzer_cache.clear()

    # Set up rate limit to be exceeded
    factory._rate_limits["test-org:github_api"] = RateLimitState(
        calls_made=5000,
        limit_per_hour=5000,
    )

    mock_settings = MagicMock()
    mock_settings.deployment_mode = DeploymentMode.CLOUD

    with patch('src.config.get_settings', return_value=mock_settings):
        with patch.object(factory, '_get_project_repo_url') as mock_get_url:
            mock_get_url.return_value = "https://github.com/org/repo"

            result = await factory.get_analyzer(
                project_id="test-project",
                org_id="test-org",
                skip_cache=True,
            )

            # Should return None due to rate limiting
            assert result is None

@pytest.mark.asyncio
async def test_analyze_with_caching():
    """Test analyze_with_caching method."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType

    factory = CodeAnalyzerFactory()
    factory._cache.clear()

    # Mock no repo URL
    with patch.object(factory, '_get_project_repo_url') as mock_get_url:
        mock_get_url.return_value = None

        result, analyzer_type = await factory.analyze_with_caching(
            project_id="test-project",
            org_id="test-org",
            selector="#btn",
        )

        assert result is None
        assert analyzer_type == AnalyzerType.NONE

@pytest.mark.asyncio
async def test_analyze_with_caching_uses_cache():
    """Test analyze_with_caching uses cached results."""
    from src.services.code_analyzer_factory import CodeAnalyzerFactory, AnalyzerType, CachedAnalysis
    from datetime import datetime, timedelta

    factory = CodeAnalyzerFactory()
    factory._cache.clear()

    # Pre-populate cache
    cache_key = factory._get_cache_key("#btn", "https://github.com/org/repo")
    cached_result = {"replacement": "#new-btn"}
    factory._cache[cache_key] = CachedAnalysis(
        selector="#btn",
        result=cached_result,
        analyzer_type=AnalyzerType.GITHUB_API,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=24),
        repo_url="https://github.com/org/repo",
    )

    with patch.object(factory, '_get_project_repo_url') as mock_get_url:
        mock_get_url.return_value = "https://github.com/org/repo"

        result, analyzer_type = await factory.analyze_with_caching(
            project_id="test-project",
            org_id="test-org",
            selector="#btn",
        )

        assert result == cached_result
        assert analyzer_type == AnalyzerType.CACHED

def test_cached_analysis_dataclass():
    """Test CachedAnalysis dataclass."""
    from src.services.code_analyzer_factory import CachedAnalysis, AnalyzerType
    from datetime import datetime, timedelta

    now = datetime.now()
    cached = CachedAnalysis(
        selector="#btn",
        result={"foo": "bar"},
        analyzer_type=AnalyzerType.GITHUB_API,
        created_at=now,
        expires_at=now + timedelta(hours=24),
        repo_url="https://github.com/org/repo",
    )

    assert cached.selector == "#btn"
    assert cached.result == {"foo": "bar"}
    assert cached.analyzer_type == AnalyzerType.GITHUB_API

def test_cached_analyzer_dataclass():
    """Test CachedAnalyzer dataclass."""
    from src.services.code_analyzer_factory import CachedAnalyzer, AnalyzerType

    cached = CachedAnalyzer(
        analyzer=None,
        analyzer_type=AnalyzerType.NONE,
        created_at=time.time(),
        repo_url="https://github.com/org/repo",
    )

    assert cached.analyzer is None
    assert cached.analyzer_type == AnalyzerType.NONE
    assert cached.repo_url == "https://github.com/org/repo"

def test_code_analyzer_protocol():
    """Test CodeAnalyzerProtocol is defined."""
    from src.services.code_analyzer_factory import CodeAnalyzerProtocol
    import typing

    # Check it's a Protocol
    assert hasattr(CodeAnalyzerProtocol, '__protocol_attrs__') or hasattr(CodeAnalyzerProtocol, '_is_protocol')

    # Check required methods
    assert hasattr(CodeAnalyzerProtocol, 'find_replacement_selector')
    assert hasattr(CodeAnalyzerProtocol, 'find_selector_changes')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
