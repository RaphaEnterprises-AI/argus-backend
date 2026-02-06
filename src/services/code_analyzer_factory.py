"""Code Analyzer Factory - Unified interface for code-aware healing.

This module provides a factory that intelligently selects the best code
analyzer based on:
1. Local repository availability (fastest, no rate limits)
2. Cached analysis results (avoid redundant API calls)
3. Remote API access (GitHub, GitLab, Bitbucket, etc.)

Architecture:
```
CodeAnalyzerFactory.get_analyzer()
    │
    ├── 1. Check: Is repo cloned locally? (MCP, dev environment)
    │   └── YES → Use GitAnalyzer (local git commands) - FREE, FAST
    │
    ├── 2. Check: Cached selector analysis?
    │   └── YES → Return from cache - INSTANT
    │
    ├── 3. Check: Remote repo configured?
    │   ├── github.com → GitHubCodeAnalyzer (with rate limiting)
    │   ├── gitlab.com → GitLabCodeAnalyzer (with rate limiting)
    │   └── others → Future support
    │
    └── 4. None available → Return None (fall back to LLM)
```

Rate Limiting Strategy:
- Cache successful analyses for 24 hours
- Track API calls per org/hour
- Graceful degradation when limits hit
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import structlog

logger = structlog.get_logger(__name__)


class AnalyzerType(str, Enum):
    """Types of code analyzers available."""
    LOCAL_GIT = "local_git"  # Local git repository
    GITHUB_API = "github_api"  # GitHub REST API
    GITLAB_API = "gitlab_api"  # GitLab REST API
    BITBUCKET_API = "bitbucket_api"  # Bitbucket REST API
    AZURE_DEVOPS = "azure_devops"  # Azure DevOps REST API
    CACHED = "cached"  # From cache
    NONE = "none"  # No analyzer available


class CodeAnalyzerProtocol(Protocol):
    """Protocol that all code analyzers must implement."""

    async def find_replacement_selector(
        self,
        failed_selector: str,
        file_hint: str | None = None,
        days: int = 30,
    ) -> Any | None:
        """Find a replacement selector by analyzing code history."""
        ...

    async def find_selector_changes(
        self,
        selector: str,
        days: int = 30,
    ) -> list[Any]:
        """Find all changes to a selector in recent history."""
        ...


@dataclass
class RateLimitState:
    """Track rate limit state for an API."""
    calls_made: int = 0
    window_start: float = field(default_factory=time.time)
    limit_per_hour: int = 1000  # Conservative default
    retry_after: float | None = None

    def can_make_call(self) -> bool:
        """Check if we can make another API call."""
        now = time.time()

        # Reset window if hour passed
        if now - self.window_start > 3600:
            self.calls_made = 0
            self.window_start = now
            self.retry_after = None

        # Check retry-after from previous 429
        if self.retry_after and now < self.retry_after:
            return False

        return self.calls_made < self.limit_per_hour

    def record_call(self, remaining: int | None = None, reset_at: float | None = None):
        """Record an API call."""
        self.calls_made += 1

        # Update from response headers if provided
        if remaining is not None:
            self.calls_made = self.limit_per_hour - remaining
        if reset_at is not None:
            self.window_start = reset_at - 3600

    def record_rate_limited(self, retry_after_seconds: int = 60):
        """Record that we hit a rate limit."""
        self.retry_after = time.time() + retry_after_seconds
        logger.warning(
            "Rate limited, will retry after",
            retry_after_seconds=retry_after_seconds,
        )


@dataclass
class CachedAnalysis:
    """Cached code analysis result."""
    selector: str
    result: Any  # SelectorChange or None
    analyzer_type: AnalyzerType
    created_at: datetime
    expires_at: datetime
    repo_url: str


@dataclass
class CachedAnalyzer:
    """Cached analyzer instance to avoid repeated DB lookups (RAP-352)."""
    analyzer: CodeAnalyzerProtocol | None
    analyzer_type: AnalyzerType
    created_at: float  # time.time() for TTL check
    repo_url: str | None = None


class CodeAnalyzerFactory:
    """Factory for creating and managing code analyzers.

    This factory:
    1. Detects the best analyzer to use (local > cached > remote)
    2. Manages rate limits for remote APIs
    3. Caches analysis results to reduce API calls
    4. Caches analyzer instances to avoid repeated DB lookups (RAP-352)
    5. Provides a unified interface regardless of backend

    Usage:
        factory = CodeAnalyzerFactory()

        # Get analyzer for a project
        analyzer = await factory.get_analyzer(
            project_id="proj_123",
            org_id="org_456",
            local_repo_path="/path/to/repo",  # Optional
        )

        if analyzer:
            change = await analyzer.find_replacement_selector("#old-btn")
    """

    # Rate limit tracking per org
    _rate_limits: dict[str, RateLimitState] = {}

    # Cache for analysis results (selector -> result)
    _cache: dict[str, CachedAnalysis] = {}

    # Cache for analyzer instances by (project_id, org_id) to avoid repeated DB lookups (RAP-352)
    _analyzer_cache: dict[str, CachedAnalyzer] = {}

    # Cache TTL for analysis results
    CACHE_TTL_HOURS = 24

    # TTL for analyzer instance cache in seconds (5 minutes) - RAP-352
    ANALYZER_CACHE_TTL_SECONDS = 300

    # Platform detection patterns
    PLATFORM_PATTERNS = {
        "github.com": AnalyzerType.GITHUB_API,
        "gitlab.com": AnalyzerType.GITLAB_API,
        "bitbucket.org": AnalyzerType.BITBUCKET_API,
        "dev.azure.com": AnalyzerType.AZURE_DEVOPS,
        "visualstudio.com": AnalyzerType.AZURE_DEVOPS,
    }

    # Multi-source priority order (RAP-351)
    # Higher priority sources are tried first
    PLATFORM_PRIORITY = [
        AnalyzerType.LOCAL_GIT,    # Highest: air-gapped compatible, no rate limits
        AnalyzerType.GITHUB_API,   # Most common remote platform
        AnalyzerType.GITLAB_API,   # Second most common
        AnalyzerType.BITBUCKET_API,  # Less common
        AnalyzerType.AZURE_DEVOPS,  # Enterprise
    ]

    # Rate limits per platform (requests per hour)
    PLATFORM_RATE_LIMITS = {
        AnalyzerType.GITHUB_API: 5000,  # GitHub authenticated
        AnalyzerType.GITLAB_API: 2000,  # GitLab default
        AnalyzerType.BITBUCKET_API: 1000,  # Bitbucket Cloud
        AnalyzerType.AZURE_DEVOPS: 1000,  # Azure DevOps
    }

    def __init__(self):
        """Initialize the factory."""
        self._analyzers: dict[str, CodeAnalyzerProtocol] = {}

    def _get_analyzer_cache_key(
        self,
        project_id: str,
        org_id: str,
    ) -> str:
        """Generate cache key for analyzer instance (RAP-352)."""
        return f"{org_id}:{project_id}"

    def _get_cached_analyzer(
        self,
        project_id: str,
        org_id: str,
    ) -> CachedAnalyzer | None:
        """Check cache for existing analyzer instance (RAP-352).

        Returns:
            CachedAnalyzer if found and not expired, None otherwise
        """
        key = self._get_analyzer_cache_key(project_id, org_id)
        cached = self._analyzer_cache.get(key)

        if cached:
            # Check TTL
            age_seconds = time.time() - cached.created_at
            if age_seconds < self.ANALYZER_CACHE_TTL_SECONDS:
                logger.debug(
                    "Analyzer cache hit",
                    project_id=project_id,
                    analyzer_type=cached.analyzer_type.value,
                    age_seconds=int(age_seconds),
                )
                return cached
            else:
                # Remove expired entry
                del self._analyzer_cache[key]
                logger.debug(
                    "Analyzer cache expired",
                    project_id=project_id,
                    age_seconds=int(age_seconds),
                )

        return None

    def _store_cached_analyzer(
        self,
        project_id: str,
        org_id: str,
        analyzer: CodeAnalyzerProtocol | None,
        analyzer_type: AnalyzerType,
        repo_url: str | None = None,
    ) -> None:
        """Store analyzer instance in cache (RAP-352)."""
        key = self._get_analyzer_cache_key(project_id, org_id)

        self._analyzer_cache[key] = CachedAnalyzer(
            analyzer=analyzer,
            analyzer_type=analyzer_type,
            created_at=time.time(),
            repo_url=repo_url,
        )

        logger.debug(
            "Cached analyzer instance",
            project_id=project_id,
            analyzer_type=analyzer_type.value,
            ttl_seconds=self.ANALYZER_CACHE_TTL_SECONDS,
        )

    def _get_cache_key(
        self,
        selector: str,
        repo_url: str,
        file_hint: str | None = None,
    ) -> str:
        """Generate cache key for a selector analysis."""
        data = f"{repo_url}:{selector}:{file_hint or ''}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def _check_cache(
        self,
        selector: str,
        repo_url: str,
        file_hint: str | None = None,
    ) -> CachedAnalysis | None:
        """Check cache for existing analysis."""
        key = self._get_cache_key(selector, repo_url, file_hint)
        cached = self._cache.get(key)

        if cached and datetime.now() < cached.expires_at:
            logger.debug(
                "Cache hit for selector analysis",
                selector=selector[:30],
                analyzer_type=cached.analyzer_type.value,
            )
            return cached

        # Remove expired entry
        if cached:
            del self._cache[key]

        return None

    def _store_cache(
        self,
        selector: str,
        repo_url: str,
        result: Any,
        analyzer_type: AnalyzerType,
        file_hint: str | None = None,
    ) -> None:
        """Store analysis result in cache."""
        key = self._get_cache_key(selector, repo_url, file_hint)
        now = datetime.now()

        self._cache[key] = CachedAnalysis(
            selector=selector,
            result=result,
            analyzer_type=analyzer_type,
            created_at=now,
            expires_at=now + timedelta(hours=self.CACHE_TTL_HOURS),
            repo_url=repo_url,
        )

        logger.debug(
            "Cached selector analysis",
            selector=selector[:30],
            ttl_hours=self.CACHE_TTL_HOURS,
        )

    def _get_rate_limit(self, org_id: str, platform: AnalyzerType) -> RateLimitState:
        """Get rate limit state for an org/platform combo."""
        key = f"{org_id}:{platform.value}"

        if key not in self._rate_limits:
            self._rate_limits[key] = RateLimitState(
                limit_per_hour=self.PLATFORM_RATE_LIMITS.get(platform, 1000)
            )

        return self._rate_limits[key]

    def _detect_platform(self, repo_url: str) -> AnalyzerType | None:
        """Detect the platform from repository URL."""
        try:
            parsed = urlparse(repo_url)
            host = parsed.netloc.lower()

            for pattern, platform in self.PLATFORM_PATTERNS.items():
                if pattern in host:
                    return platform

            # Self-hosted GitLab detection (common pattern)
            if "gitlab" in host:
                return AnalyzerType.GITLAB_API

            logger.debug(
                "Unknown repository platform",
                host=host,
                url=repo_url,
            )
            return None

        except Exception:
            return None

    def _check_local_repo(self, local_path: str | None) -> bool:
        """Check if local repository is available and valid."""
        if not local_path:
            return False

        path = Path(local_path)
        git_dir = path / ".git"

        return path.exists() and git_dir.exists()

    async def get_analyzer(
        self,
        project_id: str,
        org_id: str,
        local_repo_path: str | None = None,
        force_remote: bool = False,
        skip_cache: bool = False,
    ) -> CodeAnalyzerProtocol | None:
        """Get the best available code analyzer for a project.

        Multi-source priority order (RAP-351):
        1. Local git repository (highest priority - air-gapped compatible)
        2. GitHub API (most common)
        3. GitLab API
        4. Bitbucket API
        5. Azure DevOps API
        6. None (no analyzer available)

        Caching (RAP-352):
        - Analyzer instances are cached by (project_id, org_id) with 5-minute TTL
        - Avoids repeated DB lookups for project repo URL and credentials

        Args:
            project_id: Project ID
            org_id: Organization ID
            local_repo_path: Optional path to local repository
            force_remote: Force use of remote API even if local available
            skip_cache: Skip analyzer cache lookup (e.g., for testing)

        Returns:
            Code analyzer instance or None if unavailable
        """
        from src.config import DeploymentMode, get_settings

        config = get_settings()

        # RAP-352: Check analyzer cache first (unless forcing local or skipping cache)
        if not force_remote and not skip_cache and not local_repo_path:
            cached = self._get_cached_analyzer(project_id, org_id)
            if cached:
                return cached.analyzer

        # 1. Check local repository first (highest priority - air-gapped compatible)
        if not force_remote and self._check_local_repo(local_repo_path):
            logger.info(
                "Using local git analyzer",
                project_id=project_id,
                path=local_repo_path,
            )
            analyzer = await self._get_local_analyzer(local_repo_path)
            # Don't cache local analyzers as they depend on the local_repo_path param
            return analyzer

        # 2. Skip remote APIs in air-gapped mode
        if config.deployment_mode == DeploymentMode.AIRGAPPED:
            logger.info(
                "Air-gapped mode: skipping remote code analyzer APIs",
                project_id=project_id,
                deployment_mode=config.deployment_mode.value,
            )
            # Cache the None result to avoid repeated lookups
            self._store_cached_analyzer(
                project_id, org_id, None, AnalyzerType.NONE
            )
            return None

        # 3. Get project's repository URL
        repo_url = await self._get_project_repo_url(project_id)
        if not repo_url:
            logger.debug("No repository URL configured for project", project_id=project_id)
            # Cache the None result
            self._store_cached_analyzer(
                project_id, org_id, None, AnalyzerType.NONE
            )
            return None

        # 4. Detect platform from URL
        platform = self._detect_platform(repo_url)
        if not platform:
            logger.warning("Unsupported repository platform", repo_url=repo_url)
            self._store_cached_analyzer(
                project_id, org_id, None, AnalyzerType.NONE, repo_url
            )
            return None

        # 5. Check rate limits before attempting to get analyzer
        rate_limit = self._get_rate_limit(org_id, platform)
        if not rate_limit.can_make_call():
            logger.warning(
                "Rate limit exceeded for platform",
                platform=platform.value,
                org_id=org_id,
                retry_after=rate_limit.retry_after,
            )
            # Don't cache rate-limited result - it's temporary
            return None

        # 6. Try to get analyzer for the detected platform (RAP-351)
        # Priority is determined by PLATFORM_PRIORITY, but we use URL detection
        analyzer = await self._get_remote_analyzer(
            repo_url=repo_url,
            org_id=org_id,
            platform=platform,
        )

        # 7. Cache the result (whether analyzer or None)
        if analyzer:
            self._store_cached_analyzer(
                project_id, org_id, analyzer, platform, repo_url
            )
        else:
            # Cache None result to avoid repeated failed lookups
            self._store_cached_analyzer(
                project_id, org_id, None, AnalyzerType.NONE, repo_url
            )

        return analyzer

    async def _get_local_analyzer(
        self,
        repo_path: str,
    ) -> CodeAnalyzerProtocol | None:
        """Get local git analyzer."""
        try:
            from src.services.git_analyzer import GitAnalyzer

            return GitAnalyzer(repo_path=repo_path)
        except Exception as e:
            logger.error("Failed to create local git analyzer", error=str(e))
            return None

    async def _get_remote_analyzer(
        self,
        repo_url: str,
        org_id: str,
        platform: AnalyzerType,
    ) -> CodeAnalyzerProtocol | None:
        """Get remote API analyzer based on platform."""
        # Get credentials for this org/platform
        access_token = await self._get_platform_credentials(org_id, platform)
        if not access_token:
            logger.warning(
                "No credentials for platform",
                platform=platform.value,
                org_id=org_id,
            )
            return None

        try:
            if platform == AnalyzerType.GITHUB_API:
                from src.services.github_code_analyzer import GitHubCodeAnalyzer

                analyzer = GitHubCodeAnalyzer(
                    repo_url=repo_url,
                    access_token=access_token,
                )
                logger.info(
                    "Using GitHub API analyzer",
                    repo_url=repo_url,
                )
                return analyzer

            elif platform == AnalyzerType.GITLAB_API:
                # TODO: Implement GitLabCodeAnalyzer
                logger.warning("GitLab analyzer not yet implemented")
                return None

            elif platform == AnalyzerType.BITBUCKET_API:
                # TODO: Implement BitbucketCodeAnalyzer
                logger.warning("Bitbucket analyzer not yet implemented")
                return None

            elif platform == AnalyzerType.AZURE_DEVOPS:
                # TODO: Implement AzureDevOpsCodeAnalyzer
                logger.warning("Azure DevOps analyzer not yet implemented")
                return None

            return None

        except Exception as e:
            logger.error(
                "Failed to create remote analyzer",
                platform=platform.value,
                error=str(e),
            )
            return None

    async def _get_project_repo_url(self, project_id: str) -> str | None:
        """Fetch project's repository URL from database."""
        try:
            from src.services.supabase_client import get_supabase_client

            supabase = get_supabase_client()
            if not supabase.is_configured:
                return None

            result = await supabase.request(
                f"/projects?id=eq.{project_id}&select=repository_url"
            )
            data = result.get("data", [])

            if data and data[0].get("repository_url"):
                return data[0]["repository_url"]

            return None

        except Exception as e:
            logger.error("Failed to fetch project repo URL", error=str(e))
            return None

    async def _get_platform_credentials(
        self,
        org_id: str,
        platform: AnalyzerType,
    ) -> str | None:
        """Get API credentials for a platform from org's integrations."""
        try:
            from src.services.supabase_client import get_supabase_client

            supabase = get_supabase_client()
            if not supabase.is_configured:
                return None

            # Map platform to integration type
            platform_to_type = {
                AnalyzerType.GITHUB_API: "github",
                AnalyzerType.GITLAB_API: "gitlab",
                AnalyzerType.BITBUCKET_API: "bitbucket",
                AnalyzerType.AZURE_DEVOPS: "azure_devops",
            }

            integration_type = platform_to_type.get(platform)
            if not integration_type:
                return None

            result = await supabase.request(
                f"/integrations?organization_id=eq.{org_id}&type=eq.{integration_type}&status=eq.active&select=credentials"
            )
            data = result.get("data", [])

            if data:
                credentials = data[0].get("credentials", {})
                return credentials.get("access_token")

            return None

        except Exception as e:
            logger.error(
                "Failed to fetch platform credentials",
                platform=platform.value,
                error=str(e),
            )
            return None

    async def analyze_with_caching(
        self,
        project_id: str,
        org_id: str,
        selector: str,
        file_hint: str | None = None,
        local_repo_path: str | None = None,
    ) -> tuple[Any | None, AnalyzerType]:
        """Analyze a selector with automatic caching.

        This is the main entry point for code-aware healing. It:
        1. Checks cache first
        2. Uses local git if available
        3. Falls back to remote API with rate limiting
        4. Caches successful results

        Args:
            project_id: Project ID
            org_id: Organization ID
            selector: The failed selector to analyze
            file_hint: Optional file path hint
            local_repo_path: Optional local repo path

        Returns:
            Tuple of (SelectorChange or None, AnalyzerType used)
        """
        # Get repo URL for cache key
        repo_url = await self._get_project_repo_url(project_id)
        if not repo_url and not local_repo_path:
            return None, AnalyzerType.NONE

        cache_repo = repo_url or local_repo_path or ""

        # 1. Check cache first
        cached = self._check_cache(selector, cache_repo, file_hint)
        if cached:
            return cached.result, AnalyzerType.CACHED

        # 2. Get analyzer
        analyzer = await self.get_analyzer(
            project_id=project_id,
            org_id=org_id,
            local_repo_path=local_repo_path,
        )

        if not analyzer:
            return None, AnalyzerType.NONE

        # 3. Run analysis
        try:
            result = await analyzer.find_replacement_selector(
                failed_selector=selector,
                file_hint=file_hint,
            )

            # Determine analyzer type for logging
            analyzer_type = AnalyzerType.LOCAL_GIT
            if hasattr(analyzer, "api_base"):  # Remote analyzer
                platform = self._detect_platform(repo_url or "")
                analyzer_type = platform or AnalyzerType.GITHUB_API

                # Record API call for rate limiting
                rate_limit = self._get_rate_limit(org_id, analyzer_type)
                rate_limit.record_call()

            # 4. Cache result (even if None - prevents repeated lookups)
            self._store_cache(
                selector=selector,
                repo_url=cache_repo,
                result=result,
                analyzer_type=analyzer_type,
                file_hint=file_hint,
            )

            return result, analyzer_type

        except Exception as e:
            logger.error(
                "Code analysis failed",
                selector=selector[:30],
                error=str(e),
            )
            return None, AnalyzerType.NONE

    def clear_cache(self, repo_url: str | None = None) -> int:
        """Clear cached analyses.

        Args:
            repo_url: If provided, only clear cache for this repo

        Returns:
            Number of entries cleared
        """
        if repo_url:
            to_remove = [
                k for k, v in self._cache.items()
                if v.repo_url == repo_url
            ]
            for key in to_remove:
                del self._cache[key]
            return len(to_remove)
        else:
            count = len(self._cache)
            self._cache.clear()
            return count

    def clear_analyzer_cache(
        self,
        project_id: str | None = None,
        org_id: str | None = None,
    ) -> int:
        """Clear cached analyzer instances (RAP-352).

        Args:
            project_id: If provided with org_id, only clear cache for this project
            org_id: If provided with project_id, only clear cache for this project
                   If only org_id provided, clear all caches for this org

        Returns:
            Number of entries cleared
        """
        if project_id and org_id:
            # Clear specific project cache
            key = self._get_analyzer_cache_key(project_id, org_id)
            if key in self._analyzer_cache:
                del self._analyzer_cache[key]
                return 1
            return 0
        elif org_id:
            # Clear all caches for this org
            to_remove = [
                k for k in self._analyzer_cache.keys()
                if k.startswith(f"{org_id}:")
            ]
            for key in to_remove:
                del self._analyzer_cache[key]
            return len(to_remove)
        else:
            # Clear all analyzer caches
            count = len(self._analyzer_cache)
            self._analyzer_cache.clear()
            return count

    def clear_all_caches(self) -> dict[str, int]:
        """Clear all caches (analysis results and analyzer instances).

        Returns:
            Dict with counts of cleared entries per cache type
        """
        analysis_count = self.clear_cache()
        analyzer_count = self.clear_analyzer_cache()
        return {
            "analysis_cache": analysis_count,
            "analyzer_cache": analyzer_count,
        }

    def get_rate_limit_status(self, org_id: str) -> dict[str, Any]:
        """Get rate limit status for all platforms for an org."""
        status = {}
        for platform in AnalyzerType:
            # Skip non-API types (no rate limits)
            if platform in (AnalyzerType.LOCAL_GIT, AnalyzerType.CACHED, AnalyzerType.NONE):
                continue

            key = f"{org_id}:{platform.value}"
            if key in self._rate_limits:
                rl = self._rate_limits[key]
                status[platform.value] = {
                    "calls_made": rl.calls_made,
                    "limit_per_hour": rl.limit_per_hour,
                    "remaining": rl.limit_per_hour - rl.calls_made,
                    "can_call": rl.can_make_call(),
                    "retry_after": rl.retry_after,
                }
        return status

    def get_analyzer_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the analyzer cache (RAP-352).

        Returns:
            Dict with cache statistics
        """
        now = time.time()
        active_entries = 0
        expired_entries = 0
        by_type: dict[str, int] = {}

        for key, cached in self._analyzer_cache.items():
            age_seconds = now - cached.created_at
            if age_seconds < self.ANALYZER_CACHE_TTL_SECONDS:
                active_entries += 1
            else:
                expired_entries += 1

            type_name = cached.analyzer_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_entries": len(self._analyzer_cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "ttl_seconds": self.ANALYZER_CACHE_TTL_SECONDS,
            "by_analyzer_type": by_type,
        }


# Global factory instance
_factory: CodeAnalyzerFactory | None = None


def get_code_analyzer_factory() -> CodeAnalyzerFactory:
    """Get the global CodeAnalyzerFactory instance."""
    global _factory
    if _factory is None:
        _factory = CodeAnalyzerFactory()
    return _factory
