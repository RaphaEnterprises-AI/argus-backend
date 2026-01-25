"""
LaunchDarkly Integration for feature flag awareness.

Tracks feature flags to ensure tests cover the right variants.

API Docs: https://apidocs.launchdarkly.com/
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class FeatureFlag:
    """A LaunchDarkly feature flag."""

    key: str
    name: str
    description: str | None
    kind: str  # boolean, multivariate, etc.

    # State
    on: bool  # Is the flag on?

    # Variations
    variations: list[dict]
    default_on_variation: int
    default_off_variation: int

    # Targeting
    targets: list[dict]
    rules: list[dict]

    # Rollout percentage (if applicable)
    rollout_percentage: float | None

    # Metadata
    tags: list[str]
    created_at: datetime

    # Environment
    environment: str


@dataclass
class FlagChange:
    """A change to a feature flag (audit log entry)."""

    flag_key: str
    action: str  # 'updateOn', 'updateOff', 'updateRules', etc.
    member_email: str
    timestamp: datetime
    previous_value: dict | None
    new_value: dict | None


@dataclass
class FlagEvaluationContext:
    """Context for evaluating feature flags during tests."""

    user_key: str
    email: str | None = None
    name: str | None = None
    custom_attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to LaunchDarkly context format."""
        context = {
            "kind": "user",
            "key": self.user_key,
        }
        if self.email:
            context["email"] = self.email
        if self.name:
            context["name"] = self.name
        if self.custom_attributes:
            context.update(self.custom_attributes)
        return context


class LaunchDarklyIntegration:
    """
    LaunchDarkly Integration.

    API Docs: https://apidocs.launchdarkly.com/

    Features:
    - List feature flags
    - Get flag state and targeting rules
    - Track flag changes (audit log)
    - Calculate rollout percentages from targeting rules
    - Correlate flag changes with errors

    Usage:
        ld = LaunchDarklyIntegration(api_token="api-xxx", project_key="my-project")

        # List all flags
        flags = await ld.get_flags(environment="production")

        # Get specific flag
        flag = await ld.get_flag("new-checkout-flow", environment="production")

        # Get recent changes
        changes = await ld.get_audit_log(limit=50)

        # Close when done
        await ld.close()
    """

    def __init__(self, api_token: str | None = None, project_key: str = "default"):
        self.api_token = api_token or os.environ.get("LAUNCHDARKLY_API_TOKEN")
        self.project_key = project_key
        self.base_url = "https://app.launchdarkly.com/api/v2"
        self.http = httpx.AsyncClient(timeout=30.0)
        self.log = logger.bind(component="launchdarkly")

        if not self.api_token:
            self.log.warning(
                "No LaunchDarkly API token provided - integration will be disabled"
            )

    @property
    def headers(self) -> dict:
        """Get headers for LaunchDarkly API requests."""
        return {
            "Authorization": self.api_token,
            "Content-Type": "application/json",
        }

    async def test_connection(self) -> bool:
        """
        Test if API token is valid.

        Returns:
            True if connection is valid, False otherwise
        """
        if not self.api_token:
            return False

        try:
            response = await self.http.get(
                f"{self.base_url}/projects/{self.project_key}",
                headers=self.headers,
            )
            if response.status_code == 200:
                self.log.info("LaunchDarkly connection verified", project=self.project_key)
                return True
            elif response.status_code == 401:
                self.log.error("LaunchDarkly authentication failed - invalid token")
                return False
            elif response.status_code == 404:
                self.log.error("LaunchDarkly project not found", project=self.project_key)
                return False
            else:
                self.log.error(
                    "LaunchDarkly connection failed",
                    status_code=response.status_code,
                    response=response.text,
                )
                return False
        except Exception as e:
            self.log.error("LaunchDarkly connection error", error=str(e))
            return False

    async def get_environments(self) -> list[dict]:
        """
        List all environments for the project.

        Returns:
            List of environment dictionaries with keys: key, name, color, etc.
        """
        if not self.api_token:
            self.log.warning("LaunchDarkly token not configured, skipping get_environments")
            return []

        try:
            response = await self.http.get(
                f"{self.base_url}/projects/{self.project_key}/environments",
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            environments = data.get("items", [])
            self.log.info(
                "Fetched environments",
                count=len(environments),
                project=self.project_key,
            )
            return environments
        except Exception as e:
            self.log.error("Failed to fetch environments", error=str(e))
            return []

    async def get_flags(
        self,
        environment: str = "production",
        limit: int = 50,
        tag: str | None = None,
        filter_query: str | None = None,
    ) -> list[FeatureFlag]:
        """
        Get all feature flags for an environment.

        Args:
            environment: Environment key (e.g., 'production', 'staging')
            limit: Maximum number of flags to return
            tag: Filter by tag
            filter_query: LaunchDarkly filter query string

        Returns:
            List of FeatureFlag objects
        """
        if not self.api_token:
            self.log.warning("LaunchDarkly token not configured, skipping get_flags")
            return []

        params: dict[str, Any] = {
            "limit": limit,
            "env": environment,
        }
        if tag:
            params["tag"] = tag
        if filter_query:
            params["filter"] = filter_query

        try:
            response = await self.http.get(
                f"{self.base_url}/flags/{self.project_key}",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])

            flags = []
            for item in items:
                flag = self._parse_flag(item, environment)
                if flag:
                    flags.append(flag)

            self.log.info(
                "Fetched feature flags",
                count=len(flags),
                environment=environment,
                project=self.project_key,
            )
            return flags
        except Exception as e:
            self.log.error("Failed to fetch flags", error=str(e))
            return []

    async def get_flag(
        self,
        flag_key: str,
        environment: str = "production",
    ) -> FeatureFlag | None:
        """
        Get a single flag by key.

        Args:
            flag_key: The flag key
            environment: Environment key

        Returns:
            FeatureFlag object or None if not found
        """
        if not self.api_token:
            self.log.warning("LaunchDarkly token not configured, skipping get_flag")
            return None

        try:
            response = await self.http.get(
                f"{self.base_url}/flags/{self.project_key}/{flag_key}",
                headers=self.headers,
                params={"env": environment},
            )
            if response.status_code == 404:
                self.log.warning("Flag not found", flag_key=flag_key)
                return None

            response.raise_for_status()
            data = response.json()
            flag = self._parse_flag(data, environment)

            if flag:
                self.log.info(
                    "Fetched flag",
                    flag_key=flag_key,
                    on=flag.on,
                    environment=environment,
                )
            return flag
        except Exception as e:
            self.log.error("Failed to fetch flag", flag_key=flag_key, error=str(e))
            return None

    async def get_flag_status(
        self,
        flag_key: str,
        environment: str = "production",
    ) -> dict:
        """
        Get flag evaluation status (percentage, lastRequested, etc.).

        Args:
            flag_key: The flag key
            environment: Environment key

        Returns:
            Dictionary with status information including:
            - name: Status name (active, inactive, launched, etc.)
            - lastRequested: Last evaluation timestamp
            - default: Default variation served percentage
        """
        if not self.api_token:
            self.log.warning("LaunchDarkly token not configured, skipping get_flag_status")
            return {}

        try:
            response = await self.http.get(
                f"{self.base_url}/flag-statuses/{self.project_key}/{environment}/{flag_key}",
                headers=self.headers,
            )
            if response.status_code == 404:
                self.log.warning("Flag status not found", flag_key=flag_key)
                return {}

            response.raise_for_status()
            data = response.json()

            self.log.info(
                "Fetched flag status",
                flag_key=flag_key,
                status=data.get("name"),
                environment=environment,
            )
            return data
        except Exception as e:
            self.log.error("Failed to fetch flag status", flag_key=flag_key, error=str(e))
            return {}

    async def get_audit_log(
        self,
        limit: int = 50,
        after: datetime | None = None,
        before: datetime | None = None,
        spec: str | None = None,
    ) -> list[FlagChange]:
        """
        Get recent flag changes from audit log.

        Args:
            limit: Maximum number of entries to return
            after: Only return entries after this timestamp
            before: Only return entries before this timestamp
            spec: Filter spec (e.g., 'flags/my-flag' for specific flag)

        Returns:
            List of FlagChange objects
        """
        if not self.api_token:
            self.log.warning("LaunchDarkly token not configured, skipping get_audit_log")
            return []

        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = int(after.timestamp() * 1000)
        if before:
            params["before"] = int(before.timestamp() * 1000)
        if spec:
            params["spec"] = spec

        try:
            response = await self.http.get(
                f"{self.base_url}/auditlog",
                headers=self.headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])

            changes = []
            for item in items:
                # Filter to flag-related changes
                if not item.get("target", {}).get("resources", []):
                    continue

                resources = item["target"]["resources"]
                flag_resources = [r for r in resources if r.startswith("proj/") and "/flags/" in r]
                if not flag_resources:
                    continue

                # Extract flag key from resource path
                # Format: proj/{project}/env/{env}/flags/{flag_key}
                for resource in flag_resources:
                    parts = resource.split("/")
                    flag_key = parts[-1] if "flags" in parts else None
                    if not flag_key:
                        continue

                    change = FlagChange(
                        flag_key=flag_key,
                        action=item.get("kind", "unknown"),
                        member_email=item.get("member", {}).get("email", "unknown"),
                        timestamp=datetime.fromtimestamp(item.get("date", 0) / 1000),
                        previous_value=item.get("previousValue"),
                        new_value=item.get("currentValue"),
                    )
                    changes.append(change)

            self.log.info("Fetched audit log", count=len(changes))
            return changes
        except Exception as e:
            self.log.error("Failed to fetch audit log", error=str(e))
            return []

    async def get_flag_changes_for_timeframe(
        self,
        flag_key: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[FlagChange]:
        """
        Get changes for a specific flag within a timeframe.

        Useful for correlating flag changes with errors.

        Args:
            flag_key: The flag key to get changes for
            start_time: Start of timeframe
            end_time: End of timeframe (defaults to now)

        Returns:
            List of FlagChange objects
        """
        return await self.get_audit_log(
            limit=100,
            after=start_time,
            before=end_time,
            spec=f"proj/{self.project_key}/flags/{flag_key}",
        )

    async def get_flags_by_tags(
        self,
        tags: list[str],
        environment: str = "production",
    ) -> list[FeatureFlag]:
        """
        Get flags that have any of the specified tags.

        Useful for getting all flags related to a feature area.

        Args:
            tags: List of tags to filter by
            environment: Environment key

        Returns:
            List of FeatureFlag objects
        """
        all_flags = []
        for tag in tags:
            flags = await self.get_flags(environment=environment, tag=tag, limit=100)
            for flag in flags:
                if flag.key not in [f.key for f in all_flags]:
                    all_flags.append(flag)
        return all_flags

    async def get_active_experiments(
        self,
        environment: str = "production",
    ) -> list[FeatureFlag]:
        """
        Get flags that are currently running experiments (have rollout rules).

        Returns:
            List of FeatureFlag objects that have active rollouts
        """
        flags = await self.get_flags(environment=environment, limit=100)
        return [f for f in flags if f.rollout_percentage is not None and 0 < f.rollout_percentage < 100]

    def _parse_flag(self, data: dict, environment: str) -> FeatureFlag | None:
        """Parse API response into FeatureFlag object."""
        try:
            # Get environment-specific configuration
            env_config = data.get("environments", {}).get(environment, {})
            if not env_config:
                self.log.warning(
                    "Environment not found in flag",
                    flag_key=data.get("key"),
                    environment=environment,
                )
                return None

            # Calculate rollout percentage from targeting rules
            rollout_percentage = self._calculate_rollout_percentage(env_config)

            # Parse creation date
            created_at = datetime.now()
            if data.get("creationDate"):
                created_at = datetime.fromtimestamp(data["creationDate"] / 1000)

            return FeatureFlag(
                key=data.get("key", ""),
                name=data.get("name", ""),
                description=data.get("description"),
                kind=data.get("kind", "boolean"),
                on=env_config.get("on", False),
                variations=data.get("variations", []),
                default_on_variation=env_config.get("fallthrough", {}).get("variation", 0),
                default_off_variation=env_config.get("offVariation", 0),
                targets=env_config.get("targets", []),
                rules=env_config.get("rules", []),
                rollout_percentage=rollout_percentage,
                tags=data.get("tags", []),
                created_at=created_at,
                environment=environment,
            )
        except Exception as e:
            self.log.error("Failed to parse flag", error=str(e), flag_key=data.get("key"))
            return None

    def _calculate_rollout_percentage(self, env_config: dict) -> float | None:
        """
        Calculate the rollout percentage from targeting rules.

        Handles:
        - Percentage rollouts in fallthrough
        - Weighted variations in rules
        - Simple on/off states

        Returns:
            Rollout percentage (0-100) or None if not a rollout
        """
        # Check if flag is off - no rollout
        if not env_config.get("on", False):
            return None

        # Check fallthrough for percentage rollout
        fallthrough = env_config.get("fallthrough", {})

        # Check for rollout in fallthrough
        rollout = fallthrough.get("rollout")
        if rollout:
            weighted_variations = rollout.get("variations", [])
            if weighted_variations:
                # Calculate percentage for first variation (typically the "on" variation)
                # This handles multi-variant flags
                total_weight = sum(v.get("weight", 0) for v in weighted_variations)
                if total_weight > 0:
                    # For boolean flags, return the percentage for the "true" variation
                    # For multi-variant flags, return percentage for variation 0
                    first_weight = weighted_variations[0].get("weight", 0)
                    return (first_weight / total_weight) * 100

        # Check for experiment percentage in rules
        rules = env_config.get("rules", [])
        for rule in rules:
            rule_rollout = rule.get("rollout")
            if rule_rollout:
                weighted_variations = rule_rollout.get("variations", [])
                if weighted_variations:
                    total_weight = sum(v.get("weight", 0) for v in weighted_variations)
                    if total_weight > 0:
                        first_weight = weighted_variations[0].get("weight", 0)
                        return (first_weight / total_weight) * 100

        # If there's a simple fallthrough variation, it's either 0% or 100%
        if "variation" in fallthrough:
            variation_idx = fallthrough["variation"]
            # Return 100% if serving variation 0 (typically "true" for boolean)
            return 100.0 if variation_idx == 0 else 0.0

        return None

    async def close(self):
        """Close the HTTP client."""
        await self.http.aclose()
        self.log.info("LaunchDarkly integration closed")


def create_launchdarkly_integration(
    api_token: str | None = None,
    project_key: str = "default",
) -> LaunchDarklyIntegration:
    """Factory function for LaunchDarklyIntegration."""
    return LaunchDarklyIntegration(api_token=api_token, project_key=project_key)
