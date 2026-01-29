"""PostHog Integration Plugin for session replay and analytics.

PostHog is an open-source product analytics platform that provides session replay,
feature flags, A/B testing, and more. This plugin supports both cloud and self-hosted
instances.

API Docs: https://posthog.com/docs/api
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx
import structlog

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)

logger = structlog.get_logger()

# Default PostHog cloud URLs
POSTHOG_CLOUD_US_URL = "https://app.posthog.com"
POSTHOG_CLOUD_EU_URL = "https://eu.posthog.com"


@dataclass
class PostHogEvent:
    """A PostHog event."""

    id: str
    uuid: str
    event: str  # Event name
    distinct_id: str
    timestamp: datetime
    properties: dict[str, Any] = field(default_factory=dict)
    elements: list[dict[str, Any]] = field(default_factory=list)  # Autocapture elements
    # Session info
    session_id: str | None = None
    session_recording_id: str | None = None


@dataclass
class PostHogPerson:
    """A PostHog person (user)."""

    id: str
    uuid: str
    distinct_ids: list[str]
    created_at: datetime
    properties: dict[str, Any] = field(default_factory=dict)
    is_identified: bool = False
    # Computed properties
    name: str | None = None
    email: str | None = None


@dataclass
class PostHogSessionRecording:
    """A PostHog session recording."""

    id: str
    session_id: str
    distinct_id: str
    viewed: bool
    recording_duration: int  # seconds
    start_time: datetime
    end_time: datetime | None
    click_count: int
    keypress_count: int
    console_log_count: int
    console_warn_count: int
    console_error_count: int
    # Session metadata
    start_url: str | None = None
    person: dict[str, Any] | None = None
    matching_events: list[str] = field(default_factory=list)


@dataclass
class PostHogFeatureFlag:
    """A PostHog feature flag."""

    id: int
    key: str
    name: str
    active: bool
    created_at: datetime
    filters: dict[str, Any] = field(default_factory=dict)
    rollout_percentage: int | None = None
    # Experiment info
    is_experiment: bool = False
    experiment_variants: list[str] = field(default_factory=list)


class PostHogPlugin(IntegrationPlugin):
    """PostHog integration plugin for analytics and session replay.

    Features:
    - Sync events with filters
    - Sync persons (users)
    - Sync session recordings
    - Sync feature flags
    - Support for self-hosted instances
    - Test connection via projects endpoint

    Usage:
        async with PostHogPlugin() as plugin:
            # For cloud
            credentials = IntegrationCredentials(api_key="phx_...")

            # For self-hosted
            credentials = IntegrationCredentials(
                api_key="phx_...",
                instance_url="https://posthog.yourcompany.com"
            )

            result = await plugin.test_connection(credentials)
            if result.success:
                sync_result = await plugin.sync_data(credentials)
    """

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get PostHog integration metadata."""
        return IntegrationMetadata(
            slug="posthog",
            name="PostHog",
            description="Open-source product analytics with session replay, feature flags, and A/B testing",
            category=IntegrationCategory.SESSION_REPLAY,
            auth_type=AuthType.API_KEY,
            icon_url="https://posthog.com/static/posthog-icon.svg",
            docs_url="https://posthog.com/docs/api",
            website_url="https://posthog.com",
            features=[
                "session_replay",
                "event_tracking",
                "user_identification",
                "feature_flags",
                "ab_testing",
                "autocapture",
                "console_logging",
                "network_logging",
                "heatmaps",
                "funnels",
                "cohorts",
                "self_hosted_support",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Project API Key",
                    type="password",
                    required=True,
                    placeholder="phx_...",
                    help_text="Found in PostHog Project Settings > Project API Key",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="Instance URL",
                    type="url",
                    required=False,
                    placeholder="https://app.posthog.com",
                    help_text="For self-hosted instances. Leave empty for PostHog Cloud US",
                ),
                FieldDefinition(
                    name="project_slug",
                    label="Project ID",
                    type="text",
                    required=False,
                    placeholder="your-project-id",
                    help_text="Specific project ID if you have multiple projects",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=[
                "events",
                "persons",
                "session_recordings",
                "feature_flags",
            ],
        )

    def _get_base_url(self, credentials: IntegrationCredentials) -> str:
        """Get the base URL for PostHog API requests."""
        if credentials.instance_url:
            return credentials.instance_url.rstrip("/")
        return POSTHOG_CLOUD_US_URL

    def _get_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for PostHog API requests."""
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test PostHog connection by fetching projects.

        Args:
            credentials: The credentials to test.

        Returns:
            ConnectionTestResult with success status and details.
        """
        log = logger.bind(plugin="posthog", action="test_connection")

        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="Project API key is required",
                error_code="MISSING_API_KEY",
            )

        base_url = self._get_base_url(credentials)
        start_time = time.time()

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/",
                headers=self._get_headers(credentials),
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                projects = data.get("results", [data] if isinstance(data, dict) else [])

                log.info(
                    "PostHog connection successful",
                    instance=base_url,
                    project_count=len(projects),
                    latency_ms=latency_ms,
                )

                return ConnectionTestResult(
                    success=True,
                    message=f"Connected to PostHog with {len(projects)} project(s)",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=[
                        "read:events",
                        "read:persons",
                        "read:session_recordings",
                        "read:feature_flags",
                    ],
                    account_info={
                        "instance_url": base_url,
                        "projects": [
                            {
                                "id": p.get("id"),
                                "name": p.get("name"),
                                "uuid": p.get("uuid"),
                            }
                            for p in projects[:5]  # Limit to 5 projects
                        ],
                    },
                )

            elif response.status_code == 401:
                log.warning("PostHog authentication failed")
                return ConnectionTestResult(
                    success=False,
                    message="Invalid API key",
                    latency_ms=latency_ms,
                    error_code="UNAUTHORIZED",
                )

            elif response.status_code == 403:
                log.warning("PostHog access forbidden")
                return ConnectionTestResult(
                    success=False,
                    message="API key does not have sufficient permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                )

            else:
                log.error(
                    "PostHog connection failed",
                    status_code=response.status_code,
                    response=response.text[:500],
                )
                return ConnectionTestResult(
                    success=False,
                    message=f"Connection failed with status {response.status_code}",
                    latency_ms=latency_ms,
                    error_code=f"HTTP_{response.status_code}",
                    error_details={"response": response.text[:500]},
                )

        except httpx.TimeoutException:
            log.error("PostHog connection timed out")
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except Exception as e:
            log.error("PostHog connection error", error=str(e))
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"exception": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from PostHog.

        Args:
            credentials: The credentials to use.
            resources: Optional list of resources to sync (events, persons, session_recordings, feature_flags).
            since: Optional datetime to sync data since.
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced data and statistics.
        """
        log = logger.bind(plugin="posthog", action="sync_data")

        started_at = datetime.utcnow()
        sync_resources = resources or [
            "events",
            "persons",
            "session_recordings",
            "feature_flags",
        ]
        errors: list[str] = []
        warnings: list[str] = []
        data: dict[str, Any] = {}
        total_synced = 0

        log.info("Starting PostHog sync", resources=sync_resources, limit=limit)

        # Get project ID
        project_id = credentials.project_slug
        if not project_id:
            project_id = await self._get_default_project_id(credentials)
            if not project_id:
                return SyncResult(
                    success=False,
                    message="Could not determine project ID",
                    errors=["No project ID provided and could not fetch default"],
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

        # Sync events
        if "events" in sync_resources:
            try:
                events = await self._sync_events(credentials, project_id, since, limit)
                data["events"] = [self._event_to_dict(e) for e in events]
                total_synced += len(events)
                log.info("Synced PostHog events", count=len(events))
            except Exception as e:
                errors.append(f"Failed to sync events: {str(e)}")
                log.error("Failed to sync events", error=str(e))

        # Sync persons
        if "persons" in sync_resources:
            try:
                persons = await self._sync_persons(
                    credentials, project_id, since, limit
                )
                data["persons"] = [self._person_to_dict(p) for p in persons]
                total_synced += len(persons)
                log.info("Synced PostHog persons", count=len(persons))
            except Exception as e:
                errors.append(f"Failed to sync persons: {str(e)}")
                log.error("Failed to sync persons", error=str(e))

        # Sync session recordings
        if "session_recordings" in sync_resources:
            try:
                recordings = await self._sync_session_recordings(
                    credentials, project_id, since, limit
                )
                data["session_recordings"] = [
                    self._recording_to_dict(r) for r in recordings
                ]
                total_synced += len(recordings)
                log.info("Synced PostHog session recordings", count=len(recordings))
            except Exception as e:
                errors.append(f"Failed to sync session recordings: {str(e)}")
                log.error("Failed to sync session recordings", error=str(e))

        # Sync feature flags
        if "feature_flags" in sync_resources:
            try:
                flags = await self._sync_feature_flags(credentials, project_id, limit)
                data["feature_flags"] = [self._flag_to_dict(f) for f in flags]
                total_synced += len(flags)
                log.info("Synced PostHog feature flags", count=len(flags))
            except Exception as e:
                errors.append(f"Failed to sync feature flags: {str(e)}")
                log.error("Failed to sync feature flags", error=str(e))

        completed_at = datetime.utcnow()
        success = len(errors) == 0

        return SyncResult(
            success=success,
            message=f"Synced {total_synced} items from PostHog"
            if success
            else f"Sync completed with {len(errors)} error(s)",
            data_points_synced=total_synced,
            started_at=started_at,
            completed_at=completed_at,
            errors=errors,
            warnings=warnings,
            data=data,
        )

    async def _get_default_project_id(
        self, credentials: IntegrationCredentials
    ) -> str | None:
        """Get the default project ID."""
        base_url = self._get_base_url(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/",
                headers=self._get_headers(credentials),
            )
            if response.status_code == 200:
                data = response.json()
                projects = data.get("results", [data] if isinstance(data, dict) else [])
                if projects:
                    return str(projects[0].get("id"))
        except Exception:
            pass
        return None

    async def _sync_events(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        since: datetime | None,
        limit: int,
    ) -> list[PostHogEvent]:
        """Sync events from PostHog."""
        events = []
        base_url = self._get_base_url(credentials)

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["after"] = since.isoformat() + "Z"

        response = await self.http_client.get(
            f"{base_url}/api/projects/{project_id}/events/",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for event_data in data.get("results", []):
            event = self._parse_event(event_data)
            if event:
                events.append(event)

            if len(events) >= limit:
                break

        return events

    async def _sync_persons(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        since: datetime | None,
        limit: int,
    ) -> list[PostHogPerson]:
        """Sync persons from PostHog."""
        persons = []
        base_url = self._get_base_url(credentials)

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["created_after"] = since.isoformat() + "Z"

        response = await self.http_client.get(
            f"{base_url}/api/projects/{project_id}/persons/",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for person_data in data.get("results", []):
            person = self._parse_person(person_data)
            if person:
                persons.append(person)

            if len(persons) >= limit:
                break

        return persons

    async def _sync_session_recordings(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        since: datetime | None,
        limit: int,
    ) -> list[PostHogSessionRecording]:
        """Sync session recordings from PostHog."""
        recordings = []
        base_url = self._get_base_url(credentials)

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        if since:
            params["date_from"] = since.strftime("%Y-%m-%d")

        response = await self.http_client.get(
            f"{base_url}/api/projects/{project_id}/session_recordings/",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for recording_data in data.get("results", []):
            recording = self._parse_recording(recording_data)
            if recording:
                recordings.append(recording)

            if len(recordings) >= limit:
                break

        return recordings

    async def _sync_feature_flags(
        self,
        credentials: IntegrationCredentials,
        project_id: str,
        limit: int,
    ) -> list[PostHogFeatureFlag]:
        """Sync feature flags from PostHog."""
        flags = []
        base_url = self._get_base_url(credentials)

        params: dict[str, Any] = {
            "limit": min(limit, 100),
        }

        response = await self.http_client.get(
            f"{base_url}/api/projects/{project_id}/feature_flags/",
            headers=self._get_headers(credentials),
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        for flag_data in data.get("results", []):
            flag = self._parse_feature_flag(flag_data)
            if flag:
                flags.append(flag)

            if len(flags) >= limit:
                break

        return flags

    def _parse_event(self, data: dict[str, Any]) -> PostHogEvent | None:
        """Parse an event from API response."""
        try:
            timestamp = datetime.fromisoformat(
                data.get("timestamp", "").replace("Z", "+00:00")
            )

            properties = data.get("properties", {})

            return PostHogEvent(
                id=str(data.get("id", "")),
                uuid=data.get("uuid", ""),
                event=data.get("event", ""),
                distinct_id=data.get("distinct_id", ""),
                timestamp=timestamp,
                properties=properties,
                elements=data.get("elements", []),
                session_id=properties.get("$session_id"),
                session_recording_id=properties.get("$session_recording_id"),
            )
        except Exception:
            return None

    def _parse_person(self, data: dict[str, Any]) -> PostHogPerson | None:
        """Parse a person from API response."""
        try:
            created_at = datetime.fromisoformat(
                data.get("created_at", "").replace("Z", "+00:00")
            )

            properties = data.get("properties", {})

            return PostHogPerson(
                id=str(data.get("id", "")),
                uuid=data.get("uuid", ""),
                distinct_ids=data.get("distinct_ids", []),
                created_at=created_at,
                properties=properties,
                is_identified=data.get("is_identified", False),
                name=properties.get("name") or properties.get("$name"),
                email=properties.get("email") or properties.get("$email"),
            )
        except Exception:
            return None

    def _parse_recording(
        self, data: dict[str, Any]
    ) -> PostHogSessionRecording | None:
        """Parse a session recording from API response."""
        try:
            start_time = datetime.fromisoformat(
                data.get("start_time", "").replace("Z", "+00:00")
            )
            end_time = None
            if data.get("end_time"):
                end_time = datetime.fromisoformat(
                    data["end_time"].replace("Z", "+00:00")
                )

            return PostHogSessionRecording(
                id=data.get("id", ""),
                session_id=data.get("session_id", ""),
                distinct_id=data.get("distinct_id", ""),
                viewed=data.get("viewed", False),
                recording_duration=data.get("recording_duration", 0),
                start_time=start_time,
                end_time=end_time,
                click_count=data.get("click_count", 0),
                keypress_count=data.get("keypress_count", 0),
                console_log_count=data.get("console_log_count", 0),
                console_warn_count=data.get("console_warn_count", 0),
                console_error_count=data.get("console_error_count", 0),
                start_url=data.get("start_url"),
                person=data.get("person"),
                matching_events=data.get("matching_events", []),
            )
        except Exception:
            return None

    def _parse_feature_flag(self, data: dict[str, Any]) -> PostHogFeatureFlag | None:
        """Parse a feature flag from API response."""
        try:
            created_at = datetime.fromisoformat(
                data.get("created_at", "").replace("Z", "+00:00")
            )

            filters = data.get("filters", {})
            rollout_percentage = None
            if filters.get("groups"):
                for group in filters["groups"]:
                    if group.get("rollout_percentage") is not None:
                        rollout_percentage = group["rollout_percentage"]
                        break

            # Check if it's an experiment
            is_experiment = len(filters.get("multivariate", {}).get("variants", [])) > 0
            experiment_variants = []
            if is_experiment:
                experiment_variants = [
                    v.get("key", "")
                    for v in filters.get("multivariate", {}).get("variants", [])
                ]

            return PostHogFeatureFlag(
                id=data.get("id", 0),
                key=data.get("key", ""),
                name=data.get("name", ""),
                active=data.get("active", False),
                created_at=created_at,
                filters=filters,
                rollout_percentage=rollout_percentage,
                is_experiment=is_experiment,
                experiment_variants=experiment_variants,
            )
        except Exception:
            return None

    def _event_to_dict(self, event: PostHogEvent) -> dict[str, Any]:
        """Convert an event to a dictionary."""
        return {
            "id": event.id,
            "uuid": event.uuid,
            "event": event.event,
            "distinct_id": event.distinct_id,
            "timestamp": event.timestamp.isoformat(),
            "properties": event.properties,
            "elements": event.elements,
            "session_id": event.session_id,
            "session_recording_id": event.session_recording_id,
        }

    def _person_to_dict(self, person: PostHogPerson) -> dict[str, Any]:
        """Convert a person to a dictionary."""
        return {
            "id": person.id,
            "uuid": person.uuid,
            "distinct_ids": person.distinct_ids,
            "created_at": person.created_at.isoformat(),
            "properties": person.properties,
            "is_identified": person.is_identified,
            "name": person.name,
            "email": person.email,
        }

    def _recording_to_dict(self, recording: PostHogSessionRecording) -> dict[str, Any]:
        """Convert a session recording to a dictionary."""
        return {
            "id": recording.id,
            "session_id": recording.session_id,
            "distinct_id": recording.distinct_id,
            "viewed": recording.viewed,
            "recording_duration": recording.recording_duration,
            "start_time": recording.start_time.isoformat(),
            "end_time": recording.end_time.isoformat() if recording.end_time else None,
            "click_count": recording.click_count,
            "keypress_count": recording.keypress_count,
            "console_log_count": recording.console_log_count,
            "console_warn_count": recording.console_warn_count,
            "console_error_count": recording.console_error_count,
            "start_url": recording.start_url,
            "person": recording.person,
            "matching_events": recording.matching_events,
        }

    def _flag_to_dict(self, flag: PostHogFeatureFlag) -> dict[str, Any]:
        """Convert a feature flag to a dictionary."""
        return {
            "id": flag.id,
            "key": flag.key,
            "name": flag.name,
            "active": flag.active,
            "created_at": flag.created_at.isoformat(),
            "filters": flag.filters,
            "rollout_percentage": flag.rollout_percentage,
            "is_experiment": flag.is_experiment,
            "experiment_variants": flag.experiment_variants,
        }

    # Additional helper methods for PostHog specific operations

    async def get_recording_by_id(
        self,
        credentials: IntegrationCredentials,
        recording_id: str,
    ) -> PostHogSessionRecording | None:
        """Get detailed information about a specific session recording.

        Args:
            credentials: The credentials to use.
            recording_id: The recording ID to fetch.

        Returns:
            PostHogSessionRecording or None if not found.
        """
        project_id = credentials.project_slug
        if not project_id:
            project_id = await self._get_default_project_id(credentials)
            if not project_id:
                return None

        base_url = self._get_base_url(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/{project_id}/session_recordings/{recording_id}/",
                headers=self._get_headers(credentials),
            )
            if response.status_code == 200:
                return self._parse_recording(response.json())
        except Exception:
            pass

        return None

    async def get_person_by_distinct_id(
        self,
        credentials: IntegrationCredentials,
        distinct_id: str,
    ) -> PostHogPerson | None:
        """Get a person by distinct ID.

        Args:
            credentials: The credentials to use.
            distinct_id: The distinct ID to search for.

        Returns:
            PostHogPerson or None if not found.
        """
        project_id = credentials.project_slug
        if not project_id:
            project_id = await self._get_default_project_id(credentials)
            if not project_id:
                return None

        base_url = self._get_base_url(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/{project_id}/persons/",
                headers=self._get_headers(credentials),
                params={"distinct_id": distinct_id},
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return self._parse_person(results[0])
        except Exception:
            pass

        return None

    async def get_feature_flag_by_key(
        self,
        credentials: IntegrationCredentials,
        flag_key: str,
    ) -> PostHogFeatureFlag | None:
        """Get a feature flag by key.

        Args:
            credentials: The credentials to use.
            flag_key: The feature flag key.

        Returns:
            PostHogFeatureFlag or None if not found.
        """
        project_id = credentials.project_slug
        if not project_id:
            project_id = await self._get_default_project_id(credentials)
            if not project_id:
                return None

        base_url = self._get_base_url(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/{project_id}/feature_flags/",
                headers=self._get_headers(credentials),
                params={"key": flag_key},
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return self._parse_feature_flag(results[0])
        except Exception:
            pass

        return None

    async def get_recordings_with_errors(
        self,
        credentials: IntegrationCredentials,
        min_errors: int = 1,
        limit: int = 50,
    ) -> list[PostHogSessionRecording]:
        """Get session recordings that have console errors.

        Args:
            credentials: The credentials to use.
            min_errors: Minimum number of console errors.
            limit: Maximum number of results.

        Returns:
            List of session recordings with console errors.
        """
        project_id = credentials.project_slug
        if not project_id:
            project_id = await self._get_default_project_id(credentials)
            if not project_id:
                return []

        base_url = self._get_base_url(credentials)

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/{project_id}/session_recordings/",
                headers=self._get_headers(credentials),
                params={
                    "limit": limit * 2,  # Fetch more to filter
                    "console_error_count__gte": min_errors,
                },
            )
            if response.status_code == 200:
                recordings = []
                for recording_data in response.json().get("results", []):
                    recording = self._parse_recording(recording_data)
                    if recording and recording.console_error_count >= min_errors:
                        recordings.append(recording)
                    if len(recordings) >= limit:
                        break
                return recordings
        except Exception:
            pass

        return []

    async def get_events_for_person(
        self,
        credentials: IntegrationCredentials,
        distinct_id: str,
        event_names: list[str] | None = None,
        limit: int = 50,
    ) -> list[PostHogEvent]:
        """Get events for a specific person.

        Args:
            credentials: The credentials to use.
            distinct_id: The person's distinct ID.
            event_names: Optional list of event names to filter.
            limit: Maximum number of results.

        Returns:
            List of events for the person.
        """
        project_id = credentials.project_slug
        if not project_id:
            project_id = await self._get_default_project_id(credentials)
            if not project_id:
                return []

        base_url = self._get_base_url(credentials)

        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "distinct_id": distinct_id,
        }

        if event_names:
            params["event"] = event_names[0]  # PostHog API filters by single event

        try:
            response = await self.http_client.get(
                f"{base_url}/api/projects/{project_id}/events/",
                headers=self._get_headers(credentials),
                params=params,
            )
            if response.status_code == 200:
                events = []
                for event_data in response.json().get("results", []):
                    event = self._parse_event(event_data)
                    if event:
                        # Additional filtering for multiple event names
                        if event_names and event.event not in event_names:
                            continue
                        events.append(event)
                    if len(events) >= limit:
                        break
                return events
        except Exception:
            pass

        return []


def create_posthog_plugin(
    http_client: httpx.AsyncClient | None = None,
) -> PostHogPlugin:
    """Factory function for PostHogPlugin.

    Args:
        http_client: Optional HTTP client to use.

    Returns:
        PostHogPlugin instance.
    """
    return PostHogPlugin(http_client=http_client)
