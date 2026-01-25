"""
Amplitude Integration for product analytics.

Uses feature usage data to prioritize what tests matter most.
Focuses on understanding real user behavior to drive intelligent test prioritization.

API Docs: https://www.docs.developers.amplitude.com/analytics/apis/
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import httpx
import structlog

logger = structlog.get_logger()


class SegmentationType(str, Enum):
    """Segmentation types for Amplitude queries."""
    UNIQUES = "uniques"
    TOTALS = "totals"
    AVG = "avg"
    FORMULA = "formula"
    PROPERTY_SUM = "property_sum"


class RetentionMode(str, Enum):
    """Retention calculation modes."""
    UNBOUNDED = "unbounded"
    BRACKET = "bracket"
    NCOUNTING = "n-counting"


@dataclass
class EventMetrics:
    """Metrics for an Amplitude event."""
    event_type: str
    total_count: int
    unique_users: int
    avg_per_user: float
    # Additional metrics
    properties: dict = field(default_factory=dict)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


@dataclass
class UserPath:
    """A common user path/journey."""
    steps: list[str]  # Event sequence
    user_count: int
    conversion_rate: float
    avg_time_between_steps: list[float]  # In seconds
    # Additional context
    drop_off_at_step: dict[int, int] = field(default_factory=dict)  # step index -> drop off count


@dataclass
class FunnelStep:
    """A step in a funnel."""
    event_name: str
    count: int
    conversion_rate: float
    drop_off_rate: float
    # Additional metrics
    avg_time_to_convert: float | None = None  # Seconds to this step from previous


@dataclass
class RetentionData:
    """Retention analysis data."""
    starting_event: str
    returning_event: str
    cohort_size: int
    retention_by_day: dict[int, float]  # day number -> retention percentage
    avg_retention: float
    start_date: datetime
    end_date: datetime


@dataclass
class EventProperty:
    """Metadata about an event property."""
    property_name: str
    property_type: str
    top_values: list[dict]  # value, count


@dataclass
class ChartData:
    """Response from chart/segmentation APIs."""
    series: list[dict]
    xvalues: list[str]  # Date strings
    total: int | None = None


class AmplitudeIntegration:
    """
    Amplitude Integration for product analytics.

    API Docs: https://www.docs.developers.amplitude.com/analytics/apis/

    Features:
    - Get event metrics (which features are used most)
    - Get user paths (common journeys)
    - Get funnel data (conversion and drop-off)
    - Get retention data
    - Prioritize testing based on feature usage

    Usage:
        amplitude = AmplitudeIntegration(
            api_key="your_api_key",
            secret_key="your_secret_key"
        )

        # Get most used features
        top_events = await amplitude.get_top_events(limit=20)
        for event in top_events:
            print(f"{event.event_type}: {event.unique_users} users")

        # Get user journey data
        paths = await amplitude.get_user_paths(
            starting_event="page_view",
            limit=10
        )

        # Get funnel conversion
        funnel = await amplitude.get_funnel(
            events=["signup_start", "email_verified", "profile_complete"]
        )

        await amplitude.close()
    """

    def __init__(self, api_key: str, secret_key: str):
        """
        Initialize Amplitude integration.

        Args:
            api_key: Amplitude API Key
            secret_key: Amplitude Secret Key
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://amplitude.com/api/2"
        self.http = httpx.AsyncClient(timeout=60.0)  # Longer timeout for analytics
        self.log = logger.bind(component="amplitude")

    @property
    def auth(self) -> tuple[str, str]:
        """Get basic auth credentials."""
        return (self.api_key, self.secret_key)

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for Amplitude API (YYYYMMDD)."""
        return dt.strftime("%Y%m%d")

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime with time for Amplitude API."""
        return dt.strftime("%Y%m%dT%H")

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict | None:
        """Make an authenticated request to Amplitude API."""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = await self.http.request(
                method=method,
                url=url,
                auth=self.auth,
                params=params,
                json=json_data,
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                self.log.error("Authentication failed - check API credentials")
                return None
            elif response.status_code == 429:
                self.log.warning("Rate limited by Amplitude API")
                # Could implement retry with backoff here
                return None
            else:
                self.log.error(
                    "Amplitude API error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                return None

        except httpx.TimeoutException:
            self.log.error("Amplitude API request timed out", endpoint=endpoint)
            return None
        except Exception as e:
            self.log.error("Amplitude API request failed", error=str(e))
            return None

    async def get_event_metrics(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[EventMetrics]:
        """
        Get metrics for all events (most used features).

        Uses the Event Segmentation API to get comprehensive metrics.

        Args:
            start_date: Start of date range (default: 30 days ago)
            end_date: End of date range (default: now)
            limit: Maximum number of events to return

        Returns:
            List of EventMetrics sorted by unique users (descending)
        """
        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        # Use Event Segmentation API for all events
        params = {
            "e": '{"event_type":"_all"}',  # All events
            "m": "uniques",  # Unique users
            "start": self._format_date(start),
            "end": self._format_date(end),
            "i": "1",  # Daily granularity
        }

        data = await self._request("GET", "events/segmentation", params=params)
        if not data:
            return []

        # Parse the segmentation response
        events = []
        series_data = data.get("data", {}).get("series", [])
        event_labels = data.get("data", {}).get("seriesLabels", [])

        for idx, label in enumerate(event_labels[:limit]):
            if idx < len(series_data):
                series = series_data[idx]
                total_uniques = sum(v or 0 for v in series)

                # Get totals for this specific event
                totals = await self._get_event_totals(label, start, end)

                events.append(EventMetrics(
                    event_type=label,
                    total_count=totals.get("total", 0),
                    unique_users=total_uniques,
                    avg_per_user=totals.get("total", 0) / max(total_uniques, 1),
                ))

        # Sort by unique users
        events.sort(key=lambda e: e.unique_users, reverse=True)
        return events[:limit]

    async def _get_event_totals(
        self,
        event_type: str,
        start: datetime,
        end: datetime,
    ) -> dict:
        """Get total counts for a specific event."""
        import json
        params = {
            "e": json.dumps({"event_type": event_type}),
            "m": "totals",
            "start": self._format_date(start),
            "end": self._format_date(end),
        }

        data = await self._request("GET", "events/segmentation", params=params)
        if not data:
            return {"total": 0}

        series = data.get("data", {}).get("series", [[]])
        if series and series[0]:
            return {"total": sum(v or 0 for v in series[0])}
        return {"total": 0}

    async def get_top_events(
        self,
        limit: int = 20,
        days: int = 30,
    ) -> list[EventMetrics]:
        """
        Get the most frequently triggered events.

        Convenience method for getting top events by usage.

        Args:
            limit: Number of top events to return
            days: Number of days to analyze

        Returns:
            List of EventMetrics for top events
        """
        self.log.info("Fetching top events", limit=limit, days=days)

        start = datetime.utcnow() - timedelta(days=days)
        end = datetime.utcnow()

        return await self.get_event_metrics(
            start_date=start,
            end_date=end,
            limit=limit,
        )

    async def get_user_paths(
        self,
        starting_event: str,
        limit: int = 10,
        days: int = 30,
    ) -> list[UserPath]:
        """
        Get common user paths starting from an event.

        Uses the User Composition/Pathfinder approach to understand
        user journeys after a specific event.

        Args:
            starting_event: The event to start paths from
            limit: Maximum number of paths to return
            days: Number of days to analyze

        Returns:
            List of UserPath showing common journeys
        """
        self.log.info("Fetching user paths", starting_event=starting_event, limit=limit)

        start = datetime.utcnow() - timedelta(days=days)
        end = datetime.utcnow()

        # Use Event Segmentation with group by to approximate paths
        # Note: Full path analysis requires Amplitude's Journeys feature
        import json
        params = {
            "e": json.dumps({
                "event_type": starting_event,
                "group_by": [{"type": "event", "value": "event_type"}]
            }),
            "m": "uniques",
            "start": self._format_date(start),
            "end": self._format_date(end),
            "limit": limit,
        }

        data = await self._request("GET", "events/segmentation", params=params)
        if not data:
            return []

        # Parse paths from the response
        paths = []
        series_data = data.get("data", {}).get("series", [])
        series_labels = data.get("data", {}).get("seriesLabels", [])

        # Group by next event to find common paths
        for idx, label in enumerate(series_labels[:limit]):
            if idx < len(series_data):
                user_count = sum(v or 0 for v in series_data[idx])
                if user_count > 0:
                    paths.append(UserPath(
                        steps=[starting_event, label] if label != starting_event else [starting_event],
                        user_count=user_count,
                        conversion_rate=1.0,  # Would need funnel data for accurate rate
                        avg_time_between_steps=[0.0],  # Would need session data
                    ))

        paths.sort(key=lambda p: p.user_count, reverse=True)
        return paths[:limit]

    async def get_funnel(
        self,
        events: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        conversion_window_days: int = 14,
    ) -> list[FunnelStep]:
        """
        Get funnel conversion data for a sequence of events.

        Args:
            events: List of event names defining the funnel steps
            start_date: Start of date range (default: 30 days ago)
            end_date: End of date range (default: now)
            conversion_window_days: Window for conversion (default: 14 days)

        Returns:
            List of FunnelStep with conversion metrics
        """
        if not events:
            return []

        self.log.info("Fetching funnel data", events=events)

        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        # Build funnel events parameter
        import json
        funnel_events = [{"event_type": e} for e in events]

        params = {
            "e": json.dumps(funnel_events),
            "start": self._format_date(start),
            "end": self._format_date(end),
            "n": "any",  # Conversion mode
            "cs": str(conversion_window_days),  # Conversion window
        }

        data = await self._request("GET", "funnels", params=params)
        if not data:
            return []

        # Parse funnel response
        funnel_data = data.get("data", {})
        steps = []
        previous_count = None

        for idx, event_name in enumerate(events):
            step_key = f"step{idx + 1}"
            step_data = funnel_data.get(step_key, {})
            count = step_data.get("count", 0)

            if previous_count is None:
                conversion_rate = 100.0
                drop_off_rate = 0.0
            else:
                conversion_rate = (count / previous_count * 100) if previous_count > 0 else 0.0
                drop_off_rate = 100.0 - conversion_rate

            avg_time = step_data.get("avgTimeToConvert")

            steps.append(FunnelStep(
                event_name=event_name,
                count=count,
                conversion_rate=conversion_rate,
                drop_off_rate=drop_off_rate,
                avg_time_to_convert=avg_time,
            ))

            previous_count = count

        return steps

    async def get_retention(
        self,
        starting_event: str,
        returning_event: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        retention_days: int = 30,
    ) -> RetentionData | None:
        """
        Get retention data for users who did starting_event and returned.

        Args:
            starting_event: Event that starts the retention tracking
            returning_event: Event that counts as a return
            start_date: Start of date range (default: 60 days ago)
            end_date: End of date range (default: now)
            retention_days: Number of days to track retention

        Returns:
            RetentionData with cohort analysis or None on failure
        """
        self.log.info(
            "Fetching retention data",
            starting_event=starting_event,
            returning_event=returning_event,
        )

        start = start_date or (datetime.utcnow() - timedelta(days=60))
        end = end_date or datetime.utcnow()

        import json
        params = {
            "se": json.dumps({"event_type": starting_event}),
            "re": json.dumps({"event_type": returning_event}),
            "start": self._format_date(start),
            "end": self._format_date(end),
            "rm": "bracket",  # Retention mode
            "rb": str(retention_days),  # Retention brackets
        }

        data = await self._request("GET", "retention", params=params)
        if not data:
            return None

        # Parse retention response
        retention_data = data.get("data", {})
        series = retention_data.get("series", [])
        counts = retention_data.get("counts", [])

        if not series or not counts:
            return None

        # Calculate retention by day
        retention_by_day = {}
        total_cohort = sum(counts) if counts else 0

        for day_idx, day_series in enumerate(series):
            if day_idx < len(day_series) and counts[day_idx] > 0:
                retention_by_day[day_idx] = (day_series[day_idx] / counts[day_idx] * 100)

        avg_retention = sum(retention_by_day.values()) / max(len(retention_by_day), 1)

        return RetentionData(
            starting_event=starting_event,
            returning_event=returning_event,
            cohort_size=total_cohort,
            retention_by_day=retention_by_day,
            avg_retention=avg_retention,
            start_date=start,
            end_date=end,
        )

    async def get_event_properties(
        self,
        event_type: str,
    ) -> list[EventProperty]:
        """
        Get properties for a specific event type.

        Args:
            event_type: The event type to get properties for

        Returns:
            List of EventProperty with property metadata
        """
        import json
        params = {
            "e": json.dumps({"event_type": event_type}),
        }

        data = await self._request("GET", "events/properties", params=params)
        if not data:
            return []

        properties = []
        prop_data = data.get("data", [])

        for prop in prop_data:
            properties.append(EventProperty(
                property_name=prop.get("property_name", ""),
                property_type=prop.get("property_type", "string"),
                top_values=prop.get("top_values", []),
            ))

        return properties

    async def get_active_users(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """
        Get daily/weekly/monthly active user counts.

        Args:
            start_date: Start of date range (default: 30 days ago)
            end_date: End of date range (default: now)

        Returns:
            Dictionary with DAU, WAU, MAU data
        """
        start = start_date or (datetime.utcnow() - timedelta(days=30))
        end = end_date or datetime.utcnow()

        params = {
            "start": self._format_date(start),
            "end": self._format_date(end),
        }

        data = await self._request("GET", "users", params=params)
        if not data:
            return {}

        return {
            "dau": data.get("data", {}).get("series", []),
            "dates": data.get("data", {}).get("xvalues", []),
        }

    async def get_test_prioritization(
        self,
        limit: int = 20,
        days: int = 30,
    ) -> list[dict]:
        """
        Get prioritized list of features to test based on usage data.

        Combines event frequency, user count, and recency to prioritize
        which features should be tested first.

        Args:
            limit: Number of features to return
            days: Number of days to analyze

        Returns:
            List of dicts with event_type and priority_score
        """
        self.log.info("Calculating test prioritization", limit=limit)

        events = await self.get_top_events(limit=limit * 2, days=days)

        prioritized = []
        max_users = max(e.unique_users for e in events) if events else 1
        max_count = max(e.total_count for e in events) if events else 1

        for event in events:
            # Calculate priority score (0-100)
            # Weight: 60% unique users, 40% total count
            user_score = (event.unique_users / max_users) * 60
            count_score = (event.total_count / max_count) * 40
            priority_score = user_score + count_score

            prioritized.append({
                "event_type": event.event_type,
                "unique_users": event.unique_users,
                "total_count": event.total_count,
                "priority_score": round(priority_score, 2),
                "recommendation": self._get_priority_recommendation(priority_score),
            })

        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        return prioritized[:limit]

    def _get_priority_recommendation(self, score: float) -> str:
        """Get a test recommendation based on priority score."""
        if score >= 80:
            return "Critical - Test in every release"
        elif score >= 60:
            return "High - Test weekly"
        elif score >= 40:
            return "Medium - Test bi-weekly"
        elif score >= 20:
            return "Low - Test monthly"
        else:
            return "Minimal - Test quarterly"

    async def test_connection(self) -> bool:
        """
        Test if credentials are valid.

        Returns:
            True if connection is successful, False otherwise
        """
        self.log.info("Testing Amplitude connection")

        # Use a simple API call to verify credentials
        params = {
            "start": self._format_date(datetime.utcnow() - timedelta(days=1)),
            "end": self._format_date(datetime.utcnow()),
        }

        data = await self._request("GET", "users", params=params)

        if data is not None:
            self.log.info("Amplitude connection successful")
            return True
        else:
            self.log.error("Amplitude connection failed")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.http.aclose()
        self.log.info("Amplitude integration closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def create_amplitude_integration(
    api_key: str | None = None,
    secret_key: str | None = None,
) -> AmplitudeIntegration | None:
    """
    Factory function for AmplitudeIntegration.

    Args:
        api_key: Amplitude API Key (or from AMPLITUDE_API_KEY env var)
        secret_key: Amplitude Secret Key (or from AMPLITUDE_SECRET_KEY env var)

    Returns:
        AmplitudeIntegration instance or None if credentials missing
    """
    import os

    api_key = api_key or os.environ.get("AMPLITUDE_API_KEY")
    secret_key = secret_key or os.environ.get("AMPLITUDE_SECRET_KEY")

    if not api_key or not secret_key:
        logger.warning("Amplitude credentials not provided - integration disabled")
        return None

    return AmplitudeIntegration(api_key=api_key, secret_key=secret_key)
