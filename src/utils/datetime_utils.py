"""Enterprise-grade datetime utilities for null-safe data handling.

This module provides defense-in-depth for datetime fields coming from external
data sources (databases, APIs) where NULL values may exist despite schema expectations.

Usage:
    from src.utils.datetime_utils import safe_datetime, SafeDatetimeStr

    # In Pydantic models - automatic validation
    class MyResponse(BaseModel):
        created_at: SafeDatetimeStr
        updated_at: SafeDatetimeStr | None = None

    # In direct code - manual transformation
    created_at = safe_datetime(row.get("created_at"))

Monitoring:
    Null encounters are logged at WARNING level with context for tracking
    data quality issues upstream (e.g., missing database defaults).
"""

from datetime import UTC, datetime
from functools import lru_cache
from typing import Annotated, Any

import structlog
from pydantic import AfterValidator, BeforeValidator

logger = structlog.get_logger(__name__)

# Metrics counter for monitoring (can be wired to Prometheus/Datadog)
_null_datetime_encounters: dict[str, int] = {}


def _track_null_encounter(context: str | None = None) -> None:
    """Track null datetime encounters for monitoring and alerting.

    In production, this would emit to your metrics system (Prometheus, Datadog, etc.)
    to alert on data quality issues.
    """
    key = context or "unknown"
    _null_datetime_encounters[key] = _null_datetime_encounters.get(key, 0) + 1

    # Log at warning level - these indicate upstream data quality issues
    logger.warning(
        "Null datetime encountered - using fallback",
        context=context,
        total_encounters=_null_datetime_encounters[key],
        hint="Consider adding NOT NULL DEFAULT now() constraint to database column",
    )


def safe_datetime(
    value: str | datetime | None,
    *,
    fallback: datetime | None = None,
    context: str | None = None,
) -> str:
    """Safely convert a datetime value to ISO format string with fallback.

    Args:
        value: The datetime value (string, datetime object, or None)
        fallback: Custom fallback datetime. Defaults to current UTC time.
        context: Context string for monitoring (e.g., "tests.created_at")

    Returns:
        ISO format datetime string, never None

    Examples:
        >>> safe_datetime("2024-01-15T10:30:00Z")
        '2024-01-15T10:30:00Z'

        >>> safe_datetime(None)  # Returns current time
        '2024-01-15T12:00:00+00:00'

        >>> safe_datetime(None, context="test_runs.created_at")  # Logs warning
        '2024-01-15T12:00:00+00:00'
    """
    if value is None:
        _track_null_encounter(context)
        fallback_dt = fallback or datetime.now(UTC)
        return fallback_dt.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, str):
        # Validate it's a parseable datetime string
        if value.strip():
            return value
        # Empty string treated as null
        _track_null_encounter(context)
        fallback_dt = fallback or datetime.now(UTC)
        return fallback_dt.isoformat()

    # Unexpected type - log and fallback
    logger.error(
        "Unexpected datetime type",
        value_type=type(value).__name__,
        context=context,
    )
    _track_null_encounter(context)
    fallback_dt = fallback or datetime.now(UTC)
    return fallback_dt.isoformat()


def safe_datetime_optional(
    value: str | datetime | None,
    *,
    context: str | None = None,
) -> str | None:
    """Safely convert a datetime value, returning None for missing values.

    Use this for genuinely optional datetime fields (e.g., completed_at, deleted_at)
    where None is a valid business state, not a data quality issue.

    Args:
        value: The datetime value (string, datetime object, or None)
        context: Context string for logging

    Returns:
        ISO format datetime string or None
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, str):
        if value.strip():
            return value
        return None

    logger.error(
        "Unexpected datetime type in optional field",
        value_type=type(value).__name__,
        context=context,
    )
    return None


# =============================================================================
# Pydantic Validators for Response Models
# =============================================================================

def _validate_safe_datetime(value: Any) -> str:
    """Pydantic validator that ensures datetime is never None."""
    return safe_datetime(value, context="pydantic_model")


def _validate_safe_datetime_optional(value: Any) -> str | None:
    """Pydantic validator for optional datetime fields."""
    return safe_datetime_optional(value, context="pydantic_model")


def _coerce_datetime_before(value: Any) -> Any:
    """Before validator to handle various datetime input formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# Type aliases for use in Pydantic models
SafeDatetimeStr = Annotated[
    str,
    BeforeValidator(_coerce_datetime_before),
    AfterValidator(_validate_safe_datetime),
]
"""A datetime string that is guaranteed to be non-null.

Usage in Pydantic models:
    class TestResponse(BaseModel):
        created_at: SafeDatetimeStr  # Never None, uses current time as fallback
"""

OptionalDatetimeStr = Annotated[
    str | None,
    BeforeValidator(_coerce_datetime_before),
    AfterValidator(_validate_safe_datetime_optional),
]
"""An optional datetime string for genuinely optional fields.

Usage in Pydantic models:
    class TestResponse(BaseModel):
        completed_at: OptionalDatetimeStr  # Can be None (valid business state)
"""


# =============================================================================
# Monitoring & Diagnostics
# =============================================================================

def get_null_datetime_stats() -> dict[str, int]:
    """Get statistics on null datetime encounters for monitoring dashboards.

    Returns:
        Dict mapping context strings to encounter counts

    Example:
        >>> get_null_datetime_stats()
        {'tests.created_at': 15, 'conversations.updated_at': 3}
    """
    return dict(_null_datetime_encounters)


def reset_null_datetime_stats() -> None:
    """Reset null datetime statistics (useful for testing)."""
    _null_datetime_encounters.clear()


# =============================================================================
# Database Row Transformer
# =============================================================================

def transform_row_datetimes(
    row: dict[str, Any],
    required_fields: list[str] | None = None,
    optional_fields: list[str] | None = None,
    context_prefix: str = "",
) -> dict[str, Any]:
    """Transform all datetime fields in a database row with proper null handling.

    This is useful for batch-processing database results before converting
    to Pydantic models.

    Args:
        row: Database row as dictionary
        required_fields: Fields that must have values (use fallback if null)
        optional_fields: Fields that can be None
        context_prefix: Prefix for monitoring context (e.g., "tests")

    Returns:
        New dictionary with transformed datetime fields

    Example:
        >>> row = {"id": "123", "created_at": None, "deleted_at": None}
        >>> transform_row_datetimes(
        ...     row,
        ...     required_fields=["created_at"],
        ...     optional_fields=["deleted_at"],
        ...     context_prefix="tests"
        ... )
        {'id': '123', 'created_at': '2024-01-15T12:00:00+00:00', 'deleted_at': None}
    """
    result = dict(row)
    required_fields = required_fields or []
    optional_fields = optional_fields or []

    for field in required_fields:
        if field in result:
            context = f"{context_prefix}.{field}" if context_prefix else field
            result[field] = safe_datetime(result[field], context=context)

    for field in optional_fields:
        if field in result:
            context = f"{context_prefix}.{field}" if context_prefix else field
            result[field] = safe_datetime_optional(result[field], context=context)

    return result
