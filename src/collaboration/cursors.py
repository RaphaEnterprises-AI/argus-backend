"""Cursor position tracking and rendering for real-time collaboration."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .models import (
    CURSOR_COLORS,
    BroadcastMessage,
    CursorPosition,
    SelectionRange,
)


@dataclass
class CursorState:
    """Complete cursor state for a user."""

    user_id: str
    user_name: str
    color: str
    position: CursorPosition | None = None
    selection: SelectionRange | None = None
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for transmission."""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "color": self.color,
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "element_id": self.position.element_id,
                "step_index": self.position.step_index,
                "field_name": self.position.field_name,
            } if self.position else None,
            "selection": {
                "start": self.selection.start,
                "end": self.selection.end,
                "element_id": self.selection.element_id,
            } if self.selection else None,
            "last_updated": self.last_updated.isoformat(),
        }


class CursorTracker:
    """Tracks cursor positions for multiple users in a test editor.

    Provides:
    - Per-user cursor position tracking
    - Selection range tracking
    - Throttled broadcast updates
    - Cursor interpolation for smooth rendering
    """

    # Minimum time between cursor updates (ms)
    THROTTLE_MS = 50
    # Maximum age before cursor is considered stale (ms)
    STALE_THRESHOLD_MS = 5000

    def __init__(
        self,
        test_id: str,
        broadcast_fn: Callable[[BroadcastMessage], None] | None = None,
    ):
        """Initialize cursor tracker.

        Args:
            test_id: The test being tracked.
            broadcast_fn: Optional callback for broadcasting cursor updates.
        """
        self.test_id = test_id
        self._broadcast_fn = broadcast_fn
        # Map of user_id -> CursorState
        self._cursors: dict[str, CursorState] = {}
        # Last broadcast time per user (for throttling)
        self._last_broadcast: dict[str, datetime] = {}

    def add_user(
        self,
        user_id: str,
        user_name: str,
        color: str | None = None,
    ) -> CursorState:
        """Add a user to cursor tracking.

        Args:
            user_id: User identifier.
            user_name: Display name.
            color: Optional cursor color (auto-assigned if not provided).

        Returns:
            The created CursorState.
        """
        if color is None:
            # Assign color based on user_id hash
            color_index = hash(user_id) % len(CURSOR_COLORS)
            color = CURSOR_COLORS[color_index]

        cursor_state = CursorState(
            user_id=user_id,
            user_name=user_name,
            color=color,
        )
        self._cursors[user_id] = cursor_state
        return cursor_state

    def remove_user(self, user_id: str) -> None:
        """Remove a user from cursor tracking.

        Args:
            user_id: User identifier.
        """
        self._cursors.pop(user_id, None)
        self._last_broadcast.pop(user_id, None)

    def update_position(
        self,
        user_id: str,
        position: CursorPosition,
        broadcast: bool = True,
    ) -> bool:
        """Update a user's cursor position.

        Args:
            user_id: User identifier.
            position: New cursor position.
            broadcast: Whether to broadcast the update.

        Returns:
            True if update was broadcast, False if throttled.
        """
        if user_id not in self._cursors:
            return False

        cursor_state = self._cursors[user_id]
        cursor_state.position = position
        cursor_state.last_updated = datetime.now(UTC)

        if broadcast:
            return self._maybe_broadcast(user_id, "cursor_move")
        return False

    def update_selection(
        self,
        user_id: str,
        selection: SelectionRange,
        broadcast: bool = True,
    ) -> bool:
        """Update a user's text selection.

        Args:
            user_id: User identifier.
            selection: New selection range.
            broadcast: Whether to broadcast the update.

        Returns:
            True if update was broadcast, False if throttled.
        """
        if user_id not in self._cursors:
            return False

        cursor_state = self._cursors[user_id]
        cursor_state.selection = selection
        cursor_state.last_updated = datetime.now(UTC)

        if broadcast:
            return self._maybe_broadcast(user_id, "cursor_select")
        return False

    def clear_selection(self, user_id: str) -> None:
        """Clear a user's selection.

        Args:
            user_id: User identifier.
        """
        if user_id in self._cursors:
            self._cursors[user_id].selection = None

    def get_cursor(self, user_id: str) -> CursorState | None:
        """Get a user's cursor state.

        Args:
            user_id: User identifier.

        Returns:
            CursorState if found, None otherwise.
        """
        return self._cursors.get(user_id)

    def get_all_cursors(self) -> list[CursorState]:
        """Get all active cursor states.

        Returns:
            List of CursorState objects.
        """
        return list(self._cursors.values())

    def get_active_cursors(self) -> list[CursorState]:
        """Get non-stale cursor states.

        Returns:
            List of recently-updated CursorState objects.
        """
        now = datetime.now(UTC)
        threshold_seconds = self.STALE_THRESHOLD_MS / 1000

        return [
            cursor for cursor in self._cursors.values()
            if (now - cursor.last_updated).total_seconds() < threshold_seconds
        ]

    def get_cursors_at_element(self, element_id: str) -> list[CursorState]:
        """Get all cursors at a specific element.

        Args:
            element_id: Element identifier.

        Returns:
            List of CursorState objects at that element.
        """
        return [
            cursor for cursor in self._cursors.values()
            if cursor.position and cursor.position.element_id == element_id
        ]

    def get_cursors_at_step(self, step_index: int) -> list[CursorState]:
        """Get all cursors at a specific test step.

        Args:
            step_index: Step index.

        Returns:
            List of CursorState objects at that step.
        """
        return [
            cursor for cursor in self._cursors.values()
            if cursor.position and cursor.position.step_index == step_index
        ]

    def get_full_state(self) -> dict:
        """Get complete cursor state for initial sync.

        Returns:
            Dictionary with all cursor states.
        """
        return {
            "test_id": self.test_id,
            "cursors": [cursor.to_dict() for cursor in self._cursors.values()],
        }

    def _maybe_broadcast(self, user_id: str, event_type: str) -> bool:
        """Broadcast update if not throttled.

        Args:
            user_id: User identifier.
            event_type: Type of cursor event.

        Returns:
            True if broadcast was sent.
        """
        now = datetime.now(UTC)
        last = self._last_broadcast.get(user_id)

        if last:
            elapsed_ms = (now - last).total_seconds() * 1000
            if elapsed_ms < self.THROTTLE_MS:
                return False

        self._last_broadcast[user_id] = now

        if self._broadcast_fn and user_id in self._cursors:
            cursor_state = self._cursors[user_id]
            message = BroadcastMessage(
                channel=f"test:{self.test_id}",
                event=event_type,
                payload=cursor_state.to_dict(),
            )
            try:
                self._broadcast_fn(message)
            except Exception:
                pass

        return True


def interpolate_cursor_position(
    start: CursorPosition,
    end: CursorPosition,
    progress: float,
) -> CursorPosition:
    """Interpolate between two cursor positions for smooth animation.

    Args:
        start: Starting position.
        end: Ending position.
        progress: Interpolation progress (0.0 to 1.0).

    Returns:
        Interpolated CursorPosition.
    """
    progress = max(0.0, min(1.0, progress))

    return CursorPosition(
        x=start.x + (end.x - start.x) * progress,
        y=start.y + (end.y - start.y) * progress,
        element_id=end.element_id if progress >= 0.5 else start.element_id,
        step_index=end.step_index if progress >= 0.5 else start.step_index,
        field_name=end.field_name if progress >= 0.5 else start.field_name,
    )


def generate_cursor_label(user_name: str, max_length: int = 12) -> str:
    """Generate a display label for cursor tooltip.

    Args:
        user_name: Full user name.
        max_length: Maximum label length.

    Returns:
        Truncated label suitable for cursor display.
    """
    if len(user_name) <= max_length:
        return user_name

    # Try first name only
    parts = user_name.split()
    if parts and len(parts[0]) <= max_length:
        return parts[0]

    # Truncate with ellipsis (ellipsis is 3 chars)
    return user_name[:max_length - 3] + "..."
