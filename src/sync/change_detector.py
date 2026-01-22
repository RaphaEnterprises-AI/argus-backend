"""Change detection for test specifications."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .models import SyncEvent, SyncEventType, SyncSource


@dataclass
class Change:
    """A detected change in a test spec."""

    path: list[str]
    operation: str  # "add", "update", "delete"
    old_value: Any = None
    new_value: Any = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "operation": self.operation,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass
class DiffResult:
    """Result of diffing two test specs."""

    has_changes: bool = False
    changes: list[Change] = field(default_factory=list)
    old_checksum: str | None = None
    new_checksum: str | None = None

    @property
    def change_count(self) -> int:
        """Get number of changes."""
        return len(self.changes)

    def get_changes_by_path(self, path_prefix: list[str]) -> list[Change]:
        """Get changes matching a path prefix."""
        return [
            c for c in self.changes
            if c.path[:len(path_prefix)] == path_prefix
        ]


class ChangeDetector:
    """Detects changes between test specification versions.

    Provides:
    - Deep diff between test specs
    - Change categorization (steps, assertions, metadata)
    - Checksum calculation for validation
    - Event generation from changes
    """

    @staticmethod
    def calculate_checksum(spec: dict) -> str:
        """Calculate a deterministic checksum for a spec.

        Args:
            spec: Test specification dictionary.

        Returns:
            SHA-256 hex digest.
        """
        # Normalize the spec for consistent hashing
        normalized = json.dumps(spec, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def diff(
        self,
        old_spec: dict | None,
        new_spec: dict | None,
    ) -> DiffResult:
        """Compute differences between two specs.

        Args:
            old_spec: Previous spec (None for new test).
            new_spec: Current spec (None for deleted test).

        Returns:
            DiffResult with all detected changes.
        """
        result = DiffResult()

        if old_spec:
            result.old_checksum = self.calculate_checksum(old_spec)
        if new_spec:
            result.new_checksum = self.calculate_checksum(new_spec)

        # Handle creation/deletion
        if old_spec is None and new_spec is not None:
            result.has_changes = True
            result.changes.append(Change(
                path=[],
                operation="add",
                new_value=new_spec,
            ))
            return result

        if old_spec is not None and new_spec is None:
            result.has_changes = True
            result.changes.append(Change(
                path=[],
                operation="delete",
                old_value=old_spec,
            ))
            return result

        if old_spec is None and new_spec is None:
            return result

        # Deep diff
        changes = self._deep_diff(old_spec, new_spec, [])
        result.changes = changes
        result.has_changes = len(changes) > 0

        return result

    def _deep_diff(
        self,
        old_val: Any,
        new_val: Any,
        path: list[str],
    ) -> list[Change]:
        """Recursively diff two values.

        Args:
            old_val: Old value.
            new_val: New value.
            path: Current path.

        Returns:
            List of changes.
        """
        changes: list[Change] = []

        # Skip metadata fields that change frequently
        if path and path[-1] in ("updated_at", "version"):
            return changes

        # Same value
        if old_val == new_val:
            return changes

        # Type change
        if type(old_val) != type(new_val):
            changes.append(Change(
                path=path.copy(),
                operation="update",
                old_value=old_val,
                new_value=new_val,
            ))
            return changes

        # Dict diff
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            all_keys = set(old_val.keys()) | set(new_val.keys())
            for key in all_keys:
                new_path = path + [key]
                if key not in old_val:
                    changes.append(Change(
                        path=new_path,
                        operation="add",
                        new_value=new_val[key],
                    ))
                elif key not in new_val:
                    changes.append(Change(
                        path=new_path,
                        operation="delete",
                        old_value=old_val[key],
                    ))
                else:
                    changes.extend(
                        self._deep_diff(old_val[key], new_val[key], new_path)
                    )
            return changes

        # List diff
        if isinstance(old_val, list) and isinstance(new_val, list):
            return self._diff_lists(old_val, new_val, path)

        # Primitive change
        changes.append(Change(
            path=path.copy(),
            operation="update",
            old_value=old_val,
            new_value=new_val,
        ))
        return changes

    def _diff_lists(
        self,
        old_list: list,
        new_list: list,
        path: list[str],
    ) -> list[Change]:
        """Diff two lists.

        Args:
            old_list: Old list.
            new_list: New list.
            path: Current path.

        Returns:
            List of changes.
        """
        changes: list[Change] = []
        max_len = max(len(old_list), len(new_list))

        for i in range(max_len):
            new_path = path + [str(i)]
            if i >= len(old_list):
                # Added item
                changes.append(Change(
                    path=new_path,
                    operation="add",
                    new_value=new_list[i],
                ))
            elif i >= len(new_list):
                # Removed item
                changes.append(Change(
                    path=new_path,
                    operation="delete",
                    old_value=old_list[i],
                ))
            else:
                # Compare items
                changes.extend(
                    self._deep_diff(old_list[i], new_list[i], new_path)
                )

        return changes

    def generate_events(
        self,
        diff_result: DiffResult,
        project_id: str,
        test_id: str,
        source: SyncSource = SyncSource.IDE,
        local_version: int = 0,
    ) -> list[SyncEvent]:
        """Generate sync events from a diff result.

        Args:
            diff_result: Result from diff operation.
            project_id: Project ID.
            test_id: Test ID.
            source: Source of changes.
            local_version: Current local version.

        Returns:
            List of sync events.
        """
        if not diff_result.has_changes:
            return []

        events: list[SyncEvent] = []

        for change in diff_result.changes:
            event_type = self._get_event_type(change)
            events.append(SyncEvent(
                type=event_type,
                source=source,
                project_id=project_id,
                test_id=test_id,
                path=change.path,
                content={"new_value": change.new_value} if change.new_value else None,
                previous_content={"old_value": change.old_value} if change.old_value else None,
                local_version=local_version,
                checksum=diff_result.new_checksum,
            ))

        return events

    def _get_event_type(self, change: Change) -> SyncEventType:
        """Get event type from a change.

        Args:
            change: The change.

        Returns:
            Appropriate SyncEventType.
        """
        path = change.path

        # Test level changes
        if len(path) == 0:
            if change.operation == "add":
                return SyncEventType.TEST_CREATED
            elif change.operation == "delete":
                return SyncEventType.TEST_DELETED
            return SyncEventType.TEST_UPDATED

        # Step changes
        if path[0] == "steps":
            if change.operation == "add":
                return SyncEventType.STEP_ADDED
            elif change.operation == "delete":
                return SyncEventType.STEP_REMOVED
            return SyncEventType.STEP_UPDATED

        # Assertion changes
        if path[0] == "assertions":
            return SyncEventType.ASSERTIONS_UPDATED

        # Metadata changes
        if path[0] == "metadata":
            return SyncEventType.METADATA_UPDATED

        # Default to test updated
        return SyncEventType.TEST_UPDATED

    def detect_step_reorder(
        self,
        old_steps: list[dict],
        new_steps: list[dict],
    ) -> bool:
        """Detect if steps were reordered (vs added/removed).

        Args:
            old_steps: Old step list.
            new_steps: New step list.

        Returns:
            True if this is a reorder operation.
        """
        if len(old_steps) != len(new_steps):
            return False

        # Check if all steps exist in both lists (just different order)
        old_ids = {s.get("id") for s in old_steps if "id" in s}
        new_ids = {s.get("id") for s in new_steps if "id" in s}

        if old_ids and new_ids and old_ids == new_ids:
            # Same IDs, different order
            old_order = [s.get("id") for s in old_steps]
            new_order = [s.get("id") for s in new_steps]
            return old_order != new_order

        # Compare by content hash
        def hash_step(step: dict) -> str:
            return hashlib.md5(
                json.dumps(step, sort_keys=True).encode(),
                usedforsecurity=False
            ).hexdigest()

        old_hashes = set(hash_step(s) for s in old_steps)
        new_hashes = set(hash_step(s) for s in new_steps)

        return old_hashes == new_hashes


def diff_specs(
    old_spec: dict | None,
    new_spec: dict | None,
) -> DiffResult:
    """Convenience function to diff two specs.

    Args:
        old_spec: Previous spec.
        new_spec: Current spec.

    Returns:
        DiffResult with changes.
    """
    return ChangeDetector().diff(old_spec, new_spec)


def calculate_checksum(spec: dict) -> str:
    """Convenience function to calculate spec checksum.

    Args:
        spec: Test specification.

    Returns:
        Checksum string.
    """
    return ChangeDetector.calculate_checksum(spec)
