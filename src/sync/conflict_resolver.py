"""Conflict resolution for test specification synchronization."""

from dataclasses import dataclass, field
from typing import Any

from .change_detector import Change, ChangeDetector
from .models import (
    ConflictResolutionStrategy,
    SyncConflict,
)


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    merged_spec: dict | None = None
    conflicts: list[SyncConflict] = field(default_factory=list)
    auto_resolved: int = 0
    manual_required: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "merged_spec": self.merged_spec,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "auto_resolved": self.auto_resolved,
            "manual_required": self.manual_required,
        }


class ConflictResolver:
    """Resolves conflicts between local and remote test specifications.

    Strategies:
    - KEEP_LOCAL: Local changes always win
    - KEEP_REMOTE: Remote changes always win
    - LATEST_WINS: Most recent timestamp wins
    - MERGE: Attempt automatic merge, flag conflicts
    - MANUAL: All conflicts require user resolution
    """

    def __init__(
        self,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE,
    ):
        """Initialize conflict resolver.

        Args:
            default_strategy: Default resolution strategy.
        """
        self.default_strategy = default_strategy
        self._detector = ChangeDetector()

    def detect_conflicts(
        self,
        base_spec: dict | None,
        local_spec: dict,
        remote_spec: dict,
        test_id: str,
    ) -> list[SyncConflict]:
        """Detect conflicts between local and remote changes.

        Three-way merge conflict detection:
        1. Find changes from base to local
        2. Find changes from base to remote
        3. Identify overlapping changes (conflicts)

        Args:
            base_spec: Common ancestor spec (last synced version).
            local_spec: Local version.
            remote_spec: Remote version.
            test_id: Test ID.

        Returns:
            List of detected conflicts.
        """
        conflicts: list[SyncConflict] = []

        # Get changes from base
        local_diff = self._detector.diff(base_spec, local_spec)
        remote_diff = self._detector.diff(base_spec, remote_spec)

        # Build path sets for conflict detection
        local_paths = {tuple(c.path) for c in local_diff.changes}
        remote_paths = {tuple(c.path) for c in remote_diff.changes}

        # Find overlapping paths (conflicts)
        conflicting_paths = local_paths & remote_paths

        # Also check for parent/child path conflicts
        for local_path in local_paths:
            for remote_path in remote_paths:
                if self._paths_conflict(list(local_path), list(remote_path)):
                    conflicting_paths.add(local_path)
                    conflicting_paths.add(remote_path)

        # Create conflict objects
        for path_tuple in conflicting_paths:
            path = list(path_tuple)
            local_change = next(
                (c for c in local_diff.changes if tuple(c.path) == path_tuple),
                None
            )
            remote_change = next(
                (c for c in remote_diff.changes if tuple(c.path) == path_tuple),
                None
            )

            if local_change or remote_change:
                conflicts.append(SyncConflict(
                    test_id=test_id,
                    path=path,
                    local_value=local_change.new_value if local_change else None,
                    remote_value=remote_change.new_value if remote_change else None,
                ))

        return conflicts

    def _paths_conflict(self, path1: list[str], path2: list[str]) -> bool:
        """Check if two paths conflict (one is parent of other).

        Args:
            path1: First path.
            path2: Second path.

        Returns:
            True if paths conflict.
        """
        if path1 == path2:
            return True

        min_len = min(len(path1), len(path2))
        return path1[:min_len] == path2[:min_len]

    def resolve(
        self,
        conflict: SyncConflict,
        strategy: ConflictResolutionStrategy | None = None,
        manual_value: Any = None,
    ) -> SyncConflict:
        """Resolve a single conflict.

        Args:
            conflict: The conflict to resolve.
            strategy: Resolution strategy (uses default if None).
            manual_value: Value to use for MANUAL resolution.

        Returns:
            Updated conflict with resolution.
        """
        strategy = strategy or self.default_strategy

        if strategy == ConflictResolutionStrategy.KEEP_LOCAL:
            conflict.resolved_value = conflict.local_value
        elif strategy == ConflictResolutionStrategy.KEEP_REMOTE:
            conflict.resolved_value = conflict.remote_value
        elif strategy == ConflictResolutionStrategy.LATEST_WINS:
            if conflict.local_timestamp and conflict.remote_timestamp:
                if conflict.local_timestamp >= conflict.remote_timestamp:
                    conflict.resolved_value = conflict.local_value
                else:
                    conflict.resolved_value = conflict.remote_value
            else:
                # Default to local if no timestamps
                conflict.resolved_value = conflict.local_value
        elif strategy == ConflictResolutionStrategy.MANUAL:
            if manual_value is not None:
                conflict.resolved_value = manual_value
            else:
                # Leave unresolved for manual handling
                return conflict
        elif strategy == ConflictResolutionStrategy.MERGE:
            # Try automatic merge
            merged = self._try_auto_merge(
                conflict.local_value,
                conflict.remote_value,
            )
            if merged is not None:
                conflict.resolved_value = merged
            else:
                # Can't auto-merge, needs manual resolution
                return conflict

        conflict.resolved = True
        conflict.resolution = strategy
        return conflict

    def _try_auto_merge(self, local_val: Any, remote_val: Any) -> Any | None:
        """Try to automatically merge two values.

        Args:
            local_val: Local value.
            remote_val: Remote value.

        Returns:
            Merged value or None if can't merge.
        """
        # Same value = no conflict
        if local_val == remote_val:
            return local_val

        # One is None = take the other
        if local_val is None:
            return remote_val
        if remote_val is None:
            return local_val

        # Try dict merge for non-overlapping keys
        if isinstance(local_val, dict) and isinstance(remote_val, dict):
            merged = dict(local_val)
            for key, value in remote_val.items():
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    # Conflicting key values - can't auto merge
                    return None
            return merged

        # Try list merge (append unique items)
        if isinstance(local_val, list) and isinstance(remote_val, list):
            # For steps, we can't auto-merge different lists
            return None

        # Can't auto-merge primitives with different values
        return None

    def merge_specs(
        self,
        base_spec: dict | None,
        local_spec: dict,
        remote_spec: dict,
        test_id: str,
        strategy: ConflictResolutionStrategy | None = None,
    ) -> MergeResult:
        """Perform a three-way merge of test specs.

        Args:
            base_spec: Common ancestor spec.
            local_spec: Local version.
            remote_spec: Remote version.
            test_id: Test ID.
            strategy: Resolution strategy for conflicts.

        Returns:
            MergeResult with merged spec and any unresolved conflicts.
        """
        strategy = strategy or self.default_strategy

        # Detect conflicts
        conflicts = self.detect_conflicts(base_spec, local_spec, remote_spec, test_id)

        if not conflicts:
            # No conflicts - merge all changes
            merged = self._merge_no_conflicts(base_spec, local_spec, remote_spec)
            return MergeResult(
                success=True,
                merged_spec=merged,
                auto_resolved=0,
                manual_required=0,
            )

        # Resolve conflicts
        resolved_conflicts: list[SyncConflict] = []
        unresolved_conflicts: list[SyncConflict] = []
        auto_resolved = 0

        for conflict in conflicts:
            resolved = self.resolve(conflict, strategy)
            if resolved.resolved:
                resolved_conflicts.append(resolved)
                auto_resolved += 1
            else:
                unresolved_conflicts.append(resolved)

        if unresolved_conflicts:
            # Can't complete merge - needs manual resolution
            return MergeResult(
                success=False,
                conflicts=unresolved_conflicts,
                auto_resolved=auto_resolved,
                manual_required=len(unresolved_conflicts),
            )

        # Apply resolved conflicts to create merged spec
        merged = self._apply_resolutions(
            base_spec,
            local_spec,
            remote_spec,
            resolved_conflicts,
        )

        return MergeResult(
            success=True,
            merged_spec=merged,
            conflicts=resolved_conflicts,
            auto_resolved=auto_resolved,
            manual_required=0,
        )

    def _merge_no_conflicts(
        self,
        base_spec: dict | None,
        local_spec: dict,
        remote_spec: dict,
    ) -> dict:
        """Merge specs when there are no conflicts.

        Args:
            base_spec: Common ancestor.
            local_spec: Local version.
            remote_spec: Remote version.

        Returns:
            Merged spec.
        """
        import copy

        # Start with remote (server is source of truth)
        merged = copy.deepcopy(remote_spec)

        # Apply local changes that don't overlap with remote
        local_diff = self._detector.diff(base_spec, local_spec)
        remote_diff = self._detector.diff(base_spec, remote_spec)

        remote_paths = {tuple(c.path) for c in remote_diff.changes}

        for change in local_diff.changes:
            if tuple(change.path) not in remote_paths:
                self._apply_change(merged, change)

        return merged

    def _apply_resolutions(
        self,
        base_spec: dict | None,
        local_spec: dict,
        remote_spec: dict,
        resolved_conflicts: list[SyncConflict],
    ) -> dict:
        """Apply conflict resolutions to create merged spec.

        Args:
            base_spec: Common ancestor.
            local_spec: Local version.
            remote_spec: Remote version.
            resolved_conflicts: Resolved conflicts.

        Returns:
            Merged spec with resolutions applied.
        """
        import copy

        # Start with base, apply non-conflicting changes, then resolutions
        merged = copy.deepcopy(base_spec) if base_spec else {}

        # Get all changes
        local_diff = self._detector.diff(base_spec, local_spec)
        remote_diff = self._detector.diff(base_spec, remote_spec)

        conflict_paths = {tuple(c.path) for c in resolved_conflicts}

        # Apply non-conflicting local changes
        for change in local_diff.changes:
            if tuple(change.path) not in conflict_paths:
                self._apply_change(merged, change)

        # Apply non-conflicting remote changes
        for change in remote_diff.changes:
            if tuple(change.path) not in conflict_paths:
                self._apply_change(merged, change)

        # Apply resolutions
        for conflict in resolved_conflicts:
            if conflict.resolved_value is not None:
                self._set_at_path(merged, conflict.path, conflict.resolved_value)

        return merged

    def _apply_change(self, spec: dict, change: Change) -> None:
        """Apply a change to a spec.

        Args:
            spec: Spec to modify.
            change: Change to apply.
        """
        if change.operation == "delete":
            self._delete_at_path(spec, change.path)
        else:
            self._set_at_path(spec, change.path, change.new_value)

    def _set_at_path(self, obj: dict, path: list[str], value: Any) -> None:
        """Set value at path in object.

        Args:
            obj: Object to modify.
            path: Path to set.
            value: Value to set.
        """
        if not path:
            return

        current = obj
        for key in path[:-1]:
            if isinstance(current, dict):
                if key not in current:
                    current[key] = {}
                current = current[key]
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx]
                except (ValueError, IndexError):
                    return
            else:
                return

        final_key = path[-1]
        if isinstance(current, dict):
            current[final_key] = value
        elif isinstance(current, list):
            try:
                idx = int(final_key)
                if idx < len(current):
                    current[idx] = value
                else:
                    current.append(value)
            except ValueError:
                pass

    def _delete_at_path(self, obj: dict, path: list[str]) -> None:
        """Delete value at path in object.

        Args:
            obj: Object to modify.
            path: Path to delete.
        """
        if not path:
            return

        current = obj
        for key in path[:-1]:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx]
                except (ValueError, IndexError):
                    return
            else:
                return

        final_key = path[-1]
        if isinstance(current, dict) and final_key in current:
            del current[final_key]
        elif isinstance(current, list):
            try:
                idx = int(final_key)
                if idx < len(current):
                    del current[idx]
            except ValueError:
                pass


def resolve_conflicts(
    conflicts: list[SyncConflict],
    strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS,
) -> list[SyncConflict]:
    """Convenience function to resolve multiple conflicts.

    Args:
        conflicts: List of conflicts.
        strategy: Resolution strategy.

    Returns:
        List of resolved conflicts.
    """
    resolver = ConflictResolver(default_strategy=strategy)
    return [resolver.resolve(c, strategy) for c in conflicts]
