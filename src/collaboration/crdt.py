"""CRDT (Conflict-free Replicated Data Type) implementation for collaborative editing.

Implements Last-Write-Wins (LWW) Register and Operation-based CRDT for JSON documents.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4
import copy
import json


@dataclass
class VectorClock:
    """Vector clock for tracking causality across nodes."""

    clocks: dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str) -> "VectorClock":
        """Increment clock for a node.

        Args:
            node_id: The node to increment.

        Returns:
            New VectorClock with incremented value.
        """
        new_clocks = self.clocks.copy()
        new_clocks[node_id] = new_clocks.get(node_id, 0) + 1
        return VectorClock(clocks=new_clocks)

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge two vector clocks (pointwise maximum).

        Args:
            other: The other vector clock.

        Returns:
            Merged VectorClock.
        """
        merged = self.clocks.copy()
        for node_id, timestamp in other.clocks.items():
            merged[node_id] = max(merged.get(node_id, 0), timestamp)
        return VectorClock(clocks=merged)

    def is_concurrent(self, other: "VectorClock") -> bool:
        """Check if two clocks are concurrent (neither happened before the other).

        Args:
            other: The other vector clock.

        Returns:
            True if clocks are concurrent.
        """
        return not self.happened_before(other) and not other.happened_before(self)

    def happened_before(self, other: "VectorClock") -> bool:
        """Check if this clock happened before another.

        Args:
            other: The other vector clock.

        Returns:
            True if this clock happened before.
        """
        at_least_one_less = False
        for node_id in set(self.clocks.keys()) | set(other.clocks.keys()):
            self_time = self.clocks.get(node_id, 0)
            other_time = other.clocks.get(node_id, 0)
            if self_time > other_time:
                return False
            if self_time < other_time:
                at_least_one_less = True
        return at_least_one_less

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return self.clocks.copy()

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "VectorClock":
        """Create from dictionary."""
        return cls(clocks=data.copy())


@dataclass
class CRDTOperation:
    """A single CRDT operation."""

    id: str = field(default_factory=lambda: str(uuid4()))
    node_id: str = ""  # ID of the node that created this operation
    operation: str = "set"  # "set", "delete", "insert", "move"
    path: list[str] = field(default_factory=list)  # JSON path
    value: Any = None
    previous_value: Any = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vector_clock: VectorClock = field(default_factory=VectorClock)

    def to_dict(self) -> dict:
        """Convert to dictionary for transmission."""
        return {
            "id": self.id,
            "node_id": self.node_id,
            "operation": self.operation,
            "path": self.path,
            "value": self.value,
            "previous_value": self.previous_value,
            "timestamp": self.timestamp.isoformat(),
            "vector_clock": self.vector_clock.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CRDTOperation":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            node_id=data.get("node_id", ""),
            operation=data.get("operation", "set"),
            path=data.get("path", []),
            value=data.get("value"),
            previous_value=data.get("previous_value"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(timezone.utc),
            vector_clock=VectorClock.from_dict(data.get("vector_clock", {})),
        )


class LWWRegister:
    """Last-Write-Wins Register CRDT.

    Simple CRDT where the latest write (by timestamp) wins.
    """

    def __init__(self, node_id: str, initial_value: Any = None):
        """Initialize LWW Register.

        Args:
            node_id: Unique identifier for this node.
            initial_value: Optional initial value.
        """
        self.node_id = node_id
        self._value = initial_value
        self._timestamp = datetime.now(timezone.utc)
        self._vector_clock = VectorClock()

    @property
    def value(self) -> Any:
        """Get current value."""
        return self._value

    @property
    def timestamp(self) -> datetime:
        """Get last update timestamp."""
        return self._timestamp

    def set(self, value: Any) -> CRDTOperation:
        """Set value (local operation).

        Args:
            value: New value.

        Returns:
            The operation that was performed.
        """
        previous_value = self._value
        self._value = value
        self._timestamp = datetime.now(timezone.utc)
        self._vector_clock = self._vector_clock.increment(self.node_id)

        return CRDTOperation(
            node_id=self.node_id,
            operation="set",
            value=value,
            previous_value=previous_value,
            timestamp=self._timestamp,
            vector_clock=self._vector_clock,
        )

    def apply(self, operation: CRDTOperation) -> bool:
        """Apply a remote operation.

        Args:
            operation: The operation to apply.

        Returns:
            True if the operation was applied (newer timestamp wins).
        """
        if operation.timestamp > self._timestamp:
            self._value = operation.value
            self._timestamp = operation.timestamp
            self._vector_clock = self._vector_clock.merge(operation.vector_clock)
            return True
        return False

    def merge(self, other: "LWWRegister") -> None:
        """Merge with another register.

        Args:
            other: The other register.
        """
        if other.timestamp > self._timestamp:
            self._value = other.value
            self._timestamp = other.timestamp
        self._vector_clock = self._vector_clock.merge(other._vector_clock)


class CRDTDocument:
    """CRDT-based JSON document for collaborative editing.

    Supports:
    - Set operations at any JSON path
    - Delete operations
    - Array insert/remove operations
    - Automatic conflict resolution
    """

    def __init__(self, node_id: str, initial_doc: Optional[dict] = None):
        """Initialize CRDT document.

        Args:
            node_id: Unique identifier for this node.
            initial_doc: Optional initial document state.
        """
        self.node_id = node_id
        self._doc = initial_doc or {}
        self._vector_clock = VectorClock()
        self._operations: list[CRDTOperation] = []
        self._pending_ops: list[CRDTOperation] = []

    @property
    def document(self) -> dict:
        """Get current document state."""
        return copy.deepcopy(self._doc)

    @property
    def vector_clock(self) -> VectorClock:
        """Get current vector clock."""
        return self._vector_clock

    def get(self, path: list[str]) -> Any:
        """Get value at a JSON path.

        Args:
            path: List of keys/indices to traverse.

        Returns:
            Value at path, or None if not found.
        """
        current = self._doc
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current

    def set(self, path: list[str], value: Any) -> CRDTOperation:
        """Set value at a JSON path.

        Args:
            path: List of keys to traverse.
            value: Value to set.

        Returns:
            The operation that was performed.
        """
        previous_value = self.get(path)
        self._set_at_path(path, value)
        self._vector_clock = self._vector_clock.increment(self.node_id)

        operation = CRDTOperation(
            node_id=self.node_id,
            operation="set",
            path=path,
            value=value,
            previous_value=previous_value,
            vector_clock=self._vector_clock,
        )
        self._operations.append(operation)
        return operation

    def delete(self, path: list[str]) -> CRDTOperation:
        """Delete value at a JSON path.

        Args:
            path: List of keys to traverse.

        Returns:
            The operation that was performed.
        """
        previous_value = self.get(path)
        self._delete_at_path(path)
        self._vector_clock = self._vector_clock.increment(self.node_id)

        operation = CRDTOperation(
            node_id=self.node_id,
            operation="delete",
            path=path,
            previous_value=previous_value,
            vector_clock=self._vector_clock,
        )
        self._operations.append(operation)
        return operation

    def insert(self, path: list[str], index: int, value: Any) -> CRDTOperation:
        """Insert value into an array at path.

        Args:
            path: Path to the array.
            index: Index to insert at.
            value: Value to insert.

        Returns:
            The operation that was performed.
        """
        arr = self.get(path)
        if not isinstance(arr, list):
            arr = []
            self._set_at_path(path, arr)

        arr.insert(index, value)
        self._vector_clock = self._vector_clock.increment(self.node_id)

        operation = CRDTOperation(
            node_id=self.node_id,
            operation="insert",
            path=path + [str(index)],
            value=value,
            vector_clock=self._vector_clock,
        )
        self._operations.append(operation)
        return operation

    def apply(self, operation: CRDTOperation) -> bool:
        """Apply a remote operation.

        Args:
            operation: The operation to apply.

        Returns:
            True if operation was applied.
        """
        # Check if operation was already applied
        if any(op.id == operation.id for op in self._operations):
            return False

        # Check causality - operation should happen after all we've seen
        if not self._can_apply(operation):
            self._pending_ops.append(operation)
            return False

        self._apply_operation(operation)
        self._operations.append(operation)
        self._vector_clock = self._vector_clock.merge(operation.vector_clock)

        # Try to apply any pending operations
        self._apply_pending()
        return True

    def _apply_operation(self, operation: CRDTOperation) -> None:
        """Actually apply an operation to the document."""
        if operation.operation == "set":
            self._set_at_path(operation.path, operation.value)
        elif operation.operation == "delete":
            self._delete_at_path(operation.path)
        elif operation.operation == "insert":
            # Extract array path and index from operation path
            if operation.path:
                arr_path = operation.path[:-1]
                try:
                    index = int(operation.path[-1])
                except ValueError:
                    return
                arr = self.get(arr_path)
                if isinstance(arr, list):
                    arr.insert(index, operation.value)

    def _can_apply(self, operation: CRDTOperation) -> bool:
        """Check if operation can be applied based on causality."""
        # Simple check: operation's clock should not be before ours
        return not operation.vector_clock.happened_before(self._vector_clock)

    def _apply_pending(self) -> None:
        """Try to apply pending operations."""
        changed = True
        while changed:
            changed = False
            remaining = []
            for op in self._pending_ops:
                if self._can_apply(op):
                    self._apply_operation(op)
                    self._operations.append(op)
                    self._vector_clock = self._vector_clock.merge(op.vector_clock)
                    changed = True
                else:
                    remaining.append(op)
            self._pending_ops = remaining

    def _set_at_path(self, path: list[str], value: Any) -> None:
        """Set value at path in document."""
        if not path:
            return

        current = self._doc
        for key in path[:-1]:
            if isinstance(current, list):
                try:
                    idx = int(key)
                    current = current[idx]
                except (ValueError, IndexError):
                    return
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

        # Handle the final key
        if isinstance(current, list):
            try:
                idx = int(path[-1])
                current[idx] = value
            except (ValueError, IndexError):
                pass
        else:
            current[path[-1]] = value

    def _delete_at_path(self, path: list[str]) -> None:
        """Delete value at path in document."""
        if not path:
            return

        current = self._doc
        for key in path[:-1]:
            if key not in current:
                return
            current = current[key]

        if isinstance(current, dict) and path[-1] in current:
            del current[path[-1]]
        elif isinstance(current, list):
            try:
                idx = int(path[-1])
                del current[idx]
            except (ValueError, IndexError):
                pass

    def get_operations_since(self, vector_clock: VectorClock) -> list[CRDTOperation]:
        """Get operations that happened after a given vector clock.

        Args:
            vector_clock: The reference clock.

        Returns:
            List of operations after the clock.
        """
        return [
            op for op in self._operations
            if not op.vector_clock.happened_before(vector_clock)
            and op.vector_clock.to_dict() != vector_clock.to_dict()
        ]

    def to_dict(self) -> dict:
        """Get full state for serialization."""
        return {
            "node_id": self.node_id,
            "document": self._doc,
            "vector_clock": self._vector_clock.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CRDTDocument":
        """Create from serialized state."""
        doc = cls(
            node_id=data.get("node_id", str(uuid4())),
            initial_doc=data.get("document", {}),
        )
        doc._vector_clock = VectorClock.from_dict(data.get("vector_clock", {}))
        return doc


class TestSpecCRDT:
    """Specialized CRDT for test specification documents.

    Provides higher-level operations for test editing:
    - Add/update/delete test steps
    - Update test metadata
    - Manage assertions
    """

    def __init__(self, node_id: str, test_spec: Optional[dict] = None):
        """Initialize test spec CRDT.

        Args:
            node_id: Unique identifier for this node.
            test_spec: Optional initial test specification.
        """
        self._crdt = CRDTDocument(
            node_id=node_id,
            initial_doc=test_spec or {
                "id": str(uuid4()),
                "name": "",
                "description": "",
                "steps": [],
                "assertions": [],
                "metadata": {},
            },
        )

    @property
    def test_spec(self) -> dict:
        """Get current test specification."""
        return self._crdt.document

    @property
    def node_id(self) -> str:
        """Get node ID."""
        return self._crdt.node_id

    def set_name(self, name: str) -> CRDTOperation:
        """Set test name."""
        return self._crdt.set(["name"], name)

    def set_description(self, description: str) -> CRDTOperation:
        """Set test description."""
        return self._crdt.set(["description"], description)

    def add_step(self, step: dict, index: Optional[int] = None) -> CRDTOperation:
        """Add a test step.

        Args:
            step: Step definition.
            index: Optional index to insert at (appends if None).

        Returns:
            The operation performed.
        """
        steps = self._crdt.get(["steps"]) or []
        if index is None:
            index = len(steps)
        return self._crdt.insert(["steps"], index, step)

    def update_step(self, index: int, step: dict) -> CRDTOperation:
        """Update a test step.

        Args:
            index: Step index.
            step: New step definition.

        Returns:
            The operation performed.
        """
        return self._crdt.set(["steps", str(index)], step)

    def delete_step(self, index: int) -> CRDTOperation:
        """Delete a test step.

        Args:
            index: Step index.

        Returns:
            The operation performed.
        """
        return self._crdt.delete(["steps", str(index)])

    def add_assertion(self, assertion: dict) -> CRDTOperation:
        """Add an assertion.

        Args:
            assertion: Assertion definition.

        Returns:
            The operation performed.
        """
        assertions = self._crdt.get(["assertions"]) or []
        return self._crdt.insert(["assertions"], len(assertions), assertion)

    def set_metadata(self, key: str, value: Any) -> CRDTOperation:
        """Set metadata value.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            The operation performed.
        """
        return self._crdt.set(["metadata", key], value)

    def apply(self, operation: CRDTOperation) -> bool:
        """Apply a remote operation.

        Args:
            operation: The operation to apply.

        Returns:
            True if operation was applied.
        """
        return self._crdt.apply(operation)

    def get_operations_since(self, vector_clock: VectorClock) -> list[CRDTOperation]:
        """Get operations since a vector clock."""
        return self._crdt.get_operations_since(vector_clock)

    def to_dict(self) -> dict:
        """Get full state."""
        return self._crdt.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "TestSpecCRDT":
        """Create from serialized state."""
        crdt = cls(node_id=data.get("node_id", str(uuid4())))
        crdt._crdt = CRDTDocument.from_dict(data)
        return crdt
