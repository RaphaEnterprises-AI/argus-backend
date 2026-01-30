"""Database persistence for collaboration state.

Provides write-through caching - in-memory for real-time performance,
with async persistence to Supabase for durability.
"""

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import structlog

from src.services.supabase_client import get_supabase_client

from .crdt import CRDTOperation, TestSpecCRDT, VectorClock
from .models import CollaborativeComment, UserPresence

logger = structlog.get_logger()


class CollaborationPersistence:
    """Handles database persistence for collaboration state.

    Uses write-through caching pattern:
    - Reads first check in-memory cache, then fall back to DB
    - Writes update both in-memory and DB (async for non-blocking)
    """

    def __init__(self, organization_id: str | None = None):
        """Initialize persistence layer.

        Args:
            organization_id: Optional organization ID for multi-tenant isolation.
        """
        self.organization_id = organization_id
        self._supabase = get_supabase_client()
        self._write_queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._write_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the background write task."""
        if self._write_task is None:
            self._write_task = asyncio.create_task(self._process_writes())
            logger.info("Collaboration persistence started")

    async def stop(self) -> None:
        """Stop the background write task."""
        if self._write_task:
            self._write_task.cancel()
            try:
                await self._write_task
            except asyncio.CancelledError:
                pass
            self._write_task = None
            logger.info("Collaboration persistence stopped")

    # =========================================================================
    # Session Persistence
    # =========================================================================

    async def save_session(
        self,
        session_id: str,
        user_id: str,
        workspace_id: str | None,
        test_id: str | None,
        status: str = "active",
    ) -> None:
        """Save or update a collaboration session.

        Args:
            session_id: Unique session identifier.
            user_id: User identifier.
            workspace_id: Workspace identifier.
            test_id: Optional test identifier.
            status: Session status.
        """
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "test_id": test_id,
            "status": status,
            "last_activity": datetime.now(UTC).isoformat(),
        }
        if self.organization_id:
            data["organization_id"] = self.organization_id

        await self._queue_write("upsert_session", data)

    async def update_session_activity(self, session_id: str) -> None:
        """Update session last activity timestamp.

        Args:
            session_id: Session identifier.
        """
        await self._queue_write("update_session_activity", {
            "session_id": session_id,
            "last_activity": datetime.now(UTC).isoformat(),
        })

    async def delete_session(self, session_id: str) -> None:
        """Delete a session.

        Args:
            session_id: Session identifier.
        """
        await self._queue_write("delete_session", {"session_id": session_id})

    async def load_active_sessions(
        self,
        workspace_id: str | None = None,
    ) -> list[dict]:
        """Load active sessions from database.

        Args:
            workspace_id: Optional workspace filter.

        Returns:
            List of session dictionaries.
        """
        filters = {"status": "eq.active"}
        if workspace_id:
            filters["workspace_id"] = f"eq.{workspace_id}"
        if self.organization_id:
            filters["organization_id"] = f"eq.{self.organization_id}"

        result = await self._supabase.select(
            "collaboration_sessions",
            columns="*",
            filters=filters,
        )
        return result.get("data") or []

    # =========================================================================
    # CRDT Document Persistence
    # =========================================================================

    async def save_crdt_document(
        self,
        test_id: str,
        crdt: TestSpecCRDT,
        modified_by: str | None = None,
    ) -> None:
        """Save CRDT document state.

        Args:
            test_id: Test identifier.
            crdt: The CRDT document to save.
            modified_by: User who made the modification.
        """
        crdt_state = crdt.to_dict()
        data = {
            "test_id": test_id,
            "document_state": crdt_state.get("document", {}),
            "vector_clock": crdt_state.get("vector_clock", {}),
            "last_modified": datetime.now(UTC).isoformat(),
            "last_modified_by": modified_by,
        }
        if self.organization_id:
            data["organization_id"] = self.organization_id

        await self._queue_write("upsert_crdt_document", data)

    async def load_crdt_document(self, test_id: str) -> TestSpecCRDT | None:
        """Load CRDT document from database.

        Args:
            test_id: Test identifier.

        Returns:
            TestSpecCRDT if found, None otherwise.
        """
        filters = {"test_id": f"eq.{test_id}"}
        if self.organization_id:
            filters["organization_id"] = f"eq.{self.organization_id}"

        result = await self._supabase.select(
            "crdt_documents",
            columns="*",
            filters=filters,
        )

        data = result.get("data")
        if not data or len(data) == 0:
            return None

        doc = data[0]
        crdt = TestSpecCRDT(
            node_id="server",
            test_spec=doc.get("document_state", {}),
        )
        crdt._crdt._vector_clock = VectorClock.from_dict(
            doc.get("vector_clock", {})
        )
        return crdt

    async def save_crdt_operation(
        self,
        document_id: str,
        operation: CRDTOperation,
        user_id: str | None = None,
    ) -> None:
        """Save a CRDT operation for history/sync.

        Args:
            document_id: Document UUID (from crdt_documents table).
            operation: The operation to save.
            user_id: User who performed the operation.
        """
        data = {
            "document_id": document_id,
            "operation_id": operation.id,
            "node_id": operation.node_id,
            "operation_type": operation.operation,
            "path": operation.path,
            "value": operation.value,
            "previous_value": operation.previous_value,
            "vector_clock": operation.vector_clock.to_dict(),
            "applied_at": operation.timestamp.isoformat(),
            "user_id": user_id,
        }
        await self._queue_write("insert_crdt_operation", data)

    async def get_document_id(self, test_id: str) -> str | None:
        """Get the database UUID for a CRDT document.

        Args:
            test_id: Test identifier.

        Returns:
            Document UUID if found.
        """
        filters = {"test_id": f"eq.{test_id}"}
        result = await self._supabase.select(
            "crdt_documents",
            columns="id",
            filters=filters,
        )
        data = result.get("data")
        if data and len(data) > 0:
            return data[0].get("id")
        return None

    # =========================================================================
    # Presence Persistence (Optional - for analytics/audit)
    # =========================================================================

    async def save_presence(self, presence: UserPresence) -> None:
        """Save user presence (optional persistence).

        Args:
            presence: User presence object.
        """
        data = {
            "user_id": presence.user_id,
            "user_name": presence.user_name,
            "user_email": presence.user_email,
            "workspace_id": presence.workspace_id,
            "test_id": presence.test_id or "",  # Use empty string for NULL in UNIQUE constraint
            "status": presence.status.value,
            "cursor_position": {
                "x": presence.cursor.x,
                "y": presence.cursor.y,
                "element_id": presence.cursor.element_id,
            } if presence.cursor else None,
            "color": presence.color,
            "last_seen": datetime.now(UTC).isoformat(),
        }
        if self.organization_id:
            data["organization_id"] = self.organization_id

        await self._queue_write("upsert_presence", data)

    async def delete_presence(
        self,
        user_id: str,
        workspace_id: str,
        test_id: str | None = None,
    ) -> None:
        """Delete user presence.

        Args:
            user_id: User identifier.
            workspace_id: Workspace identifier.
            test_id: Optional test identifier.
        """
        await self._queue_write("delete_presence", {
            "user_id": user_id,
            "workspace_id": workspace_id,
            "test_id": test_id or "",
        })

    async def load_workspace_presence(
        self,
        workspace_id: str,
    ) -> list[dict]:
        """Load presence data for a workspace.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            List of presence dictionaries.
        """
        filters = {
            "workspace_id": f"eq.{workspace_id}",
            "status": "neq.offline",
        }
        if self.organization_id:
            filters["organization_id"] = f"eq.{self.organization_id}"

        result = await self._supabase.select(
            "user_presence",
            columns="*",
            filters=filters,
        )
        return result.get("data") or []

    # =========================================================================
    # Comments Persistence
    # =========================================================================

    async def save_comment(self, comment: CollaborativeComment) -> str:
        """Save a collaborative comment.

        Args:
            comment: The comment to save.

        Returns:
            Comment ID.
        """
        data = {
            "id": comment.id,
            "test_id": comment.test_id,
            "step_index": comment.step_index,
            "author_id": comment.author_id,
            "author_name": comment.author_name,
            "author_avatar": comment.author_avatar,
            "content": comment.content,
            "mentions": comment.mentions,
            "resolved": comment.resolved,
            "resolved_by": comment.resolved_by,
            "resolved_at": comment.resolved_at.isoformat() if comment.resolved_at else None,
            "parent_id": comment.parent_id,
        }
        if self.organization_id:
            data["organization_id"] = self.organization_id

        await self._queue_write("upsert_comment", data)
        return comment.id

    async def update_comment_resolved(
        self,
        comment_id: str,
        resolved_by: str,
    ) -> None:
        """Mark a comment as resolved.

        Args:
            comment_id: Comment identifier.
            resolved_by: User who resolved the comment.
        """
        await self._queue_write("update_comment_resolved", {
            "comment_id": comment_id,
            "resolved_by": resolved_by,
            "resolved_at": datetime.now(UTC).isoformat(),
        })

    async def load_comments(
        self,
        test_id: str,
        include_resolved: bool = False,
    ) -> list[CollaborativeComment]:
        """Load comments for a test.

        Args:
            test_id: Test identifier.
            include_resolved: Whether to include resolved comments.

        Returns:
            List of CollaborativeComment objects.
        """
        filters = {"test_id": f"eq.{test_id}"}
        if not include_resolved:
            filters["resolved"] = "eq.false"
        if self.organization_id:
            filters["organization_id"] = f"eq.{self.organization_id}"

        result = await self._supabase.select(
            "collaborative_comments",
            columns="*",
            filters=filters,
        )

        comments = []
        for row in result.get("data") or []:
            comment = CollaborativeComment(
                id=row["id"],
                test_id=row["test_id"],
                step_index=row.get("step_index"),
                author_id=row["author_id"],
                author_name=row.get("author_name", ""),
                author_avatar=row.get("author_avatar"),
                content=row["content"],
                mentions=row.get("mentions", []),
                resolved=row.get("resolved", False),
                resolved_by=row.get("resolved_by"),
                resolved_at=datetime.fromisoformat(row["resolved_at"]) if row.get("resolved_at") else None,
                parent_id=row.get("parent_id"),
                created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(UTC),
            )
            comments.append(comment)

        return comments

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _queue_write(self, operation: str, data: dict) -> None:
        """Queue a write operation for async processing.

        Args:
            operation: Operation type.
            data: Data to write.
        """
        await self._write_queue.put((operation, data))

    async def _process_writes(self) -> None:
        """Background task to process queued writes."""
        while True:
            try:
                operation, data = await self._write_queue.get()
                await self._execute_write(operation, data)
                self._write_queue.task_done()
            except asyncio.CancelledError:
                # Process remaining items before exiting
                while not self._write_queue.empty():
                    try:
                        operation, data = self._write_queue.get_nowait()
                        await self._execute_write(operation, data)
                    except Exception as e:
                        logger.error("Error processing final write", error=str(e))
                break
            except Exception as e:
                logger.error("Error processing write", operation=operation, error=str(e))

    async def _execute_write(self, operation: str, data: dict) -> None:
        """Execute a write operation.

        Args:
            operation: Operation type.
            data: Data to write.
        """
        try:
            if operation == "upsert_session":
                await self._supabase.request(
                    "/collaboration_sessions",
                    method="POST",
                    body=data,
                    headers={"Prefer": "resolution=merge-duplicates"},
                )

            elif operation == "update_session_activity":
                await self._supabase.update(
                    "collaboration_sessions",
                    filters={"session_id": f"eq.{data['session_id']}"},
                    data={"last_activity": data["last_activity"]},
                )

            elif operation == "delete_session":
                await self._supabase.request(
                    f"/collaboration_sessions?session_id=eq.{data['session_id']}",
                    method="DELETE",
                )

            elif operation == "upsert_crdt_document":
                # Try to get existing version first
                existing = await self._supabase.select(
                    "crdt_documents",
                    columns="id,version",
                    filters={"test_id": f"eq.{data['test_id']}"},
                )
                existing_data = existing.get("data")

                if existing_data and len(existing_data) > 0:
                    # Update existing
                    data["version"] = (existing_data[0].get("version", 0) or 0) + 1
                    await self._supabase.update(
                        "crdt_documents",
                        filters={"test_id": f"eq.{data['test_id']}"},
                        data=data,
                    )
                else:
                    # Insert new
                    data["version"] = 1
                    await self._supabase.insert("crdt_documents", data)

            elif operation == "insert_crdt_operation":
                await self._supabase.insert("crdt_operations", data)

            elif operation == "upsert_presence":
                await self._supabase.request(
                    "/user_presence",
                    method="POST",
                    body=data,
                    headers={"Prefer": "resolution=merge-duplicates"},
                )

            elif operation == "delete_presence":
                test_id = data.get("test_id") or ""
                await self._supabase.request(
                    f"/user_presence?user_id=eq.{data['user_id']}&workspace_id=eq.{data['workspace_id']}&test_id=eq.{test_id}",
                    method="DELETE",
                )

            elif operation == "upsert_comment":
                await self._supabase.request(
                    "/collaborative_comments",
                    method="POST",
                    body=data,
                    headers={"Prefer": "resolution=merge-duplicates"},
                )

            elif operation == "update_comment_resolved":
                await self._supabase.update(
                    "collaborative_comments",
                    filters={"id": f"eq.{data['comment_id']}"},
                    data={
                        "resolved": True,
                        "resolved_by": data["resolved_by"],
                        "resolved_at": data["resolved_at"],
                    },
                )

            else:
                logger.warning("Unknown write operation", operation=operation)

        except Exception as e:
            logger.error(
                "Failed to execute write",
                operation=operation,
                error=str(e),
            )


# Global instance
_persistence: CollaborationPersistence | None = None


def get_collaboration_persistence(
    organization_id: str | None = None,
) -> CollaborationPersistence:
    """Get or create collaboration persistence instance.

    Args:
        organization_id: Optional organization ID.

    Returns:
        CollaborationPersistence instance.
    """
    global _persistence
    if _persistence is None:
        _persistence = CollaborationPersistence(organization_id)
    return _persistence
