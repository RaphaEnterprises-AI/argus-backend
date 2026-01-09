"""PostgreSQL checkpointer for durable execution.

Provides memory and state persistence for chat conversations and test orchestration.
Uses PostgresSaver when DATABASE_URL is set, otherwise falls back to MemorySaver.
"""

import os
from typing import Optional, Union

from langgraph.checkpoint.memory import MemorySaver
import structlog

logger = structlog.get_logger()

# Type alias for checkpointer types
CheckpointerType = Union["PostgresSaver", MemorySaver]

# Global checkpointer instance (singleton pattern)
_checkpointer: Optional[CheckpointerType] = None


def get_checkpointer() -> CheckpointerType:
    """Get or create the checkpointer instance.

    Returns PostgresSaver if DATABASE_URL is set, otherwise falls back to MemorySaver.
    The checkpointer is cached as a singleton for the lifetime of the application.

    Returns:
        PostgresSaver for durable execution or MemorySaver for in-memory (development)
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        try:
            # Import here to avoid dependency issues when PostgresSaver is not installed
            from langgraph.checkpoint.postgres import PostgresSaver

            # Use PostgresSaver for production
            _checkpointer = PostgresSaver.from_conn_string(database_url)
            logger.info("Using PostgresSaver for durable execution")
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-postgres not installed, falling back to MemorySaver"
            )
            _checkpointer = MemorySaver()
        except Exception as e:
            logger.warning(
                f"Failed to create PostgresSaver: {e}, falling back to MemorySaver"
            )
            _checkpointer = MemorySaver()
    else:
        # Fallback to in-memory for development
        logger.info("DATABASE_URL not set, using MemorySaver (not durable)")
        _checkpointer = MemorySaver()

    return _checkpointer


async def setup_checkpointer() -> CheckpointerType:
    """Initialize checkpointer and ensure tables exist.

    This should be called during application startup to ensure the checkpointer
    is ready before handling requests.

    Returns:
        The initialized checkpointer instance
    """
    checkpointer = get_checkpointer()

    # Check if it's a PostgresSaver for logging purposes
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        if isinstance(checkpointer, PostgresSaver):
            # PostgresSaver automatically creates tables on first use
            logger.info("PostgresSaver initialized for durable execution")
    except ImportError:
        pass

    return checkpointer


def reset_checkpointer() -> None:
    """Reset the cached checkpointer instance.

    This is primarily useful for testing or when reconfiguring the database connection.
    """
    global _checkpointer
    _checkpointer = None
    logger.info("Checkpointer cache cleared")


async def list_pending_threads(checkpointer: CheckpointerType) -> list[dict]:
    """
    List all threads that are currently paused at an approval point.

    This scans through checkpoints to find threads waiting for human input.

    Args:
        checkpointer: The checkpointer to query

    Returns:
        List of pending thread information
    """
    pending = []

    try:
        # For MemorySaver, we can iterate over stored checkpoints
        if isinstance(checkpointer, MemorySaver):
            # MemorySaver stores data in storage dict
            storage = getattr(checkpointer, 'storage', {})

            for thread_id, checkpoints in storage.items():
                if checkpoints:
                    # Get the latest checkpoint
                    latest = max(checkpoints.values(), key=lambda c: c.get('ts', 0))

                    # Check if it's paused at an approval point
                    metadata = latest.get('metadata', {})
                    if metadata.get('next'):
                        next_nodes = metadata.get('next', [])
                        if any(node in ['self_heal', 'request_approval'] for node in next_nodes):
                            pending.append({
                                'thread_id': thread_id,
                                'paused_at': next_nodes[0] if next_nodes else 'unknown',
                                'checkpoint_id': latest.get('id'),
                                'created_at': metadata.get('created_at'),
                            })
        else:
            # For other checkpointers, use the list method if available
            try:
                async for checkpoint in checkpointer.alist({}):
                    metadata = checkpoint.metadata or {}
                    if metadata.get('next'):
                        next_nodes = metadata.get('next', [])
                        if any(node in ['self_heal', 'request_approval'] for node in next_nodes):
                            pending.append({
                                'thread_id': checkpoint.config.get('configurable', {}).get('thread_id'),
                                'paused_at': next_nodes[0] if next_nodes else 'unknown',
                                'checkpoint_id': checkpoint.checkpoint_id,
                                'created_at': metadata.get('created_at'),
                            })
            except AttributeError:
                logger.warning("Checkpointer does not support async listing")

    except Exception as e:
        logger.error("Failed to list pending threads", error=str(e))

    return pending


async def get_thread_state(
    checkpointer: CheckpointerType,
    thread_id: str,
) -> Optional[dict]:
    """
    Get the current state of a thread.

    Args:
        checkpointer: The checkpointer to query
        thread_id: The thread ID to look up

    Returns:
        The thread state or None if not found
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Try async get first
        try:
            checkpoint = await checkpointer.aget(config)
        except AttributeError:
            # Fall back to sync get
            checkpoint = checkpointer.get(config)

        if checkpoint:
            return {
                'thread_id': thread_id,
                'checkpoint_id': getattr(checkpoint, 'checkpoint_id', None),
                'values': getattr(checkpoint, 'channel_values', {}),
                'metadata': getattr(checkpoint, 'metadata', {}),
                'next': checkpoint.metadata.get('next') if hasattr(checkpoint, 'metadata') and checkpoint.metadata else None,
            }

    except Exception as e:
        logger.error("Failed to get thread state", thread_id=thread_id, error=str(e))

    return None


class CheckpointManager:
    """
    Manager for handling checkpoint operations.

    Provides higher-level operations for managing paused executions,
    including approval workflows and state updates.
    """

    def __init__(self, checkpointer: Optional[CheckpointerType] = None):
        self.checkpointer = checkpointer or get_checkpointer()
        self.log = logger.bind(component="checkpoint_manager")

    async def get_pending_approvals(self) -> list[dict]:
        """Get all pending approval requests."""
        return await list_pending_threads(self.checkpointer)

    async def get_approval_details(self, thread_id: str) -> Optional[dict]:
        """Get detailed information about a pending approval."""
        state = await get_thread_state(self.checkpointer, thread_id)

        if not state:
            return None

        next_nodes = state.get('next', [])
        next_node = next_nodes[0] if next_nodes else None
        values = state.get('values', {})

        # Build approval context based on what's pending
        if next_node == 'self_heal':
            return {
                'thread_id': thread_id,
                'approval_type': 'healing',
                'description': 'Approve self-healing for failed tests',
                'context': {
                    'failures': values.get('failures', []),
                    'healing_queue': values.get('healing_queue', []),
                    'passed_count': values.get('passed_count', 0),
                    'failed_count': values.get('failed_count', 0),
                },
                'options': ['approve', 'reject', 'modify'],
            }
        elif next_node == 'plan_tests':
            return {
                'thread_id': thread_id,
                'approval_type': 'test_plan',
                'description': 'Approve test plan before execution',
                'context': {
                    'test_plan': values.get('test_plan', []),
                    'testable_surfaces': values.get('testable_surfaces', []),
                },
                'options': ['approve', 'reject', 'modify'],
            }

        return {
            'thread_id': thread_id,
            'approval_type': 'unknown',
            'description': f'Paused at: {next_node}',
            'context': {},
            'options': ['resume', 'abort'],
        }

    async def approve(self, thread_id: str, modifications: Optional[dict] = None) -> dict:
        """
        Approve a pending action and prepare for resume.

        Args:
            thread_id: The thread to approve
            modifications: Optional state modifications to apply

        Returns:
            Result of the approval operation
        """
        self.log.info("Approving action", thread_id=thread_id)

        state = await get_thread_state(self.checkpointer, thread_id)
        if not state:
            return {'success': False, 'error': 'Thread not found'}

        return {
            'success': True,
            'thread_id': thread_id,
            'ready_to_resume': True,
            'modifications': modifications,
        }

    async def reject(self, thread_id: str, reason: Optional[str] = None) -> dict:
        """
        Reject a pending action.

        Args:
            thread_id: The thread to reject
            reason: Optional reason for rejection

        Returns:
            Result of the rejection operation
        """
        self.log.info("Rejecting action", thread_id=thread_id, reason=reason)

        return {
            'success': True,
            'thread_id': thread_id,
            'rejected': True,
            'reason': reason,
        }
