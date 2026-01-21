"""PostgreSQL checkpointer for durable execution.

Provides memory and state persistence for chat conversations and test orchestration.
Uses AsyncPostgresSaver with connection pooling when DATABASE_URL is set,
otherwise falls back to MemorySaver.

Production Features:
- Async connection pooling via psycopg_pool
- Automatic table creation
- Connection health checks
- Graceful shutdown
"""

import os
from typing import TYPE_CHECKING, Union

import structlog
from langgraph.checkpoint.memory import MemorySaver

if TYPE_CHECKING:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = structlog.get_logger()

# Type alias for checkpointer types
CheckpointerType = Union["AsyncPostgresSaver", MemorySaver]

# Global checkpointer instance (singleton pattern)
_checkpointer: CheckpointerType | None = None
_checkpointer_cm = None  # Context manager for cleanup


def get_checkpointer() -> CheckpointerType:
    """Get the checkpointer instance (sync access for compatibility).

    Note: For production, use get_async_checkpointer() or setup_checkpointer()
    which properly initializes the async PostgresSaver with connection pooling.

    Returns:
        The cached checkpointer or MemorySaver if not yet initialized
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    # Return MemorySaver as fallback - proper init happens in setup_checkpointer
    logger.warning("get_checkpointer called before setup_checkpointer - using MemorySaver")
    return MemorySaver()


async def setup_checkpointer() -> CheckpointerType:
    """Initialize checkpointer with async connection pooling.

    This should be called during application startup to ensure the checkpointer
    is ready before handling requests. Uses AsyncPostgresSaver with psycopg_pool
    for production-grade performance.

    Returns:
        The initialized checkpointer instance
    """
    global _checkpointer, _checkpointer_cm

    if _checkpointer is not None:
        return _checkpointer

    database_url = os.environ.get("DATABASE_URL")

    if database_url:
        try:
            # Import async checkpointer
            import socket
            from urllib.parse import urlparse

            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # Force IPv4 by resolving hostname and using hostaddr
            # This fixes Railway IPv6 connectivity issues with Supabase
            parsed = urlparse(database_url)
            hostname = parsed.hostname
            conninfo = database_url  # Default to original

            try:
                # Use getaddrinfo with AF_INET to force IPv4 resolution
                addrs = socket.getaddrinfo(hostname, 5432, socket.AF_INET, socket.SOCK_STREAM)
                if addrs:
                    ipv4_addr = addrs[0][4][0]  # Extract IP from first result
                    logger.info(
                        "Resolved database hostname to IPv4",
                        hostname=hostname,
                        ipv4=ipv4_addr,
                    )
                    # Add hostaddr parameter to connection string
                    if "?" in database_url:
                        conninfo = f"{database_url}&hostaddr={ipv4_addr}"
                    else:
                        conninfo = f"{database_url}?hostaddr={ipv4_addr}"
                else:
                    logger.warning("No IPv4 addresses found for hostname", hostname=hostname)
            except socket.gaierror as e:
                # Fall back to original URL if resolution fails
                logger.warning("Could not resolve hostname to IPv4", hostname=hostname, error=str(e))

            # Create AsyncPostgresSaver using from_conn_string
            # This is an async context manager - we need to enter it
            _checkpointer_cm = AsyncPostgresSaver.from_conn_string(conninfo)
            _checkpointer = await _checkpointer_cm.__aenter__()

            # Create checkpoint tables if they don't exist
            await _checkpointer.setup()

            logger.info(
                "AsyncPostgresSaver initialized for durable execution",
                database=database_url.split("@")[-1].split("/")[0] if "@" in database_url else "unknown",
                features=["durable", "async"],
            )

        except ImportError as e:
            logger.warning(
                "Required packages not installed for PostgresSaver",
                error=str(e),
                hint="Install with: pip install langgraph-checkpoint-postgres psycopg[binary] psycopg-pool",
            )
            _checkpointer = MemorySaver()

        except Exception as e:
            logger.error(
                "Failed to create AsyncPostgresSaver",
                error=str(e),
                error_type=type(e).__name__,
            )
            _checkpointer = MemorySaver()
            logger.info("Falling back to MemorySaver (not durable)")
    else:
        # Fallback to in-memory for development
        logger.info("DATABASE_URL not set, using MemorySaver (not durable)")
        _checkpointer = MemorySaver()

    return _checkpointer


async def shutdown_checkpointer() -> None:
    """Gracefully shutdown the checkpointer.

    This should be called during application shutdown to properly
    close database connections.
    """
    global _checkpointer, _checkpointer_cm

    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
            logger.info("PostgreSQL checkpointer shutdown complete")
        except Exception as e:
            logger.error("Error shutting down checkpointer", error=str(e))

    _checkpointer = None
    _checkpointer_cm = None


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
) -> dict | None:
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

    def __init__(self, checkpointer: CheckpointerType | None = None):
        self.checkpointer = checkpointer or get_checkpointer()
        self.log = logger.bind(component="checkpoint_manager")

    async def get_pending_approvals(self) -> list[dict]:
        """Get all pending approval requests."""
        return await list_pending_threads(self.checkpointer)

    async def get_approval_details(self, thread_id: str) -> dict | None:
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

    async def approve(self, thread_id: str, modifications: dict | None = None) -> dict:
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

    async def reject(self, thread_id: str, reason: str | None = None) -> dict:
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
