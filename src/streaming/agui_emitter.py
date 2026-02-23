"""AG-UI event emitter for streaming agent activity to clients via SSE.

Each swarm run gets its own emitter identified by run_id. The emitter
buffers events in an asyncio.Queue and yields them as SSE-formatted dicts
for use with FastAPI's StreamingResponse / EventSourceResponse.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from src.streaming.agui_events import AGUIEvent


class AGUIEmitter:
    """Buffered emitter that queues AG-UI events and streams them as SSE."""

    def __init__(self, run_id: str, max_queue_size: int = 1000) -> None:
        self.run_id = run_id
        self._queue: asyncio.Queue[AGUIEvent | None] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._closed = False

    async def emit(self, event: AGUIEvent) -> None:
        """Enqueue an AG-UI event for streaming. No-op if emitter is closed."""
        if self._closed:
            return
        await self._queue.put(event)

    def emit_sync(self, event: AGUIEvent) -> None:
        """Non-async emit for use from synchronous callbacks.

        Uses put_nowait which raises asyncio.QueueFull if the queue is at
        capacity. Silently drops events if the emitter is closed.
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop event rather than block or crash

    async def close(self) -> None:
        """Signal end-of-stream by enqueuing a sentinel None value."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)

    async def stream(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator yielding SSE-formatted dicts until closed.

        Each yielded dict has the shape:
            {"event": "<EVENT_TYPE>", "data": "<json_string>"}

        Suitable for use with sse-starlette's EventSourceResponse or
        similar SSE adapters.
        """
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield {
                "event": event.type.value,
                "data": event.to_sse_data(),
            }


# ── Module-level emitter registry ──────────────────────────────────────

_emitters: dict[str, AGUIEmitter] = {}


def create_emitter(run_id: str, max_queue_size: int = 1000) -> AGUIEmitter:
    """Create and register a new AGUIEmitter for the given run_id.

    If an emitter already exists for this run_id, it is replaced.
    """
    emitter = AGUIEmitter(run_id=run_id, max_queue_size=max_queue_size)
    _emitters[run_id] = emitter
    return emitter


def get_emitter(run_id: str) -> AGUIEmitter | None:
    """Retrieve the emitter for a given run_id, or None if not found."""
    return _emitters.get(run_id)


def remove_emitter(run_id: str) -> None:
    """Remove an emitter from the registry."""
    _emitters.pop(run_id, None)
