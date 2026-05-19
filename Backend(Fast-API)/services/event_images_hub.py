import asyncio
from collections import defaultdict
from typing import Any


class EventImagesHub:
    """In-memory pub/sub for event image SSE streams (single-process)."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, event_id: str, *, max_queue_size: int = 64) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        async with self._lock:
            self._subscribers[event_id].add(queue)
        return queue

    async def unsubscribe(self, event_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(event_id)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                del self._subscribers[event_id]

    async def publish(self, event_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(event_id, ()))

        for queue in queues:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    async def close_all(self) -> None:
        async with self._lock:
            self._subscribers.clear()


event_images_hub = EventImagesHub()
