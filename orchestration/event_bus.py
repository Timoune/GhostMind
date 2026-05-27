import asyncio
import uuid
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from core.module_base import GhostModule


@dataclass
class Event:
    """
    An event routed through the EventBus.

    Priority: 1 = highest urgency, 10 = lowest.
    Equal-priority events are ordered by timestamp (FIFO within tier).
    """
    event_type: str
    source: str
    payload: Any
    priority: int = 5

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def __lt__(self, other: "Event") -> bool:
        # Required so PriorityQueue can break ties between equal-priority events
        return self.timestamp < other.timestamp


class EventBus(GhostModule):
    """
    Async priority pub/sub event bus.
    """

    def __init__(self):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.subscribers: dict[str, list[Callable]] = defaultdict(list)
        self.running: bool = False

        # BUG FIX: track subscriber callback tasks so they cannot be
        # GC'd mid-execution and so exceptions inside callbacks are not
        # silently dropped.
        self._callback_tasks: set[asyncio.Task] = set()

    async def initialize(self):
        pass

    async def publish(self, event: Event):
        await self.queue.put((event.priority, event))

    def subscribe(self, event_type: str, callback: Callable):
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
            except ValueError:
                pass

    async def start(self):
        self.running = True

        while self.running:
            try:
                # Timeout lets the loop notice self.running = False promptly
                _, event = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            for callback in self.subscribers.get(event.event_type, []):
                # BUG FIX: store the task reference and attach a done-callback
                # that removes it from the set and logs any exception.
                task = asyncio.create_task(
                    callback(event),
                    name=f"cb:{event.event_type}"
                )
                self._callback_tasks.add(task)
                task.add_done_callback(self._handle_callback_result)

    def _handle_callback_result(self, task: asyncio.Task):
        """Remove finished callback tasks and surface any exceptions."""
        self._callback_tasks.discard(task)

        if task.cancelled():
            return

        exc = task.exception()
        if exc is not None:
            # We don't have self.logger here (EventBus is intentionally
            # lightweight), so print to stderr — replace with your logger
            # if you inject one into EventBus later.
            import sys
            print(
                f"[EventBus] callback task {task.get_name()} raised: {exc}",
                file=sys.stderr
            )

    async def stop(self):
        self.running = False

        # Wait for any in-flight callback tasks to complete
        if self._callback_tasks:
            await asyncio.gather(*self._callback_tasks, return_exceptions=True)

    async def health_check(self) -> dict:
        return {
            "status": "ok",
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "active_callbacks": len(self._callback_tasks),
            "subscriber_types": list(self.subscribers.keys())
        }
