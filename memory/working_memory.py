import asyncio


class WorkingMemory:
    """
    Short-term in-session memory — a capped sliding window of context items.

    FIX: All methods are now async and guarded by an asyncio.Lock,
    preventing race conditions when multiple tasks read/write simultaneously.
    """

    def __init__(self, max_items: int = 20):
        self.max_items = max_items
        self._context: list = []
        self._lock = asyncio.Lock()

    async def add(self, item):
        async with self._lock:
            self._context.append(item)
            if len(self._context) > self.max_items:
                self._context.pop(0)

    async def get_context(self) -> list:
        async with self._lock:
            return list(self._context)      # return a copy, not the live list

    async def clear(self):
        async with self._lock:
            self._context.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._context)
