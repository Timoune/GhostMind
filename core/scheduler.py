import asyncio
from typing import Callable


class Scheduler:
    """
    Async task scheduler.

    FIX: Both methods now accept a coroutine *factory* (a callable) plus
    optional args/kwargs, instead of a coroutine object. This is required
    because a coroutine object can only be awaited once — passing the same
    object to schedule_repeat would crash after the first iteration with
    "RuntimeError: cannot reuse already awaited coroutine".
    """

    def __init__(self):
        self.tasks: list[asyncio.Task] = []
        self.running: bool = True

    async def schedule_once(
        self,
        delay: float,
        coro_fn: Callable,
        *args,
        **kwargs
    ) -> asyncio.Task:
        """Run coro_fn(*args, **kwargs) once after `delay` seconds."""

        async def wrapper():
            await asyncio.sleep(delay)
            await coro_fn(*args, **kwargs)

        task = asyncio.create_task(wrapper())
        self.tasks.append(task)
        return task

    async def schedule_repeat(
        self,
        interval: float,
        coro_fn: Callable,
        *args,
        **kwargs
    ) -> asyncio.Task:
        """
        Call coro_fn(*args, **kwargs) repeatedly every `interval` seconds.

        A fresh coroutine is created on every iteration by calling
        coro_fn(...) each time, not by re-awaiting the same object.
        """

        async def wrapper():
            while self.running:
                try:
                    await coro_fn(*args, **kwargs)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    # Swallow errors so the loop keeps running
                    pass
                await asyncio.sleep(interval)

        task = asyncio.create_task(wrapper())
        self.tasks.append(task)
        return task

    async def shutdown(self):
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
