import asyncio
import psutil

from core.module_base import GhostModule


class Heartbeat(GhostModule):

    def __init__(
        self,
        logger,
        state_manager,
        interval_seconds: float = 5.0
    ):
        self.logger = logger
        self.state_manager = state_manager
        self.interval_seconds = interval_seconds
        self.running = False

    async def initialize(self):
        self.logger.info(
            "heartbeat_initialized",
            interval=self.interval_seconds
        )

    async def start(self):
        self.running = True

        while self.running:
            # BUG FIX: asyncio.sleep was previously OUTSIDE the try block.
            # If the task was cancelled during sleep, CancelledError would
            # propagate upward unhandled and kill the loop without any log.
            #
            # Now the entire tick — work + sleep — lives inside the try so:
            #   • CancelledError is caught explicitly → clean break + log
            #   • Regular exceptions are caught → logged, loop continues
            try:
                memory = psutil.virtual_memory().percent
                cpu    = psutil.cpu_percent(interval=None)

                await self.state_manager.update_system_state("memory_usage", memory)
                await self.state_manager.update_system_state("cpu_usage", cpu)

                self.logger.info(
                    "heartbeat",
                    memory_usage=memory,
                    cpu_usage=cpu
                )

                await asyncio.sleep(self.interval_seconds)

            except asyncio.CancelledError:
                # Task was cancelled (expected during shutdown) — exit cleanly
                self.logger.info("heartbeat_cancelled")
                break

            except Exception as e:
                self.logger.error("heartbeat_error", error=str(e))
                # Back off briefly before retrying so a persistent error
                # doesn't spin at full speed
                await asyncio.sleep(self.interval_seconds)

    async def stop(self):
        self.running = False
        self.logger.info("heartbeat_stopped")

    async def health_check(self) -> dict:
        return {
            "status": "ok",
            "running": self.running,
            "interval_seconds": self.interval_seconds
        }
