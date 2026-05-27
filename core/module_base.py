from abc import ABC, abstractmethod


class GhostModule(ABC):
    """
    Abstract base class for all GhostMind modules.

    Every module must implement this four-method lifecycle contract.
    Runtime iterates self.modules and calls each in order:

        initialize() → start() → [running] → stop()
    """

    @abstractmethod
    async def initialize(self):
        """Called once at startup, before start(). Load resources here."""
        pass

    @abstractmethod
    async def start(self):
        """Begin the module's main loop or operation."""
        pass

    @abstractmethod
    async def stop(self):
        """Gracefully shut down. Release resources, cancel loops."""
        pass

    @abstractmethod
    async def health_check(self) -> dict:
        """Return {"status": "ok" | "error", ...}."""
        pass
