import asyncio


class StateManager:

    def __init__(self):
        self._lock = asyncio.Lock()
        self.runtime_state: str = "BOOTING"

        self.cognitive_state: dict = {
            "active_goal": None,
            "current_focus": None,
            "confidence": 1.0,
            "active_tasks": [],
            "risk_level": 0
        }

        self.system_state: dict = {
            "loaded_modules": [],
            "memory_usage": 0,
            "cpu_usage": 0,
            "model_loaded": False
        }

    async def set_runtime_state(self, state: str):
        async with self._lock:
            self.runtime_state = state

    async def get_runtime_state(self) -> str:
        async with self._lock:
            return self.runtime_state

    async def update_system_state(self, key: str, value):
        async with self._lock:
            self.system_state[key] = value

    async def update_cognitive_state(self, key: str, value):
        async with self._lock:
            self.cognitive_state[key] = value

    async def get_full_state(self) -> dict:
        async with self._lock:
            return {
                "runtime_state": self.runtime_state,
                "cognitive_state": dict(self.cognitive_state),
                "system_state": dict(self.system_state)
            }
