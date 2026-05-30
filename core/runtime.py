import asyncio

from core.config_loader    import ConfigLoader
from core.logger           import LoggerManager
from core.state_manager    import StateManager
from core.scheduler        import Scheduler
from core.heartbeat        import Heartbeat
from core.autonomous_agent import AutonomousAgent
from core.module_base      import GhostModule

from orchestration.event_bus import EventBus

from memory.memory_bridge   import MemoryBridge
from memory.working_memory  import WorkingMemory
from memory.context_loader  import ContextLoader

from llm.model_client     import ModelClient
from cognition.pipeline   import CognitionPipeline


class Runtime:

    def __init__(self):
        self.running        = False
        self.config_loader  = ConfigLoader()
        self.logger_manager = LoggerManager()
        self.state_manager  = StateManager()
        self.scheduler      = Scheduler()
        self.modules: list[GhostModule] = []

        # Keep task references alive (GC safety) and allow clean shutdown
        self._bg_tasks: set[asyncio.Task] = set()

    async def initialize(self):

        # ââ Config ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.config_loader.load_all()

        # ââ Logging âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        log_dir = self.config_loader.get("runtime.logging.log_dir", "logs/")
        self.logger_manager = LoggerManager(log_dir=log_dir)
        self.logger_manager.initialize()
        self.logger = self.logger_manager.get_logger("runtime")
        self.logger.info("runtime_initializing")

        # ââ Event Bus âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.event_bus = EventBus()

        # ââ Heartbeat âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        interval = self.config_loader.get(
            "runtime.heartbeat.interval_seconds", 5
        )
        self.heartbeat = Heartbeat(
            logger=self.logger,
            state_manager=self.state_manager,
            interval_seconds=interval,
        )

        # ââ Memory ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        memory_host = self.config_loader.get("memory.api.host",             "127.0.0.1")
        memory_port = self.config_loader.get("memory.api.port",             8000)
        max_items   = self.config_loader.get("memory.working_memory.max_items", 20)

        self.memory_bridge = MemoryBridge(
            host=memory_host,
            port=memory_port,
            logger=self.logger,
        )
        await self.memory_bridge.start()

        self.working_memory = WorkingMemory(max_items=max_items)
        self.context_loader = ContextLoader(
            memory_bridge=self.memory_bridge,
            working_memory=self.working_memory,
            logger=self.logger,
        )

        try:
            health = await self.memory_bridge.health_check()
            self.logger.info("dreamcloud_connected", health=health)
        except Exception as e:
            self.logger.warning(
                "dreamcloud_unavailable",
                error=str(e),
                note="Running without long-term memory until DreamCloud starts",
            )

        # ââ LLM âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        llm_endpoint = self.config_loader.get("models.endpoint",         "http://127.0.0.1:8080")
        llm_timeout  = self.config_loader.get("models.timeout_seconds",  60)
        temperature  = self.config_loader.get("models.temperature",      0.7)
        max_tokens   = self.config_loader.get("models.max_tokens",       1024)

        self.model_client = ModelClient(
            endpoint=llm_endpoint,
            timeout_seconds=llm_timeout,
            logger=self.logger,
        )
        await self.model_client.start()

        llm_alive = await self.model_client.health_check()
        if llm_alive:
            self.logger.info("llm_connected", endpoint=llm_endpoint)
            await self.state_manager.update_system_state("model_loaded", True)
        else:
            self.logger.warning(
                "llm_unreachable",
                endpoint=llm_endpoint,
                note=(
                    "Start llama-server before calling think(). "
                    "llama-server --model MiniVon/llm/<model>.gguf --port 8080"
                ),
            )

        # ââ Cognition settings ââââââââââââââââââââââââââââââââââââââââââââââââ
        max_depth       = self.config_loader.get("cognition.max_reasoning_depth", 5)
        autonomous_mode = self.config_loader.get("safety.autonomous_mode",        False)

        # ââ Cognition Pipeline ââââââââââââââââââââââââââââââââââââââââââââââââ
        self.cognition = CognitionPipeline(
            model_client=self.model_client,
            context_loader=self.context_loader,
            working_memory=self.working_memory,
            memory_bridge=self.memory_bridge,
            state_manager=self.state_manager,
            logger=self.logger,
            event_bus=self.event_bus,
            temperature=temperature,
            max_tokens=max_tokens,
            max_reasoning_depth=max_depth,
            autonomous_mode=autonomous_mode,
        )

        # ââ Autonomous Agent ââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.autonomous_agent = AutonomousAgent(
            scheduler=self.scheduler,
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            memory_bridge=self.memory_bridge,
            working_memory=self.working_memory,
            logger=self.logger,
            autonomous_mode=autonomous_mode,
        )

        # ââ Register modules ââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.modules.extend([
            self.event_bus,
            self.heartbeat,
            self.autonomous_agent,
        ])

        for module in self.modules:
            await module.initialize()

        await self.state_manager.set_runtime_state("RUNNING")
        self.logger.info("runtime_initialized")

    async def start(self):
        self.running = True
        self.logger.info("runtime_started")

        for module in self.modules:
            task = asyncio.create_task(
                module.start(),
                name=type(module).__name__,
            )
            self._bg_tasks.add(task)
            task.add_done_callback(self._handle_task_result)

        while self.running:
            await asyncio.sleep(1)

    def _handle_task_result(self, task: asyncio.Task):
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.error(
                "background_task_crashed",
                task=task.get_name(),
                error=str(exc),
            )

    async def stop(self):
        self.logger.info("runtime_stopping")
        self.running = False
        await self.state_manager.set_runtime_state("STOPPING")

        for module in reversed(self.modules):
            try:
                await module.stop()
            except Exception as e:
                self.logger.error(
                    "module_stop_error",
                    module=type(module).__name__,
                    error=str(e),
                )

        if self._bg_tasks:
            self.logger.info(
                "runtime_awaiting_tasks",
                count=len(self._bg_tasks),
            )
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        await self.scheduler.shutdown()
        await self.model_client.stop()
        await self.memory_bridge.stop()

        await self.state_manager.set_runtime_state("STOPPED")
        self.logger.info("runtime_stopped")

    # ââ Public API ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    async def think(self, user_input: str) -> str:
        return await self.cognition.think(user_input=user_input)

    async def state(self) -> dict:
        return await self.state_manager.get_full_state()