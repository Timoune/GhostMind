from __future__ import annotations

import asyncio

from core.module_base import GhostModule
from orchestration.event_bus import Event


class AutonomousAgent(GhostModule):
    """
    Background behavior manager.

    Uses Scheduler + EventBus to run autonomous behaviors independently
    of user requests.

    Background behaviors (via Scheduler)
    ──────────────────────────────────────
    • Memory consolidation  — flush recent assistant turns to long-term memory
    • Cognitive state review — watch for anomalies (high CPU/RAM, stuck focus)
    • Idle introspection    — log a self-assessment when the system is quiet

    Event handlers (via EventBus subscriptions)
    ────────────────────────────────────────────
    • high_risk_detected    — log + update state risk level
    • reflection_retry      — log retry reason
    • cognition_complete    — reset risk level after each cycle

    Autonomous behavior management
    ──────────────────────────────
    If autonomous_mode=True (safety.yaml: autonomous_mode: true), the agent
    will proceed through critical-risk plans and can be extended to take
    proactive actions. When False (default), it only observes and logs.
    """

    # Intervals (seconds)
    _CONSOLIDATION_INTERVAL  = 300   # 5 minutes
    _STATE_REVIEW_INTERVAL   = 60    # 1 minute
    _INTROSPECTION_INTERVAL  = 120   # 2 minutes

    def __init__(
        self,
        scheduler,
        event_bus,
        state_manager,
        memory_bridge,
        working_memory,
        logger,
        autonomous_mode: bool = False,
    ):
        self.scheduler = scheduler
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.memory_bridge = memory_bridge
        self.working_memory = working_memory
        self.logger = logger
        self.autonomous_mode = autonomous_mode
        self.running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self):
        # Subscribe to pipeline events
        self.event_bus.subscribe("high_risk_detected",   self._on_high_risk)
        self.event_bus.subscribe("reflection_retry",     self._on_reflection_retry)
        self.event_bus.subscribe("cognition_complete",   self._on_cognition_complete)

        self.logger.info(
            "autonomous_agent_initialized",
            autonomous_mode=self.autonomous_mode,
        )

    async def start(self):
        self.running = True

        # Register recurring background tasks with the scheduler
        await self.scheduler.schedule_repeat(
            self._CONSOLIDATION_INTERVAL,
            self._consolidate_memories,
        )
        await self.scheduler.schedule_repeat(
            self._STATE_REVIEW_INTERVAL,
            self._review_cognitive_state,
        )
        await self.scheduler.schedule_repeat(
            self._INTROSPECTION_INTERVAL,
            self._idle_introspection,
        )

        self.logger.info("autonomous_agent_started")

        while self.running:
            await asyncio.sleep(5)

    async def stop(self):
        self.running = False
        self.logger.info("autonomous_agent_stopped")

    async def health_check(self) -> dict:
        return {
            "status": "ok",
            "running": self.running,
            "autonomous_mode": self.autonomous_mode,
        }

    # ── Background behaviors ──────────────────────────────────────────────────

    async def _consolidate_memories(self):
        """
        Flush the most recent assistant turns from working memory into
        long-term storage via memory_bridge.

        This prevents important context from being lost when the working
        memory window slides past its max_items limit.
        """
        try:
            size = await self.working_memory.size()
            if size == 0:
                return

            self.logger.info(
                "autonomous_consolidation_start",
                working_memory_size=size,
            )

            context = await self.working_memory.get_context()

            # Persist the last few assistant messages to long-term memory
            flushed = 0
            for item in reversed(context[-6:]):
                if item.get("role") == "assistant":
                    await self.memory_bridge.store_memory(
                        content=item["content"],
                        memory_type="consolidated",
                        metadata={"source": "autonomous_consolidation"},
                    )
                    flushed += 1

            self.logger.info(
                "autonomous_consolidation_complete",
                flushed=flushed,
            )

        except Exception as e:
            self.logger.error(
                "autonomous_consolidation_failed",
                error=str(e),
            )

    async def _review_cognitive_state(self):
        """
        Inspect the current cognitive and system state.
        Log warnings if resources are under pressure or
        if the system appears stuck (non-null focus for too long).
        """
        try:
            state = await self.state_manager.get_full_state()
            cog   = state.get("cognitive_state", {})
            sys_s = state.get("system_state", {})

            memory_pct = sys_s.get("memory_usage", 0)
            cpu_pct    = sys_s.get("cpu_usage", 0)
            focus      = cog.get("current_focus")
            risk       = cog.get("risk_level", 0)

            self.logger.info(
                "autonomous_state_review",
                focus_active=focus is not None,
                risk_level=risk,
                memory_pct=memory_pct,
                cpu_pct=cpu_pct,
            )

            if memory_pct > 85:
                self.logger.warning(
                    "autonomous_high_memory_warning",
                    memory_pct=memory_pct,
                )

            if cpu_pct > 90:
                self.logger.warning(
                    "autonomous_high_cpu_warning",
                    cpu_pct=cpu_pct,
                )

        except Exception as e:
            self.logger.error(
                "autonomous_state_review_failed",
                error=str(e),
            )

    async def _idle_introspection(self):
        """
        When current_focus is None (system is idle), log a brief
        self-assessment. Can be extended to trigger proactive behaviors
        (e.g. surfacing unresolved goals) when autonomous_mode is True.
        """
        try:
            state = await self.state_manager.get_full_state()
            focus = state["cognitive_state"].get("current_focus")

            if focus is not None:
                return  # System is active — nothing to do

            last_intent = state["cognitive_state"].get("last_intent", "none")
            confidence  = state["cognitive_state"].get("last_confidence", 1.0)

            self.logger.info(
                "autonomous_idle_introspection",
                last_intent=last_intent,
                last_confidence=confidence,
                autonomous_mode=self.autonomous_mode,
            )

        except Exception as e:
            self.logger.error(
                "autonomous_introspection_failed",
                error=str(e),
            )

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_high_risk(self, event: Event):
        """
        Respond to a high_risk_detected event published by the pipeline.
        Updates state and logs the risk context.
        """
        payload   = event.payload or {}
        risk      = payload.get("risk_level", "unknown")
        objective = payload.get("objective", "")
        warning   = payload.get("warning", "")

        self.logger.warning(
            "autonomous_high_risk_received",
            risk_level=risk,
            objective=objective[:120],
            warning=warning,
        )

        await self.state_manager.update_cognitive_state(
            "risk_level", risk
        )

    async def _on_reflection_retry(self, event: Event):
        """
        Log when the reflection engine determines a retry is needed.
        """
        payload = event.payload or {}
        reason  = payload.get("reason", "unknown")
        notes   = payload.get("notes", "")

        self.logger.info(
            "autonomous_reflection_retry_received",
            reason=reason,
            notes=notes[:120],
        )

    async def _on_cognition_complete(self, event: Event):
        """
        After each full cognition cycle, reset the risk level to neutral.
        """
        await self.state_manager.update_cognitive_state("risk_level", 0)

        self.logger.info("autonomous_cognition_complete_received")
