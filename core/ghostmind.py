"""
core/ghostmind.py
─────────────────
GhostMind v4.4  —  Top-level identity and self-awareness module.

Implements GhostModule so it participates in the Runtime lifecycle:

    initialize() → start() → [running] → stop()

Responsibilities
────────────────
• Own the system's self-model (identity, version, configuration)
• Run background introspection and self-review loops
• Subscribe to system events and react at the meta-cognitive level
• Publish ghostmind_* events back onto the EventBus for downstream modules

Configuration  (configs/ghostmind.yaml)
────────────────────────────────────────
ghostmind.identity.name                    — display name      (default: GhostMind)
ghostmind.identity.version                 — version string    (default: 4.4)
ghostmind.loops.self_review_interval       — seconds           (default: 180)
ghostmind.loops.introspection_interval     — seconds           (default: 90)
ghostmind.loops.max_review_depth           — recursive limit   (default: 3)
ghostmind.health.risk_threshold            — float 0–1         (default: 0.8)
ghostmind.health.memory_pressure_threshold — integer %         (default: 85)
ghostmind.health.auto_respawn_bg_tasks     — bool              (default: true)

Background tasks
────────────────
• ghostmind:self_review     — periodic coherence check across cognitive + system state
• ghostmind:introspection   — meta-cognitive identity/state consistency check

Event subscriptions
────────────────────
• high_risk_detected    — triggers urgent self-review + warning log
• reflection_complete   — logs all reflection quality scores
• reflection_retry      — tracks chronic retry patterns
• cognition_complete    — logs cycle completion
• hitl_gate_triggered   — logs human-approval requests
"""

from __future__ import annotations

import asyncio

from core.module_base import GhostModule
from orchestration.event_bus import Event


class GhostMind(GhostModule):
    """
    Top-level GhostMind module.

    This is the identity and meta-cognitive layer of the system.
    It does not call the LLM directly; it observes, logs, reacts,
    and publishes events to keep every other module informed of
    system-level health from a self-aware perspective.
    """

    VERSION = "4.4"

    # Default loop intervals (seconds) — overridden by ghostmind.yaml
    _DEFAULT_SELF_REVIEW_INTERVAL   = 180
    _DEFAULT_INTROSPECTION_INTERVAL = 90
    _DEFAULT_MAX_REVIEW_DEPTH       = 3

    def __init__(
        self,
        config_loader,
        event_bus,
        state_manager,
        logger,
    ):
        self.config_loader = config_loader
        self.event_bus     = event_bus
        self.state_manager = state_manager
        self.logger        = logger

        self.running     = False
        self._bg_tasks: set[asyncio.Task] = set()

        # ── Identity ──────────────────────────────────────────────────────────
        self.name    = config_loader.get("ghostmind.identity.name",    "GhostMind")
        self.version = config_loader.get("ghostmind.identity.version", self.VERSION)

        # ── Loop configuration ────────────────────────────────────────────────
        self._self_review_interval = config_loader.get(
            "ghostmind.loops.self_review_interval",
            self._DEFAULT_SELF_REVIEW_INTERVAL,
        )
        self._introspection_interval = config_loader.get(
            "ghostmind.loops.introspection_interval",
            self._DEFAULT_INTROSPECTION_INTERVAL,
        )
        self._max_review_depth = config_loader.get(
            "ghostmind.loops.max_review_depth",
            self._DEFAULT_MAX_REVIEW_DEPTH,
        )

        # ── Health thresholds ─────────────────────────────────────────────────
        self._risk_threshold   = config_loader.get(
            "ghostmind.health.risk_threshold", 0.8
        )
        self._mem_pressure_pct = config_loader.get(
            "ghostmind.health.memory_pressure_threshold", 85
        )
        self._auto_respawn     = config_loader.get(
            "ghostmind.health.auto_respawn_bg_tasks", True
        )

        # ── Internal counters (reset on stop) ─────────────────────────────────
        self._review_cycles: int = 0
        self._retry_count: int   = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self):
        """
        Subscribe to EventBus channels and stamp identity into StateManager.
        Called once by Runtime before start().
        """
        self.event_bus.subscribe("high_risk_detected",  self._on_high_risk)
        self.event_bus.subscribe("reflection_complete", self._on_reflection_complete)
        self.event_bus.subscribe("reflection_retry",    self._on_reflection_retry)
        self.event_bus.subscribe("cognition_complete",  self._on_cognition_complete)
        self.event_bus.subscribe("hitl_gate_triggered", self._on_hitl_triggered)

        # Publish identity into shared state so other modules can read it
        await self.state_manager.update_system_state("ghostmind_name",    self.name)
        await self.state_manager.update_system_state("ghostmind_version", self.version)

        self.logger.info(
            "ghostmind_initialized",
            name=self.name,
            version=self.version,
            self_review_interval=self._self_review_interval,
            introspection_interval=self._introspection_interval,
        )

    async def start(self):
        """
        Launch background loops as managed Tasks, then block until stop().
        Each task is tracked and optionally respawned on crash.
        """
        self.running = True

        self._spawn("ghostmind:self_review",   self._self_review_loop())
        self._spawn("ghostmind:introspection", self._introspection_loop())

        self.logger.info("ghostmind_started", version=self.version)

        while self.running:
            await asyncio.sleep(1)

    async def stop(self):
        """
        Cancel all background loops and wait for clean shutdown.
        """
        self.running = False

        for task in list(self._bg_tasks):
            task.cancel()

        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        self.logger.info(
            "ghostmind_stopped",
            review_cycles=self._review_cycles,
            retry_events_seen=self._retry_count,
        )

    async def health_check(self) -> dict:
        return {
            "status":            "ok" if self.running else "stopped",
            "name":              self.name,
            "version":           self.version,
            "running":           self.running,
            "active_bg_tasks":   len(self._bg_tasks),
            "review_cycles":     self._review_cycles,
            "retry_events_seen": self._retry_count,
        }

    # ── Background task management ────────────────────────────────────────────

    def _spawn(self, name: str, coro) -> asyncio.Task:
        """
        Create a named, tracked background task.
        The done-callback handles logging and optional respawn.
        """
        task = asyncio.create_task(coro, name=name)
        self._bg_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task):
        """
        Done-callback for every managed background task.
        Surfaces exceptions and respawns the loop if auto_respawn is enabled.
        """
        self._bg_tasks.discard(task)

        if task.cancelled():
            return

        exc = task.exception()
        if exc is None:
            return  # clean exit

        task_name = task.get_name()
        self.logger.error(
            "ghostmind_bg_task_crashed",
            task=task_name,
            error=str(exc),
        )

        if not self._auto_respawn or not self.running:
            return

        self.logger.info("ghostmind_respawning_task", task=task_name)

        if "self_review" in task_name:
            self._spawn(task_name, self._self_review_loop())
        elif "introspection" in task_name:
            self._spawn(task_name, self._introspection_loop())

    # ── Background loops ──────────────────────────────────────────────────────

    async def _self_review_loop(self):
        """
        Periodic self-review cycle.

        Waits self_review_interval seconds, then evaluates system coherence:
        risk levels, memory pressure, CPU pressure. Publishes a
        ghostmind_self_review_alert event if any threshold is breached.

        On exception the loop raises so _on_task_done can respawn it.
        """
        while self.running:
            try:
                await asyncio.sleep(self._self_review_interval)
                if not self.running:
                    break
                await self._run_self_review()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "ghostmind_self_review_loop_error", error=str(e)
                )
                await asyncio.sleep(10)   # brief back-off before respawn
                raise

    async def _introspection_loop(self):
        """
        Periodic meta-cognitive identity check.

        Validates that the self-model stored in StateManager matches the
        in-memory configuration. Corrects drift without an LLM call.
        """
        while self.running:
            try:
                await asyncio.sleep(self._introspection_interval)
                if not self.running:
                    break
                await self._run_introspection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "ghostmind_introspection_loop_error", error=str(e)
                )
                await asyncio.sleep(10)
                raise

    # ── Core review / introspection logic ────────────────────────────────────

    async def _run_self_review(self, depth: int = 0):
        """
        Inspect cognitive + system state for anomalies.

        depth — recursive call depth guard (capped by _max_review_depth).
        Publishes ghostmind_self_review_alert if any issue is detected.
        """
        if depth > self._max_review_depth:
            self.logger.warning(
                "ghostmind_self_review_depth_limit",
                max_depth=self._max_review_depth,
            )
            return

        try:
            state = await self.state_manager.get_full_state()
            cog   = state.get("cognitive_state", {})
            sys_s = state.get("system_state", {})

            risk        = cog.get("risk_level",    0)
            last_intent = cog.get("last_intent",   "none")
            mem_pct     = sys_s.get("memory_usage", 0)
            cpu_pct     = sys_s.get("cpu_usage",    0)

            issues: list[str] = []

            # Float risk (0–1) from pipeline
            if isinstance(risk, float) and risk >= self._risk_threshold:
                issues.append("elevated_risk")

            if mem_pct > self._mem_pressure_pct:
                issues.append("memory_pressure")

            if cpu_pct > 90:
                issues.append("cpu_pressure")

            self._review_cycles += 1

            self.logger.info(
                "ghostmind_self_review",
                cycle=self._review_cycles,
                depth=depth,
                issues=issues or None,
                last_intent=last_intent,
                risk_level=risk,
                mem_pct=mem_pct,
                cpu_pct=cpu_pct,
            )

            if issues:
                await self.event_bus.publish(Event(
                    event_type="ghostmind_self_review_alert",
                    source="GhostMind",
                    payload={
                        "issues":       issues,
                        "review_depth": depth,
                        "cycle":        self._review_cycles,
                    },
                    priority=2,
                ))

        except Exception as e:
            self.logger.error(
                "ghostmind_self_review_failed",
                depth=depth,
                error=str(e),
            )

    async def _run_introspection(self):
        """
        Cross-check the live self-model against values stored in StateManager.
        Corrects any name/version drift by re-stamping the correct values.
        """
        try:
            state = await self.state_manager.get_full_state()
            sys_s = state.get("system_state", {})

            stored_name    = sys_s.get("ghostmind_name",    "")
            stored_version = sys_s.get("ghostmind_version", "")
            model_loaded   = sys_s.get("model_loaded",      False)
            runtime_state  = state.get("runtime_state",     "UNKNOWN")

            drift: list[str] = []
            if stored_name and stored_name != self.name:
                drift.append(f"name:{stored_name}→{self.name}")
            if stored_version and stored_version != self.version:
                drift.append(f"version:{stored_version}→{self.version}")

            self.logger.info(
                "ghostmind_introspection",
                name=self.name,
                version=self.version,
                runtime_state=runtime_state,
                model_loaded=model_loaded,
                drift=drift or None,
            )

            if drift:
                await self.state_manager.update_system_state(
                    "ghostmind_name", self.name
                )
                await self.state_manager.update_system_state(
                    "ghostmind_version", self.version
                )
                self.logger.warning(
                    "ghostmind_identity_drift_corrected",
                    corrections=drift,
                )

        except Exception as e:
            self.logger.error("ghostmind_introspection_failed", error=str(e))

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_high_risk(self, event: Event):
        """
        React to high_risk_detected from CognitionPipeline.
        Immediately triggers an out-of-schedule self-review at depth=1.
        """
        payload   = event.payload or {}
        risk      = payload.get("risk_level", "unknown")
        objective = payload.get("objective", "")[:120]
        warning   = payload.get("warning",   "")

        self.logger.warning(
            "ghostmind_high_risk_intercepted",
            risk_level=risk,
            objective=objective,
            warning=warning,
        )

        # Urgent review — runs immediately, outside the scheduled loop
        self._spawn(
            "ghostmind:urgent_review",
            self._run_self_review(depth=1),
        )

    async def _on_reflection_complete(self, event: Event):
        """
        Log the full reflection scorecard from each cognition cycle.
        """
        payload = event.payload or {}
        self.logger.info(
            "ghostmind_reflection_received",
            coherence_score      = payload.get("coherence_score",      None),
            hallucination_risk   = payload.get("hallucination_risk",   None),
            reasoning_quality    = payload.get("reasoning_quality",    None),
            planning_quality     = payload.get("planning_quality",     None),
            confidence_alignment = payload.get("confidence_alignment", None),
            requires_retry       = payload.get("requires_retry",       False),
        )

    async def _on_reflection_retry(self, event: Event):
        """
        Track reflection retries. Emit an error log if retries become chronic
        (>= 5 per session), which may indicate model or prompt degradation.
        """
        self._retry_count += 1

        payload = event.payload or {}
        reason  = payload.get("reason", "unknown")
        notes   = payload.get("notes",  "")[:120]

        self.logger.warning(
            "ghostmind_reflection_retry",
            reason=reason,
            notes=notes,
            total_retries=self._retry_count,
        )

        if self._retry_count >= 5:
            self.logger.error(
                "ghostmind_chronic_retry_detected",
                total_retries=self._retry_count,
                note=(
                    "High retry rate may indicate model quality or "
                    "prompt template issues."
                ),
            )

    async def _on_cognition_complete(self, event: Event):
        """
        Acknowledge each full cognition cycle completion.
        """
        payload = event.payload or {}
        self.logger.info(
            "ghostmind_cycle_complete",
            path=payload.get("path",     ""),
            intent=payload.get("intent", ""),
            review_cycles=self._review_cycles,
        )

    async def _on_hitl_triggered(self, event: Event):
        """
        Log whenever a human-in-the-loop approval gate is activated.
        """
        payload   = event.payload or {}
        objective = payload.get("objective", "")[:120]
        risk      = payload.get("risk_level", "unknown")

        self.logger.info(
            "ghostmind_hitl_intercepted",
            objective=objective,
            risk_level=risk,
        )
