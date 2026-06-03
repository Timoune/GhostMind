"""
core/ghostmind.py
─────────────────
GhostMind v4.5  —  Top-level identity and self-awareness module.

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
ghostmind.identity.version                 — version string    (default: 4.5)
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

    VERSION = "4.5"

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
        self.event_bus.subscribe("safe_mode_changed",   self.on_safe_mode_changed)  # BloodyHeart governance

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


    # ── BloodyHeart Governance Handler ─────────────────────────────────────────

    def on_safe_mode_changed(self, event):
        """
        Called when BloodyHeart changes the system Safe Mode level (L1–L4).
        Updates internal state and can adjust autonomy / logging aggressiveness.
        """
        try:
            level = event.payload.get("level", "L1")
            reason = event.payload.get("reason", "")
            self._safe_mode_level = level

            self.logger.warning(
                "ghostmind_safe_mode_changed",
                level=level,
                reason=reason,
            )

            # Example behavioral adjustment (extend as needed)
            if level in ("L3", "L4"):
                self.autonomous_mode = False  # reduce autonomy on high restriction
            else:
                self.autonomous_mode = getattr(self, "autonomous_mode", False)

        except Exception as e:
            self.logger.error("safe_mode_handler_failed", error=str(e))
