"""
BloodyHeart Adapter for GhostMind

Small, optional helper that encapsulates common integration patterns
between GhostMind cognition and the BloodyHeart microkernel.

Usage (when BloodyHeart is ready):

    from core.bloodyheart_adapter import BloodyHeartAdapter

    adapter = BloodyHeartAdapter(
        event_bus=core_bus,
        pipeline=cognition_pipeline,
        ghostmind_module=ghostmind
    )

    adapter.wire()   # subscribes + sets up publishing

This keeps the main GhostMind and pipeline classes clean while providing
a single place for BloodyHeart-specific wiring logic.
"""

from __future__ import annotations
from typing import Any, Optional


class BloodyHeartAdapter:
    """
    Lightweight adapter for BloodyHeart <-> GhostMind integration.
    """

    def __init__(
        self,
        event_bus,
        pipeline=None,
        ghostmind_module=None,
        logger=None,
    ):
        self.event_bus = event_bus
        self.pipeline = pipeline
        self.ghostmind = ghostmind_module
        self.logger = logger or getattr(pipeline, "logger", None)

    def wire(self):
        """Perform all subscriptions and setup."""
        if self.event_bus is None:
            return

        # GhostMind reacts to BloodyHeart governance
        if self.ghostmind:
            self.event_bus.subscribe("safe_mode_changed", self.ghostmind.on_safe_mode_changed)
            # Future: cognitive_budget_updated, etc.

        # Ensure pipeline can publish (already supported via constructor injection)
        if self.pipeline and not getattr(self.pipeline, "event_bus", None):
            self.pipeline.event_bus = self.event_bus

        if self.logger:
            self.logger.info("bloodyheart_adapter_wired")

    async def publish_cognitive_event(self, event_type: str, payload: dict, priority: int = 5):
        """Convenience wrapper for publishing from outside the pipeline."""
        if self.pipeline:
            await self.pipeline._publish(event_type, payload, priority)
        elif self.event_bus:
            from orchestration.event_bus import Event
            event = Event(event_type=event_type, source="BloodyHeartAdapter", payload=payload, priority=priority)
            await self.event_bus.publish(event)

    async def health_snapshot(self) -> dict:
        """Aggregate health from pipeline + ghostmind if available."""
        result = {"adapter": "healthy"}
        if self.pipeline and hasattr(self.pipeline, "health_check"):
            result["pipeline"] = await self.pipeline.health_check()
        if self.ghostmind and hasattr(self.ghostmind, "health_check"):
            result["ghostmind"] = await self.ghostmind.health_check()
        return result
