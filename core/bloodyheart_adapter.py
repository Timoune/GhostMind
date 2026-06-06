"""
BloodyHeart Adapter for GhostMind v4.5

Provides clean integration so that GhostMind cognition flows through BloodyHeart:

    GhostMind (CognitionPipeline) → BloodyHeart → BigArms (tool execution)

Key responsibilities:
- Publish `tool.execute.request` events when the pipeline needs to call a tool
- Listen for `tool.execute.result` / `tool.execute.failed` from BloodyHeart
- Provide async request-response semantics for tool execution
- Wire events between GhostMind modules and BloodyHeart

Usage:

    from core.bloodyheart_adapter import BloodyHeartAdapter

    adapter = BloodyHeartAdapter(event_bus=core_bus, pipeline=cognition_pipeline)
    adapter.wire()

    # Later in pipeline / step executor:
    result = await adapter.execute_tool(
        tool_name="file_writer",
        tool_version="1.0",
        args={"path": "/tmp/test.txt", "content": "hello"},
        granted_capabilities=[{"tier": "WRITE"}]
    )
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, Optional, List

from cognition.cognition_types import TaskNode
from tracing import TraceContext


class BloodyHeartAdapter:
    """
    Adapter that makes GhostMind use BloodyHeart as the execution layer
    for all tool calls (GhostMind → BloodyHeart → BigArms).
    """

    def __init__(
        self,
        event_bus,
        pipeline=None,
        ghostmind_module=None,
        logger=None,
        default_timeout: float = 120.0,
    ):
        self.event_bus = event_bus
        self.pipeline = pipeline
        self.ghostmind = ghostmind_module
        self.logger = logger or getattr(pipeline, "logger", None)
        self.default_timeout = default_timeout

        # Pending tool executions: correlation_id -> Future
        self._pending_tool_calls: Dict[str, asyncio.Future] = {}

    def wire(self):
        """Wire event subscriptions. Call this once during Runtime initialization."""
        if self.event_bus is None:
            if self.logger:
                self.logger.warning("BloodyHeartAdapter: No event_bus provided, skipping wiring")
            return

        # Listen for results coming back from BloodyHeart
        self.event_bus.subscribe("tool.execute.result", self._on_tool_result)
        self.event_bus.subscribe("tool.execute.failed", self._on_tool_failed)

        if self.logger:
            self.logger.info("BloodyHeartAdapter wired (tool routing via BloodyHeart enabled)")

    # =====================================================================
    # Public API for GhostMind Pipeline / StepExecutor
    # =====================================================================

    async def execute_tool(
        self,
        tool_name: str,
        tool_version: str,
        args: Dict[str, Any],
        granted_capabilities: List[Dict[str, Any]],
        correlation_id: Optional[str] = None,
        timeout: Optional[float] = None,
        requested_by: str = "GhostMind",
        dry_run: bool = False,
        resource_budget: Optional[Dict[str, Any]] = None,
        trace_context: Optional[TraceContext] = None,
    ) -> dict:
        """
        Execute a tool by going through BloodyHeart (GhostMind → BloodyHeart → BigArms).

        Supports distributed tracing via `trace_context`.
        """
        if self.event_bus is None:
            return {
                "success": False,
                "error": "BloodyHeart event_bus not available",
                "tool": tool_name,
            }

        if correlation_id is None:
            correlation_id = f"ghostmind-{uuid.uuid4().hex[:12]}"

        # Create or inherit trace context
        if trace_context is None:
            trace_context = TraceContext.new()

        timeout = timeout or self.default_timeout

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_tool_calls[correlation_id] = future

        try:
            from orchestration.event_bus import Event, Priority

            payload = {
                "correlation_id": correlation_id,
                "tool_name": tool_name,
                "tool_version": tool_version,
                "args": args,
                "granted_capabilities": granted_capabilities,
                "dry_run": dry_run,
                "resource_budget": resource_budget or {},
                "requested_by": requested_by,
                "trace_context": trace_context.to_dict(),
            }

            await self.event_bus.publish(Event(
                event_type="tool.execute.request",
                source="GhostMind",
                payload=payload,
                priority=Priority.P2_AUTONOMOUS,
            ))

            if self.logger:
                self.logger.info(
                    "cross_layer_tool_request",
                    tool=f"{tool_name}@{tool_version}",
                    correlation_id=correlation_id,
                    trace_id=trace_context.trace_id,
                    span_id=trace_context.span_id,
                )

            result = await asyncio.wait_for(future, timeout=timeout)
            return result

        except asyncio.TimeoutError:
            self._pending_tool_calls.pop(correlation_id, None)
            if self.logger:
                self.logger.warning(
                    "cross_layer_tool_timeout",
                    tool=tool_name,
                    correlation_id=correlation_id,
                    trace_id=trace_context.trace_id,
                )
            return {
                "success": False,
                "error": f"Tool execution timed out after {timeout}s",
                "tool": tool_name,
                "correlation_id": correlation_id,
                "trace_id": trace_context.trace_id,
            }
        except Exception as e:
            self._pending_tool_calls.pop(correlation_id, None)
            if self.logger:
                self.logger.error("cross_layer_tool_failed", error=str(e), trace_id=trace_context.trace_id)
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "correlation_id": correlation_id,
                "trace_id": trace_context.trace_id,
            }

    # =====================================================================
    # Event Handlers (results coming back from BloodyHeart)
    # =====================================================================

    async def _on_tool_result(self, event):
        """Handle successful tool result from BloodyHeart."""
        payload = event.payload or {}
        correlation_id = payload.get("correlation_id")

        if not correlation_id or correlation_id not in self._pending_tool_calls:
            return

        future = self._pending_tool_calls.pop(correlation_id)
        if not future.done():
            # The result from BloodyHeart is usually nested under "result"
            result_data = payload.get("result", payload)
            future.set_result(result_data)

    async def _on_tool_failed(self, event):
        """Handle tool failure from BloodyHeart."""
        payload = event.payload or {}
        correlation_id = payload.get("correlation_id")

        if not correlation_id or correlation_id not in self._pending_tool_calls:
            return

        future = self._pending_tool_calls.pop(correlation_id)
        if not future.done():
            future.set_result({
                "success": False,
                "error": payload.get("error", "Unknown tool execution failure"),
                "tool": payload.get("tool_name"),
                "correlation_id": correlation_id,
            })

    # =====================================================================
    # Helper for StepExecutor integration
    # =====================================================================

    async def execute_task_via_bloodyheart(self, task: TaskNode, base_args: dict) -> dict:
        """
        Convenience method for StepExecutor to run a TaskNode through BloodyHeart.
        """
        return await self.execute_tool(
            tool_name=task.tool_scope or task.title,
            tool_version="1.0",  # TODO: get from manifest
            args=base_args,
            granted_capabilities=[],  # TODO: derive from task.risk_level
            correlation_id=task.id,
        )

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