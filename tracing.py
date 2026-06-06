"""
Lightweight Distributed Tracing for Mini Von

Provides TraceContext for end-to-end observability across:
    GhostMind (Cognition) → BloodyHeart (Orchestration) → BigArms (Execution)

Features:
- trace_id (top level request)
- span_id / parent_span_id (hierarchical tracing)
- baggage (key-value context propagation)
- Child span creation
- Serialization for event bus transport
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TraceContext:
    """
    Distributed trace context.

    This object is passed across component boundaries (via event payloads)
    to enable end-to-end observability.
    """
    trace_id: str
    parent_span_id: Optional[str] = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    baggage: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)

    def child(self, name: str = "") -> "TraceContext":
        """Create a child span under the current context."""
        child = TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            baggage={**self.baggage}
        )
        if name:
            child.baggage["span_name"] = name
        return child

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "span_id": self.span_id,
            "baggage": self.baggage,
            "start_time": self.start_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraceContext":
        ctx = cls(
            trace_id=data.get("trace_id") or str(uuid.uuid4())[:12],
            parent_span_id=data.get("parent_span_id"),
            span_id=data.get("span_id") or str(uuid.uuid4())[:12],
            baggage=data.get("baggage", {}),
        )
        ctx.start_time = data.get("start_time", time.perf_counter())
        return ctx

    @classmethod
    def new(cls, trace_id: Optional[str] = None) -> "TraceContext":
        return cls(trace_id=trace_id or str(uuid.uuid4())[:12])

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000

    def __repr__(self):
        return f"TraceContext(trace_id={self.trace_id}, span_id={self.span_id})"


def visualize_trace(summary: dict, width: int = 60, use_color: bool = True) -> str:
    """
    Generate a human-readable text visualization of a trace summary.

    Supports optional ANSI colors (disable with use_color=False).
    """
    if use_color:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
    else:
        RESET = BOLD = CYAN = GREEN = YELLOW = BLUE = MAGENTA = ""

    lines = []

    trace_id = summary.get("trace_id", "unknown")
    total_ms = summary.get("total_duration_ms", 0)
    stage_timings = summary.get("stage_timings_ms", {})
    tool_calls = summary.get("tool_calls", 0)
    successful = summary.get("successful_tool_calls", 0)
    failed = summary.get("failed_tool_calls", 0)
    avg_roundtrip = summary.get("bloodyheart_metrics", {}).get("avg_cross_layer_roundtrip_ms", 0)

    # Header
    header = f"{BOLD}{CYAN}Trace {trace_id}{RESET} ({total_ms}ms)"
    lines.append(header)
    lines.append("─" * min(width, 80))

    # Stage timings with colored bars
    max_time = max(stage_timings.values()) if stage_timings else 1
    for stage, ms in stage_timings.items():
        bar_length = int((ms / max_time) * 20) if max_time > 0 else 0
        bar = f"{BLUE}{'█' * bar_length}{RESET}"
        lines.append(f"├── {stage:<25} {YELLOW}{ms:>6.1f}ms{RESET} {bar}")

    # Tool calls section
    if tool_calls > 0:
        lines.append("")
        status_color = GREEN if failed == 0 else YELLOW
        lines.append(
            f"{BOLD}Tool Calls:{RESET} {tool_calls}  "
            f"{status_color}Success: {successful}{RESET}  "
            f"Failed: {failed}  "
            f"Avg Roundtrip: {avg_roundtrip}ms"
        )

    # BloodyHeart metrics
    bh_metrics = summary.get("bloodyheart_metrics", {})
    if bh_metrics:
        lines.append(
            f"{MAGENTA}BloodyHeart:{RESET} {bh_metrics.get('total_calls', 0)} calls, "
            f"{bh_metrics.get('successful', 0)} success"
        )

    return "\n".join(lines)


def export_trace_json(summary: dict, filepath: Optional[str] = None) -> str:
    """
    Export trace summary as pretty JSON.

    If filepath is given, also writes it to disk.
    Returns the JSON string.
    """
    import json

    json_str = json.dumps(summary, indent=2, default=str)

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)

    return json_str
