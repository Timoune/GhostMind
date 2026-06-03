# BloodyHeart Integration for GhostMind

## Purpose
This document describes how GhostMind is prepared to integrate with **BloodyHeart** (the cognitive microkernel / central nervous system of Mini Von) with minimal friction once BloodyHeart development is complete.

GhostMind remains a pure cognition module. BloodyHeart owns orchestration, scheduling, health, governance, Safe Mode, and cross-module observability.

## Current Integration Points (Already Wired)

### 1. Event Publishing (BloodyHeart Observability)
GhostMind now publishes the following events via the injected `EventBus`:

| Event Type                  | When Published                  | Priority | Payload Highlights                              | BloodyHeart Use Case                  |
|-----------------------------|----------------------------------|----------|--------------------------------------------------|---------------------------------------|
| `high_risk_detected`        | Risk gate returns require_hitl or warn_and_confirm | 2 (high) | objective, risk_level, effective_risk_score, unvalidated_high_risk_assumptions | Trigger Safe Mode escalation, audit  |
| `hitl_gate_triggered`       | HITL approval required           | 2        | objective, risk                                  | Human priority queue, notification   |
| `cognition_complete`        | Successful `think()` cycle       | 5        | conversation_id, objective, path, overall_uncertainty, assumptions_tracked, success | Observability, replay, metrics       |
| `reflection_complete` / `reflection_retry` | After reflection (future)     | 3-5      | coherence, assumption_violation_score, requires_retry | Learning loop, chronic failure detection |

**How to subscribe from BloodyHeart**:
```python
event_bus.subscribe("cognition_complete", self._on_cognition_complete)
event_bus.subscribe("high_risk_detected", self._on_high_risk)
```

### 2. Health & Capability Reporting
`CognitionPipeline` and the `GhostMind` core module now implement:

```python
async def health_check(self) -> dict:
    return {
        "status": "healthy",
        "module": "CognitionPipeline",
        "metacognition_enabled": True,
        "uncertainty_tracking": True,
        "real_engines": True,
        ...
    }
```

BloodyHeart can call this during capability propagation and health monitoring.

### 3. Dependency & Lifecycle
- GhostMind accepts `event_bus` in its constructor (already present in `Runtime` and `CognitionPipeline`).
- When BloodyHeart is ready, pass the central `EventBus` instance during `Runtime` / `GhostMind` initialization.
- GhostMind will automatically publish to it.

### 4. Governance Awareness (Future-Ready)
- `DecisionEngine` and risk gating already produce `effective_risk_score` that can be influenced by BloodyHeart Safe Mode level (L1âL4) if BloodyHeart publishes a `safe_mode_changed` event.
- Placeholder exists to respect cognitive budgets once BloodyHeart exposes them.

## Recommended Wiring (When BloodyHeart is Ready)

In your BloodyHeart orchestration code (example):

```python
# After creating EventBus and modules
ghostmind = GhostMind(
    config_loader=...,
    event_bus=core_bus,           # â pass the BloodyHeart EventBus
    state_manager=...,
    ...
)

# Subscribe GhostMind to BloodyHeart governance events
core_bus.subscribe("safe_mode_changed", ghostmind.on_safe_mode_changed)
core_bus.subscribe("cognitive_budget_updated", ghostmind.on_budget_update)

# GhostMind will automatically publish its cognitive events back
```

## Non-Goals / Boundaries
- GhostMind does **not** perform scheduling, resource governance, or Safe Mode enforcement itself.
- BloodyHeart remains the single source of truth for module health, capability graph, and priority queues.
- Personality (`DarkPassenger`) and execution (`BigArms`) stay completely decoupled.

## Testing
The test suite (`tests/full_test_suite_v1.py`) includes pipeline execution that exercises the new `_publish` paths (via mock `event_bus` if injected).

## Implemented Enhancements (June 2026)

All four requested items have been added:

1. **`reflection_complete` / `reflection_retry` publishing**  
   Published automatically after every reflection step in `CognitionPipeline.think()`. Includes coherence, assumption violation score, calibration error, and counts of validated/violated assumptions.

2. **`on_safe_mode_changed` handler in `core/ghostmind.py`**  
   Subscribed automatically. Updates internal `_safe_mode_level` and can reduce `autonomous_mode` on L3/L4. Extend the handler for deeper behavioral changes.

3. **Richer `reasoning_trace` as event payload**  
   `cognition_complete` and reflection events now include:
   - `uncertainty_evolution` across stages
   - `high_risk_assumptions` list
   - Full reflection metrics
   - `reasoning_trace` is also stored in `CognitionRecord`

4. **`core/bloodyheart_adapter.py`**  
   Small helper class that encapsulates wiring, health aggregation, and convenient publishing. Use it for clean, one-line integration when BloodyHeart is ready.

## Recommended Wiring (Updated)

```python
from core.bloodyheart_adapter import BloodyHeartAdapter

adapter = BloodyHeartAdapter(
    event_bus=your_bloodyheart_bus,
    pipeline=cognition_pipeline,
    ghostmind_module=ghostmind
)
adapter.wire()
```

GhostMind is now **fully instrumented** for BloodyHeart.
