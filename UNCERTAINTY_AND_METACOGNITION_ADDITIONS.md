# GhostMind v4.4+ — Uncertainty Quantification & Metacognition Additions

## Scientific Rationale (as implemented)

These additions address two core limitations in current LLM-centric cognitive agents:

1. **Poor uncertainty calibration** — LLMs rarely express well-calibrated confidence. Explicit structured uncertainty (epistemic vs aleatoric) + propagation allows better risk gating and deferral.

2. **Hidden assumptions** — Most reasoning failures originate in unexamined premises. Explicit assumption extraction + validation is a direct implementation of metacognition (monitoring one's own reasoning process).

Both are well-supported in cognitive science (metacognition research) and modern AI safety/agent literature (uncertainty-aware planning, assumption-based verification).

## What Was Added

### 1. Extended Type System (`cognition/cognition_types.py`)
- New `Uncertainty` dataclass with `epistemic`, `aleatoric`, `overall`, and `justification`.
- New `Assumption` dataclass with source stage, confidence, validation status, and `risk_if_false`.
- All major result types (`IntentAnalysis`, `DecompositionResult`, `ExecutionPlan`, `TaskNode`, `AgentResult`, `MultiAgentResult`, `ReflectionResult`, `CognitionRecord`) now carry `uncertainty` and `assumptions` (or aggregated versions).

This makes uncertainty and assumptions first-class, queryable, and propagatable through the entire cognition pipeline.

### 2. New `MetaReasoner` Module (`cognition/meta_reasoner.py`)
A dedicated metacognitive component with:
- `extract_assumptions(stage, content)` — LLM-driven extraction of implicit assumptions (returns structured `Assumption` list).
- `validate_assumptions(...)` — Memory/context or heuristic validation (extendable to full counterfactual or DreamCloud lookup).
- `enrich_with_uncertainty(...)` — Deterministic adjustment of overall uncertainty based on volume and severity of unvalidated high-risk assumptions.
- `process_stage(...)` — Convenience method that combines the above.

The module is model-agnostic (uses existing `ModelClient`) and designed to be called after each major reasoning stage.

### 3. Uncertainty-Aware `DecisionEngine` (`cognition/decision_engine.py`)
- `decide_path()` now factors `intent.uncertainty.overall` — high uncertainty pushes complex tasks toward `multi_step` for extra verification.
- `gate_on_risk()` computes an *effective risk score* that combines:
  - Base plan risk
  - Uncertainty contribution
  - Penalty from unvalidated high/critical-risk assumptions
- This produces more conservative and better-calibrated HITL / confirmation decisions.

## Integration Points (Recommended Next Steps)

In `cognition/pipeline.py` (orchestration layer), after each stage you would do something like:

```python
# After intent analysis
meta = await self.meta_reasoner.process_stage("intent", intent.reasoning, intent.uncertainty)
intent.assumptions = meta["assumptions"]
intent.uncertainty = meta["uncertainty"]

# Similarly after decomposition and planning...
# Then pass enriched objects to DecisionEngine and ReflectionEngine
```

In `reflection_engine.py`:
- Add computation of `assumption_violation_score` and `uncertainty_calibration_error`.
- Use validated/violated assumptions to decide `requires_retry`.

In `CognitionRecord` and `DecisionLedger`:
- Persist full `overall_uncertainty` and `tracked_assumptions` for audit and future adaptive learning.

## Benefits
- **Safety**: Better calibrated risk gating → fewer missed high-risk plans.
- **Robustness**: Explicit assumption tracking reduces hidden-failure modes.
- **Explainability**: Structured traces now include uncertainty evolution and assumption provenance.
- **Future-proof**: Foundation for causal sketches, counterfactual simulation, and guarded experience-based strategy adaptation.

All changes are fully backward-compatible with existing configs and non-uncertainty-aware code paths (defaults are conservative).

## Files Modified / Added in this increment
- `cognition/cognition_types.py` (extended)
- `cognition/meta_reasoner.py` (new)
- `cognition/decision_engine.py` (extended)
- `cognition/__init__.py` (export update)

These form a solid, minimal, scientifically grounded foundation for the higher-level additions (causal world sketches, structured reasoning traces, adaptive strategy selection) discussed previously.
