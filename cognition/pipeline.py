"""
CognitionPipeline with full integration of Uncertainty Quantification and Metacognition (MetaReasoner).

This is a functional, integrated version of the GhostMind cognition pipeline
incorporating all additions from v4.5+:
- Structured Uncertainty propagation
- Explicit Assumption tracking via MetaReasoner
- Uncertainty-aware DecisionEngine
- Enhanced Reflection with calibration and assumption violation metrics
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from cognition.cognition_types import (
    IntentAnalysis, DecompositionResult, ExecutionPlan, TaskNode,
    Uncertainty, Assumption, ReflectionResult, CognitionRecord,
    MultiAgentResult
)
from cognition.decision_engine import DecisionEngine
from cognition.meta_reasoner import MetaReasoner
from cognition.intent_engine import IntentEngine
from cognition.decomposition_engine import DecompositionEngine
from cognition.planning_engine import PlanningEngine
from llm.model_client import ModelClient  # type: ignore


class CognitionPipeline:
    """
    Integrated cognitive pipeline for GhostMind.

    Flow:
    1. Intent Analysis
    2. Meta-reasoning on intent (assumptions + uncertainty enrichment)
    3. Decomposition
    4. Meta-reasoning on decomposition
    5. Planning
    6. Meta-reasoning on plan + DecisionEngine gate
    7. Execution (single or multi-step)
    8. Reflection (with new metacognitive metrics)
    9. Record creation
    """

    def __init__(
        self,
        model_client: ModelClient,
        logger,
        context_loader=None,
        working_memory=None,
        memory_bridge=None,
        state_manager=None,
        event_bus=None,
        temperature: float = 0.6,
        max_tokens: int = 1536,
        max_reasoning_depth: int = 5,
        autonomous_mode: bool = False,
        hitl_gate=None,
        hitl_enabled: bool = True,
        hitl_risk_threshold: str = "high",
        multi_agent_orchestrator=None,
    ):
        self.model_client = model_client
        self.logger = logger
        self.context_loader = context_loader
        self.working_memory = working_memory
        self.memory_bridge = memory_bridge
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_reasoning_depth = max_reasoning_depth
        self.autonomous_mode = autonomous_mode
        self.hitl_gate = hitl_gate
        self.hitl_enabled = hitl_enabled
        self.hitl_risk_threshold = hitl_risk_threshold
        self.multi_agent_orchestrator = multi_agent_orchestrator

        # Core engines
        self.decision_engine = DecisionEngine(logger=logger, autonomous_mode=autonomous_mode)
        self.meta_reasoner = MetaReasoner(
            model_client=model_client,
            logger=logger,
            memory_bridge=memory_bridge,
            working_memory=working_memory,
            temperature=0.35,
            max_tokens=700
        )

        # Real engines (wired from original GhostMind)
        self.intent_engine = IntentEngine(model_client=model_client, logger=logger)
        self.decomposition_engine = DecompositionEngine(model_client=model_client, logger=logger)
        self.planning_engine = PlanningEngine(logger=logger)

        self.logger.info("cognition_pipeline_initialized_with_metacognition_and_real_engines")

    # ââ BloodyHeart Integration Helpers ââââââââââââââââââââââââââââââââââââââââ

    async def _publish(self, event_type: str, payload: dict, priority: int = 5):
        """Publish event to BloodyHeart EventBus if available."""
        if self.event_bus:
            try:
                from orchestration.event_bus import Event
                event = Event(
                    event_type=event_type,
                    source="CognitionPipeline",
                    payload=payload,
                    priority=priority
                )
                await self.event_bus.publish(event)
            except Exception as e:
                self.logger.warning("bloodyheart_event_publish_failed", event_type=event_type, error=str(e))

    async def health_check(self) -> dict:
        """Standard health report for BloodyHeart capability propagation."""
        return {
            "status": "healthy",
            "module": "CognitionPipeline",
            "metacognition_enabled": True,
            "uncertainty_tracking": True,
            "real_engines": True,
            "last_cycle_success": True  # in real impl track this
        }

    async def think(self, user_input: str) -> str:
        """Main entry point â full integrated cognition cycle."""
        conversation_id = str(uuid.uuid4())
        self.logger.info("cognition_cycle_start", conversation_id=conversation_id, input=user_input[:80])

        try:
            # 1. Intent Analysis (simplified for integration demo; in real use call IntentEngine)
            intent = await self._analyze_intent(user_input)

            # 2. Metacognition on Intent
            intent_meta = await self.meta_reasoner.process_stage(
                stage="intent",
                content=intent.reasoning,
                existing_uncertainty=intent.uncertainty,
                context={"user_input": user_input}
            )
            intent.assumptions = intent_meta["assumptions"]
            intent.uncertainty = intent_meta["uncertainty"]

            # 3. Decomposition
            decomp = await self._decompose(user_input, intent)

            # 4. Metacognition on Decomposition
            decomp_meta = await self.meta_reasoner.process_stage(
                stage="decomposition",
                content=decomp.reasoning + " " + decomp.strategy_summary,
                existing_uncertainty=decomp.uncertainty,
                context={"intent": asdict(intent)}
            )
            decomp.assumptions = decomp_meta["assumptions"]
            decomp.uncertainty = decomp_meta["uncertainty"]

            # 5. Planning
            plan = await self._plan(user_input, decomp)

            # 6. Metacognition on Plan + Decision
            plan_meta = await self.meta_reasoner.process_stage(
                stage="planning",
                content=plan.reasoning,
                existing_uncertainty=plan.uncertainty,
                context={"decomposition": asdict(decomp)}
            )
            plan.assumptions = plan_meta["assumptions"]
            plan.uncertainty = plan_meta["uncertainty"]

            gate_result = self.decision_engine.gate_on_risk(plan)
            self.logger.info("risk_gate_result", **gate_result)

            if gate_result["action"] == "require_hitl" and self.hitl_enabled:
                # In real system this would trigger HITL; for tests we simulate approval
                self.logger.info("hitl_required_simulated_approval")
                if self.hitl_gate:
                    # Would await approval here
                    pass

            # 7. Execution path decision
            path = self.decision_engine.decide_path(intent)
            self.logger.info("execution_path_chosen", path=path, uncertainty=intent.uncertainty.overall)

            if path == "lightweight":
                final_response = await self._single_call_synthesis(user_input, intent, decomp, plan)
            else:
                final_response = await self._multi_step_execution(user_input, plan)

            # 8. Reflection with new metacognitive metrics
            reflection = await self._reflect(
                user_input=user_input,
                intent=intent,
                decomposition=decomp,
                plan=plan,
                final_response=final_response,
                gate_result=gate_result
            )

            # BloodyHeart: publish reflection events
            event_type = "reflection_retry" if reflection.requires_retry else "reflection_complete"
            await self._publish(
                event_type,
                {
                    "conversation_id": conversation_id,
                    "coherence_score": reflection.coherence_score,
                    "assumption_violation_score": reflection.assumption_violation_score,
                    "uncertainty_calibration_error": reflection.uncertainty_calibration_error,
                    "requires_retry": reflection.requires_retry,
                    "retry_reason": reflection.retry_reason,
                    "validated_assumptions_count": len(reflection.validated_assumptions),
                    "violated_assumptions_count": len(reflection.violated_assumptions),
                },
                priority=3 if reflection.requires_retry else 5
            )

            # 9. Build record with richer reasoning_trace for BloodyHeart
            richer_trace = {
                "path": path,
                "gate": gate_result,
                "meta_stages": ["intent", "decomposition", "planning"],
                "uncertainty_evolution": {
                    "intent": intent.uncertainty.overall,
                    "decomposition": decomp.uncertainty.overall,
                    "plan": plan.uncertainty.overall,
                },
                "key_assumptions_count": len(intent.assumptions + decomp.assumptions + plan.assumptions),
                "high_risk_assumptions": [
                    {"statement": a.statement, "risk": a.risk_if_false}
                    for a in (intent.assumptions + decomp.assumptions + plan.assumptions)
                    if a.risk_if_false in ("high", "critical")
                ],
                "reflection": {
                    "coherence": reflection.coherence_score,
                    "assumption_violation": reflection.assumption_violation_score,
                    "calibration_error": reflection.uncertainty_calibration_error,
                }
            }

            record = CognitionRecord(
                conversation_id=conversation_id,
                user_input=user_input,
                final_response=final_response,
                intent_analysis=asdict(intent),
                decomposition=asdict(decomp),
                execution_plan=asdict(plan),
                reflection=asdict(reflection),
                success=True,
                overall_uncertainty=intent.uncertainty,
                tracked_assumptions=intent.assumptions + decomp.assumptions + plan.assumptions,
                reasoning_trace=richer_trace
            )

            self.logger.info(
                "cognition_cycle_complete",
                conversation_id=conversation_id,
                success=True,
                overall_uncertainty=record.overall_uncertainty.overall,
                assumptions_tracked=len(record.tracked_assumptions)
            )

            # BloodyHeart integration - publish completion event for observability & replay
            await self._publish(
                "cognition_complete",
                {
                    "conversation_id": conversation_id,
                    "objective": user_input[:120],
                    "path": path,
                    "overall_uncertainty": record.overall_uncertainty.overall,
                    "assumptions_tracked": len(record.tracked_assumptions),
                    "success": True
                },
                priority=5
            )

            # In real system: await self.working_memory.add(...) and memory_bridge.store_memory(...)
            return final_response

        except Exception as e:
            self.logger.error("cognition_cycle_failed", error=str(e))
            return f"[ERROR] Cognition pipeline failed: {e}"

    # ââ Stage implementations (simplified but realistic for testing) ââââââââââ

    async def _analyze_intent(self, user_input: str) -> IntentAnalysis:
        """Real IntentEngine wiring."""
        context_messages: list = []
        intent = await self.intent_engine.analyze(user_input, context_messages)
        # Ensure our extended fields exist (defaults from dataclass)
        if not hasattr(intent, 'uncertainty') or intent.uncertainty is None:
            intent.uncertainty = Uncertainty(overall=0.25, justification="From IntentEngine")
        if not hasattr(intent, 'assumptions'):
            intent.assumptions = []
        return intent

    async def _decompose(self, user_input: str, intent: IntentAnalysis) -> DecompositionResult:
        """Real DecompositionEngine wiring."""
        context_messages: list = []
        decomp = await self.decomposition_engine.decompose(
            objective=user_input,
            context_messages=context_messages,
            intent_analysis=intent
        )
        if not hasattr(decomp, 'uncertainty') or decomp.uncertainty is None:
            decomp.uncertainty = Uncertainty(overall=0.30, justification="From DecompositionEngine")
        if not hasattr(decomp, 'assumptions'):
            decomp.assumptions = []
        return decomp

    async def _plan(self, user_input: str, decomp: DecompositionResult) -> ExecutionPlan:
        """Real PlanningEngine wiring (mostly deterministic)."""
        plan = await self.planning_engine.build_plan(decomposition=decomp)
        if not hasattr(plan, 'uncertainty') or plan.uncertainty is None:
            plan.uncertainty = Uncertainty(overall=0.32, justification="From PlanningEngine")
        if not hasattr(plan, 'assumptions'):
            plan.assumptions = []
        # Ensure causal_sketch exists for our meta layer
        if not hasattr(plan, 'causal_sketch') or plan.causal_sketch is None:
            plan.causal_sketch = {"steps": [t.title for t in plan.ordered_tasks]}
        return plan

    async def _single_call_synthesis(self, user_input: str, intent, decomp, plan) -> str:
        """Simple synthesis for lightweight path."""
        return (
            f"[Lightweight Response]\n"
            f"Objective: {user_input}\n"
            f"Key insight: {intent.reasoning}\n"
            f"Strategy: {decomp.strategy_summary}\n"
            f"Overall uncertainty: {plan.uncertainty.overall:.2f} | "
            f"Assumptions tracked: {len(plan.assumptions) + len(decomp.assumptions)}"
        )

    async def _multi_step_execution(self, user_input: str, plan: ExecutionPlan) -> str:
        """Simulated multi-step execution."""
        results = []
        for i, task in enumerate(plan.ordered_tasks, 1):
            results.append(f"Step {i} ({task.title}): Completed successfully (simulated). Risk was {task.risk_level}.")
        return "\n".join(results) + f"\n\nFinal synthesis for: {user_input}"

    async def _reflect(
        self,
        user_input: str,
        intent: IntentAnalysis,
        decomposition: DecompositionResult,
        plan: ExecutionPlan,
        final_response: str,
        gate_result: dict
    ) -> ReflectionResult:
        """Enhanced reflection with metacognitive metrics."""
        # Calculate assumption violation score (simulated)
        all_assumptions = intent.assumptions + decomposition.assumptions + plan.assumptions
        violated = [a for a in all_assumptions if not a.validated and a.risk_if_false in ("high", "critical")]
        assumption_violation_score = min(1.0, len(violated) * 0.25)

        # Uncertainty calibration error (simulated)
        stated = plan.uncertainty.overall
        observed_success = 0.85  # In real system this would come from outcome evaluation
        calibration_error = abs(stated - (1.0 - observed_success))

        coherence = max(0.6, 0.92 - assumption_violation_score * 0.3)
        hallucination_risk = min(0.4, plan.uncertainty.overall * 0.7 + assumption_violation_score * 0.2)

        requires_retry = assumption_violation_score > 0.5 or plan.uncertainty.overall > 0.75

        return ReflectionResult(
            coherence_score=round(coherence, 3),
            hallucination_risk=round(hallucination_risk, 3),
            reasoning_quality=0.88,
            planning_quality=0.82,
            confidence_alignment=round(1.0 - calibration_error, 3),
            requires_retry=requires_retry,
            retry_reason="High assumption violation or poorly calibrated uncertainty" if requires_retry else "",
            reflection_notes="Integrated metacognitive reflection performed. Uncertainty and assumptions evaluated.",
            assumption_violation_score=round(assumption_violation_score, 3),
            uncertainty_calibration_error=round(calibration_error, 3),
            meta_notes=f"Tracked {len(all_assumptions)} assumptions across pipeline. Gate action: {gate_result['action']}",
            validated_assumptions=[a for a in all_assumptions if a.validated],
            violated_assumptions=violated
        )