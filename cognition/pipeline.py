from __future__ import annotations

import re
import time
import uuid
import dataclasses
from typing import Optional, List, Dict, Any

from cognition.intent_engine        import IntentEngine
from cognition.decomposition_engine import DecompositionEngine
from cognition.planning_engine      import PlanningEngine
from cognition.reflection_engine    import ReflectionEngine
from cognition.decision_engine      import DecisionEngine, RISK_ORDER
from cognition.step_executor        import StepExecutor
from cognition.decision_ledger      import DecisionLedger
from cognition.cognition_record     import CognitionRecord
from cognition.cognition_types      import MultiAgentResult
from orchestration.event_bus        import Event


GHOSTMIND_SYSTEM_PROMPT = """\
You are GhostMind — the reasoning core of Mini Von, an autonomous AI assistant.

Your responsibilities:
- Parse intent precisely before acting
- Decompose complex tasks into clear steps
- Reason carefully and acknowledge uncertainty
- Be direct, concise, and honest

You have access to:
- Working memory: the current session's conversation history
- Long-term memory: relevant past context retrieved from DreamCloud

Always think before you respond. For multi-step tasks, outline the steps first.
"""

# ── Multi-agent trigger detection ──────────────────────────────────────────────
_MULTI_AGENT_RE = re.compile(
    r"\b("
    r"multi[- ]?agent|multiple agents?|use agents?|"
    r"agent collaboration|agent reasoning|"
    r"with (multiple )?agents?|run (multiple )?agents?|"
    r"spawn agents?|use multi[- ]?agent|"
    r"collaborative reasoning|parallel agents?"
    r")\b",
    re.IGNORECASE,
)


def _serialize(obj) -> dict:
    """Recursively convert dataclasses to plain dicts for JSON storage."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


class CognitionPipeline:
    """
    Full cognition pipeline.

    Stages
    ──────
    0. Multi-agent detection  — if triggered, route to MultiAgentOrchestrator
    1. Context loading
    2. Intent analysis        (IntentEngine)
    3. Path selection         (DecisionEngine.decide_path)
    4. Task decomposition     (DecompositionEngine)          — executive / multi_step
    5. Planning + ordering    (PlanningEngine + DecisionEngine.prioritize_goals)
    6. Risk gating            (DecisionEngine.gate_on_risk)
       6a. HITL gate          — pause for human approval on high/critical risk
    7. Strategy selection     (DecisionEngine.select_strategy)
    8. Execution              (StepExecutor OR single LLM call)
    9. Reflection             (ReflectionEngine)
    10. Memory persistence
    11. Decision ledger
    12. Event publishing
    """

    def __init__(
        self,
        model_client,
        context_loader,
        working_memory,
        memory_bridge,
        state_manager,
        logger,
        event_bus=None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_reasoning_depth: int = 5,
        autonomous_mode: bool = False,
        # ── Human-in-the-loop ──────────────────────────────────────────
        hitl_gate=None,
        hitl_enabled: bool = False,
        hitl_risk_threshold: str = "high",
        # ── Multi-agent ────────────────────────────────────────────────
        multi_agent_orchestrator=None,
    ):
        self.model_client        = model_client
        self.context_loader      = context_loader
        self.working_memory      = working_memory
        self.memory_bridge       = memory_bridge
        self.state_manager       = state_manager
        self.logger              = logger
        self.event_bus           = event_bus
        self.temperature         = temperature
        self.max_tokens          = max_tokens
        self.max_reasoning_depth = max_reasoning_depth
        self.autonomous_mode     = autonomous_mode

        # HITL
        self.hitl_gate           = hitl_gate
        self.hitl_enabled        = hitl_enabled
        self.hitl_risk_threshold = RISK_ORDER.get(hitl_risk_threshold, RISK_ORDER["high"])

        # Multi-agent
        self.multi_agent_orchestrator = multi_agent_orchestrator

        # ── Engines ───────────────────────────────────────────────────────────
        self.intent_engine = IntentEngine(
            model_client=self.model_client,
            logger=self.logger,
        )
        self.decomposition_engine = DecompositionEngine(
            model_client=self.model_client,
            logger=self.logger,
        )
        self.planning_engine = PlanningEngine(
            logger=self.logger,
        )
        self.reflection_engine = ReflectionEngine(
            model_client=self.model_client,
            logger=self.logger,
        )
        self.decision_engine = DecisionEngine(
            logger=self.logger,
            autonomous_mode=self.autonomous_mode,
        )
        self.bloodyheart_adapter = None
        self.step_executor = StepExecutor(
            model_client=self.model_client,
            logger=self.logger,
            max_tokens=self.max_tokens,
            bloodyheart_adapter=None,
        )
        self.decision_ledger = DecisionLedger(
            logger=self.logger,
        )

    def set_bloodyheart_adapter(self, adapter):
        """Automatically called by Runtime to enable GhostMind → BloodyHeart → BigArms flow."""
        self.bloodyheart_adapter = adapter
        if self.step_executor:
            self.step_executor.bloodyheart_adapter = adapter

    def _record_trace_event(self, event_type: str, data: Dict[str, Any]):
        """Record an event for trace summary generation."""
        if not hasattr(self, "_trace_events"):
            self._trace_events = []
        self._trace_events.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        })

    def get_trace_summary(self) -> Dict[str, Any]:
        """Generate a summary of the last cognition trace."""
        if not hasattr(self, "_trace_events") or not self._trace_events:
            return {"trace_id": None, "events": 0}

        trace_id = getattr(self._current_trace_context, "trace_id", None) if hasattr(self, "_current_trace_context") else None
        duration = (time.perf_counter() - getattr(self, "_trace_start_time", time.perf_counter())) * 1000

        tool_calls = [e for e in self._trace_events if "tool" in e["type"].lower()]
        successful_tools = sum(1 for e in tool_calls if e["data"].get("success", True))

        summary = {
            "trace_id": trace_id,
            "total_duration_ms": round(duration, 1),
            "event_count": len(self._trace_events),
            "tool_calls": len(tool_calls),
            "successful_tool_calls": successful_tools,
            "failed_tool_calls": len(tool_calls) - successful_tools,
            "key_events": [e["type"] for e in self._trace_events[:10]],  # First 10 for brevity
            "bloodyheart_metrics": self.step_executor.get_bloodyheart_metrics() if hasattr(self.step_executor, "get_bloodyheart_metrics") else {},
        }

        return summary

    def visualize_last_trace(self, use_color: bool = True) -> str:
        """Return a nice text visualization of the last cognition trace."""
        from tracing import visualize_trace
        summary = self.get_trace_summary()
        return visualize_trace(summary, use_color=use_color)

    def export_last_trace_json(self, filepath: Optional[str] = None) -> str:
        """Export the last trace summary as JSON (optionally write to file)."""
        from tracing import export_trace_json
        summary = self.get_trace_summary()
        return export_trace_json(summary, filepath=filepath)

    # ── Main entry point ──────────────────────────────────────────────────────

    async def think(self, user_input: str) -> str:

        from tracing import TraceContext

        # Top-level trace context for distributed tracing across GhostMind → BloodyHeart → BigArms
        trace_context = TraceContext.new()
        self._current_trace_context = trace_context

        # Propagate to StepExecutor
        if hasattr(self, "step_executor") and self.step_executor:
            setattr(self.step_executor, "_current_trace_context", trace_context)

        self.logger.info(
            "cognition_start",
            input_preview=user_input[:100],
            trace_id=trace_context.trace_id,
        )

        self._record_trace_event("cognition_start", {"input": user_input[:120]})

        await self.state_manager.update_cognitive_state(
            "current_focus", user_input[:200]
        )

        # ── 0. Multi-agent detection ───────────────────────────────────────
        if self._is_multi_agent_request(user_input) and self.multi_agent_orchestrator:
            return await self._think_multi_agent(user_input)

        # ── 1. Context ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        context  = await self.context_loader.build_context(user_input)
        stage_timings["context_build"] = (time.perf_counter() - t0) * 1000
        messages = self._build_messages(user_input, context)

        # ── 2. Intent analysis ─────────────────────────────────────────────
        t0 = time.perf_counter()
        intent = await self.intent_engine.analyze(
            user_input=user_input,
            context_messages=messages,
        )
        stage_timings["intent_analysis"] = (time.perf_counter() - t0) * 1000

        # ── 3. Path selection ──────────────────────────────────────────────
        path = self.decision_engine.decide_path(intent)

        decomposition  = None
        execution_plan = None
        strategy       = "single_call"

        # ── 4-7. Executive / multi-step path ──────────────────────────────
        if path in ("executive", "multi_step"):

            self.logger.info("executive_cognition_enabled", path=path)

            # 4. Decompose
            decomposition = await self.decomposition_engine.decompose(
                objective=user_input,
                context_messages=messages,
                intent_analysis=intent,
            )

            # 5. Plan
            decomposition.tasks = self.decision_engine.prioritize_goals(
                decomposition.tasks
            )
            execution_plan = await self.planning_engine.build_plan(
                decomposition=decomposition,
            )

            # 6. Risk gate
            risk_decision = self.decision_engine.gate_on_risk(execution_plan)

            # ── 6a. HITL gate (high / critical risk) ──────────────────────
            if (
                self.hitl_enabled
                and self.hitl_gate is not None
                and not self.autonomous_mode
                and RISK_ORDER.get(risk_decision["risk_level"], 0)
                    >= self.hitl_risk_threshold
            ):
                task_titles = [t.title for t in execution_plan.ordered_tasks]

                self.logger.info(
                    "hitl_gate_triggered",
                    risk_level=risk_decision["risk_level"],
                    task_count=len(task_titles),
                )

                await self._publish("hitl_gate_triggered", {
                    "risk_level": risk_decision["risk_level"],
                    "objective":  execution_plan.objective,
                }, priority=1)

                approval = await self.hitl_gate.request_approval(
                    risk_level=risk_decision["risk_level"],
                    objective=execution_plan.objective,
                    warning=risk_decision.get("warning", ""),
                    task_titles=task_titles,
                )

                if not approval["approved"]:
                    reason = approval.get("reason", "denied")
                    self.logger.info(
                        "hitl_gate_denied", reason=reason
                    )
                    # Return a brief denial — GUI handles UI cleanup
                    denial_msg = (
                        f"⛔ Action paused by human review (reason: {reason}). "
                        "Please refine your request or lower the scope."
                    )
                    await self.state_manager.update_cognitive_state(
                        "current_focus", None
                    )
                    return denial_msg

                self.logger.info("hitl_gate_approved")

            # Normal risk flow (halt on critical without HITL)
            if not risk_decision["proceed"]:
                warning = risk_decision["warning"]
                self.logger.warning(
                    "cognition_halted_by_risk",
                    risk_level=risk_decision["risk_level"],
                )
                await self._publish("high_risk_detected", {
                    "risk_level": risk_decision["risk_level"],
                    "objective":  execution_plan.objective,
                    "warning":    warning,
                }, priority=1)
                await self.state_manager.update_cognitive_state(
                    "current_focus", None
                )
                return warning

            if risk_decision["warning"]:
                await self._publish("high_risk_detected", {
                    "risk_level": risk_decision["risk_level"],
                    "objective":  execution_plan.objective,
                    "warning":    risk_decision["warning"],
                }, priority=2)

            # 7. Strategy selection
            strategy = self.decision_engine.select_strategy(
                plan=execution_plan,
                path=path,
                max_depth=self.max_reasoning_depth,
            )

            if strategy == "single_call":
                risk_note = (
                    f"\n⚠ {risk_decision['warning']}"
                    if risk_decision["warning"]
                    else ""
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "COGNITIVE ANALYSIS\n\n"
                        f"Intent: {intent}\n\n"
                        "EXECUTION PLAN\n\n"
                        f"Objective : {execution_plan.objective}\n"
                        f"Risk      : {execution_plan.overall_risk}\n"
                        f"Complexity: {execution_plan.estimated_complexity}\n"
                        f"Tasks     : {len(execution_plan.ordered_tasks)}"
                        f"{risk_note}\n\n"
                        "Tasks in order:\n" +
                        "\n".join(
                            f"  {i+1}. [{t.risk_level}] {t.title}"
                            for i, t in enumerate(execution_plan.ordered_tasks)
                        )
                    ),
                })

        else:
            self.logger.info("lightweight_cognition_path")

        # ── 8. Execution ───────────────────────────────────────────────────
        response = await self._execute(
            strategy=strategy,
            execution_plan=execution_plan,
            messages=messages,
            user_input=user_input,
        )

        if response is None:
            return "[GhostMind: execution produced no output]"

        # ── 9. Reflection ──────────────────────────────────────────────────
        reflection     = None
        should_reflect = (
            path in ("executive", "multi_step")
            or len(user_input) > 120
        )

        if should_reflect:
            reflection = await self.reflection_engine.reflect(
                messages=messages,
                user_input=user_input,
                response=response,
                intent_analysis=intent,
                decomposition=decomposition,
                execution_plan=execution_plan,
            )

            # Publish full scorecard so GhostMind and other subscribers
            # can react to reflection quality without parsing CognitionRecord.
            trace_summary = self.get_trace_summary() if hasattr(self, "get_trace_summary") else {}

            await self._publish("reflection_complete", {
                "coherence_score":      reflection.coherence_score,
                "hallucination_risk":   reflection.hallucination_risk,
                "reasoning_quality":    reflection.reasoning_quality,
                "planning_quality":     reflection.planning_quality,
                "confidence_alignment": reflection.confidence_alignment,
                "requires_retry":       reflection.requires_retry,
                "retry_reason":         reflection.retry_reason,
                "trace_id":             getattr(self._current_trace_context, "trace_id", None),
                "trace_summary":        trace_summary,
            }, priority=4)

            if reflection.requires_retry:
                self.logger.warning(
                    "reflection_retry_triggered",
                    reason=reflection.retry_reason,
                )
                await self._publish("reflection_retry", {
                    "reason": reflection.retry_reason,
                    "notes":  reflection.reflection_notes,
                }, priority=3)

                retry_messages = list(messages)
                retry_messages.append({
                    "role": "system",
                    "content": (
                        "REFLECTION WARNING\n\n"
                        f"{reflection.reflection_notes}\n\n"
                        "Revise the response with improved accuracy and reasoning."
                    ),
                })
                response = await self._llm_call(
                    messages=retry_messages,
                    temperature=0.3,
                )

        # ── 10. Memory persistence ─────────────────────────────────────────
        await self._persist_memory(user_input, response, intent, path)

        # ── 11. Decision ledger ────────────────────────────────────────────
        await self._store_ledger(
            user_input=user_input,
            response=response,
            intent=intent,
            decomposition=decomposition,
            execution_plan=execution_plan,
            reflection=reflection,
        )

        # ── 12. State + events ─────────────────────────────────────────────
        await self.state_manager.update_cognitive_state(
            "last_intent", intent.primary_intent
        )
        await self.state_manager.update_cognitive_state(
            "last_confidence", intent.confidence
        )
        await self.state_manager.update_cognitive_state(
            "last_reflection", _serialize(reflection) if reflection else {}
        )
        await self.state_manager.update_cognitive_state("current_focus", None)

        await self._publish("cognition_complete", {
            "path":     path,
            "strategy": strategy,
            "intent":   intent.primary_intent,
        }, priority=5)

        self._record_trace_event("cognition_complete", {
            "path": path,
            "strategy": strategy,
            "response_length": len(response)
        })

        # Generate and log trace summary
        trace_summary = self.get_trace_summary()
        trace_summary["stage_timings_ms"] = stage_timings
        self.logger.info("trace_summary", **trace_summary)

        self.logger.info(
            "cognition_complete",
            path=path,
            strategy=strategy,
            response_length=len(response),
        )

        return response

    # ── Multi-agent path ──────────────────────────────────────────────────────

    def _is_multi_agent_request(self, user_input: str) -> bool:
        """Return True if the user explicitly requested multi-agent reasoning."""
        return bool(_MULTI_AGENT_RE.search(user_input))

    async def _think_multi_agent(self, user_input: str) -> str:
        """
        Route the request through the MultiAgentOrchestrator and handle
        memory persistence + ledger, then return the final synthesis.
        """
        self.logger.info("multi_agent_path_selected")

        await self._publish("multi_agent_start", {
            "user_input": user_input[:80],
        }, priority=4)

        # Context for agents
        context  = await self.context_loader.build_context(user_input)
        messages = self._build_messages(user_input, context)

        result: MultiAgentResult = await self.multi_agent_orchestrator.run(
            user_input=user_input,
            context_messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response = result.final_synthesis

        # Format agent summary header for the response
        agent_names = " → ".join(r.agent_name for r in result.agent_results)
        header = f"[Multi-Agent: {agent_names} → Synthesizer]\n\n"
        response = header + response

        # Persist memory
        await self._persist_memory(
            user_input=user_input,
            response=response,
            intent=None,
            path="multi_agent",
        )

        # Minimal ledger entry
        try:
            trace_summary = self.get_trace_summary() if hasattr(self, "get_trace_summary") else {}

            record = CognitionRecord(
                conversation_id=str(uuid.uuid4()),
                user_input=user_input,
                final_response=response,
                intent_analysis={"path": "multi_agent"},
                decomposition={},
                execution_plan={},
                reflection={},
                success=True,
                error=None,
                trace_summary=trace_summary,
            )
            await self.decision_ledger.store_record(record)
        except Exception as e:
            self.logger.warning("decision_ledger_store_failed", error=str(e))

        await self.state_manager.update_cognitive_state("current_focus", None)
        await self._publish("cognition_complete", {
            "path":     "multi_agent",
            "strategy": "multi_agent",
            "intent":   "multi_agent_reasoning",
        }, priority=5)

        self.logger.info(
            "multi_agent_complete",
            response_length=len(response),
        )

        return response

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _execute(
        self,
        strategy: str,
        execution_plan,
        messages: list,
        user_input: str,
    ) -> str:
        if strategy == "sequential" and execution_plan is not None:
            return await self.step_executor.execute(
                plan=execution_plan,
                base_messages=messages,
                user_input=user_input,
                max_depth=self.max_reasoning_depth,
            )
        return await self._llm_call(messages=messages)

    async def _llm_call(
        self,
        messages: list,
        temperature: float | None = None,
    ) -> str:
        try:
            return await self.model_client.complete(
                messages=messages,
                temperature=temperature if temperature is not None
                            else self.temperature,
                max_tokens=self.max_tokens,
                system_prompt=GHOSTMIND_SYSTEM_PROMPT,
            )
        except Exception as e:
            exc_type = type(e).__name__
            exc_msg  = str(e) or "(no message)"
            self.logger.error(
                "cognition_inference_failed",
                exc_type=exc_type,
                error=exc_msg,
            )
            return f"[GhostMind: inference failed — {exc_type}: {exc_msg}]"

    async def _persist_memory(
        self,
        user_input: str,
        response: str,
        intent,
        path: str,
    ):
        await self.working_memory.add({"role": "user",      "content": user_input})
        await self.working_memory.add({"role": "assistant", "content": response})
        try:
            await self.memory_bridge.store_memory(
                content=f"User: {user_input}\nGhostMind: {response}",
                memory_type="conversation",
                metadata={
                    "source": "cognition_pipeline",
                    "intent": intent.primary_intent if intent else path,
                    "path":   path,
                },
            )
        except Exception as e:
            self.logger.warning("cognition_memory_persist_failed", error=str(e))

    async def _store_ledger(
        self,
        user_input: str,
        response: str,
        intent,
        decomposition,
        execution_plan,
        reflection,
    ):
        try:
            trace_summary = self.get_trace_summary() if hasattr(self, "get_trace_summary") else {}

            record = CognitionRecord(
                conversation_id=str(uuid.uuid4()),
                user_input=user_input,
                final_response=response,
                intent_analysis=_serialize(intent) if intent else {},
                decomposition=_serialize(decomposition) if decomposition else {},
                execution_plan=_serialize(execution_plan) if execution_plan else {},
                reflection=_serialize(reflection) if reflection else {},
                success=True,
                error=None,
                trace_summary=trace_summary,
            )
            await self.decision_ledger.store_record(record)
        except Exception as e:
            self.logger.warning("decision_ledger_store_failed", error=str(e))

    async def _publish(self, event_type: str, payload: dict, priority: int = 5):
        if self.event_bus is None:
            return
        try:
            await self.event_bus.publish(Event(
                event_type=event_type,
                source="cognition_pipeline",
                payload=payload,
                priority=priority,
            ))
        except Exception as e:
            self.logger.warning(
                "event_publish_failed",
                event_type=event_type,
                error=str(e),
            )

    def _build_messages(self, user_input: str, context: dict) -> list:
        messages = []
        long_term = context.get("long_term_memory", [])
        if long_term:
            snippets = "\n---\n".join(
                str(m.get("content", m)) for m in long_term[:3]
            )
            messages.append({
                "role": "system",
                "content": f"[Retrieved memory context]\n{snippets}",
            })
        for item in context.get("working_memory", []):
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(item)
        messages.append({"role": "user", "content": user_input})
        return messages