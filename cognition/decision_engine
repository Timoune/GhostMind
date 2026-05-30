from __future__ import annotations

from cognition.cognition_types import (
    IntentAnalysis,
    ExecutionPlan,
    TaskNode,
)


# Ordered severity maps for comparison
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
COMPLEXITY_ORDER = {"low": 0, "medium": 1, "high": 2, "extreme": 3}


class DecisionEngine:
    """
    Rule-based active decision-making layer.

    Deterministic — no LLM calls — so it is fast, auditable, and
    does not consume tokens on meta-reasoning.

    Responsibilities
    ────────────────
    • decide_path          — choose lightweight / executive / multi_step
    • gate_on_risk         — proceed, warn, or halt based on plan risk
    • select_strategy      — single_call vs sequential step execution
    • prioritize_goals     — re-order tasks by deps → priority → risk → steps
    """

    def __init__(
        self,
        logger,
        autonomous_mode: bool = False,
    ):
        self.logger = logger
        self.autonomous_mode = autonomous_mode

    # ── Path selection ────────────────────────────────────────────────────────

    def decide_path(self, intent: IntentAnalysis) -> str:
        """
        Returns one of:
          'lightweight'  — simple single-call response
          'executive'    — intent + decompose + plan + single-call + reflect
          'multi_step'   — intent + decompose + plan + step executor + reflect
        """
        requires = intent.requires_planning or intent.requires_tools

        if not requires:
            path = "lightweight"

        elif intent.requires_planning and intent.confidence < 0.6:
            # Uncertain complex task → full iterative execution
            path = "multi_step"

        elif intent.requires_planning:
            path = "multi_step"

        else:
            # requires_tools only (no planning flag) → executive single call
            path = "executive"

        self.logger.info(
            "decision_path_selected",
            path=path,
            confidence=intent.confidence,
            requires_planning=intent.requires_planning,
            requires_tools=intent.requires_tools,
        )

        return path

    # ── Risk gating ───────────────────────────────────────────────────────────

    def gate_on_risk(self, plan: ExecutionPlan) -> dict:
        """
        Evaluate the overall plan risk and decide whether to proceed.

        Returns a dict with keys:
          proceed    : bool
          risk_level : str
          warning    : str | None
        """
        risk = plan.overall_risk
        score = RISK_ORDER.get(risk, 0)

        if score >= RISK_ORDER["critical"]:
            if self.autonomous_mode:
                decision = {
                    "proceed": True,
                    "risk_level": risk,
                    "warning": (
                        "⚠ CRITICAL RISK — autonomous mode active. "
                        "Proceeding with maximum caution."
                    ),
                }
            else:
                decision = {
                    "proceed": False,
                    "risk_level": risk,
                    "warning": (
                        "⛔ CRITICAL RISK — execution halted. "
                        "Please refine or simplify your request."
                    ),
                }

        elif score >= RISK_ORDER["high"]:
            decision = {
                "proceed": True,
                "risk_level": risk,
                "warning": (
                    "⚠ HIGH RISK — proceeding with caution. "
                    "Some tasks may require confirmation."
                ),
            }

        elif score >= RISK_ORDER["medium"]:
            decision = {
                "proceed": True,
                "risk_level": risk,
                "warning": (
                    "ℹ MEDIUM RISK — careful handling applied."
                ),
            }

        else:
            decision = {
                "proceed": True,
                "risk_level": risk,
                "warning": None,
            }

        self.logger.info(
            "decision_risk_gate",
            risk_level=risk,
            proceed=decision["proceed"],
            has_warning=decision["warning"] is not None,
        )

        return decision

    # ── Execution strategy ────────────────────────────────────────────────────

    def select_strategy(
        self,
        plan: ExecutionPlan,
        path: str,
        max_depth: int = 5,
    ) -> str:
        """
        Choose how to execute the plan:
          'single_call'  — LLM handles all tasks in one prompt
          'sequential'   — StepExecutor runs each task individually

        Sequential is preferred for multi_step paths when task count
        and complexity justify it.
        """
        if path != "multi_step":
            return "single_call"

        task_count = len(plan.ordered_tasks)
        complexity = COMPLEXITY_ORDER.get(plan.estimated_complexity, 0)

        if task_count == 0:
            strategy = "single_call"
        elif task_count > max_depth:
            # Too many tasks — collapse to one call with full plan context
            strategy = "single_call"
        elif complexity >= COMPLEXITY_ORDER["medium"] and task_count > 1:
            strategy = "sequential"
        else:
            strategy = "single_call"

        self.logger.info(
            "decision_strategy_selected",
            strategy=strategy,
            task_count=task_count,
            complexity=plan.estimated_complexity,
        )

        return strategy

    # ── Goal prioritization ───────────────────────────────────────────────────

    def prioritize_goals(self, tasks: list[TaskNode]) -> list[TaskNode]:
        """
        Re-order tasks by:
          1. Dependencies first (tasks with no deps execute before dependent ones)
          2. Priority (1 = highest urgency)
          3. Risk (lower risk first — safe tasks before risky ones)
          4. Estimated steps (quick wins first within same tier)
        """

        def sort_key(task: TaskNode):
            has_deps = 1 if task.dependencies else 0
            risk = RISK_ORDER.get(task.risk_level, 0)
            return (has_deps, task.priority, risk, task.estimated_steps)

        ordered = sorted(tasks, key=sort_key)

        self.logger.info(
            "decision_goals_prioritized",
            task_count=len(ordered),
            order=[t.id for t in ordered],
        )

        return ordered
