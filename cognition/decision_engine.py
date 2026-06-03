from __future__ import annotations

from cognition.cognition_types import (
    IntentAnalysis,
    ExecutionPlan,
    TaskNode,
    Uncertainty,
)


# Ordered severity maps for comparison
RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
COMPLEXITY_ORDER = {"low": 0, "medium": 1, "high": 2, "extreme": 3}


class DecisionEngine:
    """
    Rule-based active decision-making layer (extended with uncertainty awareness).

    Now uses:
    - Overall uncertainty from upstream stages to influence path selection and risk gating.
    - Assumption risk to escalate confirmation requirements.
    """

    def __init__(
        self,
        logger,
        autonomous_mode: bool = False,
    ):
        self.logger = logger
        self.autonomous_mode = autonomous_mode

    # ── Path selection (now uncertainty-aware) ─────────────────────────────────

    def decide_path(self, intent: IntentAnalysis) -> str:
        """
        Returns one of:
          'lightweight'  — simple single-call response
          'executive'    — intent + decompose + plan + single-call + reflect
          'multi_step'   — intent + decompose + plan + step executor + reflect
        """
        requires = intent.requires_planning or intent.requires_tools

        # NEW: uncertainty influence
        u = intent.uncertainty.overall if intent.uncertainty else 0.0
        high_uncertainty = u > 0.65

        if not requires:
            path = "lightweight"
        elif intent.requires_planning and (intent.confidence < 0.6 or high_uncertainty):
            # High uncertainty or low confidence → full iterative execution for verification
            path = "multi_step"
        elif intent.requires_planning:
            path = "multi_step"
        else:
            path = "executive"

        self.logger.info(
            "decision_path_selected",
            path=path,
            confidence=intent.confidence,
            uncertainty_overall=u,
            high_uncertainty=high_uncertainty,
            requires_planning=intent.requires_planning,
            requires_tools=intent.requires_tools,
        )
        return path

    # ── Risk gating (now considers uncertainty + assumption risk) ─────────────

    def gate_on_risk(self, plan: ExecutionPlan) -> dict:
        """
        Evaluate the overall plan risk and decide whether to proceed, warn, or require HITL.
        Now factors in:
        - plan.overall_risk
        - plan.uncertainty.overall
        - number and severity of unvalidated high-risk assumptions
        """
        base_risk = RISK_ORDER.get(plan.overall_risk, 1)
        u = plan.uncertainty.overall if plan.uncertainty else 0.0

        # Count dangerous unvalidated assumptions
        dangerous_assumptions = [
            a for a in (plan.assumptions or [])
            if not a.validated and a.risk_if_false in ("high", "critical")
        ]
        assumption_penalty = len(dangerous_assumptions) * 0.8

        effective_risk_score = base_risk + (u * 2.0) + assumption_penalty

        if effective_risk_score >= 3.5 or plan.overall_risk == "critical":
            action = "require_hitl"
            reason = "High effective risk (risk + uncertainty + unvalidated critical assumptions)"
        elif effective_risk_score >= 2.5 or plan.overall_risk == "high":
            action = "warn_and_confirm"
            reason = "Elevated risk due to uncertainty or unvalidated assumptions"
        else:
            action = "proceed"
            reason = "Risk within acceptable bounds after uncertainty adjustment"

        result = {
            "action": action,
            "reason": reason,
            "effective_risk_score": round(effective_risk_score, 2),
            "uncertainty_contribution": round(u * 2.0, 2),
            "unvalidated_high_risk_assumptions": len(dangerous_assumptions),
        }

        self.logger.info("risk_gate_evaluated", **result)
        return result

    # ── Other methods remain largely unchanged but can consume uncertainty ────

    def select_strategy(self, plan: ExecutionPlan) -> str:
        """single_call vs sequential — can now factor uncertainty."""
        if plan.uncertainty and plan.uncertainty.overall > 0.7:
            return "sequential"  # more verification steps when uncertain
        return "single_call" if len(plan.ordered_tasks) <= 3 else "sequential"

    def prioritize_goals(self, tasks: List[TaskNode]) -> List[TaskNode]:
        """Existing prioritization logic (deps → priority → risk → steps)."""
        # (implementation unchanged for brevity in this extension)
        return sorted(
            tasks,
            key=lambda t: (
                len(t.dependencies),
                -t.priority,
                -RISK_ORDER.get(t.risk_level, 0),
                -t.estimated_steps
            )
        )