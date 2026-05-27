from __future__ import annotations

from cognition.cognition_types import (
    DecompositionResult,
    ExecutionPlan,
)


class PlanningEngine:
    def __init__(
        self,
        logger,
    ):
        self.logger = logger

    async def build_plan(
        self,
        decomposition: DecompositionResult,
    ) -> ExecutionPlan:

        ordered_tasks = sorted(
            decomposition.tasks,
            key=lambda t: (
                t.priority,
                t.estimated_steps,
            ),
        )

        overall_risk = self._calculate_risk(
            ordered_tasks
        )

        estimated_complexity = (
            self._calculate_complexity(
                ordered_tasks
            )
        )

        requires_confirmation = any(
            task.risk_level in [
                "medium",
                "high",
                "critical",
            ]
            for task in ordered_tasks
        )

        reasoning = (
            "Execution plan synthesized from "
            "task decomposition and risk analysis."
        )

        plan = ExecutionPlan(
            objective=decomposition.objective,
            ordered_tasks=ordered_tasks,
            overall_risk=overall_risk,
            estimated_complexity=estimated_complexity,
            requires_confirmation=(
                requires_confirmation
            ),
            reasoning=reasoning,
        )

        self.logger.info(
            "planning_complete",
            task_count=len(ordered_tasks),
            overall_risk=overall_risk,
            complexity=estimated_complexity,
        )

        return plan

    def _calculate_risk(
        self,
        tasks,
    ) -> str:

        if any(
            t.risk_level == "critical"
            for t in tasks
        ):
            return "critical"

        if any(
            t.risk_level == "high"
            for t in tasks
        ):
            return "high"

        if any(
            t.risk_level == "medium"
            for t in tasks
        ):
            return "medium"

        return "low"

    def _calculate_complexity(
        self,
        tasks,
    ) -> str:

        total_steps = sum(
            t.estimated_steps
            for t in tasks
        )

        if total_steps >= 50:
            return "extreme"

        if total_steps >= 25:
            return "high"

        if total_steps >= 10:
            return "medium"

        return "low"
