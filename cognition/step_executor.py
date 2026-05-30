from __future__ import annotations

from cognition.cognition_types import ExecutionPlan


_STEP_SYSTEM_PROMPT = """\
You are GhostMind executing a specific task step.

Rules:
- Focus only on the task assigned to you in this step.
- Build on prior step results when they are provided.
- Be precise, concise, and factual.
- Do not repeat instructions — produce your result directly.
- Your output will be passed to the next step or the synthesis stage.
"""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are GhostMind synthesizing the results of a completed multi-step execution.

Rules:
- Integrate all step results into a single, coherent final response.
- Address the original user request directly.
- Do not list the steps — present a clean, unified answer.
- Be clear, direct, and well-organized.
"""


class StepExecutor:
    """
    Iterative multi-step task executor.

    Executes each TaskNode in an ExecutionPlan sequentially, passing
    accumulated results between steps, then calls the LLM a final time
    to synthesize a unified response.

    Responsibilities
    ────────────────
    • Multi-step reasoning  — one LLM call per task, up to max_depth
    • Context accumulation  — each step sees prior step results
    • Synthesis             — final call produces the answer for the user
    """

    def __init__(
        self,
        model_client,
        logger,
        max_tokens: int = 1024,
    ):
        self.model_client = model_client
        self.logger = logger
        self.max_tokens = max_tokens

    async def execute(
        self,
        plan: ExecutionPlan,
        base_messages: list,
        user_input: str,
        max_depth: int = 5,
    ) -> str:
        """
        Execute plan tasks one by one (up to max_depth), accumulate
        results, then synthesize a final response.

        Args:
            plan          : ExecutionPlan from PlanningEngine
            base_messages : Message history to prepend to every call
            user_input    : Original user request (for context + synthesis)
            max_depth     : Maximum number of task steps to execute

        Returns:
            Synthesized final response string.
        """
        tasks = plan.ordered_tasks[:max_depth]
        total = len(tasks)
        step_results: list[dict] = []

        self.logger.info(
            "step_executor_start",
            objective=plan.objective,
            task_count=total,
            max_depth=max_depth,
        )

        for i, task in enumerate(tasks, start=1):

            self.logger.info(
                "step_start",
                step=i,
                total=total,
                task_id=task.id,
                title=task.title,
                risk_level=task.risk_level,
            )

            step_messages = list(base_messages)

            # Inject all prior step results as accumulated context
            if step_results:
                prior_text = "\n\n".join(
                    f"[Step {r['step']}] {r['title']}:\n{r['result']}"
                    for r in step_results
                )
                step_messages.append({
                    "role": "system",
                    "content": f"PRIOR STEP RESULTS\n\n{prior_text}",
                })

            # Formulate the current task prompt
            deps_note = (
                f"Dependencies: {', '.join(task.dependencies)}"
                if task.dependencies
                else "No dependencies."
            )

            step_messages.append({
                "role": "user",
                "content": (
                    f"STEP {i} OF {total}\n\n"
                    f"Task ID   : {task.id}\n"
                    f"Title     : {task.title}\n"
                    f"Details   : {task.description}\n"
                    f"Risk      : {task.risk_level}\n"
                    f"{deps_note}\n\n"
                    f"Original request: {user_input}"
                ),
            })

            try:
                result = await self.model_client.complete(
                    messages=step_messages,
                    system_prompt=_STEP_SYSTEM_PROMPT,
                    temperature=0.4,
                    max_tokens=self.max_tokens,
                )

                self.logger.info(
                    "step_complete",
                    step=i,
                    task_id=task.id,
                )

            except Exception as e:
                result = f"[Step {i} error: {type(e).__name__}: {e}]"
                self.logger.error(
                    "step_failed",
                    step=i,
                    task_id=task.id,
                    error=str(e),
                )

            step_results.append({
                "step": i,
                "task_id": task.id,
                "title": task.title,
                "result": result,
            })

        # Final synthesis pass
        return await self._synthesize(
            step_results=step_results,
            base_messages=base_messages,
            user_input=user_input,
        )

    async def _synthesize(
        self,
        step_results: list[dict],
        base_messages: list,
        user_input: str,
    ) -> str:
        """
        Combine all step results into a single coherent response.
        """
        self.logger.info(
            "synthesis_start",
            step_count=len(step_results),
        )

        all_results = "\n\n".join(
            f"[Step {r['step']}] {r['title']}:\n{r['result']}"
            for r in step_results
        )

        synthesis_messages = list(base_messages)

        synthesis_messages.append({
            "role": "system",
            "content": f"COMPLETED STEP RESULTS\n\n{all_results}",
        })

        synthesis_messages.append({
            "role": "user",
            "content": (
                "Using the step results above, produce a final unified "
                f"response to the original request:\n\n{user_input}"
            ),
        })

        try:
            response = await self.model_client.complete(
                messages=synthesis_messages,
                system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                temperature=0.5,
                max_tokens=self.max_tokens,
            )

            self.logger.info("synthesis_complete")
            return response

        except Exception as e:
            self.logger.error("synthesis_failed", error=str(e))
            # Fall back: concatenate step results
            return "\n\n".join(
                f"**{r['title']}**\n{r['result']}"
                for r in step_results
            )
