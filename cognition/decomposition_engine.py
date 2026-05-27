from __future__ import annotations

import json
from pathlib import Path

from cognition.cognition_types import (
    TaskNode,
    DecompositionResult,
)


class DecompositionEngine:
    def __init__(
        self,
        model_client,
        logger,
    ):
        self.model_client = model_client
        self.logger = logger

        prompt_path = (
            Path(__file__).parent
            / "prompts"
            / "decomposition_prompt.txt"
        )

        self.system_prompt = prompt_path.read_text(
            encoding="utf-8"
        )

    async def decompose(
        self,
        objective: str,
        context_messages: list,
        intent_analysis,
    ) -> DecompositionResult:

        messages = list(context_messages)

        messages.append(
            {
                "role": "system",
                "content": (
                    "Intent Analysis:\n"
                    f"{intent_analysis}"
                ),
            }
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Decompose this objective:\n\n"
                    f"{objective}"
                ),
            }
        )

        response = await self.model_client.complete(
            messages=messages,
            system_prompt=self.system_prompt,
            temperature=0.3,
        )

        parsed = self._safe_parse(response)

        tasks = []

        for item in parsed.get(
            "tasks",
            [],
        ):

            task = TaskNode(
                id=item.get(
                    "id",
                    "",
                ),
                title=item.get(
                    "title",
                    "",
                ),
                description=item.get(
                    "description",
                    "",
                ),
                priority=int(
                    item.get(
                        "priority",
                        1,
                    )
                ),
                estimated_steps=int(
                    item.get(
                        "estimated_steps",
                        1,
                    )
                ),
                dependencies=item.get(
                    "dependencies",
                    [],
                ),
                requires_tools=bool(
                    item.get(
                        "requires_tools",
                        False,
                    )
                ),
                tool_scope=item.get(
                    "tool_scope",
                ),
                risk_level=item.get(
                    "risk_level",
                    "low",
                ),
            )

            tasks.append(task)

        result = DecompositionResult(
            objective=parsed.get(
                "objective",
                objective,
            ),
            strategy_summary=parsed.get(
                "strategy_summary",
                "",
            ),
            estimated_total_steps=int(
                parsed.get(
                    "estimated_total_steps",
                    len(tasks),
                )
            ),
            confidence=float(
                parsed.get(
                    "confidence",
                    0.5,
                )
            ),
            reasoning=parsed.get(
                "reasoning",
                "",
            ),
            tasks=tasks,
        )

        self.logger.info(
            "decomposition_complete",
            task_count=len(tasks),
            confidence=result.confidence,
        )

        return result

    def _safe_parse(
        self,
        raw: str,
    ) -> dict:

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        # that LLMs frequently add around JSON output.
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        try:
            return json.loads(text)

        except Exception as e:
            self.logger.error(
                "decomposition_parse_failed",
                error=str(e),
                raw_response=raw,
            )

            return {
                "objective": "unknown",
                "strategy_summary": "",
                "estimated_total_steps": 0,
                "confidence": 0.1,
                "reasoning": "json_parse_failure",
                "tasks": [],
            }
