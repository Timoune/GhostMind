from __future__ import annotations

import json
from pathlib import Path

from cognition.cognition_record import (
    ReflectionResult,
)


class ReflectionEngine:
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
            / "reflection_prompt.txt"
        )

        self.system_prompt = (
            prompt_path.read_text(
                encoding="utf-8"
            )
        )

    async def reflect(
        self,
        messages: list,
        user_input: str,
        response: str,
        intent_analysis,
        decomposition,
        execution_plan,
    ) -> ReflectionResult:

        reflection_messages = list(messages)

        reflection_messages.append(
            {
                "role": "system",
                "content": (
                    "INTENT ANALYSIS\n\n"
                    f"{intent_analysis}\n\n"

                    "TASK DECOMPOSITION\n\n"
                    f"{decomposition}\n\n"

                    "EXECUTION PLAN\n\n"
                    f"{execution_plan}\n\n"

                    "FINAL RESPONSE\n\n"
                    f"{response}"
                ),
            }
        )

        reflection_messages.append(
            {
                "role": "user",
                "content": (
                    "Evaluate the quality "
                    "and safety of the "
                    "cognitive process."
                ),
            }
        )

        raw = await self.model_client.complete(
            messages=reflection_messages,
            system_prompt=self.system_prompt,
            temperature=0.1,
        )

        parsed = self._safe_parse(raw)

        result = ReflectionResult(
            coherence_score=float(
                parsed.get(
                    "coherence_score",
                    0.5,
                )
            ),
            hallucination_risk=float(
                parsed.get(
                    "hallucination_risk",
                    0.5,
                )
            ),
            reasoning_quality=float(
                parsed.get(
                    "reasoning_quality",
                    0.5,
                )
            ),
            planning_quality=float(
                parsed.get(
                    "planning_quality",
                    0.5,
                )
            ),
            confidence_alignment=float(
                parsed.get(
                    "confidence_alignment",
                    0.5,
                )
            ),
            requires_retry=bool(
                parsed.get(
                    "requires_retry",
                    False,
                )
            ),
            retry_reason=parsed.get(
                "retry_reason",
                "",
            ),
            reflection_notes=parsed.get(
                "reflection_notes",
                "",
            ),
        )

        self.logger.info(
            "reflection_complete",
            coherence_score=(
                result.coherence_score
            ),
            hallucination_risk=(
                result.hallucination_risk
            ),
            requires_retry=(
                result.requires_retry
            ),
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
                "reflection_parse_failed",
                error=str(e),
                raw_response=raw,
            )

            return {
                "coherence_score": 0.1,
                "hallucination_risk": 1.0,
                "reasoning_quality": 0.1,
                "planning_quality": 0.1,
                "confidence_alignment": 0.1,
                "requires_retry": False,
                "retry_reason": "parse_failure",
                "reflection_notes": (
                    "reflection parse failure"
                ),
            }
