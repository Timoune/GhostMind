from __future__ import annotations

import json
import re
from pathlib import Path

from cognition.cognition_types import IntentAnalysis


class IntentEngine:
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
            / "intent_prompt.txt"
        )

        self.system_prompt = prompt_path.read_text(
            encoding="utf-8"
        )

    async def analyze(
        self,
        user_input: str,
        context_messages: list,
    ) -> IntentAnalysis:

        messages = list(context_messages)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Analyze the intent of the "
                    "following request:\n\n"
                    f"{user_input}"
                ),
            }
        )

        response = await self.model_client.complete(
            messages=messages,
            system_prompt=self.system_prompt,
            temperature=0.2,
        )

        parsed = self._safe_parse(response)

        result = IntentAnalysis(
            primary_intent=parsed.get(
                "primary_intent",
                "unknown",
            ),
            secondary_intents=parsed.get(
                "secondary_intents",
                [],
            ),
            confidence=float(
                parsed.get(
                    "confidence",
                    0.5,
                )
            ),
            requires_tools=bool(
                parsed.get(
                    "requires_tools",
                    False,
                )
            ),
            requires_planning=bool(
                parsed.get(
                    "requires_planning",
                    False,
                )
            ),
            urgency=parsed.get(
                "urgency",
                "normal",
            ),
            emotional_tone=parsed.get(
                "emotional_tone",
                "neutral",
            ),
            reasoning=parsed.get(
                "reasoning",
                "",
            ),
        )

        self.logger.info(
            "intent_analysis_complete",
            primary_intent=result.primary_intent,
            confidence=result.confidence,
            requires_tools=result.requires_tools,
            requires_planning=result.requires_planning,
        )

        return result

    def _safe_parse(
        self,
        raw: str,
    ) -> dict:

        # Strip markdown code fences robustly.
        # Handles ```json, ```python, ``` etc.
        # The old split("\n",1) approach left the language
        # specifier (e.g. "json") in the text when the LLM
        # emits it on its own line, causing json.loads to fail.
        text = raw.strip()
        text = re.sub(r"^```[^\n]*\n", "", text)  # drop opening fence line
        text = re.sub(r"\n?```\s*$", "", text)     # drop closing fence
        text = text.strip()

        try:
            return json.loads(text)

        except Exception as e:
            self.logger.error(
                "intent_analysis_parse_failed",
                error=str(e),
                raw_response=raw,
            )

            return {
                "primary_intent": "unknown",
                "secondary_intents": [],
                "confidence": 0.1,
                "requires_tools": False,
                "requires_planning": False,
                "urgency": "normal",
                "emotional_tone": "uncertain",
                "reasoning": "json_parse_failure",
            }
