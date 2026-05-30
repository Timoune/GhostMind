from __future__ import annotations

import json
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
        """
        Strip markdown code fences (e.g. ```json ... ```) and parse JSON.

        The LLM often wraps JSON in a code block. This method removes the
        opening fence, the optional language tag, and the closing fence
        before attempting to parse.
        """
        text = raw.strip()

        # Remove opening fence ``` and any language identifier
        if text.startswith("```"):
            text = text[3:]                     # drop ```
            if text.startswith("json"):
                text = text[4:]                  # drop 'json'
            text = text.strip()

        # Remove closing fence ``` at the end of the string
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].strip()   # drop trailing ```

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