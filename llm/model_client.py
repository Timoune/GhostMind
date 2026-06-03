"""
Minimal ModelClient for testing the integrated pipeline.
Returns schema-compliant JSON so that real IntentEngine / DecompositionEngine
parse successfully and produce rich objects for the metacognitive layer.
"""

from __future__ import annotations
from typing import Optional, List, Dict


class ModelClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:8080", timeout_seconds: int = 60, logger=None):
        self.endpoint = endpoint
        self.timeout = timeout_seconds
        self.logger = logger
        self._session = None

    async def start(self):
        pass

    async def stop(self):
        pass

    async def complete(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> str:
        """Mock completion that returns valid JSON matching the engine prompts."""
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        full_context = (system_prompt or "") + " " + last_user.lower()

        # MetaReasoner assumption extraction
        if "assumptions" in full_context or "meta-reasoner" in full_context:
            return '''{
  "assumptions": [
    {"statement": "The user wants a concrete actionable plan or answer.", "confidence": 0.85, "risk_if_false": "medium"},
    {"statement": "No external tools or elevated permissions are required unless explicitly stated.", "confidence": 0.78, "risk_if_false": "low"},
    {"statement": "The request is within normal operational and safety bounds.", "confidence": 0.92, "risk_if_false": "low"}
  ]
}'''

        # Intent analysis - exact schema from intent_prompt.txt
        if "intent" in full_context or "analyze the intent" in full_context:
            return '''{
  "primary_intent": "complex_task",
  "secondary_intents": ["information_gathering", "planning"],
  "confidence": 0.82,
  "requires_tools": false,
  "requires_planning": true,
  "urgency": "normal",
  "emotional_tone": "neutral",
  "reasoning": "User request involves creating a plan or multi-step process."
}'''

        # Decomposition - schema from decomposition_prompt.txt
        if "decompose" in full_context or "break the objective" in full_context:
            return '''{
  "objective": "User request",
  "strategy_summary": "Understand goal then produce structured output or plan.",
  "estimated_total_steps": 3,
  "confidence": 0.79,
  "reasoning": "Standard goal-oriented decomposition.",
  "tasks": [
    {"id": "t1", "title": "Clarify objective", "description": "Ensure understanding.", "priority": 10, "estimated_steps": 1, "dependencies": [], "requires_tools": false, "tool_scope": null, "risk_level": "low"},
    {"id": "t2", "title": "Produce deliverable", "description": "Synthesize into requested output.", "priority": 9, "estimated_steps": 2, "dependencies": ["t1"], "requires_tools": false, "tool_scope": null, "risk_level": "low"}
  ]
}'''

        # Safe default JSON (prevents parse failures in real engines)
        return '{"primary_intent": "simple_query", "secondary_intents": [], "confidence": 0.75, "requires_tools": false, "requires_planning": false, "urgency": "normal", "emotional_tone": "neutral", "reasoning": "Direct request."}'
