from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ReflectionResult:
    coherence_score: float
    hallucination_risk: float
    reasoning_quality: float
    planning_quality: float
    confidence_alignment: float
    requires_retry: bool
    retry_reason: str
    reflection_notes: str


@dataclass
class CognitionRecord:
    conversation_id: str
    user_input: str
    final_response: str

    intent_analysis: Dict[str, Any]
    decomposition: Dict[str, Any]
    execution_plan: Dict[str, Any]

    reflection: Dict[str, Any]

    success: bool
    error: str | None = None
