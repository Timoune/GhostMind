"""
cognition package — GhostMind v4.5

Contains the core cognitive engines and types:
- Intent analysis
- Task decomposition
- Planning and risk assessment
- Metacognition (assumption tracking + uncertainty)
- Decision making
- Reflection
"""

from .intent_engine import IntentEngine
from .decomposition_engine import DecompositionEngine
from .planning_engine import PlanningEngine
from .decision_engine import DecisionEngine
from .meta_reasoner import MetaReasoner
from .reflection_engine import ReflectionEngine

from .cognition_types import (
    IntentAnalysis,
    DecompositionResult,
    ExecutionPlan,
    TaskNode,
    ReflectionResult,
    CognitionRecord,
    Uncertainty,
    Assumption,
)

from .pipeline import CognitionPipeline


__all__ = [
    # Engines
    "IntentEngine",
    "DecompositionEngine",
    "PlanningEngine",
    "DecisionEngine",
    "MetaReasoner",
    "ReflectionEngine",

    # Main entry point
    "CognitionPipeline",

    # Public types
    "IntentAnalysis",
    "DecompositionResult",
    "ExecutionPlan",
    "TaskNode",
    "ReflectionResult",
    "CognitionRecord",
    "Uncertainty",
    "Assumption",
]