from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Uncertainty:
    """Structured uncertainty representation for cognitive outputs."""
    epistemic: float = 0.0          # Model's lack of knowledge / ignorance
    aleatoric: float = 0.0          # Inherent stochasticity or ambiguity in the world
    overall: float = 0.0            # Combined / calibrated uncertainty (0= certain, 1= highly uncertain)
    justification: str = ""         # LLM-provided or rule-based rationale for the scores


@dataclass
class Assumption:
    """Tracked assumption made during reasoning."""
    id: str
    statement: str
    source_stage: str               # e.g. "intent", "decomposition", "planning"
    confidence: float = 0.7
    validated: bool = False
    validation_method: str = ""     # "memory_check", "counterfactual", "llm_query", etc.
    validation_notes: str = ""
    risk_if_false: str = "medium"   # low | medium | high | critical


@dataclass
class IntentAnalysis:
    primary_intent: str
    secondary_intents: List[str]
    confidence: float
    requires_tools: bool
    requires_planning: bool
    urgency: str
    emotional_tone: str
    reasoning: str
    # NEW: uncertainty
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    assumptions: List[Assumption] = field(default_factory=list)


@dataclass
class TaskNode:
    id: str
    title: str
    description: str
    priority: int
    estimated_steps: int
    dependencies: List[str] = field(default_factory=list)
    requires_tools: bool = False
    tool_scope: Optional[str] = None
    risk_level: str = "low"
    # NEW
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    assumptions: List[Assumption] = field(default_factory=list)


@dataclass
class DecompositionResult:
    objective: str
    strategy_summary: str
    estimated_total_steps: int
    confidence: float
    reasoning: str
    tasks: List[TaskNode]
    # NEW
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    assumptions: List[Assumption] = field(default_factory=list)
    meta_notes: str = ""


@dataclass
class ExecutionPlan:
    objective: str
    ordered_tasks: List[TaskNode]
    overall_risk: str
    estimated_complexity: str
    requires_confirmation: bool
    reasoning: str
    # NEW
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    assumptions: List[Assumption] = field(default_factory=list)
    causal_sketch: Optional[Dict[str, Any]] = None   # lightweight world model / causal graph
    counterfactuals: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Output from a single reasoning agent."""
    agent_name: str
    role: str        # "analyst" | "critic" | "synthesizer" | "meta_reviewer"
    output: str
    # NEW
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    assumptions: List[Assumption] = field(default_factory=list)


@dataclass
class MultiAgentResult:
    """Aggregated result from the MultiAgentOrchestrator."""
    agent_results: List[AgentResult]
    final_synthesis: str
    # NEW
    aggregated_uncertainty: Uncertainty = field(default_factory=Uncertainty)
    key_assumptions: List[Assumption] = field(default_factory=list)


# ââ Reflection & Record âââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
    # NEW: metacognition & uncertainty calibration
    assumption_violation_score: float = 0.0
    uncertainty_calibration_error: float = 0.0   # |stated_overall - observed_success|
    meta_notes: str = ""
    validated_assumptions: List[Assumption] = field(default_factory=list)
    violated_assumptions: List[Assumption] = field(default_factory=list)


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
    # NEW
    overall_uncertainty: Uncertainty = field(default_factory=Uncertainty)
    tracked_assumptions: List[Assumption] = field(default_factory=list)
    reasoning_trace: Optional[Dict[str, Any]] = None   # structured full trace for audit