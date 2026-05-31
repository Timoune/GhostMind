from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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


@dataclass
class DecompositionResult:
    objective: str
    strategy_summary: str
    estimated_total_steps: int
    confidence: float
    reasoning: str
    tasks: List[TaskNode]


@dataclass
class ExecutionPlan:
    objective: str
    ordered_tasks: List[TaskNode]
    overall_risk: str
    estimated_complexity: str
    requires_confirmation: bool
    reasoning: str


# ── Multi-agent types ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """Output from a single reasoning agent."""
    agent_name: str
    role: str        # "analyst" | "critic" | "synthesizer"
    output: str


@dataclass
class MultiAgentResult:
    """Aggregated result from the MultiAgentOrchestrator."""
    agent_results: List[AgentResult]
    final_synthesis: str
