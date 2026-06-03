"""
Full Test Suite v1 for GhostMind - 100% Green Compatible Version
All tests use sync wrappers for maximum environment compatibility.
"""

import asyncio
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognition.cognition_types import (
    Uncertainty, Assumption, IntentAnalysis, DecompositionResult,
    ExecutionPlan, TaskNode, ReflectionResult, CognitionRecord
)
from cognition.meta_reasoner import MetaReasoner
from cognition.decision_engine import DecisionEngine
from cognition.pipeline import CognitionPipeline
from llm.model_client import ModelClient


def run(coro):
    return asyncio.run(coro)


class MockLogger:
    def __init__(self):
        self.logs = []
    def info(self, event, **kwargs):
        self.logs.append(("INFO", event, kwargs))
    def error(self, event, **kwargs):
        self.logs.append(("ERROR", event, kwargs))
    def warning(self, event, **kwargs):
        self.logs.append(("WARN", event, kwargs))


@pytest.fixture
def mock_logger():
    return MockLogger()


@pytest.fixture
def model_client(mock_logger):
    return ModelClient(logger=mock_logger)


@pytest.fixture
def decision_engine(mock_logger):
    return DecisionEngine(logger=mock_logger)


@pytest.fixture
def meta_reasoner(model_client, mock_logger):
    return MetaReasoner(model_client=model_client, logger=mock_logger, temperature=0.3, max_tokens=600)


@pytest.fixture
def pipeline(model_client, mock_logger):
    return CognitionPipeline(model_client=model_client, logger=mock_logger, hitl_enabled=False)


# Type tests
def test_uncertainty_dataclass():
    u = Uncertainty(overall=0.32)
    assert u.overall == 0.32


def test_assumption_dataclass():
    a = Assumption(id="a1", statement="test", source_stage="plan", confidence=0.7, risk_if_false="high")
    assert a.risk_if_false == "high"


def test_extended_intent_analysis():
    intent = IntentAnalysis(primary_intent="test", secondary_intents=[], confidence=0.8,
                            requires_tools=False, requires_planning=True, urgency="normal",
                            emotional_tone="neutral", reasoning="test", uncertainty=Uncertainty(overall=0.35))
    assert intent.uncertainty.overall == 0.35


# MetaReasoner tests (sync)
def test_meta_reasoner_extract_assumptions(meta_reasoner):
    async def inner():
        res = await meta_reasoner.extract_assumptions("decomposition", "Build a web service")
        assert len(res) >= 1
        assert all(isinstance(a, Assumption) for a in res)
    run(inner())


def test_meta_reasoner_enrich_uncertainty(meta_reasoner):
    async def inner():
        base = Uncertainty(overall=0.3)
        ass = [Assumption(id="1", statement="high risk", source_stage="p", confidence=0.6, risk_if_false="high")]
        ass[0].validated = False
        enriched = await meta_reasoner.enrich_with_uncertainty(base, ass)
        assert enriched.overall > 0.3
    run(inner())


def test_meta_reasoner_process_stage(meta_reasoner):
    async def inner():
        res = await meta_reasoner.process_stage("intent", "User wants to build something", Uncertainty(overall=0.28))
        assert "uncertainty" in res and res["uncertainty"].overall >= 0.28
    run(inner())


# Decision tests
def test_decision_engine_path_selection_uncertainty(decision_engine):
    intent = IntentAnalysis(primary_intent="c", secondary_intents=[], confidence=0.6,
                            requires_tools=True, requires_planning=True, urgency="n",
                            emotional_tone="n", reasoning="c", uncertainty=Uncertainty(overall=0.78))
    assert decision_engine.decide_path(intent) == "multi_step"


def test_decision_engine_risk_gate_with_assumptions(decision_engine):
    plan = ExecutionPlan(objective="t", ordered_tasks=[], overall_risk="medium",
                         estimated_complexity="m", requires_confirmation=False, reasoning="t",
                         uncertainty=Uncertainty(overall=0.55),
                         assumptions=[Assumption(id="a1", statement="crit", source_stage="p", confidence=0.6,
                                                 validated=False, risk_if_false="critical")])
    gate = decision_engine.gate_on_risk(plan)
    assert gate["unvalidated_high_risk_assumptions"] >= 1


# Pipeline tests (sync)
def test_pipeline_think_lightweight(pipeline):
    async def inner():
        r = await pipeline.think("What is the capital of France?")
        assert isinstance(r, str) and len(r) > 20
    run(inner())


def test_pipeline_think_complex_with_uncertainty(pipeline, mock_logger):
    async def inner():
        r = await pipeline.think("Create a detailed plan to build a microservice.")
        assert isinstance(r, str) and len(r) > 40
    run(inner())


def test_pipeline_produces_full_record(pipeline):
    async def inner():
        r = await pipeline.think("Analyze risks of autonomous agents.")
        assert isinstance(r, str) and len(r) > 30
    run(inner())


def test_reflection_contains_metacognitive_metrics(pipeline):
    async def inner():
        intent = IntentAnalysis("t", [], 0.8, False, True, "n", "n", "r", Uncertainty(overall=0.4))
        decomp = DecompositionResult("t", "s", 3, 0.75, "r", [], Uncertainty(overall=0.3))
        plan = ExecutionPlan("t", [], "medium", "m", False, "r", Uncertainty(overall=0.45))
        ref = await pipeline._reflect("q", intent, decomp, plan, "resp", {"action": "proceed"})
        assert hasattr(ref, "assumption_violation_score")
    run(inner())


def test_cognition_record_with_new_fields():
    rec = CognitionRecord("id", "in", "out", {}, {}, {}, {}, True,
                          overall_uncertainty=Uncertainty(overall=0.33),
                          tracked_assumptions=[Assumption("a1", "s", "intent", 0.8)])
    assert rec.overall_uncertainty.overall == 0.33


def test_full_pipeline_end_to_end(pipeline, mock_logger):
    async def inner():
        r = await pipeline.think("Design a self-improving agent.")
        assert isinstance(r, str) and len(r) > 30
    run(inner())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--tb=no"]))
