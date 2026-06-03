from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

from cognition.cognition_types import Assumption, Uncertainty, IntentAnalysis, DecompositionResult, ExecutionPlan
from llm.model_client import ModelClient


class MetaReasoner:
    """
    Metacognitive layer for GhostMind.

    Responsibilities:
    - Extract implicit assumptions from stage outputs (intent, decomposition, plan)
    - Assign initial confidence and risk_if_false
    - Validate assumptions against memory/context or via targeted LLM query
    - Produce structured assumption sets for downstream use (reflection, decision, audit)
    - Contribute to uncertainty calibration

    This module embodies explicit metacognition: monitoring and critiquing the reasoning process itself.
    """

    def __init__(
        self,
        model_client: ModelClient,
        logger,
        memory_bridge=None,
        working_memory=None,
        temperature: float = 0.3,
        max_tokens: int = 800
    ):
        self.model_client = model_client
        self.logger = logger
        self.memory_bridge = memory_bridge
        self.working_memory = working_memory
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def extract_assumptions(
        self,
        stage: str,
        content: str,
        context: Optional[Dict] = None
    ) -> List[Assumption]:
        """
        Use LLM to extract key assumptions from a reasoning stage output.
        Returns a list of Assumption objects with initial confidence.
        """
        prompt = f"""You are GhostMind's Meta-Reasoner.

Stage: {stage}
Content to analyze:
{content}

Task:
1. Identify the 3-7 most important implicit or explicit assumptions being made.
2. For each assumption:
   - Write a clear, atomic statement.
   - Estimate your confidence that the assumption is true in this context (0.0-1.0).
   - Assess risk_if_false: low | medium | high | critical (what happens to the plan if it is false?).
3. Return ONLY valid JSON in this exact schema:

{{
  "assumptions": [
    {{
      "statement": "...",
      "confidence": 0.85,
      "risk_if_false": "high"
    }}
  ]
}}

Be precise and conservative. Do not invent assumptions that are not reasonably implied by the content.
"""

        try:
            response = await self.model_client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system_prompt="You are a rigorous metacognitive assumption extractor for an autonomous reasoning system. Output only valid JSON."
            )

            # Simple JSON extraction (in production use proper parser + validation)
            import json
            data = json.loads(response.strip().strip('`').strip())
            assumptions = []
            for item in data.get("assumptions", []):
                assumptions.append(Assumption(
                    id=str(uuid.uuid4())[:8],
                    statement=item["statement"],
                    source_stage=stage,
                    confidence=float(item.get("confidence", 0.7)),
                    risk_if_false=item.get("risk_if_false", "medium")
                ))
            self.logger.info("meta_assumptions_extracted", stage=stage, count=len(assumptions))
            return assumptions

        except Exception as e:
            self.logger.error("meta_assumption_extraction_failed", stage=stage, error=str(e))
            return []

    async def validate_assumptions(
        self,
        assumptions: List[Assumption],
        context: Dict[str, Any]
    ) -> List[Assumption]:
        """
        Validate a list of assumptions using available memory and/or targeted reasoning.
        Updates the Assumption objects in place (validated flag + notes).
        """
        validated = []
        for ass in assumptions:
            # Placeholder validation logic â in full system this would query memory_bridge
            # or run a small counterfactual / consistency check.
            # For now we mark as "pending" with heuristic.
            if "fact" in ass.statement.lower() or "known" in ass.statement.lower():
                ass.validated = True
                ass.validation_method = "heuristic_keyword"
                ass.validation_notes = "Appears to reference established fact or known entity."
            else:
                ass.validated = False
                ass.validation_method = "pending_llm_or_memory"
                ass.validation_notes = "Requires further validation against long-term memory or simulation."

            validated.append(ass)

        self.logger.info(
            "meta_assumptions_validated",
            total=len(assumptions),
            validated_count=sum(1 for a in validated if a.validated)
        )
        return validated

    async def enrich_with_uncertainty(
        self,
        base_uncertainty: Uncertainty,
        assumptions: List[Assumption]
    ) -> Uncertainty:
        """
        Adjust overall uncertainty based on number and risk of unvalidated assumptions.
        This is a simple deterministic calibration step (can be replaced by learned model later).
        """
        if not assumptions:
            return base_uncertainty

        high_risk_unvalidated = sum(
            1 for a in assumptions
            if not a.validated and a.risk_if_false in ("high", "critical")
        )
        medium_risk = sum(
            1 for a in assumptions
            if not a.validated and a.risk_if_false == "medium"
        )

        # Simple additive penalty model
        penalty = (high_risk_unvalidated * 0.15) + (medium_risk * 0.08)
        new_overall = min(1.0, base_uncertainty.overall + penalty)

        enriched = Uncertainty(
            epistemic=max(base_uncertainty.epistemic, penalty * 0.6),
            aleatoric=base_uncertainty.aleatoric,
            overall=new_overall,
            justification=base_uncertainty.justification + f" | Meta-adjusted for {len(assumptions)} assumptions ({high_risk_unvalidated} high-risk unvalidated)."
        )
        return enriched

    async def process_stage(
        self,
        stage: str,
        content: str,
        existing_uncertainty: Optional[Uncertainty] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        One-stop helper: extract assumptions, validate, enrich uncertainty.
        Returns a dict ready to be merged into stage result objects.
        """
        assumptions = await self.extract_assumptions(stage, content, context)
        validated_assumptions = await self.validate_assumptions(assumptions, context or {})

        base_u = existing_uncertainty or Uncertainty(overall=0.4)
        enriched_u = await self.enrich_with_uncertainty(base_u, validated_assumptions)

        return {
            "assumptions": validated_assumptions,
            "uncertainty": enriched_u,
            "meta_notes": f"Processed {len(validated_assumptions)} assumptions in {stage} stage."
        }