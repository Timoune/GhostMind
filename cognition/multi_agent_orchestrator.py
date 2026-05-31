from __future__ import annotations

import asyncio
from typing import Callable, Optional

from cognition.cognition_types import AgentResult, MultiAgentResult


# ── Agent system prompts ───────────────────────────────────────────────────────

_ANALYST_PROMPT = """\
You are the Analyst in a multi-agent reasoning system.

Your role:
- Deeply examine the user's request: break down the problem, identify key facts,
  implicit assumptions, constraints, and potential risks.
- Be thorough and objective. Structure your output clearly.
- Do NOT write a final user-facing answer — provide analytical findings only.
  The Synthesizer will produce the final response.
"""

_CRITIC_PROMPT = """\
You are the Critic in a multi-agent reasoning system.

Your role:
- Challenge the framing of the user's request. Identify weaknesses, blind spots,
  alternative interpretations, and things that could go wrong.
- Be constructively critical — you are not trying to obstruct, but to improve.
- Do NOT write a final user-facing answer — provide critique and alternatives only.
  The Synthesizer will produce the final response.
"""

_SYNTHESIZER_PROMPT = """\
You are the Synthesizer in a multi-agent reasoning system.

You will receive:
  1. The original user request
  2. An Analyst's structured findings
  3. A Critic's challenges and alternatives

Your role:
- Integrate both perspectives into a single, well-reasoned, final response.
- Resolve tensions between the Analyst and Critic where needed.
- Be concise, direct, and actionable — this is what the user will read.
- Do not describe the multi-agent process; just deliver the best possible answer.
"""

_AGENTS = [
    {
        "name":          "Analyst",
        "role":          "analyst",
        "system_prompt": _ANALYST_PROMPT,
        "temperature":   0.35,
    },
    {
        "name":          "Critic",
        "role":          "critic",
        "system_prompt": _CRITIC_PROMPT,
        "temperature":   0.60,
    },
]


class MultiAgentOrchestrator:
    """
    Runs multiple specialized reasoning agents, then synthesizes their outputs.

    Architecture
    ────────────
    Phase 1 — Parallel:   Analyst + Critic reason independently
    Phase 2 — Sequential: Synthesizer integrates both → final answer

    The on_agent_update callback receives human-readable status strings so the
    GUI can update its loading indicator to show which agent is active.
    """

    def __init__(
        self,
        model_client,
        logger,
        on_agent_update: Optional[Callable[[str], None]] = None,
    ):
        self.model_client    = model_client
        self.logger          = logger
        self.on_agent_update = on_agent_update

    # ── Main entry point ──────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        context_messages: list,
        temperature: float = 0.7,
        max_tokens: int    = 1024,
    ) -> MultiAgentResult:
        """
        Execute the full multi-agent pipeline.

        Returns a MultiAgentResult with per-agent outputs + final synthesis.
        """
        self.logger.info(
            "multi_agent_start",
            user_input_preview=user_input[:80],
        )

        # ── Phase 1: Analyst + Critic in parallel ──────────────────────────
        self._notify("Analyst  +  Critic")

        phase1_tasks = [
            self._run_agent(
                agent=agent,
                user_input=user_input,
                context_messages=context_messages,
                max_tokens=max_tokens,
            )
            for agent in _AGENTS
        ]
        agent_results: list[AgentResult] = list(
            await asyncio.gather(*phase1_tasks)
        )

        # ── Phase 2: Synthesizer ───────────────────────────────────────────
        self._notify("Synthesizer")

        synthesis = await self._synthesize(
            user_input=user_input,
            agent_results=agent_results,
            context_messages=context_messages,
            max_tokens=max_tokens,
        )

        self.logger.info(
            "multi_agent_complete",
            agent_count=len(agent_results),
            synthesis_length=len(synthesis),
        )

        return MultiAgentResult(
            agent_results=agent_results,
            final_synthesis=synthesis,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_agent(
        self,
        agent: dict,
        user_input: str,
        context_messages: list,
        max_tokens: int,
    ) -> AgentResult:
        """Run a single agent and return its AgentResult."""
        self.logger.info("multi_agent_agent_start", agent=agent["name"])

        messages = list(context_messages)
        messages.append({"role": "user", "content": user_input})

        try:
            output = await self.model_client.complete(
                messages=messages,
                system_prompt=agent["system_prompt"],
                temperature=agent["temperature"],
                max_tokens=max_tokens,
            )
        except Exception as e:
            self.logger.warning(
                "multi_agent_agent_failed",
                agent=agent["name"],
                error=str(e),
            )
            output = f"[{agent['name']} failed: {type(e).__name__}: {e}]"

        self.logger.info("multi_agent_agent_done", agent=agent["name"])

        return AgentResult(
            agent_name=agent["name"],
            role=agent["role"],
            output=output,
        )

    async def _synthesize(
        self,
        user_input: str,
        agent_results: list[AgentResult],
        context_messages: list,
        max_tokens: int,
    ) -> str:
        """Integrate all agent outputs into a final response."""
        sections = "\n\n".join(
            f"=== {r.agent_name.upper()} ===\n{r.output}"
            for r in agent_results
        )

        synthesis_prompt = (
            f"ORIGINAL REQUEST:\n{user_input}\n\n"
            f"{sections}\n\n"
            "Using the above findings and critique, provide your final integrated "
            "response to the user's original request."
        )

        messages = list(context_messages)
        messages.append({"role": "user", "content": synthesis_prompt})

        try:
            return await self.model_client.complete(
                messages=messages,
                system_prompt=_SYNTHESIZER_PROMPT,
                temperature=0.45,
                max_tokens=max_tokens,
            )
        except Exception as e:
            self.logger.error("multi_agent_synthesis_failed", error=str(e))
            # Fallback: return analyst output if synthesis fails
            analyst = next(
                (r for r in agent_results if r.role == "analyst"), None
            )
            return analyst.output if analyst else "[Multi-agent synthesis failed]"

    def _notify(self, status: str):
        """Fire the GUI update callback (safe to call from async context)."""
        if self.on_agent_update:
            try:
                self.on_agent_update(status)
            except Exception:
                pass
