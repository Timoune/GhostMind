from cognition.intent_engine import IntentEngine
from cognition.decomposition_engine import DecompositionEngine
from cognition.reflection_engine import ReflectionEngine
from cognition.decision_ledger import DecisionLedger


GHOSTMIND_SYSTEM_PROMPT = """You are GhostMind — the reasoning core of Mini Von, an autonomous AI assistant.

Your responsibilities:
- Parse intent precisely before acting
- Decompose complex tasks into clear steps
- Reason carefully and acknowledge uncertainty
- Be direct, concise, and honest

You have access to:
- Working memory: the current session's conversation history
- Long-term memory: relevant past context retrieved from DreamCloud

Always think before you respond. For multi-step tasks, outline the steps first.
"""


class CognitionPipeline:

    def __init__(
        self,
        model_client,
        context_loader,
        working_memory,
        memory_bridge,
        state_manager,
        logger,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.model_client = model_client
        self.context_loader = context_loader
        self.working_memory = working_memory
        self.memory_bridge = memory_bridge
        self.state_manager = state_manager
        self.logger = logger
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.intent_engine = IntentEngine(
            model_client=self.model_client,
            logger=self.logger,
        )

        self.decomposition_engine = DecompositionEngine(
            model_client=self.model_client,
            logger=self.logger,
        )

        self.reflection_engine = ReflectionEngine(
            model_client=self.model_client,
            logger=self.logger,
        )

        self.decision_ledger = DecisionLedger(
            logger=self.logger,
        )

    async def think(self, user_input: str) -> str:

        self.logger.info(
            "cognition_start",
            input_preview=user_input[:100]
        )

        await self.state_manager.update_cognitive_state(
            "current_focus",
            user_input[:200]
        )

        context = await self.context_loader.build_context(
            user_input
        )

        messages = self._build_messages(
            user_input,
            context
        )

        intent_analysis = await self.intent_engine.analyze(
            user_input=user_input,
            context_messages=messages,
        )

        use_executive_cognition = (
            intent_analysis.requires_planning
            or intent_analysis.requires_tools
        )

        decomposition = None

        if use_executive_cognition:

            self.logger.info(
                "executive_cognition_enabled"
            )

            decomposition = (
                await self.decomposition_engine.decompose(
                    objective=user_input,
                    context_messages=messages,
                    intent_analysis=intent_analysis,
                )
            )

            messages.append({
                "role": "system",
                "content": (
                    "EXECUTIVE ANALYSIS\n\n"
                    f"{intent_analysis}\n\n"

                    "TASK DECOMPOSITION\n\n"
                    f"{decomposition}"
                )
            })

        else:

            self.logger.info(
                "lightweight_cognition_path"
            )

        try:

            response = await self.model_client.complete(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system_prompt=GHOSTMIND_SYSTEM_PROMPT
            )

        except Exception as e:

            exc_type = type(e).__name__
            exc_msg = str(e) or "(no message)"

            self.logger.error(
                "cognition_inference_failed",
                exc_type=exc_type,
                error=exc_msg
            )

            return (
                f"[GhostMind: inference failed — "
                f"{exc_type}: {exc_msg}]"
            )

        reflection = None

        should_reflect = (
            use_executive_cognition
            or len(user_input) > 120
        )

        if should_reflect:

            reflection = (
                await self.reflection_engine.reflect(
                    messages=messages,
                    user_input=user_input,
                    response=response,
                    intent_analysis=intent_analysis,
                    decomposition=decomposition,
                    execution_plan=None,
                )
            )

            if reflection.requires_retry:

                self.logger.warning(
                    "reflection_retry_triggered",
                    reason=reflection.retry_reason
                )

                retry_messages = list(messages)

                retry_messages.append({
                    "role": "system",
                    "content": (
                        "REFLECTION WARNING\n\n"
                        f"{reflection.reflection_notes}\n\n"

                        "Revise the response with "
                        "improved accuracy and reasoning."
                    )
                })

                response = await self.model_client.complete(
                    messages=retry_messages,
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    system_prompt=GHOSTMIND_SYSTEM_PROMPT
                )

        await self.working_memory.add({
            "role": "user",
            "content": user_input
        })

        await self.working_memory.add({
            "role": "assistant",
            "content": response
        })

        try:

            await self.memory_bridge.store_memory(
                content=(
                    f"User: {user_input}\n"
                    f"GhostMind: {response}"
                ),
                memory_type="conversation",
                metadata={
                    "source": "cognition_pipeline",
                    "intent": (
                        intent_analysis.primary_intent
                    ),
                }
            )

        except Exception as e:

            self.logger.warning(
                "cognition_memory_persist_failed",
                error=str(e)
            )

        try:

            await self.decision_ledger.store_record(
                {
                    "user_input": user_input,
                    "response": response,
                    "intent_analysis": vars(intent_analysis),
                    "decomposition": (
                        vars(decomposition)
                        if decomposition
                        else None
                    ),
                    "reflection": (
                        vars(reflection)
                        if reflection
                        else None
                    ),
                }
            )

        except Exception as e:

            self.logger.warning(
                "decision_ledger_store_failed",
                error=str(e)
            )

        await self.state_manager.update_cognitive_state(
            "last_intent",
            intent_analysis.primary_intent
        )

        await self.state_manager.update_cognitive_state(
            "last_confidence",
            intent_analysis.confidence
        )

        await self.state_manager.update_cognitive_state(
            "last_reflection",
            (
                vars(reflection)
                if reflection
                else {}
            )
        )

        await self.state_manager.update_cognitive_state(
            "current_focus",
            None
        )

        self.logger.info(
            "cognition_complete",
            response_length=len(response),
            executive_cognition=(
                use_executive_cognition
            ),
        )

        return response

    def _build_messages(
        self,
        user_input: str,
        context: dict
    ) -> list:

        messages = []

        long_term = context.get(
            "long_term_memory",
            []
        )

        if long_term:

            snippets = "\n---\n".join(
                str(m.get("content", m))
                for m in long_term[:3]
            )

            messages.append({
                "role": "system",
                "content": (
                    "[Retrieved memory context]\n"
                    f"{snippets}"
                )
            })

        for item in context.get(
            "working_memory",
            []
        ):

            if (
                isinstance(item, dict)
                and "role" in item
                and "content" in item
            ):

                messages.append(item)

        messages.append({
            "role": "user",
            "content": user_input
        })

        return messages
