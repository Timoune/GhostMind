class ContextLoader:
    """
    Assembles a unified context dict from both memory layers before inference.

    FIX: working_memory.get_context() is now awaited (it became async).
    Long-term retrieval failures are caught and logged as warnings so
    a DreamCloud outage does not block inference.
    """

    def __init__(self, memory_bridge, working_memory, logger):
        self.memory_bridge = memory_bridge
        self.working_memory = working_memory
        self.logger = logger

    async def build_context(self, query: str) -> dict:

        # Long-term memory (DreamCloud) — non-fatal if unavailable
        long_term: list = []
        try:
            result = await self.memory_bridge.retrieve_memory(
                query=query,
                top_k=5
            )
            long_term = result.get("results", [])
        except Exception as e:
            self.logger.warning(
                "context_long_term_unavailable",
                error=str(e)
            )

        # Short-term memory — always available (in-process)
        short_term = await self.working_memory.get_context()

        context = {
            "working_memory": short_term,
            "long_term_memory": long_term
        }

        self.logger.info(
            "context_assembled",
            query=query[:80],
            short_term_items=len(short_term),
            long_term_items=len(long_term)
        )

        return context
