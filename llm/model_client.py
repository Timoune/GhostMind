import aiohttp
from typing import Optional


class ModelClient:
    """
    Async HTTP client for the llama.cpp server.

    Uses the OpenAI-compatible /v1/chat/completions endpoint.

    The llama.cpp server must already be running and pointing at your
    GGUF file (located in MiniVon/llm/). Either start it manually:

        llama-server --model /path/to/MiniVon/llm/model.gguf --port 8080

    Or use llm/server_launcher.py to have GhostMind launch it automatically.

    A single aiohttp.ClientSession is created in start() and reused for
    all requests — opened once, closed once, never recreated per-call.
    """

    def __init__(
        self,
        endpoint: str,
        timeout_seconds: int = 60,
        logger=None
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.logger = logger
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def stop(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Returns True if llama.cpp server is reachable."""
        try:
            async with self._session.get(
                f"{self.endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages:      List of {"role": "user"|"assistant", "content": "..."}
            temperature:   Sampling temperature. 0.0 = deterministic.
            max_tokens:    Maximum tokens to generate.
            system_prompt: If provided, prepended as a system message.

        Returns:
            The model's response as a plain string.
        """
        if self._session is None or self._session.closed:
            raise RuntimeError(
                "ModelClient: session not open. Call start() first."
            )

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        payload = {
            "model": "local",           # llama.cpp ignores this field
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        async with self._session.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        text = data["choices"][0]["message"]["content"].strip()

        if self.logger:
            usage = data.get("usage", {})
            self.logger.info(
                "llm_completion",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0)
            )

        return text
