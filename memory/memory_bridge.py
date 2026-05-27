import asyncio
import aiohttp
from typing import Optional


class MemoryBridge:
    """
    HTTP client for DreamCloud (the long-term memory service).

    FIX 1: Single shared aiohttp.ClientSession created in start() and
    closed in stop(). Previously a new session was created per request,
    which is the documented anti-pattern for aiohttp and causes
    connection overhead and potential resource leaks.

    FIX 2: Full error handling with typed exceptions, HTTP status
    validation (raise_for_status), and configurable retry logic with
    exponential back-off.

    FIX 3: DreamCloud being unreachable no longer crashes the runtime —
    the caller receives a clear exception it can choose to handle.
    """

    def __init__(
        self,
        host: str,
        port: int,
        logger,
        retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = f"http://{host}:{port}"
        self.logger = logger
        self.retries = retries
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Open the shared HTTP session. Call once at startup."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def stop(self):
        """Close the shared HTTP session. Call at shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _require_session(self):
        if self._session is None or self._session.closed:
            raise RuntimeError(
                "MemoryBridge: session is not open. "
                "Ensure start() was called before making requests."
            )

    async def _post(self, endpoint: str, payload: dict) -> dict:
        self._require_session()

        for attempt in range(1, self.retries + 1):
            try:
                async with self._session.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()

            except aiohttp.ClientResponseError as e:
                self.logger.error(
                    "memory_bridge_http_error",
                    endpoint=endpoint,
                    status=e.status,
                    attempt=attempt
                )
                raise   # HTTP errors (4xx/5xx) don't benefit from retry

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(
                    "memory_bridge_connection_error",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt,
                    max_attempts=self.retries
                )
                if attempt == self.retries:
                    raise
                await asyncio.sleep(self.retry_delay * attempt)

    # ── Public API ────────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        self._require_session()
        async with self._session.get(
            f"{self.base_url}/health",
            timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def store_memory(
        self,
        content: str,
        memory_type: str = "general",
        metadata: Optional[dict] = None
    ) -> dict:
        payload = {
            "content": content,
            "memory_type": memory_type,
            "metadata": metadata or {}
        }
        data = await self._post("/store", payload)
        self.logger.info("memory_stored", memory_type=memory_type)
        return data

    async def retrieve_memory(
        self,
        query: str,
        top_k: int = 5
    ) -> dict:
        payload = {"query": query, "top_k": top_k}
        data = await self._post("/retrieve", payload)
        self.logger.info(
            "memory_retrieved",
            query=query[:80],
            results=len(data.get("results", []))
        )
        return data
