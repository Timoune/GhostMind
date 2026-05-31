from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ApprovalRequest:
    request_id: str
    risk_level: str
    objective: str
    warning: str
    task_titles: list[str]


class HumanApprovalGate:
    """
    Bridges the async cognition pipeline with a human reviewer
    for high-risk decision approval.

    Flow
    ────
    1. Pipeline calls  await gate.request_approval(...)   → suspends
    2. Gate fires on_approval_requested callback (GUI shows dialog)
    3. User clicks Approve / Deny / Modify in the GUI
    4. GUI calls  gate.resolve(request_id, approved)  via loop.call_soon_threadsafe
    5. Pipeline resumes with the human decision

    Thread safety
    ─────────────
    request_approval() runs inside the asyncio event loop.
    resolve()          is called from the tkinter main thread via
                       loop.call_soon_threadsafe(), which is safe.
    """

    def __init__(
        self,
        logger,
        on_approval_requested: Optional[Callable[["ApprovalRequest"], None]] = None,
        timeout_seconds: float = 120.0,
    ):
        self.logger               = logger
        self.on_approval_requested = on_approval_requested
        self.timeout_seconds      = timeout_seconds
        self._pending: dict[str, asyncio.Future] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def request_approval(
        self,
        risk_level: str,
        objective: str,
        warning: str,
        task_titles: list[str],
    ) -> dict:
        """
        Suspend execution and wait for human approval.

        Returns
        ───────
        {
          "approved"       : bool,
          "modified_input" : str | None,
          "reason"         : "approved" | "denied" | "modify" | "timeout",
        }
        """
        request_id = str(uuid.uuid4())
        loop       = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future

        request = ApprovalRequest(
            request_id=request_id,
            risk_level=risk_level,
            objective=objective,
            warning=warning,
            task_titles=task_titles,
        )

        self.logger.info(
            "hitl_approval_requested",
            request_id=request_id,
            risk_level=risk_level,
            objective=objective[:80],
        )

        # Notify the GUI (runs in the event loop thread, GUI uses after() for safety)
        if self.on_approval_requested:
            try:
                self.on_approval_requested(request)
            except Exception as e:
                self.logger.warning("hitl_callback_failed", error=str(e))

        # Suspend and await human decision (or timeout)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.timeout_seconds,
            )
            self.logger.info(
                "hitl_approval_resolved",
                request_id=request_id,
                approved=result.get("approved"),
                reason=result.get("reason"),
            )
            return result

        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            self.logger.warning(
                "hitl_approval_timeout",
                request_id=request_id,
                timeout=self.timeout_seconds,
            )
            return {
                "approved":       False,
                "modified_input": None,
                "reason":         "timeout",
            }

    def resolve(
        self,
        request_id: str,
        approved: bool,
        reason: str = "approved",
        modified_input: Optional[str] = None,
    ) -> bool:
        """
        Resolve a pending approval.
        Called from the GUI thread via loop.call_soon_threadsafe().

        Returns True if the request was found and resolved.
        """
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False

        future.set_result({
            "approved":       approved,
            "modified_input": modified_input,
            "reason":         reason,
        })
        return True

    def cancel_all(self):
        """Cancel all pending approvals — called on shutdown."""
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)
