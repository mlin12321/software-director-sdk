"""The director's own calls of a model: a question apart from any agent's
work, a question on top of an agent's conversation, a summary."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.models import Session
from openhands.sdk.errors import NotFound


# How many of its own model calls the director makes at once. They are made
# with the models of the org's agents and must not hold up the agents.
MODEL_CALLS_AT_ONCE = 2


@dataclass
class ModelCallService:
    """Asks the models of the org's agents on the director's behalf.

    It decides nothing: a call that finds no conversation, or takes too
    long, gives None, and its caller goes on without an answer.
    """

    launcher: Callable[[], TeamLauncher]
    """The conversations of the agents, as they are when asked."""
    sessions: Callable[[], Sequence[Session]]
    """The org's sessions, oldest first."""
    call_timeout: float
    """Seconds a question may take."""
    summary_timeout: float
    """Seconds a summary may take."""
    at_once: int = MODEL_CALLS_AT_ONCE
    _slots: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._slots = asyncio.Semaphore(self.at_once)

    def asker(self) -> str | None:
        """The conversation the director's own questions are asked from."""
        # Nobody works at the top team itself: the IC a session started
        # with, oldest first.
        return next(
            (
                session.conversation_id
                for session in self.sessions()
                if session.conversation_id
            ),
            None,
        )

    async def ask(
        self,
        conversation_id: str,
        system: str,
        user: str,
        usage_id: str,
        *,
        in_turn: bool = False,
    ) -> str | None:
        """Ask the model of an agent one question, apart from its work.

        Args:
            in_turn: Whether the call waits for one of the few made at once.
        """
        if in_turn:
            async with self._slots:
                return await self.ask(conversation_id, system, user, usage_id)
        try:
            return await asyncio.wait_for(
                self.launcher().ask(
                    UUID(conversation_id), system, user, usage_id=usage_id
                ),
                timeout=self.call_timeout,
            )
        except (NotFound, TimeoutError):
            return None

    async def ask_agent(self, conversation_id: str, question: str) -> str | None:
        """Ask an agent one question on top of its conversation so far."""
        try:
            return await asyncio.wait_for(
                self.launcher().ask_agent(UUID(conversation_id), question),
                timeout=self.call_timeout,
            )
        except (NotFound, TimeoutError):
            return None

    async def summarize(self, conversation_id: str, text: str) -> TeamSummary | None:
        """Have the model of an agent say in a title and a line what
        ``text`` is about, as one of the few calls made at once."""
        try:
            async with self._slots:
                return await asyncio.wait_for(
                    self.launcher().summarize(UUID(conversation_id), text),
                    timeout=self.summary_timeout,
                )
        except (NotFound, TimeoutError):
            return None
