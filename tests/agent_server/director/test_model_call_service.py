"""The director's own model calls, with a launcher that answers by hand
and no service."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.launch import TeamLauncher
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.sdk.errors import NotFound


class Answers:
    """A launcher that answers each call as the test set it to."""

    def __init__(self) -> None:
        self.asked: list[tuple[UUID, str]] = []
        self.running = 0
        self.most_at_once = 0
        self.wait = 0.0
        self.gone = False

    async def ask(
        self, conversation_id: UUID, system: str, user: str, *, usage_id: str
    ) -> str | None:
        self.asked.append((conversation_id, usage_id))
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            await asyncio.sleep(self.wait)
            if self.gone:
                raise NotFound("no such conversation")
            return f"{system}/{user}"
        finally:
            self.running -= 1

    async def summarize(self, conversation_id: UUID, task: str) -> TeamSummary | None:
        await asyncio.sleep(self.wait)
        return TeamSummary(title="Checkout", headline=task)


def calls(
    answers: Answers, sessions: list[Any] | None = None, **limits: float
) -> ModelCallService:
    return ModelCallService(
        launcher=lambda: cast(TeamLauncher, answers),
        sessions=lambda: sessions or [],
        call_timeout=limits.get("call_timeout", 5),
        summary_timeout=limits.get("summary_timeout", 5),
    )


def test_questions_are_asked_from_the_oldest_session_that_has_an_agent():
    first, second = str(uuid4()), str(uuid4())
    sessions = [
        SimpleNamespace(conversation_id=None),
        SimpleNamespace(conversation_id=first),
        SimpleNamespace(conversation_id=second),
    ]

    assert calls(Answers(), sessions).asker() == first
    assert calls(Answers()).asker() is None


async def test_a_question_gets_the_models_answer():
    answers = Answers()
    conversation = str(uuid4())

    answer = await calls(answers).ask(conversation, "system", "user", "routing")

    assert answer == "system/user"
    assert answers.asked == [(UUID(conversation), "routing")]


async def test_a_call_that_takes_too_long_or_finds_no_conversation_gives_none():
    slow = Answers()
    slow.wait = 0.2
    conversation = str(uuid4())

    of_slow = calls(slow, call_timeout=0.01, summary_timeout=0.01)
    assert await of_slow.ask(conversation, "s", "u", "x") is None
    assert await of_slow.summarize(conversation, "Build it") is None

    gone = Answers()
    gone.gone = True
    assert await calls(gone).ask(conversation, "s", "u", "x") is None


async def test_calls_made_in_turn_are_only_a_few_at_once():
    answers = Answers()
    answers.wait = 0.01
    service = calls(answers)
    conversation = str(uuid4())

    await asyncio.gather(
        *(service.ask(conversation, "s", "u", "x", in_turn=True) for _ in range(6))
    )

    assert answers.most_at_once == service.at_once == 2
