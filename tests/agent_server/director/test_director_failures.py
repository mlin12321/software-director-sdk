"""Agents and steps that fail, and who is told."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.director.org_store import AGENT_LOST
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.org_watch_service import (
    STALL_WAITS,
)
from openhands.agent_server.director.service.team_messaging_service import failure_waits
from openhands.agent_server.director.tools import MsgTeamTool, PruneTeamTool
from openhands.agent_server.director.tools.definition import MsgTeamAction
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict
from tests.agent_server.director.conftest import (
    PRUNE_API,
    FakeLauncher,
    ic,
    manager,
    node,
    record_of,
    split_root,
    started_with_server,
    use,
    woken,
)


API = "services.api"
NO_KEY = "LLMAuthenticationError: the key was refused"


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    """An org of API and Web whose timer never fires by itself."""
    now = datetime(2026, 11, 17, tzinfo=UTC)
    service = await started_with_server(
        repo, launcher, clock=lambda: now, stall_poll=3600
    )
    await split_root(service)
    yield service
    await service.aclose()


async def fails(
    service: DirectorService,
    launcher: FakeLauncher,
    conversation: str | None,
    said: str | None = NO_KEY,
    status: ConversationExecutionStatus = ConversationExecutionStatus.ERROR,
) -> None:
    assert conversation is not None
    if said is not None:
        launcher.failures[UUID(conversation)] = said
    launcher.report(conversation, status)
    await service.work_queue.settled()


def notices(service: DirectorService, launcher: FakeLauncher) -> list[str]:
    """What the top team's IC was told of agents that failed."""
    return [
        text
        for text in launcher.messages_to(ic(service))
        if "failed and is at rest" in text
    ]


async def test_the_ic_above_is_told_why_an_agent_of_a_team_failed(
    service: DirectorService, launcher: FakeLauncher
):
    await fails(service, launcher, ic(service, API))

    [agent] = [a for a in node(service, API).agents if a.role == "ic"]
    assert (agent.status, agent.reason) == ("blocked", NO_KEY)
    [told] = notices(service, launcher)
    assert "The IC of team 'API'" in told and NO_KEY in told
    assert "`msg_team`" in told
    assert node(service, API).escalations == []
    [event] = [e for e in record_of(service, "root") if e.kind == "failed"]
    assert event.team == "API" and NO_KEY in event.detail


async def test_it_is_told_once_and_the_reason_goes_when_the_agent_runs_again(
    service: DirectorService, launcher: FakeLauncher
):
    api = ic(service, API)
    await fails(service, launcher, api)
    await fails(service, launcher, api)
    assert (
        len(
            [
                t
                for t in launcher.messages_to(ic(service))
                if "failed and is at rest" in t
            ]
        )
        == 1
    )

    await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Try again.")
    )
    assert api is not None
    launcher.report(api, ConversationExecutionStatus.RUNNING)
    await service.work_queue.settled()

    [agent] = [a for a in node(service, API).agents if a.role == "ic"]
    assert (agent.status, agent.reason) == ("working", None)


async def test_an_agent_that_only_repeats_itself_has_a_reason_too(
    service: DirectorService, launcher: FakeLauncher
):
    await fails(
        service,
        launcher,
        ic(service, API),
        said=None,
        status=ConversationExecutionStatus.STUCK,
    )

    [agent] = [a for a in node(service, API).agents if a.role == "ic"]
    assert agent.reason == "It repeated the same step without getting anywhere."


async def test_the_human_is_asked_when_no_ic_works_above_the_one_that_failed(
    service: DirectorService, launcher: FakeLauncher
):
    top = ic(service)
    await fails(service, launcher, top)

    [asked] = node(service, "root").escalations
    assert (asked.id, asked.held_by, asked.raised_by) == (
        failure_waits("ic"),
        None,
        "ic",
    )
    assert NO_KEY in asked.detail and "run it again" in asked.detail

    service.org_run.answer_escalation(
        node(service, "root").id, asked.id, "The key is set."
    )
    await service.work_queue.settled()

    assert (
        "your conversation failed, and says:\nThe key is set."
        in launcher.messages_to(top)[-1]
    )
    assert node(service, "root").escalations == []


async def test_an_agent_that_runs_again_by_itself_is_no_longer_asked_about(
    service: DirectorService, launcher: FakeLauncher
):
    top = ic(service)
    await fails(service, launcher, top)
    assert top is not None

    launcher.report(top, ConversationExecutionStatus.RUNNING)
    await service.work_queue.settled()

    assert node(service, "root").escalations == []


async def test_an_agent_that_uses_a_tool_after_it_failed_is_no_longer_asked_about(
    service: DirectorService, launcher: FakeLauncher
):
    """A conversation that is started again by what runs it need not say
    that it runs; that it acts is enough."""
    top = ic(service)
    await fails(service, launcher, top)
    assert [e.id for e in node(service, "root").escalations] == [failure_waits("ic")]

    await use(service, top, MsgTeamTool, MsgTeamAction(team=API, text="Go on."))
    await service.work_queue.settled()

    assert node(service, "root").escalations == []
    [agent] = [a for a in node(service, "root").agents if a.role == "ic"]
    assert agent.status == "working"


async def test_a_manager_that_failed_is_told_of_as_a_manager(
    service: DirectorService, launcher: FakeLauncher
):
    await woken(service, API)
    await fails(service, launcher, manager(service, API))

    [told] = notices(service, launcher)
    assert "The Manager of team 'API'" in told and "`msg_team`" not in told


async def test_an_agent_that_cannot_be_started_is_lost_with_the_reason(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.derive_error = RuntimeError("npx is not installed")
    service.team_messaging.tell_manager(node(service, API).id, "Look in on your team.")
    await service.work_queue.settled()

    [agent] = [a for a in node(service, API).agents if a.role == "manager"]
    assert agent.status == "lost"
    assert agent.reason == "It could not be started: npx is not installed"
    [told] = notices(service, launcher)
    assert "npx is not installed" in told


async def test_a_step_of_the_director_that_fails_is_kept_in_the_record(
    service: DirectorService, monkeypatch: pytest.MonkeyPatch
):
    async def breaks(_: object) -> None:
        raise RuntimeError("the disk is full")

    monkeypatch.setitem(service.resumable_work.steps, "launch_member", breaks)
    service.team_messaging.tell_manager(node(service, API).id, "Look in on your team.")
    await service.work_queue.settled()

    [event] = [e for e in record_of(service, "root") if e.kind == "failed"]
    assert event.team == "API"
    assert event.detail == (
        "Its Manager is started: not done. RuntimeError: the disk is full"
    )


async def test_the_human_is_asked_of_a_stalled_team_with_no_ic_above(
    repo: Path, launcher: FakeLauncher
):
    now = [datetime(2026, 11, 17, tzinfo=UTC)]
    service = await started_with_server(
        repo, launcher, clock=lambda: now[0], stall_poll=3600
    )
    try:
        await service.org_watch.check_stalls()
        await service.work_queue.settled()
        now[0] += timedelta(minutes=120)
        await service.org_watch.check_stalls()
        await service.work_queue.settled()

        top = node(service, "root")
        [asked] = top.escalations
        assert (asked.id, asked.held_by) == (STALL_WAITS, None)
        assert "has shown no progress for 120 min" in asked.title

        service.org_run.answer_escalation(top.id, STALL_WAITS, "Start with the tests.")
        await service.work_queue.settled()

        assert (
            "shows no progress, and says:\nStart with the tests."
            in launcher.messages_to(ic(service))[-1]
        )
        assert node(service, "root").escalations == []
    finally:
        await service.aclose()


async def test_a_team_whose_removal_failed_is_still_told_things(
    service: DirectorService, launcher: FakeLauncher
):
    await woken(service, API)
    api_ic = ic(service, API)

    def fail_to_stop(call: str) -> None:
        if call == "stop":
            raise RuntimeError("the agent could not be stopped")

    launcher.before = fail_to_stop
    await use(service, manager(service), PruneTeamTool, PRUNE_API)
    await service.work_queue.settled()
    launcher.before = lambda _: None
    told = len(launcher.messages_to(api_ic))

    service.message_delivery.notify(api_ic, "Go on with the orders endpoint.")
    await service.work_queue.settled()

    assert launcher.messages_to(api_ic)[told:] == [
        "[director] Go on with the orders endpoint."
    ]


async def test_a_message_that_could_not_be_sent_is_sent_again_before_the_next(
    service: DirectorService, launcher: FakeLauncher
):
    top_ic = ic(service)
    failed: list[str] = []

    def fail_once(call: str) -> None:
        if call == "send" and not failed:
            failed.append(call)
            raise RuntimeError("the server could not take it")

    launcher.before = fail_once
    told = len(launcher.messages_to(top_ic))

    service.message_delivery.notify(top_ic, "first")
    service.message_delivery.notify(top_ic, "second")
    await service.work_queue.settled()

    sent = [text.splitlines()[0] for text in launcher.messages_to(top_ic)[told:]]
    assert sent == ["[director] first", "[director] second"]


async def test_no_agent_is_started_in_a_team_that_is_being_removed(
    service: DirectorService, launcher: FakeLauncher
):
    api = node(service, API)
    # A step that removes the team waits for its agents to stop.
    service.message_delivery._settling[0] = ({api.id}, api.id)
    starts = len(launcher.derived)

    with pytest.raises(Conflict, match="being replaced or removed"):
        await service.member_start._start_new(
            api, "ic", str(uuid4()), ic(service), "Build it."
        )

    assert len(launcher.derived) == starts


async def test_an_answer_about_a_lost_ic_waits_until_the_team_has_started_again(
    service: DirectorService, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    await fails(service, launcher, ic(service))
    top = node(service, "root").id
    service.org_load.org.set_agent(top, "ic", status=AGENT_LOST)
    heard: list[str | None] = []

    async def start_again(_place: str, _request: object, said: str | None = None):
        heard.append(said)
        if len(heard) == 1:
            raise Conflict("No conversation of the org runs on this server")

    monkeypatch.setattr(service.org_run, "start_again", start_again)

    service.org_run.answer_escalation(top, failure_waits("ic"), "The key is set.")
    await service.work_queue.settled()
    # It could not be started: the human is still asked.
    assert [e.id for e in node(service, "root").escalations] == [failure_waits("ic")]

    service.org_run.answer_escalation(top, failure_waits("ic"), "The key is set.")
    await service.work_queue.settled()
    assert node(service, "root").escalations == []
    assert heard == ["The key is set.", "The key is set."]
