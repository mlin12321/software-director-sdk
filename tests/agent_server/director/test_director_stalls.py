"""Teams that show no progress, and what the IC above can do about them."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.models import DirectorSettingsUpdate, SplitChild
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgTeamTool,
    ProposeTeamsTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ReviewTeamTool,
    StopTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgTeamAction,
    PruneTeamAction,
    StopTeamAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import FakeLauncher, record_of, tasked
from tests.agent_server.director.test_director_acp import started
from tests.agent_server.director.test_director_review import (
    DONE,
    api_submits,
    commit,
    reject,
    requests,
)
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    CENTS,
    SPLIT,
    ic,
    manager,
    node,
    proposing,
    split_root,
    use,
)


API, WEB = "services.api", "apps.web"
STOP_API = StopTeamAction(team=API, reason="Three hours on one failing test.")


@dataclass
class Clock:
    now: datetime = field(default_factory=lambda: datetime(2026, 11, 14, tzinfo=UTC))

    def __call__(self) -> datetime:
        return self.now

    def passes(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher(agent_answers={"shown no progress": "Stuck on the schema."})


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher, clock: Clock):
    """An org of API and Web whose timer never fires by itself."""
    service = await started(repo, launcher, clock=clock, stall_poll=3600)
    await split_root(service)
    yield service
    await service.aclose()


async def looked(service: DirectorService) -> None:
    """The director looks for teams that show no progress, as its timer does."""
    await service._check_stalls()
    await service.settled()


def notices(service: DirectorService, launcher: FakeLauncher) -> list[str]:
    """What the top team's IC was told of teams that show no progress."""
    return [
        text
        for text in launcher.messages_to(ic(service))
        if "has shown no progress" in text
    ]


async def test_a_team_without_progress_for_too_long_is_told_to_the_ic_above(
    service: DirectorService, launcher: FakeLauncher, clock: Clock
):
    await looked(service)
    clock.passes(119)
    await looked(service)
    assert notices(service, launcher) == []

    clock.passes(1)
    await looked(service)

    told = notices(service, launcher)
    assert len(told) == 2
    about_api = next(text for text in told if "Team 'API'" in text)
    assert "has shown no progress for 2 h" in about_api
    # The director asked the team's IC itself, on top of its conversation.
    assert "Asked just now, its IC says:\nStuck on the schema." in about_api
    assert "git log --oneline -5 -- services/api" in about_api
    assert "`msg_team`" in about_api and "very sparingly" in about_api
    assert (UUID(str(ic(service, API))), True) in [
        (asked, "shown no progress" in question)
        for asked, question in launcher.agents_asked
    ]


async def test_it_is_told_again_only_after_another_such_stretch(
    service: DirectorService, launcher: FakeLauncher, clock: Clock
):
    await looked(service)
    clock.passes(120)
    await looked(service)
    clock.passes(60)
    await looked(service)
    assert len(notices(service, launcher)) == 2

    clock.passes(60)
    await looked(service)

    again = notices(service, launcher)[2:]
    assert len(again) == 2 and all("for 4 h" in text for text in again)


async def test_a_commit_a_submission_or_an_escalation_is_progress(
    service: DirectorService, launcher: FakeLauncher, clock: Clock, repo: Path
):
    await looked(service)
    clock.passes(100)
    commit(repo, "services/api/app.py", "order_id = '1'\n")
    await looked(service)
    await use(service, ic(service, WEB), RaiseEscalationTool, CENTS)
    clock.passes(100)
    await looked(service)

    # API committed 100 minutes ago; Web is blocked and waits on its answer.
    assert notices(service, launcher) == []

    clock.passes(20)
    await looked(service)

    [told] = notices(service, launcher)
    assert "Team 'API'" in told and "for 2 h" in told


async def test_a_team_that_submitted_is_not_stalled_and_one_sent_back_can_be(
    service: DirectorService, launcher: FakeLauncher, clock: Clock, repo: Path
):
    await looked(service)
    await api_submits(service, repo)
    await use(service, ic(service, WEB), SubmitReviewTool, DONE)
    clock.passes(500)
    await looked(service)
    assert notices(service, launcher) == []

    await use(service, ic(service), ReviewTeamTool, reject())
    clock.passes(119)
    await looked(service)
    assert notices(service, launcher) == []
    clock.passes(1)
    await looked(service)

    [told] = notices(service, launcher)
    assert "You sent its work back (review round 1)" in told


async def test_a_stalled_team_no_longer_holds_back_the_reviews_beside_it(
    service: DirectorService, launcher: FakeLauncher, clock: Clock, repo: Path
):
    await looked(service)
    await api_submits(service, repo)
    assert requests(service, launcher) == []

    clock.passes(120)
    await looked(service)

    [told] = notices(service, launcher)
    assert "Team 'Web'" in told
    [request] = requests(service, launcher)
    assert "Team 'API'" in request


async def test_a_team_in_a_turn_is_not_asked_and_the_check_can_be_turned_off(
    service: DirectorService, launcher: FakeLauncher, clock: Clock
):
    await looked(service)
    launcher.report(str(ic(service, API)), ConversationExecutionStatus.RUNNING)
    clock.passes(120)
    await looked(service)

    about_api = next(
        text for text in notices(service, launcher) if "Team 'API'" in text
    )
    assert "Its IC is in a turn right now, so it was not asked" in about_api
    assert UUID(str(ic(service, API))) not in [
        asked for asked, _ in launcher.agents_asked
    ]

    service.update_settings(DirectorSettingsUpdate(stall_check_minutes=0))
    clock.passes(1000)
    await looked(service)
    assert len(notices(service, launcher)) == 2


async def test_the_time_an_org_is_paused_does_not_count(
    service: DirectorService, launcher: FakeLauncher, clock: Clock
):
    await looked(service)
    clock.passes(100)
    await service.pause()
    clock.passes(500)
    await looked(service)
    await service.resume()
    await service.settled()
    clock.passes(100)
    await looked(service)

    assert notices(service, launcher) == []


async def test_the_time_a_server_was_not_running_does_not_count_either(
    repo: Path, launcher: FakeLauncher, clock: Clock
):
    first = await started(repo, launcher, clock=clock, stall_poll=3600)
    await split_root(first)
    await looked(first)
    clock.passes(100)
    await first.aclose()
    clock.passes(500)

    second = DirectorService(
        repo,
        save_delay=0,
        remembered=repo.parent / "state" / "director.json",
        clock=clock,
        stall_poll=3600,
    )
    await second.start(launcher)
    await looked(second)
    clock.passes(119)
    await looked(second)
    quiet = notices(second, launcher)
    clock.passes(1)
    await looked(second)

    assert quiet == [] and len(notices(second, launcher)) == 2
    await second.aclose()


async def test_an_ic_writes_to_the_ic_of_a_team_below_and_to_no_other(
    service: DirectorService, launcher: FakeLauncher
):
    hint = MsgTeamAction(team=API, text="Use the orders table that exists.")

    sent = await use(service, ic(service), MsgTeamTool, hint)
    by_manager = await use(service, manager(service), MsgTeamTool, hint)
    sideways = await use(service, ic(service, WEB), MsgTeamTool, hint)
    empty = await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text=" ")
    )

    assert not sent.is_error and "does not answer in a message" in sent.text
    assert by_manager.is_error and sideways.is_error and empty.is_error
    told = launcher.messages_to(ic(service, API))[-1]
    assert "The IC of the team above (Checkout) writes:" in told
    assert "Use the orders table that exists." in told
    for team in ("root", API):
        [thread] = [t for t in service.threads_of(node(service, team)) if t.ic_to_ic]
        assert (thread.manager_node_id, thread.ic_node_id) == (
            node(service, "root").id,
            node(service, API).id,
        )
        assert [message.text for message in thread.messages] == [hint.text]
    assert not any(t.ic_to_ic for t in service.threads_of(node(service, WEB)))


async def test_a_stopped_team_rests_with_all_below_it_and_is_sent_nothing(
    service: DirectorService, launcher: FakeLauncher
):
    db = [SplitChild(title="DB", path="services/api/db", task="Own the schema")]
    await use(service, ic(service, API), ProposeTeamsTool, proposing(db))
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    api_ic, db_ic = str(ic(service, API)), str(ic(service, "services.api.db"))
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)
    launcher.report(db_ic, ConversationExecutionStatus.RUNNING)
    heard = len(launcher.messages_to(db_ic))

    stopped = await use(service, ic(service), StopTeamTool, STOP_API)
    twice = await use(service, ic(service), StopTeamTool, STOP_API)
    by_manager = await use(service, manager(service), StopTeamTool, STOP_API)
    service.notify(db_ic, "Something for later.")
    await service.settled()

    assert not stopped.is_error and twice.is_error and by_manager.is_error
    assert set(launcher.paused) == {UUID(api_ic), UUID(db_ic)}
    api = node(service, API)
    assert api.status == "blocked" and api.stopped is not None
    assert api.stopped.reason == "Three hours on one failing test."
    assert len(launcher.messages_to(db_ic)) == heard
    to_manager = launcher.messages_to(manager(service))[-1]
    assert "Your IC stopped the team 'API'" in to_manager
    assert "Three hours on one failing test." in to_manager

    again = await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Try v2.")
    )
    await service.settled()

    assert "goes on now" in again.text and node(service, API).stopped is None
    assert node(service, API).status != "blocked"
    assert "Try v2." in launcher.messages_to(api_ic)[-1]
    assert "Something for later." in launcher.messages_to(db_ic)[-1]
    # The team below was cut off in a turn and runs again; the stopped
    # team's own IC is run by the message.
    assert launcher.resumed == [(UUID(db_ic), None)]


async def test_a_stopped_team_does_not_stall_or_hold_back_reviews_and_is_pruned(
    service: DirectorService, launcher: FakeLauncher, clock: Clock, repo: Path
):
    await looked(service)
    await use(service, ic(service, WEB), SubmitReviewTool, DONE)
    await use(service, ic(service), StopTeamTool, STOP_API)
    await service.settled()

    # Stopped counts as blocked: the work beside it goes to its reviewer.
    [request] = requests(service, launcher)
    assert "Team 'Web'" in request
    clock.passes(500)
    await looked(service)
    assert notices(service, launcher) == []

    pruned = await use(
        service,
        manager(service),
        PruneTeamTool,
        PruneTeamAction(team=API, reason="It is not coming back."),
    )
    await service.settled()

    assert not pruned.is_error and service.store.place(API) is None
    assert "pruned the team 'API'" in launcher.messages_to(ic(service))[-1]


async def test_teams_that_take_a_stopped_teams_place_replace_it(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), StopTeamTool, STOP_API)
    await service.settled()
    services = [SplitChild(title="Services", path="services", task="Serve")]

    await use(service, ic(service), ProposeTeamsTool, proposing(services, keep=["Web"]))
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    await service.settled()

    assert service.store.place(API) is None
    replacement = node(service, "services")
    assert replacement.stopped is None and replacement.status != "blocked"


async def test_the_director_looks_by_itself(
    repo: Path, launcher: FakeLauncher, clock: Clock
):
    service = await started(repo, launcher, clock=clock, stall_poll=0.01)
    await split_root(service)
    await asyncio.sleep(0.1)
    await service.settled()
    clock.passes(120)

    await asyncio.sleep(0.1)
    await service.settled()

    assert len(notices(service, launcher)) == 2
    await service.aclose()


async def test_a_stopped_team_is_still_stopped_after_a_restart(
    repo: Path, launcher: FakeLauncher, clock: Clock
):
    first = await started(repo, launcher, clock=clock, stall_poll=3600)
    await split_root(first)
    await use(first, ic(first), StopTeamTool, STOP_API)
    api_ic = str(ic(first, API))
    heard = len(launcher.messages_to(api_ic))
    first.notify(api_ic, "Something for later.")
    await first.settled()
    await first.aclose()

    launcher.listeners.clear()
    second = DirectorService(
        repo,
        save_delay=0,
        remembered=repo.parent / "state" / "director.json",
        clock=clock,
        stall_poll=3600,
    )
    await second.start(launcher)
    await second.settled()
    if second.summary().paused is not None:
        await second.resume()
        await second.settled()

    stopped = node(second, API).stopped
    assert stopped is not None and stopped.reason == STOP_API.reason
    assert len(launcher.messages_to(api_ic)) == heard

    await use(second, ic(second), MsgTeamTool, MsgTeamAction(team=API, text="Go on."))
    await second.settled()

    after = launcher.messages_to(api_ic)[heard:]
    assert len(after) == 2 and "Something for later." in after[0]
    assert "Go on." in after[1]
    await second.aclose()


async def test_a_team_set_up_and_given_no_task_is_told_to_the_ic_above(
    repo: Path, launcher: FakeLauncher, clock: Clock
):
    service = await started(repo, launcher, clock=clock, stall_poll=3600)
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    service.tell_ic(node(service, WEB).id, "Begin.")
    await service.settled()

    await looked(service)
    clock.passes(120)
    await looked(service)

    told = [
        text
        for text in launcher.messages_to(ic(service))
        if "has been given no task" in text
    ]
    assert len(told) == 1
    api = node(service, API).id
    assert f"Team 'API' (id {api}, services/api) was set up 2 h ago" in told[0]
    assert "`msg_team`" in told[0] and "pruned" in told[0]
    # Nobody is there to ask, and it is not counted as work that stalled.
    assert not any("Team 'API'" in text for text in notices(service, launcher))
    stalled = [e.team for e in record_of(service, "root") if e.kind == "stalled"]
    assert "API" not in stalled

    # Said once: a team may be left without a task on purpose, and an org
    # that was paused in between does not say it again.
    clock.passes(240)
    await looked(service)
    await service.pause()
    await service.resume()
    clock.passes(240)
    await looked(service)
    service.tell_ic(api, "Begin.")
    await service.settled()
    clock.passes(120)
    await looked(service)
    again = [
        text
        for text in launcher.messages_to(ic(service))
        if "has been given no task" in text
    ]
    assert len(again) == 1
    await service.aclose()


async def test_the_time_a_team_was_stopped_is_not_time_without_progress(
    service: DirectorService, launcher: FakeLauncher, clock: Clock
):
    db = [SplitChild(title="DB", path="services/api/db", task="Own the schema")]
    await use(service, ic(service, API), ProposeTeamsTool, proposing(db))
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    api_ic = str(ic(service, API))
    await looked(service)
    await use(service, ic(service), StopTeamTool, STOP_API)
    await service.settled()
    clock.passes(300)
    await looked(service)

    def stalls() -> list[str]:
        told = launcher.messages_to(api_ic)
        return [text for text in told if "has shown no progress" in text]

    before = stalls()
    await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Try v2.")
    )
    await service.settled()
    clock.passes(1)
    await looked(service)

    assert stalls() == before
