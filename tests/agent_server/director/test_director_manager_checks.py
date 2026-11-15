"""The looks a Manager takes at its teams without a message from its IC."""

from pathlib import Path

import pytest

from openhands.agent_server.director.models import DirectorSettingsUpdate, SplitChild
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgManagerTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ResolveEscalationTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    PruneTeamAction,
    RaiseEscalationAction,
    ResolveEscalationAction,
)
from tests.agent_server.director.conftest import FakeLauncher, tasked
from tests.agent_server.director.test_director_acp import started
from tests.agent_server.director.test_director_review import DONE, accept, commit
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    ic,
    manager,
    node,
    proposing,
    split_root,
    use,
)


API, WEB = "services.api", "apps.web"


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    service = await started(repo, launcher, stall_poll=3600)
    await split_root(service)
    yield service
    await service.aclose()


def looks(service: DirectorService, launcher: FakeLauncher, team: str = "root"):
    """The looks the director asked the Manager of a team to take."""
    return [
        text
        for text in launcher.messages_to(manager(service, team))
        if "A look at your teams" in text
    ]


async def accepted(service: DirectorService, repo: Path, team: str, times: int = 1):
    """Have a team submit work and the top team's IC accept it."""
    directory = node(service, team).boundary.path
    for _ in range(times):
        count = len(list((repo / directory).glob("*.txt"))) if directory else 0
        commit(repo, f"{directory}/part{count}.txt", "done\n")
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        await use(service, ic(service), ReviewTeamTool, accept(team))
    await service.settled()


async def escalates(service: DirectorService, team: str, times: int = 1) -> None:
    for number in range(times):
        await use(
            service,
            ic(service, team),
            RaiseEscalationTool,
            RaiseEscalationAction(category="design_decision", title=f"Q{number}?"),
        )
    await service.settled()


async def test_a_manager_looks_after_every_so_many_reviews_its_ic_accepted(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await accepted(service, repo, API, times=4)
    assert looks(service, launcher) == []

    await accepted(service, repo, WEB)

    [look] = looks(service, launcher)
    assert "your IC has accepted 5 reviews" in look
    assert "'API' (node services.api, services/api): " in look
    assert "review accepted, round 1" in look
    assert "Nothing is asked of you." in look

    await accepted(service, repo, API, times=4)
    assert len(looks(service, launcher)) == 1
    await accepted(service, repo, API)
    assert len(looks(service, launcher)) == 2


async def test_a_manager_looks_when_one_team_keeps_escalating_on_its_work(
    service: DirectorService, launcher: FakeLauncher
):
    await escalates(service, API, times=2)
    await escalates(service, WEB, times=2)
    # Four in all, but no team has raised three.
    assert looks(service, launcher) == []

    await escalates(service, API)

    [look] = looks(service, launcher)
    assert "the IC of 'API' has escalated 3 times to your IC" in look
    assert "3 escalated to your IC on this work; 3 unsettled" in look
    assert "2 escalated to your IC on this work; 2 unsettled" in look

    # Counted from nothing again for that team; Web's two still stand.
    await escalates(service, API, times=2)
    assert len(looks(service, launcher)) == 1
    await escalates(service, WEB)
    assert len(looks(service, launcher)) == 2


async def test_what_a_team_escalated_is_counted_for_the_work_it_has_in_hand(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await escalates(service, API, times=2)
    for escalation in node(service, API).escalations:
        await use(
            service,
            ic(service),
            ResolveEscalationTool,
            ResolveEscalationAction(
                node_id=API, escalation_id=escalation.id, action="settle", text="Yes."
            ),
        )
    await accepted(service, repo, API)

    # Its work was accepted: what it raises now is on other work.
    await escalates(service, API, times=2)

    assert looks(service, launcher) == []
    await escalates(service, API)
    assert len(looks(service, launcher)) == 1


async def test_the_two_counts_are_kept_apart(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    service.update_settings(
        DirectorSettingsUpdate(
            manager_check_accepted_reviews=2, manager_check_ic_escalations=2
        )
    )
    await escalates(service, WEB)
    await accepted(service, repo, API, times=2)
    [by_reviews] = looks(service, launcher)

    # The look for the reviews did not use up what Web has escalated.
    await escalates(service, WEB)
    by_escalations = looks(service, launcher)[1:]
    # And that look did not use up the reviews counted since.
    await accepted(service, repo, API)
    still = len(looks(service, launcher))
    await accepted(service, repo, API)

    assert "accepted 2 reviews" in by_reviews
    assert (
        len(by_escalations) == 1
        and "'Web' has escalated 2 times" in (by_escalations[0])
    )
    assert (still, len(looks(service, launcher))) == (2, 3)


async def test_each_check_can_be_turned_off(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    service.update_settings(
        DirectorSettingsUpdate(
            manager_check_accepted_reviews=0, manager_check_ic_escalations=0
        )
    )

    await accepted(service, repo, API, times=6)
    await escalates(service, WEB, times=4)

    assert looks(service, launcher) == []


async def test_an_escalation_passed_up_counts_for_the_team_it_came_through(
    service: DirectorService, launcher: FakeLauncher
):
    db = [SplitChild(title="DB", path="services/api/db", task="Own the schema")]
    await use(service, ic(service, API), MsgManagerTool, proposing(db))
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    await escalates(service, "services.api.db", times=3)

    # Three reached the API team's IC: its Manager looks.
    [below] = looks(service, launcher, API)
    assert "the IC of 'DB' has escalated 3 times" in below
    assert looks(service, launcher) == []

    for escalation in node(service, "services.api.db").escalations:
        await use(
            service,
            ic(service, API),
            ResolveEscalationTool,
            ResolveEscalationAction(
                node_id="services.api.db",
                escalation_id=escalation.id,
                action="pass_up",
                text="Not mine to decide.",
            ),
        )
    await service.settled()

    [above] = looks(service, launcher)
    assert "the IC of 'API' has escalated 3 times" in above


async def test_what_was_counted_is_kept_over_a_restart(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher, stall_poll=3600)
    await split_root(first)
    await escalates(first, API, times=2)
    await first.aclose()

    launcher.listeners.clear()
    second = DirectorService(
        repo,
        save_delay=0,
        remembered=repo.parent / "state" / "director.json",
        stall_poll=3600,
    )
    await second.start(launcher)
    await second.settled()
    if second.summary().paused is not None:
        await second.resume()
        await second.settled()
    await escalates(second, API)

    assert len(looks(second, launcher)) == 1
    await second.aclose()


async def test_work_that_goes_up_without_a_review_is_not_counted(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    service.update_settings(DirectorSettingsUpdate(manager_check_accepted_reviews=1))
    db = [SplitChild(title="DB", path="services/api/db", task="Own the schema")]
    await use(service, ic(service, API), MsgManagerTool, proposing(db))
    await use(service, manager(service, API), ApproveTeamsTool, APPROVE)
    await tasked(service)
    commit(repo, "services/api/db/schema.sql", "create table orders ();\n")
    await use(service, ic(service, "services.api.db"), SubmitReviewTool, DONE)

    await use(service, ic(service, API), ReviewTeamTool, accept("services.api.db"))
    await service.settled()

    api = node(service, API).review
    assert api is not None and api.passed_up
    # The API team's IC accepted one review; the top team's IC none.
    assert len(looks(service, launcher, API)) == 1
    assert looks(service, launcher) == []


async def test_a_new_team_in_a_removed_teams_place_starts_from_nothing(
    service: DirectorService, launcher: FakeLauncher
):
    await escalates(service, API, times=2)
    await use(
        service,
        manager(service),
        PruneTeamTool,
        PruneTeamAction(node_id=API, reason="Going nowhere."),
    )
    await service.settled()
    again = [SplitChild(title="Orders", path="services/api", task="Take orders")]
    await use(service, ic(service), MsgManagerTool, proposing(again, keep=["Web"]))
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
    await service.settled()

    # The new team has the removed one's node id, and none of its count.
    assert node(service, API).title == "Orders"
    await escalates(service, API, times=2)
    assert looks(service, launcher) == []
    await escalates(service, API)
    assert len(looks(service, launcher)) == 1
