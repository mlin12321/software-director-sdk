"""What a Manager and an IC are given to judge the make-up of their teams
by, and when a prune is refused."""

from pathlib import Path

import pytest

from openhands.agent_server.director.models import TeamEvent
from openhands.agent_server.director.prompts import task_ratio, track_records
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgManagerTool,
    PruneTeamTool,
    ReviewTeamTool,
    StopTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgManagerAction,
    PruneTeamAction,
    StopTeamAction,
)
from tests.agent_server.director.conftest import FakeLauncher
from tests.agent_server.director.test_director_acp import started
from tests.agent_server.director.test_director_manager_checks import (
    accepted,
    escalates,
    looks,
)
from tests.agent_server.director.test_director_review import (
    DONE,
    commit,
    reject,
    run_git,
)
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    SPLIT,
    ic,
    manager,
    node,
    split_root,
    use,
)


API, WEB = "services.api", "apps.web"
PRUNE = PruneTeamAction(node_id=API, reason="It holds everything up.")


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    service = await started(repo, launcher, stall_poll=3600)
    await split_root(service)
    yield service
    await service.aclose()


def event(kind: str, team: str = "a") -> TeamEvent:
    return TeamEvent.model_validate(
        {"at": "2026-11-15T10:00:00+00:00", "kind": kind, "team_id": team, "team": team}
    )


@pytest.mark.parametrize(
    ("kinds", "ratio"),
    [
        (["set_up"], ""),
        (["set_up", "sent_back", "accepted"], "1 of 1 task"),
        (["set_up", "accepted", "accepted", "woken"], "2 of 3 tasks"),
        # Reached twice on one piece of work is one task.
        (["set_up", "woken", "woken", "accepted", "accepted"], "1 of 2 tasks"),
        # A team set up in the place of one that is gone starts its own.
        (["set_up", "woken", "replaced", "set_up", "accepted"], "1 of 1 task"),
    ],
)
def test_a_teams_tasks_are_counted_with_the_ones_that_reached_no_limit(
    kinds: list[str], ratio: str
):
    [record] = track_records([event(kind) for kind in kinds])

    assert task_ratio(record).startswith(ratio)
    assert not ratio or "without its escalations reaching the limit" in task_ratio(
        record
    )


async def test_a_look_says_how_many_of_each_teams_tasks_reached_no_limit(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await accepted(service, repo, WEB, times=2)
    await escalates(service, API, times=3)

    [look] = looks(service, launcher)
    web = next(line for line in look.splitlines() if line.startswith("- 'Web'"))
    api = next(line for line in look.splitlines() if line.startswith("- 'API'"))
    assert "2 of 2 tasks without its escalations reaching the limit" in web
    assert "0 of 1 task without its escalations reaching the limit" in api
    # The limit being reached is kept with the team it was reached by.
    [woken] = [e for e in service.team_memory.history("root") if e.kind == "woken"]
    assert (woken.team_id, woken.team) == (API, "API")
    assert "has escalated 3 times" in woken.detail
    assert "nothing here is a call to prune" in look
    assert "Nothing went wrong" not in look


async def test_a_look_after_accepted_work_says_that_nothing_went_wrong(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await accepted(service, repo, API, times=5)

    [look] = looks(service, launcher)
    assert "Nothing went wrong: this look comes after work that was accepted." in look
    assert "grossly inefficient" in look


async def test_work_that_is_sent_back_is_kept_and_wakes_no_manager(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    for number in range(4):
        commit(repo, f"services/api/try{number}.txt", "again\n")
        await use(service, ic(service, API), SubmitReviewTool, DONE)
        await use(service, ic(service), ReviewTeamTool, reject(API))
    await service.settled()

    assert looks(service, launcher) == []
    assert service.record_of(node(service, API))["sent_back"] == 4


async def test_a_team_with_nothing_against_it_cannot_be_pruned(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await accepted(service, repo, API)

    refused = await use(service, manager(service), PruneTeamTool, PRUNE)

    assert refused.is_error
    assert (
        "Refused: nothing on record says 'API' is grossly inefficient" in refused.text
    )
    assert "A prune is not a precaution" in refused.text
    assert node(service, "root").child_ids == [API, WEB]
    assert launcher.deleted == []


async def test_a_team_its_ic_stopped_can_be_pruned(service: DirectorService):
    await use(
        service,
        ic(service),
        StopTeamTool,
        StopTeamAction(team=API, reason="Three hours on one failing test."),
    )

    pruned = await use(service, manager(service), PruneTeamTool, PRUNE)

    assert not pruned.is_error
    assert node(service, "root").child_ids == [WEB]


async def test_a_team_whose_escalations_reached_the_limit_can_be_pruned(
    service: DirectorService,
):
    await escalates(service, API, times=3)

    pruned = await use(service, manager(service), PruneTeamTool, PRUNE)

    assert not pruned.is_error


async def test_a_team_that_was_never_given_a_task_can_be_pruned(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher, stall_poll=3600)
    await use(service, ic(service), MsgManagerTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    pruned = await use(service, manager(service), PruneTeamTool, PRUNE)

    assert not pruned.is_error
    await service.aclose()


ORDERS = """\
One team for the orders, apart from the pages.

```teams
reason: Orders and pages change for different reasons.
keep: [Web]
teams:
  - name: Orders
    path: services/api
    instruction: Keep the orders.
    reason: The order rules are a concern of their own.
```"""


async def test_a_manager_is_told_what_the_runtime_knows_of_proposed_teams(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    commit(repo, "services/api/app.py", "orders = []\n")
    commit(repo, "services/api/db.py", "rows = []\n")
    await escalates(service, API, times=3)
    await use(service, manager(service), PruneTeamTool, PRUNE)
    await service.settled()

    await use(service, ic(service), MsgManagerTool, MsgManagerAction(text=ORDERS))

    told = launcher.messages_to(manager(service))[-1]
    assert "What the runtime knows of them:" in told
    assert (
        "The IC's reason for this make-up: Orders and pages change for "
        "different reasons."
    ) in told
    assert "- 'Orders' (services/api): 2 files there now." in told
    assert "The IC's reason: The order rules are a concern of their own." in told
    assert (
        "- A team 'API' had services/api before and was pruned: It holds everything up."
    ) in told
    proposal = node(service, "root").pending_split
    assert proposal is not None
    assert proposal.reason == "Orders and pages change for different reasons."
    assert proposal.children[0].reason == "The order rules are a concern of their own."


def both(repo: Path, number: int) -> None:
    """Change a file of each directory in one commit."""
    paths = [f"services/api/b{number}.py", f"apps/web/v{number}.js"]
    for path in paths:
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x = 1\n")
    run_git(repo, "add", *paths)
    run_git(repo, "commit", "--quiet", "-m", "Change both")


async def test_directories_that_were_changed_together_are_pointed_out(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher, stall_poll=3600)
    commit(repo, "services/api/alone.py", "x = 1\n")
    for number in range(3):
        both(repo, number)

    await use(service, ic(service), MsgManagerTool, SPLIT)

    told = launcher.messages_to(manager(service))[-1]
    # Fewer than three says nothing: a first commit holds everything.
    assert (
        "- services/api and apps/web were changed in the same commit 3 times: "
        "weigh whether they are one concern."
    ) in told
    await service.aclose()


async def test_what_an_ic_proposed_and_what_was_approved_is_kept(
    service: DirectorService,
):
    await use(service, ic(service), MsgManagerTool, MsgManagerAction(text=ORDERS))
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await service.settled()

    kept = [
        (event.kind, event.detail)
        for event in service.team_memory.history("root")
        if event.kind in ("proposed", "approved")
    ]
    assert kept[-2:] == [
        ("proposed", "Orders (services/api); in place of API"),
        ("approved", "Orders (services/api); in place of API"),
    ]
