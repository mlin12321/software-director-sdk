"""What a Manager and an IC are given to judge the make-up of their teams
by."""

from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    ReviewFinding,
    SplitChild,
    TeamEvent,
)
from openhands.agent_server.director.org_store import team_id_of
from openhands.agent_server.director.prompts import (
    needs_between,
    success_rate,
    track_records,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgManagerTool,
    PruneTeamTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgManagerAction,
    PruneTeamAction,
    ReviewTeamAction,
)
from tests.agent_server.director.conftest import FakeLauncher
from tests.agent_server.director.test_director_acp import started
from tests.agent_server.director.test_director_manager_checks import (
    accepted,
    escalates,
    looks,
    sent_back,
)
from tests.agent_server.director.test_director_review import (
    DONE,
    commit,
    requests,
    run_git,
)
from tests.agent_server.director.test_director_service import attempt, create
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    SPLIT,
    ic,
    manager,
    node,
    proposing,
    split_root,
    sub_teams,
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
        (["set_up", "sent_back", "accepted"], "success rate 1 of 1 task (100%)"),
        (
            ["set_up", "accepted", "accepted", "woken"],
            "success rate 2 of 3 tasks (67%)",
        ),
        # Reached twice on one piece of work is one task.
        (
            ["set_up", "woken", "woken", "accepted", "accepted"],
            "success rate 1 of 2 tasks (50%)",
        ),
        # A team set up in the place of one that is gone starts its own.
        (
            ["set_up", "woken", "replaced", "set_up", "accepted"],
            "success rate 1 of 1 task (100%)",
        ),
    ],
)
def test_a_teams_success_rate_is_its_tasks_accepted_without_reaching_the_limit(
    kinds: list[str], ratio: str
):
    [record] = track_records([event(kind) for kind in kinds])

    assert success_rate(record) == ratio


async def test_a_look_tells_the_success_rate_of_each_team(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await accepted(service, repo, WEB, times=2)
    await escalates(service, API, times=3)

    [look] = looks(service, launcher)
    web = next(line for line in look.splitlines() if line.startswith("- 'Web'"))
    api = next(line for line in look.splitlines() if line.startswith("- 'API'"))
    assert web.endswith("success rate 2 of 2 tasks (100%)")
    assert api.endswith("success rate 0 of 1 task (0%)")
    # The number it is counted against is the one that wakes the Manager.
    limit = service.settings.manager_check_ic_setbacks
    assert (
        "A team's success rate is the share of its tasks that were accepted "
        f"without its IC escalating {limit} times on them"
    ) in look
    assert "wait until a team has had enough tasks to judge it by" in look
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


def record_of_api(service: DirectorService) -> dict:
    [api] = [
        record
        for record in track_records(service.team_memory.history("root"))
        if record["node_id"] == API
    ]
    return api


async def test_work_sent_back_less_often_than_the_limit_wakes_no_manager(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await sent_back(service, repo, API, times=2)

    assert looks(service, launcher) == []
    assert record_of_api(service)["sent_back"] == 2


async def test_work_sent_back_as_often_as_the_limit_wakes_the_manager_once(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await sent_back(service, repo, API, times=4)

    [look] = looks(service, launcher)
    assert "your IC has sent the work of 'API' back 3 times." in look
    assert "its work being sent back 3 times" in look
    # The task counts against the team, also once it is accepted.
    assert success_rate(record_of_api(service)) == "success rate 0 of 1 task (0%)"
    await accepted(service, repo, API)
    assert success_rate(record_of_api(service)) == "success rate 0 of 1 task (0%)"


async def test_send_backs_wake_nobody_when_the_check_is_off(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    service.update_settings(DirectorSettingsUpdate(manager_check_ic_setbacks=0))

    await sent_back(service, repo, API, times=3)

    assert looks(service, launcher) == []


async def test_a_manager_may_prune_a_team_with_nothing_against_it(
    service: DirectorService, repo: Path
):
    await accepted(service, repo, API)

    pruned = await use(service, manager(service), PruneTeamTool, PRUNE)

    assert not pruned.is_error
    assert [team_id_of(team) for team in sub_teams(service)] == [WEB]


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


def needed(team: str, other: str, how: str = "write refused") -> TeamEvent:
    return TeamEvent.model_validate(
        {
            "at": "2026-11-15T10:00:00+00:00",
            "kind": "needed",
            "team_id": team,
            "team": team,
            "other_id": other,
            "other": other,
            "detail": how,
        }
    )


def test_teams_that_needed_each_other_are_counted_as_a_pair_while_both_are_there():
    counted = needs_between(
        [
            event("set_up", "a"),
            event("set_up", "b"),
            event("set_up", "c"),
            needed("a", "b"),
            needed("b", "a", "review finding"),
            needed("a", "b"),
            needed("c", "a"),
            # A team set up anew in c's place starts from nothing.
            event("replaced", "c"),
            event("set_up", "c"),
        ]
    )

    assert counted == [
        {
            "node_ids": ["a", "b"],
            "teams": ["a", "b"],
            "writes_refused": 2,
            "review_findings": 1,
            "requests": 0,
        }
    ]


def refused_in_web(service: DirectorService, repo: Path, name: str) -> str | None:
    """API's IC tries to write a file of Web."""
    path = repo / "apps" / "web" / name
    return attempt(service, ic(service, API), "file_editor", create(path))


async def test_a_refused_write_into_a_team_beside_is_kept_once_for_each_file(
    service: DirectorService, repo: Path
):
    assert refused_in_web(service, repo, "page.js") is not None
    assert refused_in_web(service, repo, "page.js") is not None
    # A file that is no team's says nothing of two teams.
    attempt(service, ic(service, API), "file_editor", create(repo / "README.md"))
    await service.settled()

    [kept] = [e for e in service.team_memory.history("root") if e.kind == "needed"]
    assert (kept.team_id, kept.other_id, kept.other) == (API, WEB, "Web")
    assert (kept.detail, kept.paths) == ("write refused", ["apps/web/page.js"])
    # Once is not worth saying.
    assert service.needs_below("root") == []


async def test_a_need_is_kept_once_across_a_restart_of_the_server(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    refused_in_web(service, repo, "page.js")
    await service.settled()
    again = DirectorService(
        repo,
        save_delay=0,
        remembered=repo.parent / "state" / "director.json",
        server_url="http://127.0.0.1:8000",
    )
    await again.start(launcher)
    try:
        refused_in_web(again, repo, "page.js")
        refused_in_web(again, repo, "other.js")
        await again.settled()
        needed_files = [
            event.paths
            for event in again.team_memory.history("root")
            if event.kind == "needed"
        ]
    finally:
        await again.aclose()

    assert needed_files == [["apps/web/page.js"], ["apps/web/other.js"]]


async def test_an_ic_is_told_of_teams_that_keep_needing_each_other(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    for name in ("a.js", "b.js"):
        refused_in_web(service, repo, name)
    commit(repo, "services/api/app.py", "orders = []\n")
    await use(service, ic(service, API), SubmitReviewTool, DONE)
    finding = ReviewFinding(path="apps/web/form.js", issue="Posts the wrong field.")
    await use(
        service,
        ic(service),
        ReviewTeamTool,
        ReviewTeamAction(node_id=API, decision="reject", findings=[finding]),
    )
    await use(service, ic(service, API), SubmitReviewTool, DONE)
    # A review is asked for once every team below has submitted.
    commit(Path(node(service, WEB).worktree_path), "apps/web/page.js", "x = 1\n")
    await use(service, ic(service, WEB), SubmitReviewTool, DONE)

    words = (
        "needed each other's directory 3 times (2 writes refused, 1 review "
        "finding): weigh whether the two are one concern"
    )
    asked = "\n".join(requests(service, launcher))
    assert f"Its work so far: 1 sent back; it and 'Web' {words}." in asked
    assert f"Its work so far: it and 'API' {words}." in asked
    context = service.context_of(node(service, "root"), "ic", in_full=True)
    assert context["teams_below_that_needed_each_other"] == [f"'API' and 'Web' {words}"]


async def test_a_manager_is_told_how_often_teams_replaced_together_needed_each_other(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    refused_in_web(service, repo, "page.js")
    redrawn = [
        SplitChild(title="Services", path="services", task="Serve the shop"),
        SplitChild(title="Apps", path="apps", task="Show the shop"),
    ]

    await use(service, ic(service), MsgManagerTool, proposing(redrawn))

    told = launcher.messages_to(manager(service))[-1]
    assert (
        "- 'API' and 'Web' needed each other's directory 1 time (1 write "
        "refused): weigh whether the two are one concern."
    ) in told
