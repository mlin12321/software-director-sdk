"""Limits on new teams, and many teams at work at once."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    CreateOrgRequest,
    DirectorSettingsUpdate,
    RouteMessageRequest,
    SplitChild,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    DecideSplitTool,
    ProposeBoundarySplitTool,
    SetStatusTool,
)
from openhands.agent_server.director.tools.definition import (
    DecideSplitAction,
    ProposeBoundarySplitAction,
    SetStatusAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_node,
    root_conversation,
    run_git,
)
from tests.agent_server.director.test_director_tools import (
    ic,
    manager,
    node,
    split_root,
    use,
)


APPROVE = DecideSplitAction(decision="approve")


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


async def started(repo: Path, launcher: FakeLauncher, **options) -> DirectorService:
    service = DirectorService(
        repo,
        save_delay=0,
        remembered=repo.parent / "state" / "director.json",
        **{"run_limit_retry": 0, **options},
    )
    await service.start(launcher)
    await service.create_org(
        CreateOrgRequest(
            title="Checkout", task="Build", conversation=root_conversation(repo)
        )
    )
    await service.settled()
    return service


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    service = await started(repo, launcher)
    yield service
    await service.aclose()


def split(below: str, count: int) -> ProposeBoundarySplitAction:
    return ProposeBoundarySplitAction(
        children=[
            SplitChild(title=f"{below} {n}", path=f"{below}/t{n}", task="Work")
            for n in range(count)
        ],
        contracts="",
    )


async def eventually(condition) -> None:
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("It never happened")


async def test_a_split_that_gives_a_team_too_many_agents_is_refused_and_says_what_fits(
    service: DirectorService,
):
    service.update_settings(DirectorSettingsUpdate(max_agents=8))

    refused = await use(service, ic(service), ProposeBoundarySplitTool, split("a", 4))

    assert refused.is_error and node(service, "root").pending_split is None
    assert "at most 8 agents" in refused.text and "has 2" in refused.text
    assert "At most 3 more sub-teams fit" in refused.text

    fits = await use(service, ic(service), ProposeBoundarySplitTool, split("a", 3))
    await use(service, manager(service), DecideSplitTool, APPROVE)

    assert not fits.is_error and len(node(service, "root").child_ids) == 3


async def test_the_agents_of_the_teams_above_count_too(service: DirectorService):
    await split_root(service)
    service.update_settings(DirectorSettingsUpdate(max_agents=8))

    refused = await use(
        service,
        ic(service, "services.api"),
        ProposeBoundarySplitTool,
        split("services/api", 2),
    )

    assert refused.is_error and "'Checkout' has 6" in refused.text
    assert "At most 1 more sub-teams fit" in refused.text


async def test_teams_cannot_go_deeper_than_the_limit(service: DirectorService):
    service.update_settings(DirectorSettingsUpdate(max_depth=2))
    await split_root(service)

    refused = await use(
        service,
        ic(service, "services.api"),
        ProposeBoundarySplitTool,
        split("services/api", 1),
    )

    assert refused.is_error and "at most 2 levels" in refused.text
    assert "would be level 3" in refused.text


async def test_a_lowered_limit_removes_nothing_and_stops_the_next_split(
    service: DirectorService,
):
    await split_root(service)

    state = service.update_settings(DirectorSettingsUpdate(max_agents=2, max_depth=1))

    assert (state.agents, state.depth) == (6, 2)
    assert len(service.store) == 3
    refused = await use(service, ic(service), ProposeBoundarySplitTool, split("a", 1))
    assert refused.is_error


async def test_a_split_approved_after_the_org_grew_goes_back_to_the_ic(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), ProposeBoundarySplitTool, split("a", 3))
    service.update_settings(DirectorSettingsUpdate(max_agents=4))

    await use(service, manager(service), DecideSplitTool, APPROVE)

    root = node(service, "root")
    assert root.child_ids == [] and root.pending_split is not None
    assert root.pending_split.status == "refine_requested"
    assert "cannot be carried out" in launcher.messages_to(ic(service))[-1]
    assert "cannot be carried out" in launcher.messages_to(manager(service))[-1]


async def test_settings_are_kept_for_the_next_start_beside_the_repository(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    first.update_settings(DirectorSettingsUpdate(max_depth=3))
    await first.aclose()
    kept = json.loads((repo.parent / "state" / "director.json").read_text())

    again = DirectorService(repo, remembered=repo.parent / "state" / "director.json")

    assert kept["settings"] == {
        "max_agents": 64,
        "max_depth": 3,
        "allowed_profiles": [],
    }
    assert kept["repo_path"] == str(repo)
    assert again.settings.max_depth == 3 and again.store.root_id == "root"


async def test_a_team_is_found_by_words_of_its_title_or_directory(
    service: DirectorService,
):
    await split_root(service)

    assert [found.id for found in service.search("API", 5)] == ["services.api"]
    assert [found.id for found in service.search("apps web", 5)] == ["apps.web"]
    assert service.search("  ", 5) == [] and service.search("nothing", 5) == []
    assert len(service.search("e", 1)) == 1


async def test_a_team_whose_review_is_slow_does_not_hold_up_another_team(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "app.py").write_text("x = 1\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "--quiet", "-m", "work")
    launcher.agents_silent = asyncio.Event()
    done = asyncio.create_task(
        use(
            service,
            ic(service, "services.api"),
            SetStatusTool,
            SetStatusAction(status="done", summary="Shipped."),
        )
    )
    await eventually(lambda: launcher.agents_asked)

    await service.route_message(RouteMessageRequest(text="First", path="apps/web"))
    await service.route_message(RouteMessageRequest(text="Second", path="apps/web"))
    await eventually(lambda: len(launcher.messages_to(ic(service, "apps.web"))) == 2)

    assert launcher.messages_to(ic(service, "apps.web")) == ["First", "Second"]
    assert not done.done()
    launcher.agents_silent.set()
    await done
    await service.aclose()


async def test_teams_wait_for_a_free_run_and_start_when_one_ends(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher, run_limit_retry=3600)
    launcher.full_for = 1_000_000
    await use(service, ic(service), ProposeBoundarySplitTool, split("a", 2))
    approve = asyncio.create_task(
        use(service, manager(service), DecideSplitTool, APPROVE)
    )
    await eventually(
        lambda: (
            [team.waiting_to_start for team in service.store.children("root")]
            == [True, True]
        )
    )
    root = node(service, "root")
    assert root.pending_split is not None and len(launcher.derived) == 1

    launcher.full_for = 0
    root_ic = ic(service)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.IDLE)
    await approve

    teams = service.store.children("root")
    assert [team.waiting_to_start for team in teams] == [False, False]
    assert len(launcher.derived) == 5 and node(service, "root").pending_split is None
    await service.aclose()


def test_a_thousand_agents_changing_status_cost_little():
    store = OrgStore()
    nodes = [make_node("root", None, path="")]
    for a in range(10):
        nodes.append(make_node(f"d{a}", "root", path=f"d{a}"))
        for b in range(7):
            nodes.append(make_node(f"d{a}.s{b}", f"d{a}", path=f"d{a}/s{b}"))
            for c in range(6):
                nodes.append(
                    make_node(f"d{a}.s{b}.m{c}", f"d{a}.s{b}", path=f"d{a}/s{b}/m{c}")
                )
    store.replace_all(nodes)
    seen: list[str] = []
    for focus in ("root", "d3", "d3.s2"):
        store.subscribe(focus, lambda event: seen.append(event.type))
    store.subscribe("root", lambda event: seen.append(event.type), full=True)
    leaves = [item.id for item in nodes if item.id.count(".") == 2]
    assert len(nodes) == 501

    began = time.perf_counter()
    for turn in range(10):
        for leaf in leaves:
            store.update_node(leaf, status="working" if turn % 2 == 0 else "idle")
    spent = time.perf_counter() - began

    assert len(seen) > 4200 and "escalations" not in seen
    # 4,200 changes; about 1 s on a laptop. The bound only catches a change
    # that makes each one cost the whole tree.
    assert spent < 15
