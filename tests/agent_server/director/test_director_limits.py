"""Limits on new teams, and many teams at work at once."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.models import (
    CreateOrgRequest,
    DirectorSettingsUpdate,
    SplitChild,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    ProposeTeamsAction,
    ReviewTeamAction,
    SubmitReviewAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    FakeLauncher,
    root_conversation,
    run_git,
    tasked,
)
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    division_lead_ic,
    ic,
    manager,
    node,
    proposing,
    split_root,
    use,
    woken,
)


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


def split(below: str, count: int) -> ProposeTeamsAction:
    return proposing(
        [
            SplitChild(title=f"{below} {n}", path=f"{below}/t{n}", task="Work")
            for n in range(count)
        ]
    )


async def accepted(service: DirectorService) -> None:
    """Have the work of the session opened first accepted, so that the org
    has that session's sub-teams."""
    await use(
        service, ic(service), SubmitReviewTool, SubmitReviewAction(summary="Done.")
    )
    await use(
        service,
        division_lead_ic(service),
        ReviewTeamTool,
        ReviewTeamAction(team=node(service, "root").id, decision="accept"),
    )


async def eventually(condition) -> None:
    for _ in range(400):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("It never happened")


async def test_a_split_that_makes_too_many_teams_is_refused_and_says_what_fits(
    service: DirectorService,
):
    service.update_settings(DirectorSettingsUpdate(max_teams=4))

    refused = await use(service, ic(service), ProposeTeamsTool, split("a", 4))

    assert refused.is_error and node(service, "root").pending_split is None
    assert "at most 4 teams" in refused.text and "'Checkout' is 1" in refused.text
    assert "At most 3 more sub-teams fit" in refused.text

    fits = await use(service, ic(service), ProposeTeamsTool, split("a", 3))
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    assert not fits.is_error and len(node(service, "root").child_ids) == 3


async def test_the_teams_above_count_too(service: DirectorService):
    await split_root(service)
    service.update_settings(DirectorSettingsUpdate(max_teams=4))

    refused = await use(
        service,
        ic(service, "services.api"),
        ProposeTeamsTool,
        split("services/api", 2),
    )

    assert refused.is_error and "'Checkout' is 3" in refused.text
    assert "At most 1 more sub-teams fit" in refused.text


async def test_teams_cannot_go_deeper_than_the_limit(service: DirectorService):
    service.update_settings(DirectorSettingsUpdate(max_depth=2))
    await split_root(service)

    refused = await use(
        service,
        ic(service, "services.api"),
        ProposeTeamsTool,
        split("services/api", 1),
    )

    assert refused.is_error and "at most 2 levels" in refused.text
    assert "would be level 3" in refused.text


async def test_a_lowered_limit_removes_nothing_and_stops_the_next_split(
    service: DirectorService,
):
    await split_root(service)
    await accepted(service)

    state = service.update_settings(DirectorSettingsUpdate(max_teams=1, max_depth=1))

    assert (state.teams, state.depth) == (3, 2)
    assert len(service.store) == 3
    refused = await use(service, ic(service), ProposeTeamsTool, split("a", 1))
    assert refused.is_error


async def test_sub_teams_approved_after_the_limits_were_lowered_go_back_to_the_ic(
    service: DirectorService, launcher: FakeLauncher
):
    await use(service, ic(service), ProposeTeamsTool, split("a", 3))
    service.update_settings(DirectorSettingsUpdate(max_teams=2))

    told_the_manager = launcher.messages_to(manager(service))

    refused = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    root = node(service, "root")
    assert refused.is_error and "at most 2 teams" in refused.text
    assert root.child_ids == [] and root.pending_split is not None
    assert root.pending_split.status == "proposed"
    assert launcher.messages_to(manager(service)) == told_the_manager


async def test_settings_are_kept_for_the_next_start_beside_the_repository(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    first.update_settings(DirectorSettingsUpdate(max_depth=3))
    await first.aclose()
    kept = json.loads((repo.parent / "state" / "director.json").read_text())

    again = DirectorService(repo, remembered=repo.parent / "state" / "director.json")

    assert kept["settings"] == {
        "max_teams": 32,
        "max_depth": 3,
        "allowed_profiles": [],
        "default_agent_type": "software-engineer",
        "allowed_agent_types": [],
        "resume_on_start": False,
        "report_compaction_chars": 24_000,
        "soft_check_in_minutes": 120,
        "hard_check_in_minutes": 360,
        "manager_check_accepted_reviews": 5,
        "manager_check_ic_setbacks": 3,
        "route_requests": True,
        "session_cost_limit": 0.0,
        "max_open_sessions": 0,
    }
    assert kept["repo_path"] == str(repo)
    assert again.settings.max_depth == 3 and again.store.root_id == "root"


async def test_a_limit_kept_as_agents_is_no_longer_read(
    repo: Path, launcher: FakeLauncher
):
    remembered = repo.parent / "state" / "director.json"
    remembered.parent.mkdir(parents=True, exist_ok=True)
    remembered.write_text(json.dumps({"settings": {"max_agents": 10}}))

    service = DirectorService(repo, remembered=remembered)

    assert service.settings.max_teams == 32


async def test_a_team_is_found_by_words_of_its_title_or_directory(
    service: DirectorService,
):
    await split_root(service)
    await accepted(service)

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
            SubmitReviewTool,
            SubmitReviewAction(summary="Shipped."),
        )
    )
    await eventually(lambda: launcher.agents_asked)

    web = node(service, "apps.web").id
    service.tell_ic(web, "First", human=True)
    service.tell_ic(web, "Second", human=True)
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
    await woken(service)
    launcher.full_for = 1_000_000
    await use(service, ic(service), ProposeTeamsTool, split("a", 2))
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    top = node(service, "root").id
    for team in service.store.below(top):
        service.tell_ic(team.id, "Begin.")
    await eventually(
        lambda: (
            [team.waiting_to_start for team in service.store.below(top)] == [True, True]
        )
    )
    assert len(launcher.derived) == 1

    launcher.full_for = 0
    root_ic = ic(service)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.IDLE)
    await service.settled()

    teams = service.store.below(top)
    assert [team.waiting_to_start for team in teams] == [False, False]
    assert len(launcher.derived) == 3 and node(service, "root").pending_split is None
    await service.aclose()
