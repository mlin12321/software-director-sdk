"""Sub-teams that take the place of ones a team already has."""

from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.models import SplitChild
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgIcTool,
    MsgManagerTool,
    PruneTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import MsgIcAction
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import FakeLauncher, run_git
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_pause import (
    again,
    cut_off,
    end_server_at,
)
from tests.agent_server.director.test_director_review import (
    DONE,
    commit,
    in_own_worktree,
)
from tests.agent_server.director.test_director_tools import (
    APPROVE,
    PRUNE_API,
    ic,
    manager,
    node,
    proposing,
    split_root,
    use,
)


API = "services.api"
DB = "services.api.db"
SERVICES = proposing([SplitChild(title="Services", path="services", task="Serve")])


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher):
    """An org of API (with DB below it) and Web, Web in its own checkout."""
    service = await started(repo, launcher)
    await split_root(service)
    await propose_and_approve(
        service,
        API,
        SplitChild(title="DB", path="services/api/db", task="Own the schema"),
    )
    yield service
    await service.aclose()


async def propose_and_approve(
    service: DirectorService, node_id: str, *children: SplitChild
) -> None:
    await use(service, ic(service, node_id), MsgManagerTool, proposing(list(children)))
    await use(service, manager(service, node_id), ApproveTeamsTool, APPROVE)


def below(service: DirectorService, node_id: str) -> list[str]:
    return sorted(team.id for team in service.store.children(node_id))


async def test_a_team_over_existing_ones_replaces_them_and_keeps_their_sub_teams(
    service: DirectorService, launcher: FakeLauncher
):
    gone = {ic(service, API), manager(service, API)}
    kept = ic(service, DB)

    proposed = await use(service, ic(service), MsgManagerTool, SERVICES)
    assert "take the place of: API" in launcher.messages_to(manager(service))[-1]
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    assert not proposed.is_error
    assert below(service, "root") == ["apps.web", "services"]
    assert below(service, "services") == [DB]
    assert service.store.get(API) is None
    assert {str(conversation) for conversation in launcher.deleted} == gone
    assert ic(service, DB) == kept
    assert node(service, DB).boundary.path == "services/api/db"
    told = launcher.messages_to(ic(service))[-1]
    assert "Services (services)" in told and "The teams they replaced are gone" in told
    # The new team's IC continues from the IC that proposed it.
    starts = {str(start.conversation_id): start for start in launcher.derived}
    assert str(starts[str(ic(service, "services"))].parent_id) == ic(service)


async def test_teams_inside_an_existing_one_replace_it_and_the_rest_goes_back(
    service: DirectorService,
):
    await propose_and_approve(
        service,
        "root",
        SplitChild(title="HTTP", path="services/api/http", task="Routes"),
        SplitChild(title="Jobs", path="services/api/jobs", task="Workers"),
    )

    # Nothing new holds services/api/db, so its team reports to the proposer.
    assert below(service, "root") == [
        "apps.web",
        DB,
        "services.api.http",
        "services.api.jobs",
    ]
    assert service.store.get(API) is None
    assert node(service, DB).parent_id == "root"


async def test_a_new_team_inside_a_sub_team_that_stays_is_refused(
    service: DirectorService,
):
    refused = await use(
        service,
        ic(service),
        MsgManagerTool,
        proposing(
            [SplitChild(title="Tables", path="services/api/db/tables", task="x")]
        ),
    )

    assert refused.is_error and "would sit inside 'DB'" in refused.text
    assert node(service, "root").pending_split is None


async def test_what_a_replaced_team_committed_in_its_own_checkout_is_brought_in(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, "apps/web/page.tsx", "export {}\n")
    (worktree / "apps" / "web" / "draft.tsx").write_text("half\n")

    await propose_and_approve(
        service, "root", SplitChild(title="Apps", path="apps", task="All the apps")
    )

    assert below(service, "root") == ["apps", API]
    assert (repo / "apps" / "web" / "page.tsx").read_text() == "export {}\n"
    [saved] = run_git(
        repo, "for-each-ref", "--format=%(refname)", "refs/agents/pruned"
    ).split()
    assert run_git(repo, "show", f"{saved}:apps/web/draft.tsx") == "half\n"


async def test_work_that_does_not_merge_leaves_every_team_as_it_was(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    worktree = in_own_worktree(service, tmp_path, repo)
    commit(worktree, "apps/web/page.tsx", "theirs\n")
    commit(repo, "apps/web/page.tsx", "ours\n")

    await propose_and_approve(
        service, "root", SplitChild(title="Apps", path="apps", task="All the apps")
    )

    assert below(service, "root") == ["apps.web", API]
    assert node(service, "root").pending_split is None
    assert launcher.deleted == []
    told = launcher.messages_to(ic(service))[-1]
    assert "could not be set up" in told and "apps/web/page.tsx" in told
    assert "Nothing was changed" in told


async def test_teams_are_replaced_only_once_their_agents_are_at_rest(
    service: DirectorService, launcher: FakeLauncher
):
    api_ic, db_ic = ic(service, API), ic(service, DB)
    assert api_ic is not None and db_ic is not None
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)
    launcher.report(db_ic, ConversationExecutionStatus.RUNNING)
    await use(service, ic(service), MsgManagerTool, SERVICES)

    answer = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await use(
        service, manager(service), MsgIcTool, MsgIcAction(text="Hold on.", team=API)
    )

    assert "once their agents have finished" in answer.text
    assert below(service, "root") == ["apps.web", API]
    # Nothing new is started in a team that is about to change.
    assert not any("Hold on." in text for text in launcher.messages_to(api_ic))

    launcher.report(api_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()
    assert below(service, "root") == ["apps.web", API]

    launcher.report(db_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()

    assert below(service, "root") == ["apps.web", "services"]
    assert below(service, "services") == [DB]


async def test_a_team_is_pruned_only_once_it_and_the_teams_below_are_at_rest(
    service: DirectorService, launcher: FakeLauncher
):
    db_ic = ic(service, DB)
    assert db_ic is not None
    launcher.report(db_ic, ConversationExecutionStatus.RUNNING)

    await use(service, manager(service), PruneTeamTool, PRUNE_API)

    assert service.store.get(API) is not None and launcher.stopped == []

    launcher.report(db_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()

    assert service.store.get(API) is None and service.store.get(DB) is None
    assert "services/api is yours again" in launcher.messages_to(ic(service))[-1]


async def test_work_a_kept_sub_team_had_in_review_goes_to_the_ic_it_now_reports_to(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await split_root(service)
    await propose_and_approve(
        service, API, SplitChild(title="DB", path="services/api/db", task="Schema")
    )
    commit(repo, "services/api/db/schema.sql", "create table orders (id int);\n")
    await use(service, ic(service, DB), SubmitReviewTool, DONE)
    assert node(service, DB).review is not None

    await use(service, ic(service), MsgManagerTool, SERVICES)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    request = launcher.messages_to(ic(service, "services"))[-1]
    assert "Review mode" in request and DB in request
    await service.aclose()


async def test_a_replacement_cut_off_by_a_crash_is_finished_once(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await propose_and_approve(
        first, API, SplitChild(title="DB", path="services/api/db", task="Schema")
    )
    api_ic, db_ic = ic(first, API), ic(first, DB)
    assert api_ic is not None
    await use(first, ic(first), MsgManagerTool, SERVICES)
    # The approval is saved while the replaced team is still in a turn.
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)
    await use(first, manager(first), ApproveTeamsTool, APPROVE)
    # The server ends as it deletes the second agent of the replaced team.
    ended = end_server_at(launcher, "delete", nth=2)
    launcher.report(api_ic, ConversationExecutionStatus.FINISHED)
    await ended.wait()
    assert first.store.get(API) is not None

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert below(second, "root") == ["apps.web", "services"]
    assert below(second, "services") == [DB]
    assert ic(second, DB) == db_ic
    assert node(second, "root").pending_split is None
    started_as = [start.title for start in launcher.derived]
    assert started_as.count("Services (IC)") == 1
    assert UUID(str(db_ic)) not in launcher.deleted
    await second.aclose()


async def test_an_approval_the_saved_tree_does_not_have_is_dropped_and_given_again(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await use(first, ic(first), MsgManagerTool, SERVICES)
    proposed = run_git(repo, "rev-parse", "refs/agents/org").strip()
    ended = end_server_at(launcher, "stop")
    await cut_off(
        launcher, ended, use(first, manager(first), ApproveTeamsTool, APPROVE)
    )
    # Nothing of the first server may save the tree after this.
    await first.aclose()
    # The server ended before it saved the tree with the approval in it.
    run_git(repo, "update-ref", "refs/agents/org", proposed)

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert below(second, "root") == ["apps.web", API]
    waiting = node(second, "root").pending_split
    assert waiting is not None and waiting.status == "proposed"

    await use(second, manager(second), ApproveTeamsTool, APPROVE)

    assert below(second, "root") == ["apps.web", "services"]
    await second.aclose()
