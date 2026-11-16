"""Tests for sessions: lines of work the human opens at a team beside the
org's own, each with agents and a checkout of its own."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.git import git
from openhands.agent_server.director.merge_back import merge_back
from openhands.agent_server.director.models import (
    OpenedSession,
    OpenSessionRequest,
    RouteMessageRequest,
    SplitChild,
)
from openhands.agent_server.director.org_store import AGENT_LOST
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import (
    MsgManagerTool,
    MsgTeamTool,
    PruneTeamTool,
    ReadNodeContextTool,
    ReportTool,
    ReviewTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgTeamAction,
    PruneTeamAction,
    ReadNodeContextAction,
    ReportAction,
)
from openhands.sdk.errors import Conflict, NotFound
from tests.agent_server.director.conftest import FakeLauncher, run_git
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_review import DONE, accept, commit
from tests.agent_server.director.test_director_teams import (
    API,
    DB,
    propose_and_approve,
)
from tests.agent_server.director.test_director_tools import (
    ic,
    node,
    proposing,
    split_root,
    use,
    woken,
)


WEB = "apps.web"
REFUNDS = "Add refunds to the checkout API."


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    """An org of API (with DB below it) and Web, every team started."""
    service = await started(repo, launcher, stall_poll=3600)
    await split_root(service)
    await propose_and_approve(
        service,
        API,
        SplitChild(title="DB", path="services/api/db", task="Own the schema"),
    )
    yield service
    await service.aclose()


async def opened(
    service: DirectorService,
    at: str | None = API,
    text: str = REFUNDS,
    title: str = "Refunds",
) -> OpenedSession:
    session = await service.open_session(
        OpenSessionRequest(text=text, title=title, node_id=at)
    )
    await service.settled()
    return session


def own_checkout(
    service: DirectorService, repo: Path, tmp_path: Path, node_id: str
) -> Path:
    """Give a session's team the worktree its conversation would have made."""
    name = node_id.replace("@", "-")
    worktree = tmp_path / name
    run_git(repo, "worktree", "add", "--quiet", "-b", name, str(worktree))
    for team in service.store.session(node(service, node_id).session):
        service.store.update_node(team.id, worktree_path=str(worktree), branch=name)
    return worktree


async def test_a_session_puts_the_team_and_its_sub_teams_there_once_more(
    service: DirectorService, launcher: FakeLauncher
):
    teams_before = service.store.teams_below("root")

    session = await opened(service)

    top = node(service, session.session.node_id)
    assert top.id == f"{API}@{session.session.id}"
    # Under the team whose IC reviews it, beside the team it is there for.
    assert (top.parent_id, top.instance_of, top.title) == ("root", API, "API")
    assert top.boundary == node(service, API).boundary
    assert (top.session_title, top.isolation) == ("Refunds", "worktree")
    [db] = service.store.children(top.id)
    assert (db.instance_of, db.session, db.agents) == (DB, top.session, [])
    # The session's IC starts from the reviewer's, in a checkout of its own.
    start = launcher.derived[-1]
    assert str(start.conversation_id) == ic(service, top.id)
    assert (str(start.parent_id), start.isolation) == (ic(service), "worktree")
    assert REFUNDS in start.task and "session 'Refunds'" in start.task
    assert session.conversation_id == ic(service, top.id)
    # A session adds no team to the org.
    assert service.store.teams_below("root") == teams_before
    assert [team.id for team in service.store.children("root")] == [API, WEB]
    assert [team.id for team in service.store.opened_under("root")] == [top.id]
    assert node(service, "root").session_ids == [top.id]
    assert node(service, top.id).session_ids == []
    assert [(found.id, found.team_id, found.teams) for found in service.sessions()] == [
        (session.session.id, API, 1)
    ]
    [event] = [
        event
        for event in service.team_memory.history("root")
        if event.kind == "session_opened"
    ]
    assert (event.team_id, event.session, event.detail) == (top.id, "Refunds", REFUNDS)


async def test_a_session_at_the_top_team_is_reviewed_by_its_own_ic(
    service: DirectorService,
):
    session = await opened(service, "root", "Rename the product everywhere.")

    top = node(service, session.session.node_id)
    assert (top.parent_id, top.instance_of) == ("root", "root")
    # The whole org is there for it, and none of it has started.
    assert {team.instance_of for team in service.store.session(top.session)} == {
        "root",
        API,
        DB,
        WEB,
    }
    assert session.session.teams == 1
    # The top team's own IC keeps its sub-teams and gains none.
    assert [team.id for team in service.store.children("root")] == [API, WEB]


async def test_the_director_chooses_the_team_when_none_is_named(
    service: DirectorService, launcher: FakeLauncher
):
    launcher.answers[USAGE_ID] = f"Teams: {WEB}\nReason: It is the cart page."

    session = await opened(service, None, "Show the order id on the cart page.")

    assert session.session.team_id == WEB
    assert (session.chosen_by, session.reason) == ("model", "It is the cart page.")


async def test_a_session_is_refused_where_it_cannot_be_reviewed_or_found(
    service: DirectorService,
):
    with pytest.raises(NotFound):
        await opened(service, "no.such.team")

    service.store.set_agent("root", "ic", status=AGENT_LOST)
    with pytest.raises(Conflict, match="reviews the work"):
        await opened(service)
    assert service.sessions() == []


async def test_a_sub_team_joins_a_session_when_its_ic_gives_it_a_task(
    service: DirectorService, launcher: FakeLauncher
):
    session = await opened(service)
    top = session.session.node_id
    db = f"{DB}@{session.session.id}"
    to_db = len(launcher.messages_to(ic(service, DB)))

    await use(
        service,
        ic(service, top),
        MsgTeamTool,
        MsgTeamAction(team=db, text="Add a refunds table."),
    )

    start = launcher.derived[-1]
    assert str(start.conversation_id) == ic(service, db)
    assert str(start.parent_id) == ic(service, top)
    assert "Add a refunds table." in start.task and "Own the schema" in start.task
    # The team's own DB hears nothing of it.
    assert len(launcher.messages_to(ic(service, DB))) == to_db


async def test_each_session_is_reviewed_by_the_ic_above_and_the_second_meets_the_first(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    first = (await opened(service, title="Refunds")).session.node_id
    second = (await opened(service, title="Vouchers")).session.node_id
    for team, content in ((first, "refunds\n"), (second, "vouchers\n")):
        worktree = own_checkout(service, repo, tmp_path, team)
        commit(worktree, "services/api/routes.ts", content)
        before = len(launcher.messages_to(ic(service)))
        await use(service, ic(service, team), SubmitReviewTool, DONE)
        # Web has not submitted, and the session's review does not wait for it.
        [request] = launcher.messages_to(ic(service))[before:]
        assert "The human opened the session" in request and team in request

    await use(service, ic(service), ReviewTeamTool, accept(first))
    await use(service, ic(service), ReviewTeamTool, accept(second))

    assert (repo / "services/api/routes.ts").read_text() == "refunds\n"
    reviews = [node(service, team).review for team in (first, second)]
    assert [review.state for review in reviews if review] == [
        "accepted",
        "changes_requested",
    ]
    assert "do not merge" in launcher.messages_to(ic(service, second))[-1]


async def test_work_uncommitted_beside_a_session_is_not_put_aside_for_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    """The team's own IC works in the checkout the session is merged into.
    What it has not committed there is its work, not something in the way."""
    top = (await opened(service)).session.node_id
    worktree = own_checkout(service, repo, tmp_path, top)
    commit(worktree, "services/api/routes.ts", "refunds\n")
    (repo / "services/api").mkdir(parents=True, exist_ok=True)
    (repo / "services/api/routes.ts").write_text("the team's own, not committed\n")
    await use(service, ic(service, top), SubmitReviewTool, DONE)

    await use(service, ic(service), ReviewTeamTool, accept(top))

    assert (repo / "services/api/routes.ts").read_text() == (
        "the team's own, not committed\n"
    )
    review = node(service, top).review
    assert review is not None and review.state == "pending"
    assert "not committed" in launcher.messages_to(ic(service))[-1]


async def test_a_sessions_work_never_goes_up_without_a_review(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    """A team that changed nothing itself passes its sub-teams' accepted
    work up unreviewed. The IC above a session gave it no task, so it
    reviews what the session delivers all the same."""
    session = await opened(service)
    top, db = session.session.node_id, f"{DB}@{session.session.id}"
    await use(service, ic(service, top), MsgTeamTool, MsgTeamAction(team=db, text="Go"))
    commit(repo, "services/api/db/schema.sql", "create table refunds;\n")
    await use(service, ic(service, db), SubmitReviewTool, DONE)

    await use(service, ic(service, top), ReviewTeamTool, accept(db))

    accepted = node(service, db).review
    assert accepted is not None and accepted.state == "accepted"
    assert node(service, top).review is None


async def test_sub_teams_are_not_changed_where_a_session_works(
    service: DirectorService,
):
    session = await opened(service)
    top = session.session.node_id
    more = proposing(
        [SplitChild(title="Routes", path="services/api/routes", task="Route")],
        keep=["DB"],
    )

    inside = await use(service, ic(service, top), MsgManagerTool, more)
    beside = await use(service, ic(service, API), MsgManagerTool, more)
    pruned = await use(
        service,
        await woken(service),
        PruneTeamTool,
        PruneTeamAction(node_id=API, reason="Too slow"),
    )
    closed = await use(
        service,
        await woken(service),
        PruneTeamTool,
        PruneTeamAction(node_id=top, reason="Not needed"),
    )

    assert "not changed inside a session" in inside.text
    assert "while the session 'Refunds' works in it" in beside.text
    assert "sessions (Refunds) work in it" in pruned.text
    assert "Only the human closes a session" in closed.text
    assert node(service, API).pending_split is None

    service.close_session(session.session.id)
    await service.settled()
    again = await use(service, ic(service, API), MsgManagerTool, more)

    assert "Sent to your Manager with the sub-teams" in again.text


async def test_the_sessions_of_a_team_share_its_reports_and_know_of_each_other(
    service: DirectorService,
):
    top = (await opened(service)).session.node_id
    await use(
        service, ic(service, API), ReportTool, ReportAction(text="Routes are typed.")
    )
    await use(
        service, ic(service, top), ReportTool, ReportAction(text="Refunds are async.")
    )

    own = service.context_of(node(service, API), "ic")
    session = service.context_of(node(service, top), "ic")
    above = service.context_of(node(service, "root"), "ic")

    assert [
        (report["by"], report["text"]) for report in own["reports_of_your_team"]
    ] == [
        ("ic", "Routes are typed."),
        ("ic in 'Refunds'", "Refunds are async."),
    ]
    assert [
        (report["by"], report["text"]) for report in session["reports_of_your_team"]
    ] == [
        ('ic in "the team\'s own work"', "Routes are typed."),
        ("ic", "Refunds are async."),
    ]
    assert session["you"]["session"] == "Refunds"
    assert [other["node_id"] for other in session["other_sessions_of_your_team"]] == [
        API
    ]
    assert [other["session"] for other in own["other_sessions_of_your_team"]] == [
        "Refunds"
    ]
    assert [found["node_id"] for found in above["sessions_whose_work_you_review"]] == [
        top
    ]
    assert [team["node_id"] for team in above["sub_teams"]] == [API, WEB]


async def test_an_agent_reads_the_other_sessions_with_its_context(
    service: DirectorService,
):
    await opened(service)

    observation = await use(
        service, ic(service, API), ReadNodeContextTool, ReadNodeContextAction()
    )

    assert "other_sessions_of_your_team" in observation.text


async def test_a_message_reaches_a_session_only_when_it_is_written_to_it(
    service: DirectorService, launcher: FakeLauncher
):
    top = (await opened(service)).session.node_id
    launcher.answers[USAGE_ID] = f"Teams: {API}\nReason: It is the API's."

    routed = await service.route_message(
        RouteMessageRequest(text="Make services/api faster.", path="", auto=True)
    )
    written = await service.route_message(
        RouteMessageRequest(text="Also refund partial orders.", node_id=top)
    )
    await service.settled()

    assert routed.node_id == API
    assert (written.node_id, written.conversation_id) == (top, ic(service, top))
    assert "Also refund partial orders." in launcher.messages_to(ic(service, top))[-1]


async def test_closing_a_session_removes_it_and_keeps_what_was_not_merged(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    session = await opened(service)
    top = session.session.node_id
    conversation = ic(service, top)
    worktree = own_checkout(service, repo, tmp_path, top)
    commit(worktree, "services/api/routes.ts", "refunds\n")
    tip = run_git(worktree, "rev-parse", "HEAD").strip()
    (repo / "services/api").mkdir(parents=True, exist_ok=True)
    (repo / "services/api/own.ts").write_text("the team's own, not committed\n")

    service.close_session(session.session.id)
    await service.settled()

    assert service.sessions() == []
    assert service.store.session(session.session.id) == []
    assert conversation is not None and UUID(conversation) in launcher.deleted
    kept = run_git(
        repo, "for-each-ref", "--format=%(objectname)", "refs/agents/sessions/"
    ).split()
    assert kept == [tip]
    assert (repo / "services/api/own.ts").exists()
    assert "closed the session 'Refunds'" in launcher.messages_to(ic(service))[-1]
    [event] = [
        event
        for event in service.team_memory.history("root")
        if event.kind == "session_closed"
    ]
    assert event.detail == "with work that was not merged"
    with pytest.raises(NotFound):
        service.close_session(session.session.id)


async def test_closing_a_session_that_never_started_touches_no_checkout(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    """Until its IC has started, a session's team names the checkout of
    the team above, where the work is somebody else's."""
    session = await opened(service)
    top = session.session.node_id
    assert node(service, top).worktree_path == node(service, "root").worktree_path
    (repo / "services/api").mkdir(parents=True, exist_ok=True)
    (repo / "services/api/own.ts").write_text("the team's own, not committed\n")

    service.close_session(session.session.id)
    await service.settled()

    assert (repo / "services/api/own.ts").read_text() == (
        "the team's own, not committed\n"
    )
    assert run_git(repo, "for-each-ref", "refs/agents/pruned/") == ""


def test_a_merge_reaches_the_index_of_an_agent_that_owns_the_same_directory(
    repo: Path, tmp_path: Path
):
    """An agent's own index does not follow a merge into its checkout by
    itself. Left behind, its next commit would take the merged files out."""
    commit(repo, "api/routes.ts", "v1\n")
    commit(repo, "api/own.ts", "v1\n")
    worktree = tmp_path / "session"
    run_git(repo, "worktree", "add", "--quiet", "-b", "session", str(worktree))
    commit(worktree, "api/routes.ts", "from the session\n")
    commit(worktree, "api/new.ts", "from the session\n")
    git_dir = Path(run_git(repo, "rev-parse", "--absolute-git-dir").strip())
    own = {"GIT_INDEX_FILE": str(git_dir / "agent-index-abc")}
    git(repo, ["read-tree", "HEAD"], env=own)
    (repo / "api/own.ts").write_text("staged by the agent\n")
    git(repo, ["add", "api/own.ts"], env=own)

    result = merge_back(repo, worktree, "api", "API")
    git(repo, ["commit", "--quiet", "-m", "The agent's own"], env=own)

    assert result.outcome == "merged"
    committed = {
        path: run_git(repo, "show", f"HEAD:{path}")
        for path in ("api/routes.ts", "api/new.ts", "api/own.ts")
    }
    assert committed == {
        "api/routes.ts": "from the session\n",
        "api/new.ts": "from the session\n",
        "api/own.ts": "staged by the agent\n",
    }
