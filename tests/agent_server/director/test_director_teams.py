"""Sub-teams that take the place of ones a team already has."""

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.models import (
    RouteMessageRequest,
    SplitChild,
    Stopped,
)
from openhands.agent_server.director.org_store import team_id_of
from openhands.agent_server.director.routing import USAGE_ID
from openhands.agent_server.director.run_state import SESSION, RunState
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import failure_waits
from openhands.agent_server.director.session_state import kept_sessions
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    MsgIcTool,
    MsgTeamTool,
    ProposeTeamsTool,
    PruneTeamTool,
    StopTeamTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import (
    MsgIcAction,
    MsgTeamAction,
    StopTeamAction,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    API,
    APPROVE,
    DB,
    DONE,
    PRUNE_API,
    SPLIT,
    FakeLauncher,
    ServerEnded,
    again,
    attempt,
    commit,
    create,
    cut_off,
    end_server_at,
    has_place,
    ic,
    in_own_worktree,
    landed,
    manager,
    node,
    propose_and_approve,
    proposing,
    record_of,
    run_git,
    split_root,
    started,
    sub_teams,
    tasked,
    use,
    woken,
)


SERVICES = proposing(
    [SplitChild(title="Services", path="services", task="Serve")], keep=["Web"]
)


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


def below(service: DirectorService, node_id: str) -> list[str]:
    return sorted(team_id_of(team) for team in sub_teams(service, node_id))


async def test_a_team_over_existing_ones_replaces_them_and_keeps_their_sub_teams(
    service: DirectorService, launcher: FakeLauncher
):
    gone = {ic(service, API), manager(service, API)}
    kept = ic(service, DB)

    proposed = await use(service, ic(service), ProposeTeamsTool, SERVICES)
    told = launcher.messages_to(manager(service))[-1]
    assert "take the place of: API" in told and "Kept as they are: Web." in told
    proposal = node(service, "root").pending_split
    assert proposal is not None
    assert (proposal.replaces, proposal.keeps) == (
        [node(service, API).id],
        [node(service, "apps.web").id],
    )
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    assert not proposed.is_error
    assert below(service, "root") == ["apps.web", "services"]
    assert below(service, "services") == [DB]
    assert not has_place(service, API)
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
        keep=["Web"],
    )

    # Nothing new holds services/api/db, so its team reports to the proposer.
    assert below(service, "root") == [
        "apps.web",
        DB,
        "services.api.http",
        "services.api.jobs",
    ]
    assert not has_place(service, API)
    assert node(service, DB).parent_id == node(service, "root").id


async def test_a_new_team_inside_a_sub_team_that_stays_is_refused(
    service: DirectorService,
):
    refused = await use(
        service,
        ic(service),
        ProposeTeamsTool,
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
        service,
        "root",
        SplitChild(title="Apps", path="apps", task="All the apps"),
        keep=["API"],
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
        service,
        "root",
        SplitChild(title="Apps", path="apps", task="All the apps"),
        keep=["API"],
    )

    assert below(service, "root") == ["apps.web", API]
    assert node(service, "root").pending_split is None
    assert launcher.deleted == []
    told = launcher.messages_to(ic(service))[-1]
    assert "could not be set up" in told and "apps/web/page.tsx" in told
    assert "Nothing was changed" in told
    # The Manager that approved them finds it in the record.
    [kept] = [event for event in record_of(service, "root") if event.kind == "declined"]
    assert kept.detail.startswith("approved, and the runtime could not set them up")
    assert kept.paths == ["apps"]


async def test_teams_are_replaced_only_once_their_agents_are_at_rest(
    service: DirectorService, launcher: FakeLauncher
):
    api_ic, db_ic = ic(service, API), ic(service, DB)
    assert api_ic is not None and db_ic is not None
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)
    launcher.report(db_ic, ConversationExecutionStatus.RUNNING)
    await use(service, ic(service), ProposeTeamsTool, SERVICES)

    answer = await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)
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

    assert has_place(service, API) and launcher.stopped == []

    launcher.report(db_ic, ConversationExecutionStatus.FINISHED)
    await service.settled()

    assert not has_place(service, API) and not has_place(service, DB)
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

    await use(service, ic(service), ProposeTeamsTool, SERVICES)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    # The request is the first thing sent to the new team, so it starts it.
    [request] = [
        s.task for s in launcher.derived if s.title == "Services · Checkout (IC)"
    ]
    assert "Review request" in request and node(service, DB).id in request
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
    await use(first, ic(first), ProposeTeamsTool, SERVICES)
    # The approval is saved while the replaced team is still in a turn.
    launcher.report(api_ic, ConversationExecutionStatus.RUNNING)
    await use(first, manager(first), ApproveTeamsTool, APPROVE)
    await tasked(first)
    # The server ends as it deletes the second agent of the replaced team.
    ended = end_server_at(launcher, "delete", nth=2)
    launcher.report(api_ic, ConversationExecutionStatus.FINISHED)
    await ended.wait()
    assert has_place(first, API)

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()
    await tasked(second)

    assert below(second, "root") == ["apps.web", "services"]
    assert below(second, "services") == [DB]
    assert ic(second, DB) == db_ic
    assert node(second, "root").pending_split is None
    started_as = [start.title for start in launcher.derived]
    assert started_as.count("Services · Checkout (IC)") == 1
    assert UUID(str(db_ic)) not in launcher.deleted
    await second.aclose()


async def test_an_approval_the_kept_session_does_not_have_is_dropped_and_given_again(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await use(first, ic(first), ProposeTeamsTool, SERVICES)
    proposed = kept_sessions(first.org)
    ended = end_server_at(launcher, "stop")
    await cut_off(
        launcher, ended, use(first, manager(first), ApproveTeamsTool, APPROVE)
    )
    # Nothing of the first server may save the session after this.
    await first.aclose()
    # The server ended before it saved the session with the approval in it.
    run = RunState(repo / ".git" / "director" / "state.db")
    run.replace_marks(SESSION, proposed)
    run.close()

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert below(second, "root") == ["apps.web", API]
    waiting = node(second, "root").pending_split
    assert waiting is not None and waiting.status == "proposed"

    await use(second, manager(second), ApproveTeamsTool, APPROVE)
    await tasked(second)

    assert below(second, "root") == ["apps.web", "services"]
    await second.aclose()


@pytest.mark.parametrize(
    ("keep", "problem"),
    [
        (None, "'Web' is neither replaced by a proposed team nor listed under"),
        (["Web", "API"], "'API' cannot be kept and replaced"),
        (["Web", "Mobile"], "`keep` names 'Mobile', which is not among"),
    ],
)
async def test_a_proposal_must_say_what_becomes_of_every_sub_team(
    service: DirectorService, keep: list[str] | None, problem: str
):
    services = [SplitChild(title="Services", path="services", task="Serve")]

    refused = await use(
        service, ic(service), ProposeTeamsTool, proposing(services, keep=keep)
    )

    assert refused.is_error and problem in refused.text
    assert node(service, "root").pending_split is None


async def test_a_sub_team_is_kept_by_its_name_or_its_team_id(
    service: DirectorService,
):
    mobile = [SplitChild(title="Mobile", path="apps/mobile", task="The app")]

    refused = await use(
        service, ic(service), ProposeTeamsTool, proposing(mobile, keep=["API"])
    )
    taken = await use(
        service,
        ic(service),
        ProposeTeamsTool,
        proposing(mobile, keep=[node(service, API).id, "Web"]),
    )

    assert refused.is_error and "'Web' is neither replaced" in refused.text
    proposal = node(service, "root").pending_split
    assert not taken.is_error and proposal is not None
    assert (proposal.replaces, proposal.keeps) == (
        [],
        [node(service, API).id, node(service, "apps.web").id],
    )


@pytest.fixture
async def fresh(repo: Path, launcher: FakeLauncher):
    """An org of one team."""
    service = await started(repo, launcher)
    yield service
    await service.aclose()


async def test_approved_teams_are_set_up_and_start_nothing(
    fresh: DirectorService, launcher: FakeLauncher
):
    service = fresh
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    answer = await use(service, manager(service), ApproveTeamsTool, APPROVE)

    teams = sub_teams(service)
    assert [(team_id_of(team), team.agents, team.status) for team in teams] == [
        (API, [], "idle"),
        ("apps.web", [], "idle"),
    ]
    assert node(service, "root").pending_split is None
    assert len(launcher.derived) == 1
    assert "None of them starts until your IC gives it its task" in answer.text
    [told] = [
        text
        for text in launcher.messages_to(ic(service))
        if "your Manager approved" in text
    ]
    assert "a team starts when you give it its task with msg_team" in told
    # What only an IC with sub-teams must know comes with them.
    assert "list under `keep`" in told
    assert "goes on up without another review" in told


async def test_a_split_says_what_the_ic_left_and_lets_it_finish_until_the_team_starts(
    fresh: DirectorService, launcher: FakeLauncher, repo: Path
):
    service = fresh
    pay = repo / "services" / "api" / "pay.py"
    pay.parent.mkdir(parents=True)
    pay.write_text("draft\n")
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    [told] = [
        text
        for text in launcher.messages_to(ic(service))
        if "your Manager approved" in text
    ]
    assert "How the split changes your own work:" in told
    assert "  - API (services/api): services/api/pay.py" in told
    assert "You may finish them" in told and "Web (apps/web):" not in told
    # The team has not started, so the IC may finish what it has there.
    assert attempt(service, ic(service), "file_editor", create(pay)) is None

    given = await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Take payments")
    )
    await service.settled()

    assert "The 1 file(s) you left uncommitted in its folder" in given.text
    assert "belongs to your sub-team 'API'" in (
        attempt(service, ic(service), "file_editor", create(pay)) or ""
    )
    assert "did not commit them: services/api/pay.py. They are yours now." in (
        launcher.derived[-1].task
    )


async def test_a_split_that_leaves_nothing_unfinished_says_so(
    fresh: DirectorService, launcher: FakeLauncher
):
    service = fresh
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    [told] = [
        text
        for text in launcher.messages_to(ic(service))
        if "your Manager approved" in text
    ]
    assert "You left nothing uncommitted there." in told
    assert "You have changes there" not in told


async def test_what_is_uncommitted_in_a_sub_teams_directory_is_not_the_ics_to_submit(
    service: DirectorService, repo: Path
):
    schema = repo / "services" / "api" / "db" / "schema.sql"
    schema.parent.mkdir(parents=True)
    schema.write_text("create table orders (id int);\n")

    refused = await use(service, ic(service, API), SubmitReviewTool, DONE)

    assert refused.is_error
    assert "services/api/db/schema.sql in the folder of your sub-team 'DB'" in (
        refused.text
    )
    assert "Wait until it has submitted its work" in refused.text


async def test_a_team_starts_with_the_first_message_from_the_ic_above(
    fresh: DirectorService, launcher: FakeLauncher
):
    service = fresh
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)

    first = await use(
        service,
        ic(service),
        MsgTeamTool,
        MsgTeamAction(team=API, text="Build POST /checkout first."),
    )
    second = await use(
        service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="And GET.")
    )

    assert "starts with what you wrote as its task" in first.text
    assert "starts with" not in second.text
    [start] = launcher.derived[1:]
    assert start.title == "API · Checkout (IC)"
    # What the team is for, then what it was started with.
    assert start.task.index("Serve the checkout API") < start.task.index(
        "Build POST /checkout first."
    )
    assert launcher.messages_to(ic(service, API))[-1].endswith("And GET.")
    assert node(service, "apps.web").agents == []


async def test_a_message_from_the_human_starts_the_team_in_a_session_of_its_own(
    fresh: DirectorService, launcher: FakeLauncher
):
    service = fresh
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await landed(service)
    before = len(launcher.derived)

    routed = await service.route_message(
        RouteMessageRequest(text="Start with the cart page.", path="apps/web")
    )
    await service.settled()

    [start] = launcher.derived[before:]
    assert start.title.startswith("Web · ") and start.title.endswith("(IC)")
    assert "Start with the cart page." in start.task
    # The team the first session set up and never started stays as it was.
    assert routed.node_id != node(service, "apps.web").id
    assert node(service, "apps.web").agents == []


async def test_a_team_given_no_task_owes_nothing(
    repo: Path, fresh: DirectorService, launcher: FakeLauncher
):
    service = fresh
    await use(service, ic(service), ProposeTeamsTool, SPLIT)
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await use(service, ic(service), MsgTeamTool, MsgTeamAction(team=API, text="Begin."))
    commit(repo, "services/api/app.py", "order_id = '1'\n")

    await use(service, ic(service, API), SubmitReviewTool, DONE)

    # The review of the team that works does not wait for the one that was
    # never started.
    assert any("Review request" in text for text in launcher.messages_to(ic(service)))
    assert node(service, "apps.web").agents == []


async def test_work_that_does_not_merge_takes_out_the_teams_merged_before_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    web = in_own_worktree(service, tmp_path, repo)
    api = in_own_worktree(service, tmp_path, repo, API)
    commit(web, "apps/web/page.tsx", "merges\n")
    commit(api, "services/api/main.py", "theirs\n")
    commit(repo, "services/api/main.py", "ours\n")
    before = run_git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text("uncommitted\n")

    await propose_and_approve(
        service,
        "root",
        SplitChild(title="Apps", path="apps", task="All the apps"),
        SplitChild(title="Services", path="services", task="Serve"),
    )

    assert "Nothing was changed" in launcher.messages_to(ic(service))[-1]
    assert run_git(repo, "rev-parse", "HEAD") == before
    assert (repo / "notes.txt").read_text() == "uncommitted\n"


async def test_a_team_made_in_place_of_one_is_not_taken_for_it(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    await split_root(first)
    api = node(first, API).id
    await use(
        first,
        ic(first),
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="Orders", path="services/api", task="Take orders")],
            keep=["Web"],
        ),
    )
    ended = asyncio.Event()
    save = first._save

    async def server_ends() -> None:
        made = first.org.place(api)
        if made is None or made.title != "Orders":
            return await save()
        # The session is kept with the new team in it, and the split is
        # not on record as done.
        first._runs().replace_marks(SESSION, kept_sessions(first.org))
        ended.set()
        raise ServerEnded

    monkeypatch.setattr(first, "_save", server_ends)
    await cut_off(
        launcher, ended, use(first, manager(first), ApproveTeamsTool, APPROVE)
    )

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert node(second, API).title == "Orders"
    gone = [event for event in record_of(second, "root") if event.kind == "replaced"]
    assert len(gone) == 1
    await second.aclose()


async def test_nothing_is_sent_straight_to_a_team_under_a_stopped_one(
    service: DirectorService, launcher: FakeLauncher
):
    await woken(service, DB)
    launcher.answers[USAGE_ID] = "Teams: services.api.db\nReason: The schema is DB's."
    [session] = service.org.sessions()
    service.org.update_place(
        node(service, API).id,
        stopped=Stopped(at="2026-01-01T00:00:00+00:00", reason="Waits", running=[]),
    )

    said = await service.route_message(
        RouteMessageRequest(text="Add an index on orders.", session_id=session.id)
    )

    # DB is stopped with API: it would only be held there.
    assert said.node_id == node(service, "root").id


async def test_sub_teams_that_fail_to_be_set_up_are_put_aside_and_told(
    service: DirectorService, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    def fails(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(service, "_record_contracts", fails)
    await use(
        service,
        ic(service, "apps.web"),
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="Cart", path="apps/web/cart", task="The cart")],
            contracts="Cart owns the cart page.",
        ),
    )
    await use(service, manager(service, "apps.web"), ApproveTeamsTool, APPROVE)
    await service.settled()

    # Another proposal is not refused for one that is "still being set up".
    assert node(service, "apps.web").pending_split is None
    assert "disk full" in launcher.messages_to(ic(service, "apps.web"))[-1]


async def test_a_manager_that_could_not_be_started_is_started_again_by_the_answer(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    top = node(service, "root")
    launcher.derive_error = RuntimeError("npx is not installed")
    service.tell_overseer(top.id, "Look in on your team.")
    await service.settled()
    [asked] = node(service, "root").escalations
    assert asked.id == failure_waits("manager")
    derived = len(launcher.derived)

    service.answer_escalation(top.id, asked.id, "Fixed, go on.")
    await service.settled()

    top = node(service, "root")
    [manager] = [agent for agent in top.agents if agent.role == "manager"]
    assert manager.status != "lost" and top.escalations == []
    assert len(launcher.derived) == derived + 1
    told = launcher.messages_to(manager.conversation_id)
    assert any("Fixed, go on." in text for text in told)
    await service.aclose()


async def test_what_was_held_for_a_pruned_team_does_not_pause_the_org_on_a_restart(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await woken(first, "root")
    await use(first, ic(first), StopTeamTool, StopTeamAction(team=API, reason="Stuck."))
    first.notify(ic(first, API), "A notice for the stopped team.")
    await first.settled()
    assert first._runs().held()
    await use(first, manager(first), PruneTeamTool, PRUNE_API)
    await first.settled()
    assert first._runs().held() == []
    await first.aclose()

    second = await again(repo, launcher)
    assert second.summary().paused is None
    await second.aclose()
