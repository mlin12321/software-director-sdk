"""Pausing an org, and taking up what a stopped server left unfinished."""

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.git_blackboard import CONTRACTS_FILE
from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    RouteMessageRequest,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.recovery import describe, restore_worktrees
from openhands.agent_server.director.run_state import RunState, RunStateError
from openhands.agent_server.director.service import (
    AGENT_PAUSED,
    RESTART_NOTE,
    DirectorService,
)
from openhands.agent_server.director.tools import (
    DecideSplitTool,
    ProposeBoundarySplitTool,
    PruneTeamTool,
    ReviewTeamTool,
    SetStatusTool,
)
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import FakeLauncher, make_node, run_git
from tests.agent_server.director.test_director_limits import (
    APPROVE,
    eventually,
    split,
    started,
)
from tests.agent_server.director.test_director_review import (
    DONE,
    accept,
    api_reports_done,
    commit,
    reject,
)
from tests.agent_server.director.test_director_tools import (
    PRUNE_API,
    SPLIT,
    ic,
    manager,
    node,
    split_root,
    use,
)


class ServerEnded(BaseException):
    """Ends whatever the director was doing, as a killed server does."""


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


async def again(repo: Path, launcher: FakeLauncher, **options) -> DirectorService:
    """The director of the server's next start, on the same org."""
    launcher.listeners.clear()
    launcher.before = lambda _: None
    launcher.on_derived = lambda _: None
    service = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        remembered=repo.parent / "state" / "director.json",
        **options,
    )
    await service.start(launcher)
    await service.settled()
    return service


def end_server_at(launcher: FakeLauncher, call: str, nth: int = 1) -> asyncio.Event:
    """Have the server end at the ``nth`` time the director makes ``call``.

    Returns:
        Set once it has ended.
    """
    ended = asyncio.Event()
    seen = 0

    def before(name: str) -> None:
        nonlocal seen
        seen += name == call
        if ended.is_set() or seen >= nth:
            ended.set()
            raise ServerEnded

    launcher.before = before
    launcher.on_derived = lambda _: before("derived")
    return ended


def kinds(recovery) -> list[tuple[str, str]]:
    return [(item.node_id, item.kind) for item in recovery.items]


async def test_a_paused_org_holds_messages_and_delivers_them_in_order_on_resume(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    before = len(launcher.sent)

    paused = await service.pause()
    await service.route_message(RouteMessageRequest(text="First"))
    service.notify(ic(service), "Second")
    await service.settled()

    assert paused.paused == "human" and service.summary().paused == "human"
    assert len(launcher.sent) == before
    held = await service.recovery()
    assert held.messages_held == 2 and ("root", "message") in kinds(held)

    resumed = await service.resume()
    await service.settled()

    assert resumed.paused is None and service.summary().paused is None
    first, second = launcher.messages_to(ic(service))[-2:]
    assert first == "First" and "Second" in second
    assert (await service.recovery()).items == []
    await service.aclose()


async def test_pausing_stops_the_agents_that_run_and_resuming_runs_only_those(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    root_ic = ic(service)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)

    await service.pause()
    await service.settled()

    assert launcher.paused == [UUID(root_ic)]
    statuses = {agent.role: agent.status for agent in node(service, "root").agents}
    assert statuses == {"ic": AGENT_PAUSED, "manager": "idle"}
    assert ("root", "run_again") in kinds(await service.recovery())

    await service.resume()
    await service.settled()

    assert launcher.resumed == [(UUID(root_ic), None)]
    assert node(service, "root").status == "working"
    await service.aclose()


async def test_what_an_agent_decides_while_the_org_is_paused_is_done_after_resume(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await use(service, ic(service), ProposeBoundarySplitTool, SPLIT)

    await service.pause()
    answer = await use(service, manager(service), DecideSplitTool, APPROVE)

    assert not answer.is_error and node(service, "root").child_ids == []
    assert ("root", "continue") in kinds(await service.recovery())

    await service.resume()
    await service.settled()

    assert node(service, "root").child_ids == ["services.api", "apps.web"]
    assert node(service, "root").pending_split is None
    assert len(launcher.derived) == 5
    await service.aclose()


async def test_a_server_stopped_while_teams_wait_starts_them_in_order_once_resumed(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher, run_limit_retry=3600)
    launcher.full_for = 1_000_000
    await use(first, ic(first), ProposeBoundarySplitTool, split("a", 3))
    approve = asyncio.create_task(use(first, manager(first), DecideSplitTool, APPROVE))
    await eventually(
        lambda: sum(team.waiting_to_start for team in first.store.children("root")) == 3
    )
    await first.aclose()
    approve.cancel()
    await asyncio.gather(approve, return_exceptions=True)

    launcher.full_for = 0
    second = await again(repo, launcher)
    left = await second.recovery()

    assert second.summary().paused == "shutdown" and len(launcher.derived) == 1
    waited = [node_id for node_id, kind in kinds(left) if kind == "start"]
    assert sorted(waited) == ["a.t0", "a.t1", "a.t2"]
    # A team that has not been started yet has lost nothing.
    assert {kind for _, kind in kinds(left)} == {"start"}
    assert {
        agent.status for team in second.store.children("root") for agent in team.agents
    } == {"idle"}

    await second.resume()
    await second.settled()

    teams = second.store.children("root")
    assert [team.id for team in teams] == ["a.t0", "a.t1", "a.t2"]
    assert node(second, "root").pending_split is None
    starts = [str(start.conversation_id) for start in launcher.derived[1:]]
    assert len(set(starts)) == len(starts) == 6
    in_order = [ic(second, node_id) for node_id in waited]
    assert [start for start in starts if start in in_order] == in_order
    await second.aclose()


async def test_a_split_cut_off_by_a_crash_is_finished_without_a_second_of_anything(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await use(first, ic(first), ProposeBoundarySplitTool, SPLIT)
    # The server ends as the first sub-team's IC has just been started.
    ended = end_server_at(launcher, "derived")
    approve = asyncio.create_task(use(first, manager(first), DecideSplitTool, APPROVE))
    await ended.wait()
    approve.cancel()
    await asyncio.gather(approve, return_exceptions=True)
    assert len(launcher.derived) < 5

    second = await again(repo, launcher)

    assert second.summary().paused == "crash"
    assert node(second, "root").pending_split is not None

    await second.resume()
    await second.settled()

    assert node(second, "root").child_ids == ["services.api", "apps.web"]
    assert node(second, "root").pending_split is None
    starts = [str(start.conversation_id) for start in launcher.derived]
    assert len(set(starts)) == len(starts) == 5
    contracts = second.blackboard.read(CONTRACTS_FILE) or ""
    assert contracts.count("POST /checkout returns {order_id}") == 1
    told = [text for text in launcher.messages_to(ic(second)) if "approved" in text]
    assert len(told) == 1
    await second.aclose()


async def test_a_prune_cut_off_by_a_crash_is_finished_and_recorded_once(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "half.py").write_text("broken\n")
    ended = end_server_at(launcher, "delete", nth=2)
    prune = asyncio.create_task(use(first, manager(first), PruneTeamTool, PRUNE_API))
    await ended.wait()
    prune.cancel()
    await asyncio.gather(prune, return_exceptions=True)
    assert len(launcher.deleted) == 1

    second = await again(repo, launcher)
    assert ("services.api", "continue") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    assert second.store.get("services.api") is None
    assert len(set(launcher.deleted)) == 2
    [autopsy] = second.blackboard.autopsies("services/api")
    kept = run_git(repo, "for-each-ref", "--format=%(refname)", "refs/agents/pruned")
    assert kept.split() == [autopsy.saved_ref]
    told = [text for text in launcher.messages_to(ic(second)) if "pruned" in text]
    assert len(told) == 1
    await second.aclose()


async def cut_off(launcher: FakeLauncher, ended: asyncio.Event, doing) -> None:
    """Let the server end in the middle of ``doing``."""
    task = asyncio.create_task(doing)
    await ended.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def end_server_at_message(launcher: FakeLauncher, part: str) -> asyncio.Event:
    """Have the server end as it is about to send a message with ``part``."""
    ended = asyncio.Event()

    def before(name: str) -> None:
        if ended.is_set() or (name == "send" and part in launcher.sending):
            ended.set()
            raise ServerEnded

    launcher.before = before
    return ended


async def test_the_end_of_a_split_cut_off_by_a_crash_is_told_once(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await use(first, ic(first), ProposeBoundarySplitTool, SPLIT)
    ended = end_server_at_message(launcher, "Your split was approved")
    await cut_off(launcher, ended, use(first, manager(first), DecideSplitTool, APPROVE))

    second = await again(repo, launcher)
    assert ("root", "message") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    assert node(second, "root").pending_split is None
    assert len(launcher.derived) == 5
    told = [text for text in launcher.messages_to(ic(second)) if "approved" in text]
    assert len(told) == 1
    await second.aclose()


async def test_a_review_request_cut_off_by_a_crash_reaches_the_reviewer_once(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    commit(repo, "services/api/app.py", "order_id = '1'\n")
    # The server ends as the team is asked for its account of the work.
    ended = end_server_at(launcher, "ask_agent")
    await cut_off(
        launcher, ended, use(first, ic(first, "services.api"), SetStatusTool, DONE)
    )

    second = await again(repo, launcher)
    assert ("services.api", "continue") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    review = node(second, "services.api").review
    assert review is not None and review.state == "pending"
    asked = [text for text in launcher.messages_to(ic(second)) if "Review mode" in text]
    assert len(asked) == 1
    await second.aclose()


async def test_an_acceptance_the_server_ended_before_saving_is_not_lost(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    await split_root(first)
    await api_reports_done(first, repo)
    ended = asyncio.Event()

    def nothing_more(_: str) -> None:
        if ended.is_set():
            raise ServerEnded

    async def end_instead_of_saving() -> None:
        ended.set()
        raise ServerEnded

    # The server ends with the acceptance made and the tree not yet saved.
    launcher.before = nothing_more
    monkeypatch.setattr(first, "_save", end_instead_of_saving)
    await cut_off(launcher, ended, use(first, ic(first), ReviewTeamTool, accept()))

    second = await again(repo, launcher)
    saved = node(second, "services.api").review
    assert saved is not None and saved.state == "pending"
    assert ("services.api", "continue") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    review = node(second, "services.api").review
    assert review is not None and review.state == "accepted"
    told = [
        text
        for text in launcher.messages_to(ic(second, "services.api"))
        if "accepted" in text
    ]
    # Told at least once; a message recorded before the end is sent as well.
    assert 1 <= len(told) <= 2
    await second.aclose()


async def test_stopping_a_team_cut_off_by_a_crash_stops_all_its_agents(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    for attempt in range(3):
        commit(repo, "services/api/app.py", f"order_id = {attempt}\n")
        await use(first, ic(first, "services.api"), SetStatusTool, DONE)
        if attempt < 2:
            await use(first, ic(first), ReviewTeamTool, reject())
    # The third rejection stops the team; the server ends at the first stop.
    ended = end_server_at(launcher, "stop")
    await cut_off(launcher, ended, use(first, ic(first), ReviewTeamTool, reject()))
    assert launcher.stopped == []
    await first._save()

    second = await again(repo, launcher)
    assert ("services.api", "continue") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    api = node(second, "services.api")
    assert api.status == "blocked"
    assert {UUID(str(agent.conversation_id)) for agent in api.agents} <= set(
        launcher.stopped
    )
    await second.aclose()


async def test_an_agent_cut_off_in_a_step_shows_paused_is_told_and_runs_again(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    root_ic = ic(first)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    # The server ends; what was running is in error when it is loaded again.
    launcher.known[UUID(root_ic)] = ConversationExecutionStatus.ERROR

    second = await again(repo, launcher)

    assert second.summary().paused == "crash"
    root = node(second, "root")
    assert root.agents[0].status == AGENT_PAUSED and root.status != "blocked"
    assert ("root", "run_again") in kinds(await second.recovery())

    await second.resume()
    await second.settled()

    assert launcher.resumed == [(UUID(root_ic), RESTART_NOTE)]
    await second.aclose()


async def test_a_server_stopped_with_nothing_under_way_comes_back_running(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await first.aclose()

    second = await again(repo, launcher)

    assert second.summary().paused is None
    assert (await second.recovery()).items == []
    await second.aclose()


async def test_an_org_set_to_resume_on_start_takes_its_work_up_unasked(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    first.update_settings(DirectorSettingsUpdate(resume_on_start=True))
    await first.pause()
    first.notify(ic(first), "Carry on")
    await first.settled()
    await first.aclose()
    before = len(launcher.sent)

    second = await again(repo, launcher)

    # A human's pause holds across restarts; only a stop or a crash does not.
    assert second.summary().paused == "human" and len(launcher.sent) == before
    await second.resume()
    await second.settled()
    assert "Carry on" in launcher.messages_to(ic(second))[-1]

    launcher.report(str(ic(second)), ConversationExecutionStatus.RUNNING)
    await second.aclose()
    launcher.known[UUID(str(ic(second)))] = ConversationExecutionStatus.PAUSED
    third = await again(repo, launcher)

    assert third.summary().paused is None
    assert launcher.resumed[-1] == (UUID(str(ic(third))), None)
    await third.aclose()


async def test_run_state_of_a_team_the_tree_no_longer_has_is_put_aside_and_named(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    api_ic = ic(first, "services.api")
    await first.pause()
    first.notify(api_ic, "Rename the endpoint")
    first.notify(ic(first), "Carry on")
    await first.settled()
    await first.aclose()
    # The tree moves under the run state, as a pull does.
    board, pulled = create_org_board(repo), OrgStore()
    load_org(board, pulled)
    pulled.remove("services.api")
    save_org(board, pulled)
    before = len(launcher.sent)

    second = await again(repo, launcher)
    left = await second.recovery()

    dropped = [item for item in left.items if item.kind == "dropped"]
    assert [item.node_id for item in dropped] == ["services.api"]
    assert "no longer in the org" in dropped[0].detail
    assert "Rename the endpoint" in dropped[0].detail
    assert left.messages_held == 1

    resumed = await second.resume()
    await second.settled()

    assert [item.node_id for item in resumed.items] == ["services.api"]
    assert all(
        "Rename the endpoint" not in text for text in launcher.messages_to(api_ic)
    )
    assert len(launcher.sent) == before + 1
    await second.aclose()


async def test_run_state_a_team_changed_under_is_put_aside(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await first.pause()
    first.notify(ic(first), "Carry on")
    await first.settled()
    await first.aclose()
    board, pulled = create_org_board(repo), OrgStore()
    load_org(board, pulled)
    pulled.update_node("root", branch="someone-elses")
    save_org(board, pulled)
    before = len(launcher.sent)

    second = await again(repo, launcher)
    resumed = await second.resume()
    await second.settled()

    assert ["changed since" in item.detail for item in resumed.items] == [True]
    assert len(launcher.sent) == before
    await second.aclose()


async def test_run_state_written_by_another_server_is_not_used(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await first.pause()
    first.notify(ic(first), "Carry on")
    await first.settled()
    await first.aclose()
    before = len(launcher.sent)

    # The repository, run state and all, is opened by another server.
    launcher.listeners.clear()
    other = DirectorService(
        repo, save_delay=0, remembered=repo.parent / "elsewhere" / "director.json"
    )
    await other.start(launcher)
    assert other.store.root_id == "root"
    await other.settled()

    assert other.summary().paused is None
    await other.resume()
    await other.settled()
    assert len(launcher.sent) == before
    await other.aclose()


def test_a_worktree_that_is_gone_is_made_again_from_its_branch(repo: Path):
    worktree = repo.parent / "worktrees" / "api"
    run_git(repo, "worktree", "add", "--quiet", "-b", "team/api", str(worktree))
    (worktree / "api.py").write_text("ok\n")
    run_git(worktree, "add", "api.py")
    run_git(worktree, "commit", "--quiet", "-m", "api")
    run_git(repo, "worktree", "remove", "--force", str(worktree))
    kept = make_node("api", "root").model_copy(
        update={"worktree_path": str(worktree), "branch": "team/api"}
    )
    gone = make_node("web", "root").model_copy(
        update={"worktree_path": f"{worktree}-web", "branch": "team/web"}
    )
    root = make_node("root", None).model_copy(
        update={"isolation": "shared", "worktree_path": str(repo)}
    )
    store = OrgStore()
    store.replace_all([root, kept, gone])

    told = describe(RunState(None), store, {}, repo)
    restore_worktrees(repo, store.all(), "")

    assert [(item.node_id, item.kind) for item in told] == [
        ("api", "continue"),
        ("web", "cannot_resume"),
    ]
    assert (worktree / "api.py").read_text() == "ok\n"
    assert run_git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == "team/api"
    assert not Path(str(worktree) + "-web").exists()


def test_run_state_is_kept_as_it_was_written_and_in_order(tmp_path: Path):
    path = tmp_path / "git" / "director" / "state.db"
    run = RunState(path)
    split_ = run.begin("split", "root", {"why": "too large"})
    run.begin("accept", "api", {})
    done = run.begin("halt", "web", {})
    run.finish(done.id)
    run.step_done(split_, "planned", teams=["a", "b"])
    sent = run.hold("c1", "root", "one")
    run.hold("c2", "api", "two")
    run.delivered(sent)
    run.mark("waiting", "b")
    run.mark("waiting", "a")
    run.mark("waiting", "b", "again")
    run.close()

    kept = RunState(path)

    first, second = kept.open_intents()
    assert (first.kind, first.steps, first.payload) == (
        "split",
        frozenset({"planned"}),
        {"why": "too large", "teams": ["a", "b"]},
    )
    assert second.kind == "accept"
    assert [(held.conversation_id, held.text) for held in kept.held()] == [
        ("c2", "two")
    ]
    assert list(kept.marked("waiting").items()) == [("b", "again"), ("a", "")]


def test_run_state_of_a_newer_director_is_refused(tmp_path: Path):
    path = tmp_path / "state.db"
    newer = RunState(path)
    newer.put("schema_version", "99")
    newer.close()

    with pytest.raises(RunStateError, match="version 99"):
        RunState(path)


def test_run_state_stays_out_of_the_repository(repo: Path):
    service = DirectorService(repo)
    service.store.replace_all([make_node("root", None)])

    assert service.summary().paused is None
    assert run_git(repo, "status", "--porcelain") == ""
    assert (repo / ".git" / "director" / "state.db").is_file()
