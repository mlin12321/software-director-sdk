"""Pausing an org, and taking up what a stopped server left unfinished."""

import asyncio
import sqlite3
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from openhands.agent_server.director.git_store import CONTRACTS_FILE
from openhands.agent_server.director.intents import Accept, Prune, Split
from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    OpenSessionRequest,
)
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
    save_org,
)
from openhands.agent_server.director.org_store import OrgStore, team_id_of
from openhands.agent_server.director.recovery import describe, restore_worktrees
from openhands.agent_server.director.run_state import (
    RUNNING,
    SESSION,
    RunState,
    RunStateError,
)
from openhands.agent_server.director.service import (
    AGENT_PAUSED,
    RESTART_NOTE,
    DirectorService,
)
from openhands.agent_server.director.service.state import _Held
from openhands.agent_server.director.session_state import kept_sessions
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    PruneTeamTool,
    ReviewTeamTool,
    SettleTeamsTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import SettleTeamsAction
from openhands.sdk.conversation.state import ConversationExecutionStatus
from tests.agent_server.director.conftest import (
    APPROVE,
    DONE,
    JOBS,
    PRUNE_API,
    SCHEMA,
    SPLIT,
    FakeLauncher,
    ServerEnded,
    accept,
    again,
    api_submits,
    at_api,
    commit,
    cut_off,
    division_lead_manager,
    end_server_at,
    eventually,
    former_of,
    ic,
    landed,
    make_team,
    manager,
    memory_of,
    node,
    open_session_in,
    org_gains,
    path_of,
    record_of,
    run_git,
    session_accepted,
    sets_up_jobs,
    shipped_type,
    split,
    split_api,
    split_root,
    started,
    tasked,
    use,
    waiting,
    woken,
)


def kinds(recovery) -> list[tuple[str, str]]:
    return [(item.node_id, item.kind) for item in recovery.items]


def place(service: DirectorService, team: str = "root") -> str:
    """The id of what the session opened first has at a team."""
    return node(service, team).id


async def test_a_paused_org_holds_messages_and_delivers_them_in_order_on_resume(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    before = len(launcher.sent)

    paused = await service.pause()
    service.tell_writer(place(service), "First", human=True)
    service.notify(ic(service), "Second")
    await service.settled()

    assert paused.paused == "human" and service.summary().paused == "human"
    assert len(launcher.sent) == before
    held = await service.recovery()
    assert held.messages_held == 2 and (place(service), "message") in kinds(held)

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
    assert statuses == {"ic": AGENT_PAUSED}
    assert (place(service), "run_again") in kinds(await service.recovery())

    await service.resume()
    await service.settled()

    assert launcher.resumed == [(UUID(root_ic), None)]
    assert node(service, "root").status == "working"
    await service.aclose()


async def test_what_an_agent_decides_while_the_org_is_paused_is_done_after_resume(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    await use(service, ic(service), ProposeTeamsTool, SPLIT)

    await service.pause()
    answer = await use(service, manager(service), ApproveTeamsTool, APPROVE)

    assert not answer.is_error and node(service, "root").child_ids == []
    assert (place(service), "continue") in kinds(await service.recovery())

    await service.resume()
    await service.settled()
    assert len(launcher.derived) == 1
    await tasked(service)

    assert node(service, "root").child_ids == [
        place(service, "services.api"),
        place(service, "apps.web"),
    ]
    assert node(service, "root").pending_split is None
    assert len(launcher.derived) == 3
    await service.aclose()


async def test_a_server_stopped_while_teams_wait_starts_them_in_order_once_resumed(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher, run_limit_retry=3600)
    await woken(first)
    launcher.full_for = 1_000_000
    await use(first, ic(first), ProposeTeamsTool, split("a", 3))
    await use(first, manager(first), ApproveTeamsTool, APPROVE)
    top = place(first)
    for team in first.org.below(top):
        first.tell_writer(team.id, "Begin.")
    await eventually(
        lambda: sum(team.waiting_to_start for team in first.org.below(top)) == 3
    )
    await first.aclose()

    launcher.full_for = 0
    second = await again(repo, launcher)
    left = await second.recovery()

    assert second.summary().paused == "shutdown" and len(launcher.derived) == 1
    waited = [node_id for node_id, kind in kinds(left) if kind == "start"]
    assert sorted(team_id_of(node(second, node_id)) for node_id in waited) == [
        "a.t0",
        "a.t1",
        "a.t2",
    ]
    # A team that has not been started yet has lost nothing.
    assert {kind for _, kind in kinds(left)} == {"start"}
    assert {
        agent.status for team in second.org.below(top) for agent in team.agents
    } == {"idle"}

    await second.resume()
    await second.settled()

    teams = second.org.below(top)
    assert [team_id_of(team) for team in teams] == ["a.t0", "a.t1", "a.t2"]
    assert node(second, "root").pending_split is None
    starts = [str(start.conversation_id) for start in launcher.derived[1:]]
    assert len(set(starts)) == len(starts) == 3
    in_order = [ic(second, node_id) for node_id in waited]
    assert [start for start in starts if start in in_order] == in_order
    await second.aclose()


async def test_a_team_start_cut_off_by_a_crash_is_finished_without_a_second_of_anything(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await use(first, ic(first), ProposeTeamsTool, SPLIT)
    await use(first, manager(first), ApproveTeamsTool, APPROVE)
    # The server ends as the first sub-team's IC has just been started.
    ended = end_server_at(launcher, "derived")
    api = place(first, "services.api")
    first.tell_writer(api, "Begin.")
    await ended.wait()
    assert len(launcher.derived) <= 2

    second = await again(repo, launcher)

    assert second.summary().paused == "crash"
    assert (api, "start") in kinds(await second.recovery())

    await second.resume()
    await second.settled()
    await tasked(second)

    assert node(second, "root").child_ids == [api, place(second, "apps.web")]
    assert node(second, "root").pending_split is None
    starts = [str(start.conversation_id) for start in launcher.derived]
    assert len(set(starts)) == len(starts) == 3
    contracts = memory_of(second, CONTRACTS_FILE) or ""
    assert contracts.count("POST /checkout returns {order_id}") == 1
    told = [
        text
        for text in launcher.messages_to(ic(second))
        if "Your team was split" in text
    ]
    assert len(told) == 1
    await second.aclose()


async def test_a_prune_cut_off_by_a_crash_is_finished_and_recorded_once(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await woken(first, "services.api")
    api = place(first, "services.api")
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "half.py").write_text("broken\n")
    ended = end_server_at(launcher, "delete", nth=2)
    prune = asyncio.create_task(use(first, manager(first), PruneTeamTool, PRUNE_API))
    await ended.wait()
    prune.cancel()
    await asyncio.gather(prune, return_exceptions=True)
    assert len(launcher.deleted) == 1

    second = await again(repo, launcher)
    assert (api, "continue") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    assert second.org.place(api) is None
    assert len(set(launcher.deleted)) == 2
    kept = run_git(repo, "for-each-ref", "--format=%(refname)", "refs/agents/pruned")
    assert len(kept.split()) == 1
    assert len(former_of(second, "services.api")) == 1
    told = [text for text in launcher.messages_to(ic(second)) if "pruned" in text]
    assert len(told) == 1
    await second.aclose()


async def test_a_prune_cut_off_after_it_is_recorded_is_recorded_once(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    await split_root(first)
    await woken(first, "services.api")
    api = place(first, "services.api")
    ended = asyncio.Event()

    def server_ends(*_: object) -> None:
        ended.set()
        raise ServerEnded

    # The team is on record as pruned by now, and not yet removed.
    monkeypatch.setattr(first, "_specs_left_in", server_ends)
    await cut_off(launcher, ended, use(first, manager(first), PruneTeamTool, PRUNE_API))

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert second.org.place(api) is None
    pruned = [event for event in record_of(second, "root") if event.kind == "pruned"]
    assert len(pruned) == 1
    await second.aclose()


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
    await use(first, ic(first), ProposeTeamsTool, SPLIT)
    ended = end_server_at_message(launcher, "Your team was split")
    await cut_off(
        launcher, ended, use(first, manager(first), ApproveTeamsTool, APPROVE)
    )

    second = await again(repo, launcher)
    assert (place(second), "message") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    assert node(second, "root").pending_split is None
    assert len(launcher.derived) == 1
    told = [
        text
        for text in launcher.messages_to(ic(second))
        if "Your team was split" in text
    ]
    assert len(told) == 1
    await second.aclose()


async def test_the_end_of_a_split_cut_off_before_it_is_told_is_told(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    await use(first, ic(first), ProposeTeamsTool, SPLIT)
    ended = asyncio.Event()

    def server_ends(*_: object) -> None:
        # The session is kept as it is now, before the IC is told.
        first._runs().replace_marks(SESSION, kept_sessions(first.org))
        ended.set()
        raise ServerEnded

    monkeypatch.setattr(first, "_confine", server_ends)
    await cut_off(
        launcher, ended, use(first, manager(first), ApproveTeamsTool, APPROVE)
    )

    second = await again(repo, launcher)
    await second.resume()
    await second.settled()

    assert node(second, "root").pending_split is None
    told = [
        text
        for text in launcher.messages_to(ic(second))
        if "Your team was split" in text
    ]
    assert len(told) == 1
    await second.aclose()


async def test_a_team_in_line_to_start_is_held_when_the_org_is_paused(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    service._line.extend(["ahead", "behind"])

    async def start() -> None:
        raise AssertionError("it is not its turn")

    waits = asyncio.create_task(service._when_there_is_room("behind", start))
    await asyncio.sleep(0)
    await service.pause()

    # The one ahead of it may never start now.
    with pytest.raises(_Held):
        await asyncio.wait_for(waits, 1)
    await service.aclose()


async def test_a_review_request_cut_off_by_a_crash_reaches_the_reviewer_once(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    commit(repo, "apps/web/page.tsx", "export {}\n")
    await use(first, ic(first, "apps.web"), SubmitReviewTool, DONE)
    commit(repo, "services/api/app.py", "order_id = '1'\n")
    # The server ends as the last team is asked for its account of the work.
    ended = end_server_at(launcher, "ask_agent")
    await cut_off(
        launcher, ended, use(first, ic(first, "services.api"), SubmitReviewTool, DONE)
    )
    # A save the ended server had begun is not one the next start reads half of.
    await asyncio.to_thread(first._wait_for_save)

    second = await again(repo, launcher)
    assert (place(second, "services.api"), "continue") in kinds(await second.recovery())
    await second.resume()
    await second.settled()

    review = node(second, "services.api").review
    assert review is not None and review.state == "pending"
    asked = "\n".join(
        text for text in launcher.messages_to(ic(second)) if "Review request" in text
    )
    # Each went when its team submitted, and neither twice: the one the
    # crash cut off goes after it, and names the other as still to decide.
    assert asked.count("Team 'Web' (id") == 1 and asked.count("Team 'API' (id") == 1
    assert "Still waiting for your decision" in asked
    await second.aclose()


async def test_an_acceptance_the_server_ended_before_saving_is_not_lost(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    await split_root(first)
    await api_submits(first, repo)
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
    assert (place(second, "services.api"), "continue") in kinds(await second.recovery())
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


async def test_accepted_work_the_server_ended_before_passing_up_still_goes_up(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    await split_root(first)
    await split_api(first)
    commit(repo, "services/api/schema/order.py", "id: int\n")
    await use(first, ic(first, SCHEMA), SubmitReviewTool, DONE)
    ended = asyncio.Event()

    async def end_instead(node_id: str, *, acted: bool) -> None:
        await first._save()
        ended.set()
        raise ServerEnded

    # The server ends with the sub-team's work accepted and saved, before
    # the team above it is looked at.
    monkeypatch.setattr(first, "_pass_up", end_instead)
    await cut_off(
        launcher,
        ended,
        use(first, ic(first, "services.api"), ReviewTeamTool, accept(SCHEMA)),
    )

    second = await again(repo, launcher)
    saved = node(second, SCHEMA).review
    assert saved is not None and saved.state == "accepted"
    assert node(second, "services.api").review is None
    await second.resume()
    await second.settled()

    api = node(second, "services.api")
    assert api.review is not None and api.review.state == "accepted"
    assert api.status == "done"
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
    assert (place(second), "run_again") in kinds(await second.recovery())

    await second.resume()
    await second.settled()

    assert launcher.resumed == [(UUID(root_ic), RESTART_NOTE)]
    await second.aclose()


async def test_an_agent_cut_off_in_a_step_is_not_a_failure_when_it_is_run_again(
    repo: Path, launcher: FakeLauncher, monkeypatch: pytest.MonkeyPatch
):
    first = await started(repo, launcher)
    root_ic = ic(first)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    launcher.known[UUID(root_ic)] = ConversationExecutionStatus.ERROR
    second = await again(repo, launcher)

    async def told_before_it_runs(conversation_id: UUID, note: str | None) -> None:
        # The conversation is loaded and told; its run starts a moment later.
        launcher.resumed.append((conversation_id, note))
        launcher.report(str(conversation_id), ConversationExecutionStatus.ERROR)

    monkeypatch.setattr(launcher, "resume", told_before_it_runs)

    await second.resume()
    await second.settled()

    root = node(second, "root")
    assert root.agents[0].status == AGENT_PAUSED and root.escalations == []
    assert [e for e in record_of(second, "root") if e.kind == "failed"] == []

    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    assert node(second, "root").agents[0].status == "working"
    # From here on an error is its own.
    launcher.report(root_ic, ConversationExecutionStatus.ERROR)
    await second.settled()
    assert node(second, "root").agents[0].status == "blocked"
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


async def test_a_session_stays_when_the_org_no_longer_has_its_team(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await split_root(first)
    await landed(first)
    at_api = await first.open_session(
        OpenSessionRequest(text="Rename it", node_id="services.api")
    )
    at_web = await first.open_session(
        OpenSessionRequest(text="Restyle it", node_id="apps.web")
    )
    await first.settled()
    await first.pause()
    first.notify(at_api.conversation_id, "Rename the endpoint")
    first.notify(at_web.conversation_id, "Carry on")
    await first.settled()
    await first.aclose()
    # The tree moves under the run state, as a pull does.
    board, pulled = create_org_board(repo), OrgStore(shipped_type)
    load_org(board, pulled)
    pulled.remove("services.api")
    save_org(board, pulled)
    before = len(launcher.sent)

    second = await again(repo, launcher)
    left = await second.recovery()

    # A session is not dropped with its team: what becomes of the team is
    # decided when the session's work is reviewed.
    assert second.org.team("services.api") is None
    open_now = [session.id for session in second.org.sessions()]
    assert at_api.session.id in open_now and at_web.session.id in open_now
    assert [item for item in left.items if item.kind == "dropped"] == []
    assert left.messages_held == 2

    await second.resume()
    await second.settled()

    assert "Rename the endpoint" in launcher.messages_to(at_api.conversation_id)[-1]
    assert "Carry on" in launcher.messages_to(at_web.conversation_id)[-1]
    assert len(launcher.sent) == before + 2
    await second.aclose()


async def test_a_sessions_run_state_is_kept_when_its_team_changed_in_the_org(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    await first.pause()
    first.notify(ic(first), "Carry on")
    await first.settled()
    await first.aclose()
    board, pulled = create_org_board(repo), OrgStore(shipped_type)
    load_org(board, pulled)
    pulled.update_team("root", headline="Changed elsewhere")
    save_org(board, pulled)
    before = len(launcher.sent)

    second = await again(repo, launcher)
    resumed = await second.resume()
    await second.settled()

    # What a session has at a team is this machine's, whatever the org's
    # team became elsewhere.
    assert resumed.items == []
    assert "Carry on" in launcher.messages_to(ic(second))[-1]
    assert len(launcher.sent) == before + 1
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
    assert other.org.root_id == "root"
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
    store = OrgStore(shipped_type)
    store.replace_all(
        [
            make_team("root", None, path=""),
            make_team("api", "root"),
            make_team("web", "root"),
        ]
    )
    open_session_in(store, checkout=str(repo))
    store.update_place("api@s", worktree_path=str(worktree), branch="team/api")
    store.update_place("web@s", worktree_path=f"{worktree}-web", branch="team/web")

    told = describe(RunState(None), store, {}, repo)
    restore_worktrees(repo, store.places(), "")

    assert [(item.node_id, item.kind) for item in told] == [
        ("api@s", "continue"),
        ("web@s", "cannot_resume"),
    ]
    assert (worktree / "api.py").read_text() == "ok\n"
    assert run_git(worktree, "rev-parse", "--abbrev-ref", "HEAD").strip() == "team/api"
    assert not Path(str(worktree) + "-web").exists()


def test_run_state_is_kept_as_it_was_written_and_in_order(tmp_path: Path):
    path = tmp_path / "git" / "director" / "state.db"
    run = RunState(path)
    split_ = run.begin("root", Split())
    run.begin("api", Accept(needed_above="The API waits on it."))
    done = run.begin("web", Prune(reason="Not needed."))
    run.finish(done.id)
    run.step_done(split_, "planned", replaces=["a", "b"], stamp="s1")
    sent = run.hold("c1", "root", "one")
    run.hold("c2", "api", "two")
    run.delivered(sent)
    run.mark("waiting", "b")
    run.mark("waiting", "a")
    run.mark("waiting", "b", "again")
    run.close()

    kept = RunState(path)

    first, second = kept.open_intents()
    assert (first.kind, first.steps) == ("split", frozenset({"planned"}))
    assert first.payload == Split(replaces=["a", "b"], stamp="s1")
    assert second.payload == Accept(needed_above="The API waits on it.")
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
    service.org.replace_all([make_team("root", None)])

    assert service.summary().paused is None
    assert run_git(repo, "status", "--porcelain") == ""
    assert (repo / ".git" / "director" / "state.db").is_file()


async def test_teams_that_wait_are_still_there_after_a_restart(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    top, conversation = await at_api(first)
    sets_up_jobs(first, top)
    org_gains(first, JOBS, "services/api/workers")
    await session_accepted(first, top, conversation)
    await first.aclose()

    second = await again(repo, launcher)

    assert waiting(second, top) == [JOBS]
    await use(
        second, division_lead_manager(second, top), SettleTeamsTool, SettleTeamsAction()
    )
    assert path_of(second, JOBS) == "services/api/jobs"
    await second.aclose()


async def test_another_servers_run_state_brings_back_none_of_its_sessions(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    first = await started(repo, launcher)
    await split_root(first)
    await first.aclose()
    launcher.listeners.clear()
    other = DirectorService(
        repo, save_delay=0, remembered=tmp_path / "elsewhere" / "director.json"
    )

    await other.start(launcher)
    await other.settled()

    # Their drafts, reports and place in the order are not used either.
    assert other.org.sessions() == []
    await other.aclose()


def test_a_state_file_that_is_no_database_leaves_the_director_working(repo: Path):
    state = repo / ".git" / "director" / "state.db"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"this is not a database" * 100)
    service = DirectorService(repo, save_delay=0)

    assert service._runs().path is None


async def test_resuming_waits_for_a_pause_that_is_still_stopping_agents(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    root_ic = ic(service)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    stopped, pause = asyncio.Event(), launcher.pause

    async def slow_pause(conversation_id: UUID) -> None:
        await stopped.wait()
        await pause(conversation_id)

    launcher.pause = slow_pause  # type: ignore[method-assign]

    await service.pause()
    resumed = asyncio.create_task(service.resume())
    await asyncio.sleep(0.05)
    waits = not resumed.done()
    stopped.set()
    await resumed
    await service.settled()

    # Or the pause would stop the agent after it was set going again.
    assert waits
    assert launcher.resumed == [(UUID(root_ic), None)]
    await service.aclose()


async def test_an_agent_not_yet_run_again_is_taken_up_by_the_next_start(
    repo: Path, launcher: FakeLauncher
):
    first = await started(repo, launcher)
    root_ic = ic(first)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    launcher.known[UUID(root_ic)] = ConversationExecutionStatus.ERROR
    second = await again(repo, launcher)
    never = asyncio.Event()

    async def cut_off(conversation_id: UUID, note: str | None) -> None:
        await never.wait()

    launcher.resume = cut_off  # type: ignore[method-assign]

    await second.resume()
    await asyncio.sleep(0.05)

    # The server ends here a second time: the agent is still one to run again.
    assert root_ic in second._runs().marked(RUNNING)
    never.set()
    await second.aclose()


def test_a_failed_change_of_several_marks_leaves_the_run_state_usable(
    tmp_path: Path,
):
    run = RunState(tmp_path / "state.db")
    run.replace_marks("kind", {"a": "1"})

    with pytest.raises(sqlite3.Error):
        run.replace_marks("kind", {"b": cast(str, object())})

    # Nothing of the failed change stays, and the next one is taken.
    assert run.marked("kind") == {"a": "1"}
    run.replace_marks("kind", {"c": "3"})
    assert run.marked("kind") == {"c": "3"}
    run.close()


async def test_two_resumes_at_once_run_an_agent_again_once(
    repo: Path, launcher: FakeLauncher
):
    service = await started(repo, launcher)
    root_ic = ic(service)
    assert root_ic is not None
    launcher.report(root_ic, ConversationExecutionStatus.RUNNING)
    await service.pause()
    await service.settled()

    await asyncio.gather(service.resume(), service.resume())
    await service.settled()

    assert launcher.resumed == [(UUID(root_ic), None)]
    await service.aclose()


def test_a_step_cannot_record_what_its_kind_does_not_have(tmp_path: Path):
    run = RunState(tmp_path / "state.db")
    prune = run.begin("web", Prune(reason="Not needed."))

    with pytest.raises(ValidationError):
        run.step_done(prune, "planned", tital="Web")

    [kept] = run.open_intents()
    assert kept.steps == frozenset() and kept.payload == Prune(reason="Not needed.")


def test_run_state_that_holds_something_it_cannot_read_is_refused(tmp_path: Path):
    path = tmp_path / "state.db"
    run = RunState(path)
    run.begin("web", Prune(reason="Not needed."))
    run.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE intents SET kind = 'unheard_of'")

    with pytest.raises(RunStateError, match="unheard_of"):
        RunState(path).open_intents()
