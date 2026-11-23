"""An org pulled from and pushed to a remote of its repository, as a merge."""

from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.git_store import SPEC_FILE
from openhands.agent_server.director.models import (
    OrgNode,
    ResolveMergeRequest,
    RouteMessageRequest,
    SnapshotEvent,
    StartAgainRequest,
)
from openhands.agent_server.director.org_merge import by_id, merge_orgs
from openhands.agent_server.director.org_persistence import create_org_board
from openhands.agent_server.director.org_store import (
    AGENT_LOST,
    OrgChange,
    team_id_of,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools import SubmitReviewTool
from openhands.sdk.errors import Conflict, Unprocessable
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_team,
    root_conversation,
    run_git,
)
from tests.agent_server.director.test_director_landing import landed
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_review import DONE
from tests.agent_server.director.test_director_tools import (
    division_lead_ic,
    division_lead_of,
    ic,
    manager,
    node,
    split_root,
    use,
)


API, WEB = "services.api", "apps.web"


def tree(*changes: tuple[str, str, object]) -> dict[str, dict]:
    """A root with teams ``a`` and ``b``, as saved, with fields changed."""
    nodes = by_id(
        [
            make_team("root", None, path="").model_dump(),
            make_team("a", "root").model_dump(),
            make_team("b", "root").model_dump(),
        ]
    )
    for node_id, field, value in changes:
        nodes[node_id] = {**nodes[node_id], field: value}
    return nodes


def without(nodes: dict[str, dict], *gone: str) -> dict[str, dict]:
    return {node_id: node for node_id, node in nodes.items() if node_id not in gone}


def test_changes_to_different_teams_and_fields_are_all_kept():
    mine = tree(("a", "headline", "A moved on"), ("b", "title", "Billing"))
    theirs = tree(("b", "headline", "B moved on"), ("root", "headline", "Done"))

    merged = merge_orgs(tree(), mine, theirs, {})

    assert merged.conflicts == []
    assert by_id(merged.nodes) == tree(
        ("a", "headline", "A moved on"),
        ("b", "title", "Billing"),
        ("b", "headline", "B moved on"),
        ("root", "headline", "Done"),
    )


def test_one_field_changed_on_both_sides_waits_for_a_choice():
    mine, theirs = tree(("a", "headline", "Mine")), tree(("a", "headline", "Theirs"))

    open_ = merge_orgs(tree(), mine, theirs, {})
    chosen = merge_orgs(tree(), mine, theirs, {"field:a:headline": "theirs"})
    kept = merge_orgs(tree(), mine, theirs, {"field:a:headline": "mine"})

    (conflict,) = open_.conflicts
    assert (conflict.kind, conflict.node_id, conflict.field) == (
        "field",
        "a",
        "headline",
    )
    assert (conflict.mine, conflict.theirs, conflict.choice) == ("Mine", "Theirs", None)
    assert by_id(open_.nodes)["a"]["headline"] == "Mine"
    assert by_id(chosen.nodes)["a"]["headline"] == "Theirs"
    assert by_id(kept.nodes)["a"]["headline"] == "Mine"
    assert chosen.conflicts[0].choice == "theirs"


def test_the_same_change_on_both_sides_is_no_conflict():
    both = tree(("a", "headline", "Agreed"))

    merged = merge_orgs(tree(), both, both, {})

    assert merged.conflicts == [] and by_id(merged.nodes) == both


def test_a_team_removed_on_one_side_and_untouched_on_the_other_is_removed():
    below = make_team("a.x", "a").model_dump()
    base = {**tree(), "a.x": below}

    mine_removed = merge_orgs(base, without(base, "a", "a.x"), base, {})
    theirs_removed = merge_orgs(base, base, without(base, "a", "a.x"), {})

    assert mine_removed.conflicts == [] and theirs_removed.conflicts == []
    assert set(by_id(mine_removed.nodes)) == {"root", "b"}
    assert set(by_id(theirs_removed.nodes)) == {"root", "b"}


def test_a_team_the_removing_side_moved_out_from_under_it_is_kept_where_it_went():
    below = make_team("a.x", "a").model_dump()
    base = {**tree(), "a.x": below}
    # A team replaced: what was below it is now below the team above.
    replaced = {**without(base, "a"), "a.x": {**below, "parent_id": "root"}}

    for merged in (
        merge_orgs(base, base, replaced, {}),
        merge_orgs(base, replaced, base, {}),
    ):
        assert merged.conflicts == [] and by_id(merged.nodes) == replaced


def test_a_team_moved_from_under_a_team_the_other_side_removed_is_a_conflict():
    below = make_team("a.x", "a").model_dump()
    base = {**tree(), "a.x": below}
    moved = {**base, "a.x": {**below, "parent_id": "b"}}
    removed = without(base, "a", "a.x")

    mine_moved = merge_orgs(base, moved, removed, {})
    theirs_moved = merge_orgs(base, removed, moved, {})
    taken = merge_orgs(base, removed, moved, {"team:a.x": "theirs"})

    for merged in (mine_moved, theirs_moved):
        assert [c.id for c in merged.conflicts] == ["team:a.x"]
    # The removal of the team it was under stands; it is kept where it went.
    assert by_id(mine_moved.nodes).keys() == {"root", "b", "a.x"}
    assert by_id(theirs_moved.nodes).keys() == {"root", "b"}
    assert by_id(taken.nodes)["a.x"]["parent_id"] == "b"


def test_a_team_moved_under_a_team_the_other_side_removed_stays_where_it_was():
    below = make_team("a.x", "a").model_dump()
    base = {**tree(), "a.x": below}
    mine = {**base, "a.x": {**below, "parent_id": "b"}}
    theirs = without(base, "b")

    removed = merge_orgs(base, mine, theirs, {"team:b": "theirs"})

    assert by_id(removed.nodes).keys() == {"root", "a", "a.x"}
    assert by_id(removed.nodes)["a.x"]["parent_id"] == "a"


def test_a_team_removed_on_one_side_and_changed_below_on_the_other_is_a_conflict():
    below = make_team("a.x", "a").model_dump()
    base = {**tree(), "a.x": below}
    theirs = {**base, "a.x": {**below, "headline": "Still going"}}
    mine = without(base, "a", "a.x")

    open_ = merge_orgs(base, mine, theirs, {})
    removed = merge_orgs(base, mine, theirs, {"team:a": "mine"})
    kept = merge_orgs(base, mine, theirs, {"team:a": "theirs"})
    reverse = merge_orgs(base, theirs, mine, {"team:a": "mine"})

    (conflict,) = open_.conflicts
    assert (conflict.kind, conflict.node_id) == ("team", "a")
    assert conflict.mine is None and conflict.theirs is not None
    assert set(by_id(open_.nodes)) == {"root", "b"}
    assert set(by_id(removed.nodes)) == {"root", "b"}
    assert by_id(kept.nodes)["a.x"]["headline"] == "Still going"
    assert by_id(reverse.nodes)["a.x"]["headline"] == "Still going"
    assert [node["id"] for node in kept.nodes].index("a") < [
        node["id"] for node in kept.nodes
    ].index("a.x")


def test_teams_added_on_either_side_are_kept():
    mine = {**tree(), "c": make_team("c", "root").model_dump()}
    theirs = {**tree(), "d": make_team("d", "b").model_dump()}

    merged = merge_orgs(tree(), mine, theirs, {})

    assert merged.conflicts == []
    assert set(by_id(merged.nodes)) == {"root", "a", "b", "c", "d"}


@pytest.fixture
def remote(repo: Path, tmp_path: Path) -> Path:
    """A bare repository ``repo`` has as its ``origin``."""
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--quiet", "--bare", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    run_git(repo, "push", "--quiet", "origin", "main")
    return bare


async def elsewhere(remote: Path, tmp_path: Path) -> tuple[DirectorService, Path]:
    """A director on a second clone of the remote, with no agents of its own."""
    clone = tmp_path / "elsewhere" / "repo"
    clone.parent.mkdir()
    run_git(tmp_path, "clone", "--quiet", str(remote), str(clone))
    service = DirectorService(
        clone,
        save_delay=0,
        run_limit_retry=0,
        remembered=clone.parent / "state" / "director.json",
    )
    await service.start(FakeLauncher())
    return service, clone


async def an_org_on_two_machines(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
) -> tuple[DirectorService, DirectorService]:
    """An org whose first session set up API and Web, and a second clone
    that pulled it."""
    here = await started(repo, launcher)
    await split_root(here)
    await landed(here)
    await here.push_org("origin")
    there, _ = await elsewhere(remote, tmp_path)
    await there.pull_org("origin")
    await there.settled()
    return here, there


def team(service: DirectorService, team_id: str) -> OrgNode:
    """A team as the org has it, whatever a session has there."""
    found = service.org.team(team_id)
    assert found is not None
    return found


def change(service: DirectorService, node_id: str, **fields: object) -> None:
    service.org.update_team(node_id, **fields)


async def test_a_pushed_org_is_whole_in_another_clone_except_for_its_sessions(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here = await started(repo, launcher)
    await split_root(here)
    here.report(node(here, API), "ic", "Never log card numbers")
    await landed(here)

    pushed = await here.push_org("origin")
    there, _ = await elsewhere(remote, tmp_path)
    pulled = await there.pull_org("origin")
    await there.settled()

    assert pushed.result == "pushed" and pulled.result == "fast_forward"
    assert [found.id for found in there.org.teams()] == ["root", API, WEB]
    assert team(there, API).title == "API" and team(there, API).spec_summary
    [carried] = there.reports_of(team(there, API))
    assert carried.text == "Never log card numbers"
    assert there.team_memory.read("contracts.md")
    # Who works on it is the first machine's alone.
    assert there.org.sessions() == [] and len(here.org.sessions()) == 1
    assert there.org.places() == []
    assert there.summary().paused is None and here.summary().paused is None
    await here.aclose()
    await there.aclose()


async def test_a_pull_takes_only_the_org_and_team_memory(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here = await started(repo, launcher)
    await split_root(here)
    await landed(here)
    await here.push_org("origin")
    # What a server keeps for itself is never given, and never taken where
    # someone put it on the remote all the same.
    head = run_git(repo, "rev-parse", "HEAD").strip()
    run_git(repo, "push", "--quiet", "origin", f"{head}:refs/agents/aside/left")
    assert sorted(
        line.split()[-1]
        for line in run_git(repo, "ls-remote", "origin", "refs/agents/*").splitlines()
    ) == ["refs/agents/aside/left", "refs/agents/memory", "refs/agents/org"]

    there, clone = await elsewhere(remote, tmp_path)
    await there.pull_org("origin")
    await there.settled()

    assert sorted(
        run_git(
            clone, "for-each-ref", "--format=%(refname)", "refs/remotes/origin/agents"
        ).split()
    ) == ["refs/remotes/origin/agents/memory", "refs/remotes/origin/agents/org"]
    await here.aclose()
    await there.aclose()


async def test_what_each_side_changed_is_merged_and_goes_back_with_the_next_push(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    mine = ic(here, API)

    change(here, API, headline="API has its routes")
    change(there, WEB, headline="Web has its pages")
    change(there, API, title="Orders API")
    await here.push_org("origin")
    with pytest.raises(Conflict, match="Pull first"):
        await there.push_org("origin")
    merged = await there.pull_org("origin")
    await there.push_org("origin")
    back = await here.pull_org("origin")
    await here.settled()

    assert merged.result == "merged" and merged.merge is None
    for side in (here, there):
        assert team(side, API).headline == "API has its routes"
        assert team(side, API).title == "Orders API"
        assert team(side, WEB).headline == "Web has its pages"
    assert back.result in ("merged", "fast_forward")
    # The session that works here is as it was.
    assert ic(here, API) == mine
    assert merged.head is not None
    clone = there.repo_dir
    commit, *parents = run_git(
        clone, "rev-list", "--parents", "-n", "1", merged.head
    ).split()
    assert commit == merged.head and len(parents) == 2
    await here.aclose()
    await there.aclose()


async def conflicting(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
) -> tuple[DirectorService, DirectorService]:
    """Two machines that both changed the headline of team API; the first
    has pushed."""
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    change(here, API, headline="Mine")
    change(there, API, headline="Theirs")
    change(there, WEB, headline="Web moved on")
    await there.push_org("origin")
    return here, there


async def test_a_pull_stops_on_a_conflict_and_writes_nothing(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await conflicting(repo, launcher, remote, tmp_path)
    before = create_org_board(repo).head()

    pulled = await here.pull_org("origin")

    assert pulled.result == "conflicts" and pulled.merge is not None
    (conflict,) = pulled.merge.conflicts
    assert (conflict.id, conflict.mine, conflict.theirs) == (
        f"field:{API}:headline",
        "Mine",
        "Theirs",
    )
    assert team(here, API).headline == "Mine" and team(here, WEB).headline != (
        "Web moved on"
    )
    assert create_org_board(repo).head() == before
    assert here.summary().merging and here.summary().paused == "human"
    assert await here.merge() == pulled.merge
    with pytest.raises(Conflict, match="merge waits"):
        await here.pull_org("origin")
    with pytest.raises(Conflict, match="merge waits"):
        await here.push_org("origin")
    await here.aclose()
    await there.aclose()


async def test_a_pull_that_fails_while_it_merges_leaves_the_org_running(
    repo: Path,
    launcher: FakeLauncher,
    remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    here, there = await conflicting(repo, launcher, remote, tmp_path)

    def fails(*_):
        raise OSError("disk full")

    monkeypatch.setattr(here, "_weigh", fails)
    with pytest.raises(OSError, match="disk full"):
        await here.pull_org("origin")

    # Nothing waits to be resolved, so nothing holds the org.
    assert not here.summary().merging and here.summary().paused is None
    await here.aclose()
    await there.aclose()


@pytest.mark.parametrize(("side", "headline"), [("mine", "Mine"), ("theirs", "Theirs")])
async def test_a_conflict_is_resolved_either_way_and_the_merge_is_finished(
    repo: Path,
    launcher: FakeLauncher,
    remote: Path,
    tmp_path: Path,
    side: str,
    headline: str,
):
    here, there = await conflicting(repo, launcher, remote, tmp_path)
    await here.pull_org("origin")

    done = await here.resolve_merge(
        ResolveMergeRequest.model_validate({"choices": {f"field:{API}:headline": side}})
    )
    await here.settled()

    assert done.result == "merged" and done.merge is None
    assert team(here, API).headline == headline
    assert team(here, WEB).headline == "Web moved on"
    summary = here.summary()
    assert not summary.merging and summary.paused is None
    assert (await here.push_org("origin")).result == "pushed"
    await here.aclose()
    await there.aclose()


async def test_a_merge_that_waits_survives_a_restart_and_can_be_abandoned(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await conflicting(repo, launcher, remote, tmp_path)
    await here.pull_org("origin")
    await here.aclose()
    launcher.listeners.clear()
    again = DirectorService(
        repo,
        save_delay=0,
        run_limit_retry=0,
        remembered=repo.parent / "state" / "director.json",
    )
    await again.start(launcher)
    await again.settled()

    waiting = await again.merge()
    with pytest.raises(Unprocessable, match="no conflict named"):
        await again.resolve_merge(
            ResolveMergeRequest.model_validate({"choices": {"field:x:y": "mine"}})
        )
    abandoned = await again.abort_merge()
    await again.settled()

    assert waiting is not None and len(waiting.conflicts) == 1
    assert abandoned.result == "up_to_date" and await again.merge() is None
    assert team(again, API).headline == "Mine"
    assert not again.summary().merging and again.summary().paused is None
    taken = await again.pull_org("origin")
    everything = await again.resolve_merge(
        ResolveMergeRequest.model_validate({"rest": "theirs"})
    )
    assert taken.result == "conflicts" and everything.result == "merged"
    assert team(again, API).headline == "Theirs"
    await again.aclose()
    await there.aclose()


async def test_a_spec_both_sides_rewrote_is_a_conflict_about_the_file(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    here.team_memory.write("services/api", SPEC_FILE, "Serve orders over REST\n")
    there.team_memory.write("services/api", SPEC_FILE, "Serve orders over gRPC\n")
    there.team_memory.write("apps/web", SPEC_FILE, "Pages for checkout\n")
    await there.push_org("origin")

    pulled = await here.pull_org("origin")
    done = await here.resolve_merge(
        ResolveMergeRequest.model_validate(
            {"choices": {"file:services/api/spec.md": "theirs"}}
        )
    )

    assert pulled.merge is not None
    (conflict,) = pulled.merge.conflicts
    assert (conflict.kind, conflict.field) == ("file", "services/api/spec.md")
    assert conflict.mine == "Serve orders over REST\n"
    assert conflict.theirs == "Serve orders over gRPC\n"
    assert done.result == "merged"
    assert done.memory is not None and done.memory.kept_local == []
    assert here.team_memory.read("services/api/spec.md") == "Serve orders over gRPC\n"
    assert here.team_memory.read("apps/web/spec.md") == "Pages for checkout\n"
    await here.aclose()
    await there.aclose()


async def test_a_session_goes_on_with_a_team_the_pull_removed(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    there.org.remove(API)
    await there.push_org("origin")
    await here.pause()
    here.notify(ic(here, API), "Use idempotency keys")
    await here.settled()

    pulled = await here.pull_org("origin")

    assert pulled.result in ("merged", "fast_forward")
    assert here.org.team(API) is None
    # The session has the team as it set it up; it is not made to catch up
    # while it works.
    assert team_id_of(node(here, API)) == API and ic(here, API) is not None
    assert here.summary().paused == "human"
    assert (await here.recovery()).messages_held == 1
    await here.aclose()
    await there.aclose()


async def test_a_team_whose_conversation_is_gone_is_started_again_on_its_work(
    repo: Path, launcher: FakeLauncher
):
    here = await started(repo, launcher)
    await split_root(here)
    here.team_memory.write("services/api", SPEC_FILE, "Serve orders over REST\n")
    lost = ic(here, API)
    assert lost is not None
    del launcher.known[UUID(lost)]
    derived = len(launcher.derived)
    api = node(here, API).id

    started_again = await here.start_again(
        api, StartAgainRequest(conversation=root_conversation(repo))
    )
    await here.settled()

    assert ic(here, API) != lost
    assert launcher.root is not None
    assert launcher.root.workspace.working_dir == node(here, API).worktree_path
    assert str(launcher.root.conversation_id) == ic(here, API)
    (briefing,) = launcher.messages_to(ic(here, API))
    assert "Serve orders over REST" in briefing and "worked on before" in briefing
    assert len(launcher.derived) == derived
    assert {agent.status for agent in started_again.agents} != {AGENT_LOST}
    with pytest.raises(Conflict, match="has a conversation here"):
        await here.start_again(
            api, StartAgainRequest(conversation=root_conversation(repo))
        )
    await here.aclose()


async def test_such_a_team_is_started_again_like_the_ic_above_in_its_own_checkout(
    repo: Path, launcher: FakeLauncher
):
    here = await started(repo, launcher)
    await split_root(here)
    api, branch = node(here, API).id, node(here, API).branch
    top_ic, lost = ic(here), ic(here, API)
    assert top_ic is not None and lost is not None
    del launcher.known[UUID(lost)]

    # Nobody works at a team of the org itself.
    with pytest.raises(Conflict, match="open a session there"):
        await here.start_again("root", StartAgainRequest())
    # With no conversation of the org left here there is nothing to start
    # it like.
    kept = launcher.known.pop(UUID(top_ic))
    with pytest.raises(Conflict, match="Start it again with a conversation"):
        await here.start_again(api, StartAgainRequest())
    launcher.known[UUID(top_ic)] = kept

    before = len(launcher.derived)
    await here.start_again(api, StartAgainRequest())
    await here.settled()

    [again] = launcher.derived[before:]
    assert str(again.parent_id) == top_ic
    assert str(again.conversation_id) == ic(here, API) != lost
    assert again.checkout == node(here, API).worktree_path
    assert "worked on before" in again.task
    assert {agent.status for agent in node(here, API).agents} != {AGENT_LOST}
    assert node(here, API).branch == branch
    await here.aclose()


async def test_a_message_with_a_conversation_opens_a_session_in_a_pulled_org(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    clone = there.repo_dir
    fake = there._require_launcher()
    assert isinstance(fake, FakeLauncher)

    routed = await there.route_message(
        RouteMessageRequest(
            text="Add a health check",
            path="services/api",
            conversation=root_conversation(clone),
        )
    )
    await there.settled()

    # The team nearest to where it was said, in a session of this machine.
    [session] = there.org.sessions()
    assert (session.team_id, session.request) == (API, "Add a health check")
    assert routed.node_id == session.teams[0].id
    assert fake.root is not None
    assert routed.conversation_id == str(fake.root.conversation_id)
    assert routed.conversation_id == ic(there, routed.node_id)
    assert "Add a health check" in fake.messages_to(routed.conversation_id)[-1]
    await here.aclose()
    await there.aclose()


async def test_a_pull_shows_the_tree_again_to_a_page_that_has_it_open(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    change(here, API, title="Payments")
    await here.push_org("origin")
    events: list[OrgChange] = []
    there.org.subscribe("root", events.append, full=True)

    await there.pull_org("origin")

    shown = events[-1]
    assert isinstance(shown, SnapshotEvent)
    assert {node.id: node.title for node in shown.nodes}[API] == "Payments"
    await here.aclose()
    await there.aclose()


async def test_a_sessions_reviewer_that_is_lost_is_started_again(
    repo: Path, launcher: FakeLauncher
):
    here = await started(repo, launcher)
    await use(here, ic(here), SubmitReviewTool, DONE)
    await here.settled()
    lead, lost = division_lead_of(here), division_lead_ic(here)
    assert lost is not None
    del launcher.known[UUID(lost)]

    before = len(launcher.derived)
    await here.start_again(lead, StartAgainRequest())
    await here.settled()

    # Like the first time: from the IC the session works with.
    [again] = launcher.derived[before:]
    assert str(again.parent_id) == ic(here)
    assert str(again.conversation_id) == division_lead_ic(here) != lost
    await here.aclose()


async def test_a_team_started_again_keeps_its_manager_and_hears_the_human(
    repo: Path, launcher: FakeLauncher
):
    here = await started(repo, launcher)
    await split_root(here)
    api = node(here, API).id
    here.tell_overseer(api, "Look in on your team.")
    await here.settled()
    kept, lost = manager(here, API), ic(here, API)
    assert kept is not None and lost is not None
    del launcher.known[UUID(lost)]

    before = len(launcher.derived)
    await here.start_again(api, StartAgainRequest(), said="Use the smaller migration.")
    await here.settled()

    assert manager(here, API) == kept and ic(here, API) != lost
    [again] = launcher.derived[before:]
    assert "Use the smaller migration." in again.task
    await here.aclose()
