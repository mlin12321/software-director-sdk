"""An org pulled from and pushed to a remote of its repository, as a merge."""

from pathlib import Path

import pytest

from openhands.agent_server.director import org_sync
from openhands.agent_server.director.git_blackboard import SPEC_FILE
from openhands.agent_server.director.models import (
    OrgNode,
    OrgNodeInput,
    ResolveMergeRequest,
    RouteMessageRequest,
    StartAgainRequest,
)
from openhands.agent_server.director.org_merge import by_id, merge_orgs, rehomed
from openhands.agent_server.director.org_persistence import create_org_board
from openhands.agent_server.director.service import AGENT_LOST, DirectorService
from openhands.sdk.errors import Conflict, Unprocessable
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_node,
    root_conversation,
    run_git,
)
from tests.agent_server.director.test_director_limits import started
from tests.agent_server.director.test_director_tools import ic, node, split_root


API, WEB = "services.api", "apps.web"


def tree(*changes: tuple[str, str, object]) -> dict[str, dict]:
    """A root with teams ``a`` and ``b``, as saved, with fields changed."""
    nodes = by_id(
        [
            make_node("root", None, path="").model_dump(),
            make_node("a", "root").model_dump(),
            make_node("b", "root").model_dump(),
        ]
    )
    for node_id, field, value in changes:
        nodes[node_id] = {**nodes[node_id], field: value}
    return nodes


def without(nodes: dict[str, dict], *gone: str) -> dict[str, dict]:
    return {node_id: node for node_id, node in nodes.items() if node_id not in gone}


def test_changes_to_different_teams_and_fields_are_all_kept():
    mine = tree(("a", "headline", "A moved on"), ("b", "title", "Billing"))
    theirs = tree(("b", "headline", "B moved on"), ("root", "status", "done"))

    merged = merge_orgs(tree(), mine, theirs, {})

    assert merged.conflicts == []
    assert by_id(merged.nodes) == tree(
        ("a", "headline", "A moved on"),
        ("b", "title", "Billing"),
        ("b", "headline", "B moved on"),
        ("root", "status", "done"),
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
    below = make_node("a.x", "a").model_dump()
    base = {**tree(), "a.x": below}

    mine_removed = merge_orgs(base, without(base, "a", "a.x"), base, {})
    theirs_removed = merge_orgs(base, base, without(base, "a", "a.x"), {})

    assert mine_removed.conflicts == [] and theirs_removed.conflicts == []
    assert set(by_id(mine_removed.nodes)) == {"root", "b"}
    assert set(by_id(theirs_removed.nodes)) == {"root", "b"}


def test_a_team_removed_on_one_side_and_changed_below_on_the_other_is_a_conflict():
    below = make_node("a.x", "a").model_dump()
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
    mine = {**tree(), "c": make_node("c", "root").model_dump()}
    theirs = {**tree(), "d": make_node("d", "b").model_dump()}

    merged = merge_orgs(tree(), mine, theirs, {})

    assert merged.conflicts == []
    assert set(by_id(merged.nodes)) == {"root", "a", "b", "c", "d"}


def test_where_a_team_runs_is_never_taken_from_the_other_side():
    theirs = tree(("a", "worktree_path", "/elsewhere/a"), ("a", "agents", []))
    mine = tree(("a", "headline", "Mine"))

    merged = merge_orgs(tree(), mine, theirs, {})

    assert merged.conflicts == [] and by_id(merged.nodes) == mine


def test_teams_from_another_machine_are_pointed_at_this_repository():
    nodes = [
        {**make_node("root", None, path="").model_dump(), "worktree_path": "/there"},
        {**make_node("a", "root").model_dump(), "worktree_path": "/there/a"},
        {**make_node("b", "root").model_dump(), "worktree_path": "/tmp/trees/b"},
    ]

    here = by_id(rehomed(nodes, {}, "/there", "/here"))
    known = by_id(rehomed(nodes, by_id(nodes), "/there", "/here"))

    assert here["root"]["worktree_path"] == "/here"
    assert here["a"]["worktree_path"] == "/here/a"
    assert here["b"]["worktree_path"] == "/tmp/trees/b"
    assert known == by_id(nodes)


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


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
    here = await started(repo, launcher)
    await split_root(here)
    await here.push_org("origin")
    there, _ = await elsewhere(remote, tmp_path)
    await there.pull_org("origin")
    await there.settled()
    return here, there


def change(service: DirectorService, node_id: str, **fields: object) -> None:
    service.store.update_node(node_id, **fields)


async def test_a_pushed_org_is_whole_in_another_clone_except_for_its_agents(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here = await started(repo, launcher)
    await split_root(here)
    here.blackboard.teach("", "Never log card numbers")

    pushed = await here.push_org("origin")
    there, clone = await elsewhere(remote, tmp_path)
    pulled = await there.pull_org("origin")
    await there.settled()

    assert pushed.result == "pushed" and pulled.result == "fast_forward"
    assert [team.id for team in there.store.all()] == ["root", API, WEB]
    assert node(there, API).title == "API" and node(there, API).spec_summary
    assert there.blackboard.memory("").rules == ["Never log card numbers"]
    assert there.blackboard.applies_to("services/api", "")[1]
    assert node(there, "root").worktree_path == str(clone)
    assert {agent.status for team in there.store.all() for agent in team.agents} == {
        AGENT_LOST
    }
    assert there.summary().paused is None and here.summary().paused is None
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
        assert node(side, API).headline == "API has its routes"
        assert node(side, API).title == "Orders API"
        assert node(side, WEB).headline == "Web has its pages"
    assert back.result in ("merged", "fast_forward")
    assert ic(here, API) == mine and node(here, "root").worktree_path == str(repo)
    assert merged.head is not None
    clone = Path(node(there, "root").worktree_path)
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
    assert node(here, API).headline == "Mine" and node(here, WEB).headline != (
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
    assert node(here, API).headline == headline
    assert node(here, WEB).headline == "Web moved on"
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
    assert node(again, API).headline == "Mine"
    assert not again.summary().merging and again.summary().paused is None
    taken = await again.pull_org("origin")
    everything = await again.resolve_merge(
        ResolveMergeRequest.model_validate({"rest": "theirs"})
    )
    assert taken.result == "conflicts" and everything.result == "merged"
    assert node(again, API).headline == "Theirs"
    await again.aclose()
    await there.aclose()


async def test_a_spec_both_sides_rewrote_is_a_conflict_about_the_file(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    here.blackboard.write("services/api", SPEC_FILE, "Serve orders over REST\n")
    there.blackboard.write("services/api", SPEC_FILE, "Serve orders over gRPC\n")
    there.blackboard.write("apps/web", SPEC_FILE, "Pages for checkout\n")
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
    assert done.blackboard is not None and done.blackboard.kept_local == []
    assert here.blackboard.read("services/api/spec.md") == "Serve orders over gRPC\n"
    assert here.blackboard.read("apps/web/spec.md") == "Pages for checkout\n"
    await here.aclose()
    await there.aclose()


async def test_unfinished_work_for_a_team_the_pull_removed_is_put_aside(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    there.store.remove(API)
    await there.push_org("origin")
    await here.pause()
    here.notify(ic(here, API), "Use idempotency keys")
    await here.settled()

    pulled = await here.pull_org("origin")

    assert pulled.result in ("merged", "fast_forward")
    assert here.store.get(API) is None
    (dropped,) = pulled.put_aside
    assert dropped.node_id == API and "idempotency" in dropped.detail
    assert here.summary().paused == "human"
    assert (await here.recovery()).messages_held == 0
    await here.aclose()
    await there.aclose()


async def test_a_team_without_agents_here_is_started_again_on_its_work(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    clone = Path(node(there, "root").worktree_path)
    there.blackboard.write("services/api", SPEC_FILE, "Serve orders over REST\n")
    fake = there._require_launcher()
    assert isinstance(fake, FakeLauncher)

    team = await there.start_again(
        API, StartAgainRequest(conversation=root_conversation(clone))
    )
    await there.settled()

    assert ic(there, API) != ic(here, API)
    assert fake.root is not None
    assert fake.root.workspace.working_dir == node(there, API).worktree_path
    assert str(fake.root.conversation_id) == ic(there, API)
    (briefing,) = fake.messages_to(ic(there, API))
    assert "Serve orders over REST" in briefing and "worked on before" in briefing
    assert [start.title for start in fake.derived] == ["API (Manager)"]
    assert {agent.status for agent in team.agents} != {AGENT_LOST}
    with pytest.raises(Conflict, match="has a conversation here"):
        await there.start_again(
            API, StartAgainRequest(conversation=root_conversation(clone))
        )
    await here.aclose()
    await there.aclose()


async def test_a_message_with_a_conversation_starts_the_nearest_lost_team_again(
    repo: Path, launcher: FakeLauncher, remote: Path, tmp_path: Path
):
    here, there = await an_org_on_two_machines(repo, launcher, remote, tmp_path)
    clone = Path(node(there, "root").worktree_path)
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

    assert routed.node_id == API and routed.conversation_id == ic(there, API)
    assert fake.messages_to(ic(there, API))[-1].startswith("Add a health check")
    assert node(there, "root").agents[0].status == AGENT_LOST
    await here.aclose()
    await there.aclose()


def team_with_branch(repo: Path, branch: str) -> OrgNodeInput:
    return make_node("a", "root").model_copy(
        update={"branch": branch, "worktree_path": str(repo / "nowhere")}
    )


def commit_on(repo: Path, branch: str, file: str) -> str:
    exists = branch in run_git(repo, "branch", "--list", branch)
    run_git(repo, "checkout", "--quiet", *([] if exists else ["-b"]), branch)
    (repo / file).write_text(f"{file}\n")
    run_git(repo, "add", file)
    run_git(repo, "commit", "--quiet", "-m", f"add {file}")
    head = run_git(repo, "rev-parse", "HEAD").strip()
    run_git(repo, "checkout", "--quiet", "main")
    return head


def test_a_teams_branch_travels_and_a_diverged_one_is_kept_beside(
    repo: Path, remote: Path, tmp_path: Path
):
    team = OrgNode(**team_with_branch(repo, "team-a").model_dump())
    first = commit_on(repo, "team-a", "one.txt")
    clone = tmp_path / "clone"
    run_git(tmp_path, "clone", "--quiet", str(remote), str(clone))
    run_git(clone, "config", "user.name", "Test")
    run_git(clone, "config", "user.email", "test@example.invalid")

    pushed = org_sync.push(repo, "origin", [team])
    org_sync.fetch(clone, "origin")
    created = org_sync.take_in_branches(clone, "origin", [team])
    second = commit_on(repo, "team-a", "two.txt")
    org_sync.push(repo, "origin", [team])
    org_sync.fetch(clone, "origin")
    forwarded = org_sync.take_in_branches(clone, "origin", [team])
    commit_on(clone, "team-a", "theirs.txt")
    third = commit_on(repo, "team-a", "three.txt")
    org_sync.push(repo, "origin", [team])
    org_sync.fetch(clone, "origin")
    diverged = org_sync.take_in_branches(clone, "origin", [team])
    refused = org_sync.push(clone, "origin", [team])

    assert [branch.result for branch in pushed] == ["pushed"]
    assert [branch.result for branch in created] == ["created"]
    assert [branch.result for branch in forwarded] == ["fast_forwarded"]
    assert first != second
    (kept,) = diverged
    assert (kept.result, kept.ref) == ("diverged", "refs/agents/teams/a")
    assert run_git(clone, "rev-parse", "refs/agents/teams/a").strip() == third
    assert "theirs.txt" in run_git(clone, "ls-tree", "--name-only", "team-a")
    assert [branch.result for branch in refused] == ["diverged"]
    assert run_git(remote, "for-each-ref", "refs/heads").count("team-a") == 0
