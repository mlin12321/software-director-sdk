"""What a team changed, and changing one of its files by hand."""

from pathlib import Path

import pytest

from openhands.agent_server.director.changes import (
    file_sides,
    team_changes,
    write_by_hand,
)
from openhands.agent_server.director.models import Boundary, OrgNode
from openhands.sdk.errors import Forbidden, Invalid, NotFound
from tests.agent_server.director.conftest import make_node, run_git


def team(repo: Path, path: str, *, readonly: list[str] | None = None) -> OrgNode:
    """A team that owns ``path`` of ``repo`` and started at its HEAD."""
    given = make_node(path or "root", None, path=path).model_dump()
    given.update(
        boundary=Boundary(
            path=path,
            writable=[f"{path}/**" if path else "**"],
            readonly=readonly or [],
        ),
        worktree_path=str(repo),
        base_commit=run_git(repo, "rev-parse", "HEAD").strip(),
    )
    return OrgNode(**given)


def commit(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    run_git(repo, "add", path)
    run_git(repo, "commit", "--quiet", "-m", f"change {path}")


def test_lists_what_a_team_committed_left_uncommitted_and_added(repo: Path):
    commit(repo, "api/old.py", "a\nb\n")
    commit(repo, "api/gone.py", "x\n")
    api = team(repo, "api")
    commit(repo, "api/old.py", "a\nc\nd\n")
    commit(repo, "web/page.tsx", "not the team's\n")
    run_git(repo, "rm", "--quiet", "api/gone.py")
    run_git(repo, "commit", "--quiet", "-m", "remove")
    (repo / "api" / "new.py").write_text("one\ntwo\n")

    changes = team_changes(api)

    assert changes.base == api.base_commit and not changes.truncated
    assert [
        (file.path, file.kind, file.added, file.removed, file.committed)
        for file in changes.files
    ] == [
        ("api/gone.py", "deleted", 0, 1, True),
        ("api/new.py", "added", 2, 0, False),
        ("api/old.py", "modified", 2, 1, True),
    ]


def test_a_committed_file_changed_again_is_marked_uncommitted(repo: Path):
    api = team(repo, "api")
    commit(repo, "api/a.py", "1\n")
    (repo / "api" / "a.py").write_text("1\n2\n")

    [file] = team_changes(api).files

    assert (file.kind, file.added, file.committed) == ("added", 2, False)


def test_a_team_without_a_base_or_a_checkout_has_no_list(repo: Path, tmp_path: Path):
    unstarted = team(repo, "api").model_copy(update={"base_commit": ""})
    elsewhere = team(repo, "api").model_copy(
        update={"worktree_path": str(tmp_path / "missing")}
    )

    assert team_changes(unstarted).files == []
    with pytest.raises(NotFound):
        team_changes(elsewhere)


def test_shows_a_file_before_and_now(repo: Path):
    commit(repo, "api/a.py", "old\n")
    api = team(repo, "api")
    (repo / "api" / "a.py").write_text("new\n")
    (repo / "api" / "b.py").write_text("added\n")
    (repo / "api" / "blob.bin").write_bytes(b"\0\1")

    changed = file_sides(api, "api/a.py")
    added = file_sides(api, "api/b.py")
    binary = file_sides(api, "api/blob.bin")

    assert (changed.before, changed.after) == ("old\n", "new\n")
    assert (added.before, added.after) == (None, "added\n")
    assert binary.unreadable and binary.after is None
    with pytest.raises(Forbidden):
        file_sides(api, "README.md")
    with pytest.raises(Invalid):
        file_sides(api, "api/../README.md")


def test_a_change_by_hand_is_committed_alone(repo: Path):
    commit(repo, "api/a.py", "old\n")
    api = team(repo, "api")
    (repo / "api" / "wip.py").write_text("the agent's, not committed\n")
    run_git(repo, "add", "api/wip.py")

    edit = write_by_hand(api, [], "api/a.py", "by hand\n")

    assert edit.commit == run_git(repo, "rev-parse", "HEAD").strip()
    assert run_git(repo, "show", "--name-only", "--format=%s", "HEAD").split() == [
        "Edited",
        "by",
        "hand:",
        "api/a.py",
        "api/a.py",
    ]
    assert (repo / "api" / "a.py").read_text() == "by hand\n"
    assert run_git(repo, "status", "--porcelain").strip() == "A  api/wip.py"


def test_a_change_by_hand_can_add_a_file_and_does_nothing_when_nothing_changed(
    repo: Path,
):
    api = team(repo, "api")

    added = write_by_hand(api, [], "api/deep/new.py", "x\n")
    same = write_by_hand(api, [], "api/deep/new.py", "x\n")

    assert added.commit is not None and same.commit is None
    assert run_git(repo, "rev-parse", "HEAD").strip() == added.commit


def test_a_change_by_hand_is_held_to_what_the_team_may_write(
    repo: Path, tmp_path: Path
):
    (repo / ".gitignore").write_text("*.log\n")
    commit(repo, "api/sub/x.py", "x\n")
    commit(repo, "api/locked/y.py", "y\n")
    api = team(repo, "api", readonly=["api/locked/**"])
    sub = team(repo, "api/sub")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "api" / "link").symlink_to(outside)
    head = run_git(repo, "rev-parse", "HEAD")

    for path in ("README.md", "api/sub/x.py", "api/locked/y.py", "api/link/z.py"):
        with pytest.raises(Forbidden):
            write_by_hand(api, [sub], path, "no\n")
    for path in ("/etc/x", "api/../README.md", "", "api/sub", "api/run.log"):
        with pytest.raises(Invalid):
            write_by_hand(api, [], path, "no\n")
    with pytest.raises(Forbidden):
        write_by_hand(team(repo, ""), [], ".git/config", "no\n")

    assert run_git(repo, "rev-parse", "HEAD") == head
    assert list(outside.iterdir()) == []
    assert (repo / "api" / "sub" / "x.py").read_text() == "x\n"
