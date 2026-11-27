"""What a team changed, and changing one of its files by hand."""

from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git.changes import (
    accepted_changes,
    accepted_file,
    file_sides,
    team_changes,
    write_by_hand,
)
from openhands.agent_server.director.environment.git.cli import git, uncommitted
from openhands.agent_server.director.models import Boundary, SessionTeam
from openhands.sdk.errors import Forbidden, Invalid, NotFound
from tests.agent_server.director.conftest import commit, make_place, make_team, run_git


def team(repo: Path, path: str, *, readonly: list[str] | None = None) -> SessionTeam:
    """What a session has at the team that owns ``path`` of ``repo``, started
    at its HEAD."""
    boundary = Boundary(
        path=path, writable=[f"{path}/**" if path else "**"], readonly=readonly or []
    )
    place = make_place(
        make_team(path or "root", None, path=path), "s", None, checkout=str(repo)
    )
    return SessionTeam(
        **place.model_dump(exclude={"boundary", "base_commit"}),
        boundary=boundary,
        base_commit=run_git(repo, "rev-parse", "HEAD").strip(),
    )


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


def test_what_an_accept_took_is_read_from_its_two_commits(repo: Path):
    commit(repo, "api/old.py", "a\nb\n")
    base = run_git(repo, "rev-parse", "HEAD").strip()
    commit(repo, "api/old.py", "a\nc\n")
    commit(repo, "api/new.py", "one\n")
    commit(repo, "web/page.tsx", "not the team's\n")
    head = run_git(repo, "rev-parse", "HEAD").strip()
    # What came after the accept, committed or not, is no part of it.
    commit(repo, "api/later.py", "later\n")
    (repo / "api" / "old.py").write_text("by hand\n")

    changes = accepted_changes(repo, base, head, "api")
    everywhere = accepted_changes(repo, base, head, "")
    sides = accepted_file(repo, base, head, "api/old.py")
    added = accepted_file(repo, base, head, "api/new.py")

    assert (changes.base, changes.head, changes.truncated) == (base, head, False)
    assert [(f.path, f.kind, f.added, f.removed) for f in changes.files] == [
        ("api/new.py", "added", 1, 0),
        ("api/old.py", "modified", 1, 1),
    ]
    assert [f.path for f in everywhere.files][-1] == "web/page.tsx"
    assert (sides.before, sides.after) == ("a\nb\n", "a\nc\n")
    assert (added.before, added.after) == (None, "one\n")


def test_commits_that_are_no_commits_are_refused(repo: Path):
    head = run_git(repo, "rev-parse", "HEAD").strip()
    gone = "0123456789abcdef0123456789abcdef01234567"

    # Nothing but a commit id reaches git.
    for not_an_id in ("HEAD", "--output=x", "main..other", ""):
        with pytest.raises(Invalid, match="is not a commit id"):
            accepted_changes(repo, not_an_id, head, "")
        with pytest.raises(Invalid, match="is not a commit id"):
            accepted_file(repo, head, not_an_id, "README.md")
    with pytest.raises(Invalid):
        accepted_changes(repo, head, head, "../elsewhere")
    with pytest.raises(NotFound, match="cannot be read"):
        accepted_changes(repo, gone, head, "")
    with pytest.raises(NotFound, match="has no commit 0123456789ab"):
        accepted_file(repo, gone, head, "README.md")


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


def test_a_change_by_hand_reaches_the_index_of_an_agent_that_works_there(
    repo: Path,
):
    commit(repo, "api/a.py", "old\n")
    api = team(repo, "api")
    own = {"GIT_INDEX_FILE": str(repo / ".git" / "agent-index-ic")}
    git(repo, ["read-tree", "HEAD"], env=own)

    write_by_hand(api, [], "api/a.py", "by hand\n")

    # Or the agent's next commit would put the old file back.
    assert git(repo, ["diff", "--cached", "--name-only", "HEAD"], env=own) == ""


def test_a_change_by_hand_does_not_follow_a_link_into_another_teams_folder(
    repo: Path,
):
    commit(repo, "web/w.py", "theirs\n")
    (repo / "api").mkdir()
    (repo / "api" / "link").symlink_to("../web")

    with pytest.raises(Forbidden):
        write_by_hand(team(repo, "api"), [], "api/link/w.py", "mine\n")

    assert (repo / "web" / "w.py").read_text() == "theirs\n"


def test_uncommitted_files_are_named_as_they_are(repo: Path):
    (repo / "api").mkdir()
    (repo / "api" / "a b.py").write_text("x\n")
    (repo / "api" / "é.py").write_text("x\n")

    assert sorted(uncommitted(repo, "api")) == ["api/a b.py", "api/é.py"]
