"""Sharing a blackboard through a remote, between two clones."""

from pathlib import Path

import pytest

from openhands.agent_server.director import blackboard_sync
from openhands.agent_server.director.git import git_version
from openhands.agent_server.director.git_blackboard import (
    BLACKBOARD_REF,
    GitBlackboard,
)
from openhands.sdk.errors import Conflict, NotFound, UpstreamFailed
from tests.agent_server.director.conftest import run_git


@pytest.fixture
def remote(repo: Path, tmp_path: Path) -> Path:
    if git_version(repo) < (2, 38):
        pytest.skip("needs git 2.38")
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "clone", "--quiet", "--bare", str(repo), str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    return bare


@pytest.fixture
def ours(repo: Path, remote: Path) -> GitBlackboard:
    return GitBlackboard(repo)


@pytest.fixture
def theirs(remote: Path, tmp_path: Path) -> GitBlackboard:
    clone = tmp_path / "theirs"
    run_git(tmp_path, "clone", "--quiet", str(remote), str(clone))
    return GitBlackboard(clone)


def test_what_one_clone_pushes_the_other_pulls(
    ours: GitBlackboard, theirs: GitBlackboard
):
    ours.teach("svc/a", "Amounts are cents.")

    pushed = blackboard_sync.push(ours, "origin")
    pulled = blackboard_sync.pull(theirs, "origin")

    assert pushed.pushed and not pushed.changed
    assert pulled.changed and not pulled.pushed
    assert pulled.head == ours.head()
    assert theirs.memory("svc/a").rules == ["Amounts are cents."]


def test_a_push_takes_in_what_the_remote_gained_first(
    ours: GitBlackboard, theirs: GitBlackboard
):
    ours.teach("svc/a", "Ours.")
    ours.write("svc/a", "spec.md", "our spec")
    theirs.teach("svc/a", "Theirs.")
    theirs.write("svc/a", "spec.md", "their spec")
    blackboard_sync.push(theirs, "origin")

    pushed = blackboard_sync.push(ours, "origin")

    assert pushed.pushed and pushed.changed
    assert pushed.kept_local == ["svc/a/spec.md"]
    assert ours.memory("svc/a").rules == ["Ours.", "Theirs."]
    assert ours.read("svc/a/spec.md") == "our spec"
    on_remote = run_git(theirs.repo_dir, "ls-remote", "origin", BLACKBOARD_REF)
    assert on_remote.split()[0] == ours.head()


def test_both_clones_end_on_the_same_blackboard(
    ours: GitBlackboard, theirs: GitBlackboard
):
    ours.teach("svc/a", "Ours.")
    theirs.teach("svc/a", "Theirs.")
    blackboard_sync.push(theirs, "origin")
    blackboard_sync.push(ours, "origin")

    pulled = blackboard_sync.pull(theirs, "origin")

    assert pulled.changed
    assert pulled.head == ours.head()
    assert not blackboard_sync.pull(theirs, "origin").changed


def test_pulling_from_a_remote_without_a_blackboard_changes_nothing(
    ours: GitBlackboard,
):
    ours.teach("", "Rule.")
    head = ours.head()

    pulled = blackboard_sync.pull(ours, "origin")

    assert (pulled.head, pulled.changed) == (head, False)


def test_only_the_blackboard_reaches_the_remote(ours: GitBlackboard, remote: Path):
    ours.teach("", "Rule.")
    GitBlackboard(ours.repo_dir, ref="refs/agents/org").write("", "org.json", "{}")

    blackboard_sync.push(ours, "origin")

    refs = run_git(remote, "for-each-ref", "--format=%(refname)", "refs/agents")
    assert refs.split() == [BLACKBOARD_REF]


def test_a_remote_that_does_not_exist_or_cannot_be_reached_is_reported(
    ours: GitBlackboard, tmp_path: Path
):
    ours.teach("", "Rule.")
    run_git(ours.repo_dir, "remote", "add", "gone", str(tmp_path / "nowhere.git"))

    with pytest.raises(NotFound, match="upstream"):
        blackboard_sync.pull(ours, "upstream")
    with pytest.raises(UpstreamFailed):
        blackboard_sync.pull(ours, "gone")
    with pytest.raises(UpstreamFailed):
        blackboard_sync.push(ours, "gone")


def test_an_empty_blackboard_cannot_be_pushed(ours: GitBlackboard):
    with pytest.raises(Conflict):
        blackboard_sync.push(ours, "origin")
