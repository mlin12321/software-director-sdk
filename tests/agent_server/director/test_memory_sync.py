"""Sharing a team memory through a remote, between two clones."""

from pathlib import Path

import pytest

from openhands.agent_server.director import memory_sync
from openhands.agent_server.director.git import git_version
from openhands.agent_server.director.git_store import (
    MEMORY_REF,
    GitStore,
)
from openhands.agent_server.director.models import Report
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
def ours(repo: Path, remote: Path) -> GitStore:
    return GitStore(repo)


def note(store: GitStore, text: str) -> None:
    at = (
        "2026-11-02T00:00:00+00:00"
        if text == "Theirs."
        else "2026-11-01T00:00:00+00:00"
    )
    store.append_report("agent-1", Report(at=at, role="ic", text=text))


def noted(store: GitStore) -> list[str]:
    return [report.text for report in store.reports("agent-1")]


@pytest.fixture
def theirs(remote: Path, tmp_path: Path) -> GitStore:
    clone = tmp_path / "theirs"
    run_git(tmp_path, "clone", "--quiet", str(remote), str(clone))
    return GitStore(clone)


def test_what_one_clone_pushes_the_other_pulls(ours: GitStore, theirs: GitStore):
    note(ours, "Amounts are cents.")

    pushed = memory_sync.push(ours, "origin")
    pulled = memory_sync.pull(theirs, "origin")

    assert pushed.pushed and not pushed.changed
    assert pulled.changed and not pulled.pushed
    assert pulled.head == ours.head()
    assert noted(theirs) == ["Amounts are cents."]


def test_a_push_takes_in_what_the_remote_gained_first(ours: GitStore, theirs: GitStore):
    note(ours, "Ours.")
    ours.write("svc/a", "spec.md", "our spec")
    note(theirs, "Theirs.")
    theirs.write("svc/a", "spec.md", "their spec")
    memory_sync.push(theirs, "origin")

    pushed = memory_sync.push(ours, "origin")

    assert pushed.pushed and pushed.changed
    assert pushed.kept_local == ["svc/a/spec.md"]
    assert noted(ours) == ["Ours.", "Theirs."]
    assert ours.read("svc/a/spec.md") == "our spec"
    on_remote = run_git(theirs.repo_dir, "ls-remote", "origin", MEMORY_REF)
    assert on_remote.split()[0] == ours.head()


def test_both_clones_end_on_the_same_team_memory(ours: GitStore, theirs: GitStore):
    note(ours, "Ours.")
    note(theirs, "Theirs.")
    memory_sync.push(theirs, "origin")
    memory_sync.push(ours, "origin")

    pulled = memory_sync.pull(theirs, "origin")

    assert pulled.changed
    assert pulled.head == ours.head()
    assert not memory_sync.pull(theirs, "origin").changed


def test_pulling_from_a_remote_without_a_team_memory_changes_nothing(
    ours: GitStore,
):
    note(ours, "Rule.")
    head = ours.head()

    pulled = memory_sync.pull(ours, "origin")

    assert (pulled.head, pulled.changed) == (head, False)


def test_only_the_team_memory_reaches_the_remote(ours: GitStore, remote: Path):
    note(ours, "Rule.")
    GitStore(ours.repo_dir, ref="refs/agents/org").write("", "org.json", "{}")

    memory_sync.push(ours, "origin")

    refs = run_git(remote, "for-each-ref", "--format=%(refname)", "refs/agents")
    assert refs.split() == [MEMORY_REF]


def test_a_remote_that_does_not_exist_or_cannot_be_reached_is_reported(
    ours: GitStore, tmp_path: Path
):
    note(ours, "Rule.")
    run_git(ours.repo_dir, "remote", "add", "gone", str(tmp_path / "nowhere.git"))

    with pytest.raises(NotFound, match="upstream"):
        memory_sync.pull(ours, "upstream")
    with pytest.raises(UpstreamFailed):
        memory_sync.pull(ours, "gone")
    with pytest.raises(UpstreamFailed):
        memory_sync.push(ours, "gone")


def test_an_empty_team_memory_cannot_be_pushed(ours: GitStore):
    with pytest.raises(Conflict):
        memory_sync.push(ours, "origin")
