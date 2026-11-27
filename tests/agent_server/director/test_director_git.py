"""The director's own git operations on a checkout."""

import subprocess
from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git.cli import (
    DIRECTOR_IDENTITY,
    GitError,
    bring_over,
    git,
    save_and_discard,
)
from tests.agent_server.director.conftest import run_git


REF = "refs/agents/pruned/team/1"


def test_uncommitted_work_under_a_path_is_discarded_and_kept_in_a_commit(repo: Path):
    (repo / "svc").mkdir()
    (repo / "svc" / "kept.py").write_text("committed\n")
    (repo / "svc" / "gone.py").write_text("committed\n")
    (repo / ".gitignore").write_text("*.log\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "svc")
    (repo / "svc" / "kept.py").write_text("edited\n")
    (repo / "svc" / "gone.py").unlink()
    (repo / "svc" / "staged.py").write_text("staged\n")
    run_git(repo, "add", "svc/staged.py")
    (repo / "svc" / "new" / "deep").mkdir(parents=True)
    (repo / "svc" / "new" / "deep" / "untracked.py").write_text("untracked\n")
    (repo / "svc" / "build.log").write_text("ignored\n")
    (repo / "README.md").write_text("outside the path\n")

    commit = save_and_discard(repo, "svc", REF)

    assert commit == run_git(repo, "rev-parse", REF).strip()
    assert (repo / "svc" / "kept.py").read_text() == "committed\n"
    assert (repo / "svc" / "gone.py").read_text() == "committed\n"
    assert not (repo / "svc" / "staged.py").exists()
    assert not (repo / "svc" / "new").exists()
    assert (repo / "svc" / "build.log").read_text() == "ignored\n"
    assert (repo / "README.md").read_text() == "outside the path\n"
    assert run_git(repo, "status", "--porcelain").split() == ["M", "README.md"]
    kept = run_git(repo, "ls-tree", "-r", "--name-only", REF).split()
    assert kept == [
        ".gitignore",
        "README.md",
        "svc/kept.py",
        "svc/new/deep/untracked.py",
        "svc/staged.py",
    ]
    assert run_git(repo, "show", f"{REF}:svc/kept.py") == "edited\n"
    # What was uncommitted outside the path is not in the copy.
    assert run_git(repo, "show", f"{REF}:README.md") == "hello\n"


def test_a_path_with_only_untracked_files_is_cleared(repo: Path):
    (repo / "svc").mkdir()
    (repo / "svc" / "draft.py").write_text("draft\n")

    assert save_and_discard(repo, "svc", REF) is not None

    assert not (repo / "svc" / "draft.py").exists()


def test_nothing_is_saved_when_everything_is_committed(repo: Path):
    (repo / "README.md").write_text("outside the path\n")

    assert save_and_discard(repo, "svc", REF) is None

    assert run_git(repo, "for-each-ref", "refs/agents/pruned") == ""
    assert (repo / "README.md").read_text() == "outside the path\n"


def with_a_second_checkout(repo: Path, tmp_path: Path) -> Path:
    (repo / "svc").mkdir()
    (repo / "svc" / "kept.py").write_text("committed\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "svc")
    other = tmp_path / "other"
    run_git(repo, "worktree", "add", "--quiet", "-b", "team", str(other))
    return other


def test_uncommitted_work_is_brought_over_to_a_checkout_on_the_same_commit(
    repo: Path, tmp_path: Path
):
    other = with_a_second_checkout(repo, tmp_path)
    (repo / "svc" / "new.py").write_text("new\n")
    (repo / "svc" / "kept.py").write_text("changed\n")
    (repo / "README.md").write_text("not theirs\n")

    moved = bring_over(repo, other, "svc", REF)

    assert sorted(moved) == ["svc/kept.py", "svc/new.py"]
    assert (other / "svc" / "new.py").read_text() == "new\n"
    assert (other / "svc" / "kept.py").read_text() == "changed\n"
    assert run_git(repo, "status", "--porcelain").split() == ["M", "README.md"]
    assert sorted(run_git(other, "status", "--porcelain", "-uall").split()) == [
        "??",
        "M",
        "svc/kept.py",
        "svc/new.py",
    ]


def test_nothing_is_brought_over_to_a_checkout_on_another_commit(
    repo: Path, tmp_path: Path
):
    other = with_a_second_checkout(repo, tmp_path)
    (other / "svc" / "theirs.py").write_text("theirs\n")
    run_git(other, "add", "-A")
    run_git(other, "commit", "--quiet", "-m", "theirs")
    (repo / "svc" / "kept.py").write_text("changed\n")

    assert bring_over(repo, other, "svc", REF) == []
    assert (repo / "svc" / "kept.py").read_text() == "changed\n"


def test_changed_files_are_named_from_the_top_whatever_git_is_set_to_print(
    repo: Path,
):
    (repo / "svc" / "api").mkdir(parents=True)
    (repo / "svc" / "api" / "main.py").write_text("v1\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "api")
    run_git(repo, "config", "diff.relative", "true")

    named = git(repo / "svc" / "api", ["diff", "--name-only", "HEAD~1", "HEAD"])

    assert named.split() == ["svc/api/main.py"]


def test_a_git_that_does_not_finish_fails_like_any_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def never(*args: object, **kwargs: object):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(subprocess, "run", never)

    with pytest.raises(GitError, match="did not finish"):
        git(tmp_path, ["status"], timeout=1)


def test_what_was_kept_already_is_found_again(repo: Path):
    (repo / "svc").mkdir()
    (repo / "svc" / "draft.py").write_text("draft\n")
    kept = save_and_discard(repo, "svc", REF)

    # A server that ended before it could say so asks again.
    assert kept is not None and save_and_discard(repo, "svc", REF) == kept
    assert save_and_discard(repo, "svc", "refs/agents/pruned/team/2") is None


def test_git_is_not_sent_to_another_repository_by_a_planted_commondir(
    repo: Path, tmp_path: Path
):
    run_git(repo, "worktree", "add", "-q", "-b", "wt", str(tmp_path / "wt"))
    assert git(tmp_path / "wt", ["rev-parse", "HEAD"])  # A worktree's own is git's.
    elsewhere = tmp_path / "elsewhere.git"
    run_git(tmp_path, "init", "-q", "--bare", str(elsewhere))

    # What an agent can write from its sandbox.
    (repo / ".git" / "commondir").write_text(f"{elsewhere}\n")

    for where in (repo, repo / "sub", tmp_path / "wt"):
        where.mkdir(exist_ok=True)
        with pytest.raises(GitError, match="commondir"):
            git(where, ["status"])


def test_no_command_the_repositorys_config_names_is_run(repo: Path, tmp_path: Path):
    ran = tmp_path / "ran"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name in ("pre-commit", "fsmonitor"):
        script = hooks / name
        script.write_text(f"#!/bin/sh\ntouch {ran}\n")
        script.chmod(0o755)
    # What an agent can write from its sandbox.
    run_git(repo, "config", "core.hooksPath", str(hooks))
    run_git(repo, "config", "core.fsmonitor", str(hooks / "fsmonitor"))

    git(repo, ["status", "--porcelain"])
    git(repo, ["commit", "--allow-empty", "-m", "x"], env=DIRECTOR_IDENTITY)

    assert not ran.exists()
