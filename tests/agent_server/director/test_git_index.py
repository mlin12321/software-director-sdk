"""An agent's own git index, exercised with the real git."""

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from openhands.agent_server.director.git_index import excludes_file, git_script
from openhands.agent_server.director.sandbox import write_shell
from tests.agent_server.director.conftest import run_git


pytestmark = pytest.mark.skipif(os.name == "nt", reason="the wrapper is a sh script")

type Git = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def shared(repo: Path) -> Path:
    """A checkout with a committed file in each of two teams' directories."""
    for name in ("api", "web"):
        (repo / "svc" / name).mkdir(parents=True)
        (repo / "svc" / name / "main.py").write_text(f"{name} v1\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "teams")
    return repo


@pytest.fixture
def agent(shared: Path, tmp_path: Path) -> Callable[..., Git]:
    """Makes the ``git`` of an agent whose team owns a directory."""
    real = shutil.which("git")
    assert real is not None
    common = run_git(
        shared, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()

    def make(name: str, directory: str, *sub_teams: str) -> Git:
        home = tmp_path / "agents" / name
        excludes = home / "exclude"
        home.mkdir(parents=True, exist_ok=True)
        excludes.write_text(excludes_file(shared, directory, sub_teams))
        write_shell(
            home / "git",
            git_script(
                real,
                common_dir=common,
                conversation_id=name,
                directory=directory,
                sub_team_directories=sub_teams,
                excludes=excludes,
                flock=shutil.which("flock"),
            ),
        )
        env = {
            **os.environ,
            "PATH": f"{home}{os.pathsep}{os.environ['PATH']}",
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": f"{name}@example.invalid",
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": f"{name}@example.invalid",
        }

        def run(*args: str, cwd: Path = shared) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *args], cwd=cwd, env=env, capture_output=True, text=True
            )

        return run

    return make


def files_of(repo: Path, commit: str) -> list[str]:
    return run_git(repo, "show", "--name-only", "--format=", commit).split()


def test_a_commit_of_everything_takes_only_the_teams_own_files(
    shared: Path, agent: Callable[..., Git]
):
    api, web = agent("api", "svc/api"), agent("web", "svc/web")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    (shared / "svc" / "api" / "new.py").write_text("api new\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    (shared / "svc" / "web" / "new.py").write_text("web new\n")

    assert web("add", "-A").returncode == 0
    assert api("add", "-A").returncode == 0
    committed = api("commit", "-m", "api work")

    assert committed.returncode == 0, committed.stderr
    assert files_of(shared, "HEAD") == ["svc/api/main.py", "svc/api/new.py"]
    # The other team's work is still there, uncommitted and staged for it.
    assert (shared / "svc" / "web" / "main.py").read_text() == "web v2\n"
    assert web("commit", "-m", "web work").returncode == 0
    assert files_of(shared, "HEAD") == ["svc/web/main.py", "svc/web/new.py"]
    # The second commit kept what the first one changed.
    assert run_git(shared, "show", "HEAD:svc/api/main.py") == "api v2\n"


def test_commit_all_does_not_take_another_teams_edits(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")

    assert api("commit", "-am", "api work").returncode == 0

    assert files_of(shared, "HEAD") == ["svc/api/main.py"]
    assert (shared / "svc" / "web" / "main.py").read_text() == "web v2\n"


def test_an_agent_sees_only_its_own_teams_changes(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")
    (shared / "svc" / "api" / "draft.py").write_text("mine\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    (shared / "svc" / "web" / "draft.py").write_text("theirs\n")
    (shared / "notes.txt").write_text("the team above\n")

    status = api("status", "--porcelain", "-uall")

    assert status.stdout.split() == ["??", "svc/api/draft.py"]


def test_the_checkouts_own_index_follows_an_agents_commit(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    run_git(shared, "add", "svc/web/main.py")

    assert api("commit", "-am", "api work", cwd=shared / "svc" / "api").returncode == 0

    # What the human had staged is still staged; the commit is not shown
    # as something to undo.
    assert run_git(shared, "status", "--porcelain").split() == [
        "M",
        "svc/web/main.py",
    ]


def test_a_team_does_not_stage_what_it_handed_to_a_sub_team(
    shared: Path, agent: Callable[..., Git]
):
    root = agent("root", "", "svc/web")
    (shared / "README.md").write_text("root v2\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    (shared / "svc" / "web" / "new.py").write_text("web new\n")

    assert root("add", "-A").returncode == 0
    assert root("commit", "-m", "root work").returncode == 0

    assert files_of(shared, "HEAD") == ["README.md"]


def test_a_directory_that_comes_back_to_the_team_is_staged_again(
    shared: Path, agent: Callable[..., Git]
):
    agent("root", "", "svc/web")("status")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")

    root = agent("root", "")

    assert root("commit", "-am", "took web back").returncode == 0
    assert files_of(shared, "HEAD") == ["svc/web/main.py"]


def test_git_in_another_repository_is_the_real_git(
    shared: Path, agent: Callable[..., Git], tmp_path: Path
):
    api = agent("api", "svc/api")
    other = tmp_path / "other"
    other.mkdir()

    assert api("init", "--quiet", "--initial-branch=main", cwd=other).returncode == 0
    (other / "file.txt").write_text("x\n")
    assert api("add", "-A", cwd=other).returncode == 0
    assert api("commit", "-m", "first", cwd=other).returncode == 0

    assert files_of(other, "HEAD") == ["file.txt"]
    assert not list((other / ".git").glob("agent-index-*"))
    assert run_git(other, "status", "--porcelain") == ""


def test_an_agent_in_a_worktree_of_the_org_keeps_its_index_there(
    shared: Path, agent: Callable[..., Git], tmp_path: Path
):
    checkout = tmp_path / "worktree"
    run_git(shared, "worktree", "add", "--quiet", "-b", "team", str(checkout))
    web = agent("web", "svc/web")
    (checkout / "svc" / "web" / "main.py").write_text("web v2\n")
    (checkout / "svc" / "api" / "main.py").write_text("not mine\n")

    assert web("commit", "-am", "web work", cwd=checkout).returncode == 0

    assert files_of(shared, "team") == ["svc/web/main.py"]
    assert run_git(shared, "rev-parse", "main") != run_git(shared, "rev-parse", "team")


def test_the_exit_code_and_output_of_git_come_through(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")

    missing = api("show", "no-such-ref")
    log = api("log", "--format=%s", "-1")

    assert missing.returncode != 0 and "no-such-ref" in missing.stderr
    assert log.stdout == "teams\n"
