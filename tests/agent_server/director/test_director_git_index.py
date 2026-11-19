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
                readonly=(),
                writable=[f"{directory}/**" if directory else "**"],
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
    # As on a machine set up to print paths from where git is run; the
    # commit below is made two directories down.
    run_git(shared, "config", "diff.relative", "true")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    run_git(shared, "add", "svc/web/main.py")

    committed = api("commit", "-am", "api work", cwd=shared / "svc" / "api")

    assert committed.returncode == 0
    assert "index was not updated" not in committed.stderr, committed.stderr
    # What the human had staged is still staged; the commit is not shown
    # as something to undo.
    assert run_git(shared, "status", "--porcelain").split() == [
        "M",
        "svc/web/main.py",
    ]


def test_a_committed_name_that_reads_as_a_pattern_leaves_the_rest_staged(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")
    (shared / "svc" / "api" / "f2").write_text("staged by the human\n")
    run_git(shared, "add", "svc/api/f2")
    (shared / "svc" / "api" / "f?").write_text("x\n")

    assert api("--literal-pathspecs", "add", "svc/api/f?").returncode == 0
    assert api("commit", "-qm", "odd name").returncode == 0
    assert (
        run_git(shared, "show", "--name-only", "--format=", "HEAD").strip()
        == "svc/api/f?"
    )

    assert "A  svc/api/f2" in run_git(shared, "status", "--porcelain").splitlines()


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


def test_a_directory_that_comes_back_keeps_what_was_committed_in_it_meanwhile(
    shared: Path, agent: Callable[..., Git]
):
    assert agent("root", "", "svc/web")("status").returncode == 0
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    assert agent("web", "svc/web")("commit", "-qam", "web work").returncode == 0

    # The sub-team is gone, and its folder is the team's again.
    root = agent("root", "")
    (shared / "README.md").write_text("more\n")
    assert root("add", "README.md").returncode == 0
    assert root("commit", "-qm", "my own change").returncode == 0

    assert files_of(shared, "HEAD") == ["README.md"]
    assert run_git(shared, "show", "HEAD:svc/web/main.py") == "web v2\n"


def test_a_reset_that_would_take_away_another_teams_commit_is_not_done(
    shared: Path, agent: Callable[..., Git]
):
    api, web = agent("api", "svc/api"), agent("web", "svc/web")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    assert api("commit", "-qam", "api work").returncode == 0
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    assert web("commit", "-qam", "web work").returncode == 0
    head = run_git(shared, "rev-parse", "HEAD")
    # What the other team has not committed yet, and what this one staged.
    (shared / "svc" / "web" / "main.py").write_text("web v3, not committed\n")
    (shared / "svc" / "api" / "new.py").write_text("x\n")
    assert api("add", "svc/api/new.py").returncode == 0

    for how in ("--soft", "--mixed", "--hard"):
        refused = api("reset", how, "HEAD~2")

        assert refused.returncode != 0 and "other teams" in refused.stderr
        assert run_git(shared, "rev-parse", "HEAD") == head
        assert (shared / "svc" / "web" / "main.py").read_text() == (
            "web v3, not committed\n"
        )
        assert api("status", "--porcelain").stdout == "A  svc/api/new.py\n"
    # Naming files moves no branch, and its own last commit is its own.
    assert api("reset", "-q", "HEAD~2", "--", "svc/api/new.py").returncode == 0
    assert api("add", "svc/api/new.py").returncode == 0
    assert api("commit", "-qm", "api again").returncode == 0
    assert api("reset", "--soft", "HEAD~1").returncode == 0


def test_a_team_that_hands_a_folder_on_keeps_what_it_staged_in_its_own(
    shared: Path, agent: Callable[..., Git]
):
    root = agent("root", "")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    (shared / "top.txt").write_text("x\n")
    assert root("add", "svc/api/main.py", "top.txt").returncode == 0

    # A sub-team now owns web.
    root = agent("root", "", "svc/web")

    assert root("status", "--porcelain").stdout == "M  svc/api/main.py\nA  top.txt\n"


def test_a_merge_that_brings_in_other_teams_changes_is_undone(
    shared: Path, agent: Callable[..., Git]
):
    api, web = agent("api", "svc/api"), agent("web", "svc/web")
    run_git(shared, "checkout", "-q", "-b", "other")
    (shared / "svc" / "web" / "main.py").write_text("web elsewhere\n")
    run_git(shared, "commit", "-qam", "elsewhere")
    run_git(shared, "checkout", "-q", "-")
    head = run_git(shared, "rev-parse", "HEAD")

    merged = api("merge", "-q", "--no-edit", "other")

    assert merged.returncode != 0 and "outside your team" in merged.stderr
    assert run_git(shared, "rev-parse", "HEAD") == head
    assert (shared / "svc" / "web" / "main.py").read_text() == "web v1\n"
    assert web("status", "--porcelain").stdout == ""


def test_a_commit_of_what_the_team_may_only_read_is_undone(
    shared: Path, tmp_path: Path
):
    real = shutil.which("git")
    assert real is not None
    home = tmp_path / "held"
    home.mkdir()
    excludes = home / "exclude"
    excludes.write_text(excludes_file(shared, "svc/api", ()))
    common = run_git(
        shared, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    write_shell(
        home / "git",
        git_script(
            real,
            common_dir=common,
            conversation_id="held",
            directory="svc/api",
            sub_team_directories=(),
            excludes=excludes,
            flock=shutil.which("flock"),
            readonly=["svc/api/main.py"],
            writable=["svc/api/**"],
        ),
    )
    head = run_git(shared, "rev-parse", "HEAD")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")

    committed = subprocess.run(
        [str(home / "git"), "commit", "-qam", "contract"],
        cwd=shared,
        capture_output=True,
        text=True,
    )

    assert committed.returncode != 0 and "outside your team" in committed.stderr
    assert run_git(shared, "rev-parse", "HEAD") == head


def test_a_team_that_may_write_nothing_commits_nothing(shared: Path, tmp_path: Path):
    real = shutil.which("git")
    assert real is not None
    home = tmp_path / "reader"
    home.mkdir()
    excludes = home / "exclude"
    excludes.write_text(excludes_file(shared, "svc/api", ()))
    common = run_git(
        shared, "rev-parse", "--path-format=absolute", "--git-common-dir"
    ).strip()
    write_shell(
        home / "git",
        git_script(
            real,
            common_dir=common,
            conversation_id="reader",
            directory="svc/api",
            sub_team_directories=(),
            excludes=excludes,
            flock=shutil.which("flock"),
            readonly=(),
            writable=(),
        ),
    )
    head = run_git(shared, "rev-parse", "HEAD")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")

    committed = subprocess.run(
        [str(home / "git"), "commit", "-qam", "change"],
        cwd=shared,
        capture_output=True,
        text=True,
    )

    assert committed.returncode != 0 and "outside your team" in committed.stderr
    assert run_git(shared, "rev-parse", "HEAD") == head


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


def test_a_checkout_named_with_dash_c_is_used_like_any_other(
    shared: Path, agent: Callable[..., Git], tmp_path: Path
):
    api = agent("api", "svc/api")
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")

    api("-C", str(shared), "add", "-A", cwd=tmp_path)
    committed = api("-C", str(shared), "commit", "-m", "api", cwd=tmp_path)

    assert committed.returncode == 0, committed.stderr
    assert files_of(shared, "HEAD") == ["svc/api/main.py"]


def test_a_commit_that_changes_another_teams_file_is_undone(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")
    head = run_git(shared, "rev-parse", "HEAD").strip()
    blob = run_git(shared, "hash-object", "-w", "svc/api/main.py").strip()

    api("update-index", "--cacheinfo", f"100644,{blob},svc/web/main.py")
    committed = api("commit", "-m", "theirs")

    assert committed.returncode != 0
    assert "outside your team's folder" in committed.stderr
    assert run_git(shared, "rev-parse", "HEAD").strip() == head
    # Its index holds the other team's file as it is committed again.
    assert api("diff", "--cached", "--quiet").returncode == 0


def test_no_commit_is_made_on_an_index_that_could_not_be_brought_up_to_date(
    shared: Path, agent: Callable[..., Git]
):
    api, web = agent("api", "svc/api"), agent("web", "svc/web")
    api("status")
    (shared / "svc" / "web" / "main.py").write_text("web v2\n")
    web("commit", "-am", "web")
    head = run_git(shared, "rev-parse", "HEAD").strip()
    lock = shared / ".git" / "agent-index-api.lock"
    lock.touch()
    (shared / "svc" / "api" / "main.py").write_text("api v2\n")

    refused = api("commit", "-am", "api")

    assert refused.returncode != 0
    assert run_git(shared, "rev-parse", "HEAD").strip() == head
    lock.unlink()
    assert api("commit", "-am", "api").returncode == 0
    assert files_of(shared, "HEAD") == ["svc/api/main.py"]
    assert run_git(shared, "show", "HEAD:svc/web/main.py") == "web v2\n"


@pytest.mark.parametrize(
    "command",
    [("checkout", "-b", "mine"), ("switch", "-c", "mine"), ("checkout", "other")],
    ids=["checkout -b", "switch -c", "checkout"],
)
def test_a_checkout_stays_on_the_branch_its_work_is_taken_from(
    shared: Path, agent: Callable[..., Git], command: tuple[str, ...]
):
    api, web = agent("api", "svc/api"), agent("web", "svc/web")
    branch = run_git(shared, "symbolic-ref", "--short", "HEAD").strip()
    run_git(shared, "branch", "other")

    switched = api(*command)

    assert switched.returncode != 0 and f"stays on {branch}" in switched.stderr
    assert run_git(shared, "symbolic-ref", "--short", "HEAD").strip() == branch
    # A branch it made on the way is gone; one that was there stays.
    assert run_git(shared, "branch", "--format=%(refname:short)").split() == sorted(
        [branch, "other"]
    )
    # Both teams' commits are then on the one branch that is accepted.
    for git, folder in ((api, "api"), (web, "web")):
        (shared / "svc" / folder / "main.py").write_text(f"{folder} v2\n")
        git("add", "-A")
        assert git("commit", "-m", folder).returncode == 0
    assert files_of(shared, "HEAD~1") + files_of(shared, "HEAD") == [
        "svc/api/main.py",
        "svc/web/main.py",
    ]


def test_a_file_is_still_restored_with_checkout(
    shared: Path, agent: Callable[..., Git]
):
    api = agent("api", "svc/api")
    kept = (shared / "svc" / "api" / "main.py").read_text()
    (shared / "svc" / "api" / "main.py").write_text("scratch\n")

    restored = api("checkout", "--", "svc/api/main.py")

    assert restored.returncode == 0, restored.stderr
    assert (shared / "svc" / "api" / "main.py").read_text() == kept
