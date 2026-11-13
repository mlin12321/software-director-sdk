"""An agent's shell, confined to the directory its team owns."""

import subprocess
from pathlib import Path

import pytest

from openhands.agent_server.director.sandbox import (
    find_bwrap,
    shell_script,
    write_shell,
)
from tests.agent_server.director.conftest import run_git


SHELLS = ".git/director-shells"


def confined(
    repo: Path,
    directory: str | None,
    *,
    sub_teams: tuple[str, ...] = (),
    cwd: Path | None = None,
    checkouts: tuple[Path, ...] = (),
):
    """A function that runs a command in the confined shell of ``directory``."""
    bwrap = find_bwrap()
    if bwrap is None:
        pytest.skip("bubblewrap cannot make a sandbox on this machine")
    script = repo / SHELLS / f"{directory or 'none'}.sh".replace("/", "-")
    write_shell(
        script,
        shell_script(
            bwrap,
            "/bin/sh",
            checkouts=[repo, *checkouts],
            private=repo / SHELLS,
            directory=directory,
            sub_team_directories=sub_teams,
        ),
    )

    def run(command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), "-c", command],
            cwd=cwd or repo,
            capture_output=True,
            text=True,
            timeout=60,
        )

    return run


def test_the_shell_writes_its_teams_directory_and_nothing_else_in_the_checkout(
    repo: Path,
):
    (repo / "svc" / "web").mkdir(parents=True)
    run = confined(repo, "svc/api")

    assert run("pwd").stdout.strip() == str(repo / "svc" / "api")
    assert run("echo x > pay.py").returncode == 0
    assert (repo / "svc" / "api" / "pay.py").read_text() == "x\n"

    for target in ("../web/page.tsx", "../../README.md", "../../new.txt"):
        refused = run(f"echo x > {target}")
        assert refused.returncode != 0 and "Read-only file system" in refused.stderr
    assert (repo / "README.md").read_text() == "hello\n"
    assert not (repo / "svc" / "web" / "page.tsx").exists()


def test_the_shell_can_commit_but_not_change_its_own_confinement(repo: Path):
    run = confined(repo, "svc/api")

    committed = run("echo x > pay.py && git add pay.py && git commit -qm pay")

    assert committed.returncode == 0, committed.stderr
    assert "pay" in run_git(repo, "log", "--format=%s", "-1")
    assert run(f"echo x >> {repo / SHELLS / 'svc-api.sh'}").returncode != 0


def test_a_directory_handed_to_a_sub_team_is_closed_to_the_team_above(repo: Path):
    (repo / "svc" / "api").mkdir(parents=True)
    run = confined(repo, "", sub_teams=("svc/api",))

    assert run("echo x > top.txt && echo x > svc/other.txt").returncode == 0
    assert run("echo x > svc/api/pay.py").returncode != 0


def test_a_shell_without_a_directory_writes_nowhere_in_the_checkout(repo: Path):
    run = confined(repo, None)

    assert run("pwd").stdout.strip() == str(repo)
    assert run("echo x > note.txt").returncode != 0
    assert run("git commit -q --allow-empty -m no").returncode != 0


def test_files_outside_the_checkouts_stay_writable(repo: Path, tmp_path: Path):
    outside = tmp_path / "scratch.txt"

    assert confined(repo, "svc/api")(f"echo x > {outside}").returncode == 0
    assert outside.read_text() == "x\n"


def test_in_a_worktree_the_shell_writes_that_worktrees_directory(
    repo: Path, tmp_path: Path
):
    worktrees = tmp_path / "worktrees"
    worktree = worktrees / "web" / "repo"
    run_git(repo, "worktree", "add", "--quiet", "-b", "web", str(worktree))
    run = confined(repo, "apps/web", cwd=worktree, checkouts=(worktrees,))

    assert run("pwd").stdout.strip() == str(worktree / "apps" / "web")
    done = run("echo x > page.tsx && git add page.tsx && git commit -qm page")
    assert done.returncode == 0, done.stderr
    assert run(f"echo x > {worktree / 'README.md'}").returncode != 0
    assert run(f"echo x > {repo / 'README.md'}").returncode != 0


def test_an_unchanged_script_is_not_rewritten(repo: Path):
    script = repo / SHELLS / "a.sh"
    write_shell(script, "#!/bin/sh\n")
    written = script.stat().st_mtime_ns

    write_shell(script, "#!/bin/sh\n")

    assert script.stat().st_mtime_ns == written
    assert script.stat().st_mode & 0o777 == 0o700
