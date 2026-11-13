"""A thin wrapper over the git CLI for the plumbing commands the director uses."""

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path


_TIMEOUT_SECONDS = 30


class GitError(Exception):
    """A git command exited with a non-zero status."""

    def __init__(self, args: list[str], code: int, stderr: str, stdout: str) -> None:
        super().__init__(f"git {' '.join(args)} exited {code}: {stderr.strip()}")
        self.code = code
        self.stderr = stderr
        self.stdout = stdout


def git(
    repo_dir: Path,
    args: list[str],
    *,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> str:
    """Run one git command in ``repo_dir`` and return its stdout.

    Raises:
        GitError: The command exited with a non-zero status.
    """
    # Bytes in and out: text mode would rewrite line endings in file content.
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        env={**os.environ, **(env or {})},
        input=None if stdin is None else stdin.encode(),
        # Without input, do not let git read whatever stdin the server has.
        stdin=subprocess.DEVNULL if stdin is None else None,
        capture_output=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )
    stdout = result.stdout.decode(errors="replace")
    if result.returncode != 0:
        raise GitError(
            args, result.returncode, result.stderr.decode(errors="replace"), stdout
        )
    return stdout


def git_version(repo_dir: Path) -> tuple[int, int]:
    """Return git's ``(major, minor)`` version."""
    match = re.search(r"(\d+)\.(\d+)", git(repo_dir, ["--version"]))
    if match is None:
        raise RuntimeError("Could not parse `git --version` output")
    return int(match.group(1)), int(match.group(2))


def is_git_repository(repo_dir: Path) -> bool:
    """Whether ``repo_dir`` is inside a git repository."""
    if not repo_dir.is_dir():
        return False
    try:
        git(repo_dir, ["rev-parse", "--git-dir"])
    except (GitError, OSError):
        return False
    return True


def enclosing_repository(repo_dir: Path) -> Path | None:
    """The checkout that has the repository of ``repo_dir`` as a submodule.

    None when there is none, or when it cannot be read.
    """
    try:
        top = git(repo_dir, ["rev-parse", "--show-superproject-working-tree"]).strip()
    except (GitError, OSError):
        return None
    return Path(top) if top else None
