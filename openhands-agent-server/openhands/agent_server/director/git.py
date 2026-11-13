"""A thin wrapper over the git CLI for the plumbing commands the director uses."""

import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path


_TIMEOUT_SECONDS = 30
# Who the commits the director makes itself are by.
DIRECTOR_IDENTITY = {
    "GIT_AUTHOR_NAME": "Agent Director",
    "GIT_AUTHOR_EMAIL": "director@agent-director.invalid",
    "GIT_COMMITTER_NAME": "Agent Director",
    "GIT_COMMITTER_EMAIL": "director@agent-director.invalid",
}


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
    timeout: float = _TIMEOUT_SECONDS,
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
        timeout=timeout,
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


def pathspec(path: str) -> str:
    """What names ``path`` of a checkout, from anywhere in it; '' is all of it."""
    return f":(top,literal){path}" if path else ":(top)"


def head_of(checkout: Path) -> str | None:
    """The commit a checkout is on; None before its first commit."""
    try:
        return git(checkout, ["rev-parse", "--verify", "--quiet", "HEAD"]).strip()
    except (GitError, OSError):
        return None


def uncommitted(checkout: Path, path: str) -> list[str]:
    """The files under ``path`` that differ from the last commit or are new,
    ignored files aside."""
    listed = git(checkout, ["status", "--porcelain", "-uall", "--", pathspec(path)])
    return [line[3:] for line in listed.splitlines() if line]


def save_and_discard(checkout: Path, path: str, ref: str) -> str | None:
    """Discard what is uncommitted under ``path`` of a checkout, keeping it
    as a commit at ``ref``.

    Ignored files are left alone. The checkout's own index is reset for
    ``path``; nothing else of it changes.

    Args:
        path: Relative to the top of the checkout; '' for all of it.

    Returns:
        The commit that holds what was discarded, or None when everything
        under ``path`` was committed already.
    """
    spec = [pathspec(path)]
    if not git(checkout, ["status", "--porcelain", "-uall", "--", *spec]):
        return None
    with tempfile.TemporaryDirectory(prefix="director-discard-") as scratch:
        env = {"GIT_INDEX_FILE": str(Path(scratch) / "index")}
        git(checkout, ["read-tree", "HEAD"], env=env)
        git(checkout, ["add", "--all", "--", *spec], env=env)
        tree = git(checkout, ["write-tree"], env=env).strip()
    commit = git(
        checkout,
        ["commit-tree", tree, "-p", "HEAD", "-m", f"Discarded from {path or '/'}"],
        env=DIRECTOR_IDENTITY,
    ).strip()
    git(checkout, ["update-ref", ref, commit])
    git(checkout, ["reset", "--quiet", "--", *spec])
    # `git restore` refuses a path that has no tracked file.
    if git(checkout, ["ls-files", "--", *spec]):
        git(checkout, ["restore", "--worktree", "--", *spec])
    git(checkout, ["clean", "--force", "-d", "--quiet", "--", *spec])
    return commit
