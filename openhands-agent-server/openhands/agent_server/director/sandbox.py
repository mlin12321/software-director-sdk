"""Confines an agent's shell to the directory its team owns.

The shell runs under bubblewrap in a mount namespace where the org's
checkouts are read-only except for the team's own directory. Everything
outside the checkouts is as writable as it is for the server.
"""

import shlex
import shutil
import subprocess
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from openhands.sdk.utils.files import atomic_write_text


_PROBE_TIMEOUT_SECONDS = 10


@cache
def find_bwrap() -> str | None:
    """The bubblewrap binary, when this machine lets it make a sandbox."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return None
    try:
        probe = subprocess.run(
            [bwrap, "--dev-bind", "/", "/", "true"],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bwrap if probe.returncode == 0 else None


def shell_script(
    bwrap: str | None,
    shell: str,
    *,
    checkouts: Sequence[Path],
    private: Path,
    directory: str | None,
    sub_team_directories: Sequence[str] = (),
    rcfile: Path | None = None,
) -> str:
    """A script that runs ``shell`` confined to one team's directory.

    It is started in the agent's workspace and finds the checkout from there,
    so it holds for a workspace that is a git worktree made after the script
    was written.

    Args:
        bwrap: The bubblewrap binary; None for a shell that is not confined.
        rcfile: What bash reads at its start in place of ``~/.bashrc``.
        checkouts: Where the org's checkouts are; nothing in them is writable
            but the team's directory.
        private: A directory of the director inside the git directory, kept
            read-only although the git directory is writable.
        directory: The repo-relative directory the shell may write, with ''
            for the whole checkout; None for a shell that may write none of
            it. It is created if missing, and the shell starts in it.
        sub_team_directories: Repo-relative directories inside ``directory``
            that sub-teams own and this shell may not write.
    """
    quote = shlex.quote
    start = f"{quote(shell)} --rcfile {quote(str(rcfile))}" if rcfile else quote(shell)
    return _confined(
        bwrap,
        start,
        checkouts=checkouts,
        private=private,
        directory=directory,
        sub_team_directories=sub_team_directories,
    )


def command_script(
    bwrap: str | None,
    *,
    checkouts: Sequence[Path],
    private: Path,
    directory: str | None,
    sub_team_directories: Sequence[str] = (),
    first_on_path: Path | None = None,
) -> str:
    """A script that runs the command it is given confined like a shell of
    ``shell_script``, for an agent whose whole process is to be held.

    Args:
        first_on_path: A directory of programs the command finds before any
            other of the same name.
    """
    start = (
        f'env PATH={shlex.quote(str(first_on_path))}:"$PATH"' if first_on_path else ""
    )
    return _confined(
        bwrap,
        start,
        checkouts=checkouts,
        private=private,
        directory=directory,
        sub_team_directories=sub_team_directories,
        stay=True,
        die_with_parent=True,
    )


def _confined(
    bwrap: str | None,
    start: str,
    *,
    checkouts: Sequence[Path],
    private: Path,
    directory: str | None,
    sub_team_directories: Sequence[str],
    stay: bool = False,
    die_with_parent: bool = False,
) -> str:
    """``start`` followed by the script's own arguments, under ``bwrap``.

    Args:
        stay: Keep the directory the script was started in, instead of
            moving to the team's.
        die_with_parent: End the command when the process that started the
            script ends, however that happens. Not for a shell: the kernel
            counts the thread that started it as the parent, and a
            terminal's may end while its shell is still wanted.
    """
    quote = shlex.quote
    run = f'{start} "$@"'.strip()
    if bwrap is None:
        return f"#!/bin/sh\nexec {run}\n"

    def read_only(path: Path) -> str:
        return f"--ro-bind-try {quote(str(path))} {quote(str(path))}"

    lines = [
        "#!/bin/sh",
        "top=$(git rev-parse --show-toplevel 2>/dev/null) || top=$(pwd)",
    ]
    # A later mount covers an earlier one, so each entry narrows or reopens
    # what the entries before it set.
    mounts = [*(read_only(path) for path in checkouts), '--ro-bind "$top" "$top"']
    if directory is not None:
        lines += [
            f'own="$top"/{quote(directory)}' if directory else 'own="$top"',
            'mkdir -p "$own"',
            "git_dir=$(git rev-parse --path-format=absolute --git-common-dir)",
        ]
        mounts.append('--bind "$own" "$own"')
        mounts += [
            f'--ro-bind-try "$top"/{quote(sub)} "$top"/{quote(sub)}'
            for sub in sub_team_directories
        ]
        # Committing writes the repository's git directory.
        mounts += ['--bind "$git_dir" "$git_dir"', read_only(private)]
        if not stay:
            mounts.append('--chdir "$own"')
    command = [
        f"exec {quote(bwrap)} --dev-bind / /",
        *(["--die-with-parent"] if die_with_parent else []),
        *mounts,
        f"-- {run}",
    ]
    lines.append(" \\\n  ".join(command))
    return "\n".join(lines) + "\n"


def write_shell(path: Path, script: str) -> None:
    """Store ``script`` as an executable, unless it is already there unchanged."""
    try:
        if path.read_text() == script:
            return
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, script, mode=0o700)
