"""Brings the accepted work of a team with its own worktree into the
checkout of the team above."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openhands.agent_server.director.git import (
    DIRECTOR_IDENTITY,
    GitError,
    git,
    head_of,
)
from openhands.agent_server.director.git_index import LOCK_FILE
from openhands.agent_server.director.ownership import can_merge_in_memory, conflicts


if sys.platform != "win32":
    import fcntl

_MERGE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class MergeBack:
    """How bringing a team's commits into the checkout above went.

    ``merged``: they are in it now. ``nothing``: they already were.
    ``conflicts``: ``files`` do not merge. ``outside``: the team committed
    ``files`` that are not in its directory. ``refused``: git would not
    merge into the checkout as it is; ``detail`` says why.
    """

    outcome: Literal["merged", "nothing", "conflicts", "outside", "refused"]
    tip: str = ""
    files: tuple[str, ...] = ()
    detail: str = ""


@contextmanager
def _one_at_a_time(checkout: Path) -> Iterator[None]:
    """Hold the lock the agents' git takes for commands that move ``HEAD``."""
    if sys.platform == "win32":
        yield
        return
    common = git(
        checkout, ["rev-parse", "--path-format=absolute", "--git-common-dir"]
    ).strip()
    with open(Path(common) / LOCK_FILE, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def merge_back(above: Path, team: Path, directory: str, title: str) -> MergeBack:
    """Merge what a team committed in its own worktree into the checkout of
    the team above.

    Nothing is merged unless all of it is inside the team's directory and
    merges cleanly.

    Args:
        above: The checkout of the team above.
        team: The team's worktree.
        directory: The directory the team owns, relative to the repository.

    Raises:
        GitError: A checkout could not be read.
    """
    tip = head_of(team)
    ours = head_of(above)
    if tip is None or ours is None:
        raise GitError(["rev-parse", "HEAD"], 1, "A checkout has no commit", "")
    ancestors = git(above, ["merge-base", "--all", ours, tip]).split()
    if tip in ancestors:
        return MergeBack("nothing", tip)
    changed = {
        path
        for base in ancestors
        for path in git(above, ["diff", "--name-only", "-z", base, tip]).split("\0")
        if path
    }
    outside = sorted(
        path
        for path in changed
        if directory and path != directory and not path.startswith(f"{directory}/")
    )
    if outside:
        return MergeBack("outside", tip, tuple(outside))
    if can_merge_in_memory(above):
        clashing = conflicts(above, ours, tip)
        if clashing:
            return MergeBack("conflicts", tip, clashing)
    with _one_at_a_time(above):
        try:
            git(
                above,
                ["merge", "--no-ff", "--no-verify", "--no-gpg-sign", "--quiet"]
                + ["-m", f"Merge the work of team {title} ({directory or '/'})", tip],
                env=DIRECTOR_IDENTITY,
                timeout=_MERGE_TIMEOUT_SECONDS,
            )
        except GitError as error:
            with suppress(GitError):
                git(above, ["merge", "--abort"])
            return MergeBack(
                "refused", tip, detail=(error.stderr or error.stdout).strip()
            )
    return MergeBack("merged", tip)
