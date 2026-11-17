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
    common_dir,
    git,
    head_of,
    top_of,
)
from openhands.agent_server.director.git_index import LOCK_FILE
from openhands.agent_server.director.ownership import can_merge_in_memory, conflicts
from openhands.agent_server.director.reconcile import MERGE_SUBJECT


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
    with open(common_dir(checkout) / LOCK_FILE, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def merge_message(title: str, directory: str) -> str:
    return f"{MERGE_SUBJECT}{title} ({directory or '/'})"


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
                + ["-m", merge_message(title, directory), tip],
                env=DIRECTOR_IDENTITY,
                timeout=_MERGE_TIMEOUT_SECONDS,
            )
        except GitError as error:
            with suppress(GitError):
                git(above, ["merge", "--abort"])
            return MergeBack(
                "refused", tip, detail=(error.stderr or error.stdout).strip()
            )
        _bring_up_to_date(above, ours)
    return MergeBack("merged", tip)


def take_merge(above: Path, commit: str) -> MergeBack:
    """Move the checkout above on to a merge that was made on top of the
    commit it is on.

    Raises:
        GitError: The checkout could not be read.
    """
    ours = head_of(above)
    if ours is None:
        raise GitError(["rev-parse", "HEAD"], 1, "A checkout has no commit", "")
    with _one_at_a_time(above):
        try:
            git(
                above,
                ["merge", "--ff-only", "--no-verify", "--quiet", commit],
                timeout=_MERGE_TIMEOUT_SECONDS,
            )
        except GitError as error:
            return MergeBack(
                "refused", commit, detail=(error.stderr or error.stdout).strip()
            )
        _bring_up_to_date(above, ours)
    return MergeBack("merged", commit)


def _bring_up_to_date(checkout: Path, before: str) -> None:
    """Take a merge that moved the ``HEAD`` of ``checkout`` on from
    ``before`` into the index of every agent that works there.

    An agent's git holds what is outside its team's directory at ``HEAD``
    by itself. What is inside it does not follow, so the index of an agent
    whose directory the merge changed would still have the files as they
    were, and its next commit would take the merge out again. Each merged
    file the agent has staged nothing for is set to what the merge made of
    it; what it staged stays staged.
    """
    merged = _names(git(checkout, ["diff", "--name-only", "-z", before, "HEAD"]))
    if not merged:
        return
    git_dir = Path(git(checkout, ["rev-parse", "--absolute-git-dir"]).strip())
    # The names are relative to the top of the checkout.
    top = Path(top_of(checkout))
    for index in git_dir.glob("agent-index-*"):
        if index.suffix:
            # What the agent's git keeps beside its index.
            continue
        own = {"GIT_INDEX_FILE": str(index)}
        with suppress(GitError):
            staged = _names(
                git(top, ["diff", "--cached", "--name-only", "-z", before], env=own)
            )
            behind = sorted(merged - staged)
            if behind:
                git(
                    top,
                    ["reset", "-q", "HEAD", "--pathspec-from-file=-"]
                    + ["--pathspec-file-nul"],
                    env={**own, "GIT_LITERAL_PATHSPECS": "1"},
                    stdin="\0".join(behind),
                )


def _names(listed: str) -> set[str]:
    return {name for name in listed.split("\0") if name}
