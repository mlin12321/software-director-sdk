"""A merge that does not go through by itself, laid out for the IC that
reviews the work to finish.

The director starts the merge in a checkout of its own and leaves git's
conflict markers in the files that do not merge. The reviewer edits those
files and nothing else; the director stages them and makes the commit, so
what is merged outside them is what git made of it.
"""

import shutil
from contextlib import suppress
from pathlib import Path

from pydantic import BaseModel

from openhands.agent_server.director.git import DIRECTOR_IDENTITY, GitError, git


_MARKERS = ("<<<<<<< ", ">>>>>>> ")
_MERGE_TIMEOUT_SECONDS = 120


class MergeCheckout(BaseModel):
    """A merge a reviewer finishes."""

    where: str
    # The commit of the checkout above the work is merged onto.
    ours: str
    # The commit of the work.
    tip: str
    # The files that do not merge, relative to the repository.
    files: list[str]


class Unfinished(Exception):
    """The merge is not as it has to be to be taken."""


def prepare(repo_dir: Path, where: Path, ours: str, tip: str) -> None:
    """Lay out the merge of ``tip`` onto ``ours`` in a checkout at
    ``where``, with what does not merge left marked in the files.

    Raises:
        GitError: The checkout could not be made.
    """
    discard(repo_dir, where)
    where.parent.mkdir(parents=True, exist_ok=True)
    git(repo_dir, ["worktree", "add", "--detach", "--quiet", str(where), ours])
    with suppress(GitError):
        # It stops at the files that do not merge, which is what is wanted.
        git(
            where,
            ["merge", "--no-commit", "--no-ff", "--no-verify", "--quiet", tip],
            env=DIRECTOR_IDENTITY,
            timeout=_MERGE_TIMEOUT_SECONDS,
        )


def finish(merge: MergeCheckout, message: str) -> str:
    """Commit the merge as its reviewer left it.

    Returns:
        The merge commit.

    Raises:
        Unfinished: A file still has conflict markers or is gone, or a
            file that merged by itself was changed.
        GitError: git would not commit it.
    """
    where = Path(merge.where)
    listed = set(merge.files)
    head = git(where, ["rev-parse", "HEAD"]).strip()
    if head != merge.ours:
        # It was committed before and only waits to be taken.
        return head
    changed = {
        path for path in git(where, ["diff", "--name-only", "-z"]).split("\0") if path
    }
    added = {
        path
        for path in git(
            where, ["ls-files", "--others", "--exclude-standard", "-z"]
        ).split("\0")
        if path
    }
    other = sorted((changed | added) - listed)
    if other:
        raise Unfinished(
            "These files merged by themselves and are not yours to change "
            f"there: {', '.join(other)}. Put them back as they were."
        )
    marked = []
    for path in merge.files:
        file = where / path
        if not file.is_file():
            continue
        text = file.read_text(errors="replace")
        if any(line.startswith(_MARKERS) for line in text.splitlines()):
            marked.append(path)
    if marked:
        raise Unfinished(
            f"These files still have conflict markers: {', '.join(marked)}."
        )
    git(
        where,
        ["add", "--all", "--pathspec-from-file=-", "--pathspec-file-nul"],
        env={"GIT_LITERAL_PATHSPECS": "1"},
        stdin="\0".join(merge.files),
    )
    git(
        where,
        ["commit", "--no-verify", "--no-gpg-sign", "--quiet", "-m", message],
        env=DIRECTOR_IDENTITY,
    )
    return git(where, ["rev-parse", "HEAD"]).strip()


def discard(repo_dir: Path, where: Path) -> None:
    """Take a merge checkout away, whatever state it is in."""
    with suppress(GitError):
        git(repo_dir, ["worktree", "remove", "--force", str(where)])
    shutil.rmtree(where, ignore_errors=True)
    with suppress(GitError):
        git(repo_dir, ["worktree", "prune"])
