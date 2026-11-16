"""What a session's work has to be reconciled with before it is merged.

Sessions at one team work side by side on the same directory, each from the
commit it started at. git says when two of them changed the same lines; it
says nothing when they changed the same file in ways that merge, or when a
later one rewrote what an earlier one had just delivered. So before a
session is merged, the files it changed are compared with what was merged
into the checkout above since it started, and the IC that reviews it is
asked to reconcile the two where they meet.
"""

from dataclasses import dataclass
from pathlib import Path

from openhands.agent_server.director.git import git, head_of


# How the commit that merges a team's accepted work starts.
MERGE_SUBJECT = "Merge the work of team "


@dataclass(frozen=True)
class Changes:
    """What a session's worktree and the checkout above each changed since
    they parted."""

    own: frozenset[str]
    above: frozenset[str]
    # The teams whose work was merged into the checkout above meanwhile.
    landed: tuple[str, ...] = ()
    tip: str = ""

    @property
    def both(self) -> list[str]:
        """The files both changed."""
        return sorted(self.own & self.above)


def _changed(checkout: Path, bases: list[str], tip: str) -> frozenset[str]:
    return frozenset(
        path
        for base in bases
        for path in git(checkout, ["diff", "--name-only", "-z", base, tip]).split("\0")
        if path
    )


def changes_of(above: Path, team: Path) -> Changes | None:
    """The files a team's worktree and the checkout above changed since the
    commits they have in common.

    Returns:
        None when either has no commit, or the worktree has nothing that is
        not already in the checkout above.

    Raises:
        GitError: A checkout could not be read.
    """
    tip, ours = head_of(team), head_of(above)
    if tip is None or ours is None:
        return None
    bases = git(above, ["merge-base", "--all", ours, tip]).split()
    if not bases or tip in bases:
        return None
    merges = git(
        above, ["log", "--first-parent", "--merges", "--format=%s", f"{tip}..{ours}"]
    ).splitlines()
    return Changes(
        own=_changed(above, bases, tip),
        above=_changed(above, bases, ours),
        landed=tuple(
            dict.fromkeys(
                subject.removeprefix(MERGE_SUBJECT).rsplit(" (", 1)[0]
                for subject in merges
                if subject.startswith(MERGE_SUBJECT)
            )
        ),
        tip=tip,
    )
