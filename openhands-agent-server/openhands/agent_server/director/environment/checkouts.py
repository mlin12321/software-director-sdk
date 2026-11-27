"""Where each team's work is and how it moves between teams.

A team works in a checkout: a directory with the files of its repository
at some version. Teams may share one or have one each. `Checkouts` reads
the state of a team's work there, compares the work of two teams, and
brings the work of one into the checkout of another.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from openhands.agent_server.director.models import (
    AcceptedChanges,
    FileEdit,
    FileSides,
    NodeOwnership,
    OrgNode,
    SessionTeam,
    TeamChanges,
)


def time_stamp() -> str:
    """The time now, as it is written into the name work is kept under."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class MergeBack:
    """How bringing a team's work into the checkout above went.

    ``merged``: it is in it now. ``nothing``: it already was.
    ``conflicts``: ``files`` do not merge. ``outside``: the team changed
    ``files`` that are not in its directory, or that it may not write there.
    ``refused``: the checkout would not take a merge as it is; ``detail``
    says why.
    """

    outcome: Literal["merged", "nothing", "conflicts", "outside", "refused"]
    tip: str = ""
    files: tuple[str, ...] = ()
    detail: str = ""


class MergeCheckout(BaseModel):
    """A merge a reviewer finishes."""

    where: str
    # The version of the checkout above the work is merged onto.
    ours: str
    # The version of the work.
    tip: str
    # The files that do not merge, relative to the repository.
    files: list[str]


class Unfinished(Exception):
    """The merge is not as it has to be to be taken."""


@dataclass(frozen=True)
class Overlap:
    """Where the work of a team meets what came into the checkout above
    since the team began."""

    tip: str
    """The version of the team's work that was compared."""
    files: list[str]
    """The team's files that both sides changed and that merge."""
    landed: tuple[str, ...]
    """Whose work came into the checkout above meanwhile."""
    seen: str
    """Names the state of ``files`` above: it differs once one changes."""


class Checkouts(ABC):
    """Reads and moves the work of teams.

    A ``workspace`` is any directory inside a checkout. Every operation
    may raise `EnvironmentFailed` unless it says what it returns instead.
    """

    # Repositories and checkouts.

    @abstractmethod
    def is_repository(self, directory: Path) -> bool:
        """Whether ``directory`` is in a repository teams can work in."""

    @abstractmethod
    def root_of(self, workspace: str | Path) -> Path:
        """The top of the checkout ``workspace`` is in, symlinks resolved;
        ``workspace`` itself where it is in none."""

    @abstractmethod
    def enclosing(self, repository: Path) -> Path | None:
        """The repository that ``repository`` is a part of, if it is."""

    @abstractmethod
    def is_apart(self, team: SessionTeam, above: SessionTeam) -> bool:
        """Whether a team works in a checkout of its own, not in the one
        of the team above."""

    @abstractmethod
    def missing(
        self, repository: Path, teams: Iterable[SessionTeam]
    ) -> list[tuple[SessionTeam, bool]]:
        """The teams whose own checkout is gone, parents first, each with
        whether it can be made again."""

    @abstractmethod
    def restore(
        self, repository: Path, teams: Iterable[SessionTeam], within: str
    ) -> None:
        """Make the checkouts of ``teams`` that are gone again, each as its
        team left its work. One that cannot be made is logged.

        Args:
            within: Where in a checkout the teams' workspaces are; '' at
                its top.
        """

    # The state of work.

    @abstractmethod
    def version(self, workspace: str | Path) -> str | None:
        """What names the work a checkout holds now; None when it holds
        none or cannot be asked."""

    @abstractmethod
    def branch(self, workspace: str | Path) -> str:
        """The line of work a checkout is on, as the environment names
        it; '' when it cannot be asked."""

    @abstractmethod
    def uncommitted(self, workspace: str | Path, path: str) -> list[str]:
        """The files under ``path`` whose changes are not part of any
        version yet."""

    @abstractmethod
    def files(self, workspace: str | Path) -> list[str]:
        """Every file the checkout keeps, from its top; none when it
        cannot be asked."""

    @abstractmethod
    def file_count(self, workspace: str | Path, path: str) -> int:
        """How many files the checkout keeps under ``path``."""

    @abstractmethod
    def changed_together(self, workspace: str | Path, path: str, other: str) -> int:
        """How often ``path`` and ``other`` changed in one step lately."""

    @abstractmethod
    def last_change(self, workspace: str | Path, path: str) -> str | None:
        """What names the last change under ``path``; None when there is
        none or it cannot be asked."""

    @abstractmethod
    def astray(
        self, team: SessionTeam, below: Sequence[SessionTeam], repository: Path
    ) -> str | None:
        """Why what a team's checkout is on is not all of its work, in
        words for an agent; None when the work is all there.

        Args:
            below: The teams directly below it.
            repository: The org's own checkout, which is the human's.
        """

    @abstractmethod
    def changes_listed(self, team: SessionTeam) -> str | None:
        """What a team changed in its directory since it began, listed
        for its own account of it; '' when nothing, None when it cannot
        be told."""

    @abstractmethod
    def own_changes(
        self, team: SessionTeam, below: Sequence[SessionTeam]
    ) -> list[str] | None:
        """The files a team changed itself since its work was last taken
        up: those in its directory that are in no sub-team's. None when
        it cannot be told."""

    @abstractmethod
    def overlap(self, team: SessionTeam, above: SessionTeam) -> Overlap | None:
        """Where the work of a team meets what came into the checkout
        above meanwhile; None for a team that shares that checkout."""

    # Moving work.

    @abstractmethod
    def merge(self, team: SessionTeam, above: SessionTeam, title: str) -> MergeBack:
        """Bring the work of a team with a checkout of its own into the
        checkout of the team above."""

    @abstractmethod
    def undo_merge(self, workspace: str | Path, before: str) -> None:
        """Take a checkout back to ``before``, keeping what is uncommitted
        in it."""

    @abstractmethod
    def lay_out_merge(self, repository: Path, where: Path, ours: str, tip: str) -> None:
        """Start the merge of ``tip`` onto ``ours`` in a checkout of its
        own at ``where`` and leave what does not merge marked there."""

    @abstractmethod
    def finish_merge(self, merge: MergeCheckout, title: str, path: str) -> str:
        """Make a laid-out merge a version, once its files are settled.

        Returns:
            What names the merged work.

        Raises:
            Unfinished: A file still does not merge.
        """

    @abstractmethod
    def take_merge(self, workspace: str | Path, version: str) -> MergeBack:
        """Bring a finished merge into the checkout it was made for."""

    @abstractmethod
    def drop_merge(self, repository: Path, where: Path) -> None:
        """Remove the checkout a merge was laid out in."""

    @abstractmethod
    def set_aside(self, workspace: str | Path, path: str, ref: str) -> str | None:
        """Keep what is uncommitted under ``path`` as ``ref`` and take it
        out of the checkout.

        Returns:
            What names the kept work; None when there was nothing.
        """

    @abstractmethod
    def bring_over(
        self, source: str | Path, target: str | Path, path: str, ref: str
    ) -> list[str]:
        """Move what is uncommitted under ``path`` from one checkout into
        another, keeping it as ``ref`` on the way.

        Returns:
            The files that were moved.
        """

    @abstractmethod
    def keep_unmerged(self, own: str | Path, into: str | Path, ref: str) -> bool:
        """Keep the work of the checkout ``own`` as ``ref`` unless all of
        it is in the checkout ``into``.

        Returns:
            Whether anything was kept.
        """

    # What the human reads and edits by hand.

    @abstractmethod
    def ownership(
        self, node: OrgNode, reports: Sequence[OrgNode], workspace: str | Path
    ) -> NodeOwnership:
        """Which files of a team's directory are its own and which are
        its sub-teams'."""

    @abstractmethod
    def team_changes(self, team: SessionTeam) -> TeamChanges:
        """What a team changed since its work was last taken up."""

    @abstractmethod
    def file_sides(self, team: SessionTeam, path: str) -> FileSides:
        """A file of a team as it was and as it is."""

    @abstractmethod
    def accepted_changes(
        self, repository: Path, base: str, head: str, path: str
    ) -> AcceptedChanges:
        """What accepted work changed under ``path`` between two versions."""

    @abstractmethod
    def accepted_file(
        self, repository: Path, base: str, head: str, path: str
    ) -> FileSides:
        """A file as it was before and after accepted work."""

    @abstractmethod
    def require_editable(
        self, team: SessionTeam, sub_teams: Sequence[SessionTeam], path: str
    ) -> str:
        """``path`` in its plain form, when a person may change it by
        hand in a team's checkout.

        Raises:
            Invalid: ``path`` does not name a file of a checkout.
            Forbidden: The team's agents may not write the file either.
        """

    @abstractmethod
    def write_by_hand(
        self,
        team: SessionTeam,
        sub_teams: Sequence[SessionTeam],
        path: str,
        content: str,
    ) -> FileEdit:
        """Write a file in a team's checkout as the human's own change."""
