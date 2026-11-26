"""What kind of organisation a tree of teams is.

A kind of organisation is a subclass of ``OrgBase``. It says which types
of team it ships and what holds for the whole tree: who may write to whom,
who may pass a document to whom, who may read what another team keeps.
What one document does as a team's work goes on is not said here but by
the document's own protocol (``documents/protocol.py``).
"""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from openhands.agent_server.director.documents.store import DocumentRef, TeamRef
from openhands.agent_server.director.documents.tree import TeamTree


@dataclass(frozen=True)
class TeamMember:
    """One agent of the org: a member of a team.

    Attributes:
        team: Its team.
        member: What its team's type calls it.
    """

    team: TeamRef
    member: str


class OrgBase(ABC):
    """A kind of organisation. By itself it is one with no types of team
    of its own, in which nothing is refused."""

    name: ClassVar[str] = ""
    """What an org's settings call this kind by."""
    types_folder: ClassVar[Path | None] = None
    """The folder of the types of team it ships, one YAML file for each."""

    def __init__(self, tree: TeamTree) -> None:
        """
        Args:
            tree: Where each team of the org stands among the others.
        """
        self.tree = tree

    def allows_msg(
        self,
        sender: TeamMember,  # noqa: ARG002
        receiver: TeamMember,  # noqa: ARG002
    ) -> bool:
        """Whether ``sender`` may write a message to ``receiver``."""
        return True

    def allows_ref(
        self,
        sender: TeamMember,  # noqa: ARG002
        receiver: TeamMember,  # noqa: ARG002
        ref: DocumentRef,  # noqa: ARG002
    ) -> bool:
        """Whether ``sender`` may pass the document ``ref`` to
        ``receiver`` to read."""
        return True

    def allows_read(
        self,
        reader: TeamRef,  # noqa: ARG002
        owner: TeamRef,  # noqa: ARG002
        document: str,  # noqa: ARG002
    ) -> bool:
        """Whether the team ``reader`` may read the document named
        ``document`` that ``owner`` keeps. It is asked only of a document
        whose own protocol would have ``reader`` read it."""
        return True
