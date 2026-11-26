"""What a document is told of the work of the team that keeps it."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from openhands.agent_server.director.documents.store import DocRef


@dataclass(frozen=True)
class TeamRef:
    """The team a document is told of: the work of one team in a session.

    Attributes:
        id: The id of the session's place of the team.
        session: The session the work is of.
        folder: The team's folder, where its documents are kept.
    """

    id: str
    session: str
    folder: str

    def keeps(self, name: str) -> DocRef:
        """The team's document of that name."""
        return DocRef(self.folder, name)


@dataclass(frozen=True)
class SubTeamsCreated:
    """A proposal was approved and the team now has the sub-teams it
    named.

    Attributes:
        team: The team that has them.
        given: What the proposal gave for documents, by the name of each.
        replaced: Whether the proposal replaced sub-teams the team had.
    """

    team: TeamRef
    given: Mapping[str, str] = field(default_factory=dict)
    replaced: bool = False


@dataclass(frozen=True)
class WorkAccepted:
    """The team's reviewer accepted what a sub-team submitted.

    Attributes:
        team: The team whose reviewer accepted.
        given: What the reviewer gave for documents, by the name of each.
    """

    team: TeamRef
    given: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HandedOver:
    """The team above passed on to the team what the teams that were
    there before it left.

    Attributes:
        team: The team it was passed on to.
        given: What was given for documents, by the name of each.
    """

    team: TeamRef
    given: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TeamRemoved:
    """The team was replaced or pruned.

    Attributes:
        team: The team that is gone.
    """

    team: TeamRef
