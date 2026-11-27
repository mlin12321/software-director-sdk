"""What a document is told of the work of the team that keeps it."""

from dataclasses import dataclass

from openhands.agent_server.director.documents.store import TeamRef


@dataclass(frozen=True)
class SubTeamsProposed:
    """The team's writer is about to put a proposal of sub-teams before
    the member that settles them.

    Attributes:
        team: The team that proposes.
    """

    team: TeamRef


@dataclass(frozen=True)
class TeamRemoved:
    """The team was replaced or pruned.

    Attributes:
        team: The team that is gone.
    """

    team: TeamRef
