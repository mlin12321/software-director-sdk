"""What a document is told of the work of the team that keeps it."""

from dataclasses import dataclass

from openhands.agent_server.director.documents.store import TeamRef


@dataclass(frozen=True)
class SubTeamsCreated:
    """A proposal was approved and the team now has the sub-teams it
    named.

    Attributes:
        team: The team that has them.
    """

    team: TeamRef


@dataclass(frozen=True)
class SubTeamWorkAccepted:
    """The team's reviewer accepted what a sub-team submitted.

    Attributes:
        team: The team whose reviewer accepted.
    """

    team: TeamRef


@dataclass(frozen=True)
class RemovedTeamHandedOver:
    """The team above passed on to the team what the teams that were
    there before it left.

    Attributes:
        team: The team it was passed on to.
        text: What the team above said of it, in its own words.
        writer: The member of that team that works in its folder.
    """

    team: TeamRef
    text: str = ""
    writer: str = ""


@dataclass(frozen=True)
class TeamRemoved:
    """The team was replaced or pruned.

    Attributes:
        team: The team that is gone.
    """

    team: TeamRef
