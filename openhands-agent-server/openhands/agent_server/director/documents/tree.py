"""The teams around a team, as a document or an org may ask of them."""

from typing import Protocol

from openhands.agent_server.director.documents.store import TeamRef


class TeamTree(Protocol):
    """Where a team stands among the others. It is read, never changed."""

    def above(self, team: TeamRef) -> TeamRef | None:
        """The team directly above ``team``; None at the top."""
        ...

    def below(self, team: TeamRef) -> list[TeamRef]:
        """The teams directly below ``team``."""
        ...
