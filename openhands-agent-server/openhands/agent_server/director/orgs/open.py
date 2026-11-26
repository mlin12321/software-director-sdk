"""An organisation in which nothing is refused."""

from typing import ClassVar

from openhands.agent_server.director.org import OrgBase


class OpenOrg(OrgBase):
    """Any member may write to any other and pass it any document. It
    ships no types of team: its teams are of the types of the other kinds,
    or of the org's own."""

    name: ClassVar[str] = "open"
