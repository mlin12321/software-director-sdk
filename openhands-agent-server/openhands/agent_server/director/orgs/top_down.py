"""An organisation in which what is written goes down the tree only."""

from openhands.agent_server.director.documents.events import TeamRef
from openhands.agent_server.director.org import OrgBase


class TopDownOrg(OrgBase):
    """A team reads what it keeps itself and what a team above it keeps
    for it. Nothing a team keeps is read by the teams above or beside it."""

    def allows_read(self, reader: TeamRef, owner: TeamRef, document: str) -> bool:  # noqa: ARG002
        if owner.folder == reader.folder:
            return True
        above = self.tree.above(reader)
        while above is not None:
            if above.id == owner.id:
                return True
            above = self.tree.above(above)
        return False
