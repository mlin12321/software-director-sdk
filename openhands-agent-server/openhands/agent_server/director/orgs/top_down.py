"""An organisation in which what is written goes down the tree only."""

from openhands.agent_server.director.documents.store import DocumentRef, TeamRef
from openhands.agent_server.director.org import OrgBase, TeamMember


class TopDownOrg(OrgBase):
    """A team reads what it keeps itself and what a team above it keeps
    for it. Nothing a team keeps is read by the teams above or beside it,
    and a document is passed only to a team below."""

    def _is_or_is_above(self, above: TeamRef, team: TeamRef) -> bool:
        if above.folder == team.folder:
            return True
        found = self.tree.above(team)
        while found is not None:
            if found.id == above.id:
                return True
            found = self.tree.above(found)
        return False

    def allows_read(self, reader: TeamRef, owner: TeamRef, document: str) -> bool:  # noqa: ARG002
        return self._is_or_is_above(owner, reader)

    def allows_ref(
        self,
        sender: TeamMember,
        receiver: TeamMember,
        ref: DocumentRef,  # noqa: ARG002
    ) -> bool:
        return sender.team.id != receiver.team.id and self._is_or_is_above(
            sender.team, receiver.team
        )
