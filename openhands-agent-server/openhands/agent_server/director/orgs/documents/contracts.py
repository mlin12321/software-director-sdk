"""What the sub-teams of a team promise each other."""

from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    register_protocol,
)


class ContractsProtocol(DocumentProtocolBase):
    """Contracts are true of the teams they are between, so they become
    the org's when those teams do."""

    def waits_for_team_changes(self) -> bool:
        return True


register_protocol("contracts", ContractsProtocol)
