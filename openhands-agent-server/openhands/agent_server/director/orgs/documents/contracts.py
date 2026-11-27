"""What the sub-teams of a team promise each other."""

from collections.abc import Sequence
from typing import Any, ClassVar

from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.registry import register_protocol
from openhands.agent_server.director.documents.schema import (
    BriefedPart,
    DocumentsBriefed,
    DocumentToolCall,
    DocumentToolResult,
    ShownDocument,
)
from openhands.agent_server.director.documents.store import TeamRef
from openhands.agent_server.director.documents.writer import (
    DocumentWriterBase,
    EditorWriter,
)
from openhands.sdk.errors import Invalid


class ContractsWriter(EditorWriter):
    """Writes contracts for the member that settles them, and says who
    reads them now."""

    def __call__(self, call: DocumentToolCall) -> DocumentToolResult:
        words = call.words
        if call.leads_session:
            raise Invalid(
                "Contracts are not yours as the Division Lead "
                f"{words.get('reviewer', '')}: the Division "
                f"Lead {words.get('manager', '')} settles them"
            )
        super().__call__(call)
        if self.tree.below(call.team):
            return DocumentToolResult(
                "The contracts between your sub-teams are changed. Each "
                "sub-team is told with its next message; write to one with "
                "msg_team if it must know at once."
            )
        return DocumentToolResult(
            "Saved. Your team has no sub-teams yet: the ones you propose read "
            "these contracts once they are set up."
        )


class ContractsProtocol(DocumentProtocolBase):
    """Contracts are true of the teams they are between, so they become
    the org's when those teams do. A team writes the ones between its own
    sub-teams, and each of those reads them."""

    # A team keeps to the contracts the team above set between it and the
    # teams beside it, and sets those between its own sub-teams.
    read_by_sub_teams: ClassVar[bool] = True
    # Contracts that name teams which are gone are restated where what
    # those left is passed on.
    usable_in_hand_over: ClassVar[bool] = True
    waits_for_team_changes: ClassVar[bool] = True
    # Sub-teams cannot see each other's work: what they must keep between
    # them has to be there before they are.
    needed_before_sub_teams: ClassVar[bool] = True
    writer_class: ClassVar[type[DocumentWriterBase]] = ContractsWriter

    def told_to_reader(
        self,
        reader: TeamRef,  # noqa: ARG002
        shown: Sequence[ShownDocument],
    ) -> dict[str, Any]:
        return {
            "contracts_with_the_teams_beside_you": [
                found.text for found in shown if not found.own
            ],
            "contracts_between_your_sub_teams": [
                found.text for found in shown if found.own
            ],
        }

    def told_in_briefing(self, briefed: DocumentsBriefed) -> Sequence[BriefedPart]:
        return [
            BriefedPart(
                "Contracts between your sub-teams:"
                if found.own
                else "Contracts your team must keep with the teams beside it "
                "(set by the team above; read-only):",
                found.text,
            )
            for found in briefed.shown
        ]


register_protocol("contracts", ContractsProtocol)
