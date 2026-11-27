"""How a team works inside its folder."""

from typing import ClassVar

from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    DocumentToolCall,
    DocumentToolResult,
    register_protocol,
)


class SpecProtocol(DocumentProtocolBase):
    """A plain document of the team's own. Its first line says what the
    team is for, and the member that takes over what a removed team left
    may write it while it does."""

    usable_in_hand_over: ClassVar[bool] = True
    # How a removed team worked is where the next team in its folder starts.
    stays_when_team_is_removed: ClassVar[bool] = True

    def write(self, call: DocumentToolCall) -> DocumentToolResult:
        written = super().write(call)
        ref = call.team.keeps(self.document.name)
        text = self.store.read(call.team.session, ref) or ""
        first = text.strip().splitlines()[0] if text.strip() else ""
        return DocumentToolResult(
            written.answer, summary=first.lstrip("# ").strip()[:200]
        )


register_protocol("spec", SpecProtocol)
