"""How a team works inside its folder."""

from collections.abc import Sequence
from typing import ClassVar

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import (
    DocumentArgument,
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

    @classmethod
    def arguments_in_other_tools(
        cls,
        document: Document,  # noqa: ARG003
    ) -> Sequence[DocumentArgument]:
        return (
            DocumentArgument(
                "review_team",
                "specs",
                "Only when you accept the work of a session: for the spec of a "
                "folder in it, whether the org's stays, the session's holds, "
                "or a text you write from both does. Unnamed, the session's "
                "holds.",
                per_folder=True,
            ),
        )

    def write(self, call: DocumentToolCall) -> DocumentToolResult:
        written = super().write(call)
        ref = call.team.keeps(self.document.name)
        text = self.store.read(call.team.session, ref) or ""
        first = text.strip().splitlines()[0] if text.strip() else ""
        return DocumentToolResult(
            written.answer, summary=first.lstrip("# ").strip()[:200]
        )


register_protocol("spec", SpecProtocol)
