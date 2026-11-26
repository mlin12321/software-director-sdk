"""How a team works inside its folder."""

from typing import ClassVar

from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    Writing,
    Written,
    register_protocol,
)


class SpecProtocol(DocumentProtocolBase):
    """A plain document of the team's own. Its first line says what the
    team is for, and the member that takes over what a removed team left
    may write it while it does."""

    usable_in_hand_over: ClassVar[bool] = True

    def write(self, writing: Writing) -> Written:
        written = super().write(writing)
        ref = writing.team.keeps(self.document.name)
        text = self.store.read(writing.team.session, ref) or ""
        first = text.strip().splitlines()[0] if text.strip() else ""
        return Written(written.answer, purpose=first.lstrip("# ").strip()[:200])


register_protocol("spec", SpecProtocol)
