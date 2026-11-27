"""What the sub-teams of a team promise each other."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.events import (
    RemovedTeamHandedOver,
    SubTeamsCreated,
    SubTeamWorkAccepted,
    TeamRef,
    TeamRemoved,
)
from openhands.agent_server.director.documents.protocol import (
    INSERT_LINE,
    NEW_STR,
    OLD_STR,
    BriefedPart,
    DocumentArgument,
    DocumentProtocolBase,
    DocumentsBriefed,
    DocumentToolCall,
    DocumentToolResult,
    ShownDocument,
    register_protocol,
)
from openhands.sdk.errors import Invalid


class EditContractsArguments(BaseModel):
    """Change one place in the contracts between your sub-teams, or add to
    them.

    The commands and arguments of the file editor, without `create`.
    """

    command: Literal["str_replace", "insert"] = Field(
        description="`str_replace`: one place changes. `insert`: text is added."
    )
    old_str: str = Field(default="", description=OLD_STR)
    new_str: str = Field(default="", description=NEW_STR)
    insert_line: int | None = Field(default=None, ge=0, description=INSERT_LINE)


class ContractsProtocol(DocumentProtocolBase):
    """Contracts are true of the teams they are between, so they become
    the org's when those teams do."""

    @classmethod
    def arguments_for(cls, document: Document) -> type[BaseModel]:  # noqa: ARG003
        return EditContractsArguments

    @classmethod
    def arguments_in_other_tools(
        cls,
        document: Document,
    ) -> Sequence[DocumentArgument]:
        name = document.name
        return (
            DocumentArgument(
                "propose_teams",
                name,
                "What the sub-teams must keep between them: types, endpoints, "
                "file formats. They cannot see each other's work or your spec, "
                "so this must hold everything they have to respect. What teams "
                "promise each other is written only here, not in a spec.",
            ),
            DocumentArgument(
                "review_team",
                name,
                "Only when you accept: what this work establishes that every "
                "team below yours may rely on from now, a line for each. It is "
                "added to the contracts between your sub-teams, which each of "
                "them reads. Leave empty otherwise.",
                "What you wrote was added to the contracts between your "
                "sub-teams; each is told with its next message.",
            ),
            DocumentArgument(
                "hand_over",
                name,
                "The contracts this team's own sub-teams must keep between "
                "them. Takes the place of the ones it has. Only for a team that "
                "has, or took over, sub-teams.",
                "the contracts between its sub-teams are written",
            ),
            DocumentArgument(
                "settle_teams",
                name,
                "For contracts that wait: which holds for each folder. "
                "Unnamed, the session's hold.",
                per_folder=True,
            ),
        )

    def write(self, call: DocumentToolCall) -> DocumentToolResult:
        # Contracts are first written by the proposal that makes the
        # sub-teams, so the tool only changes them.
        given = call.arguments
        assert isinstance(given, EditContractsArguments)
        words = call.words
        if call.leads_session:
            raise Invalid(
                "Contracts are not yours as the Division Lead "
                f"{words.get('reviewer', '')}: the Division "
                f"Lead {words.get('manager', '')} settles them"
            )
        if not self.tree.below(call.team):
            raise Invalid(
                "Your team has no sub-teams, so it sets no contracts. The ones "
                "you keep are the team's above; raise what must change in them "
                "with raise_escalation."
            )
        ref = call.team.keeps(self.document.name)
        now = self.store.read(call.team.session, ref) or ""
        text = self.edited(now, given)
        text = f"{text.strip()}\n"
        if text != now:
            self.store.write(call.team.session, ref, text)
        return DocumentToolResult(
            "The contracts between your sub-teams are changed. Each sub-team is "
            "told with its next message; write to one with msg_team if it must "
            "know at once."
        )

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

    def read_from(self, reader: TeamRef) -> list[TeamRef]:
        # A team keeps to the contracts the team above set between it and
        # the teams beside it, and sets those between its own sub-teams.
        above = self.tree.above(reader)
        return [reader] if above is None else [above, reader]

    def waits_for_team_changes(self) -> bool:
        return True

    def when_sub_teams_are_created(self, created: SubTeamsCreated) -> None:
        # Contracts that named teams which are gone do not stay beside the
        # new ones.
        self._record(created.team, created.given, replace=created.replaced)

    def when_sub_team_work_is_accepted(self, accepted: SubTeamWorkAccepted) -> None:
        self._record(accepted.team, accepted.given, replace=False)

    def when_removed_team_is_handed_over(self, handed: RemovedTeamHandedOver) -> None:
        self._record(handed.team, handed.given, replace=True)

    def when_team_is_removed(self, removed: TeamRemoved) -> str | None:
        # The folder is left without the contracts of a team that is gone.
        ref = removed.team.keeps(self.document.name)
        contracts = self.store.read(removed.team.session, ref)
        if contracts:
            self.store.remove(removed.team.session, ref)
        return contracts or None

    def _record(
        self, team: TeamRef, given: Mapping[str, str], *, replace: bool
    ) -> None:
        """Add what was given to the team's contracts, or put it in their
        place."""
        contracts = given.get(self.document.name, "")
        if not contracts.strip():
            return
        ref = team.keeps(self.document.name)
        earlier = self.store.read(team.session, ref)
        text = f"{contracts.strip()}\n"
        if replace:
            if earlier != text:
                self.store.write(team.session, ref, text)
            return
        if earlier and text.strip() in earlier:
            return
        self.store.write(
            team.session,
            ref,
            f"{earlier.rstrip()}\n\n{text}" if earlier and earlier.strip() else text,
        )


register_protocol("contracts", ContractsProtocol)
