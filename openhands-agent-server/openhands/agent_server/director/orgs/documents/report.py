"""What the members of a team record for themselves as they work."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.events import RemovedTeamHandedOver
from openhands.agent_server.director.documents.protocol import (
    BriefedPart,
    DocumentArgument,
    DocumentProtocolBase,
    DocumentsBriefed,
    DocumentToolCall,
    DocumentToolResult,
    ShownDocument,
    register_protocol,
)
from openhands.agent_server.director.documents.store import LogEntry, TeamRef
from openhands.sdk.errors import Invalid


class ReportArguments(BaseModel):
    """Record a design decision, a setback or a reflection."""

    text: str = Field(description="What to record, in plain English.")


class ReportProtocol(DocumentProtocolBase):
    """A log that is only added to: each call of its tool adds an entry,
    and the members of the team read all of them."""

    @classmethod
    def arguments_for(cls, document: Document) -> type[BaseModel]:  # noqa: ARG003
        return ReportArguments

    @classmethod
    def arguments_in_other_tools(
        cls,
        document: Document,  # noqa: ARG003
    ) -> Sequence[DocumentArgument]:
        return (
            DocumentArgument(
                "hand_over",
                "notes",
                "What of the removed teams' reports this team should know, in "
                "your own words or quoted. Kept with its reports.",
                "your notes are with its reports, marked as handed over",
            ),
        )

    def write(self, call: DocumentToolCall) -> DocumentToolResult:
        given = call.arguments
        assert isinstance(given, ReportArguments)
        text = given.text.strip()
        if not text:
            raise Invalid("A report cannot be empty")
        self.store.append(
            call.team, self.document.name, LogEntry(at="", role=call.member, text=text)
        )
        return DocumentToolResult("Recorded for you and the other agent of your team.")

    def told_to_reader(
        self,
        reader: TeamRef,
        shown: Sequence[ShownDocument],  # noqa: ARG002
    ) -> dict[str, Any]:
        return {
            "reports_of_your_team": [
                {"by": _written_by(entry, reader), "at": entry.at, "text": entry.text}
                for entry in self.store.entries(reader, self.document.name)
            ]
        }

    def told_in_briefing(self, briefed: DocumentsBriefed) -> Sequence[BriefedPart]:
        entries = self.store.entries(briefed.reader, self.document.name)
        if not entries:
            return []
        return [
            BriefedPart(
                "What your team has reported so far:",
                "\n".join(
                    f"- {entry_line(entry, briefed.titles, briefed.above_writer)}"
                    for entry in entries
                ),
            )
        ]

    def when_removed_team_is_handed_over(self, handed: RemovedTeamHandedOver) -> None:
        # The notes of the team above on what the teams before left.
        notes = handed.given.get(self.document.name, "").strip()
        if notes:
            self.store.append(
                handed.team,
                self.document.name,
                LogEntry(at="", role=handed.writer, text=notes, handed_over=True),
            )


def entry_line(entry: LogEntry, titles: Mapping[str, str], above_writer: str) -> str:
    """One report, with which agent of the team wrote it.

    Args:
        titles: What each member of the team is called.
        above_writer: What the member that works in the folder of the team
            above is called.
    """
    if entry.handed_over:
        return (
            f"Handed over by the {above_writer} of the team above, from the "
            "teams that "
            f"were here before yours, {entry.at}: {entry.text}"
        )
    writer = titles.get(entry.role, entry.role)
    return f"{writer}, {entry.at}: {entry.text}"


def _written_by(entry: LogEntry, reader: TeamRef) -> str:
    """Who wrote an entry, as the members of a team are told."""
    if entry.handed_over:
        return "ic_of_the_team_above"
    if entry.session_id == reader.session:
        return entry.role
    return f"{entry.role} in {entry.session!r}"


register_protocol("report", ReportProtocol)
