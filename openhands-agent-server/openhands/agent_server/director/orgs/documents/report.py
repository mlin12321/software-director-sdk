"""What the members of a team record for themselves as they work."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.registry import register_protocol
from openhands.agent_server.director.documents.schema import (
    BriefedPart,
    DocumentsBriefed,
    DocumentToolCall,
    DocumentToolResult,
    ShownDocument,
)
from openhands.agent_server.director.documents.store import LogEntry, TeamRef
from openhands.agent_server.director.documents.writer import DocumentWriterBase
from openhands.sdk.errors import Invalid


class ReportArguments(BaseModel):
    """Record a design decision, a setback or a reflection."""

    text: str = Field(description="What to record, in plain English.")


class ReportWriter(DocumentWriterBase):
    """Adds an entry to the log with each call."""

    @classmethod
    def arguments_for(cls, document: Document) -> type[BaseModel]:  # noqa: ARG003
        return ReportArguments

    def __call__(self, call: DocumentToolCall) -> DocumentToolResult:
        given = call.arguments
        assert isinstance(given, ReportArguments)
        text = given.text.strip()
        if not text:
            raise Invalid("A report cannot be empty")
        self.store.append(
            call.team, self.document.name, LogEntry(at="", role=call.member, text=text)
        )
        return DocumentToolResult("Recorded for you and the other agent of your team.")


class ReportProtocol(DocumentProtocolBase):
    """A log that is only added to: each call of its tool adds an entry,
    and the members of the team read all of them."""

    writer_class: ClassVar[type[DocumentWriterBase]] = ReportWriter

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
                    f"- {entry_line(entry, briefed.titles)}" for entry in entries
                ),
            )
        ]


def entry_line(entry: LogEntry, titles: Mapping[str, str]) -> str:
    """One report, with which agent of the team wrote it.

    Args:
        titles: What each member of the team is called.
    """
    writer = titles.get(entry.role, entry.role)
    return f"{writer}, {entry.at}: {entry.text}"


def _written_by(entry: LogEntry, reader: TeamRef) -> str:
    """Who wrote an entry, as the members of a team are told."""
    if entry.session_id == reader.session:
        return entry.role
    return f"{entry.role} in {entry.session!r}"


register_protocol("report", ReportProtocol)
