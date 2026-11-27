"""What a document is as the work of the team that keeps it goes on.

A document that needs nothing of this is a plain text its team rewrites:
it is kept with :class:`DocumentProtocolBase` itself. One that does more
names a protocol in its YAML file, a subclass registered under that name
(``documents/registry.py``). What carries out a call of its tool is its
writer (``documents/writer.py``).
"""

from abc import ABC
from collections.abc import Sequence
from typing import Any, ClassVar

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.schema import (
    BriefedPart,
    DocumentsBriefed,
    ShownDocument,
)
from openhands.agent_server.director.documents.store import DocumentStore, TeamRef
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.documents.writer import (
    DocumentWriterBase,
    EditorWriter,
)


class DocumentProtocolBase(ABC):
    """What one document is: a few facts the director reads, who reads the
    document, and what they are told of it. Every answer here is the plain one,
    and a subclass gives another only where its document differs."""

    usable_in_hand_over: ClassVar[bool] = False
    """Whether its tool also works in a hand-over conversation."""
    read_by_sub_teams: ClassVar[bool] = False
    """Whether a team also reads the one the team above it keeps."""
    waits_for_team_changes: ClassVar[bool] = False
    """Whether what a session wrote of it becomes the org's only once the
    session's changes to the teams are settled, and not with its code."""
    stays_when_team_is_removed: ClassVar[bool] = False
    """Whether the one a removed team wrote stays in its folder, for the
    team that has the folder next to read as an old one. Otherwise it goes
    out of the folder and into the record of the team that is gone."""
    needed_before_sub_teams: ClassVar[bool] = False
    """Whether a team must have written it before its writer may propose
    sub-teams."""
    writer_class: ClassVar[type[DocumentWriterBase]] = EditorWriter
    """What carries out a call of its tool."""

    def __init__(
        self, document: Document, store: DocumentStore, tree: TeamTree
    ) -> None:
        """
        Args:
            document: The document this is the protocol of.
            store: Where documents are kept.
            tree: Where each team stands among the others.
        """
        self.document = document
        self.store = store
        self.tree = tree
        self.writer = self.writer_class(document, store, tree)

    def told_to_reader(
        self,
        reader: TeamRef,  # noqa: ARG002
        shown: Sequence[ShownDocument],
    ) -> dict[str, Any]:
        """What the context of an agent says of this document: the parts
        it adds to it, by the name of each. The agent is told of a part
        again whenever it changes.

        Args:
            reader: The team whose agent is told.
            shown: The documents of this kind the agent's team reads, as
                ``read_from`` names them; none that is empty.

        The plain answer: its team's own, under the document's name.
        """
        own = next((found.text for found in shown if found.own), None)
        return {self.document.name: own}

    def told_in_briefing(self, briefed: DocumentsBriefed) -> Sequence[BriefedPart]:
        """What an agent's first message says of this document.

        The plain answer: its team's own, when it has one.
        """
        return [
            BriefedPart(f"Your team's {self.document.name}:", found.text)
            for found in briefed.shown
            if found.own
        ]

    def read_from(self, reader: TeamRef) -> list[TeamRef]:
        """The teams whose document of this kind the team ``reader``
        reads as a matter of course, in the order it is shown them. It is
        worked out from the tree whenever it is asked and never kept, so
        it holds for a team wherever it has been moved.

        The plain answer: its own, and before it the one of the team
        above where the document is ``read_by_sub_teams``."""
        above = self.tree.above(reader) if self.read_by_sub_teams else None
        return [reader] if above is None else [above, reader]
