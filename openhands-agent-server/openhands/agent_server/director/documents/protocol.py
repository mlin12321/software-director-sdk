"""What a document does as the work of the team that keeps it goes on.

A document that needs nothing of this is a plain text its team rewrites:
it is kept with :class:`DocumentProtocolBase` itself. One that does more
names a protocol in its YAML file, a subclass registered under that name.
"""

from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, create_model

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.events import (
    RemovedTeamHandedOver,
    SubTeamsCreated,
    SubTeamWorkAccepted,
    TeamRef,
    TeamRemoved,
)
from openhands.agent_server.director.documents.store import DocumentStore
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.memory_edit import edited
from openhands.sdk.errors import Invalid


class UnknownProtocol(ValueError):
    """A document names a protocol nobody registered."""


@dataclass(frozen=True)
class DocumentToolCall:
    """One call of a document's tool.

    Attributes:
        team: The team of the member that called.
        member: The member, by the name its team's type gives it.
        arguments: What the tool was called with.
        hand_over: Whether it called from its hand-over conversation.
        leads_session: Whether its team only reviews the work of a session.
        words: What the members of its team are called, for an answer.
    """

    team: TeamRef
    member: str
    arguments: BaseModel
    hand_over: bool = False
    leads_session: bool = False
    words: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ShownDocument:
    """A document as a team that reads it is shown it.

    Attributes:
        folder: The folder of the team that keeps it.
        name: The name of the document.
        text: What it says, as the reader's session has it.
        own: Whether the team that reads it is the one that keeps it.
    """

    folder: str
    name: str
    text: str
    own: bool


@dataclass(frozen=True)
class DocumentToolResult:
    """What came of a call of a document's tool.

    Attributes:
        answer: What the member that called is answered.
        purpose: What the team is for, in a line, where the document now
            says so; None where it says nothing of it.
    """

    answer: str
    purpose: str | None = None


OLD_STR = (
    "For `str_replace`: the text now there, exactly as it stands and there only once."
)
NEW_STR = (
    "For `str_replace`: what `old_str` is to read, empty to remove it. For "
    "`insert`: the text to insert."
)
INSERT_LINE = (
    "For `insert`: the line `new_str` goes after; 0 is the top. Left out: at the end."
)


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


@cache
def editor_arguments(tool: str, name: str) -> type[BaseModel]:
    """What the tool ``tool`` of a plain document named ``name`` takes: the
    commands and arguments of the file editor."""
    return create_model(
        f"{_camel(tool)}Arguments",
        __doc__=(
            f"Write the {name} of your team's folder, or change one place in it."
            "\n\nThe commands and arguments of the file editor."
        ),
        command=(
            Literal["create", "str_replace", "insert"],
            Field(
                default="create",
                description=(
                    f"`create`: `file_text` is the whole {name}. `str_replace`: one "
                    "place changes. `insert`: text is added after a line."
                ),
            ),
        ),
        file_text=(
            str,
            Field(
                default="", description=f"For `create`: the whole {name}, in Markdown."
            ),
        ),
        old_str=(str, Field(default="", description=OLD_STR)),
        new_str=(str, Field(default="", description=NEW_STR)),
        insert_line=(
            int | None,
            Field(default=None, ge=0, description=INSERT_LINE),
        ),
    )


class DocumentProtocolBase(ABC):
    """What one document does. Every answer here is the plain one, and a
    subclass gives another only where its document differs."""

    usable_in_hand_over: ClassVar[bool] = False
    """Whether its tool also works in a hand-over conversation."""

    @classmethod
    def arguments_for(cls, document: Document) -> type[BaseModel]:
        """What the tool of ``document`` takes. The plain answer: the
        commands and arguments of the file editor."""
        return editor_arguments(document.written_with, document.name)

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

    def write(self, call: DocumentToolCall) -> DocumentToolResult:
        """Carry out a call of the document's tool.

        The plain document is rewritten whole, or changed in one place,
        for the session of the team alone until its work is accepted.

        Raises:
            Invalid: The call cannot be carried out as it was made.
        """
        name = self.document.name
        given = call.arguments
        command: Any = getattr(given, "command", "create")
        text = getattr(given, "file_text", "")
        old_str = getattr(given, "old_str", "")
        new_str = getattr(given, "new_str", "")
        insert_line = getattr(given, "insert_line", None)
        ref = call.team.keeps(name)
        if command != "create":
            if text.strip():
                raise Invalid(
                    f"`file_text` is the whole {name} and goes with `create`; "
                    f"`{command}` takes `old_str`, `new_str` and `insert_line`"
                )
            own = self.store.read(call.team.session, ref)
            if own is None:
                raise Invalid(
                    f"Your team has no {name} to change yet: write it with `create`"
                )
            text = self.edited(own, given)
        elif old_str or new_str or insert_line is not None:
            raise Invalid(
                f"`create` takes the whole {name} in `file_text`; to change one "
                "place use `str_replace` or `insert`"
            )
        if not text.strip():
            raise Invalid(f"A {name} cannot be empty")
        self.store.write(call.team.session, ref, text)
        return DocumentToolResult(
            f"Saved the {name} of {call.team.folder or 'the repository'}."
        )

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

    def edited(self, text: str, arguments: BaseModel) -> str:
        """``text`` after the command of the file editor that ``arguments``
        holds: `str_replace` or `insert`.

        Raises:
            Invalid: The editor refuses the change.
        """
        command: Any = getattr(arguments, "command", "")
        return edited(
            text,
            command,
            old_str=getattr(arguments, "old_str", ""),
            new_str=getattr(arguments, "new_str", ""),
            insert_line=getattr(arguments, "insert_line", None),
            name=f"the {self.document.name}",
        )

    def read_from(self, reader: TeamRef) -> list[TeamRef]:
        """The teams whose document of this kind the team ``reader``
        reads as a matter of course, in the order it is shown them. It is
        worked out from the tree whenever it is asked and never kept, so
        it holds for a team wherever it has been moved.

        The plain answer: its own and no other team's."""
        return [reader]

    def waits_for_team_changes(self) -> bool:
        """Whether what a session wrote of this document becomes the
        org's only once the session's changes to the teams are settled,
        and not with its code."""
        return False

    def when_sub_teams_are_created(self, created: SubTeamsCreated) -> None:
        """A proposal was approved and the team now has sub-teams."""

    def when_sub_team_work_is_accepted(self, accepted: SubTeamWorkAccepted) -> None:
        """The team's reviewer accepted what a sub-team submitted."""

    def when_removed_team_is_handed_over(self, handed: RemovedTeamHandedOver) -> None:
        """The team above passed on to the team what the teams before it
        left."""

    def when_team_is_removed(
        self,
        removed: TeamRemoved,  # noqa: ARG002
    ) -> str | None:
        """The team was replaced or pruned.

        Returns:
            What is kept of the document in the record of the team that
            is gone; None for nothing.
        """
        return None


_PROTOCOLS: dict[str, type[DocumentProtocolBase]] = {}


def register_protocol(name: str, protocol: type[DocumentProtocolBase]) -> None:
    """Make ``protocol`` the one a document names as ``name``."""
    _PROTOCOLS[name] = protocol


def protocol_class(document: Document) -> type[DocumentProtocolBase]:
    """The class of what ``document`` does.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
    """
    if document.protocol is None:
        return DocumentProtocolBase
    protocol = _PROTOCOLS.get(document.protocol)
    if protocol is None:
        raise UnknownProtocol(
            f"The document {document.name!r} names the protocol "
            f"{document.protocol!r}, and there is none of that name"
        )
    return protocol


def protocol_of(
    document: Document, store: DocumentStore, tree: TeamTree
) -> DocumentProtocolBase:
    """What ``document`` does: the protocol it names, or the plain one.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
    """
    return protocol_class(document)(document, store, tree)


# Who is told of each document an org declares, and the ones told so far.
# The service that reads an org's documents does not know what gives its
# agents their tools; that part listens here.
_declared: dict[str, tuple[Document, type[DocumentProtocolBase]]] = {}
_listeners: list[Callable[[Document, type[DocumentProtocolBase]], None]] = []


def declare(document: Document) -> None:
    """Tell of a document an org keeps, so that its tool can be made.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
        ValueError: Its tool cannot be made: another of that name takes
            something else.
    """
    found = (document, protocol_class(document))
    for tell in _listeners:
        tell(*found)
    _declared[document.name] = found


def when_declared(
    tell: Callable[[Document, type[DocumentProtocolBase]], None],
) -> None:
    """Have ``tell`` called with every document declared, the ones before
    now as well."""
    _listeners.append(tell)
    for found in list(_declared.values()):
        tell(*found)
