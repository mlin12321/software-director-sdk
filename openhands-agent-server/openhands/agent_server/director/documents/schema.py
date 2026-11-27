"""What passes between a document and the director: a call of its tool
and what came of it, and a document as a team that reads it is shown it."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, create_model

from openhands.agent_server.director.documents.store import TeamRef


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
class DocumentsBriefed:
    """An agent is given its first message, which tells it what its team
    works to.

    Attributes:
        reader: The team whose agent is told.
        shown: The documents of this kind the team reads, as ``read_from``
            names them; none that is empty.
        titles: What each member of the team is called.
    """

    reader: TeamRef
    shown: Sequence[ShownDocument]
    titles: Mapping[str, str]


class BriefedPart(NamedTuple):
    """One part of an agent's first message: a line that says what it is,
    and the text."""

    heading: str
    text: str


@dataclass(frozen=True)
class DocumentToolResult:
    """What came of a call of a document's tool.

    Attributes:
        answer: What the member that called is answered.
        summary: What the team is for, in a line, where the document now
            says so; None where it says nothing of it.
    """

    answer: str
    summary: str | None = None


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
