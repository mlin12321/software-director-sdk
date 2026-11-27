"""What carries out a call of a document's tool.

A document's protocol says what the document is; its writer does the
writing. :class:`EditorWriter` is the plain one: the text is rewritten
whole or changed in one place.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.schema import (
    DocumentToolCall,
    DocumentToolResult,
    editor_arguments,
)
from openhands.agent_server.director.documents.store import DocumentStore
from openhands.agent_server.director.documents.tree import TeamTree
from openhands.agent_server.director.memory_edit import edited
from openhands.sdk.errors import Invalid


class DocumentWriterBase(ABC):
    """Carries out a call of one document's tool."""

    @classmethod
    @abstractmethod
    def arguments_for(cls, document: Document) -> type[BaseModel]:
        """What the tool of ``document`` takes."""

    def __init__(
        self, document: Document, store: DocumentStore, tree: TeamTree
    ) -> None:
        """
        Args:
            document: The document this writes.
            store: Where documents are kept.
            tree: Where each team stands among the others.
        """
        self.document = document
        self.store = store
        self.tree = tree

    @abstractmethod
    def __call__(self, call: DocumentToolCall) -> DocumentToolResult:
        """Carry out a call of the document's tool.

        Raises:
            Invalid: The call cannot be carried out as it was made.
        """


class EditorWriter(DocumentWriterBase):
    """Writes a plain document: it is rewritten whole, or changed in one
    place, for the session of the team alone until its work is accepted."""

    @classmethod
    def arguments_for(cls, document: Document) -> type[BaseModel]:
        """The commands and arguments of the file editor."""
        return editor_arguments(document.written_with, document.name)

    def __call__(self, call: DocumentToolCall) -> DocumentToolResult:
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
