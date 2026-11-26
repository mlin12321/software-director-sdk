"""What a document is said to be: one YAML file for each.

The file holds data only: what the document is called, what it holds, and
what its tool and its protocol are named. How it is written, merged or
shortened is not said here.
"""

import re
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# The folder of an org's own documents, beside its ``director.yaml``.
DOCUMENTS_FOLDER = "documents"
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


class InvalidDocument(ValueError):
    """A document's file does not say what a document may say."""


class Document(BaseModel):
    """A document teams may keep."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = Field(
        min_length=1,
        description=(
            "What it holds, in a line or two. A type of team may say what it "
            "holds for its teams instead."
        ),
    )
    tool: str = Field(
        default="",
        description=(
            "The name of the tool its team writes it with. Empty: `write_<name>`."
        ),
    )
    tool_description: str = Field(
        default="",
        description=(
            "What the agents that have its tool are told the tool is for. "
            "Empty: that it writes the document."
        ),
    )
    protocol: str | None = Field(
        default=None,
        description=(
            "The name of what the document does as its team's work goes on. "
            "None: it is a plain text its team rewrites."
        ),
    )

    @field_validator("name")
    @classmethod
    def _named_plainly(cls, name: str) -> str:
        if not _NAME.fullmatch(name):
            raise ValueError(
                "a document's name is lower-case letters, digits, hyphens "
                "and underscores"
            )
        return name

    @property
    def written_with(self) -> str:
        """The name of the tool its team writes it with."""
        return self.tool or f"write_{self.name.replace('-', '_')}"

    @property
    def tool_summary(self) -> str:
        """What the agents that have its tool are told it is for."""
        return " ".join(self.tool_description.split()) or (
            f"Write your team's {self.name}: {self.description.strip()} `create` "
            "with the whole of it in `file_text`, `str_replace` to change "
            "one place, `insert` to add text after a line."
        )


def read_document(path: Path) -> Document:
    """The document a file says there is.

    Raises:
        InvalidDocument: The file is not a document, or is not named as
            its document is.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise InvalidDocument(f"{path}: cannot be read as YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise InvalidDocument(f"{path}: a document is a mapping")
    loaded.setdefault("name", path.stem)
    try:
        found = Document.model_validate(loaded)
    except ValidationError as error:
        raise InvalidDocument(f"{path}: {error}") from error
    if found.name != path.stem:
        raise InvalidDocument(
            f"{path}: the document is named {found.name!r}; name the file after it"
        )
    return found


def load_documents(folders: Iterable[Path]) -> dict[str, Document]:
    """The documents of the YAML files in ``folders``, by name; a later
    folder's document takes the place of an earlier one of the same name.

    A file that is no document is left out and logged: one bad file does
    not take the others with it.
    """
    found: dict[str, Document] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")]):
            try:
                found[path.stem] = read_document(path)
            except InvalidDocument:
                logger.exception("Ignoring the document in %s", path)
    return found
