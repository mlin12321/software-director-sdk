"""The types of agent a team may be: what its two agents are told.

A type is a YAML file of text. It says what the IC and the Manager of a
team of that type are for, what the team's spec, contracts and reports
hold, what each of the director's tools means to it, and how its work is
written up and reviewed. It changes nothing the runtime enforces: every
team keeps its folder, is reviewed by the IC above and lands its work the
same way, whatever its type.

The types shipped with the director are in ``director/types``; an org adds
its own in the ``types`` folder beside its ``director.yaml``, where a file
of the same name takes the place of a shipped one.
"""

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from openhands.agent_server.director.models import AgentRole


logger = logging.getLogger(__name__)

# The type of a team whose proposal names none, and of every team set up
# before teams had types.
DEFAULT_TYPE = "software-engineer"
SHIPPED_TYPES = Path(__file__).parent / "types"
# The folder of an org's own types, beside its ``director.yaml``.
TYPES_FOLDER = "types"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

Primitive = Literal["spec", "contracts", "report"]


class InvalidAgentType(ValueError):
    """A type's file does not say what a type may say."""


class ReviewTexts(BaseModel):
    """How the work of a team of a type is written up and reviewed."""

    model_config = ConfigDict(extra="forbid")

    write_up: str = Field(
        default="",
        description=(
            "What the team's IC is asked to write of its own work for the "
            "IC above, before the list of changed files. Empty: the "
            "director's own question."
        ),
    )
    checklist: str = Field(
        default="",
        description="What the IC above checks the work by, in its own words.",
    )
    checklist_skill: str | None = Field(
        default=None,
        description=(
            "The name of a public skill to check the work by, such as "
            "'codereview'. It is put before `checklist`."
        ),
    )


class AgentType(BaseModel):
    """What the agents of a team of one type are told."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = Field(
        min_length=1,
        description=(
            "What a team of this type is for, in a line or two: an IC and a "
            "person choose the type of a team by it."
        ),
    )
    ic: str = Field(default="", description="What the team's IC is for.")
    manager: str = Field(default="", description="What the team's Manager is for.")
    primitives: dict[Primitive, str] = Field(
        default_factory=dict,
        description="What the team's spec, contracts and reports hold.",
    )
    actions: dict[str, str] = Field(
        default_factory=dict,
        description="What a tool of the director means to this type, by its name.",
    )
    review: ReviewTexts = Field(default_factory=ReviewTexts)
    sas_profile: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sas_profile", "model"),
        description=(
            "The agent profile of the server (a software-agent-sdk profile) "
            "that a new team of this type runs on when its proposal names "
            "none. It is the one thing a type says of what runs it: a type "
            "without it runs on whatever the team above does, and any "
            "profile can run any type. `model` is read as this too."
        ),
    )
    omit_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tools of the director a team of this type is not offered. A "
            "type takes tools away; it never adds one."
        ),
    )

    @field_validator("name")
    @classmethod
    def _named_plainly(cls, name: str) -> str:
        if not _NAME.fullmatch(name):
            raise ValueError("a type's name is lower-case letters, digits and hyphens")
        return name

    def text_for(self, role: AgentRole) -> str:
        """What an agent of ``role`` in a team of this type is told, as one
        block; '' when the type says nothing to it."""
        lines: list[str] = []
        said = (self.ic if role == "ic" else self.manager).strip()
        if said:
            lines.append(said)
        if self.primitives:
            lines += ["", "What your team keeps:"]
            lines += [
                f"- {name}: {text.strip()}" for name, text in self.primitives.items()
            ]
        if self.actions:
            lines += ["", "What the director's tools mean for your team:"]
            lines += [
                f"- `{name}`: {text.strip()}" for name, text in self.actions.items()
            ]
        body = "\n".join(lines).strip()
        if not body:
            return ""
        return f'<TEAM_TYPE name="{self.name}">\n{body}\n</TEAM_TYPE>'


def _read(path: Path, tools: Iterable[str]) -> AgentType:
    """The type a file defines.

    Raises:
        InvalidAgentType: The file is not a type, names a tool the
            director does not have, or is not named as its type is.
    """
    known = set(tools)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise InvalidAgentType(f"{path}: cannot be read as YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise InvalidAgentType(f"{path}: a type is a mapping")
    loaded.setdefault("name", path.stem)
    try:
        found = AgentType.model_validate(loaded)
    except ValidationError as error:
        raise InvalidAgentType(f"{path}: {error}") from error
    if found.name != path.stem:
        raise InvalidAgentType(
            f"{path}: the type is named {found.name!r}; name the file after it"
        )
    unknown = sorted({*found.actions, *found.omit_tools} - known)
    if unknown:
        raise InvalidAgentType(
            f"{path}: the director has no tool named {', '.join(unknown)}"
        )
    return found


def load_types(folders: Iterable[Path], tools: Iterable[str]) -> dict[str, AgentType]:
    """The types of the YAML files in ``folders``, by name; a later folder's
    type takes the place of an earlier one of the same name.

    A file that is no type is left out and logged: one bad file does not
    take the others with it.
    """
    known = tuple(tools)
    found: dict[str, AgentType] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")]):
            try:
                found[path.stem] = _read(path, known)
            except InvalidAgentType:
                logger.exception("Ignoring the agent type in %s", path)
    return found
