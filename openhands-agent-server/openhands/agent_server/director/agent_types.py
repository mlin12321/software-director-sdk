"""The kinds of team an org may have: who is in one and what each is told.

A type is a YAML file, a team template. It names the agents of a team of
that type (its members), the director's tools each has, and which of them
works in the team's folder; and it says what each is for, what documents
the team keeps and what each holds, what each of the director's tools
means to it, and how its work is written up and reviewed.

A member is whatever the type names it. What it may do is the tools it is
given: the director finds whom to turn to by tool, so the reviewer of a
team is the member of the team above that has ``review_team``. Every team
keeps its folder and lands its work the same way, whatever its type.

The types shipped with the director are in ``director/types``; an org adds
its own in the ``types`` folder beside its ``director.yaml``, where a file
of the same name takes the place of a shipped one.

Every type starts from the ``_defaults.yaml`` of those folders, the
shipped one first and the org's over it, and says only what it changes:
mappings are merged key by key, a list or a plain value takes the place of
the default.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# The type of a team whose proposal names none.
DEFAULT_TYPE = "software-engineer"
SHIPPED_TYPES = Path(__file__).parent / "types"
# The folder of an org's own types, beside its ``director.yaml``.
TYPES_FOLDER = "types"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

# What every type of the folder starts from.
DEFAULTS_FILE = "_defaults.yaml"

# The documents the director keeps itself, for every team of every type:
# where each is kept, who writes it and who reads it are the director's.
DIRECTOR_DOCUMENTS = ("spec", "contracts", "report")
# Which other teams read a document of a type's own, seen from the team
# that keeps it.
Reader = Literal["above", "below", "beside"]
_READERS: dict[Reader, str] = {
    "above": "the team above",
    "below": "your sub-teams",
    "beside": "the teams beside yours",
}

# Tools the director finds one agent of a team by: a type gives each to
# exactly one member. The last is found among the tools members have where
# a team only reviews the work of a session.
_HELD_BY_ONE = ("submit_review", "review_team", "propose_teams", "approve_teams")
_LEAD_HELD_BY_ONE = ("settle_teams",)


class InvalidAgentType(ValueError):
    """A type's file does not say what a type may say."""


def _a(name: str) -> str:
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"


_COUNTS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight")


@dataclass(frozen=True)
class Names:
    """What the members of a team are called, for what the director writes
    to that team's agents. Every text of the director's own that names a
    member takes the name from here.

    Attributes:
        writer: The member that works in the team's folder.
        reviewer: The member that reviews the work of the teams below.
        overseer: The member that settles what the team's sub-teams are.
        writers: More than one writer; '' is the name with an s.
        overseers: More than one overseer; '' is the name with an s.
        members: Every member with what it does, as in "an IC, who does
            the work".
        above_writer: The writer of the team above; '' where that team is
            of the same type or there is none.
        above_reviewer: The reviewer of the team above, likewise.
    """

    writer: str
    reviewer: str
    overseer: str
    writers: str = ""
    overseers: str = ""
    members: tuple[str, ...] = ()
    above_writer: str = ""
    above_reviewer: str = ""

    def __post_init__(self) -> None:
        for name, default in (
            ("writers", f"{self.writer}s"),
            ("overseers", f"{self.overseer}s"),
            ("above_writer", self.writer),
            ("above_reviewer", self.reviewer),
        ):
            if not getattr(self, name):
                object.__setattr__(self, name, default)

    @property
    def a_writer(self) -> str:
        return _a(self.writer)

    @property
    def a_overseer(self) -> str:
        return _a(self.overseer)

    @property
    def A_overseer(self) -> str:  # noqa: N802
        return "A" + _a(self.overseer)[1:]

    @property
    def reviews_own(self) -> bool:
        """Whether the member that works also reviews the teams below."""
        return self.reviewer == self.writer

    def under(self, above: "Names") -> "Names":
        """These names in a team whose team above is called by ``above``."""
        return replace(self, above_writer=above.writer, above_reviewer=above.reviewer)

    def words(self) -> dict[str, str]:
        """What a text of the director's own may put in for a member."""
        parts = [self.writer] if self.reviews_own else [self.writer, self.reviewer]
        count = len(self.members)
        return {
            "writer": self.writer,
            "a_writer": self.a_writer,
            "writers": self.writers,
            "reviewer": self.reviewer,
            "overseer": self.overseer,
            "a_overseer": self.a_overseer,
            "overseers": self.overseers,
            "above_writer": self.above_writer,
            "above_reviewer": self.above_reviewer,
            "count": _COUNTS[count] if count < len(_COUNTS) else str(count),
            "members": _listed(self.members),
            "other_parts": f"the {' and '.join(parts)} part"
            + ("" if len(parts) == 1 else "s"),
            "review_of_yours": "your review"
            if self.reviews_own
            else f"the review of your {self.reviewer}",
        }


def _listed(items: Sequence[str]) -> str:
    """Items that hold commas themselves, as one list in a sentence."""
    if len(items) < 3:
        return ", and ".join(items)
    return "; ".join(items[:-1]) + f"; and {items[-1]}"


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


class Document(BaseModel):
    """One document a team of a type keeps."""

    model_config = ConfigDict(extra="forbid")

    holds: str = Field(min_length=1, description="What is written in it.")
    path: str = Field(
        default="",
        description=(
            "Where a document of the type's own is, in the team's folder. "
            "The member that works there writes and commits it as any file. "
            "Not said of a document the director keeps."
        ),
    )
    readers: list[Reader] = Field(
        default_factory=list,
        description=(
            "Which other teams are given a document of the type's own with "
            "their context: the team `above` and the teams `beside` it, as "
            "it was last accepted, and the teams `below`, as it was last "
            "committed."
        ),
    )

    @field_validator("path")
    @classmethod
    def _in_the_folder(cls, path: str) -> str:
        parts = PurePosixPath(path).parts
        if path and (
            path.startswith("/")
            or "\\" in path
            or ".." in parts
            or parts[0] == ".git"
            or str(PurePosixPath(path)) != path
        ):
            raise ValueError(
                "a document's path is a plain one inside the team's folder, "
                "such as findings.md"
            )
        return path

    @property
    def read_by(self) -> str:
        """Who reads it and when, as said to the team that keeps it. What
        goes up or beside a team waits for its work to be accepted; what
        it says to its own sub-teams does not."""
        later = " and ".join(
            _READERS[reader] for reader in ("above", "beside") if reader in self.readers
        )
        parts = [f"{later} read what of it was accepted"] if later else []
        if "below" in self.readers:
            parts.append(f"{_READERS['below']} read it as you last committed it")
        return ", and ".join(parts)


class Member(BaseModel):
    """One agent of a team of a type, under the name the type gives it."""

    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(
        default_factory=list,
        description=(
            "The director's tools it has. They are all that sets one "
            "member apart from another in what it may do."
        ),
    )
    lead_tools: list[str] = Field(
        default_factory=list,
        description=(
            "The tools it has instead where its team only reviews the work "
            "of a session."
        ),
    )
    writes: bool = Field(
        default=False,
        description="Whether it works in the team's folder. One member does.",
    )
    title: str = Field(
        default="",
        description=(
            "What it is called in what the director writes to agents and "
            "people. Empty: its name."
        ),
    )
    plural: str = Field(
        default="",
        description="What several of it are called. Empty: the title with an s.",
    )
    text: str = Field(default="", description="What it is for.")
    protocol: str = Field(
        default="",
        description=(
            "What it is told of its part in how the org works, after what "
            "the director itself says of that."
        ),
    )


class TeamTemplate(BaseModel):
    """A kind of team: its members and what they are told."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = Field(
        min_length=1,
        description=(
            "What a team of this type is for, in a line or two: an IC and a "
            "person choose the type of a team by it."
        ),
    )
    agents: dict[str, Member] = Field(
        min_length=1,
        description="The members of a team of this type, by the name of each.",
    )
    documents: dict[str, Document] = Field(
        default_factory=dict,
        description=(
            "The documents a team of this type keeps, by name. `spec`, "
            "`contracts` and `report` are the director's own, and a type "
            "says only what each holds. Any other is the type's own: a "
            "file in the team's folder that the teams it names read."
        ),
    )
    actions: dict[str, str] = Field(
        default_factory=dict,
        description="What a tool of the director means to this type, by its name.",
    )
    review: ReviewTexts = Field(default_factory=ReviewTexts)
    sas_profile: str | None = Field(
        default=None,
        description=(
            "The agent profile of the server (a software-agent-sdk profile) "
            "that a new team of this type runs on when its proposal names "
            "none. It is the one thing a type says of what runs it: a type "
            "without it runs on whatever the team above does, and any "
            "profile can run any type."
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

    @model_validator(mode="after")
    def _has_whom_the_director_turns_to(self) -> "TeamTemplate":
        agents = self.agents
        for name in agents:
            if not _NAME.fullmatch(name):
                raise ValueError(
                    "a member's name is lower-case letters, digits and hyphens"
                )
        paths: set[str] = set()
        for name, document in self.documents.items():
            if not _NAME.fullmatch(name):
                raise ValueError(
                    "a document's name is lower-case letters, digits and hyphens"
                )
            if name in DIRECTOR_DOCUMENTS:
                if document.path or document.readers:
                    raise ValueError(
                        f"`{name}` is a document the director keeps: a type "
                        "says what it holds, not where it is or who reads it"
                    )
                continue
            if not document.path or not document.readers:
                raise ValueError(
                    f"the document `{name}` is the type's own: say its "
                    "`path` in the team's folder and its `readers`"
                )
            if document.path in paths:
                raise ValueError(f"two documents are kept in {document.path!r}")
            paths.add(document.path)
        writers = [name for name, member in agents.items() if member.writes]
        if len(writers) != 1:
            raise ValueError(
                "one member of a type works in the team's folder (`writes`); "
                f"this type has {len(writers)}"
            )
        for tools, held in (
            (_HELD_BY_ONE, lambda member: member.tools),
            (_LEAD_HELD_BY_ONE, lambda member: member.lead_tools),
        ):
            for tool in tools:
                holders = [
                    name for name, member in agents.items() if tool in held(member)
                ]
                if len(holders) != 1:
                    raise ValueError(
                        f"`{tool}` is held by one member of a type; this "
                        f"type gives it to {len(holders)}"
                    )
        return self

    @property
    def writer(self) -> str:
        """The member that works in the team's folder."""
        return next(name for name, member in self.agents.items() if member.writes)

    def member_with(self, tool: str, *, lead: bool = False) -> str | None:
        """The first member that has ``tool``.

        Args:
            lead: Among the tools members have where a team only reviews
                the work of a session.
        """
        for name, member in self.agents.items():
            if tool in (member.lead_tools if lead else member.tools):
                return name
        return None

    def has(self, member: str, tool: str) -> bool:
        """Whether ``member`` has ``tool``, in a team or where one only
        reviews the work of a session."""
        found = self.agents.get(member)
        return found is not None and (tool in found.tools or tool in found.lead_tools)

    @property
    def reviewer(self) -> str:
        """The member that reviews the work of the teams below."""
        return (
            self.member_with("review_team")
            or self.member_with("review_team", lead=True)
            or self.writer
        )

    @property
    def overseer(self) -> str:
        """The member that settles what a team's sub-teams are."""
        return (
            self.member_with("approve_teams")
            or self.member_with("settle_teams", lead=True)
            or self.writer
        )

    @property
    def names(self) -> Names:
        """What the members are called, in a team with a team of the same
        type above it."""
        writer = self.writer
        reviewer = (
            self.member_with("review_team")
            or self.member_with("review_team", lead=True)
            or writer
        )
        overseer = (
            self.member_with("approve_teams")
            or self.member_with("settle_teams", lead=True)
            or writer
        )

        def does(name: str) -> str:
            did = [
                what
                for held, what in (
                    (name == writer, "does the work"),
                    (
                        name == reviewer and name != writer,
                        "reviews what sub-teams submit",
                    ),
                    (name == overseer, "decides which sub-teams exist"),
                )
                if held
            ]
            called = _a(self.title_of(name))
            return f"{called}, who {' and '.join(did)}" if did else called

        return Names(
            writer=self.title_of(writer),
            reviewer=self.title_of(reviewer),
            overseer=self.title_of(overseer),
            writers=self.agents[writer].plural,
            overseers=self.agents[overseer].plural,
            members=tuple(does(name) for name in self.agents),
        )

    @property
    def own_documents(self) -> dict[str, Document]:
        """The documents of the type's own, by name."""
        return {
            name: document
            for name, document in self.documents.items()
            if name not in DIRECTOR_DOCUMENTS
        }

    @property
    def titles(self) -> dict[str, str]:
        """What each member is called in what the director writes."""
        return {name: self.title_of(name) for name in self.agents}

    def title_of(self, member: str) -> str:
        """What ``member`` is called in what the director writes."""
        found = self.agents.get(member)
        return (found.title if found else "") or member

    def text_for(self, role: str) -> str:
        """What the member ``role`` of a team of this type is told, as one
        block; '' when the type says nothing to it."""
        lines: list[str] = []
        member = self.agents.get(role)
        said = member.text.strip() if member else ""
        if said:
            lines.append(said)
        if self.documents:
            lines += ["", "What your team keeps:"]
            lines += [
                f"- {name}{_kept(document)}: {document.holds.strip()}"
                for name, document in self.documents.items()
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


def _kept(document: Document) -> str:
    """Where a document of a type's own is and who reads it; '' for one
    the director keeps."""
    if not document.path:
        return ""
    return (
        f" (`{document.path}` in your team's folder; commit it with your "
        f"work, and {document.read_by})"
    )


def _over(defaults: dict[str, Any], said: dict[str, Any]) -> dict[str, Any]:
    """What a type says, over what every type starts from: mappings key by
    key, anything else whole."""
    merged = dict(defaults)
    for key, value in said.items():
        was = merged.get(key)
        merged[key] = (
            _over(was, value)
            if isinstance(was, dict) and isinstance(value, dict)
            else value
        )
    return merged


def _mapping(path: Path) -> dict[str, Any]:
    """The mapping a YAML file holds.

    Raises:
        InvalidAgentType: The file cannot be read or holds no mapping.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise InvalidAgentType(f"{path}: cannot be read as YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise InvalidAgentType(f"{path}: a type is a mapping")
    return loaded


def read_defaults(folders: Iterable[Path]) -> dict[str, Any]:
    """What every type starts from: the ``_defaults.yaml`` of ``folders``,
    a later folder's over an earlier one's. A file that cannot be read is
    left out and logged."""
    found: dict[str, Any] = {}
    for folder in folders:
        path = folder / DEFAULTS_FILE
        if not path.is_file():
            continue
        try:
            found = _over(found, _mapping(path))
        except InvalidAgentType:
            logger.exception("Ignoring the defaults in %s", path)
    return found


def read_type(
    path: Path, tools: Iterable[str], defaults: dict[str, Any] | None = None
) -> TeamTemplate:
    """The type a file defines, over ``defaults``.

    Raises:
        InvalidAgentType: The file is not a type, names a tool the
            director does not have, or is not named as its type is.
    """
    known = set(tools)
    loaded = _over(defaults or {}, _mapping(path))
    loaded.setdefault("name", path.stem)
    try:
        found = TeamTemplate.model_validate(loaded)
    except ValidationError as error:
        raise InvalidAgentType(f"{path}: {error}") from error
    if found.name != path.stem:
        raise InvalidAgentType(
            f"{path}: the type is named {found.name!r}; name the file after it"
        )
    given = {
        tool
        for member in found.agents.values()
        for tool in (*member.tools, *member.lead_tools)
    }
    unknown = sorted({*found.actions, *found.omit_tools, *given} - known)
    if unknown:
        raise InvalidAgentType(
            f"{path}: the director has no tool named {', '.join(unknown)}"
        )
    return found


def load_types(
    folders: Iterable[Path], tools: Iterable[str]
) -> dict[str, TeamTemplate]:
    """The types of the YAML files in ``folders``, by name; a later folder's
    type takes the place of an earlier one of the same name. Each is read
    over the defaults of all the folders.

    A file that is no type is left out and logged: one bad file does not
    take the others with it.
    """
    known = tuple(tools)
    folders = list(folders)
    defaults = read_defaults(folders)
    found: dict[str, TeamTemplate] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")]):
            if path.name == DEFAULTS_FILE:
                continue
            try:
                found[path.stem] = read_type(path, known, defaults)
            except InvalidAgentType:
                logger.exception("Ignoring the agent type in %s", path)
    return found
