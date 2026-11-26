"""The kinds of team an org may have: who is in one and what each is told.

A type is a YAML file, a team template. It names the agents of a team of
that type (its members), the director's tools each has, and which of them
works in the team's folder; and it says what each is for, which documents
the team keeps and what each holds, what each of the director's tools
means to it, and how its work is written up and reviewed.

A member is whatever the type names it. What it may do is the tools it is
given: the director finds whom to turn to by tool, so the reviewer of a
team is the member of the team above that has ``review_team``. Every team
keeps its folder and lands its work the same way, whatever its type.

The types shipped with the director are those of the orgs under
``director/orgs``; an org adds its own in the ``types`` folder beside its
``director.yaml``, where a file of the same name takes the place of a
shipped one.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from openhands.agent_server.director.documents.declared import Document
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# The type of a team whose proposal names none.
DEFAULT_TYPE = "software-engineer"
# The folder of an org's own types, beside its ``director.yaml``.
TYPES_FOLDER = "types"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

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
    documents: dict[str, str | None] = Field(
        default_factory=dict,
        description=(
            "The documents the team keeps, by the name of each, and what "
            "each holds for a team of this type. Nothing said of one: what "
            "the document itself says it holds."
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
                f"- {name}: {(text or '').strip()}"
                for name, text in self.documents.items()
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


def read_type(
    path: Path, tools: Iterable[str], documents: Mapping[str, Document]
) -> TeamTemplate:
    """The type a file defines.

    Args:
        tools: The names of the director's own tools.
        documents: The documents a team may keep, by name. Each brings
            the tool it is written with.

    Raises:
        InvalidAgentType: The file is not a type, names a tool the
            director does not have or a document there is none of, or is
            not named as its type is.
    """
    known = {*tools, *(found.written_with for found in documents.values())}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise InvalidAgentType(f"{path}: cannot be read as YAML: {error}") from error
    if not isinstance(loaded, dict):
        raise InvalidAgentType(f"{path}: a type is a mapping")
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
    unknown = sorted(set(found.documents) - set(documents))
    if unknown:
        raise InvalidAgentType(
            f"{path}: there is no document named {', '.join(unknown)}"
        )
    return found.model_copy(
        update={
            "documents": {
                name: text or documents[name].holds
                for name, text in found.documents.items()
            }
        }
    )


def load_types(
    folders: Iterable[Path],
    tools: Iterable[str],
    documents: Mapping[str, Document],
) -> dict[str, TeamTemplate]:
    """The types of the YAML files in ``folders``, by name; a later folder's
    type takes the place of an earlier one of the same name.

    A file that is no type is left out and logged: one bad file does not
    take the others with it.
    """
    known = tuple(tools)
    found: dict[str, TeamTemplate] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted([*folder.glob("*.yaml"), *folder.glob("*.yml")]):
            try:
                found[path.stem] = read_type(path, known, documents)
            except InvalidAgentType:
                logger.exception("Ignoring the agent type in %s", path)
    return found
