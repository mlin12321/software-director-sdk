"""The kinds of team an org may have: who is in one and what each is told.

A type is a YAML file, a team template. It names the agents of a team of
that type (its members), the director's tools each has, and which of them
works in the team's folder; and it says what each is for, what the team's
spec, contracts and reports hold, what each of the director's tools means
to it, and how its work is written up and reviewed. A type that names no
members has the two every team had before types named them: an IC and a
Manager.

A member is whatever the type names it. What it may do is the tools it is
given: the director finds whom to turn to by tool, so the reviewer of a
team is the member of the team above that has ``review_team``. Every team
keeps its folder and lands its work the same way, whatever its type.

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
    model_validator,
)


logger = logging.getLogger(__name__)

# The type of a team whose proposal names none, and of every team set up
# before teams had types.
DEFAULT_TYPE = "software-engineer"
SHIPPED_TYPES = Path(__file__).parent / "types"
# The folder of an org's own types, beside its ``director.yaml``.
TYPES_FOLDER = "types"
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")

Primitive = Literal["spec", "contracts", "report"]

# Tools the director finds one agent of a team by: a type gives each to
# exactly one member. The last is found among the tools members have where
# a team only reviews the work of a session.
_HELD_BY_ONE = ("submit_review", "review_team", "propose_teams", "approve_teams")
_LEAD_HELD_BY_ONE = ("settle_teams",)


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
    like: str | None = Field(
        default=None,
        description=(
            "The member it is started like: it runs on what that member "
            "runs on. None: it is started as the team is."
        ),
    )
    text: str = Field(default="", description="What it is for.")


def default_agents() -> dict[str, Member]:
    """The members of a type that names none: an IC and a Manager."""
    both = ["read_team_context", "raise_escalation", "resolve_escalation", "report"]
    return {
        "ic": Member(
            writes=True,
            tools=[
                *both,
                "write_spec",
                "edit_contracts",
                "msg_manager",
                "propose_teams",
                "msg_team",
                "hand_over",
                "stop_team",
                "submit_review",
                "review_team",
            ],
            lead_tools=["read_team_context", "report", "review_team"],
        ),
        "manager": Member(
            like="ic",
            tools=[
                *both,
                "msg_ic",
                "approve_teams",
                "decline_teams",
                "prune_team",
            ],
            lead_tools=["read_team_context", "report", "settle_teams"],
        ),
    }


class AgentType(BaseModel):
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
        default_factory=default_agents,
        description=(
            "The members of a team of this type, by the name of each. "
            "Left out: an IC (`ic`) and a Manager (`manager`)."
        ),
    )
    ic: str = Field(
        default="",
        description="What the member `ic` is for: the same as its `text`.",
    )
    manager: str = Field(
        default="",
        description="What the member `manager` is for: the same as its `text`.",
    )
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

    @model_validator(mode="after")
    def _has_whom_the_director_turns_to(self) -> "AgentType":
        agents = self.agents
        for name in ("ic", "manager"):
            said = getattr(self, name)
            if not said:
                continue
            if name not in agents:
                raise ValueError(
                    f"`{name}:` says what the member {name!r} is for, and "
                    "this type has no member of that name"
                )
            if agents[name].text and agents[name].text != said:
                raise ValueError(
                    f"the member {name!r} is told two things: give `{name}:` "
                    "or its `text`, not both"
                )
            agents[name] = agents[name].model_copy(update={"text": said})
        for name, member in agents.items():
            if not _NAME.fullmatch(name):
                raise ValueError(
                    "a member's name is lower-case letters, digits and hyphens"
                )
            if member.like is not None and (
                member.like == name or member.like not in agents
            ):
                raise ValueError(
                    f"the member {name!r} is started like {member.like!r}, "
                    "which is no other member of this type"
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

    def text_for(self, role: str) -> str:
        """What the member ``role`` of a team of this type is told, as one
        block; '' when the type says nothing to it."""
        lines: list[str] = []
        member = self.agents.get(role)
        said = member.text.strip() if member else ""
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


# A type is a team template; the older name stays for what reads it.
TeamTemplate = AgentType
