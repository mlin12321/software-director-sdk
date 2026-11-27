"""What the director records of each thing of several steps it begins.

One model for each kind. What a step finds out is kept in the same model, so
that the thing can be finished after a restart from what was recorded.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from openhands.agent_server.director.models import SessionTeamInput, SplitChild


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Brief(_Payload):
    """Start the first agent of a team with its task."""

    kind: Literal["brief"] = "brief"
    role: str
    task: str


class LaunchMember(_Payload):
    """Start a member of a team that has not run yet."""

    kind: Literal["launch_member"] = "launch_member"
    role: str
    task: str


class StartTeam(_Payload):
    """Start a team the team above gave its task.

    Attributes:
        parent: The place its agent is derived from.
        purpose: What it is for, when that is not all of the task.
        checkout: The checkout it has already.
    """

    kind: Literal["start_team"] = "start_team"
    task: str
    parent: str | None = None
    purpose: str | None = None
    checkout: str | None = None


class RequestReview(_Payload):
    kind: Literal["request_review"] = "request_review"


class Accept(_Payload):
    """Take in work its reviewer accepted.

    Attributes:
        needed_above: Why a team further up waits on the work.
        reconciled: What the reviewer decided about what the work meets.
        specs: What it decided about the specs of a session's work.
        base: The work goes into what the org shares.
    """

    kind: Literal["accept"] = "accept"
    needed_above: str
    reconciled: str = ""
    specs: list[dict[str, str]] = Field(default_factory=list)
    base: bool = False


class LandOrg(_Payload):
    kind: Literal["land_org"] = "land_org"
    org_holds: bool = False


class FinishSplit(_Payload):
    kind: Literal["finish_split"] = "finish_split"


class Split(_Payload):
    """Carry out an approved split. All of it is fixed by the step
    ``planned``, before anything changes."""

    kind: Literal["split"] = "split"
    teams: list[SessionTeamInput] = Field(default_factory=list)
    replaces: list[str] = Field(default_factory=list)
    moves: dict[str, str] = Field(default_factory=dict)
    conversations: list[str] = Field(default_factory=list)
    cut_off: list[str] = Field(default_factory=list)
    was_stopped: bool = False
    stamp: str = ""


class Prune(_Payload):
    """Remove a team and what is below it. Fixed by the step ``planned``,
    but for ``saved``, which the step ``discarded`` finds out."""

    kind: Literal["prune"] = "prune"
    reason: str
    title: str = ""
    path: str = ""
    parent: str = ""
    teams: list[str] = Field(default_factory=list)
    conversations: list[str] = Field(default_factory=list)
    stamp: str = ""
    saved: str | None = None


class CloseSession(_Payload):
    """Close a session. Fixed by the step ``planned``, but for ``kept``,
    which the step of that name finds out."""

    kind: Literal["close_session"] = "close_session"
    session: str = ""
    title: str = ""
    team: str = ""
    parent: str = ""
    teams: list[str] = Field(default_factory=list)
    conversations: list[str] = Field(default_factory=list)
    stamp: str = ""
    known: bool = False
    kept: str | None = None


class HandOver(_Payload):
    """Start the conversation in which a team's writer passes on what
    removed teams left.

    Attributes:
        former: The removed teams, each by its id and when it was removed.
    """

    kind: Literal["hand_over"] = "hand_over"
    conversation: str
    ended: Literal["replaced", "pruned"]
    former: list[tuple[str, str]]
    children: list[SplitChild] = Field(default_factory=list)
    title: str = ""
    where: str = ""
    reason: str = ""


IntentPayload = Annotated[
    Brief
    | LaunchMember
    | StartTeam
    | RequestReview
    | Accept
    | LandOrg
    | FinishSplit
    | Split
    | Prune
    | CloseSession
    | HandOver,
    Field(discriminator="kind"),
]
PAYLOAD: TypeAdapter[IntentPayload] = TypeAdapter(IntentPayload)
