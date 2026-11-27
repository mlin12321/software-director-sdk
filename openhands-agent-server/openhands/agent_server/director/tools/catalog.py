"""The director's own tools: what each takes and what it does.

Nothing here knows how an agent reaches a tool. ``definition`` offers
these to an agent of the server, ``remote`` to one that runs its own
tools.
"""

from typing import Literal

from pydantic import BaseModel, Field

from openhands.agent_server.director import agent_actions
from openhands.agent_server.director.models import (
    DocumentChoice,
    EscalationKind,
    ReviewFinding,
    TeamChoice,
)
from openhands.agent_server.director.split import ProposedTeam
from openhands.agent_server.director.tools.spec import ToolSpec


class ReadTeamContextArguments(BaseModel):
    """Read who you are in the org and what concerns your team."""


_OLD_STR = (
    "For `str_replace`: the text now there, exactly as it stands and there only once."
)
_NEW_STR = (
    "For `str_replace`: what `old_str` is to read, empty to remove it. For "
    "`insert`: the text to insert."
)
_INSERT_LINE = (
    "For `insert`: the line `new_str` goes after; 0 is the top. Left out: at the end."
)


class MsgManagerArguments(BaseModel):
    """Write to your team's Manager."""

    text: str = Field(description="The message, in your own words.")


class ProposeTeamsArguments(BaseModel):
    """Propose sub-teams to your team's Manager."""

    teams: list[ProposedTeam] = Field(
        min_length=1,
        description=(
            "The sub-teams. Each owns one folder strictly inside what your "
            "team may write; no two may overlap. A team whose folder covers "
            "sub-teams you already have replaces them, and so does one that "
            "takes part of an existing sub-team's folder; what the new "
            "teams leave of it is yours again."
        ),
    )
    keep: list[str] = Field(
        default_factory=list,
        description=(
            "When you already have sub-teams: the name of every one that "
            "stays as it is. Leave empty when you have none."
        ),
    )
    text: str = Field(
        default="", description="Why you want these teams, for your Manager."
    )


class MsgIcArguments(BaseModel):
    """Write to an IC."""

    text: str = Field(description="The message, in your own words.")
    team: str = Field(
        default="",
        description=(
            "The id of a team directly below yours, to write to its "
            "IC. Leave empty for your own team's IC."
        ),
    )


class ApproveTeamsArguments(BaseModel):
    """Approve the sub-teams your IC proposed, as written."""


class SettleTeamsArguments(BaseModel):
    """Settle the sub-teams of a session that wait for you."""

    teams: list[TeamChoice] = Field(
        default_factory=list,
        description=(
            "For teams that wait: 'org' where what the org has now holds, "
            "'session' where the team becomes what the session made of it, "
            "which you may put under another team of the org or name "
            "otherwise. Unnamed, a team becomes what the session made of it."
        ),
    )
    note: str = Field(
        default="",
        description="Why, in a sentence or two; the session's IC is told.",
    )
    merge_choices: list[DocumentChoice] = Field(
        default_factory=list,
        description=(
            "For a document that waits with the teams: whether the org's "
            "stays, the session's holds, or a text you write from both does. "
            "Unnamed, the session's holds."
        ),
    )


class DeclineTeamsArguments(BaseModel):
    """Decline the sub-teams your IC proposed."""

    reason: str = Field(
        description=(
            "Why they are declined and what would change your mind; your IC "
            "is told, and it is kept in the record."
        )
    )


class RaiseEscalationArguments(BaseModel):
    """Raise something your team cannot settle itself."""

    category: EscalationKind | Literal["misrouted"] = Field(
        description=(
            "`critical_issue`: the work cannot continue. `design_decision`: a "
            "choice that reaches beyond your folder. `misrouted`: what the "
            "human sent your team directly is not your team's; it goes to "
            "the team above and you do not wait on it."
        )
    )
    title: str
    detail: str = Field(
        default="", description="What you need decided, and the options you see."
    )


class ResolveEscalationArguments(BaseModel):
    """Settle an escalation your team holds, or pass it up."""

    team: str = Field(description="The team that raised the escalation.")
    escalation_id: str
    action: Literal["settle", "pass_up"]
    text: str = Field(
        description=(
            "Your answer to the team that raised it, or the reason you cannot "
            "settle it."
        )
    )


class SubmitReviewArguments(BaseModel):
    """Submit your team's work for review."""

    summary: str = Field(description="What your team delivered.")


class ReviewTeamArguments(BaseModel):
    """Decide on the work a team below yours submitted."""

    team: str = Field(description="A team directly below yours.")
    decision: Literal["accept", "reject"]
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="What must change; required to reject.",
    )
    needed_above: str = Field(
        default="",
        description=(
            "Only when a team above yours waits on this work: which one and "
            "for what. Leave empty otherwise."
        ),
    )
    reconciled: str = Field(
        default="",
        description=(
            "Only when you were told this work meets other work that was "
            "merged since it began: what you decided, having read both. It "
            "holds for what you were shown: if that changed since, you are "
            "shown it again. Leave empty otherwise."
        ),
    )
    merge_choices: list[DocumentChoice] = Field(
        default_factory=list,
        description=(
            "Only when you accept the work of a session: for a document in "
            "it that the org also changed, whether the org's stays, the "
            "session's holds, or a text you write from both does. Unnamed, "
            "the session's holds."
        ),
    )


class MsgTeamArguments(BaseModel):
    """Write to the IC of a team below yours."""

    team: str = Field(description="The id of a team directly below yours.")
    text: str = Field(description="The message, in your own words.")


class SendDocumentArguments(BaseModel):
    """Pass a document of your team to another team, to read."""

    team: str = Field(description="The id of the team that is to read it.")
    document: str = Field(
        description="The name of the document, as your team's context lists it."
    )


class HandOverArguments(BaseModel):
    """Pass on to a team below yours what the teams before it left."""

    team: str = Field(description="The id of a team directly below yours.")
    text: str = Field(
        description=(
            "What the team should know of what the teams before it left, in "
            "your own words or quoted. Where a folder is now divided among "
            "several teams, the part that concerns this team's folder."
        )
    )


class StopTeamArguments(BaseModel):
    """Stop a team below yours that has gone on too long without progress."""

    team: str = Field(description="The id of a team directly below yours.")
    reason: str = Field(description="Why the team is stopped; your Manager is told.")


class PruneTeamArguments(BaseModel):
    """Remove a team below yours that is grossly inefficient."""

    team: str = Field(description="A team directly below yours.")
    reason: str = Field(description="Why the team is removed; its IC above is told.")


READ_TEAM_CONTEXT = ToolSpec(
    name="read_team_context",
    summary="Read your place in the org: your team, role and folder, your spec, "
    "the contracts you keep and what your team has reported, your sub-teams "
    "and their status, a split that "
    "is waiting, and the escalations you raised or must settle.",
    arguments=ReadTeamContextArguments,
    act=lambda service, caller, _action: agent_actions.node_context(service, caller),
    read_only=True,
)

MSG_MANAGER = ToolSpec(
    name="msg_manager",
    summary="IC only, with sub-teams. Write to your team's Manager, in your own "
    "words: say how the teams below are doing, say when one struggles "
    "and you cannot fix it, or when two of them are at odds. To propose "
    "sub-teams use propose_teams.",
    arguments=MsgManagerArguments,
    act=lambda service, caller, action: agent_actions.msg_manager(
        service, caller, action.text
    ),
)

PROPOSE_TEAMS = ToolSpec(
    name="propose_teams",
    summary="IC only. Propose sub-teams to your team's Manager. The proposal is "
    "checked at once, and nothing is set up until the Manager approves; "
    "you are told either way.",
    arguments=ProposeTeamsArguments,
    act=lambda service, caller, action: agent_actions.propose_teams(
        service,
        caller,
        action.teams,
        action.keep,
        action.text,
    ),
)

MSG_IC = ToolSpec(
    name="msg_ic",
    summary="Manager only. Write to your team's IC, or to the IC of a team "
    "directly below: feedback, a decision, what you are about to "
    "change. A message decides nothing: sub-teams your IC proposed "
    "still wait for approve_teams or decline_teams.",
    arguments=MsgIcArguments,
    act=lambda service, caller, action: agent_actions.msg_ic(
        service, caller, action.text, action.team
    ),
)

MSG_TEAM = ToolSpec(
    name="msg_team",
    summary="IC only. Write to the IC of a team directly below yours: its task, "
    "a hint, a correction, a change of priority. A team that was set up "
    "and not started yet starts with the first message, which is its "
    "task. Its IC does not answer in a message; it acts on what you "
    "wrote or raises an escalation. Writing to a team you stopped sets "
    "it going again.",
    arguments=MsgTeamArguments,
    act=lambda service, caller, action: agent_actions.msg_team(
        service, caller, action.team, action.text
    ),
)

SEND_DOCUMENT = ToolSpec(
    name="send_document",
    summary="Pass one of your team's documents to another team of your "
    "session to read from now on: it is told of it, and of every change to "
    "it, with its next message. Whether you may pass it to that team is "
    "your org's to say.",
    arguments=SendDocumentArguments,
    act=lambda service, caller, action: agent_actions.send_document(
        service, caller, action.team, action.document
    ),
)

STOP_TEAM = ToolSpec(
    name="stop_team",
    summary="IC only. A last resort, to be used very sparingly: stop a team "
    "directly below yours that has gone on too long without progress "
    "and that a message did not help. Its agents and those of every "
    "team below it end the step they are in and are sent nothing until "
    "you write to the team with msg_team. Your Manager is told and may "
    "remove the team.",
    arguments=StopTeamArguments,
    act=lambda service, caller, action: agent_actions.stop_team(
        service, caller, action.team, action.reason
    ),
)

APPROVE_TEAMS = ToolSpec(
    name="approve_teams",
    summary="Manager only. Approve the sub-teams your IC proposed. They are set "
    "up exactly as its message wrote them; you cannot change them here. "
    "To decline them use decline_teams.",
    arguments=ApproveTeamsArguments,
    act=lambda service, caller, _action: agent_actions.approve_teams(service, caller),
)

DECLINE_TEAMS = ToolSpec(
    name="decline_teams",
    summary="Manager only. Decline the sub-teams your IC proposed, saying why. "
    "Nothing is set up, your IC is told the reason, and the reason is "
    "kept in the record, where it is shown when the same folders "
    "are proposed again. Also the way to ask for a different division: "
    "say what to change, and your IC proposes again.",
    arguments=DeclineTeamsArguments,
    act=lambda service, caller, action: agent_actions.decline_teams(
        service, caller, action.reason
    ),
)

SETTLE_TEAMS = ToolSpec(
    name="settle_teams",
    summary="Division Lead Manager only. Settle the teams, and the documents "
    "that wait with them, that both a session and other accepted work "
    "changed: for each, the org's or the session's; a team taken from "
    "the session may go under another team or get another name; a "
    "document may be a text you write (`merge_choices`). It becomes the "
    "org's once the session's code is accepted and it is the session's "
    "turn.",
    arguments=SettleTeamsArguments,
    act=lambda service, caller, action: agent_actions.settle_teams(
        service,
        caller,
        action.teams,
        action.note,
        action.merge_choices,
    ),
)

RAISE_ESCALATION = ToolSpec(
    name="raise_escalation",
    summary="Raise something you cannot settle yourself. An IC's goes to the IC "
    "of the team above and a Manager's to the Manager there, which "
    "settles it or passes it up; it reaches the human only when no team "
    "could. A team is shown as blocked while its IC waits on one.",
    arguments=RaiseEscalationArguments,
    act=lambda service, caller, action: agent_actions.raise_escalation(
        service,
        caller,
        action.category,
        action.title,
        action.detail,
    ),
)

RESOLVE_ESCALATION = ToolSpec(
    name="resolve_escalation",
    summary="Settle an escalation that the agent of your own role in a team "
    "below raised and your team holds, with your answer; or pass it one "
    "team up with the reason you cannot settle it.",
    arguments=ResolveEscalationArguments,
    act=lambda service, caller, action: agent_actions.resolve_escalation(
        service,
        caller,
        action.team,
        action.escalation_id,
        action.action,
        action.text,
    ),
)

SUBMIT_REVIEW = ToolSpec(
    name="submit_review",
    summary="IC only. Submit your team's work when it is complete: commit it "
    "first, say what you delivered, then stop. The IC of the team above "
    "reviews it and you are woken with the outcome. Work of your "
    "sub-teams that you accepted goes with it.",
    arguments=SubmitReviewArguments,
    act=lambda service, caller, action: agent_actions.submit_review(
        service, caller, action.summary
    ),
)

REVIEW_TEAM = ToolSpec(
    name="review_team",
    summary="IC only. Accept the work a team directly below yours submitted, or "
    "reject it with what must change. Accepted work of a team with its "
    "own worktree is merged into your checkout. If your team changed "
    "nothing itself, accepted work goes on up without another review. "
    "Work is accepted in the order it was submitted. Work that changed "
    "files which also changed in your checkout since it began is not "
    "merged until you say in `reconciled` what you decided about them. "
    "Work that does not merge by itself is laid out for you to merge. "
    "Of a session's work you may decide each document the org also "
    "changed in `merge_choices`.",
    arguments=ReviewTeamArguments,
    act=lambda service, caller, action: agent_actions.review_team(
        service,
        caller,
        action.team,
        action.decision,
        action.findings,
        action.needed_above,
        action.reconciled,
        action.merge_choices,
    ),
)

PRUNE_TEAM = ToolSpec(
    name="prune_team",
    summary="Manager only. Remove a team directly below yours that is grossly "
    "inefficient, with every team below it. Rare, and never a "
    "precaution: wait until the team has had enough tasks to judge it "
    "by. Its agents are deleted and the uncommitted changes in its "
    "folder are discarded (git keeps a copy). The folder goes back "
    "to your team's IC, who is told why.",
    arguments=PruneTeamArguments,
    act=lambda service, caller, action: agent_actions.prune_team(
        service,
        caller,
        action.team,
        action.reason,
    ),
)

HAND_OVER = ToolSpec(
    name="hand_over",
    summary="IC only. After teams below yours were replaced or pruned, pass on "
    "to a team directly below what it should know of what they left. "
    "Nothing of a removed team reaches another team unless you do "
    "this. The team is told with its next message and records what "
    "should last itself. It starts nothing and can be done again.",
    arguments=HandOverArguments,
    act=lambda service, caller, action: agent_actions.hand_over(
        service, caller, action.team, action.text
    ),
)


# Every tool of the director's own, by name.
DIRECTOR_TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        READ_TEAM_CONTEXT,
        MSG_MANAGER,
        PROPOSE_TEAMS,
        MSG_IC,
        MSG_TEAM,
        SEND_DOCUMENT,
        STOP_TEAM,
        APPROVE_TEAMS,
        DECLINE_TEAMS,
        SETTLE_TEAMS,
        RAISE_ESCALATION,
        RESOLVE_ESCALATION,
        SUBMIT_REVIEW,
        REVIEW_TEAM,
        PRUNE_TEAM,
        HAND_OVER,
    )
}
