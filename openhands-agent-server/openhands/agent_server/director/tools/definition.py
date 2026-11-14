"""Actions and tool definitions of the director's agent tools."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import Field

from openhands.agent_server.director import pod_actions
from openhands.agent_server.director.models import (
    EscalationKind,
    ReviewFinding,
)
from openhands.agent_server.director.pod_actions import Caller, DeclaredStatus
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.tools.impl import (
    DirectorExecutor,
    DirectorObservation,
)
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import Action, ToolAnnotations, ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class ReadNodeContextAction(Action):
    """Read who you are in the org and what concerns your team."""


class WriteSpecAction(Action):
    """Replace the spec of your team's boundary."""

    content: str = Field(description="The whole spec, in Markdown.")


TEAMS_EXAMPLE = """\
```teams
contracts: |
  What the sub-teams must keep between them: types, endpoints, file formats.
teams:
  - name: API
    path: services/api
    instruction: What this team is to deliver.
    model: a-profile-name      # optional
    writable: [services/api/**]  # optional; all of path when left out
    readonly: [shared/**]        # optional
    isolation: shared            # optional; or worktree
```"""


class MsgManagerAction(Action):
    """Write to your team's Manager."""

    text: str = Field(
        description=(
            "The message, in your own words. To propose sub-teams, include "
            "one block like this:\n"
            f"{TEAMS_EXAMPLE}\n"
            "Each team owns one directory strictly inside what your team "
            "may write; no two may overlap. `model` is one of the profiles "
            "read_node_context lists. A team whose directory covers "
            "sub-teams you already have replaces them, and so does one that "
            "takes part of an existing sub-team's directory; what the new "
            "teams leave of it is yours again. The sub-teams cannot see "
            "each other's work or your spec, so the contracts must hold "
            "everything they have to respect."
        )
    )


class MsgIcAction(Action):
    """Write to an IC."""

    text: str = Field(description="The message, in your own words.")
    team: str = Field(
        default="",
        description=(
            "The node id of a team directly below yours, to write to its "
            "IC. Leave empty for your own team's IC."
        ),
    )


class ApproveTeamsAction(Action):
    """Approve the sub-teams your IC proposed, as written."""


class RaiseEscalationAction(Action):
    """Raise something your team cannot settle itself."""

    category: EscalationKind = Field(
        description=(
            "`critical_issue`: the work cannot continue. `design_decision`: a "
            "choice that reaches beyond your boundary."
        )
    )
    title: str
    detail: str = Field(
        default="", description="What you need decided, and the options you see."
    )


class ResolveEscalationAction(Action):
    """Settle an escalation your team holds, or pass it up."""

    node_id: str = Field(description="The node that raised the escalation.")
    escalation_id: str
    action: Literal["settle", "pass_up"]
    text: str = Field(
        description=(
            "Your answer to the team that raised it, or the reason you cannot "
            "settle it."
        )
    )


class SetStatusAction(Action):
    """Declare whether your team is working, done or blocked."""

    status: DeclaredStatus
    summary: str = Field(
        default="", description="What was delivered, or what blocks the team."
    )


class ReviewTeamAction(Action):
    """Decide on the work a team below yours reported done."""

    node_id: str = Field(description="A team directly below yours.")
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


class ReportAction(Action):
    """Record a design decision, a setback or a reflection."""

    text: str = Field(description="What to record, in plain English.")


class PruneTeamAction(Action):
    """Remove a team below yours that is stuck."""

    node_id: str = Field(description="A team directly below yours.")
    reason: str = Field(description="Why the team is removed; its IC above is told.")


class _DirectorTool[ActionT: Action](ToolDefinition[ActionT, DirectorObservation]):
    """A tool bound to the conversation of the agent that owns it."""

    # Given by the director to the agents it starts, not picked from the catalog.
    user_selectable: ClassVar[bool] = False
    summary: ClassVar[str]
    read_only: ClassVar[bool] = False

    @classmethod
    def action(cls) -> type[ActionT]:
        raise NotImplementedError

    @classmethod
    def act(cls) -> Callable[[DirectorService, Caller, ActionT], str]:
        raise NotImplementedError

    @classmethod
    def create(cls, conv_state: "ConversationState", **params) -> Sequence[Self]:
        if params:
            raise ValueError(f"{cls.__name__} does not accept parameters")
        return [
            cls(
                description=cls.summary,
                action_type=cls.action(),
                observation_type=DirectorObservation,
                executor=DirectorExecutor(conv_state.id, cls.act()),
                annotations=ToolAnnotations(
                    readOnlyHint=cls.read_only,
                    destructiveHint=False,
                    idempotentHint=cls.read_only,
                    openWorldHint=False,
                ),
            )
        ]


class ReadNodeContextTool(_DirectorTool[ReadNodeContextAction]):
    summary = (
        "Read your place in the org: your team, role and boundary, your spec, "
        "the contracts you keep and what your team has reported, your sub-teams "
        "and their status, a split that "
        "is waiting, and the escalations you raised or must settle."
    )
    read_only = True

    @classmethod
    def action(cls):
        return ReadNodeContextAction

    @classmethod
    def act(cls):
        return lambda service, caller, _action: pod_actions.node_context(
            service, caller
        )


class WriteSpecTool(_DirectorTool[WriteSpecAction]):
    summary = (
        "IC only. Replace the spec of your boundary. It is kept outside the "
        "working tree and is your team's own: sub-teams do not see it, only "
        "the contracts of the split that made them."
    )

    @classmethod
    def action(cls):
        return WriteSpecAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.write_spec(
            service, caller, action.content
        )


class MsgManagerTool(_DirectorTool[MsgManagerAction]):
    summary = (
        "IC only. Write to your team's Manager, in your own words: propose "
        "sub-teams, say how the teams below are doing, say when one "
        "struggles and you cannot fix it, or when two of them are at odds. "
        "To propose sub-teams, put one ```teams block of YAML in the "
        "message; it is checked at once and nothing starts until the "
        "Manager approves. A team without sub-teams uses this only to "
        "propose some."
    )

    @classmethod
    def action(cls):
        return MsgManagerAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.msg_manager(
            service, caller, action.text
        )


class MsgIcTool(_DirectorTool[MsgIcAction]):
    summary = (
        "Manager only. Write to your team's IC, or to the IC of a team "
        "directly below: feedback, a decision, what you are about to "
        "change. Answering your IC while its proposed sub-teams wait puts "
        "that proposal aside."
    )

    @classmethod
    def action(cls):
        return MsgIcAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.msg_ic(
            service, caller, action.text, action.team
        )


class ApproveTeamsTool(_DirectorTool[ApproveTeamsAction]):
    summary = (
        "Manager only. Approve the sub-teams your IC proposed. They are set "
        "up exactly as its message wrote them; you cannot change them here. "
        "To ask for changes or decline, answer with msg_ic instead."
    )

    @classmethod
    def action(cls):
        return ApproveTeamsAction

    @classmethod
    def act(cls):
        return lambda service, caller, _action: pod_actions.approve_teams(
            service, caller
        )


class RaiseEscalationTool(_DirectorTool[RaiseEscalationAction]):
    summary = (
        "Raise something your team cannot settle itself. It goes to the "
        "Manager of the team above, which settles it or passes it up; it "
        "reaches the human only when no team could."
    )

    @classmethod
    def action(cls):
        return RaiseEscalationAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.raise_escalation(
            service, caller, action.category, action.title, action.detail
        )


class ResolveEscalationTool(_DirectorTool[ResolveEscalationAction]):
    summary = (
        "Manager only. Settle an escalation a team below raised and your "
        "team holds, with your answer; or pass it one team up with the reason "
        "you cannot settle it."
    )

    @classmethod
    def action(cls):
        return ResolveEscalationAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.resolve_escalation(
            service,
            caller,
            action.node_id,
            action.escalation_id,
            action.action,
            action.text,
        )


class SetStatusTool(_DirectorTool[SetStatusAction]):
    summary = (
        "Declare that your team is done or blocked, or working again. Commit "
        "the work first: done work is reviewed by the IC of the team above. "
        "The Manager of the team above is told of a block."
    )

    @classmethod
    def action(cls):
        return SetStatusAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.set_status(
            service, caller, action.status, action.summary
        )


class ReviewTeamTool(_DirectorTool[ReviewTeamAction]):
    summary = (
        "IC only. Accept the work a team directly below yours reported done, "
        "or reject it with what must change. Accepted work of a team with its "
        "own worktree is merged into your checkout. A team whose work is "
        "rejected a third time is stopped and left blocked."
    )

    @classmethod
    def action(cls):
        return ReviewTeamAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.review_team(
            service,
            caller,
            action.node_id,
            action.decision,
            action.findings,
            action.needed_above,
        )


class ReportTool(_DirectorTool[ReportAction]):
    summary = (
        "Record something worth keeping: a design decision and its reason, "
        "a setback, an approach that failed and why, a reflection. Only you "
        "and the other agent of your team read it. Reports are only added "
        "to; once they grow long they are rewritten shorter for you."
    )

    @classmethod
    def action(cls):
        return ReportAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.report(
            service, caller, action.text
        )


class PruneTeamTool(_DirectorTool[PruneTeamAction]):
    summary = (
        "Manager only. Remove a team directly below yours that is stuck or "
        "keeps failing, with every team below it. Its agents are deleted and "
        "the uncommitted changes in its directory are discarded (git keeps a "
        "copy). The directory goes back to your team's IC, who is told why."
    )

    @classmethod
    def action(cls):
        return PruneTeamAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: pod_actions.prune_team(
            service,
            caller,
            action.node_id,
            action.reason,
        )


register_tool(ReadNodeContextTool.name, ReadNodeContextTool)
register_tool(WriteSpecTool.name, WriteSpecTool)
register_tool(MsgManagerTool.name, MsgManagerTool)
register_tool(MsgIcTool.name, MsgIcTool)
register_tool(ApproveTeamsTool.name, ApproveTeamsTool)
register_tool(RaiseEscalationTool.name, RaiseEscalationTool)
register_tool(ResolveEscalationTool.name, ResolveEscalationTool)
register_tool(SetStatusTool.name, SetStatusTool)
register_tool(PruneTeamTool.name, PruneTeamTool)
register_tool(ReviewTeamTool.name, ReviewTeamTool)
register_tool(ReportTool.name, ReportTool)


DIRECTOR_TOOLS: dict[str, type[_DirectorTool]] = {
    tool.name: tool
    for tool in (
        ReadNodeContextTool,
        WriteSpecTool,
        MsgManagerTool,
        MsgIcTool,
        ApproveTeamsTool,
        RaiseEscalationTool,
        ResolveEscalationTool,
        SetStatusTool,
        ReviewTeamTool,
        ReportTool,
        PruneTeamTool,
    )
}
