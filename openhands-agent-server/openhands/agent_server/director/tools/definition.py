"""Actions and tool definitions of the director's agent tools."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import Field

from openhands.agent_server.director import pod_actions
from openhands.agent_server.director.models import (
    EscalationKind,
    ReviewFinding,
    SplitChild,
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


class ProposeBoundarySplitAction(Action):
    """Propose to hand parts of your boundary to sub-teams."""

    children: list[SplitChild] = Field(
        description=(
            "The sub-teams. Each owns one directory strictly inside your "
            "boundary; no two may overlap. `isolation` is `shared` to work in "
            "your directory or `worktree` for an own git worktree and branch."
        )
    )
    contracts: str = Field(
        description=(
            "The interfaces the sub-teams must keep between them: types, "
            "endpoints, file formats. They cannot see each other's work or "
            "your spec, so include all of it they must respect."
        )
    )


class DecideSplitAction(Action):
    """Decide the split your IC proposed."""

    decision: Literal["approve", "refine"]
    feedback: str = Field(
        default="", description="What to change; required when asking to refine."
    )


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


class PruneTeamAction(Action):
    """Remove a team below yours that is stuck."""

    node_id: str = Field(description="A team directly below yours.")
    approach: str = Field(description="What the team tried.")
    why_failed: str = Field(
        description="Why it did not work, for whoever works there next."
    )
    replacement_task: str = Field(
        default="",
        description=(
            "The task of a fresh team that takes over the directory. Leave "
            "empty to give the directory back to your own team."
        ),
    )


class _DirectorTool[ActionT: Action](ToolDefinition[ActionT, DirectorObservation]):
    """A tool bound to the conversation of the agent that owns it."""

    # Given by the director to the agents it starts, not picked from the catalog.
    user_selectable: ClassVar[bool] = False
    summary: ClassVar[str]
    read_only: ClassVar[bool] = False

    @classmethod
    def _action_type(cls) -> type[ActionT]:
        raise NotImplementedError

    @classmethod
    def _act(cls) -> Callable[[DirectorService, Caller, ActionT], str]:
        raise NotImplementedError

    @classmethod
    def create(cls, conv_state: "ConversationState", **params) -> Sequence[Self]:
        if params:
            raise ValueError(f"{cls.__name__} does not accept parameters")
        return [
            cls(
                description=cls.summary,
                action_type=cls._action_type(),
                observation_type=DirectorObservation,
                executor=DirectorExecutor(conv_state.id, cls._act()),
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
        "the contracts you keep and the rules you inherit, your sub-teams "
        "and their status, a split that "
        "is waiting, and the escalations you raised or must settle."
    )
    read_only = True

    @classmethod
    def _action_type(cls):
        return ReadNodeContextAction

    @classmethod
    def _act(cls):
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
    def _action_type(cls):
        return WriteSpecAction

    @classmethod
    def _act(cls):
        return lambda service, caller, action: pod_actions.write_spec(
            service, caller, action.content
        )


class ProposeBoundarySplitTool(_DirectorTool[ProposeBoundarySplitAction]):
    summary = (
        "IC only. Propose to hand directories of your boundary to sub-teams, "
        "each with its own IC and Manager, when the work is too much for one "
        "agent. Your Manager decides. Nothing starts until it approves."
    )

    @classmethod
    def _action_type(cls):
        return ProposeBoundarySplitAction

    @classmethod
    def _act(cls):
        return lambda service, caller, action: pod_actions.propose_split(
            service, caller, action.children, action.contracts
        )


class DecideSplitTool(_DirectorTool[DecideSplitAction]):
    summary = (
        "Manager only. Approve the split your IC proposed, which starts the "
        "sub-teams, or ask the IC to refine it and say what to change."
    )

    @classmethod
    def _action_type(cls):
        return DecideSplitAction

    @classmethod
    def _act(cls):
        return lambda service, caller, action: pod_actions.decide_split(
            service, caller, action.decision, action.feedback
        )


class RaiseEscalationTool(_DirectorTool[RaiseEscalationAction]):
    summary = (
        "Raise something your team cannot settle itself. It goes to the "
        "Manager of the team above, which settles it or passes it up; it "
        "reaches the human only when no team could."
    )

    @classmethod
    def _action_type(cls):
        return RaiseEscalationAction

    @classmethod
    def _act(cls):
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
    def _action_type(cls):
        return ResolveEscalationAction

    @classmethod
    def _act(cls):
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
    def _action_type(cls):
        return SetStatusAction

    @classmethod
    def _act(cls):
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
    def _action_type(cls):
        return ReviewTeamAction

    @classmethod
    def _act(cls):
        return lambda service, caller, action: pod_actions.review_team(
            service,
            caller,
            action.node_id,
            action.decision,
            action.findings,
            action.needed_above,
        )


class PruneTeamTool(_DirectorTool[PruneTeamAction]):
    summary = (
        "Manager only. Remove a team directly below yours that is stuck or "
        "keeps failing, with every team below it. Its agents are deleted and "
        "the uncommitted changes in its directory are discarded (git keeps a "
        "copy). What you say it tried and why it failed is kept for whoever "
        "works there next."
    )

    @classmethod
    def _action_type(cls):
        return PruneTeamAction

    @classmethod
    def _act(cls):
        return lambda service, caller, action: pod_actions.prune_team(
            service,
            caller,
            action.node_id,
            action.approach,
            action.why_failed,
            action.replacement_task,
        )


register_tool(ReadNodeContextTool.name, ReadNodeContextTool)
register_tool(WriteSpecTool.name, WriteSpecTool)
register_tool(ProposeBoundarySplitTool.name, ProposeBoundarySplitTool)
register_tool(DecideSplitTool.name, DecideSplitTool)
register_tool(RaiseEscalationTool.name, RaiseEscalationTool)
register_tool(ResolveEscalationTool.name, ResolveEscalationTool)
register_tool(SetStatusTool.name, SetStatusTool)
register_tool(PruneTeamTool.name, PruneTeamTool)
register_tool(ReviewTeamTool.name, ReviewTeamTool)
