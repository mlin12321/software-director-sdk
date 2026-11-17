"""Actions and tool definitions of the director's agent tools."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from pydantic import Field

from openhands.agent_server.director import agent_actions
from openhands.agent_server.director.agent_actions import Caller
from openhands.agent_server.director.models import (
    ContractsChoice,
    EscalationKind,
    ReviewFinding,
    SpecChoice,
    TeamChoice,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.split import ProposedTeam
from openhands.agent_server.director.tools.impl import (
    DirectorExecutor,
    DirectorObservation,
)
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import Action, ToolAnnotations, ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class ReadTeamContextAction(Action):
    """Read who you are in the org and what concerns your team."""


class WriteSpecAction(Action):
    """Replace the spec of your team's folder."""

    text: str = Field(description="The whole spec, in Markdown.")


class MsgManagerAction(Action):
    """Write to your team's Manager."""

    text: str = Field(description="The message, in your own words.")


class ProposeTeamsAction(Action):
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
    contracts: str = Field(
        default="",
        description=(
            "What the sub-teams must keep between them: types, endpoints, "
            "file formats. They cannot see each other's work or your spec, "
            "so this must hold everything they have to respect. What teams "
            "promise each other is written only here, not in a spec."
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


class SettleTeamsAction(Action):
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
    contracts: list[ContractsChoice] = Field(
        default_factory=list,
        description=(
            "For contracts that wait: which holds for each directory. "
            "Unnamed, the session's hold."
        ),
    )
    note: str = Field(
        default="",
        description="Why, in a sentence or two; the session's IC is told.",
    )


class DeclineTeamsAction(Action):
    """Decline the sub-teams your IC proposed."""

    reason: str = Field(
        description=(
            "Why they are declined and what would change your mind; your IC "
            "is told, and it is kept in the record."
        )
    )


class RaiseEscalationAction(Action):
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
    team_for: str = Field(
        default="",
        description=(
            "For an IC that needs something of a team that is not above it: "
            "what it needs, of which files or of which part of the product. "
            "The runtime then takes the escalation straight to the IC of the "
            "team that holds it, when it is sure which team that is, and "
            "otherwise to the IC of the team above as without this. Leave "
            "empty for what the team above must decide."
        ),
    )


class ResolveEscalationAction(Action):
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


class SubmitReviewAction(Action):
    """Submit your team's work for review."""

    summary: str = Field(description="What your team delivered.")


class ReviewTeamAction(Action):
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
    specs: list[SpecChoice] = Field(
        default_factory=list,
        description=(
            "Only when you accept the work of a session: for the spec of a "
            "directory in it, whether the org's stays, the session's holds, "
            "or a text you write from both does. Unnamed, the session's "
            "holds."
        ),
    )


class ReportAction(Action):
    """Record a design decision, a setback or a reflection."""

    text: str = Field(description="What to record, in plain English.")


class MsgTeamAction(Action):
    """Write to the IC of a team below yours."""

    team: str = Field(description="The node id of a team directly below yours.")
    text: str = Field(description="The message, in your own words.")


class HandOverAction(Action):
    """Pass on to a team below yours what the teams before it left."""

    team: str = Field(description="The node id of a team directly below yours.")
    notes: str = Field(
        default="",
        description=(
            "What of the removed teams' reports this team should know, in "
            "your own words or quoted. Kept with its reports."
        ),
    )
    contracts: str = Field(
        default="",
        description=(
            "The contracts this team's own sub-teams must keep between "
            "them. Takes the place of the ones it has. Only for a team that "
            "has, or took over, sub-teams."
        ),
    )


class StopTeamAction(Action):
    """Stop a team below yours that has gone on too long without progress."""

    team: str = Field(description="The node id of a team directly below yours.")
    reason: str = Field(description="Why the team is stopped; your Manager is told.")


class PruneTeamAction(Action):
    """Remove a team below yours that is grossly inefficient."""

    team: str = Field(description="A team directly below yours.")
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


class ReadTeamContextTool(_DirectorTool[ReadTeamContextAction]):
    summary = (
        "Read your place in the org: your team, role and folder, your spec, "
        "the contracts you keep and what your team has reported, your sub-teams "
        "and their status, a split that "
        "is waiting, and the escalations you raised or must settle."
    )
    read_only = True

    @classmethod
    def action(cls):
        return ReadTeamContextAction

    @classmethod
    def act(cls):
        return lambda service, caller, _action: agent_actions.node_context(
            service, caller
        )


class WriteSpecTool(_DirectorTool[WriteSpecAction]):
    summary = (
        "IC only. Replace the spec of your folder: how your team works "
        "inside it. It is kept outside the working tree and is your team's "
        "own; sub-teams see only the contracts of the split that made them. "
        "What teams promise each other goes in contracts, not here."
    )

    @classmethod
    def action(cls):
        return WriteSpecAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.write_spec(
            service, caller, action.text
        )


class MsgManagerTool(_DirectorTool[MsgManagerAction]):
    summary = (
        "IC only, with sub-teams. Write to your team's Manager, in your own "
        "words: say how the teams below are doing, say when one struggles "
        "and you cannot fix it, or when two of them are at odds. To propose "
        "sub-teams use propose_teams."
    )

    @classmethod
    def action(cls):
        return MsgManagerAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.msg_manager(
            service, caller, action.text
        )


class ProposeTeamsTool(_DirectorTool[ProposeTeamsAction]):
    summary = (
        "IC only. Propose sub-teams to your team's Manager. The proposal is "
        "checked at once, and nothing is set up until the Manager approves; "
        "you are told either way."
    )

    @classmethod
    def action(cls):
        return ProposeTeamsAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.propose_teams(
            service,
            caller,
            action.teams,
            action.contracts,
            action.keep,
            action.text,
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
        return lambda service, caller, action: agent_actions.msg_ic(
            service, caller, action.text, action.team
        )


class MsgTeamTool(_DirectorTool[MsgTeamAction]):
    summary = (
        "IC only. Write to the IC of a team directly below yours: its task, "
        "a hint, a correction, a change of priority. A team that was set up "
        "and not started yet starts with the first message, which is its "
        "task. Its IC does not answer in a message; it acts on what you "
        "wrote or raises an escalation. Writing to a team you stopped sets "
        "it going again."
    )

    @classmethod
    def action(cls):
        return MsgTeamAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.msg_team(
            service, caller, action.team, action.text
        )


class HandOverTool(_DirectorTool[HandOverAction]):
    summary = (
        "IC only. After teams below yours were replaced or pruned, pass on "
        "to a team directly below what it should know of what they left: "
        "notes from their reports, and the contracts its own sub-teams must "
        "keep. Nothing of a removed team reaches another team unless you do "
        "this. It starts nothing and can be done again."
    )

    @classmethod
    def action(cls):
        return HandOverAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.hand_over(
            service, caller, action.team, action.notes, action.contracts
        )


class StopTeamTool(_DirectorTool[StopTeamAction]):
    summary = (
        "IC only. A last resort, to be used very sparingly: stop a team "
        "directly below yours that has gone on too long without progress "
        "and that a message did not help. Its agents and those of every "
        "team below it end the step they are in and are sent nothing until "
        "you write to the team with msg_team. Your Manager is told and may "
        "remove the team."
    )

    @classmethod
    def action(cls):
        return StopTeamAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.stop_team(
            service, caller, action.team, action.reason
        )


class ApproveTeamsTool(_DirectorTool[ApproveTeamsAction]):
    summary = (
        "Manager only. Approve the sub-teams your IC proposed. They are set "
        "up exactly as its message wrote them; you cannot change them here. "
        "To decline them use decline_teams."
    )

    @classmethod
    def action(cls):
        return ApproveTeamsAction

    @classmethod
    def act(cls):
        return lambda service, caller, _action: agent_actions.approve_teams(
            service, caller
        )


class SettleTeamsTool(_DirectorTool[SettleTeamsAction]):
    summary = (
        "Division Lead Manager only. Settle the teams and contracts that "
        "both a session and other accepted work changed: for each, the "
        "org's or the session's; a team taken from the session may go "
        "under another team or get another name; contracts may be a text "
        "you write. It becomes the org's once the session's code is "
        "accepted and it is the session's turn."
    )

    @classmethod
    def action(cls):
        return SettleTeamsAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.settle_teams(
            service, caller, action.teams, action.note, action.contracts
        )


class DeclineTeamsTool(_DirectorTool[DeclineTeamsAction]):
    summary = (
        "Manager only. Decline the sub-teams your IC proposed, saying why. "
        "Nothing is set up, your IC is told the reason, and the reason is "
        "kept in the record, where it is shown when the same directories "
        "are proposed again. Also the way to ask for a different division: "
        "say what to change, and your IC proposes again."
    )

    @classmethod
    def action(cls):
        return DeclineTeamsAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.decline_teams(
            service, caller, action.reason
        )


class RaiseEscalationTool(_DirectorTool[RaiseEscalationAction]):
    summary = (
        "Raise something you cannot settle yourself. An IC's goes to the IC "
        "of the team above and a Manager's to the Manager there, which "
        "settles it or passes it up; it reaches the human only when no team "
        "could. A team is shown as blocked while its IC waits on one."
    )

    @classmethod
    def action(cls):
        return RaiseEscalationAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.raise_escalation(
            service,
            caller,
            action.category,
            action.title,
            action.detail,
            action.team_for,
        )


class ResolveEscalationTool(_DirectorTool[ResolveEscalationAction]):
    summary = (
        "Settle an escalation that the agent of your own role in a team "
        "below raised and your team holds, with your answer; or pass it one "
        "team up with the reason you cannot settle it."
    )

    @classmethod
    def action(cls):
        return ResolveEscalationAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.resolve_escalation(
            service,
            caller,
            action.team,
            action.escalation_id,
            action.action,
            action.text,
        )


class SubmitReviewTool(_DirectorTool[SubmitReviewAction]):
    summary = (
        "IC only. Submit your team's work when it is complete: commit it "
        "first, say what you delivered, then stop. The IC of the team above "
        "reviews it and you are woken with the outcome. Work of your "
        "sub-teams that you accepted goes with it."
    )

    @classmethod
    def action(cls):
        return SubmitReviewAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.submit_review(
            service, caller, action.summary
        )


class ReviewTeamTool(_DirectorTool[ReviewTeamAction]):
    summary = (
        "IC only. Accept the work a team directly below yours submitted, or "
        "reject it with what must change. Accepted work of a team with its "
        "own worktree is merged into your checkout. If your team changed "
        "nothing itself, accepted work goes on up without another review. "
        "Work is accepted in the order it was submitted. Work that changed "
        "files which also changed in your checkout since it began is not "
        "merged until you say in `reconciled` what you decided about them. "
        "Work that does not merge by itself is laid out for you to merge. "
        "Of a session's work you may decide each spec in `specs`."
    )

    @classmethod
    def action(cls):
        return ReviewTeamAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.review_team(
            service,
            caller,
            action.team,
            action.decision,
            action.findings,
            action.needed_above,
            action.reconciled,
            action.specs,
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
        return lambda service, caller, action: agent_actions.report(
            service, caller, action.text
        )


class PruneTeamTool(_DirectorTool[PruneTeamAction]):
    summary = (
        "Manager only. Remove a team directly below yours that is grossly "
        "inefficient, with every team below it. Rare, and never a "
        "precaution: wait until the team has had enough tasks to judge it "
        "by. Its agents are deleted and the uncommitted changes in its "
        "directory are discarded (git keeps a copy). The directory goes back "
        "to your team's IC, who is told why."
    )

    @classmethod
    def action(cls):
        return PruneTeamAction

    @classmethod
    def act(cls):
        return lambda service, caller, action: agent_actions.prune_team(
            service,
            caller,
            action.team,
            action.reason,
        )


DIRECTOR_TOOLS: dict[str, type[_DirectorTool]] = {
    tool.name: tool
    for tool in (
        ReadTeamContextTool,
        WriteSpecTool,
        MsgManagerTool,
        ProposeTeamsTool,
        MsgIcTool,
        MsgTeamTool,
        StopTeamTool,
        ApproveTeamsTool,
        DeclineTeamsTool,
        SettleTeamsTool,
        RaiseEscalationTool,
        ResolveEscalationTool,
        SubmitReviewTool,
        ReviewTeamTool,
        ReportTool,
        PruneTeamTool,
        HandOverTool,
    )
}

for _tool in DIRECTOR_TOOLS.values():
    register_tool(_tool.name, _tool)
