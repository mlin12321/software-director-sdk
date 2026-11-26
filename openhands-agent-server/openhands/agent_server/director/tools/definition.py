"""The director's tools as an agent of the server is given them.

What each tool takes and does is in ``catalog``; this is the one module
that makes tools of the agent SDK out of them.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from openhands.agent_server.director.tools import catalog
from openhands.agent_server.director.tools.impl import (
    DirectorExecutor,
    DirectorObservation,
)
from openhands.agent_server.director.tools.spec import ToolSpec
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import Action, ToolAnnotations, ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class _DirectorTool[ActionT: Action](ToolDefinition[ActionT, DirectorObservation]):
    """A tool bound to the conversation of the agent that owns it."""

    # Given by the director to the agents it starts, not picked from the catalog.
    user_selectable: ClassVar[bool] = False
    spec: ClassVar[ToolSpec]

    @classmethod
    def action(cls) -> type[ActionT]:
        """What the tool takes, as an action of the agent SDK."""
        raise NotImplementedError

    @classmethod
    def create(cls, conv_state: "ConversationState", **params) -> Sequence[Self]:
        if params:
            raise ValueError(f"{cls.__name__} does not accept parameters")
        return [
            cls(
                description=cls.spec.summary,
                action_type=cls.action(),
                observation_type=DirectorObservation,
                executor=DirectorExecutor(conv_state.id, cls.spec.act),
                annotations=ToolAnnotations(
                    readOnlyHint=cls.spec.read_only,
                    destructiveHint=False,
                    idempotentHint=cls.spec.read_only,
                    openWorldHint=False,
                ),
            )
        ]


register_tool(_DirectorTool.name, _DirectorTool)


class ReadTeamContextAction(catalog.ReadTeamContextArguments, Action):
    __doc__ = catalog.ReadTeamContextArguments.__doc__


class ReadTeamContextTool(_DirectorTool[ReadTeamContextAction]):
    spec = catalog.READ_TEAM_CONTEXT

    @classmethod
    def action(cls):
        return ReadTeamContextAction


register_tool(ReadTeamContextTool.name, ReadTeamContextTool)


class WriteSpecAction(catalog.WriteSpecArguments, Action):
    __doc__ = catalog.WriteSpecArguments.__doc__


class WriteSpecTool(_DirectorTool[WriteSpecAction]):
    spec = catalog.WRITE_SPEC

    @classmethod
    def action(cls):
        return WriteSpecAction


register_tool(WriteSpecTool.name, WriteSpecTool)


class EditContractsAction(catalog.EditContractsArguments, Action):
    __doc__ = catalog.EditContractsArguments.__doc__


class EditContractsTool(_DirectorTool[EditContractsAction]):
    spec = catalog.EDIT_CONTRACTS

    @classmethod
    def action(cls):
        return EditContractsAction


register_tool(EditContractsTool.name, EditContractsTool)


class MsgManagerAction(catalog.MsgManagerArguments, Action):
    __doc__ = catalog.MsgManagerArguments.__doc__


class MsgManagerTool(_DirectorTool[MsgManagerAction]):
    spec = catalog.MSG_MANAGER

    @classmethod
    def action(cls):
        return MsgManagerAction


register_tool(MsgManagerTool.name, MsgManagerTool)


class ProposeTeamsAction(catalog.ProposeTeamsArguments, Action):
    __doc__ = catalog.ProposeTeamsArguments.__doc__


class ProposeTeamsTool(_DirectorTool[ProposeTeamsAction]):
    spec = catalog.PROPOSE_TEAMS

    @classmethod
    def action(cls):
        return ProposeTeamsAction


register_tool(ProposeTeamsTool.name, ProposeTeamsTool)


class MsgIcAction(catalog.MsgIcArguments, Action):
    __doc__ = catalog.MsgIcArguments.__doc__


class MsgIcTool(_DirectorTool[MsgIcAction]):
    spec = catalog.MSG_IC

    @classmethod
    def action(cls):
        return MsgIcAction


register_tool(MsgIcTool.name, MsgIcTool)


class MsgTeamAction(catalog.MsgTeamArguments, Action):
    __doc__ = catalog.MsgTeamArguments.__doc__


class MsgTeamTool(_DirectorTool[MsgTeamAction]):
    spec = catalog.MSG_TEAM

    @classmethod
    def action(cls):
        return MsgTeamAction


register_tool(MsgTeamTool.name, MsgTeamTool)


class StopTeamAction(catalog.StopTeamArguments, Action):
    __doc__ = catalog.StopTeamArguments.__doc__


class StopTeamTool(_DirectorTool[StopTeamAction]):
    spec = catalog.STOP_TEAM

    @classmethod
    def action(cls):
        return StopTeamAction


register_tool(StopTeamTool.name, StopTeamTool)


class ApproveTeamsAction(catalog.ApproveTeamsArguments, Action):
    __doc__ = catalog.ApproveTeamsArguments.__doc__


class ApproveTeamsTool(_DirectorTool[ApproveTeamsAction]):
    spec = catalog.APPROVE_TEAMS

    @classmethod
    def action(cls):
        return ApproveTeamsAction


register_tool(ApproveTeamsTool.name, ApproveTeamsTool)


class DeclineTeamsAction(catalog.DeclineTeamsArguments, Action):
    __doc__ = catalog.DeclineTeamsArguments.__doc__


class DeclineTeamsTool(_DirectorTool[DeclineTeamsAction]):
    spec = catalog.DECLINE_TEAMS

    @classmethod
    def action(cls):
        return DeclineTeamsAction


register_tool(DeclineTeamsTool.name, DeclineTeamsTool)


class SettleTeamsAction(catalog.SettleTeamsArguments, Action):
    __doc__ = catalog.SettleTeamsArguments.__doc__


class SettleTeamsTool(_DirectorTool[SettleTeamsAction]):
    spec = catalog.SETTLE_TEAMS

    @classmethod
    def action(cls):
        return SettleTeamsAction


register_tool(SettleTeamsTool.name, SettleTeamsTool)


class RaiseEscalationAction(catalog.RaiseEscalationArguments, Action):
    __doc__ = catalog.RaiseEscalationArguments.__doc__


class RaiseEscalationTool(_DirectorTool[RaiseEscalationAction]):
    spec = catalog.RAISE_ESCALATION

    @classmethod
    def action(cls):
        return RaiseEscalationAction


register_tool(RaiseEscalationTool.name, RaiseEscalationTool)


class ResolveEscalationAction(catalog.ResolveEscalationArguments, Action):
    __doc__ = catalog.ResolveEscalationArguments.__doc__


class ResolveEscalationTool(_DirectorTool[ResolveEscalationAction]):
    spec = catalog.RESOLVE_ESCALATION

    @classmethod
    def action(cls):
        return ResolveEscalationAction


register_tool(ResolveEscalationTool.name, ResolveEscalationTool)


class SubmitReviewAction(catalog.SubmitReviewArguments, Action):
    __doc__ = catalog.SubmitReviewArguments.__doc__


class SubmitReviewTool(_DirectorTool[SubmitReviewAction]):
    spec = catalog.SUBMIT_REVIEW

    @classmethod
    def action(cls):
        return SubmitReviewAction


register_tool(SubmitReviewTool.name, SubmitReviewTool)


class ReviewTeamAction(catalog.ReviewTeamArguments, Action):
    __doc__ = catalog.ReviewTeamArguments.__doc__


class ReviewTeamTool(_DirectorTool[ReviewTeamAction]):
    spec = catalog.REVIEW_TEAM

    @classmethod
    def action(cls):
        return ReviewTeamAction


register_tool(ReviewTeamTool.name, ReviewTeamTool)


class ReportAction(catalog.ReportArguments, Action):
    __doc__ = catalog.ReportArguments.__doc__


class ReportTool(_DirectorTool[ReportAction]):
    spec = catalog.REPORT

    @classmethod
    def action(cls):
        return ReportAction


register_tool(ReportTool.name, ReportTool)


class PruneTeamAction(catalog.PruneTeamArguments, Action):
    __doc__ = catalog.PruneTeamArguments.__doc__


class PruneTeamTool(_DirectorTool[PruneTeamAction]):
    spec = catalog.PRUNE_TEAM

    @classmethod
    def action(cls):
        return PruneTeamAction


register_tool(PruneTeamTool.name, PruneTeamTool)


class HandOverAction(catalog.HandOverArguments, Action):
    __doc__ = catalog.HandOverArguments.__doc__


class HandOverTool(_DirectorTool[HandOverAction]):
    spec = catalog.HAND_OVER

    @classmethod
    def action(cls):
        return HandOverAction


register_tool(HandOverTool.name, HandOverTool)


DIRECTOR_TOOLS: dict[str, type[_DirectorTool]] = {
    tool.name: tool
    for tool in (
        ReadTeamContextTool,
        WriteSpecTool,
        EditContractsTool,
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
