"""The director's tools as an agent of the server is given them.

What each tool takes and does is in ``catalog``; this is the one module
that makes tools of the agent SDK out of them.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import Field, create_model

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import (
    DocumentArgument,
    DocumentProtocolBase,
    when_declared,
)
from openhands.agent_server.director.tools import catalog
from openhands.agent_server.director.tools.documents import document_tool
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
        PruneTeamTool,
        HandOverTool,
    )
}


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def tool_of(spec: ToolSpec) -> type[_DirectorTool]:
    """The tool of the agent SDK that ``spec`` is, registered under its
    name. A tool is made once: told again with what it takes unchanged,
    it only takes on what it now says."""
    made = DIRECTOR_TOOLS.get(spec.name)
    if made is not None and made.spec.arguments is spec.arguments:
        made.spec = spec
        return made
    if made is not None:
        raise ValueError(f"The director already has a tool named {spec.name!r}")
    action = type(
        f"{_camel(spec.name)}Action",
        (spec.arguments, Action),
        {"__doc__": spec.arguments.__doc__, "__module__": __name__},
    )
    tool = type(
        f"{_camel(spec.name)}Tool",
        (_DirectorTool,),
        {
            "name": spec.name,
            "spec": spec,
            "action": classmethod(lambda cls: action),  # noqa: ARG005
            "__module__": __name__,
        },
    )
    register_tool(spec.name, tool)
    DIRECTOR_TOOLS[spec.name] = tool
    return tool


def tool_named(name: str) -> type[_DirectorTool] | None:
    """The director's tool of that name, its own or a document's."""
    return DIRECTOR_TOOLS.get(name)


SendDocumentTool = tool_of(catalog.SEND_DOCUMENT)


# The arguments documents added to the director's own tools, so that one
# declared again adds nothing.
_added: set[tuple[str, str]] = set()


def _ask_for(asked: DocumentArgument) -> None:
    """Have a tool of the director's own take one more text."""
    if (asked.tool, asked.name) in _added:
        return
    tool = DIRECTOR_TOOLS.get(asked.tool)
    if tool is None:
        raise ValueError(f"The director has no tool named {asked.tool!r}")
    takes = tool.action()
    if asked.name in takes.model_fields:
        raise ValueError(f"{asked.tool} already takes {asked.name!r}")
    added: dict[str, Any] = {
        asked.name: (str, Field(default="", description=asked.description))
    }
    extended = create_model(
        takes.__name__,
        __base__=takes,
        __doc__=takes.__doc__,
        __module__=__name__,
        **added,
    )
    setattr(tool, "action", classmethod(lambda cls: extended))  # noqa: ARG005, B010
    _added.add((asked.tool, asked.name))


def _tool_of_document(document: Document, protocol: type[DocumentProtocolBase]) -> None:
    # A document written with a tool of the director's own brings none.
    if document.written_with not in catalog.DIRECTOR_TOOL_SPECS:
        tool_of(document_tool(document, protocol))
    for asked in protocol.arguments_in_other_tools(document):
        _ask_for(asked)


# The tools an org's documents bring are made as the org is read.
when_declared(_tool_of_document)
