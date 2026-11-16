"""The tools the agents of an org use to work as one.

Importing this package registers them. They run in the server's own process
against its director, so they exist only in the agent server.
"""

from openhands.agent_server.director.tools.definition import (
    ApproveTeamsTool,
    DeclineTeamsTool,
    HandOverTool,
    MsgIcTool,
    MsgManagerTool,
    MsgTeamTool,
    PruneTeamTool,
    RaiseEscalationTool,
    ReadNodeContextTool,
    ReportTool,
    ResolveEscalationTool,
    ReviewTeamTool,
    SettleTeamsTool,
    StopTeamTool,
    SubmitReviewTool,
    WriteSpecTool,
)


__all__ = [
    "ApproveTeamsTool",
    "SettleTeamsTool",
    "DeclineTeamsTool",
    "HandOverTool",
    "MsgIcTool",
    "MsgManagerTool",
    "MsgTeamTool",
    "PruneTeamTool",
    "RaiseEscalationTool",
    "ReadNodeContextTool",
    "ReportTool",
    "ResolveEscalationTool",
    "ReviewTeamTool",
    "StopTeamTool",
    "SubmitReviewTool",
    "WriteSpecTool",
]
