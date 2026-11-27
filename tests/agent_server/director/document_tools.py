"""The tools of the shipped documents, for the tests that call them.

A document's tool is made when an org that keeps the document is read, so
there is nothing to import it from: this reads the shipped documents as a
director does.
"""

from typing import Any

import openhands.agent_server.director.tools  # noqa: F401
from openhands.agent_server.director.documents.declared import load_documents
from openhands.agent_server.director.documents.registry import declare
from openhands.agent_server.director.models import Report, SessionTeam
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import team_ref
from openhands.agent_server.director.tools.definition import (
    HandOverAction,
    ProposeTeamsAction,
    ReviewTeamAction,
    SettleTeamsAction,
    tool_named,
)


__all__ = [
    "HandOverAction",
    "ProposeTeamsAction",
    "ReviewTeamAction",
    "SettleTeamsAction",
    "ReportAction",
    "ReportTool",
    "WriteContractsAction",
    "WriteContractsTool",
    "WriteSpecAction",
    "WriteSpecTool",
    "report",
]


for _document in load_documents([SHIPPED_DOCUMENTS]).values():
    declare(_document)

WriteSpecTool: Any = tool_named("write_spec")
WriteSpecAction: Any = WriteSpecTool.action()
WriteContractsTool: Any = tool_named("write_contracts")
WriteContractsAction: Any = WriteContractsTool.action()
ReportTool: Any = tool_named("report")
ReportAction: Any = ReportTool.action()


def report(
    service: DirectorService, place: SessionTeam, member: str, text: str
) -> None:
    """Record ``text`` as the member of a team does with its tool."""
    service.documents.append(
        team_ref(place), "report", Report(at="", role=member, text=text)
    )
