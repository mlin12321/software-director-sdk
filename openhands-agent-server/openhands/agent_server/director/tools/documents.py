"""The tool a document is written with, made of what the document says."""

from typing import Any

from openhands.agent_server.director import agent_actions
from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.tools.spec import ToolSpec


def document_tool(
    document: Document, protocol: type[DocumentProtocolBase]
) -> ToolSpec[Any]:
    """The tool a team writes ``document`` with: its name and what agents
    are told of it from the document, what it takes from the document's
    protocol."""
    name = document.name
    return ToolSpec(
        name=document.written_with,
        summary=document.tool_summary,
        arguments=protocol.arguments_for(document),
        act=lambda service, caller, arguments: agent_actions.write_document(
            service, caller, name, arguments
        ),
    )
