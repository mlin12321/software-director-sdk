"""The tools of the shipped documents, for the tests that call them.

A document's tool is made when an org that keeps the document is read, so
there is nothing to import it from: this reads the shipped documents as a
director does.
"""

from typing import Any

import openhands.agent_server.director.tools  # noqa: F401
from openhands.agent_server.director.documents.declared import load_documents
from openhands.agent_server.director.documents.protocol import declare
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS
from openhands.agent_server.director.tools.definition import tool_named


for _document in load_documents([SHIPPED_DOCUMENTS]).values():
    declare(_document)

WriteSpecTool: Any = tool_named("write_spec")
WriteSpecAction: Any = WriteSpecTool.action()
EditContractsTool: Any = tool_named("edit_contracts")
EditContractsAction: Any = EditContractsTool.action()
