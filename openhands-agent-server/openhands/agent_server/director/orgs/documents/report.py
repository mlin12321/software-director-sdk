"""What the agents of a team record for themselves as they work."""

from collections.abc import Sequence
from typing import Any

from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    ShownDocument,
    register_protocol,
)


class ReportProtocol(DocumentProtocolBase):
    """A log that is only added to. The director still keeps it, adds to
    it and tells it itself, so nothing is said of it here yet."""

    def told_to_reader(self, shown: Sequence[ShownDocument]) -> dict[str, Any]:  # noqa: ARG002
        return {}


register_protocol("report", ReportProtocol)
