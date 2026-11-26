"""What a document does as the work of the team that keeps it goes on.

A document that needs nothing of this is a plain text its team rewrites:
it is kept with :class:`DocumentProtocolBase` itself. One that does more
names a protocol in its YAML file, a subclass registered under that name.
"""

from abc import ABC

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.events import (
    HandedOver,
    SubTeamsCreated,
    TeamRemoved,
    WorkAccepted,
)
from openhands.agent_server.director.documents.store import DocumentStore


class UnknownProtocol(ValueError):
    """A document names a protocol nobody registered."""


class DocumentProtocolBase(ABC):
    """What one document does. Every answer here is the plain one, and a
    subclass gives another only where its document differs."""

    def __init__(self, document: Document, store: DocumentStore) -> None:
        """
        Args:
            document: The document this is the protocol of.
            store: Where documents are kept.
        """
        self.document = document
        self.store = store

    def waits_for_team_changes(self) -> bool:
        """Whether what a session wrote of this document becomes the
        org's only once the session's changes to the teams are settled,
        and not with its code."""
        return False

    def when_sub_teams_are_created(self, created: SubTeamsCreated) -> None:
        """A proposal was approved and the team now has sub-teams."""

    def when_sub_team_work_is_accepted(self, accepted: WorkAccepted) -> None:
        """The team's reviewer accepted what a sub-team submitted."""

    def when_removed_team_is_handed_over(self, handed: HandedOver) -> None:
        """The team above passed on to the team what the teams before it
        left."""

    def when_team_is_removed(
        self,
        removed: TeamRemoved,  # noqa: ARG002
    ) -> str | None:
        """The team was replaced or pruned.

        Returns:
            What is kept of the document in the record of the team that
            is gone; None for nothing.
        """
        return None


_PROTOCOLS: dict[str, type[DocumentProtocolBase]] = {}


def register_protocol(name: str, protocol: type[DocumentProtocolBase]) -> None:
    """Make ``protocol`` the one a document names as ``name``."""
    _PROTOCOLS[name] = protocol


def protocol_of(document: Document, store: DocumentStore) -> DocumentProtocolBase:
    """What ``document`` does: the protocol it names, or the plain one.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
    """
    if document.protocol is None:
        return DocumentProtocolBase(document, store)
    protocol = _PROTOCOLS.get(document.protocol)
    if protocol is None:
        raise UnknownProtocol(
            f"The document {document.name!r} names the protocol "
            f"{document.protocol!r}, and there is none of that name"
        )
    return protocol(document, store)
