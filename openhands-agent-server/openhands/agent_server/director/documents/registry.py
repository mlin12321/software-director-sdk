"""The protocols documents may name, and the documents orgs declare."""

from collections.abc import Callable

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import DocumentProtocolBase
from openhands.agent_server.director.documents.store import DocumentStore
from openhands.agent_server.director.documents.tree import TeamTree


class UnknownProtocol(ValueError):
    """A document names a protocol nobody registered."""


_PROTOCOLS: dict[str, type[DocumentProtocolBase]] = {}


def register_protocol(name: str, protocol: type[DocumentProtocolBase]) -> None:
    """Make ``protocol`` the one a document names as ``name``."""
    _PROTOCOLS[name] = protocol


def protocol_class(document: Document) -> type[DocumentProtocolBase]:
    """The class of what ``document`` does.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
    """
    if document.protocol is None:
        return DocumentProtocolBase
    protocol = _PROTOCOLS.get(document.protocol)
    if protocol is None:
        raise UnknownProtocol(
            f"The document {document.name!r} names the protocol "
            f"{document.protocol!r}, and there is none of that name"
        )
    return protocol


def protocol_of(
    document: Document, store: DocumentStore, tree: TeamTree
) -> DocumentProtocolBase:
    """What ``document`` does: the protocol it names, or the plain one.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
    """
    return protocol_class(document)(document, store, tree)


# Who is told of each document an org declares, and the ones told so far.
# The service that reads an org's documents does not know what gives its
# agents their tools; that part listens here.
_declared: dict[str, tuple[Document, type[DocumentProtocolBase]]] = {}
_listeners: list[Callable[[Document, type[DocumentProtocolBase]], None]] = []


def declare(document: Document) -> None:
    """Tell of a document an org keeps, so that its tool can be made.

    Raises:
        UnknownProtocol: It names a protocol nobody registered.
        ValueError: Its tool cannot be made: another of that name takes
            something else.
    """
    found = (document, protocol_class(document))
    for tell in _listeners:
        tell(*found)
    _declared[document.name] = found


def when_declared(
    tell: Callable[[Document, type[DocumentProtocolBase]], None],
) -> None:
    """Have ``tell`` called with every document declared, the ones before
    now as well."""
    _listeners.append(tell)
    for found in list(_declared.values()):
        tell(*found)
