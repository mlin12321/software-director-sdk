"""The director the agents of this server belong to.

Agent tools run inside conversations and have no request to take the
director from, so the server binds the one it runs here.
"""

from typing import TYPE_CHECKING

from openhands.sdk.errors import Unavailable


if TYPE_CHECKING:
    from openhands.agent_server.director.service import DirectorService


_director: "DirectorService | None" = None


def bind_director(service: "DirectorService | None") -> None:
    global _director
    _director = service


def current_director() -> "DirectorService":
    """The running director.

    Raises:
        Unavailable: No director is running in this process.
    """
    if _director is None:
        raise Unavailable("The director is not running")
    return _director
