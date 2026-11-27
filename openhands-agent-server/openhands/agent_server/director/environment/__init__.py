"""The coding environment the director's teams work in.

Everything the director does to the code its teams write, and to the
records it keeps of them, goes through a `CodingEnvironment`: where each
team's work is and how it is compared and merged (`Checkouts`), where
what must outlive the server is kept (`RecordStore`), and how an org is
exchanged with another machine (`Remote`). The director ships one, over
git (`director.environment.git`).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.environment.confinement import Confinement
from openhands.agent_server.director.environment.errors import EnvironmentFailed
from openhands.agent_server.director.environment.records import RecordStore, Staged
from openhands.agent_server.director.environment.remote import Remote


__all__ = [
    "Checkouts",
    "CodingEnvironment",
    "Confinement",
    "EnvironmentFailed",
    "RecordStore",
    "Remote",
]


class CodingEnvironment(ABC):
    """Where the teams' work and the org's records are kept, and how
    they are moved."""

    @property
    @abstractmethod
    def checkouts(self) -> Checkouts:
        """Where each team's work is and how it moves between teams."""

    @property
    @abstractmethod
    def remote(self) -> Remote:
        """Exchanges an org's records with a remote of its repository."""

    @abstractmethod
    def confinement(
        self,
        repository: Path,
        *,
        checkouts: tuple[Path, ...],
        sandbox: str | None,
        secret: Path | None,
        server_url: str | None,
        key: Callable[[], bytes],
    ) -> Confinement:
        """What holds the agents of ``repository`` to their scopes.

        Args:
            checkouts: Where its checkouts are.
            sandbox: The binary shells are confined with; None leaves
                them unconfined.
            secret: A file of the director's own that no agent may open.
            server_url: Where an agent that runs its own tools reaches
                the server; None when that is not known.
            key: Gives what the tokens of such agents are signed with.
        """

    @abstractmethod
    def team_memory(self, repository: Path) -> RecordStore:
        """Where the teams of the org in ``repository`` keep what they
        write for each other and for later."""

    @abstractmethod
    def org_tree(self, repository: Path) -> RecordStore:
        """Where the tree of the org in ``repository`` is kept."""

    @abstractmethod
    def land_together(self, repository: Path, staged: Sequence[Staged]) -> bool:
        """Make every staged change its store's, all of them or none.

        Returns:
            False when a store was changed by someone else meanwhile.
        """

    @abstractmethod
    def state_dir(self, repository: Path) -> Path:
        """The directory of a repository in which the director keeps
        what is this machine's alone.

        Raises:
            EnvironmentFailed: ``repository`` is not one.
        """
