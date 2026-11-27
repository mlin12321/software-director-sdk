"""An org on more than one machine: its tree and team memory given to a
remote of its repository and brought from one."""

from abc import ABC, abstractmethod
from pathlib import Path


class Remote(ABC):
    """Exchanges an org's two stores with a remote.

    Nothing is exchanged on its own. A fetch brings the remote's stores
    here, beside what is here, for a merge; a push gives the remote ours.
    """

    @abstractmethod
    def fetch(self, repository: Path, remote: str) -> None:
        """Bring the remote's org tree and team memory beside ours.

        Raises:
            NotFound: The repository has no such remote.
            UpstreamFailed: The remote could not be reached.
        """

    @abstractmethod
    def behind(self, repository: Path, remote: str) -> bool:
        """Whether what was fetched has changes ours lacks."""

    @abstractmethod
    def push(self, repository: Path, remote: str) -> None:
        """Give the remote the org tree and the team memory.

        Raises:
            UpstreamFailed: The remote could not be reached or refused.
        """

    @abstractmethod
    def org_tree_of(self, repository: Path, remote: str) -> str | None:
        """The version of the org tree that was fetched; None when the
        remote has none."""

    @abstractmethod
    def team_memory_of(self, repository: Path, remote: str) -> str | None:
        """The version of the team memory that was fetched; None when the
        remote has none."""

    @abstractmethod
    def team_memory_name(self, remote: str) -> str:
        """What the team memory that was fetched is called here, as
        `RecordStore.merge` takes it."""
