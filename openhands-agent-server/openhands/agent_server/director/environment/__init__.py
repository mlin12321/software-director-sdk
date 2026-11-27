"""The coding environment the director's teams work in.

Everything the director does to the code its teams write goes through a
`CodingEnvironment`: where each team's work is, how it is compared and
merged, and where what must outlive the server is kept. The director
ships one, over git (`director.environment.git`).
"""

from abc import ABC, abstractmethod
from pathlib import Path

from openhands.agent_server.director.environment.checkouts import Checkouts
from openhands.agent_server.director.environment.errors import EnvironmentFailed


__all__ = ["Checkouts", "CodingEnvironment", "EnvironmentFailed"]


class CodingEnvironment(ABC):
    """Where the teams' work is kept and how it is moved."""

    @property
    @abstractmethod
    def checkouts(self) -> Checkouts:
        """Where each team's work is and how it moves between teams."""

    @abstractmethod
    def state_dir(self, repository: Path) -> Path:
        """The directory of a repository in which the director keeps
        what is this machine's alone.

        Raises:
            EnvironmentFailed: ``repository`` is not one.
        """
