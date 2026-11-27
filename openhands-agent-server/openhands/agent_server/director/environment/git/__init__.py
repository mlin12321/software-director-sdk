"""The coding environment over git: a checkout is a git worktree, and a
version is a commit."""

from pathlib import Path

from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.environment.git.checkouts import GitCheckouts
from openhands.agent_server.director.git import common_dir


__all__ = ["GitCheckouts", "GitCodingEnvironment"]


class GitCodingEnvironment(CodingEnvironment):
    """The teams' work in git worktrees of one repository."""

    def __init__(self) -> None:
        self._checkouts = GitCheckouts()

    @property
    def checkouts(self) -> GitCheckouts:
        return self._checkouts

    def state_dir(self, repository: Path) -> Path:
        return common_dir(repository)
