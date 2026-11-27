"""The coding environment over git: a checkout is a git worktree, a
version is a commit, and a store of records is a hidden ref."""

from collections.abc import Sequence
from pathlib import Path

from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.environment.git import org_sync
from openhands.agent_server.director.environment.git.checkouts import GitCheckouts
from openhands.agent_server.director.environment.git.cli import common_dir
from openhands.agent_server.director.environment.git.git_store import (
    ORG_REF,
    GitStore,
    swap_together,
)
from openhands.agent_server.director.environment.records import Staged
from openhands.agent_server.director.environment.remote import Remote


__all__ = ["GitCheckouts", "GitCodingEnvironment", "GitRemote", "GitStore"]


class GitRemote(Remote):
    """The org's two refs, fetched from and pushed to a git remote."""

    def fetch(self, repository: Path, remote: str) -> None:
        org_sync.fetch(repository, remote)

    def behind(self, repository: Path, remote: str) -> bool:
        return org_sync.behind(repository, remote)

    def push(self, repository: Path, remote: str) -> None:
        org_sync.push(repository, remote)

    def org_tree_of(self, repository: Path, remote: str) -> str | None:
        return org_sync.commit_at(repository, org_sync.remote_org(remote))

    def team_memory_of(self, repository: Path, remote: str) -> str | None:
        return org_sync.commit_at(repository, org_sync.remote_memory(remote))

    def team_memory_name(self, remote: str) -> str:
        return org_sync.remote_memory(remote)


class GitCodingEnvironment(CodingEnvironment):
    """The teams' work in git worktrees of one repository, and the org's
    records under ``refs/agents`` of it."""

    def __init__(self) -> None:
        self._checkouts = GitCheckouts()
        self._remote = GitRemote()

    @property
    def checkouts(self) -> GitCheckouts:
        return self._checkouts

    @property
    def remote(self) -> GitRemote:
        return self._remote

    def team_memory(self, repository: Path) -> GitStore:
        return GitStore(repository)

    def org_tree(self, repository: Path) -> GitStore:
        return GitStore(repository, ref=ORG_REF)

    def land_together(self, repository: Path, staged: Sequence[Staged]) -> bool:
        return swap_together(repository, staged)

    def state_dir(self, repository: Path) -> Path:
        return common_dir(repository)
