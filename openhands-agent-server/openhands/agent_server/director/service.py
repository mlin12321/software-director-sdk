"""The director's state for one server: an org tree and its blackboard."""

import logging
import threading
from pathlib import Path

from openhands.agent_server.config import Config
from openhands.agent_server.director.fixture import build_fixture_nodes
from openhands.agent_server.director.git import is_git_repository
from openhands.agent_server.director.git_blackboard import GitBlackboard
from openhands.agent_server.director.org_persistence import (
    create_org_board,
    load_org,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.sdk.errors import Unavailable


logger = logging.getLogger(__name__)


class DirectorService:
    """Owns the org tree and the blackboard of one repository.

    Nothing is read from disk until the tree or the blackboard is first used,
    so creating the service is free for servers that never use the director.
    """

    def __init__(self, repo_dir: Path, *, fixture: bool = False) -> None:
        self._repo_dir = repo_dir
        self._fixture = fixture
        self._lock = threading.Lock()
        self._store: OrgStore | None = None

    @classmethod
    def from_config(cls, config: Config) -> "DirectorService":
        return cls(
            config.director_repo_path or config.workspace_path,
            fixture=config.director_fixture,
        )

    @property
    def store(self) -> OrgStore:
        """The org tree, loaded on first use."""
        with self._lock:
            if self._store is None:
                self._store = self._load_store()
            return self._store

    @property
    def blackboard(self) -> GitBlackboard:
        """The blackboard of the director's repository.

        Raises:
            Unavailable: The director's directory is not a git repository.
        """
        if not is_git_repository(self._repo_dir):
            raise Unavailable(
                f"The director has no blackboard: {self._repo_dir} is not a git "
                "repository. Point OH_DIRECTOR_REPO_PATH at one."
            )
        return GitBlackboard(self._repo_dir)

    def _load_store(self) -> OrgStore:
        store = OrgStore()
        if self._fixture:
            store.replace_all(build_fixture_nodes())
            logger.info("Director serving the fixture tree (%d nodes)", len(store))
        elif is_git_repository(self._repo_dir) and load_org(
            create_org_board(self._repo_dir), store
        ):
            logger.info("Director loaded %d nodes from %s", len(store), self._repo_dir)
        return store
