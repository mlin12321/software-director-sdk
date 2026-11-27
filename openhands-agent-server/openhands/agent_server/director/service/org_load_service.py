"""Where the org is and the org as it is loaded: found in a repository,
loaded on first use, and remembered between starts."""

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from openhands.agent_server.director.agent_types import TeamTemplate
from openhands.agent_server.director.documents.store import TeamRef
from openhands.agent_server.director.environment import CodingEnvironment
from openhands.agent_server.director.environment.records import RecordStore
from openhands.agent_server.director.models import OrgNodeInput, OrgSummary
from openhands.agent_server.director.org_persistence import load_org
from openhands.agent_server.director.org_store import OrgStore, team_ref
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import SESSION
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.session_state import restore_sessions
from openhands.agent_server.director.split import clean_path
from openhands.sdk.errors import DomainError, Unavailable, Unprocessable
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class TeamsOfTheOrg:
    """Where a team stands among the others: the org's tree for a team of
    the org, its session's for the work of one in a session."""

    def __init__(self, org: Callable[[], OrgStore]) -> None:
        self._org = org

    def above(self, team: TeamRef) -> TeamRef | None:
        store = self._org()
        node = store.get(team.id)
        above = store.get(node.parent_id) if node and node.parent_id else None
        return team_ref(above) if above else None

    def below(self, team: TeamRef) -> list[TeamRef]:
        store = self._org()
        if store.place(team.id) is not None:
            return [team_ref(place) for place in store.below(team.id)]
        return [team_ref(node) for node in store.sub_teams(team.id)]


@dataclass
class OrgLoadService:
    """Finds, loads and remembers the org of one server.

    An org lives in the repository its first conversation was opened in,
    which need not be the directory the server was started for. Until
    there is an org, the one saved in that directory is looked for; a
    repository that has none and is a submodule of one that has stands
    for the nearest such repository above it. Nothing is read from disk
    until :attr:`org` is first used.
    """

    directory: Path
    """The directory the server was started for."""
    environment: CodingEnvironment
    """Where the org's tree and its checkouts are kept."""
    remembered: RememberedFile
    """Keeps the org's repository and the name of this server between
    starts."""
    org_state: OrgStateService
    """The run state of the repository the org is in."""
    type_of: Callable[[OrgNodeInput], TeamTemplate]
    """The type a team's name of a type stands for, as the tree needs it."""
    tree_changed: Callable[[], None]
    """Told of every change of the tree, from any thread."""
    loaded: Callable[[], None]
    """Told when the org's store has been loaded, so that what the org had
    begun is taken up."""
    is_sandboxed: Callable[[], bool]
    """Whether the agents' shells are confined on this server."""
    _repo_dir: Path = field(init=False)
    _store: OrgStore | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        # Where the org is; moves up when the store is loaded from above.
        self._repo_dir = self.directory

    @property
    def org(self) -> OrgStore:
        """The org tree, loaded on first use."""
        with self._lock:
            if self._store is None:
                self._store = self._load_store()
            return self._store

    def find_org(self) -> OrgStore | None:
        """The org tree if it has been loaded; asking does not load it."""
        return self._store

    @property
    def has_org(self) -> bool:
        """Whether the org tree has been loaded."""
        return self._store is not None

    @property
    def repo_dir(self) -> Path:
        """The repository the org is in."""
        _ = self.org  # settles which repository the org is in
        return self._repo_dir

    def repo_dir_now(self) -> Path:
        """The repository the org is in as far as is known; asking does
        not load the org."""
        return self._repo_dir

    def move_to(self, repository: Path) -> Path:
        """Have the org be in ``repository`` from now on.

        Returns:
            The repository it was in.
        """
        before, self._repo_dir = self._repo_dir, repository
        return before

    @property
    def team_memory(self) -> RecordStore:
        """The team memory of the org's repository.

        Raises:
            Unavailable: The director's directory is not a git repository.
        """
        _ = self.org  # settles which repository the org is in
        if not self.environment.checkouts.is_repository(self._repo_dir):
            raise Unavailable(
                f"The director has no team memory: {self._repo_dir} is not a git "
                "repository. Point OH_DIRECTOR_REPO_PATH at one."
            )
        return self.memory()

    def memory(self) -> RecordStore:
        """The team memory of the repository the org is in as far as is
        known, whether or not it is a repository."""
        return self.environment.team_memory(self._repo_dir)

    def summary(self) -> OrgSummary:
        store = self.org
        return OrgSummary(
            root_id=store.root_id,
            node_count=len(store),
            repo_path=str(self._repo_dir),
            directory=self.home(),
            is_git_repository=self.environment.checkouts.is_repository(self._repo_dir),
            boundary_enforcement="sandbox" if self.is_sandboxed() else "checks",
            paused=self.org_state.paused if self.org_state.run_state() else None,
            merging=self.org_state.run_state().get("merge") is not None,
        )

    def repository(self) -> Path:
        """The repository the org is in, or would be in.

        Raises:
            Unavailable: The director's directory is not in a git repository.
        """
        if self.org.root_id is not None:
            return self._repo_dir
        checkouts = self.environment.checkouts
        if not checkouts.is_repository(self.directory):
            raise Unavailable(
                f"An org needs a git repository; {self.directory} is not one"
            )
        return checkouts.root_of(str(self.directory))

    def instance(self) -> str:
        """Names this server, to tell its run state from another's."""
        if self.remembered.path is None:
            return ""
        kept = self.remembered.read().get("instance")
        if not isinstance(kept, str):
            kept = uuid4().hex
            self.remembered.keep(instance=kept)
        return kept

    def working(self) -> None:
        """Record that the org is run by this server and has not been
        stopped."""
        run = self.org_state.run_state()
        run.put("instance", self.instance())
        run.put("clean", "0")

    def home(self) -> str:
        """The directory the server was started for, relative to the org's
        repository; '' when it is not below its root."""
        home = Path(os.path.realpath(self.directory))
        top = self.environment.checkouts.root_of(str(self._repo_dir))
        return home.relative_to(top).as_posix() if top in home.parents else ""

    def repo_relative(self, path: str) -> str:
        """``path`` as a clean path relative to the repository.

        Raises:
            Unprocessable: It is not inside the director's directory.
        """
        if not Path(path).is_absolute():
            try:
                return clean_path(path)
            except DomainError as error:
                raise Unprocessable(str(error)) from error
        target = Path(os.path.realpath(path))
        top = self.environment.checkouts.root_of(str(self._repo_dir))
        if not target.is_relative_to(top):
            raise Unprocessable(f"{path} is outside the org's repository {top}")
        relative = target.relative_to(top).as_posix()
        return "" if relative == "." else relative

    def join(self, workspace: Path, climb: bool = True) -> bool:
        """Take up the org saved at ``workspace``, or above it when ``climb``."""
        if not self._load_from(workspace, self.org, climb=climb):
            return False
        self.remembered.keep(repo_path=str(self._repo_dir))
        return True

    def remembered_repo(self) -> Path | None:
        """The repository the org was last in, if that was kept."""
        repo_path = self.remembered.read().get("repo_path")
        return Path(repo_path) if isinstance(repo_path, str) else None

    def _load_store(self) -> OrgStore:
        store = OrgStore(self.type_of)
        remembered = self.remembered_repo()
        if remembered is None or not self._load_from(remembered, store):
            self._load_from(self.directory, store)
        store.on_change(self.tree_changed)
        self.loaded()
        return store

    def _load_from(
        self, directory: Path, store: OrgStore, *, climb: bool = True
    ) -> bool:
        """Load the org saved in the repository of ``directory``.

        With ``climb``, a repository that has none is followed by the nearest
        one above that has it as a submodule. Without it, only a ``.git`` in
        ``directory`` itself counts.

        Returns:
            Whether one was found. The director then works in its repository.
        """
        checkouts = self.environment.checkouts
        repository: Path | None = directory
        while repository is not None and checkouts.is_repository(repository):
            if not climb and not (repository / ".git").exists():
                return False
            if load_org(self.environment.org_tree(repository), store):
                self._repo_dir = checkouts.root_of(str(repository))
                logger.info("Director loaded %d teams from %s", len(store), repository)
                kept = self.org_state.open(self._repo_dir)
                try:
                    # None of another server's run state is used, so its
                    # sessions are not taken up without it either.
                    if kept.get("instance") in (None, self.instance()):
                        restore_sessions(store, kept.marked(SESSION))
                finally:
                    kept.close()
                return True
            if not climb:
                return False
            repository = checkouts.enclosing(repository)
        return False
