"""Saving: the org's tree is written to its repository as it changes, and
what its sessions have is kept beside it on this machine."""

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from openhands.agent_server.director.environment import (
    CodingEnvironment,
    EnvironmentFailed,
)
from openhands.agent_server.director.environment.records import (
    Landing,
    RecordStore,
    taking_in,
)
from openhands.agent_server.director.org_persistence import (
    save_org,
    saving,
)
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import (
    SESSION,
    RunState,
    RunStateError,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.agent_server.director.session_state import kept_sessions
from openhands.sdk.errors import OperationFailed
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# How often a landing is tried when another writer moves a ref first.
_LANDING_ATTEMPTS = 8


@dataclass
class OrgSaveService:
    """Saves the org's tree and what its sessions have.

    Changes of the tree are gathered for ``delay`` seconds and saved as
    one commit; what a session has is only on this machine, so it is kept
    at once. Nothing is saved on a timer before it is entered or after it
    is exited; exiting saves what was still waiting.
    """

    org: Callable[[], OrgStore]
    """The tree that is saved."""
    environment: CodingEnvironment
    """Where the org's records are kept."""
    repo_dir: Callable[[], Path]
    """The repository the org is in."""
    team_memory: Callable[[], RecordStore]
    """The team memory a landing is taken into."""
    run_state: Callable[[], RunState]
    """Where what the sessions have is kept."""
    work_queue: WorkQueueService
    """What carries the saves out."""
    delay: float
    """How many seconds changes of the tree are gathered before a save."""
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    """Held while the tree is saved, and by whoever changes the tree and
    team memory in one step, so that no save comes in between."""
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    _timer: asyncio.TimerHandle | None = field(default=None, init=False)
    _save_asked: bool = field(default=False, init=False)
    _sessions_asked: bool = field(default=False, init=False)

    async def __aenter__(self) -> "OrgSaveService":
        self._loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self._loop = None
        # A save that was cut off goes on in its thread: nothing may write
        # the tree after this has returned.
        await asyncio.to_thread(self.wait)
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
            await self.save()

    def tree_changed(self) -> None:
        """Have the tree saved once its changes have been gathered. Safe
        from any thread."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._arm)

    def _arm(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if not self._sessions_asked:
            self._sessions_asked = True
            self.work_queue.enqueue(self._keep_sessions)
        if self._timer is None:
            self._timer = loop.call_later(
                self.delay, lambda: self.work_queue.enqueue(self._save_due)
            )

    async def _keep_sessions(self) -> None:
        self._sessions_asked = False
        store = self.org()

        def keep() -> None:
            with self.lock:
                self.run_state().replace_marks(SESSION, kept_sessions(store))

        try:
            await asyncio.to_thread(keep)
        except (EnvironmentFailed, RunStateError):
            logger.exception("Could not keep the sessions of %s", self.repo_dir())

    async def _save_due(self) -> None:
        self._timer = None
        await self.save()

    def save_soon(self) -> None:
        """Save the tree without waiting to gather more changes: what was
        just recorded as begun refers to it."""
        if not self._save_asked:
            self._save_asked = True
            self.work_queue.enqueue(self._save_asked_for)

    async def _save_asked_for(self) -> None:
        self._save_asked = False
        await self.save()

    async def save(self) -> None:
        """Save the tree now. A save that fails is logged: the next one
        saves what it would have."""
        try:
            await asyncio.to_thread(self.save_tree, self.org())
        except EnvironmentFailed:
            logger.exception("Could not save the org tree to %s", self.repo_dir())

    def save_tree(self, store: OrgStore) -> None:
        """Write ``store`` to the org's repository, and what its sessions
        have beside it.

        Raises:
            EnvironmentFailed: The repository could not be written.
        """
        with self.lock:
            save_org(self.environment.org_tree(self.repo_dir()), store)
            # Sessions are this machine's: kept here, not with the org.
            self.run_state().replace_marks(SESSION, kept_sessions(store))

    def commit_landing(
        self, landing: Landing, message: str, tree: bool
    ) -> dict[str, str | None]:
        """Take a landing into team memory, and with ``tree`` save the org
        tree as it is now with it: both refs move together or neither.

        Returns:
            The files that were not written, with the text each has now.

        Raises:
            OperationFailed: Another writer kept moving a ref first.
        """
        memory = self.team_memory()
        if not tree:
            return memory.take_in(landing, message)
        store = self.org()
        repo = self.repo_dir()
        with self.lock:
            for _ in range(_LANDING_ATTEMPTS):
                mutate, left = taking_in(landing)
                staged = [
                    made
                    for made in (
                        memory.stage(mutate, message),
                        self.environment.org_tree(repo).stage(
                            saving(store), f"org: {message}"
                        ),
                    )
                    if made is not None
                ]
                if not staged or self.environment.land_together(repo, staged):
                    self.run_state().replace_marks(SESSION, kept_sessions(store))
                    return dict(left)
        raise OperationFailed(
            f"Gave up landing in {repo} after {_LANDING_ATTEMPTS} contended attempts"
        )

    def wait(self) -> None:
        """Return once no save is being written."""
        with self.lock:
            pass
