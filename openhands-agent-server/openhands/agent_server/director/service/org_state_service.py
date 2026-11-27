"""The state of an org's run: what the director is in the middle of, kept
in the org's repository, and whether the org is paused or merging."""

import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openhands.agent_server.director.git import (
    GitError,
    common_dir,
    is_git_repository,
)
from openhands.agent_server.director.models import (
    OrgStateEvent,
    PauseReason,
    RecoveryItem,
)
from openhands.agent_server.director.run_state import (
    PROGRESS,
    RUNNING,
    RunState,
    RunStateError,
)
from openhands.agent_server.director.service.stall_watch_service import progress_of
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Where, in the repository's git directory, the director keeps what it is in
# the middle of.
_RUN_STATE = "director/state.db"
_PAUSED = "paused"
_MERGE = "merge"
_DROPPED = "dropped"


def _reason(kept: str | None) -> PauseReason | None:
    if kept == "human" or kept == "shutdown" or kept == "crash":
        return kept
    return None


@dataclass
class OrgStateService:
    """Opens the run state of the repository the org is in, and says and
    changes whether the org is paused or in the middle of a merge.

    The run state follows the org: when the org is found in another
    repository, the next :meth:`run_state` opens that one's.
    """

    repo_dir: Callable[[], Path]
    """The repository the org is in, settled before it is given."""
    clock: Callable[[], datetime]
    """The time, for counting how long a team has shown no progress."""
    running: set[str] = field(default_factory=set, init=False)
    """The conversations last recorded as running."""
    _run: RunState | None = field(default=None, init=False)
    _run_of: Path | None = field(default=None, init=False)
    _paused: PauseReason | None = field(default=None, init=False)
    _listeners: list[Callable[[OrgStateEvent], None]] = field(
        default_factory=list, init=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def run_state(self) -> RunState:
        """What the director is in the middle of, in the org's repository."""
        repo_dir = self.repo_dir()
        with self._lock:
            if self._run is None or self._run_of != repo_dir:
                if self._run is not None:
                    self._run.close()
                self._run = self.open(repo_dir)
                self._run_of = repo_dir
                self._paused = _reason(self._run.get(_PAUSED))
                self.running = set(self._run.marked(RUNNING))
                self._restart_clocks(self._run)
            return self._run

    def open(self, repo_dir: Path) -> RunState:
        """The run state kept in a repository, for the caller to close.
        One that cannot be opened is kept in memory only."""
        if not is_git_repository(repo_dir):
            return RunState(None)
        try:
            return RunState(common_dir(repo_dir) / _RUN_STATE)
        except (GitError, OSError, RunStateError, sqlite3.Error):
            logger.exception("Run state of %s is kept in memory only", repo_dir)
            return RunState(None)

    def close(self) -> None:
        """Close the run state; the next use opens it again."""
        with self._lock:
            if self._run is not None:
                self._run.close()
                self._run = None

    @property
    def paused(self) -> PauseReason | None:
        """Why the org is paused, as last read or set; None while it runs."""
        return self._paused

    def state(self) -> OrgStateEvent:
        """Whether the org is paused or in the middle of a merge."""
        run = self.run_state()
        return OrgStateEvent(paused=self._paused, merging=run.get(_MERGE) is not None)

    def watch(self, listener: Callable[[OrgStateEvent], None]) -> Callable[[], None]:
        """Call ``listener`` whenever the org is paused, resumed, or begins
        or ends a merge. It may be called from any thread.

        Returns:
            A function that stops the calls.
        """
        with self._lock:
            self._listeners.append(listener)

        def unwatch() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unwatch

    def _announce(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        if not listeners:
            return
        state = self.state()
        for listener in listeners:
            listener(state)

    def refresh(self) -> None:
        """Read again whether the org is paused, after the run state was
        changed without :meth:`hold` or :meth:`release`."""
        self._paused = _reason(self.run_state().get(_PAUSED))
        self._announce()

    def hold(self, reason: PauseReason) -> None:
        """Pause the org, and keep that it is paused."""
        self.run_state().put(_PAUSED, reason)
        self._paused = reason
        self._announce()

    def release(self) -> None:
        """Let the org run again. The time it was paused counts against
        no team, and what was put aside when it stopped is forgotten."""
        run = self.run_state()
        run.put(_PAUSED, None)
        run.put(_DROPPED, None)
        self._paused = None
        self._restart_clocks(run)
        self._announce()

    def dropped(self) -> list[RecoveryItem]:
        """What was put aside when the org was last taken up again."""
        kept = self.run_state().get(_DROPPED)
        return [RecoveryItem.model_validate(item) for item in json.loads(kept or "[]")]

    def merging(self) -> dict[str, Any] | None:
        """The merge with another copy of the org that waits for choices;
        None when there is none."""
        kept = self.run_state().get(_MERGE)
        return json.loads(kept) if kept else None

    def set_merge(self, state: dict[str, Any] | None) -> None:
        """Keep the merge that waits for choices; None when it is over."""
        self.run_state().put(_MERGE, json.dumps(state) if state is not None else None)
        self._announce()

    def _restart_clocks(self, run: RunState) -> None:
        """Count the time teams show no progress from now: the time an org
        was paused, or its server was not running, is nobody's."""
        now = self.clock().isoformat()
        for place_id, kept in run.marked(PROGRESS).items():
            seen = progress_of(kept)
            # A team without a task was told of once, and stays told.
            told = {"told": now} if "told" in seen else {}
            again = {"at": now, "head": seen.get("head"), "checked": now}
            run.mark(PROGRESS, place_id, json.dumps(again | told))
