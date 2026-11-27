"""The line teams wait in to start: a server runs only so many
conversations at once, and a team that finds it full waits its turn."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field

from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import WAITING
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.event_service import ConversationRunLimitExceeded
from openhands.sdk.errors import NotFound
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


@dataclass
class StartLineService:
    """Starts a team when the server has room for it.

    A start that finds the server at its run limit waits for a
    conversation to stop and tries again, and starts are served in the
    order they began to wait. A team that waits is marked so in the run
    state and in the tree; while the org is paused nothing starts, and a
    team keeps its place in line for when it resumes.

    All of it is called on the loop the director works on, except where a
    method says otherwise.
    """

    org: Callable[[], OrgStore]
    """The org whose teams wait."""
    org_state: OrgStateService
    """Whether the org is paused, and where a team that waits is marked."""
    retry: float
    """How many seconds a full server is given before a start is tried
    again without a conversation of the org having stopped."""
    # Teams that waited when the org was paused, in the order they waited:
    # each tries to start only when it is the first.
    _line: deque[str] = field(default_factory=deque, init=False)
    _line_moved: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    # Set when a conversation stopped running, so a start may fit now.
    _room: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    # Held by the start that waits for room, so that starts are served in
    # the order they began to wait.
    _gate: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def line_up(self, place_ids: Iterable[str]) -> None:
        """Put teams in line in this order, in place of whoever was."""
        self._line = deque(place_ids)

    def leave(self, place_id: str) -> None:
        """Take a team out of the line, and let the one behind it try."""
        if place_id in self._line:
            self._line.remove(place_id)
            self._moved()

    def _moved(self) -> None:
        moved, self._line_moved = self._line_moved, asyncio.Event()
        moved.set()

    def room_made(self) -> None:
        """A conversation stopped running: a start may fit now."""
        self._room.set()

    def org_held(self) -> None:
        """The org was paused: those that wait see it at once."""
        self._room.set()
        self._moved()

    async def when_there_is_room[T](
        self, place_id: str, start: Callable[[], Awaitable[T]]
    ) -> T:
        """Run ``start`` for a team; while the server is full, wait in line
        for a conversation to stop running and try again.

        Raises:
            Held: The org was paused while the team waited. It is still
                next when the org resumes.
        """
        while place_id in self._line and self._line[0] != place_id:
            if self.org_state.paused is not None:
                # The one ahead of it may never start now: it is in line
                # again when the org resumes.
                self.org_state.run_state().mark(WAITING, place_id)
                raise Held
            await self._line_moved.wait()
        try:
            return await start()
        except ConversationRunLimitExceeded:
            logger.info("Server is at its run limit; team %s waits", place_id)
        finally:
            # Nothing is awaited from here to the gate, so the team is in
            # line at the gate before the next one tries.
            self.leave(place_id)
        store = self.org()
        run = self.org_state.run_state()
        run.mark(WAITING, place_id)
        with suppress(NotFound):
            store.update_place(place_id, waiting_to_start=True)
        keeps_place = False
        try:
            async with self._gate:
                while True:
                    if self.org_state.paused is not None:
                        raise Held
                    self._room.clear()
                    try:
                        return await start()
                    except ConversationRunLimitExceeded:
                        pass
                    # Also after a pause: not every conversation is the org's.
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._room.wait(), timeout=self.retry)
        except (Held, asyncio.CancelledError):
            # Stopped while it waited: it is still next when the org resumes.
            keeps_place = True
            raise
        finally:
            if not keeps_place:
                run.unmark(WAITING, place_id)
            with suppress(NotFound):
                store.update_place(place_id, waiting_to_start=False)
