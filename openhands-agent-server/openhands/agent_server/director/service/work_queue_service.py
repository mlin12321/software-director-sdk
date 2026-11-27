"""Lanes: what is asked for is carried out on worker tasks, one thing at a
time in each lane and in the order it was asked for there."""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field

from openhands.sdk.errors import Unavailable
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

type Effect = Callable[[], Awaitable[None]]

# The lane the running effect is carried out in; '' outside the lanes.
_LANE: ContextVar[str] = ContextVar("director_lane", default="")


@dataclass
class WorkQueueService:
    """Carries out what it is asked to, lane by lane.

    A lane's work is carried out in the order it was asked for, and lanes
    do not wait for each other. A lane exists while it has work, so any
    name is a lane. An effect that raises is logged and the lane goes on.

    It carries nothing out before it is entered or after it is exited.
    """

    default_lane: str
    """The lane of what names none."""
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False)
    # What each lane still has to carry out.
    _waiting: dict[str, deque[Effect]] = field(default_factory=dict, init=False)
    _workers: set[asyncio.Task[None]] = field(default_factory=set, init=False)
    # Effects asked for and not carried out yet.
    _unsettled: int = field(default=0, init=False)
    _quiet: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    def __post_init__(self) -> None:
        self._quiet.set()

    async def __aenter__(self) -> "WorkQueueService":
        self._loop = asyncio.get_running_loop()
        self._waiting = {}
        self._unsettled = 0
        self._quiet.set()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self._loop = None
        workers = list(self._workers)
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    @property
    def is_open(self) -> bool:
        """Whether what is asked for now would be carried out."""
        return self._loop is not None

    @property
    def current(self) -> str:
        """The lane the caller is carried out in; '' outside the lanes."""
        return _LANE.get()

    def enqueue(self, effect: Effect, lane: str | None = None) -> None:
        """Ask for ``effect`` in ``lane``. Safe from any thread.

        Raises:
            Unavailable: Nothing is carried out here now.
        """
        loop = self._loop
        if loop is None:
            raise Unavailable("The director is not running agents on this server")
        loop.call_soon_threadsafe(self._put, effect, lane or self.default_lane)

    def _put(self, effect: Effect, lane: str) -> None:
        self._unsettled += 1
        self._quiet.clear()
        waiting = self._waiting.get(lane)
        if waiting is not None:
            waiting.append(effect)
            return
        self._waiting[lane] = deque([effect])
        worker = asyncio.create_task(self._work(lane), name=f"director-{lane}")
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)

    async def _work(self, lane: str) -> None:
        waiting = self._waiting[lane]
        _LANE.set(lane)
        try:
            while waiting:
                effect = waiting.popleft()
                try:
                    await effect()
                except Exception:
                    logger.exception("A director action failed")
                finally:
                    self._unsettled -= 1
                    if not self._unsettled:
                        self._quiet.set()
        finally:
            del self._waiting[lane]

    async def run_in(self, lane: str, effect: Effect) -> None:
        """Carry ``effect`` out in ``lane`` and wait for it: after
        everything that was asked for there before. What it raises is
        raised here. Called on the loop the lanes work on."""
        if _LANE.get() == lane:
            await effect()
            return
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        async def carried_out() -> None:
            try:
                await effect()
            except Exception as error:
                done.set_exception(error)
            else:
                done.set_result(None)

        self._put(carried_out, lane)
        await done

    async def after(self, lane: str) -> None:
        """Wait until what ``lane`` was asked to do so far is done."""
        done = asyncio.Event()

        async def reached() -> None:
            done.set()

        self.enqueue(reached, lane)
        await done.wait()

    async def settled(self) -> None:
        """Wait until everything asked for so far has been carried out."""
        # Let what other threads asked for reach the queue first.
        await asyncio.sleep(0)
        # What is carried out may ask for more.
        while self._unsettled:
            await self._quiet.wait()
            await asyncio.sleep(0)
