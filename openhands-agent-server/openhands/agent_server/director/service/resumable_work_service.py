"""Intents: something of several steps the director does for a team, kept
before its first step so that a restart or a pause finishes it."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from openhands.agent_server.director.intents import IntentPayload
from openhands.agent_server.director.run_state import Intent, RunState
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.errors import Unavailable
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

type Step = Callable[[Intent], Awaitable[None]]


class Held(Exception):
    """A step could not be finished now. Its intent stays open and is
    carried out again."""


@dataclass
class ResumableWorkService:
    """Records what the director begins and carries each kind out with the
    step registered for it.

    An intent is finished from the steps that were done, after a restart
    or a pause too, so each step must be safe to take twice.
    """

    run_state: Callable[[], RunState]
    """Where intents are kept."""
    work_queue: WorkQueueService
    """What carries the steps out."""
    lane_of: Callable[[Intent], str]
    """The lane an intent is carried out in."""
    is_paused: Callable[[], bool]
    """Whether nothing is to be carried out now."""
    save: Callable[[], Awaitable[None]]
    """Saves what a step changed, at once."""
    save_soon: Callable[[], None]
    """Asks for a save of what an intent that was just kept refers to."""
    let_go: Callable[[Intent], None]
    """Releases what waited for a step that failed."""
    keep_failure: Callable[[Intent, Exception], None]
    """Keeps a step that failed where it is read."""
    left: Callable[[str], None]
    """Told the team of an intent that is no longer carried out."""
    saved_before_forgotten: frozenset[str] = frozenset()
    """The kinds whose outcome is saved before the intent is finished: a
    server that ends in between does the step again, never loses it."""
    steps: dict[str, Step] = field(default_factory=dict, init=False)
    """What carries out each kind of intent; see :meth:`register`."""
    # The intents being carried out right now, so that taking one up
    # again does not carry it out a second time.
    _in_flight: set[int] = field(default_factory=set, init=False)

    def register(self, kind: str, step: Step) -> None:
        """Name what carries out the intents of one kind.

        Raises:
            ValueError: The kind has a step already.
        """
        if kind in self.steps:
            raise ValueError(f"The step {kind!r} is registered already")
        self.steps[kind] = step

    def begin(self, place_id: str, payload: IntentPayload) -> None:
        """Keep an intent for a team, then carry it out. Safe from any
        thread.

        Raises:
            Unavailable: Nothing is carried out here now.
        """
        if not self.work_queue.is_open:
            raise Unavailable("The director is not running agents on this server")
        intent = self.run_state().begin(place_id, payload)
        self.save_soon()
        self.take_up(intent.id, self.lane_of(intent))

    def take_up(self, intent_id: int, lane: str) -> None:
        """Ask for an open intent to be carried out in ``lane``."""
        self.work_queue.enqueue(lambda: self.carry_out(intent_id), lane)

    async def carry_out(self, intent_id: int) -> None:
        """Take the step of an open intent, unless it is being taken."""
        run = self.run_state()
        if self.is_paused() or intent_id in self._in_flight:
            return
        intent = next(
            (open_ for open_ in run.open_intents() if open_.id == intent_id), None
        )
        if intent is None:
            return
        step = self.steps.get(intent.kind)
        if step is None:
            run.finish(intent.id)
            self.left(intent.node_id)
            return
        self._in_flight.add(intent.id)
        try:
            await step(intent)
        except Held:
            return
        except Exception as error:
            logger.exception("A director action failed")
            run.finish(intent.id)
            # What waited for the teams it was to change waits no longer.
            self.let_go(intent)
            self.keep_failure(intent, error)
        else:
            if intent.kind in self.saved_before_forgotten:
                await self.save()
            run.finish(intent.id)
        finally:
            self._in_flight.discard(intent.id)
            self.left(intent.node_id)
