"""Intents by themselves, with a run state in memory and steps made by
hand. What each step of the director does is tested through the service."""

from typing import Any

import pytest

from openhands.agent_server.director.intents import Prune, Split
from openhands.agent_server.director.run_state import Intent, RunState
from openhands.agent_server.director.service.resumable_work_service import (
    Held,
    ResumableWorkService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.errors import Unavailable


class Seen:
    """What an intent service told its owner."""

    def __init__(self) -> None:
        self.paused = False
        self.said: list[Any] = []

    def resumable_work(
        self, work_queue: WorkQueueService, run: RunState
    ) -> ResumableWorkService:
        async def save() -> None:
            self.said.append("saved")

        return ResumableWorkService(
            run_state=lambda: run,
            work_queue=work_queue,
            lane_of=lambda intent: intent.node_id,
            is_paused=lambda: self.paused,
            save=save,
            save_soon=lambda: self.said.append("save soon"),
            let_go=lambda intent: self.said.append(("let go", intent.kind)),
            keep_failure=lambda intent, error: self.said.append(
                ("failed", intent.kind, str(error))
            ),
            left=lambda place_id: self.said.append(("left", place_id)),
            saved_before_forgotten=frozenset({"prune"}),
        )


@pytest.fixture
def seen() -> Seen:
    return Seen()


async def test_an_intent_is_kept_then_carried_out_in_its_lane_and_finished(
    seen: Seen,
):
    run = RunState(None)
    async with WorkQueueService(default_lane="org") as work_queue:
        resumable_work = seen.resumable_work(work_queue, run)

        async def split(intent: Intent) -> None:
            # Kept before its first step.
            assert [open_.id for open_ in run.open_intents()] == [intent.id]
            seen.said.append(("split", work_queue.current))

        resumable_work.register("split", split)
        resumable_work.begin("team", Split())
        await work_queue.settled()

    assert seen.said == ["save soon", ("split", "team"), ("left", "team")]
    assert run.open_intents() == []


async def test_a_kind_named_so_is_saved_before_it_is_finished(seen: Seen):
    run = RunState(None)
    async with WorkQueueService(default_lane="org") as work_queue:
        resumable_work = seen.resumable_work(work_queue, run)

        async def prune(_: Intent) -> None:
            seen.said.append("pruned")

        resumable_work.register("prune", prune)
        resumable_work.begin("team", Prune(reason="done"))
        await work_queue.settled()

    assert seen.said == ["save soon", "pruned", "saved", ("left", "team")]


async def test_a_step_that_is_held_stays_open_and_is_taken_up_again(seen: Seen):
    run = RunState(None)
    tried: list[int] = []
    async with WorkQueueService(default_lane="org") as work_queue:
        resumable_work = seen.resumable_work(work_queue, run)

        async def split(intent: Intent) -> None:
            tried.append(intent.id)
            if len(tried) == 1:
                raise Held

        resumable_work.register("split", split)
        resumable_work.begin("team", Split())
        await work_queue.settled()
        (held,) = run.open_intents()

        resumable_work.take_up(held.id, "team")
        await work_queue.settled()

    assert tried == [held.id, held.id]
    assert run.open_intents() == []
    assert ("let go", "split") not in seen.said


async def test_a_step_that_fails_is_finished_let_go_and_kept(seen: Seen):
    run = RunState(None)
    async with WorkQueueService(default_lane="org") as work_queue:
        resumable_work = seen.resumable_work(work_queue, run)

        async def split(_: Intent) -> None:
            raise RuntimeError("the disk is full")

        resumable_work.register("split", split)
        resumable_work.begin("team", Split())
        await work_queue.settled()

    assert seen.said == [
        "save soon",
        ("let go", "split"),
        ("failed", "split", "the disk is full"),
        ("left", "team"),
    ]
    assert run.open_intents() == []


async def test_nothing_is_carried_out_while_paused(seen: Seen):
    run = RunState(None)
    async with WorkQueueService(default_lane="org") as work_queue:
        resumable_work = seen.resumable_work(work_queue, run)

        async def split(_: Intent) -> None:
            seen.said.append("split")

        resumable_work.register("split", split)
        seen.paused = True
        resumable_work.begin("team", Split())
        await work_queue.settled()
        assert seen.said == ["save soon"]
        (kept,) = run.open_intents()

        seen.paused = False
        resumable_work.take_up(kept.id, "team")
        await work_queue.settled()

    assert "split" in seen.said


async def test_an_intent_no_step_is_registered_for_is_finished(seen: Seen):
    run = RunState(None)
    async with WorkQueueService(default_lane="org") as work_queue:
        resumable_work = seen.resumable_work(work_queue, run)
        resumable_work.begin("team", Split())
        await work_queue.settled()

    assert seen.said == ["save soon", ("left", "team")]
    assert run.open_intents() == []


async def test_a_kind_has_one_step(seen: Seen):
    async def split(_: Intent) -> None:
        pass

    resumable_work = seen.resumable_work(
        WorkQueueService(default_lane="org"), RunState(None)
    )
    resumable_work.register("split", split)

    with pytest.raises(ValueError):
        resumable_work.register("split", split)


def test_nothing_is_begun_where_nothing_is_carried_out(seen: Seen):
    run = RunState(None)
    resumable_work = seen.resumable_work(WorkQueueService(default_lane="org"), run)

    with pytest.raises(Unavailable):
        resumable_work.begin("team", Split())
    assert run.open_intents() == []
