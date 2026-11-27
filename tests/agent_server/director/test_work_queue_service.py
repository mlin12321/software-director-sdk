"""The work_queue by themselves: order within a lane, none between work_queue."""

import asyncio

import pytest

from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from openhands.sdk.errors import Unavailable


async def test_a_lane_carries_out_its_work_in_the_order_it_was_asked_for():
    done: list[int] = []

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    async def quick() -> None:
        done.append(2)

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(slow)
        work_queue.enqueue(quick)
        await work_queue.settled()

    assert done == [1, 2]


async def test_lanes_do_not_wait_for_each_other():
    done: list[str] = []
    let_go = asyncio.Event()

    async def waits() -> None:
        await let_go.wait()
        done.append("one")

    async def goes() -> None:
        done.append(work_queue.current)
        let_go.set()

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(waits, "one")
        work_queue.enqueue(goes, "other")
        await work_queue.settled()

    assert done == ["other", "one"]


async def test_settled_waits_for_what_an_effect_asks_for():
    done: list[str] = []

    async def second() -> None:
        done.append("second")

    async def first() -> None:
        work_queue.enqueue(second, "elsewhere")

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(first)
        await work_queue.settled()

        assert done == ["second"]


async def test_an_effect_that_raises_leaves_its_lane_working():
    done: list[str] = []

    async def breaks() -> None:
        raise RuntimeError("no")

    async def goes() -> None:
        done.append("after")

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(breaks)
        work_queue.enqueue(goes)
        await work_queue.settled()

    assert done == ["after"]


async def test_run_in_waits_its_turn_and_raises_what_the_effect_raises():
    done: list[str] = []

    async def before() -> None:
        await asyncio.sleep(0.01)
        done.append("before")

    async def breaks() -> None:
        done.append(work_queue.current)
        raise RuntimeError("no")

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(before, "base")
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            await work_queue.run_in("base", breaks)

    assert done == ["before", "base"]


async def test_run_in_from_the_lane_itself_does_not_wait_for_itself():
    done: list[str] = []

    async def inner() -> None:
        done.append("inner")

    async def outer() -> None:
        await work_queue.run_in("base", inner)
        done.append("outer")

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(outer, "base")
        await work_queue.settled()

    assert done == ["inner", "outer"]


async def test_after_waits_for_what_a_lane_was_asked_so_far():
    done: list[str] = []

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.append("slow")

    async with WorkQueueService(default_lane="org") as work_queue:
        work_queue.enqueue(slow, "pause")
        await work_queue.after("pause")

        assert done == ["slow"]


async def test_nothing_is_carried_out_outside_its_time():
    async def never() -> None:
        raise AssertionError("it is closed")

    work_queue = WorkQueueService(default_lane="org")
    with pytest.raises(Unavailable):
        work_queue.enqueue(never)

    started = asyncio.Event()

    async def hangs() -> None:
        started.set()
        await asyncio.Event().wait()

    async with work_queue:
        assert work_queue.is_open
        work_queue.enqueue(hangs)
        await started.wait()

    # What was under way is cut off, and nothing more is taken.
    assert not work_queue.is_open
    with pytest.raises(Unavailable):
        work_queue.enqueue(never)
