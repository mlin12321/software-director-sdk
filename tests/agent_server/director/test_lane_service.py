"""The lanes by themselves: order within a lane, none between lanes."""

import asyncio

import pytest

from openhands.agent_server.director.service.lane_service import LaneService
from openhands.sdk.errors import Unavailable


async def test_a_lane_carries_out_its_work_in_the_order_it_was_asked_for():
    done: list[int] = []

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.append(1)

    async def quick() -> None:
        done.append(2)

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(slow)
        lanes.enqueue(quick)
        await lanes.settled()

    assert done == [1, 2]


async def test_lanes_do_not_wait_for_each_other():
    done: list[str] = []
    let_go = asyncio.Event()

    async def waits() -> None:
        await let_go.wait()
        done.append("one")

    async def goes() -> None:
        done.append(lanes.current)
        let_go.set()

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(waits, "one")
        lanes.enqueue(goes, "other")
        await lanes.settled()

    assert done == ["other", "one"]


async def test_settled_waits_for_what_an_effect_asks_for():
    done: list[str] = []

    async def second() -> None:
        done.append("second")

    async def first() -> None:
        lanes.enqueue(second, "elsewhere")

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(first)
        await lanes.settled()

        assert done == ["second"]


async def test_an_effect_that_raises_leaves_its_lane_working():
    done: list[str] = []

    async def breaks() -> None:
        raise RuntimeError("no")

    async def goes() -> None:
        done.append("after")

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(breaks)
        lanes.enqueue(goes)
        await lanes.settled()

    assert done == ["after"]


async def test_run_in_waits_its_turn_and_raises_what_the_effect_raises():
    done: list[str] = []

    async def before() -> None:
        await asyncio.sleep(0.01)
        done.append("before")

    async def breaks() -> None:
        done.append(lanes.current)
        raise RuntimeError("no")

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(before, "base")
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            await lanes.run_in("base", breaks)

    assert done == ["before", "base"]


async def test_run_in_from_the_lane_itself_does_not_wait_for_itself():
    done: list[str] = []

    async def inner() -> None:
        done.append("inner")

    async def outer() -> None:
        await lanes.run_in("base", inner)
        done.append("outer")

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(outer, "base")
        await lanes.settled()

    assert done == ["inner", "outer"]


async def test_after_waits_for_what_a_lane_was_asked_so_far():
    done: list[str] = []

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.append("slow")

    async with LaneService(default_lane="org") as lanes:
        lanes.enqueue(slow, "pause")
        await lanes.after("pause")

        assert done == ["slow"]


async def test_nothing_is_carried_out_outside_its_time():
    async def never() -> None:
        raise AssertionError("it is closed")

    lanes = LaneService(default_lane="org")
    with pytest.raises(Unavailable):
        lanes.enqueue(never)

    started = asyncio.Event()

    async def hangs() -> None:
        started.set()
        await asyncio.Event().wait()

    async with lanes:
        assert lanes.is_open
        lanes.enqueue(hangs)
        await started.wait()

    # What was under way is cut off, and nothing more is taken.
    assert not lanes.is_open
    with pytest.raises(Unavailable):
        lanes.enqueue(never)
