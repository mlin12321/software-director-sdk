"""The line teams wait in to start, by itself: a store of teams and the
state of a run kept in memory."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import WAITING
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.resumable_work_service import Held
from openhands.agent_server.director.service.start_line_service import (
    StartLineService,
)
from openhands.agent_server.event_service import ConversationRunLimitExceeded
from tests.agent_server.director.conftest import (
    open_session_in,
    shipped_type,
    small_tree,
)


@pytest.fixture
def line(tmp_path: Path) -> StartLineService:
    store = OrgStore(shipped_type)
    for node in small_tree():
        store.upsert(node)
    open_session_in(store, "root", "s1")
    org_state = OrgStateService(
        repo_dir=lambda: tmp_path, clock=lambda: datetime.now(UTC)
    )
    return StartLineService(org=lambda: store, org_state=org_state, retry=60)


def full_until(room: asyncio.Event, started: list[str], name: str):
    async def start() -> str:
        if not room.is_set():
            raise ConversationRunLimitExceeded(1)
        started.append(name)
        return name

    return start


async def test_a_team_starts_at_once_when_there_is_room(line: StartLineService):
    async def start() -> str:
        return "started"

    assert await line.when_there_is_room("a@s1", start) == "started"
    assert line.org_state.run_state().marked(WAITING) == {}


async def test_a_team_waits_while_the_server_is_full_and_starts_when_it_has_room(
    line: StartLineService,
):
    room, started = asyncio.Event(), []
    waits = asyncio.create_task(
        line.when_there_is_room("a@s1", full_until(room, started, "a"))
    )
    await asyncio.sleep(0)
    place = line.org().place("a@s1")
    assert place is not None and place.waiting_to_start
    assert "a@s1" in line.org_state.run_state().marked(WAITING)

    room.set()
    line.room_made()

    assert await waits == "a"
    place = line.org().place("a@s1")
    assert place is not None and not place.waiting_to_start
    assert line.org_state.run_state().marked(WAITING) == {}


async def test_starts_are_served_in_the_order_they_began_to_wait(
    line: StartLineService,
):
    room, started = asyncio.Event(), []
    first = asyncio.create_task(
        line.when_there_is_room("a@s1", full_until(room, started, "a"))
    )
    await asyncio.sleep(0)
    second = asyncio.create_task(
        line.when_there_is_room("b@s1", full_until(room, started, "b"))
    )
    await asyncio.sleep(0)

    room.set()
    line.room_made()
    await asyncio.gather(first, second)

    assert started == ["a", "b"]


async def test_a_team_that_waits_keeps_its_place_when_the_org_is_paused(
    line: StartLineService,
):
    room, started = asyncio.Event(), []
    waits = asyncio.create_task(
        line.when_there_is_room("a@s1", full_until(room, started, "a"))
    )
    await asyncio.sleep(0)

    line.org_state.hold("human")
    line.org_held()

    with pytest.raises(Held):
        await waits
    assert "a@s1" in line.org_state.run_state().marked(WAITING)
    assert started == []


async def test_a_team_behind_another_in_line_does_not_start_before_it(
    line: StartLineService,
):
    line.line_up(["a@s1", "b@s1"])
    started: list[str] = []
    room = asyncio.Event()
    room.set()
    behind = asyncio.create_task(
        line.when_there_is_room("b@s1", full_until(room, started, "b"))
    )
    await asyncio.sleep(0)
    assert started == []

    await line.when_there_is_room("a@s1", full_until(room, started, "a"))
    await behind

    assert started == ["a", "b"]
