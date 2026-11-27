"""When the stall watch looks at nothing, with collaborators made by hand
and no service. What it finds in a tree is tested through the service
(``test_director_stalls.py``)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from openhands.agent_server.director.environment.git import (
    GitCheckouts,
)
from openhands.agent_server.director.models import DirectorSettings
from openhands.agent_server.director.service.model_call_service import (
    ModelCallService,
)
from openhands.agent_server.director.service.stall_watch_service import (
    StallWatchService,
    progress_of,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService


def watch(looked: list[str], **given: Any) -> StallWatchService:
    def never(*said: Any, **named: Any) -> None:
        raise AssertionError(f"nothing is said or kept: {said} {named}")

    def org() -> Any:
        looked.append("org")
        return SimpleNamespace(places=lambda: [])

    settings = DirectorSettings(
        soft_check_in_minutes=given.get("soft", 10),
        hard_check_in_minutes=given.get("hard", 30),
    )
    return StallWatchService(
        checkouts=GitCheckouts(),
        org=org,
        has_org=lambda: given.get("has_org", True),
        settings=lambda: settings,
        run_state=lambda: cast(
            Any, SimpleNamespace(marked=lambda kind: {}, unmark=never, mark=never)
        ),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        paused=lambda: given.get("paused", False),
        running=lambda: set(),
        model_calls=cast(ModelCallService, None),
        check_managers=lambda: looked.append("managers"),
        work_queue=WorkQueueService(default_lane="org"),
        progressed=never,
        release_reviews=never,
        record_event=never,
        tell_writer=never,
        tell_above=never,
    )


@pytest.mark.parametrize(
    "given",
    [{"has_org": False}, {"paused": True}, {"soft": 0, "hard": 0}],
    ids=["no org is loaded", "the org is paused", "both checks are off"],
)
async def test_nothing_is_looked_at(given: dict[str, Any]):
    looked: list[str] = []

    await watch(looked, **given).check_stalls()

    assert looked == []


async def test_an_org_without_teams_is_looked_at_and_nobody_is_told():
    looked: list[str] = []

    await watch(looked).check_stalls()

    assert sorted(looked) == ["managers", "org"]


def test_what_was_kept_of_a_teams_progress_is_read_back_or_is_nothing():
    assert progress_of('{"at": "2026-01-01T00:00:00"}') == {"at": "2026-01-01T00:00:00"}
    assert progress_of(None) == progress_of("not json") == progress_of("[1]") == {}
