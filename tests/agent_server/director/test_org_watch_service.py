"""When the org watch looks at nothing, with what it holds made by hand
and no director. What it finds in a tree is tested through the service
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
from openhands.agent_server.director.service.org_state_service import progress_of
from openhands.agent_server.director.service.org_watch_service import (
    OrgWatchService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService


def watch(looked: list[str], **given: Any) -> OrgWatchService:
    def never(*said: Any, **named: Any) -> None:
        raise AssertionError(f"nothing is said or kept: {said} {named}")

    def org() -> Any:
        looked.append("org")
        return SimpleNamespace(places=lambda: [])

    settings = DirectorSettings(
        soft_check_in_minutes=given.get("soft", 10),
        hard_check_in_minutes=given.get("hard", 30),
    )
    silent = SimpleNamespace(
        progressed=never, tell_writer=never, tell_above=never, record_event=never
    )
    return OrgWatchService(
        org=org,
        has_org=lambda: given.get("has_org", True),
        checkouts=GitCheckouts(),
        launcher=cast(Any, never),
        poll_seconds=60,
        org_state=cast(
            Any,
            SimpleNamespace(
                paused="human" if given.get("paused") else None,
                running=set(),
                clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
                run_state=lambda: SimpleNamespace(
                    marked=lambda kind: {}, unmark=never, mark=never
                ),
            ),
        ),
        org_settings=cast(Any, SimpleNamespace(settings=settings)),
        org_save=cast(Any, None),
        team_types=cast(Any, None),
        team_history=cast(Any, silent),
        team_messaging=cast(Any, silent),
        reviews=cast(Any, SimpleNamespace(release_held=never)),
        session_merges=cast(
            Any, SimpleNamespace(check_managers=lambda: looked.append("managers"))
        ),
        model_calls=cast(ModelCallService, None),
        work_queue=WorkQueueService(default_lane="org"),
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
