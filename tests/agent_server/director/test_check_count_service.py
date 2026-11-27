"""What a Manager's checks count, by itself: a store of teams and a run
state in memory."""

import json

import pytest

from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import CHECKS, RunState
from openhands.agent_server.director.service.check_count_service import (
    CheckCountService,
)
from tests.agent_server.director.conftest import (
    open_session_in,
    shipped_type,
    small_tree,
)


ROOT, A, B = "root@s1", "a@s1", "b@s1"


@pytest.fixture
def counts() -> CheckCountService:
    store = OrgStore(shipped_type)
    for node in small_tree():
        store.upsert(node)
    open_session_in(store, "root", "s1")
    run = RunState(None)
    return CheckCountService(org=lambda: store, run_state=lambda: run)


def keep(counts: CheckCountService, place_id: str, kept: object) -> None:
    counts.run_state().mark(CHECKS, place_id, json.dumps(kept))


def test_nothing_is_counted_where_nothing_was_kept(counts: CheckCountService):
    assert counts.counted(ROOT) == {"accepted": 0, "escalations": {}}


def test_escalations_of_teams_no_longer_below_are_left_out(
    counts: CheckCountService,
):
    keep(counts, ROOT, {"accepted": 2, "escalations": {A: 1, "gone@s1": 4}})

    assert counts.counted(ROOT) == {"accepted": 2, "escalations": {A: 1}}


def test_what_cannot_be_read_counts_as_nothing(counts: CheckCountService):
    counts.run_state().mark(CHECKS, ROOT, "not json")

    assert counts.counted(ROOT) == {"accepted": 0, "escalations": {}}


def test_accepted_work_ends_the_count_of_its_teams_escalations(
    counts: CheckCountService,
):
    keep(counts, ROOT, {"accepted": 1, "escalations": {A: 2, B: 1}})
    team = counts.org().place(A)
    assert team is not None

    counts.work_accepted(team)

    assert counts.counted(ROOT) == {"accepted": 1, "escalations": {B: 1}}
