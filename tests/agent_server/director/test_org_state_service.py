"""The state of an org's run by itself: a repository and a clock."""

import json
from datetime import UTC, datetime
from pathlib import Path

from openhands.agent_server.director.environment.git import (
    GitCodingEnvironment,
)
from openhands.agent_server.director.models import OrgStateEvent, RecoveryItem
from openhands.agent_server.director.run_state import PROGRESS, RUNNING
from openhands.agent_server.director.service.org_state_service import OrgStateService


NOW = datetime(2026, 1, 2, tzinfo=UTC)


def state_of(repo: Path) -> OrgStateService:
    return OrgStateService(
        repo_dir=lambda: repo, clock=lambda: NOW, environment=GitCodingEnvironment()
    )


def test_a_pause_is_kept_for_the_next_start_of_the_server(repo: Path):
    first = state_of(repo)
    first.hold("human")
    first.close()

    again = state_of(repo)

    assert again.state() == OrgStateEvent(paused="human", merging=False)
    assert again.paused == "human"


def test_those_who_watch_are_told_of_a_pause_a_resume_and_a_merge(repo: Path):
    org_state = state_of(repo)
    told: list[OrgStateEvent] = []
    unwatch = org_state.watch(told.append)

    org_state.hold("human")
    org_state.release()
    org_state.set_merge({"remote": "origin"})
    unwatch()
    org_state.set_merge(None)

    assert told == [
        OrgStateEvent(paused="human", merging=False),
        OrgStateEvent(paused=None, merging=False),
        OrgStateEvent(paused=None, merging=True),
    ]
    assert org_state.merging() is None


def test_the_time_an_org_was_paused_counts_against_no_team(repo: Path):
    org_state = state_of(repo)
    long_ago = {"at": "2025-01-01T00:00:00+00:00", "head": "abc", "checked": "x"}
    org_state.run_state().mark(PROGRESS, "a@s1", json.dumps(long_ago))
    org_state.hold("human")

    org_state.release()

    seen = json.loads(org_state.run_state().marked(PROGRESS)["a@s1"])
    assert (seen["at"], seen["head"]) == (NOW.isoformat(), "abc")


def test_resuming_forgets_what_was_put_aside(repo: Path):
    org_state = state_of(repo)
    item = RecoveryItem(
        kind="dropped", node_id="a", title="A", detail="gone"
    ).model_dump()
    org_state.run_state().put("dropped", json.dumps([item]))
    assert [found.node_id for found in org_state.dropped()] == ["a"]

    org_state.release()

    assert org_state.dropped() == []


def test_the_run_state_follows_the_org_to_another_repository(
    repo: Path, tmp_path: Path
):
    where = [repo]
    org_state = OrgStateService(
        repo_dir=lambda: where[0], clock=lambda: NOW, environment=GitCodingEnvironment()
    )
    org_state.run_state().mark(RUNNING, "c1", "a@s1")
    org_state.hold("human")

    # Not a repository: its run state is kept in memory only.
    where[0] = tmp_path

    assert org_state.run_state().marked(RUNNING) == {}
    assert org_state.paused is None and org_state.running == set()
    where[0] = repo
    assert org_state.state().paused == "human"
    assert org_state.running == {"c1"}


def test_a_pause_changed_in_the_run_state_itself_is_read_again(repo: Path):
    org_state = state_of(repo)
    org_state.hold("crash")
    org_state.run_state().put("paused", None)

    org_state.refresh()

    assert org_state.paused is None
