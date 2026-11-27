"""The history of the org's teams by itself: a repository, a store of
teams and a run state in memory, and no director."""

from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.models import Review, SessionTeam
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.prompts import NEED_WRITE
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from tests.agent_server.director.conftest import (
    open_session_in,
    shipped_type,
    small_tree,
)


SESSION = "s1"


class History:
    def __init__(self, repo: Path) -> None:
        run = RunState(None)
        self.store = OrgStore(shipped_type)
        for node in small_tree():
            self.store.upsert(node)
        open_session_in(self.store, "root", SESSION)
        self.service = TeamHistoryService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            require_team_memory=lambda: GitStore(repo),
            session_drafts=SessionDraftService(run_state=lambda: run),
        )

    def place(self, team_id: str) -> SessionTeam:
        found = self.store.place(f"{team_id}@{SESSION}")
        assert found is not None
        return found


@pytest.fixture
def history(repo: Path) -> History:
    return History(repo)


def test_an_event_is_read_in_the_record_of_the_team_above(history: History):
    a = history.place("a")

    history.service.record_event(a.parent_id, "woken", a, "it stood still")

    [event] = history.service.history_of(f"root@{SESSION}")
    assert (event.kind, event.team_id, event.detail) == ("woken", "a", "it stood still")
    assert event.unaccepted


def test_a_change_to_the_org_is_noted_only_once_a_sessions_work_is_accepted(
    history: History,
):
    a = history.place("a")
    history.service.record_event(a.parent_id, "pruned", a)
    assert history.service.changed_since_accept == {}

    history.store.set_review(
        f"root@{SESSION}",
        Review(state="accepted", seq=1, round=1, at="2026-01-02T00:00:00+00:00"),
    )
    history.service.record_event(a.parent_id, "woken", a, "it stood still")
    assert history.service.changed_since_accept == {}
    history.service.record_event(a.parent_id, "pruned", a)

    assert history.service.changed_since_accept == {SESSION: False}


def test_nothing_is_kept_for_the_team_above_the_top(history: History):
    history.service.record_event(None, "woken", history.place("root"))

    assert history.service.org_history(10).events == []


def test_what_is_landed_is_the_orgs_and_read_before_what_a_session_kept(
    history: History,
):
    a, b = history.place("a"), history.place("b")
    history.service.record_event(a.parent_id, "woken", a)

    history.service.land_event("root", history.service.event("set_up", b))

    assert [event.kind for event in history.service.history_of("root")] == ["set_up"]
    kinds = [
        (event.kind, event.unaccepted)
        for event in history.service.org_history(10).events
    ]
    assert sorted(kinds) == [("set_up", False), ("woken", True)]


def test_a_file_needed_the_same_way_twice_is_kept_once(history: History):
    a = history.place("a")

    history.service.record_need(a, "svc/b/notes.txt", NEED_WRITE)
    history.service.record_need(a, "svc/b/notes.txt", NEED_WRITE)
    history.service.record_need(a, "elsewhere/notes.txt", NEED_WRITE)

    [needed] = history.service.history_of(f"root@{SESSION}")
    assert (needed.kind, needed.other_id, needed.paths) == (
        "needed",
        "b",
        ["svc/b/notes.txt"],
    )
    # No event of the org: it is counted for the IC above alone.
    assert history.service.org_history(10).events == []


def test_a_request_sent_straight_to_a_team_is_kept_with_an_event_above(
    history: History,
):
    a = history.place("a")

    history.service.keep_routed("asked", a, "Fix the parser", "model", "its folder")

    [request] = history.service.routed_requests()
    assert (request.team_id, request.text) == (a.id, "Fix the parser")
    [event] = history.service.history_of(f"root@{SESSION}")
    assert (event.kind, event.detail) == ("asked", "Fix the parser")


def test_a_page_of_the_orgs_history_says_whether_there_is_more(history: History):
    a = history.place("a")
    for day, detail in enumerate(("first", "second", "third"), start=1):
        history.service.land_event(
            "root",
            history.service.event("woken", a, detail).model_copy(
                update={"at": f"2026-01-0{day}T00:00:00Z"}
            ),
        )

    page = history.service.org_history(2)

    assert [event.detail for event in page.events] == ["third", "second"]
    assert page.earlier
    assert [
        event.detail
        for event in history.service.org_history(2, page.events[-1].at).events
    ] == ["first"]
