"""What a session has not had accepted yet, by itself, over a run state
in memory."""

import pytest

from openhands.agent_server.director.models import (
    FormerTeam,
    Report,
    RoutedRequest,
    TeamEvent,
    ThreadMessage,
)
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
    thread_name,
)


AT = "2026-01-01T00:00:00Z"


@pytest.fixture
def drafts() -> SessionDraftService:
    run = RunState(None)
    return SessionDraftService(run_state=lambda: run)


def event(detail: str) -> TeamEvent:
    return TeamEvent(at=AT, kind="set_up", detail=detail)


def request(name: str) -> RoutedRequest:
    return RoutedRequest(at=AT, kind="asked", team_id="t", team="T", text=name, id=name)


def former(name: str) -> FormerTeam:
    return FormerTeam(
        id=name, title=name, above="root", path=name, purpose="", ended="pruned", at=AT
    )


def test_events_are_kept_in_order_for_a_session_and_a_team(
    drafts: SessionDraftService,
):
    drafts.keep_event("one", "root", event("first"))
    drafts.keep_event("one", "root", event("second"))
    drafts.keep_event("one", "apps", event("apps"))
    drafts.keep_event("two", "root", event("other"))

    assert [e.detail for e in drafts.events("one", "root")] == ["first", "second"]
    assert drafts.events("none", "root") == []
    assert set(drafts.events_by_team("one")) == {("one", "root"), ("one", "apps")}
    assert len(drafts.events_by_team()) == 3

    drafts.forget_events("one", "root")

    assert set(drafts.events_by_team("one")) == {("one", "apps")}


def test_routed_requests_are_a_sessions(drafts: SessionDraftService):
    drafts.keep_routed("one", request("a"))
    drafts.keep_routed("two", request("b"))

    assert [kept.id for kept in drafts.routed("one")] == ["a"]
    assert sorted(kept.id for kept in drafts.routed()) == ["a", "b"]

    drafts.forget_routed("one", request("a"))

    assert [kept.id for kept in drafts.routed()] == ["b"]


def test_a_report_is_named_once_and_kept_for_its_place(
    drafts: SessionDraftService,
):
    drafts.keep_report("place", Report(at=AT, role="ic", text="Chose REST."))
    drafts.keep_report("place", Report(at=AT, role="ic", text="Again.", id="mine"))

    first, second = drafts.reports("place")
    assert first.id and second.id == "mine"
    assert drafts.reports("elsewhere") == []

    drafts.forget_reports("place")

    assert drafts.reports("place") == []


def test_removed_teams_are_kept_by_session(drafts: SessionDraftService):
    drafts.keep_former("one", former("cart"))
    drafts.keep_former("two", former("pay"))

    assert [kept.id for kept in drafts.former("one")] == ["cart"]
    # A team of the org has no session and keeps none.
    assert drafts.former("") == []

    drafts.forget_former("one", former("cart"))

    assert drafts.former("one") == []
    assert [kept.id for kept in drafts.former("two")] == ["pay"]


def test_threads_are_taken_with_the_place_that_wrote_or_read_them(
    drafts: SessionDraftService,
):
    said = ThreadMessage(at=AT, sender="ic", text="Done.")
    drafts.keep_message(thread_name("above", "place", False), said)
    drafts.keep_message(thread_name("place", "below", True), said)
    drafts.keep_message(thread_name("above", "beside", False), said)

    assert drafts.thread(thread_name("above", "place", False)) == [said]

    taken = drafts.take_threads("place")

    assert sorted(thread.file for thread in taken) == [
        ".threads-ic/place/below.json",
        ".threads/above/place.json",
    ]
    assert drafts.thread(thread_name("above", "place", False)) == []
    assert drafts.thread(thread_name("above", "beside", False)) == [said]


def test_a_document_is_passed_to_a_place_once(drafts: SessionDraftService):
    drafts.keep_passed("place", "apps", "spec")
    drafts.keep_passed("place", "apps", "spec")
    drafts.keep_passed("place", "lib", "contracts")

    assert drafts.passed_to("place") == [("apps", "spec"), ("lib", "contracts")]

    drafts.forget_passed("place")

    assert drafts.passed_to("place") == []


def test_what_was_handed_over_is_kept_in_the_order_it_was_said(
    drafts: SessionDraftService,
):
    drafts.keep_handed_over("place", {"by": "the IC", "at": AT, "text": "one"})
    drafts.keep_handed_over("place", {"by": "the IC", "at": AT, "text": "two"})

    assert [said["text"] for said in drafts.handed_over_to("place")] == ["one", "two"]

    drafts.forget_handed_over("place")

    assert drafts.handed_over_to("place") == []


def test_forgetting_a_session_leaves_the_others_and_what_places_keep(
    drafts: SessionDraftService,
):
    for session in ("one", "two"):
        drafts.keep_event(session, "root", event(session))
        drafts.keep_routed(session, request(session))
        drafts.keep_former(session, former(session))
    drafts.keep_report("root@one", Report(at=AT, role="ic", text="Kept."))

    drafts.forget_session("one")

    assert set(drafts.events_by_team()) == {("two", "root")}
    assert [kept.id for kept in drafts.routed()] == ["two"]
    assert drafts.former("one") == [] and len(drafts.former("two")) == 1
    assert len(drafts.reports("root@one")) == 1
