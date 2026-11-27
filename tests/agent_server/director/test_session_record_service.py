"""A session's own records by themselves, over a run state in memory."""

import pytest

from openhands.agent_server.director.models import (
    FormerTeam,
    Report,
    RoutedRequest,
    TeamEvent,
    ThreadMessage,
)
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service.session_record_service import (
    SessionRecordService,
    thread_name,
)


AT = "2026-01-01T00:00:00Z"


@pytest.fixture
def records() -> SessionRecordService:
    run = RunState(None)
    return SessionRecordService(run_state=lambda: run)


def event(detail: str) -> TeamEvent:
    return TeamEvent(at=AT, kind="set_up", detail=detail)


def request(name: str) -> RoutedRequest:
    return RoutedRequest(at=AT, kind="asked", team_id="t", team="T", text=name, id=name)


def former(name: str) -> FormerTeam:
    return FormerTeam(
        id=name, title=name, above="root", path=name, purpose="", ended="pruned", at=AT
    )


def test_events_are_kept_in_order_for_a_session_and_a_team(
    records: SessionRecordService,
):
    records.keep_event("one", "root", event("first"))
    records.keep_event("one", "root", event("second"))
    records.keep_event("one", "apps", event("apps"))
    records.keep_event("two", "root", event("other"))

    assert [e.detail for e in records.events("one", "root")] == ["first", "second"]
    assert records.events("none", "root") == []
    assert set(records.events_by_team("one")) == {("one", "root"), ("one", "apps")}
    assert len(records.events_by_team()) == 3

    records.forget_events("one", "root")

    assert set(records.events_by_team("one")) == {("one", "apps")}


def test_routed_requests_are_a_sessions(records: SessionRecordService):
    records.keep_routed("one", request("a"))
    records.keep_routed("two", request("b"))

    assert [kept.id for kept in records.routed("one")] == ["a"]
    assert sorted(kept.id for kept in records.routed()) == ["a", "b"]

    records.forget_routed("one", request("a"))

    assert [kept.id for kept in records.routed()] == ["b"]


def test_a_report_is_named_once_and_kept_for_its_place(
    records: SessionRecordService,
):
    records.keep_report("place", Report(at=AT, role="ic", text="Chose REST."))
    records.keep_report("place", Report(at=AT, role="ic", text="Again.", id="mine"))

    first, second = records.reports("place")
    assert first.id and second.id == "mine"
    assert records.reports("elsewhere") == []

    records.forget_reports("place")

    assert records.reports("place") == []


def test_removed_teams_are_kept_by_session(records: SessionRecordService):
    records.keep_former("one", former("cart"))
    records.keep_former("two", former("pay"))

    assert [kept.id for kept in records.former("one")] == ["cart"]
    # A team of the org has no session and keeps none.
    assert records.former("") == []

    records.forget_former("one", former("cart"))

    assert records.former("one") == []
    assert [kept.id for kept in records.former("two")] == ["pay"]


def test_threads_are_taken_with_the_place_that_wrote_or_read_them(
    records: SessionRecordService,
):
    said = ThreadMessage(at=AT, sender="ic", text="Done.")
    records.keep_message(thread_name("above", "place", False), said)
    records.keep_message(thread_name("place", "below", True), said)
    records.keep_message(thread_name("above", "beside", False), said)

    assert records.thread(thread_name("above", "place", False)) == [said]

    taken = records.take_threads("place")

    assert sorted(thread.file for thread in taken) == [
        ".threads-ic/place/below.json",
        ".threads/above/place.json",
    ]
    assert records.thread(thread_name("above", "place", False)) == []
    assert records.thread(thread_name("above", "beside", False)) == [said]


def test_a_document_is_passed_to_a_place_once(records: SessionRecordService):
    records.keep_passed("place", "apps", "spec")
    records.keep_passed("place", "apps", "spec")
    records.keep_passed("place", "lib", "contracts")

    assert records.passed_to("place") == [("apps", "spec"), ("lib", "contracts")]

    records.forget_passed("place")

    assert records.passed_to("place") == []


def test_what_was_handed_over_is_kept_in_the_order_it_was_said(
    records: SessionRecordService,
):
    records.keep_handed_over("place", {"by": "the IC", "at": AT, "text": "one"})
    records.keep_handed_over("place", {"by": "the IC", "at": AT, "text": "two"})

    assert [said["text"] for said in records.handed_over_to("place")] == ["one", "two"]

    records.forget_handed_over("place")

    assert records.handed_over_to("place") == []


def test_forgetting_a_session_leaves_the_others_and_what_places_keep(
    records: SessionRecordService,
):
    for session in ("one", "two"):
        records.keep_event(session, "root", event(session))
        records.keep_routed(session, request(session))
        records.keep_former(session, former(session))
    records.keep_report("root@one", Report(at=AT, role="ic", text="Kept."))

    records.forget_session("one")

    assert set(records.events_by_team()) == {("two", "root")}
    assert [kept.id for kept in records.routed()] == ["two"]
    assert records.former("one") == [] and len(records.former("two")) == 1
    assert len(records.reports("root@one")) == 1
