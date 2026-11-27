"""How team memory is laid out in a store, on a store that keeps its
files in memory: nothing of it needs git."""

from openhands.agent_server.director.environment.memory import (
    TEAM_MEMORY,
    MemoryRecordStore,
    MemoryRepository,
)
from openhands.agent_server.director.environment.records import (
    reports_file,
)
from openhands.agent_server.director.models import Report, TeamEvent
from tests.agent_server.director.conftest import take_in


def a_store() -> MemoryRecordStore:
    return MemoryRepository().store(TEAM_MEMORY)


def report(role: str, text: str, at: str) -> Report:
    return Report(at=at, role=role, text=text)


def test_a_file_is_written_inside_the_folder_it_is_written_for():
    store = a_store()

    store.write("services/api", "notes.md", "First.")

    assert store.read("services/api/notes.md") == "First."
    assert store.head() is not None


def test_what_a_team_recorded_is_read_by_member_and_replaced_only_as_it_was_read():
    store = a_store()
    first = report("ic", "Chose sqlite.", "2026-01-01T00:00:00Z")
    other = report("manager", "Checked in.", "2026-01-02T00:00:00Z")
    take_in(store, reports={"services/api": [first, other]})

    assert store.reports("services/api") == [first, other]
    assert store.reports("services/api", "ic") == [first]

    shorter = report("ic", "sqlite.", "2026-01-03T00:00:00Z")
    assert not store.replace_reports("services/api", "ic", [], [shorter])
    assert store.replace_reports("services/api", "ic", [first], [shorter])
    assert store.reports("services/api") == [other, shorter]
    assert store.read(reports_file("services/api")) is not None


def test_what_became_of_the_teams_below_is_added_once():
    store = a_store()
    event = TeamEvent(id="e1", at="2026-01-01T00:00:00Z", kind="accepted", team="api")

    take_in(store, events={"root": [event]})
    take_in(store, events={"root": [event]})

    assert store.history("root") == [event]
    assert store.routed() == []
    assert store.former("api") == []
