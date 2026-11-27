"""How team memory is laid out in a store, on a store that keeps its
files in memory: nothing of it needs git."""

from collections.abc import Callable, Collection

from openhands.agent_server.director.environment.records import (
    Clash,
    Merged,
    RecordStore,
    Staged,
    StoreChange,
    StoreReader,
    reports_file,
)
from openhands.agent_server.director.models import Report, TeamEvent
from tests.agent_server.director.conftest import take_in


class StoreInMemory(RecordStore):
    """Every version is a whole copy of the files; a version is named by
    its number."""

    def __init__(self) -> None:
        self.versions: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return "memory"

    def head(self) -> str | None:
        return str(len(self.versions) - 1) if self.versions else None

    def read_at(self, commit: str | None, path: str) -> str | None:
        return None if commit is None else self.versions[int(commit)].get(path)

    def files(self, directory: str) -> list[str]:
        now = self.versions[-1] if self.versions else {}
        return sorted(path for path in now if path.startswith(f"{directory}/"))

    def _apply(self, changes: list[StoreChange]) -> str:
        files = dict(self.versions[-1]) if self.versions else {}
        for path, content in changes:
            if content is None:
                files.pop(path, None)
            else:
                files[path] = content
        self.versions.append(files)
        return str(len(self.versions) - 1)

    def commit(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
    ) -> str | None:
        parent = self.head()
        changes = mutate(lambda path: self.read_at(parent, path))
        return self._apply(changes) if changes else parent

    def land(
        self, changes: list[StoreChange], message: str, *, merging: str | None = None
    ) -> bool:
        self._apply(changes)
        return True

    def stage(
        self, mutate: Callable[[StoreReader], list[StoreChange]], message: str
    ) -> Staged | None:
        raise NotImplementedError

    def clashes(self, theirs_ref: str) -> list[Clash]:
        raise NotImplementedError

    def merge(self, theirs_ref: str, take_theirs: Collection[str] = ()) -> Merged:
        raise NotImplementedError

    def move_to(self, commit: str) -> bool:
        raise NotImplementedError

    def contains(self, other: str) -> bool:
        return self.head() is not None and int(other) <= len(self.versions) - 1

    def base_with(self, other: str) -> str | None:
        return other if self.contains(other) else None


def report(role: str, text: str, at: str) -> Report:
    return Report(at=at, role=role, text=text)


def test_a_file_is_written_inside_the_folder_it_is_written_for():
    store = StoreInMemory()

    store.write("services/api", "notes.md", "First.")

    assert store.read("services/api/notes.md") == "First."
    assert store.head() == "0"


def test_what_a_team_recorded_is_read_by_member_and_replaced_only_as_it_was_read():
    store = StoreInMemory()
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
    store = StoreInMemory()
    event = TeamEvent(id="e1", at="2026-01-01T00:00:00Z", kind="accepted", team="api")

    take_in(store, events={"root": [event]})
    take_in(store, events={"root": [event]})

    assert store.history("root") == [event]
    assert store.routed() == []
    assert store.former("api") == []
