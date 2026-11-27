"""What every store of records does, whatever keeps it: the same tests
against git and against memory, so that the two cannot drift and the
interface says nothing only git can do."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.environment.git.cli import git_version
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.environment.memory import MemoryRepository
from openhands.agent_server.director.environment.records import (
    RecordStore,
    Staged,
    StoreChange,
    StoreReader,
)
from openhands.agent_server.director.models import Report


@dataclass
class Kept:
    """Two stores of one repository, and how staged changes land in both."""

    ours: RecordStore
    theirs: RecordStore
    land_together: Callable[[Sequence[Staged]], bool]


def in_git(repo: Path) -> Kept:
    if git_version(repo) < (2, 38):
        pytest.skip("a merge needs git 2.38")
    environment = GitCodingEnvironment()
    return Kept(
        GitStore(repo),
        GitStore(repo, "refs/agents/theirs"),
        lambda staged: environment.land_together(repo, staged),
    )


def in_memory(repo: Path) -> Kept:
    repository = MemoryRepository()
    return Kept(
        repository.store("ours"), repository.store("theirs"), repository.land_together
    )


@pytest.fixture(params=[in_git, in_memory], ids=["git", "memory"])
def kept(request: pytest.FixtureRequest, repo: Path) -> Kept:
    return request.param(repo)


@pytest.fixture
def store(kept: Kept) -> RecordStore:
    return kept.ours


def writing(
    path: str, content: str | None
) -> Callable[[StoreReader], list[StoreChange]]:
    return lambda read: [StoreChange(path, content)]


def a_report(text: str, at: str) -> Report:
    return Report(at=at, role="ic", text=text)


def test_nothing_is_read_before_the_first_write(store: RecordStore):
    assert store.head() is None
    assert store.read("svc/a/spec.md") is None
    assert store.read_at(None, "svc/a/spec.md") is None
    assert store.files("svc") == []


def test_what_is_written_is_read_back_now_and_as_it_was(store: RecordStore):
    first = store.commit(writing("svc/a/spec.md", "one"), "first")
    second = store.commit(writing("svc/a/spec.md", "two"), "second")

    assert first is not None and second is not None and first != second
    assert store.head() == second
    assert store.read("svc/a/spec.md") == "two"
    assert store.read_at(first, "svc/a/spec.md") == "one"


def test_the_files_under_a_directory_are_listed(store: RecordStore):
    store.commit(
        lambda read: [
            StoreChange("svc/a/spec.md", "A"),
            StoreChange("svc/a/notes/one.md", "1"),
            StoreChange("svc/ab/spec.md", "AB"),
        ],
        "three files",
    )

    assert sorted(store.files("svc/a")) == ["svc/a/notes/one.md", "svc/a/spec.md"]


def test_a_mutation_reads_what_it_builds_on(store: RecordStore):
    store.commit(writing("count", "1"), "first")

    store.commit(
        lambda read: [StoreChange("count", f"{int(read('count') or 0) + 1}")], "next"
    )

    assert store.read("count") == "2"


def test_a_mutation_without_changes_makes_no_version(store: RecordStore):
    head = store.commit(writing("svc/a/spec.md", "one"), "first")

    assert store.commit(lambda read: [], "nothing") == head
    assert store.stage(lambda read: [], "nothing") is None
    assert store.head() == head


def test_a_change_with_no_content_removes_the_file(store: RecordStore):
    store.commit(writing("svc/a/spec.md", "one"), "first")

    store.commit(writing("svc/a/spec.md", None), "remove")

    assert store.read("svc/a/spec.md") is None
    assert store.files("svc/a") == []


def test_what_is_staged_is_not_the_stores_until_it_lands(kept: Kept):
    head = kept.ours.commit(writing("a", "1"), "first")

    staged = kept.ours.stage(writing("a", "2"), "second")

    assert staged is not None and staged.parent == head
    assert kept.ours.head() == head and kept.ours.read("a") == "1"
    assert kept.ours.read_at(staged.commit, "a") == "2"


def test_staged_changes_to_two_stores_land_together(kept: Kept):
    one = kept.ours.stage(writing("a", "ours"), "ours")
    other = kept.theirs.stage(writing("a", "theirs"), "theirs")
    assert one is not None and other is not None

    assert kept.land_together([one, other])

    assert (kept.ours.read("a"), kept.theirs.read("a")) == ("ours", "theirs")


def test_nothing_lands_when_one_store_changed_meanwhile(kept: Kept):
    one = kept.ours.stage(writing("a", "ours"), "ours")
    other = kept.theirs.stage(writing("a", "theirs"), "theirs")
    assert one is not None and other is not None
    kept.theirs.commit(writing("b", "someone else"), "meanwhile")

    assert not kept.land_together([one, other])

    assert kept.ours.head() is None
    assert kept.theirs.read("a") is None


def test_changes_land_once_on_the_head(store: RecordStore):
    assert store.land([StoreChange("a", "1")], "first")
    assert store.land([StoreChange("b", "2")], "second")

    assert (store.read("a"), store.read("b")) == ("1", "2")


def test_what_lands_as_a_merge_has_the_other_in_its_history(kept: Kept):
    kept.ours.commit(writing("a", "1"), "ours")
    theirs = kept.theirs.commit(writing("b", "2"), "theirs")
    assert theirs is not None and not kept.ours.contains(theirs)

    assert kept.ours.land([StoreChange("b", "2")], "merge", merging=theirs)

    assert kept.ours.contains(theirs)


def test_a_store_moves_only_to_a_version_that_comes_from_its_head(kept: Kept):
    behind = kept.ours.commit(writing("a", "1"), "first")
    staged = kept.ours.stage(writing("a", "2"), "ahead")
    apart = kept.theirs.commit(writing("a", "theirs"), "apart")
    assert behind is not None and staged is not None and apart is not None

    assert not kept.ours.move_to(apart)
    assert kept.ours.move_to(staged.commit)
    assert kept.ours.read("a") == "2"
    assert not kept.ours.move_to(behind)
    assert kept.ours.head() == staged.commit


def test_a_store_has_in_its_history_what_it_was(store: RecordStore):
    first = store.commit(writing("a", "1"), "first")
    second = store.commit(writing("a", "2"), "second")
    assert first is not None and second is not None

    assert store.contains(first) and store.contains(second)
    assert store.base_with(first) == first


def test_two_stores_come_from_the_version_they_last_shared(kept: Kept):
    shared = kept.ours.commit(writing("a", "1"), "shared")
    assert shared is not None
    kept.theirs.merge(kept.ours.name)
    kept.ours.commit(writing("b", "ours"), "ours")
    kept.theirs.commit(writing("c", "theirs"), "theirs")

    assert kept.ours.base_with(kept.theirs.name) == shared
    assert kept.theirs.contains(shared)
    assert not kept.ours.contains(kept.theirs.head() or "")


def test_an_empty_store_becomes_the_one_merged_into_it(kept: Kept):
    theirs = kept.theirs.commit(writing("a", "theirs"), "theirs")

    merged = kept.ours.merge(kept.theirs.name)

    assert merged.head == theirs == kept.ours.head()
    assert merged.kept_local == []


def test_merging_what_is_already_merged_changes_nothing(kept: Kept):
    kept.ours.commit(writing("a", "ours"), "ours")
    kept.theirs.commit(writing("b", "theirs"), "theirs")
    first = kept.ours.merge(kept.theirs.name)

    again = kept.ours.merge(kept.theirs.name)

    assert again.head == first.head == kept.ours.head()


def test_a_merge_combines_changes_to_different_files(kept: Kept):
    kept.ours.write("svc/a", "spec.md", "A")
    kept.theirs.write("svc/b", "spec.md", "B")

    merged = kept.ours.merge(kept.theirs.name)

    assert kept.ours.read("svc/a/spec.md") == "A"
    assert kept.ours.read("svc/b/spec.md") == "B"
    assert merged.kept_local == []
    assert kept.ours.contains(kept.theirs.head() or "")


def test_a_merge_takes_what_only_the_other_side_changed_or_removed(kept: Kept):
    kept.ours.commit(
        lambda read: [StoreChange("stays", "1"), StoreChange("goes", "1")], "shared"
    )
    kept.theirs.merge(kept.ours.name)
    kept.theirs.commit(
        lambda read: [StoreChange("stays", "2"), StoreChange("goes", None)], "theirs"
    )
    kept.ours.commit(writing("mine", "1"), "ours")

    merged = kept.ours.merge(kept.theirs.name)

    assert kept.ours.read("stays") == "2"
    assert kept.ours.read("goes") is None
    assert kept.ours.read("mine") == "1"
    assert merged.kept_local == []


def test_of_a_file_both_sides_changed_ours_is_kept_and_named(kept: Kept):
    kept.ours.write("svc/a", "spec.md", "ours")
    kept.theirs.write("svc/a", "spec.md", "theirs")

    [clash] = kept.ours.clashes(kept.theirs.name)
    merged = kept.ours.merge(kept.theirs.name)

    assert (clash.path, clash.ours, clash.theirs) == ("svc/a/spec.md", "ours", "theirs")
    assert merged.kept_local == ["svc/a/spec.md"]
    assert kept.ours.read("svc/a/spec.md") == "ours"


def test_the_other_sides_text_is_taken_where_that_is_asked(kept: Kept):
    kept.ours.write("svc/a", "spec.md", "ours")
    kept.theirs.write("svc/a", "spec.md", "theirs")

    merged = kept.ours.merge(kept.theirs.name, take_theirs=["svc/a/spec.md"])

    assert merged.kept_local == []
    assert kept.ours.read("svc/a/spec.md") == "theirs"


def test_reports_both_sides_added_are_all_kept_in_order(kept: Kept):
    early = a_report("ours", "2026-11-01T00:00:00+00:00")
    late = a_report("theirs", "2026-11-02T00:00:00+00:00")
    kept.ours.replace_reports("svc/a", "ic", [], [early])
    kept.theirs.replace_reports("svc/a", "ic", [], [late])

    assert kept.ours.clashes(kept.theirs.name) == []
    merged = kept.ours.merge(kept.theirs.name)

    assert merged.kept_local == []
    assert [report.text for report in kept.ours.reports("svc/a")] == ["ours", "theirs"]


def test_stores_that_have_each_other_have_nothing_to_settle(kept: Kept):
    kept.ours.commit(writing("a", "1"), "ours")
    kept.theirs.merge(kept.ours.name)
    kept.theirs.commit(writing("a", "2"), "theirs went on")

    assert kept.ours.clashes(kept.theirs.name) == []
    assert kept.ours.merge(kept.theirs.name).head == kept.theirs.head()
