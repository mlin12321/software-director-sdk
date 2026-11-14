"""GitStore against real temporary git repositories."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openhands.agent_server.director.git import (
    GitError,
    git,
    git_version,
    is_git_repository,
)
from openhands.agent_server.director.git_store import (
    MEMORY_REF,
    GitStore,
    StoreChange,
    StoreScopeError,
    adopt_former_memory,
)
from openhands.agent_server.director.models import Report
from openhands.sdk.errors import NotSupported
from tests.agent_server.director.conftest import run_git


@pytest.fixture
def board(repo: Path) -> GitStore:
    return GitStore(repo)


def a_report(text: str, at: str = "2026-11-01T00:00:00+00:00") -> Report:
    return Report(at=at, role="ic", text=text)


def test_git_reports_the_exit_code_and_stderr_of_a_failed_command(repo: Path):
    with pytest.raises(GitError) as raised:
        git(repo, ["hash-object", "--no-such-flag", "--stdin"], stdin="x" * 200_000)

    assert raised.value.code != 0
    assert "no-such-flag" in raised.value.stderr


def test_a_plain_or_missing_directory_is_not_a_repository(tmp_path: Path, repo: Path):
    assert is_git_repository(repo)
    assert not is_git_repository(tmp_path)
    assert not is_git_repository(tmp_path / "missing")


def test_reads_nothing_before_the_first_write(board: GitStore):
    assert board.head() is None
    assert board.read("spec.md") is None
    assert board.applies_to("svc/a", "svc") == ([], [])
    assert board.reports("agent-1") == []


def test_writes_only_under_the_hidden_ref(repo: Path, board: GitStore):
    branch_before = run_git(repo, "rev-parse", "main")

    commit = board.write("svc/a", "spec.md", "# A\n")

    assert run_git(repo, "rev-parse", MEMORY_REF).strip() == commit
    assert run_git(repo, "ls-tree", "-r", "--name-only", MEMORY_REF) == (
        "svc/a/spec.md\n"
    )
    assert run_git(repo, "status", "--porcelain") == ""
    assert run_git(repo, "ls-files") == "README.md\n"
    assert run_git(repo, "rev-parse", "main") == branch_before


def test_writing_again_keeps_earlier_files_and_history(board: GitStore):
    first = board.write("svc/a", "spec.md", "# A\n")
    board.write("svc/b", "spec.md", "# B\n")

    assert board.read("svc/a/spec.md") == "# A\n"
    assert board.read("svc/b/spec.md") == "# B\n"
    assert run_git(board.repo_dir, "rev-parse", f"{MEMORY_REF}~1").strip() == first


def test_content_round_trips_byte_for_byte(board: GitStore):
    content = "naïve ✓ 日本語\r\nno trailing newline"

    board.write("", "notes.txt", content)

    assert board.read("notes.txt") == content


def test_a_team_works_to_its_own_spec_and_the_contracts_around_it(
    board: GitStore,
):
    board.write("", "spec.md", "root spec")
    board.write("", "contracts.md", "set at the root")
    board.write("svc", "spec.md", "spec of the team above")
    board.write("svc", "contracts.md", "between a and b")
    board.write("svc/a", "spec.md", "own spec")
    board.write("svc/a", "contracts.md", "between the teams below a")
    board.write("svc/b", "spec.md", "spec of the team beside")
    board.write("svc/a/one", "spec.md", "spec of a team below")

    specs, contracts = board.applies_to("svc/a", "svc")

    assert [(spec.path, spec.spec, spec.readonly) for spec in specs] == [
        ("svc/a", "own spec", False)
    ]
    assert [(agreed.path, agreed.contracts, agreed.own) for agreed in contracts] == [
        ("svc", "between a and b", False),
        ("svc/a", "between the teams below a", True),
    ]


def test_the_root_team_has_nothing_from_above(board: GitStore):
    board.write("", "contracts.md", "between the first teams")

    specs, contracts = board.applies_to("", None)

    assert specs == []
    assert [(agreed.path, agreed.own) for agreed in contracts] == [("", True)]


@pytest.mark.parametrize(
    ("scope", "file"),
    [
        ("svc/a", "../b/spec.md"),
        ("svc/a", ".git/config"),
        ("../outside", "spec.md"),
        ("svc/a", ""),
        ("svc/a", "."),
    ],
)
def test_rejects_writes_that_leave_the_scope(board: GitStore, scope: str, file: str):
    with pytest.raises(StoreScopeError):
        board.write(scope, file, "x")

    assert board.head() is None


def test_an_absolute_looking_file_stays_inside_the_scope(board: GitStore):
    board.write("svc/a", "/etc/passwd", "x")

    assert board.read("svc/a/etc/passwd") == "x"


def test_reports_are_added_in_order_and_kept_apart_by_agent(board: GitStore):
    first, second = a_report("Chose SQLite."), a_report("The ORM was too slow.")

    board.append_report("agent-1", first)
    kept = board.append_report("agent-1", second)

    assert kept == board.reports("agent-1") == [first, second]
    assert board.reports("agent-2") == []


def test_reports_are_replaced_only_as_they_were_read(board: GitStore):
    read = board.append_report("agent-1", a_report("Long."))
    shorter = [a_report("Short.").model_copy(update={"compacted": True})]
    board.append_report("agent-1", a_report("Added meanwhile."))

    assert not board.replace_reports("agent-1", read, shorter)
    assert board.replace_reports("agent-1", board.reports("agent-1"), shorter)
    assert board.reports("agent-1") == shorter


def test_racing_writers_lose_no_update(board: GitStore):
    reports = [a_report(f"report {i}") for i in range(6)]

    with ThreadPoolExecutor(max_workers=len(reports)) as pool:
        list(pool.map(lambda report: board.append_report("agent-1", report), reports))

    assert sorted(report.text for report in board.reports("agent-1")) == [
        report.text for report in reports
    ]


def test_memory_under_its_former_name_is_moved(repo: Path):
    former = GitStore(repo, "refs/agents/blackboard")
    head = former.write("svc/a", "spec.md", "x")

    adopt_former_memory(repo)
    adopt_former_memory(repo)

    assert GitStore(repo).head() == head
    assert former.head() is None


def test_a_change_with_no_content_deletes_the_file(board: GitStore):
    board.write("svc/a", "spec.md", "x")
    board.write("svc/a", "keep.md", "y")

    board.commit(lambda _read: [StoreChange("svc/a/spec.md", None)], "delete")

    assert board.read("svc/a/spec.md") is None
    assert board.read("svc/a/keep.md") == "y"


def test_a_mutation_without_changes_makes_no_commit(board: GitStore):
    head = board.write("svc/a", "spec.md", "x")

    assert board.commit(lambda _read: [], "nothing") == head
    assert board.head() == head


def test_a_linked_worktree_shares_the_team_memory(
    repo: Path, tmp_path: Path, board: GitStore
):
    linked = tmp_path / "linked"
    run_git(repo, "worktree", "add", "--quiet", "-b", "feature", str(linked))

    GitStore(linked).write("svc/a", "spec.md", "From the worktree.")

    assert board.read("svc/a/spec.md") == "From the worktree."
    assert run_git(linked, "status", "--porcelain") == ""


class TestMerge:
    @pytest.fixture
    def merge_supported(self, repo: Path) -> bool:
        return git_version(repo) >= (2, 38)

    @pytest.fixture
    def theirs(self, repo: Path) -> GitStore:
        return GitStore(repo, ref="refs/agents/theirs")

    def test_old_git_is_reported_as_unsupported(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if merge_supported:
            pytest.skip("the installed git can merge")
        theirs.write("svc/b", "spec.md", "B")

        with pytest.raises(NotSupported, match="2.38"):
            board.merge(theirs.ref)

    def test_combines_changes_to_different_files(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        board.write("svc/a", "spec.md", "A")
        theirs.write("svc/b", "spec.md", "B")

        board.merge(theirs.ref)

        assert board.read("svc/a/spec.md") == "A"
        assert board.read("svc/b/spec.md") == "B"

    def test_reports_both_sides_added_are_all_kept_in_order(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        shared = a_report("Shared.", "2026-11-01T00:00:00+00:00")
        ours = a_report("Ours.", "2026-11-03T00:00:00+00:00")
        their = a_report("Theirs.", "2026-11-02T00:00:00+00:00")
        board.append_report("agent-1", shared)
        board.append_report("agent-1", ours)
        theirs.append_report("agent-1", shared)
        theirs.append_report("agent-1", their)

        merged = board.merge(theirs.ref)

        assert merged.kept_local == []
        assert board.reports("agent-1") == [shared, their, ours]

    def test_of_a_spec_both_sides_changed_ours_is_kept_and_named(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        board.write("svc/a", "spec.md", "ours")
        theirs.write("svc/a", "spec.md", "theirs")
        theirs.write("svc/b", "spec.md", "only theirs")

        merged = board.merge(theirs.ref)

        assert merged.kept_local == ["svc/a/spec.md"]
        assert board.read("svc/a/spec.md") == "ours"
        assert board.read("svc/b/spec.md") == "only theirs"
        history = git(board.repo_dir, ["log", "--format=%H", board.ref]).split()
        assert theirs.head() in history

    def test_merging_what_is_already_merged_changes_nothing(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        board.write("svc/a", "spec.md", "A")
        theirs.write("svc/b", "spec.md", "B")
        head = board.merge(theirs.ref).head

        assert board.merge(theirs.ref).head == head

    def test_an_empty_team_memory_adopts_the_other_ref(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        theirs.write("svc/b", "spec.md", "B")

        assert board.merge(theirs.ref).head == theirs.head()
        assert board.read("svc/b/spec.md") == "B"
