"""GitStore against real temporary git repositories."""

from collections.abc import Mapping
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
    Landing,
    StoreChange,
    StoreScopeError,
)
from openhands.agent_server.director.models import (
    FormerTeam,
    Report,
    RoutedRequest,
    TeamEvent,
)
from openhands.sdk.errors import NotSupported
from tests.agent_server.director.conftest import run_git, take_in


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
    assert board.reports("api", "ic") == []


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


def test_reports_are_replaced_only_as_they_were_read(board: GitStore):
    take_in(board, reports={"api": [a_report("Long.")]})
    read = board.reports("api", "ic")
    shorter = [a_report("Short.").model_copy(update={"compacted": True})]
    take_in(board, reports={"api": [a_report("Added meanwhile.")]})

    assert not board.replace_reports("api", "ic", read, shorter)
    assert board.replace_reports("api", "ic", board.reports("api", "ic"), shorter)
    assert board.reports("api", "ic") == shorter


def a_landing(
    files: Mapping[str, tuple[str | None, str | None]] | None = None,
    seen: Mapping[str, str | None] | None = None,
    reports: dict[str, list[Report]] | None = None,
    former: list[FormerTeam] | None = None,
    forget: list[str] | None = None,
    events: dict[str, list[TeamEvent]] | None = None,
    routed: list[RoutedRequest] | None = None,
) -> Landing:
    return Landing(
        dict(files or {}),
        dict(seen or {}),
        reports or {},
        former or [],
        forget or [],
        events or {},
        routed or [],
    )


def test_a_file_is_taken_in_only_over_the_text_it_was_written_over(board: GitStore):
    board.write("api", "spec.md", "v1")
    board.write("web", "contracts.md", "c1")
    board.write("web", "contracts.md", "c2")

    left = board.take_in(
        a_landing(
            {
                "api/spec.md": ("mine", "v1"),
                "web/contracts.md": ("mine", "c1"),
                "db/spec.md": ("new", None),
                "db/contracts.md": (None, None),
            }
        ),
        "accepted",
    )

    # What someone else changed meanwhile is left and told, not overwritten.
    assert left == {"web/contracts.md": "c2"}
    assert board.read("api/spec.md") == "mine"
    assert board.read("web/contracts.md") == "c2"
    assert board.read("db/spec.md") == "new"


def test_a_file_is_put_in_place_of_the_text_its_reviewer_was_shown(board: GitStore):
    board.write("api", "spec.md", "v2")
    files = {"api/spec.md": ("mine", "v1")}

    assert board.take_in(a_landing(files, seen={"api/spec.md": "v1.5"}), "a") == {
        "api/spec.md": "v2"
    }
    assert board.take_in(a_landing(files, seen={"api/spec.md": "v2"}), "a") == {}
    assert board.read("api/spec.md") == "mine"
    # Taking in what is there already writes nothing.
    head = board.head()
    assert board.take_in(a_landing(files), "a") == {}
    assert board.head() == head


def test_a_file_a_session_removed_is_removed_where_it_still_stands(board: GitStore):
    board.write("api", "contracts.md", "c1")

    assert board.take_in(a_landing({"api/contracts.md": (None, "c1")}), "a") == {}
    assert board.read("api/contracts.md") is None


def test_taking_in_the_same_reports_twice_adds_them_once(board: GitStore):
    take_in(board, reports={"api": [a_report("Before.")]})
    mine = [
        a_report("Chose REST.").model_copy(update={"id": "r1"}),
        Report(at="2026-11-02T00:00:00+00:00", role="manager", text="Agreed.", id="r2"),
    ]

    board.take_in(a_landing(reports={"api": mine}), "accepted")
    board.take_in(a_landing(reports={"api": mine}), "accepted again")

    assert [r.text for r in board.reports("api", "ic")] == ["Before.", "Chose REST."]
    assert [r.text for r in board.reports("api", "manager")] == ["Agreed."]


def test_events_and_routed_requests_are_taken_in_once_and_in_order(board: GitStore):
    def event(name: str, kind: str = "accepted") -> TeamEvent:
        return TeamEvent.model_validate(
            {
                "at": "2026-11-01T00:00:00+00:00",
                "kind": kind,
                "team_id": "api",
                "id": name,
            }
        )

    asked = RoutedRequest(
        at="2026-11-01T00:00:00+00:00",
        kind="asked",
        team_id="api@s1",
        team="API",
        text="Add an index.",
        id="q1",
    )
    take_in(board, events={"root": [event("e1", "set_up")]})
    landing = a_landing(
        events={"root": [event("e1", "set_up"), event("e2")]}, routed=[asked]
    )

    board.take_in(landing, "accepted")
    board.take_in(landing, "accepted again")

    assert [e.id for e in board.history("root")] == ["e1", "e2"]
    assert board.history("api") == []
    assert board.routed() == [asked]


def test_a_team_that_is_forgotten_leaves_its_record_and_none_of_its_reports(
    board: GitStore,
):
    take_in(board, reports={"api": [a_report("The old team's.")]})
    record = FormerTeam(
        id="api",
        title="API",
        path="services/api",
        purpose="Serve orders",
        ended="replaced",
        at="20261116T100000Z",
    )
    mine = [a_report("The new team's.").model_copy(update={"id": "r1"})]

    board.take_in(
        a_landing(reports={"api": mine}, former=[record], forget=["api"]), "accepted"
    )
    board.take_in(a_landing(former=[record.model_copy(update={"title": "x"})]), "again")

    assert [r.text for r in board.reports("api", "ic")] == ["The new team's."]
    assert board.former("api") == [record]


def test_racing_writers_lose_no_update(board: GitStore):
    reports = [a_report(f"report {i}") for i in range(6)]

    with ThreadPoolExecutor(max_workers=len(reports)) as pool:
        list(pool.map(lambda r: take_in(board, reports={"api": [r]}), reports))

    assert sorted(report.text for report in board.reports("api", "ic")) == [
        report.text for report in reports
    ]


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
        take_in(board, reports={"api": [shared, ours]})
        take_in(theirs, reports={"api": [shared, their]})

        merged = board.merge(theirs.ref)

        assert merged.kept_local == []
        assert board.reports("api", "ic") == [shared, their, ours]

    def test_events_both_sides_added_are_all_kept_in_order(
        self, board: GitStore, theirs: GitStore, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")

        def event(name: str, at: str) -> TeamEvent:
            return TeamEvent.model_validate(
                {"at": at, "kind": "accepted", "team_id": "api", "id": name}
            )

        shared = event("shared", "2026-11-01T00:00:00+00:00")
        ours = event("ours", "2026-11-03T00:00:00+00:00")
        their = event("theirs", "2026-11-02T00:00:00+00:00")
        take_in(board, events={"root": [shared, ours]})
        take_in(theirs, events={"root": [shared, their]})

        merged = board.merge(theirs.ref)

        assert merged.kept_local == []
        assert board.history("root") == [shared, their, ours]

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
