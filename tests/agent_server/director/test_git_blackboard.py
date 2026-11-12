"""GitBlackboard against real temporary git repositories."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openhands.agent_server.director.git import (
    GitError,
    git,
    git_version,
    is_git_repository,
)
from openhands.agent_server.director.git_blackboard import (
    BLACKBOARD_REF,
    BlackboardMergeConflictError,
    BlackboardScopeError,
    BoardChange,
    GitBlackboard,
)
from openhands.agent_server.director.models import Autopsy
from openhands.sdk.errors import Invalid, NotSupported
from tests.agent_server.director.conftest import run_git


@pytest.fixture
def board(repo: Path) -> GitBlackboard:
    return GitBlackboard(repo)


def test_git_reports_the_exit_code_and_stderr_of_a_failed_command(repo: Path):
    with pytest.raises(GitError) as raised:
        git(repo, ["hash-object", "--no-such-flag", "--stdin"], stdin="x" * 200_000)

    assert raised.value.code != 0
    assert "no-such-flag" in raised.value.stderr


def test_a_plain_or_missing_directory_is_not_a_repository(tmp_path: Path, repo: Path):
    assert is_git_repository(repo)
    assert not is_git_repository(tmp_path)
    assert not is_git_repository(tmp_path / "missing")


def test_reads_nothing_before_the_first_write(board: GitBlackboard):
    assert board.head() is None
    assert board.read("spec.md") is None
    assert board.inherited("svc/a") == []
    assert board.memory("svc/a").rules == []
    assert board.autopsies("svc/a") == []


def test_writes_only_under_the_hidden_ref(repo: Path, board: GitBlackboard):
    branch_before = run_git(repo, "rev-parse", "main")

    commit = board.write("svc/a", "spec.md", "# A\n")

    assert run_git(repo, "rev-parse", BLACKBOARD_REF).strip() == commit
    assert run_git(repo, "ls-tree", "-r", "--name-only", BLACKBOARD_REF) == (
        "svc/a/spec.md\n"
    )
    assert run_git(repo, "status", "--porcelain") == ""
    assert run_git(repo, "ls-files") == "README.md\n"
    assert run_git(repo, "rev-parse", "main") == branch_before


def test_writing_again_keeps_earlier_files_and_history(board: GitBlackboard):
    first = board.write("svc/a", "spec.md", "# A\n")
    board.write("svc/b", "spec.md", "# B\n")

    assert board.read("svc/a/spec.md") == "# A\n"
    assert board.read("svc/b/spec.md") == "# B\n"
    assert run_git(board.repo_dir, "rev-parse", f"{BLACKBOARD_REF}~1").strip() == first


def test_content_round_trips_byte_for_byte(board: GitBlackboard):
    content = "naïve ✓ 日本語\r\nno trailing newline"

    board.write("", "notes.txt", content)

    assert board.read("notes.txt") == content


def test_inherited_specs_run_root_to_boundary_and_skip_gaps(board: GitBlackboard):
    board.write("", "spec.md", "root")
    board.write("svc/a", "spec.md", "leaf")
    board.write("svc/b", "spec.md", "sibling")

    specs = board.inherited("svc/a")

    # ``svc`` has no spec, and the sibling's spec does not apply.
    assert [(s.path, s.spec, s.readonly) for s in specs] == [
        ("", "root", True),
        ("svc/a", "leaf", False),
    ]


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
def test_rejects_writes_that_leave_the_scope(
    board: GitBlackboard, scope: str, file: str
):
    with pytest.raises(BlackboardScopeError):
        board.write(scope, file, "x")

    assert board.head() is None


def test_an_absolute_looking_file_stays_inside_the_scope(board: GitBlackboard):
    board.write("svc/a", "/etc/passwd", "x")

    assert board.read("svc/a/etc/passwd") == "x"


def test_teach_accumulates_rules_once_and_rejects_blanks(board: GitBlackboard):
    board.teach("svc/a", "  Use idempotency keys.  ")
    board.teach("svc/a", "Never log card numbers.")
    head = board.head()
    board.teach("svc/a", "Use idempotency keys.")

    assert board.memory("svc/a").rules == [
        "Use idempotency keys.",
        "Never log card numbers.",
    ]
    assert board.head() == head
    assert board.memory("svc/b").rules == []
    with pytest.raises(Invalid):
        board.teach("svc/a", "   ")


def test_autopsies_append_in_order(board: GitBlackboard):
    first = Autopsy(
        node_id="a", at="2026-11-01T00:00:00Z", approach="x", why_failed="y"
    )
    second = first.model_copy(update={"approach": "z"})

    board.append_autopsy("svc/a", first)
    board.append_autopsy("svc/a", second)

    assert board.autopsies("svc/a") == [first, second]


def test_racing_writers_lose_no_update(board: GitBlackboard):
    rules = [f"rule {i}" for i in range(6)]

    with ThreadPoolExecutor(max_workers=len(rules)) as pool:
        list(pool.map(lambda rule: board.teach("svc/a", rule), rules))

    assert sorted(board.memory("svc/a").rules) == rules


def test_a_change_with_no_content_deletes_the_file(board: GitBlackboard):
    board.write("svc/a", "spec.md", "x")
    board.write("svc/a", "keep.md", "y")

    board.commit(lambda _read: [BoardChange("svc/a/spec.md", None)], "delete")

    assert board.read("svc/a/spec.md") is None
    assert board.read("svc/a/keep.md") == "y"


def test_a_mutation_without_changes_makes_no_commit(board: GitBlackboard):
    head = board.write("svc/a", "spec.md", "x")

    assert board.commit(lambda _read: [], "nothing") == head
    assert board.head() == head


def test_a_linked_worktree_shares_the_blackboard(
    repo: Path, tmp_path: Path, board: GitBlackboard
):
    linked = tmp_path / "linked"
    run_git(repo, "worktree", "add", "--quiet", "-b", "feature", str(linked))

    GitBlackboard(linked).teach("svc/a", "From the worktree.")

    assert board.memory("svc/a").rules == ["From the worktree."]
    assert run_git(linked, "status", "--porcelain") == ""


class TestMerge:
    @pytest.fixture
    def merge_supported(self, repo: Path) -> bool:
        return git_version(repo) >= (2, 38)

    @pytest.fixture
    def theirs(self, repo: Path) -> GitBlackboard:
        return GitBlackboard(repo, ref="refs/agents/theirs")

    def test_old_git_is_reported_as_unsupported(
        self, board: GitBlackboard, theirs: GitBlackboard, merge_supported: bool
    ):
        if merge_supported:
            pytest.skip("the installed git can merge")
        theirs.teach("svc/b", "B")

        with pytest.raises(NotSupported, match="2.38"):
            board.merge(theirs.ref)

    def test_combines_changes_to_different_files(
        self, board: GitBlackboard, theirs: GitBlackboard, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        board.teach("svc/a", "A")
        theirs.teach("svc/b", "B")

        board.merge(theirs.ref)

        assert board.memory("svc/a").rules == ["A"]
        assert board.memory("svc/b").rules == ["B"]

    def test_a_conflict_leaves_the_ref_unchanged(
        self, board: GitBlackboard, theirs: GitBlackboard, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        head = board.write("svc/a", "spec.md", "ours")
        theirs.write("svc/a", "spec.md", "theirs")

        with pytest.raises(BlackboardMergeConflictError):
            board.merge(theirs.ref)

        assert board.head() == head

    def test_an_empty_blackboard_adopts_the_other_ref(
        self, board: GitBlackboard, theirs: GitBlackboard, merge_supported: bool
    ):
        if not merge_supported:
            pytest.skip("needs git 2.38")
        theirs.teach("svc/b", "B")

        assert board.merge(theirs.ref) == theirs.head()
        assert json.loads(board.read("svc/b/memory.json") or "")["rules"] == ["B"]
