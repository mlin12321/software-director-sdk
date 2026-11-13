"""The director's own git operations on a checkout."""

from pathlib import Path

from openhands.agent_server.director.git import save_and_discard
from tests.agent_server.director.conftest import run_git


REF = "refs/agents/pruned/team/1"


def test_uncommitted_work_under_a_path_is_discarded_and_kept_in_a_commit(repo: Path):
    (repo / "svc").mkdir()
    (repo / "svc" / "kept.py").write_text("committed\n")
    (repo / "svc" / "gone.py").write_text("committed\n")
    (repo / ".gitignore").write_text("*.log\n")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "svc")
    (repo / "svc" / "kept.py").write_text("edited\n")
    (repo / "svc" / "gone.py").unlink()
    (repo / "svc" / "staged.py").write_text("staged\n")
    run_git(repo, "add", "svc/staged.py")
    (repo / "svc" / "new" / "deep").mkdir(parents=True)
    (repo / "svc" / "new" / "deep" / "untracked.py").write_text("untracked\n")
    (repo / "svc" / "build.log").write_text("ignored\n")
    (repo / "README.md").write_text("outside the path\n")

    commit = save_and_discard(repo, "svc", REF)

    assert commit == run_git(repo, "rev-parse", REF).strip()
    assert (repo / "svc" / "kept.py").read_text() == "committed\n"
    assert (repo / "svc" / "gone.py").read_text() == "committed\n"
    assert not (repo / "svc" / "staged.py").exists()
    assert not (repo / "svc" / "new").exists()
    assert (repo / "svc" / "build.log").read_text() == "ignored\n"
    assert (repo / "README.md").read_text() == "outside the path\n"
    assert run_git(repo, "status", "--porcelain").split() == ["M", "README.md"]
    kept = run_git(repo, "ls-tree", "-r", "--name-only", REF).split()
    assert kept == [
        ".gitignore",
        "README.md",
        "svc/kept.py",
        "svc/new/deep/untracked.py",
        "svc/staged.py",
    ]
    assert run_git(repo, "show", f"{REF}:svc/kept.py") == "edited\n"
    # What was uncommitted outside the path is not in the copy.
    assert run_git(repo, "show", f"{REF}:README.md") == "hello\n"


def test_a_path_with_only_untracked_files_is_cleared(repo: Path):
    (repo / "svc").mkdir()
    (repo / "svc" / "draft.py").write_text("draft\n")

    assert save_and_discard(repo, "svc", REF) is not None

    assert not (repo / "svc" / "draft.py").exists()


def test_nothing_is_saved_when_everything_is_committed(repo: Path):
    (repo / "README.md").write_text("outside the path\n")

    assert save_and_discard(repo, "svc", REF) is None

    assert run_git(repo, "for-each-ref", "refs/agents/pruned") == ""
    assert (repo / "README.md").read_text() == "outside the path\n"
