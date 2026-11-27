"""The git environment's checkouts, by themselves on real repositories."""

from pathlib import Path

import pytest

from openhands.agent_server.director.environment import EnvironmentFailed
from openhands.agent_server.director.environment.git import GitCheckouts
from openhands.agent_server.director.models import Boundary, Review, SessionTeam
from openhands.agent_server.director.org_store import utc_now
from tests.agent_server.director.conftest import (
    commit,
    make_place,
    make_team,
    run_git,
)


def team_at(checkout: Path, path: str, *, branch: str = "") -> SessionTeam:
    """What a session has at the team that owns ``path``, in ``checkout``,
    started at the commit that is checked out there."""
    place = make_place(
        make_team(path or "root", None, path=path), "s", None, checkout=str(checkout)
    )
    return SessionTeam(
        **place.model_dump(exclude={"boundary", "base_commit", "branch", "isolation"}),
        boundary=Boundary(path=path, writable=[f"{path}/**" if path else "**"]),
        base_commit=run_git(checkout, "rev-parse", "HEAD").strip(),
        branch=branch,
        isolation="worktree" if branch else "shared",
    )


def own_checkout(repo: Path, path: str, branch: str) -> SessionTeam:
    """A team with a worktree of its own on ``branch``."""
    worktree = repo.parent / branch.replace("/", "-")
    run_git(repo, "worktree", "add", "--quiet", "-b", branch, str(worktree))
    return team_at(worktree, path, branch=branch)


@pytest.fixture
def checkouts() -> GitCheckouts:
    return GitCheckouts()


def test_a_checkout_names_its_work_and_the_branch_it_is_on(
    repo: Path, checkouts: GitCheckouts, tmp_path: Path
):
    head = run_git(repo, "rev-parse", "HEAD").strip()

    assert checkouts.is_repository(repo)
    assert not checkouts.is_repository(tmp_path)
    assert checkouts.version(repo) == head
    assert checkouts.branch(repo) == "main"
    assert checkouts.branch(tmp_path / "nowhere") == ""
    (repo / "docs").mkdir()
    assert checkouts.root_of(repo / "docs") == repo.resolve()
    assert checkouts.root_of(tmp_path / "nowhere") == tmp_path / "nowhere"
    assert "README.md" in checkouts.files(repo)


def test_a_team_with_a_worktree_of_its_own_is_apart_from_the_team_above(
    repo: Path, checkouts: GitCheckouts
):
    root = team_at(repo, "")
    shared = team_at(repo, "web")
    api = own_checkout(repo, "api", "team/api")

    assert checkouts.is_apart(api, root)
    assert not checkouts.is_apart(shared, root)


def test_what_a_team_changed_itself_leaves_out_its_sub_teams_folders(
    repo: Path, checkouts: GitCheckouts
):
    root, api = team_at(repo, ""), team_at(repo, "api")
    commit(repo, "api/app.py", "x = 1\n")
    commit(repo, "notes.md", "committed\n")
    (repo / "todo.md").write_text("not committed\n")

    assert checkouts.own_changes(root, [api]) == ["notes.md", "todo.md"]
    assert checkouts.uncommitted(repo, "") == ["todo.md"]


def test_work_meets_what_came_into_the_checkout_above_in_the_same_file(
    repo: Path, checkouts: GitCheckouts
):
    commit(repo, "api/app.py", "one\ntwo\nthree\n")
    root = team_at(repo, "")
    api = own_checkout(repo, "api", "team/api")
    commit(Path(api.worktree_path), "api/app.py", "one\ntwo\nthree\nfour\n")
    commit(repo, "api/app.py", "zero\none\ntwo\nthree\n")

    found = checkouts.overlap(api, root)

    assert found is not None
    assert found.files == ["api/app.py"]
    assert found.tip == checkouts.version(api.worktree_path)
    assert "api/app.py" in found.seen
    assert checkouts.overlap(team_at(repo, "web"), root) is None


def test_the_work_of_a_team_is_merged_into_the_checkout_above(
    repo: Path, checkouts: GitCheckouts
):
    root = team_at(repo, "")
    api = own_checkout(repo, "api", "team/api")
    commit(Path(api.worktree_path), "api/app.py", "x = 1\n")

    merged = checkouts.merge(api, root, "API")

    assert merged.outcome == "merged"
    assert (repo / "api" / "app.py").read_text() == "x = 1\n"
    assert checkouts.merge(api, root, "API").outcome == "nothing"


def test_work_outside_the_teams_folder_is_not_merged(
    repo: Path, checkouts: GitCheckouts
):
    root = team_at(repo, "")
    api = own_checkout(repo, "api", "team/api")
    commit(Path(api.worktree_path), "web/page.tsx", "not the team's\n")

    merged = checkouts.merge(api, root, "API")

    assert merged.outcome == "outside"
    assert merged.files == ("web/page.tsx",)
    assert not (repo / "web").exists()


def test_a_merge_is_undone_to_where_the_checkout_was(
    repo: Path, checkouts: GitCheckouts
):
    root = team_at(repo, "")
    api = own_checkout(repo, "api", "team/api")
    commit(Path(api.worktree_path), "api/app.py", "x = 1\n")
    before = checkouts.version(repo)
    assert before is not None
    checkouts.merge(api, root, "API")

    checkouts.undo_merge(repo, before)

    assert checkouts.version(repo) == before
    assert not (repo / "api").exists()


def test_a_checkout_off_its_branch_holds_work_that_is_astray(
    repo: Path, checkouts: GitCheckouts
):
    api = own_checkout(repo, "api", "team/api")
    assert checkouts.astray(api, [], repo) is None
    run_git(Path(api.worktree_path), "checkout", "--quiet", "-b", "elsewhere")

    told = checkouts.astray(api, [], repo)

    assert told is not None
    assert "'elsewhere'" in told and "'team/api'" in told


def test_accepted_work_that_is_not_in_the_checkout_is_astray(
    repo: Path, checkouts: GitCheckouts
):
    api = own_checkout(repo, "api", "team/api")
    checkout = Path(api.worktree_path)
    commit(checkout, "api/db/schema.sql", "create table t ();\n")
    accepted = run_git(checkout, "rev-parse", "HEAD").strip()
    run_git(checkout, "reset", "--quiet", "--hard", "HEAD~1")
    db = team_at(checkout, "api/db").model_copy(
        update={
            "review": Review(
                state="accepted", summary="done", commit=accepted, at=utc_now()
            )
        }
    )

    told = checkouts.astray(api, [db], repo)

    assert told is not None and accepted[:10] in told


def test_uncommitted_work_is_set_aside_and_kept(repo: Path, checkouts: GitCheckouts):
    (repo / "api").mkdir()
    (repo / "api" / "draft.py").write_text("x = 1\n")

    kept = checkouts.set_aside(repo, "api", "refs/agents/aside/api")

    assert kept is not None
    assert not (repo / "api" / "draft.py").exists()
    assert "draft.py" in run_git(repo, "show", "--stat", "refs/agents/aside/api")
    assert checkouts.set_aside(repo, "api", "refs/agents/aside/again") is None


def test_work_that_is_not_in_the_checkout_above_is_kept_when_it_is_left(
    repo: Path, checkouts: GitCheckouts
):
    root = team_at(repo, "")
    api = own_checkout(repo, "api", "team/api")
    ref = "refs/agents/sessions/s/kept"
    assert not checkouts.keep_unmerged(api.worktree_path, repo, ref)
    commit(Path(api.worktree_path), "api/app.py", "x = 1\n")

    assert checkouts.keep_unmerged(api.worktree_path, repo, ref)
    assert run_git(repo, "rev-parse", ref).strip() == checkouts.version(
        api.worktree_path
    )
    checkouts.merge(api, root, "API")
    assert not checkouts.keep_unmerged(api.worktree_path, repo, "refs/agents/other")


def test_a_checkout_that_is_gone_is_made_again_from_its_teams_branch(
    repo: Path, checkouts: GitCheckouts
):
    api = own_checkout(repo, "api", "team/api")
    gone = api.model_copy(update={"branch": "team/gone", "id": "gone@s"})
    gone = gone.model_copy(update={"worktree_path": f"{api.worktree_path}-gone"})
    run_git(repo, "worktree", "remove", "--force", api.worktree_path)

    assert checkouts.missing(repo, [api, gone]) == [(api, True), (gone, False)]
    checkouts.restore(repo, [api, gone], "")

    assert Path(api.worktree_path).is_dir()
    assert checkouts.branch(api.worktree_path) == "team/api"
    assert checkouts.missing(repo, [api]) == []


def test_what_git_refuses_is_raised_as_the_environments_failure(
    repo: Path, checkouts: GitCheckouts
):
    with pytest.raises(EnvironmentFailed):
        checkouts.undo_merge(repo, "0" * 40)
