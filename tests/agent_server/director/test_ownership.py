"""Who owns which file below a node, and where checkouts collide."""

from pathlib import Path

import pytest

from openhands.agent_server.director import ownership
from openhands.agent_server.director.models import (
    AgentRef,
    Boundary,
    OrgNode,
    StatusCounts,
)
from openhands.agent_server.director.ownership import node_ownership
from tests.agent_server.director.conftest import run_git


def team(node_id: str, path: str, checkout: Path, parent: str | None = "root"):
    return OrgNode(
        id=node_id,
        parent_id=parent,
        child_ids=[],
        title=node_id.upper(),
        boundary=Boundary(path=path),
        isolation="shared",
        worktree_path=str(checkout),
        branch="main",
        agents=[AgentRef(role="ic", status="idle")],
        status="idle",
        descendant_counts=StatusCounts(),
        escalation_count=0,
    )


def commit(checkout: Path, path: str, content: str) -> None:
    (checkout / path).parent.mkdir(parents=True, exist_ok=True)
    (checkout / path).write_text(content)
    run_git(checkout, "add", path)
    run_git(checkout, "commit", "--quiet", "-m", path)


@pytest.fixture
def org(repo: Path) -> Path:
    commit(repo, "svc/api/app.py", "api\n")
    commit(repo, "svc/api/db/schema.sql", "schema\n")
    commit(repo, "svc/web/index.html", "web\n")
    commit(repo, "shared.txt", "one\ntwo\nthree\n")
    return repo


def worktree(repo: Path, tmp_path: Path, name: str) -> Path:
    checkout = tmp_path / name
    run_git(repo, "worktree", "add", "--quiet", "-b", name, str(checkout))
    return checkout


def test_each_file_belongs_to_the_report_whose_directory_it_is_in(org: Path):
    (org / "svc" / "web" / "draft.html").write_text("untracked\n")
    (org / ".gitignore").write_text("*.log\n")
    (org / "svc" / "web" / "build.log").write_text("ignored\n")
    root = team("root", "", org, parent=None)
    reports = [team("api", "svc/api", org), team("web", "svc/web", org)]

    found = node_ownership(root, reports)

    assert {file.path: file.owner for file in found.files} == {
        ".gitignore": None,
        "README.md": None,
        "shared.txt": None,
        "svc/api/app.py": "api",
        "svc/api/db/schema.sql": "api",
        "svc/web/draft.html": "web",
        "svc/web/index.html": "web",
    }
    assert [(owner.node_id, owner.title, owner.path) for owner in found.owners] == [
        ("api", "API", "svc/api"),
        ("web", "WEB", "svc/web"),
    ]
    assert (found.truncated, found.collisions, found.collisions_checked) == (
        False,
        [],
        True,
    )


def test_only_the_files_of_the_nodes_own_boundary_are_listed(org: Path):
    api = team("api", "svc/api", org)

    found = node_ownership(api, [team("db", "svc/api/db", org, parent="api")])

    assert [(file.path, file.owner) for file in found.files] == [
        ("svc/api/app.py", None),
        ("svc/api/db/schema.sql", "db"),
    ]


def test_a_long_listing_is_cut_and_says_so(org: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ownership, "MAX_FILES", 2)

    found = node_ownership(team("root", "", org, parent=None), [])

    assert [file.path for file in found.files] == ["README.md", "shared.txt"]
    assert found.truncated


def test_work_committed_in_two_checkouts_that_does_not_merge_is_a_collision(
    org: Path, tmp_path: Path
):
    api, web = worktree(org, tmp_path, "api"), worktree(org, tmp_path, "web")
    commit(org, "shared.txt", "ONE\ntwo\nthree\n")
    commit(api, "shared.txt", "uno\ntwo\nthree\n")
    commit(api, "svc/api/app.py", "api v2\n")
    commit(web, "shared.txt", "eins\ntwo\nthree\n")
    commit(web, "svc/web/index.html", "web v2\n")
    reports = [team("api", "svc/api", api), team("web", "svc/web", web)]

    found = node_ownership(team("root", "", org, parent=None), reports)

    assert [(clash.path, clash.node_ids) for clash in found.collisions] == [
        ("shared.txt", ["root", "api", "web"])
    ]


def test_changes_that_merge_and_uncommitted_ones_are_not_collisions(
    org: Path, tmp_path: Path
):
    api = worktree(org, tmp_path, "api")
    commit(org, "shared.txt", "ONE\ntwo\nthree\n")
    commit(api, "shared.txt", "one\ntwo\nTHREE\n")
    (api / "README.md").write_text("uncommitted in one\n")
    (org / "README.md").write_text("uncommitted in the other\n")

    found = node_ownership(
        team("root", "", org, parent=None), [team("api", "svc/api", api)]
    )

    assert found.collisions == [] and found.collisions_checked


def test_an_old_git_lists_the_files_and_says_it_did_not_look_for_collisions(
    org: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ownership, "git_version", lambda _: (2, 30))

    found = node_ownership(team("root", "", org, parent=None), [])

    assert found.files and not found.collisions_checked


def test_a_checkout_that_is_gone_gives_an_empty_map(tmp_path: Path):
    gone = team("root", "", tmp_path / "nowhere", parent=None)

    found = node_ownership(gone, [team("api", "svc/api", tmp_path / "nowhere")])

    assert (found.files, found.collisions, found.collisions_checked) == ([], [], False)
    assert [owner.node_id for owner in found.owners] == ["api"]
