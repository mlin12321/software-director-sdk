"""An agent is held to a scope by what knows nothing of teams."""

import subprocess
from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git import GitConfinement
from openhands.agent_server.director.scope import Closed, MergeArea, Scope


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "svc" / "api").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def confinement(repo: Path, secret: Path | None = None) -> GitConfinement:
    return GitConfinement(
        repo_dir=repo,
        checkouts=(repo,),
        bwrap=None,
        secret=secret,
        server_url=None,
        key=lambda: b"key",
    )


def scope(repo: Path, **changes) -> Scope:
    given = {
        "checkout": str(repo),
        "directory": "svc",
        "writable": ("svc/**",),
        **changes,
    }
    return Scope(**given)


def write(path: str) -> dict[str, str]:
    return {"command": "create", "path": path}


def test_a_write_inside_the_scope_is_allowed(repo: Path):
    held = confinement(repo)

    assert held.check(scope(repo), "file_editor", write("svc/a.py")) is None
    assert held.check(scope(repo), "file_editor", {"command": "view", "path": "x"}) is (
        None
    )


@pytest.mark.parametrize(
    ("changes", "path", "kind"),
    [
        ({}, "other/a.py", "outside"),
        ({}, ".git/config", "git"),
        ({"edits": False}, "svc/a.py", "reads_only"),
        ({"readonly": ("svc/gen/**",)}, "svc/gen/a.py", "readonly"),
        ({"writable": ("svc/*.md",)}, "svc/a.py", "not_writable"),
        ({"closed": (Closed("svc/api", "API", "t1"),)}, "svc/api/a.py", "closed"),
        ({"aside": True}, "svc/a.py", "aside"),
    ],
)
def test_a_write_the_scope_does_not_allow_is_denied(
    repo: Path, changes: dict, path: str, kind: str
):
    denied = confinement(repo).check(scope(repo, **changes), "file_editor", write(path))

    assert denied is not None and denied.kind == kind
    assert denied.named == path


def test_only_what_did_not_merge_is_written_in_a_merge(repo: Path, tmp_path: Path):
    where = tmp_path / "merge"
    where.mkdir()
    held = confinement(repo)

    def check(own: bool, name: str):
        merge = MergeArea(where=str(where), files=("svc/a.py",), own=own)
        return held.check(
            scope(repo, merges=(merge,)), "file_editor", write(str(where / name))
        )

    assert check(True, "svc/a.py") is None
    by_itself = check(True, "svc/b.py")
    assert by_itself is not None and by_itself.kind == "merged_by_itself"
    assert by_itself.files == ("svc/a.py",)
    another = check(False, "svc/a.py")
    assert another is not None and another.kind == "merge_of_another"


def test_the_directors_own_file_is_closed_to_every_conversation(
    repo: Path, tmp_path: Path
):
    secret = tmp_path / "kept" / "director.json"
    secret.parent.mkdir()
    secret.write_text("{}")
    held = confinement(repo, secret)

    read = held.check(None, "read_file", {"file_path": str(secret)})
    searched = held.check(None, "grep", {"path": str(secret.parent)})

    assert read is not None and read.kind == "private"
    assert searched is not None and searched.kind == "holds_private"
    assert held.check(None, "grep", {"path": str(repo)}) is None


def test_an_agent_that_writes_gets_a_shell_and_one_that_does_not_gets_none(
    repo: Path,
):
    held = confinement(repo)

    params = held.confine("c1", scope(repo))
    assert held.confine("c2", scope(repo, writes=False)) is None

    assert params is not None
    shell = Path(params["terminal"]["shell_path"])
    assert shell == held.shells / "c1.sh" and shell.exists()
    assert (held.shells / "c1.bin" / "git").exists()

    held.forget(["c1"], repo)
    assert not list(held.shells.glob("c1.*"))


def test_what_holds_an_agent_follows_from_the_server(repo: Path):
    held = confinement(repo)

    assert held.enforcement(own_tools=False) == "checks"
    assert held.enforcement(own_tools=True) == "merge"
