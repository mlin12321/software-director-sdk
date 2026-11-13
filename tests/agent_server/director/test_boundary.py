"""Which team may write where, and which team a request goes to."""

import pytest

from openhands.agent_server.director.boundary import (
    glob_matches,
    nearest_team,
    write_refusal,
    written_paths,
)
from openhands.agent_server.director.models import AgentRef, Boundary, OrgNode


def team(
    node_id: str,
    path: str,
    *,
    writable: list[str] | None = None,
    readonly: list[str] | None = None,
    ic: str | None = "idle",
) -> OrgNode:
    return OrgNode(
        id=node_id,
        parent_id=None if not path else "root",
        title=node_id,
        boundary=Boundary(
            path=path,
            writable=writable
            if writable is not None
            else [f"{path}/**" if path else "**"],
            readonly=readonly or [],
        ),
        isolation="shared",
        worktree_path=".",
        branch="main",
        agents=[
            AgentRef(
                role="ic",
                conversation_id=None if ic is None else f"{node_id}-ic",
                status=ic or "idle",
            )
        ],
        status="idle",
    )


@pytest.mark.parametrize(
    ("glob", "path", "matches"),
    [
        ("**", "a/b.py", True),
        ("svc/api/**", "svc/api/handlers/pay.py", True),
        ("svc/api/**", "svc/api", False),
        ("svc/api/**", "svc/apiary/x.py", False),
        ("svc/*.py", "svc/a.py", True),
        ("svc/*.py", "svc/deep/a.py", False),
        ("svc/**/*.py", "svc/a.py", True),
        ("svc/**/*.py", "svc/deep/er/a.py", True),
        ("svc/?.py", "svc/ab.py", False),
        ("svc/a+b.py", "svc/a+b.py", True),
    ],
)
def test_a_glob_stays_in_a_segment_unless_it_is_a_double_star(glob, path, matches):
    assert glob_matches(glob, path) is matches


@pytest.mark.parametrize(
    ("tool", "action", "paths"),
    [
        ("file_editor", {"command": "create", "path": "/r/a.py"}, ["/r/a.py"]),
        ("file_editor", {"command": "str_replace", "path": "/r/a.py"}, ["/r/a.py"]),
        ("file_editor", {"command": "view", "path": "/r/a.py"}, []),
        ("write_file", {"file_path": "a.py", "content": "x"}, ["a.py"]),
        ("edit", {"file_path": "a.py"}, ["a.py"]),
        ("terminal", {"command": "rm -rf /r"}, []),
        ("read_node_context", {}, []),
    ],
)
def test_only_file_tools_that_write_name_the_files_they_change(tool, action, paths):
    assert written_paths(tool, action) == paths


def test_a_patch_names_every_file_it_adds_updates_deletes_or_moves():
    patch = (
        "*** Begin Patch\n*** Add File: svc/new.py\n+x\n"
        "*** Update File: svc/old.py\n*** Move to: lib/moved.py\n@@\n-a\n+b\n"
        "*** Delete File: svc/gone.py\n*** End Patch\n"
    )

    assert written_paths("apply_patch", {"patch": patch}) == [
        "svc/new.py",
        "svc/old.py",
        "lib/moved.py",
        "svc/gone.py",
    ]


def test_an_ic_may_write_what_its_team_owns():
    api = team("api", "svc/api")

    assert write_refusal(api, "ic", [], "svc/api/handlers/pay.py") is None


@pytest.mark.parametrize("path", ["svc/web/page.tsx", "README.md", "svc/apiary/x.py"])
def test_an_ic_may_not_write_outside_its_boundary(path):
    refusal = write_refusal(team("api", "svc/api"), "ic", [], path)

    assert refusal is not None and "outside your team's boundary" in refusal


def test_an_ic_may_not_write_what_it_handed_to_a_sub_team():
    root, api = team("root", ""), team("api", "svc/api")

    refusal = write_refusal(root, "ic", [api], "svc/api/handlers/pay.py")

    assert refusal is not None and "sub-team 'api'" in refusal
    assert write_refusal(root, "ic", [api], "README.md") is None


def test_read_only_globs_and_narrow_writable_globs_are_kept():
    api = team(
        "api", "svc/api", writable=["svc/api/**/*.py"], readonly=["svc/api/gen/**"]
    )

    assert write_refusal(api, "ic", [], "svc/api/pay.py") is None
    assert "read-only" in (write_refusal(api, "ic", [], "svc/api/gen/x.py") or "")
    assert "may write" in (write_refusal(api, "ic", [], "svc/api/notes.md") or "")


def test_a_manager_may_write_nothing():
    refusal = write_refusal(team("api", "svc/api"), "manager", [], "svc/api/pay.py")

    assert refusal is not None and "Manager" in refusal


def test_a_request_goes_to_the_team_that_owns_the_nearest_directory_above():
    root, svc, api = team("root", ""), team("svc", "svc"), team("api", "svc/api")
    nodes = [root, svc, api]

    def goes_to(path: str) -> str | None:
        found = nearest_team(nodes, path, lost="lost")
        return found.id if found else None

    assert goes_to("svc/api/handlers/deep") == "api"
    assert goes_to("svc/api") == "api"
    assert goes_to("svc/web") == "svc"
    assert goes_to("docs") == "root"
    assert goes_to("") == "root"


def test_a_team_without_a_running_ic_is_passed_over():
    nodes = [
        team("root", ""),
        team("svc", "svc", ic="lost"),
        team("api", "svc/api", ic=None),
    ]

    found = nearest_team(nodes, "svc/api/handlers", lost="lost")

    assert found is not None and found.id == "root"


def test_a_team_that_may_only_read_the_directory_is_passed_over():
    nodes = [
        team("root", ""),
        team("api", "svc/api", readonly=["svc/api/gen/**"]),
    ]

    def goes_to(path: str) -> str | None:
        found = nearest_team(nodes, path, lost="lost")
        return found.id if found else None

    assert goes_to("svc/api/gen") == "root"
    assert goes_to("svc/api/gen/models") == "root"
    assert goes_to("svc/api/handlers") == "api"


def test_no_team_takes_a_request_when_even_the_root_ic_is_gone():
    assert nearest_team([team("root", "", ic="lost")], "svc", lost="lost") is None
