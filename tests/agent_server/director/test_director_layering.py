"""Which parts of the director may know of which."""

import ast
from pathlib import Path

import pytest

import openhands.agent_server.director as director


DIRECTOR = Path(director.__file__).parent


def _imports(module: str) -> set[str]:
    tree = ast.parse((DIRECTOR / module).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize(
    "module", ["tools/spec.py", "tools/catalog.py", "tools/documents.py"]
)
def test_what_a_tool_is_does_not_know_the_agent_sdk(module: str):
    # What the director's tools take and do is said without the agent SDK:
    # only `tools/definition.py` makes tools of the SDK out of them.
    reached = {
        name
        for name in _imports(module)
        if name.startswith(("openhands.sdk", "openhands.tools"))
    }

    assert not reached


def test_what_a_tool_is_knows_no_tool_of_the_directors():
    assert not {
        name
        for name in _imports("tools/spec.py")
        if name.startswith("openhands.agent_server")
    }


@pytest.mark.parametrize(
    "module",
    [
        "org.py",
        "documents/store.py",
        "documents/declared.py",
        "documents/events.py",
        "documents/protocol.py",
        "documents/tree.py",
        "tools/spec.py",
    ],
)
def test_the_directors_own_parts_do_not_know_the_shipped_orgs_or_the_service(
    module: str,
):
    # What is particular to an organisation is under `orgs`, and the
    # service puts the parts together: neither is known to a part.
    reached = {
        name
        for name in _imports(module)
        if name.startswith(
            (
                "openhands.agent_server.director.orgs",
                "openhands.agent_server.director.service",
            )
        )
    }

    assert not reached


def test_where_documents_are_kept_knows_no_tool_type_or_tree():
    reached = {
        name.removeprefix("openhands.agent_server.director.")
        for name in _imports("documents/store.py")
        if name.startswith("openhands.agent_server.director.")
    }

    assert reached == {"git_store", "run_state"}


@pytest.mark.parametrize("module", ["orgs/documents/contracts.py", "orgs/top_down.py"])
def test_what_a_document_does_reaches_the_director_only_through_documents(
    module: str,
):
    # A document acts through the store it is given. It knows nothing of
    # how teams are split, reviewed or merged.
    reached = {
        name.removeprefix("openhands.agent_server.director.")
        for name in _imports(module)
        if name.startswith("openhands.agent_server.director.")
    }

    assert all(name.startswith(("documents.", "org")) for name in reached)


@pytest.mark.parametrize(
    "module", ["tools/catalog.py", "agent_actions.py", "service/records.py"]
)
def test_the_director_has_no_tool_for_any_one_document(module: str):
    # A document's tool is made of its YAML and its protocol.
    text = (DIRECTOR / module).read_text(encoding="utf-8")

    assert "def write_spec" not in text and "def edit_contracts" not in text
    assert "WRITE_SPEC" not in text and "EDIT_CONTRACTS" not in text
