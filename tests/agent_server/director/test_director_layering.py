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


@pytest.mark.parametrize("module", ["tools/spec.py", "tools/catalog.py"])
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
