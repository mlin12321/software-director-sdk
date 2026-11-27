"""Which parts of the director may know of which."""

import ast
import re
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
        "documents/protocol.py",
        "documents/schema.py",
        "documents/writer.py",
        "documents/registry.py",
        "documents/team_documents.py",
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


SERVICES_THE_DIRECTOR_HOLDS = [
    "service/org_save_service.py",
    "service/resumable_work_service.py",
    "service/work_queue_service.py",
    "service/model_call_service.py",
    "service/request_routing_service.py",
    "service/session_draft_service.py",
    "service/stall_watch_service.py",
    "service/team_type_service.py",
]


@pytest.mark.parametrize("module", SERVICES_THE_DIRECTOR_HOLDS)
def test_a_service_the_director_holds_is_not_given_the_director(module: str):
    # A part of the service is given what it needs and is held by the
    # service; it does not reach back into the mixins that hold it.
    reached = {
        name
        for name in _imports(module)
        if name.startswith("openhands.agent_server.director.service")
    }
    held = {
        "openhands.agent_server.director." + path.removesuffix(".py").replace("/", ".")
        for path in SERVICES_THE_DIRECTOR_HOLDS
    }

    # One such service may hold another, as the agent server's own do.
    assert reached <= held
    assert "DirectorService" not in (DIRECTOR / module).read_text(encoding="utf-8")


def test_where_documents_are_kept_knows_no_tool_type_or_tree():
    reached = {
        name.removeprefix("openhands.agent_server.director.")
        for name in _imports("documents/store.py")
        if name.startswith("openhands.agent_server.director.")
    }

    # `models` for what an entry of a log is.
    assert reached == {"git_store", "models", "run_state"}


@pytest.mark.parametrize(
    "module",
    [
        "orgs/documents/contracts.py",
        "orgs/documents/report.py",
        "orgs/documents/spec.py",
        "orgs/top_down.py",
    ],
)
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

    assert "def write_spec" not in text and "def write_contracts" not in text
    assert "WRITE_SPEC" not in text and "EDIT_CONTRACTS" not in text
    assert "def report(" not in text and "REPORT = ToolSpec" not in text


_DOCUMENT_NAMES = re.compile(
    r"\b(specs?|contracts?)\b|write_spec|write_contracts", re.IGNORECASE
)
# Names of the director's own that hold the word and are no document's.
_NOT_A_DOCUMENT = re.compile(r"pathspecs?|ToolSpec|tools?\.spec\b")


@pytest.mark.parametrize(
    "module",
    [
        "prompts.py",
        "context.py",
        "models.py",
        "agent_actions.py",
        "intents.py",
        "tools/catalog.py",
        *sorted(f"service/{path.name}" for path in (DIRECTOR / "service").glob("*.py")),
    ],
)
def test_the_director_names_no_document_an_org_declares(module: str):
    # Spec and contracts are documents of the shipped orgs: what the
    # director does with a document it does by what the document's
    # protocol says, and what it tells of one comes from the document.
    named = [
        line.strip()
        for line in (DIRECTOR / module).read_text(encoding="utf-8").splitlines()
        if _DOCUMENT_NAMES.search(_NOT_A_DOCUMENT.sub("", line))
    ]

    assert named == []
