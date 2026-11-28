"""Which parts of the director may know of which."""

import ast
import re
import subprocess
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


SERVICES_THE_DIRECTOR_HOLDS = sorted(
    f"service/{path.name}" for path in (DIRECTOR / "service").glob("*_service.py")
)


@pytest.mark.parametrize("module", SERVICES_THE_DIRECTOR_HOLDS)
def test_a_service_the_director_holds_is_not_given_the_director(module: str):
    # A part of the service is given what it needs and is held by the
    # service; it does not reach back into what holds it.
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
    assert reached == {"environment.records", "models", "run_state"}


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
    "module", ["tools/catalog.py", "agent_actions.py", "service/__init__.py"]
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


GIT = "openhands.agent_server.director.environment.git"
# The one place the environment is built.
_MAY_REACH_GIT = {"director/service/__init__.py"}


def _reached(path: Path) -> set[str]:
    """What a module imports, a module taken from its package included."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_only_the_environment_knows_git():
    # Checkouts, the org's records and its remote are reached through
    # `CodingEnvironment`; nothing else imports what runs git.
    modules = [
        path
        for path in [*DIRECTOR.rglob("*.py"), DIRECTOR.parent / "director_router.py"]
        if "environment/git" not in path.as_posix()
        and path.relative_to(DIRECTOR.parent).as_posix() not in _MAY_REACH_GIT
    ]
    reached = {
        f"{path.relative_to(DIRECTOR.parent)} imports {name}"
        for path in modules
        for name in _reached(path)
        if name == GIT or name.startswith(f"{GIT}.")
    }

    assert not reached


def test_records_kept_in_memory_run_no_git():
    # The second implementation of the records and the remote is what
    # shows that their interfaces say nothing only git can do.
    reached = {
        f"{path.name} imports {name}"
        for path in (DIRECTOR / "environment" / "memory").glob("*.py")
        for name in _reached(path)
        if name == GIT or name.startswith(f"{GIT}.") or name == "subprocess"
    }

    assert not reached


@pytest.mark.parametrize(
    "module",
    [
        "__init__.py",
        "checkouts.py",
        "confinement.py",
        "records.py",
        "remote.py",
        "errors.py",
    ],
)
def test_the_environment_is_said_without_git(module: str):
    prefix = "openhands.agent_server.director."
    reached = {
        name.removeprefix(prefix)
        for name in _imports(f"environment/{module}")
        if name.startswith(prefix)
    }

    assert reached <= {
        "models",
        "scope",
        "environment.checkouts",
        "environment.confinement",
        "environment.errors",
        "environment.records",
        "environment.remote",
    }


def test_the_director_is_one_class_that_holds_services():
    # Everything else it does is done by a service it holds.
    from openhands.agent_server.director.service import DirectorService

    assert DirectorService.__mro__ == (DirectorService, object)
    modules = {path.name for path in (DIRECTOR / "service").glob("*.py")}
    assert {name for name in modules if not name.endswith("_service.py")} == {
        "__init__.py"
    }


# The files of OpenHands the director changes, said with the reason for each
# in ``director/AGENTS.md``. A change to one more conflicts when upstream is
# merged: look for a way from outside before adding to these.
CHANGED_IN_OPENHANDS = {
    "openhands-agent-server/openhands/agent_server/README.md",
    "openhands-agent-server/openhands/agent_server/__main__.py",
    "openhands-agent-server/openhands/agent_server/agent-server.spec",
    "openhands-agent-server/openhands/agent_server/api.py",
    "openhands-agent-server/openhands/agent_server/config.py",
    "openhands-agent-server/openhands/agent_server/conversation_service.py",
    "openhands-agent-server/openhands/agent_server/dependencies.py",
    "openhands-agent-server/openhands/agent_server/event_service.py",
    "openhands-agent-server/openhands/agent_server/init_router.py",
    "openhands-agent-server/openhands/agent_server/server_context.py",
    "openhands-sdk/openhands/sdk/agent/acp_agent.py",
    "openhands-sdk/openhands/sdk/agent/base.py",
    "openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py",
}
UPSTREAM = "origin/upstream-base"
PACKAGES = [
    "openhands-agent-server/openhands",
    "openhands-sdk/openhands",
    "openhands-tools/openhands",
    "openhands-workspace/openhands",
]


def test_the_director_changes_no_more_of_openhands():
    root = DIRECTOR.parents[3]

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )

    if git("rev-parse", "--verify", "--quiet", UPSTREAM).returncode != 0:
        pytest.skip(f"{UPSTREAM} is not fetched here")
    # The upstream this checkout has merged: the ref may be ahead of it.
    merged = git("merge-base", UPSTREAM, "HEAD")
    assert merged.returncode == 0, merged.stderr
    # Files OpenHands has that differ here; new files (A) are the director's.
    changed = git(
        "diff",
        "--name-only",
        "--diff-filter=MDR",
        merged.stdout.strip(),
        "--",
        *PACKAGES,
    )

    assert changed.returncode == 0, changed.stderr
    assert set(changed.stdout.split()) <= CHANGED_IN_OPENHANDS
