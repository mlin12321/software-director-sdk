"""A document an org keeps brings the tool it is written with."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.documents.declared import Document
from openhands.agent_server.director.documents.protocol import (
    DocumentProtocolBase,
    declare,
    register_protocol,
)
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.state import team_ref
from openhands.agent_server.director.tools.definition import tool_named
from openhands.agent_server.director.tools.remote import call_tool, list_tools
from tests.agent_server.director.conftest import (
    BENCH,
    LAB,
    FakeLauncher,
    agent_conversation,
    an_org,
    node,
    use,
)


# A document that is one YAML file: no protocol, no tool named, no Python.
GLOSSARY = "description: What the words of this team's work mean.\n"


@pytest.fixture
async def service(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> AsyncIterator[DirectorService]:
    config = tmp_path / "config" / "director.yaml"
    (config.parent / "types").mkdir(parents=True)
    (config.parent / "documents").mkdir()
    config.write_text("")
    (config.parent / "documents" / "glossary.yaml").write_text(GLOSSARY)
    lab = LAB.replace("tools: [read_team_context, raise", GIVEN_TO_THE_BENCH, 1)
    assert lab != LAB
    (config.parent / "types" / "lab.yaml").write_text(lab + KEEPS_GLOSSARY)
    service = await an_org(repo, launcher, config, agent_type="lab")
    yield service
    await service.aclose()


GIVEN_TO_THE_BENCH = "tools: [write_glossary, read_team_context, raise"
KEEPS_GLOSSARY = "documents:\n    glossary:\n"


def bench(service: DirectorService) -> str:
    found = agent_conversation(node(service, "root"), "bench")
    assert found is not None
    return found


def glossary(service: DirectorService) -> str | None:
    root = node(service, "root")
    return service.documents.read(root.session, team_ref(root).keeps("glossary"))


async def test_a_document_of_the_orgs_own_is_written_with_a_tool_made_for_it(
    service: DirectorService, launcher: FakeLauncher
):
    assert "write_glossary" in service.tool_names
    assert launcher.root_tools == ["write_glossary", *BENCH]
    offered = {tool.name: tool for tool in list_tools(service, UUID(bench(service)))}
    assert "glossary" in offered["write_glossary"].description

    tool = tool_named("write_glossary")
    assert tool is not None
    answer = await use(
        service,
        bench(service),
        tool,
        tool.action()(file_text="# Glossary\nA run is one training job."),
    )

    assert not answer.is_error
    assert glossary(service) == "# Glossary\nA run is one training job."


async def test_a_plain_document_is_changed_in_one_place_as_a_file_is(
    service: DirectorService,
):
    conversation = UUID(bench(service))
    call_tool(service, conversation, "write_glossary", {"file_text": "A run is a job."})

    changed = call_tool(
        service,
        conversation,
        "write_glossary",
        {"command": "str_replace", "old_str": "a job", "new_str": "one job"},
    )
    empty = call_tool(service, conversation, "write_glossary", {"file_text": " "})

    assert not changed.is_error
    assert glossary(service) == "A run is one job."
    assert empty.is_error and "cannot be empty" in empty.text


async def test_only_what_a_document_allows_is_offered_in_a_hand_over(
    service: DirectorService,
):
    # The spec and the contracts may be written by the member that takes
    # over what a removed team left; a plain document may not.
    assert service.hand_over_tools == (
        "read_team_context",
        "hand_over",
        "write_contracts",
        "write_spec",
    )


async def test_a_team_is_told_the_documents_its_type_keeps_under_their_names(
    service: DirectorService,
):
    root = node(service, "root")
    before = service.context_of(root, "bench")
    call_tool(
        service,
        UUID(bench(service)),
        "write_glossary",
        {"file_text": "A run is a job."},
    )

    after = service.context_of(root, "bench")

    assert "glossary" in before and before["glossary"] is None
    assert after["glossary"] == "A run is a job."
    # Whoever looks at the team is given it as the team reads it.
    assert [found.model_dump() for found in service.documents_of(root)] == [
        {
            "name": "glossary",
            "description": "What the words of this team's work mean.",
            "path": "",
            "text": "A run is a job.",
            "own": True,
            "left": False,
            "left_by": None,
        }
    ]
    # Its type keeps the glossary alone.
    assert "spec" not in after


async def test_a_team_is_told_nothing_of_a_document_it_does_not_keep(
    repo: Path, launcher: FakeLauncher
):
    service = await an_org(repo, launcher)

    context = service.context_of(node(service, "root"), "ic")

    assert "glossary" not in context and "report" not in context
    assert context["spec"] is None
    assert context["contracts_between_your_sub_teams"] == []
    await service.aclose()


async def test_a_report_is_an_entry_its_teams_members_are_told(
    repo: Path, launcher: FakeLauncher
):
    service = await an_org(repo, launcher)
    root = node(service, "root")
    ic = agent_conversation(root, "ic")
    assert ic is not None

    recorded = call_tool(service, UUID(ic), "report", {"text": " Chose SQLite. "})
    empty = call_tool(service, UUID(ic), "report", {"text": "  "})

    assert not recorded.is_error
    assert empty.is_error and "cannot be empty" in empty.text
    [told] = service.context_of(root, "manager")["reports_of_your_team"]
    assert (told["by"], told["text"]) == ("ic", "Chose SQLite.")
    [kept] = service.own_reports(root)
    assert kept.session_id == root.session and kept.at and kept.id
    await service.aclose()


_OWN_TOOLS = ("propose_teams", "review_team", "hand_over", "settle_teams")


def _taken(tool: str) -> dict[str, str]:
    """What a tool of the director's own takes: each argument and what
    its caller is told of it."""
    made = tool_named(tool)
    assert made is not None
    return {
        name: field.description or ""
        for name, field in made.action().model_fields.items()
    }


class _Minutes(DocumentProtocolBase):
    """A document with a protocol of an org's own."""


def test_a_document_adds_nothing_to_the_tools_of_the_directors_own():
    before = {tool: _taken(tool) for tool in _OWN_TOOLS}
    register_protocol("minutes-of-meetings", _Minutes)

    declare(
        Document(
            name="minutes",
            description="What was agreed.",
            protocol="minutes-of-meetings",
        )
    )

    # Every org is offered the same tools, whatever documents it keeps.
    assert {tool: _taken(tool) for tool in _OWN_TOOLS} == before
    assert tool_named("write_minutes") is not None
    assert set(before["propose_teams"]) == {"teams", "keep", "text"}
    assert set(before["hand_over"]) == {"team", "text"}
    # Which of two versions of a document holds is asked by the tools that
    # merge a session's work, of whatever documents the org keeps.
    for tool in ("review_team", "settle_teams"):
        said = before[tool]["merge_choices"]
        assert "spec" not in said and "contracts" not in said


async def test_a_team_is_not_given_a_document_its_type_does_not_keep(
    service: DirectorService,
):
    # The lab's type keeps a glossary and no contracts.
    refused = call_tool(
        service,
        UUID(bench(service)),
        "write_contracts",
        {"command": "create", "file_text": "Rig serves Assay."},
    )

    assert refused.is_error
