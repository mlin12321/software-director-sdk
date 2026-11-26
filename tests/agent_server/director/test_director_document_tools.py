"""A document an org keeps brings the tool it is written with."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

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
GLOSSARY = "holds: What the words of this team's work mean.\n"


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
    # The spec may be written by the member that takes over what a removed
    # team left; a plain document may not.
    assert service.hand_over_tools == ("read_team_context", "hand_over", "write_spec")
