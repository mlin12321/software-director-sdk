"""The agent types shipped for research."""

from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import load_types
from openhands.agent_server.director.documents.declared import load_documents
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS, SHIPPED_TYPES
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.team_type_service import (
    DIRECTOR_TOOL_NAMES,
)
from tests.agent_server.director.conftest import (
    FakeLauncher,
    an_org,
    node,
)


DOCUMENTS = load_documents([SHIPPED_DOCUMENTS])


def test_the_research_types_load_and_say_how_research_is_done_and_reviewed():
    types = load_types(SHIPPED_TYPES, DIRECTOR_TOOL_NAMES, DOCUMENTS)

    assert {"software-engineer", "researcher", "research-engineer"} <= set(types)
    researcher = types["researcher"]
    assert "Split by hypothesis" in researcher.text_for("ic")
    assert "not a failing team" in researcher.text_for("manager")
    assert "Check the claims, not the prose." in researcher.review.checklist
    assert "command that reproduces it" in researcher.review.write_up
    assert types["research-engineer"].review.checklist_skill == "codereview"
    for kind in types.values():
        assert kind.text_for("ic") and kind.text_for("manager")


@pytest.mark.parametrize("kind", ["researcher", "research-engineer"])
async def test_an_org_can_have_a_research_type_at_its_top(
    repo: Path, launcher: FakeLauncher, kind: str
):
    service: DirectorService = await an_org(repo, launcher, agent_type=kind)
    try:
        assert node(service, "root").agent_type == kind
        assert launcher.sent or launcher.root is not None
        briefed = "\n".join(text for _, text in launcher.sent)
        assert f'<TEAM_TYPE name="{kind}">' in briefed
    finally:
        await service.aclose()
