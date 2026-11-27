"""Team types by themselves: the files that declare them, a store of
teams, and no director."""

import os
from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import DEFAULT_TYPE
from openhands.agent_server.director.models import OrgNodeInput
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.service.team_type_service import (
    DIRECTOR_TOOL_NAMES,
    TeamTypeService,
)
from tests.agent_server.director.conftest import (
    ANALYST,
    open_session_in,
    org_config,
    small_tree,
)
from tests.agent_server.director.test_team_documents import (
    RootAboveApi,
    TextsInADict,
)


def service_for(config_file: Path | None) -> TeamTypeService:
    store = OrgStore(lambda node: team_types.type_of(node))
    team_types = TeamTypeService(
        config_file=config_file,
        documents=TextsInADict(),
        tree=RootAboveApi(),
        org=lambda: store,
    )
    for node in small_tree():
        store.upsert(node)
    return team_types


@pytest.fixture
def team_types() -> TeamTypeService:
    return service_for(None)


def team(team_types: TeamTypeService, team_id: str) -> OrgNodeInput:
    found = team_types.org().team(team_id)
    assert found is not None
    return found


def test_the_types_the_director_ships_are_there_without_an_org_of_its_own(
    team_types: TeamTypeService,
):
    assert DEFAULT_TYPE in team_types.agent_types
    assert "report" in team_types.protocols


def test_an_orgs_own_type_is_read_from_beside_its_settings(tmp_path: Path):
    team_types = service_for(org_config(tmp_path))

    assert "analyst" in team_types.agent_types


def test_a_type_added_later_is_found_without_a_restart(tmp_path: Path):
    config = org_config(tmp_path)
    team_types = service_for(config)
    assert "auditor" not in team_types.agent_types

    (config.parent / "types" / "auditor.yaml").write_text(ANALYST)

    assert "auditor" in team_types.agent_types


def test_a_type_that_changed_is_read_again(tmp_path: Path):
    config = org_config(tmp_path)
    team_types = service_for(config)
    file = config.parent / "types" / "analyst.yaml"
    before = team_types.agent_types["analyst"].description

    file.write_text(ANALYST.replace("Reads results", "Weighs results"))
    stamp = file.stat().st_mtime_ns + 1_000_000
    os.utime(file, ns=(stamp, stamp))

    assert team_types.agent_types["analyst"].description != before


def test_a_team_whose_type_is_gone_is_read_as_the_default_type(
    team_types: TeamTypeService,
):
    gone = team(team_types, "a").model_copy(update={"agent_type": "astrologer"})

    assert team_types.type_of(gone).name == DEFAULT_TYPE


def test_every_tool_offered_is_the_directors_or_a_documents(
    team_types: TeamTypeService,
):
    written_with = {
        protocol.document.written_with for protocol in team_types.protocols.values()
    }

    assert set(team_types.tool_names) == {*DIRECTOR_TOOL_NAMES, *written_with}
    assert set(team_types.hand_over_tools) <= set(team_types.tool_names)


def test_a_type_cannot_take_what_an_agent_needs_to_hand_its_work_on(
    tmp_path: Path,
):
    team_types = service_for(org_config(tmp_path))
    opened = open_session_in(team_types.org(), "root", "s1")
    place = opened.model_copy(update={"agent_type": "analyst"})
    writer = team_types.team(place).writer

    offered = team_types.tools_for(place, writer)

    # The type omits both; only the one no type removes stays.
    assert "read_team_context" in offered
    assert "propose_teams" not in offered


def test_a_member_the_type_does_not_name_is_offered_nothing(
    team_types: TeamTypeService,
):
    place = open_session_in(team_types.org(), "root", "s1")

    assert team_types.tools_for(place, "nobody") == ()


def test_a_team_that_is_gone_is_named_as_one_of_the_default_type(
    team_types: TeamTypeService,
):
    assert team_types.names_at("gone") == team_types.agent_types[DEFAULT_TYPE].names


def test_the_report_is_not_among_the_documents_a_team_is_told_it_keeps(
    team_types: TeamTypeService,
):
    root = team(team_types, "root")

    kept = team_types.kept_documents(root)

    assert "report" in team_types.type_of(root).documents
    assert kept.reports and "report" not in kept.names
    assert set(kept.names) <= set(team_types.reader_of(root).keeps)
