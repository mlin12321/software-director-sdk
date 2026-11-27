"""The reports of a team by themselves: a repository, a store of teams
and a run state in memory."""

import dataclasses
from pathlib import Path

import pytest

from openhands.agent_server.director.documents.store import REPORT, TeamRef
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.models import Report
from openhands.agent_server.director.org_store import OrgStore, team_ref
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.sdk.errors import NotFound, NotSupported
from tests.agent_server.director.conftest import (
    open_session_in,
    shipped_type,
    small_tree,
)


A = "a@s1"


@pytest.fixture
def reports(repo: Path) -> TeamHistoryService:
    store = OrgStore(shipped_type)
    for node in small_tree():
        store.upsert(node)
    open_session_in(store, "root", "s1")
    run = RunState(None)
    return TeamHistoryService(
        require_team_memory=lambda: GitStore(repo),
        org_changed=lambda session: None,
        org=lambda: store,
        team_memory=lambda: GitStore(repo),
        session_drafts=SessionDraftService(run_state=lambda: run),
    )


def team(reports: TeamHistoryService) -> TeamRef:
    place = reports.org().place(A)
    assert place is not None
    return team_ref(place)


def test_what_a_member_records_is_read_back_with_its_session(
    reports: TeamHistoryService,
):
    reports.append(team(reports), REPORT, Report(at="", role="ic", text="Parser done"))

    [entry] = reports.entries(team(reports), REPORT)
    assert (entry.role, entry.text, entry.session_id) == ("ic", "Parser done", "s1")
    assert entry.at


def test_the_director_keeps_no_other_log(reports: TeamHistoryService):
    with pytest.raises(NotSupported):
        reports.append(team(reports), "journal", Report(at="", role="ic", text="x"))

    assert reports.entries(team(reports), "journal") == []


def test_a_team_that_is_no_place_of_a_session_has_no_log_to_add_to(
    reports: TeamHistoryService,
):
    gone = dataclasses.replace(team(reports), id="gone@s1")

    with pytest.raises(NotFound):
        reports.append(gone, REPORT, Report(at="", role="ic", text="x"))
