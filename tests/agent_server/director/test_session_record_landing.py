"""A session's records landing in the org by themselves: a repository, a
store of teams and a run state in memory, and no director."""

from pathlib import Path
from typing import cast

import pytest

from openhands.agent_server.director.documents.store import (
    DocumentRef,
    RecordDocumentStore,
)
from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.models import (
    Report,
    SessionTeam,
    TeamEvent,
    TeamEventKind,
)
from openhands.agent_server.director.org_persistence import load_org
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import RunState
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.org_save_service import OrgSaveService
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.session_record_service import (
    SessionRecordService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.agent_server.director.service.work_queue_service import WorkQueueService
from tests.agent_server.director.conftest import open_session_in, small_tree
from tests.agent_server.director.test_team_documents import AnyOrg, RootAboveApi


AT = "2026-01-01T00:00:00Z"
SESSION = "s1"
A_SPEC = DocumentRef("svc/a", "spec")
B_SPEC = DocumentRef("svc/b", "spec")


def event(kind: TeamEventKind, team: SessionTeam, detail: str) -> TeamEvent:
    return TeamEvent(at=AT, kind=kind, team_id=team.team_id, detail=detail)


@pytest.fixture
def landing(repo: Path) -> SessionRecordService:
    run = RunState(None)
    store = OrgStore(lambda node: team_types.type_of(node))
    documents = RecordDocumentStore(lambda: GitStore(repo), lambda: run)
    team_types = TeamTypeService(
        config_file=None,
        documents=documents,
        tree=RootAboveApi(),
        org=lambda: store,
    )
    for node in small_tree():
        store.upsert(node)
    open_session_in(store, "root", SESSION)
    return SessionRecordService(
        org=lambda: store,
        run_state=lambda: run,
        team_memory=lambda: GitStore(repo),
        documents=documents,
        require_team_memory=lambda: GitStore(repo),
        team_documents=lambda: TeamDocuments(
            documents, team_types.protocols, AnyOrg(RootAboveApi())
        ),
        confinement=cast(AgentConfinementService, None),
        team_types=team_types,
        session_drafts=SessionDraftService(run_state=lambda: run),
        org_save=OrgSaveService(
            environment=GitCodingEnvironment(),
            org=lambda: store,
            repo_dir=lambda: repo,
            team_memory=lambda: GitStore(repo),
            run_state=lambda: run,
            work_queue=WorkQueueService(default_lane="org"),
            delay=60,
        ),
        event=event,
    )


def top(landing: SessionRecordService) -> SessionTeam:
    return place(landing, "root")


def place(landing: SessionRecordService, team_id: str) -> SessionTeam:
    found = landing.org().instance(team_id, SESSION)
    assert found is not None
    return found


def report(text: str) -> Report:
    return Report(at=AT, role="ic", text=text)


def test_a_document_a_session_wrote_becomes_the_orgs_and_is_its_draft_no_more(
    landing: SessionRecordService,
):
    landing.documents.write(SESSION, A_SPEC, "Take orders.")

    taken = landing.land(top(landing))

    assert taken.left == []
    assert landing.team_memory().read(A_SPEC.path) == "Take orders."
    assert landing.documents.drafts(SESSION) == {}


def test_what_agents_recorded_lands_with_their_team_and_is_forgotten(
    landing: SessionRecordService,
):
    a = place(landing, "a")
    landing.session_drafts.keep_report(a.id, report("Orders are idempotent."))

    taken = landing.land(top(landing))

    assert taken.grown == [("svc/a", "ic")]
    kept = landing.team_memory().reports("svc/a")
    assert [entry.text for entry in kept] == ["Orders are idempotent."]
    assert landing.session_drafts.reports(a.id) == []


def test_what_is_of_a_team_that_waits_stays_the_sessions(
    landing: SessionRecordService,
):
    a, b = place(landing, "a"), place(landing, "b")
    landing.documents.write(SESSION, A_SPEC, "Take orders.")
    landing.documents.write(SESSION, B_SPEC, "Bill them.")
    landing.session_drafts.keep_report(a.id, report("Waits."))
    landing.session_drafts.keep_report(b.id, report("Lands."))

    landing.land(top(landing), waits=["a"])

    memory = landing.team_memory()
    assert (memory.read(A_SPEC.path), memory.read(B_SPEC.path)) == (None, "Bill them.")
    assert set(landing.documents.drafts(SESSION)) == {A_SPEC}
    assert [entry.text for entry in landing.session_drafts.reports(a.id)] == ["Waits."]
    assert landing.session_drafts.reports(b.id) == []


def test_a_file_that_is_held_stays_a_draft(landing: SessionRecordService):
    landing.documents.write(SESSION, A_SPEC, "Take orders.")

    landing.land(top(landing), held=[A_SPEC.path])

    assert landing.team_memory().read(A_SPEC.path) is None
    assert set(landing.documents.drafts(SESSION)) == {A_SPEC}


def test_a_document_someone_else_changed_meanwhile_is_not_written_over(
    landing: SessionRecordService,
):
    memory = landing.team_memory()
    memory.write("svc/a", "spec.md", "First.")
    landing.documents.write(SESSION, A_SPEC, "The session's.")
    memory.write("svc/a", "spec.md", "Someone else's.")

    taken = landing.land(top(landing))

    assert taken.left == [A_SPEC.path]
    assert landing.team_memory().read(A_SPEC.path) == "Someone else's."
    # It stays the session's, to be shown to a reviewer.
    assert set(landing.documents.drafts(SESSION)) == {A_SPEC}


def test_a_changed_document_is_written_over_where_its_reviewer_was_shown_it(
    landing: SessionRecordService,
):
    memory = landing.team_memory()
    memory.write("svc/a", "spec.md", "First.")
    landing.documents.write(SESSION, A_SPEC, "The session's.")
    memory.write("svc/a", "spec.md", "Someone else's.")

    taken = landing.land(top(landing), seen={A_SPEC.path: "Someone else's."})

    assert taken.left == []
    assert landing.team_memory().read(A_SPEC.path) == "The session's."


def test_the_tree_lands_with_the_records_only_when_it_is_asked_to(
    landing: SessionRecordService, repo: Path
):
    landing.documents.write(SESSION, A_SPEC, "Take orders.")
    landing.land(top(landing))
    assert GitCodingEnvironment().org_tree(repo).head() is None

    landing.documents.write(SESSION, B_SPEC, "Bill them.")
    landing.land(top(landing), tree=True)

    saved = OrgStore(landing.team_types.type_of)
    assert load_org(GitCodingEnvironment().org_tree(repo), saved)
    assert {team.id for team in saved.teams()} == {"root", "a", "b", "a1", "a2"}


def test_what_a_lost_team_left_goes_to_the_nearest_team_that_holds_its_folder(
    landing: SessionRecordService,
):
    session_top = top(landing)

    assert landing.home_of(session_top, "svc/a/one/deep") == "a1"
    assert landing.home_of(session_top, "svc/a/three") == "a"
    assert landing.home_of(session_top, "tools") == "root"


def test_a_place_is_the_orgs_only_while_the_org_has_that_very_team(
    landing: SessionRecordService,
):
    store = landing.org()
    a = place(landing, "a")

    assert store.is_orgs(a)
    # Another team was set up in the same folder since: it has the id.
    assert not store.is_orgs(a.model_copy(update={"since": "later"}))
    assert not store.is_orgs(a.model_copy(update={"team_id": "gone"}))
