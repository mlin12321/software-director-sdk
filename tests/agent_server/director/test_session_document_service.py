"""What a session wrote of team memory and what it leaves, by itself: a
repository, a store of teams and a run state in memory."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from openhands.agent_server.director.documents.store import (
    DocumentRef,
    RecordDocumentStore,
)
from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.environment.git.git_store import GitStore
from openhands.agent_server.director.models import Report, SessionTeam
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.run_state import USE, RunState
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.session_document_service import (
    SessionDocumentService,
)
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from tests.agent_server.director.conftest import open_session_in, small_tree
from tests.agent_server.director.test_team_documents import AnyOrg, RootAboveApi


SESSION = "s1"
ROOT, A = f"root@{SESSION}", f"a@{SESSION}"
SPEC = DocumentRef("svc/a", "spec").path


class Documents:
    def __init__(self, repo: Path) -> None:
        self.run = RunState(None)
        self.store = OrgStore(lambda node: team_types.type_of(node))
        documents = RecordDocumentStore(lambda: GitStore(repo), lambda: self.run)
        tree = RootAboveApi()
        team_types = TeamTypeService(
            config_file=None, documents=documents, tree=tree, org=lambda: self.store
        )
        for node in small_tree():
            self.store.upsert(node)
        open_session_in(self.store, "root", SESSION)
        self.drafts = SessionDraftService(run_state=lambda: self.run)
        self.dropped: list[str] = []
        self.service = SessionDocumentService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            require_team_memory=lambda: GitStore(repo),
            run_state=lambda: self.run,
            documents=documents,
            team_documents=lambda: TeamDocuments(
                documents, team_types.protocols, AnyOrg(tree)
            ),
            team_types=team_types,
            session_drafts=self.drafts,
            home_of=lambda top, path: "root",
            confinement=cast(
                AgentConfinementService,
                SimpleNamespace(drop_merge_checkout=self.dropped.append),
            ),
        )

    def place(self, place_id: str) -> SessionTeam:
        found = self.store.place(place_id)
        assert found is not None
        return found


@pytest.fixture
def documents(repo: Path) -> Documents:
    return Documents(repo)


def test_what_a_session_writes_is_its_own_to_read(documents: Documents):
    documents.service.draft(SESSION, SPEC, "# Parser")

    assert documents.service.read_file(documents.place(A), SPEC) == "# Parser"
    assert documents.service.drafts(SESSION) == {SPEC: ("# Parser", None)}

    documents.service.undraft(SESSION, SPEC)

    assert documents.service.drafts(SESSION) == {}
    assert documents.service.read_file(documents.place(A), SPEC) is None


def test_a_closed_session_takes_what_it_kept_to_itself_with_it(
    documents: Documents,
):
    documents.service.draft(SESSION, SPEC, "# Parser")
    documents.drafts.keep_report(A, Report(at="t", role="ic", text="Half done"))
    documents.run.mark(USE, SESSION, "{}")

    went = documents.service.leave(documents.place(ROOT))

    assert "1 reports" in went and SPEC in went
    assert documents.service.drafts(SESSION) == {}
    assert documents.drafts.reports(A) == []
    assert documents.run.marked(USE) == {}
    assert A in documents.dropped


def test_a_session_that_kept_nothing_leaves_nothing(documents: Documents):
    assert documents.service.leave(documents.place(ROOT)) == ""


def test_what_the_team_above_says_of_the_teams_before_is_kept_for_the_new_one(
    documents: Documents,
):
    documents.service.removed_team_handed_over(
        documents.place(A), documents.place(ROOT), "ic", "The lexer was half done."
    )

    [entry] = documents.drafts.handed_over_to(A)
    assert entry["text"] == "The lexer was half done."
    assert entry["by"].endswith("of root")
