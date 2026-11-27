"""What an agent reads of its place, by itself: a repository, a store of
teams and a run state in memory, and no director."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from openhands.agent_server.director.documents.team_documents import TeamDocuments
from openhands.agent_server.director.git_store import GitStore
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.prompts import NEED_WRITE
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import JOINED
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.org_state_service import OrgStateService
from openhands.agent_server.director.service.session_draft_service import (
    SessionDraftService,
)
from openhands.agent_server.director.service.team_context_service import (
    TeamContextService,
)
from openhands.agent_server.director.service.team_history_service import (
    TeamHistoryService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.errors import NotFound
from tests.agent_server.director.conftest import open_session_in, small_tree
from tests.agent_server.director.test_team_documents import (
    AnyOrg,
    RootAboveApi,
    TextsInADict,
)


SESSION = "s1"
ROOT, A, B = f"root@{SESSION}", f"a@{SESSION}", f"b@{SESSION}"


class Context:
    def __init__(self, repo: Path) -> None:
        self.store = OrgStore(lambda node: team_types.type_of(node))
        documents = TextsInADict()
        tree = RootAboveApi()
        team_types = TeamTypeService(
            config_file=None, documents=documents, tree=tree, org=lambda: self.store
        )
        for node in small_tree():
            self.store.upsert(node)
        open_session_in(self.store, "root", SESSION, checkout=str(repo))
        self.agent = str(uuid4())
        self.store.set_agent(A, "ic", conversation_id=self.agent)
        self.org_state = OrgStateService(
            repo_dir=lambda: repo, clock=lambda: datetime.now(UTC)
        )
        session_drafts = SessionDraftService(run_state=self.org_state.run_state)
        self.history = TeamHistoryService(
            org=lambda: self.store,
            team_memory=lambda: GitStore(repo),
            require_team_memory=lambda: GitStore(repo),
            session_drafts=session_drafts,
            org_changed=lambda session: None,
        )
        self.service = TeamContextService(
            org=lambda: self.store,
            org_state=self.org_state,
            org_settings=OrgSettingsService(
                config_file=None,
                remembered=RememberedFile(None),
                team_types=team_types,
                tree=tree,
                org=lambda: self.store,
                repo_dir=lambda: repo,
                max_concurrent_runs=None,
            ),
            team_types=team_types,
            team_history=self.history,
            session_drafts=session_drafts,
            documents=documents,
            team_documents=lambda: TeamDocuments(
                documents, team_types.protocols, AnyOrg(tree)
            ),
        )


@pytest.fixture
def context(repo: Path) -> Context:
    return Context(repo)


def test_an_agent_reads_who_it_is_and_the_teams_below_it(context: Context):
    top = context.store.place(ROOT)
    assert top is not None

    told = context.service.context_of(top, "ic")

    assert told["status"] == "idle"
    assert {team["team_id"] for team in told["sub_teams"]} == {A, B}


def test_what_became_of_the_teams_below_is_told_when_asked_in_full(context: Context):
    top, a = context.store.place(ROOT), context.store.place(A)
    assert top is not None and a is not None
    context.history.record_event(ROOT, "woken", a, "it stood still")

    told = context.service.context_of(top, "manager", in_full=True)

    [line] = told["what_became_of_them_lately"]
    assert "it stood still" in line


def test_what_the_human_asked_a_team_directly_is_its_own_to_read(context: Context):
    a, b = context.store.place(A), context.store.place(B)
    assert a is not None and b is not None
    context.history.keep_routed("asked", a, "Fix the parser", "model", "its folder")

    assert context.service.asked_of(a) == ["Fix the parser"]
    assert context.service.asked_of(b) == []


def test_teams_that_needed_each_others_files_are_counted_for_the_team_above(
    context: Context,
):
    a = context.store.place(A)
    assert a is not None
    for _ in range(3):
        context.history.record_event(
            ROOT, "needed", a, NEED_WRITE, ["svc/b/x.py"], context.store.place(B)
        )

    [need] = context.service.needs_below(ROOT)

    assert sorted(need["team_ids"]) == ["a", "b"]
    assert need["writes_refused"] == 3


def test_a_conversation_is_placed_in_its_team_and_session(context: Context):
    context.org_state.run_state().mark(JOINED, context.agent)

    found = context.service.place_of(context.agent)

    assert (found.node_id, found.role, found.session_id) == ("a", "ic", SESSION)
    assert found.human_joined
    with pytest.raises(NotFound):
        context.service.place_of(str(uuid4()))
