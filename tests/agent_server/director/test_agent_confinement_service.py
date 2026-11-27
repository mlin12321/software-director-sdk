"""What holds the org's agents by itself: a repository, a store of teams
and a run state in memory, and no director."""

from pathlib import Path
from uuid import uuid4

import pytest

from openhands.agent_server.director.acp import agent_token
from openhands.agent_server.director.environment.git import (
    GitCodingEnvironment,
)
from openhands.agent_server.director.environment.git.merge_checkout import MergeCheckout
from openhands.agent_server.director.models import SessionTeam
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.run_state import MERGING, RunState
from openhands.agent_server.director.service.agent_confinement_service import (
    AgentConfinementService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.errors import NotFound
from openhands.tools.file_editor.definition import FileEditorAction
from tests.agent_server.director.conftest import open_session_in, small_tree
from tests.agent_server.director.test_team_documents import (
    RootAboveApi,
    TextsInADict,
)


def service_for(
    repo: Path, run: RunState | None = None, sandbox: str | None = None
) -> AgentConfinementService:
    store = OrgStore(lambda node: team_types.type_of(node))
    team_types = TeamTypeService(
        config_file=None,
        documents=TextsInADict(),
        tree=RootAboveApi(),
        org=lambda: store,
    )
    for node in small_tree():
        store.upsert(node)
    kept = run or RunState(None)
    return AgentConfinementService(
        environment=GitCodingEnvironment(),
        repo_dir=lambda: repo,
        worktree_root=None,
        find_sandbox=lambda: sandbox,
        remembered=RememberedFile(repo.parent / "director.json"),
        server_url=None,
        org=lambda: store,
        team_types=team_types,
        run_state=lambda: kept,
    )


@pytest.fixture
def confinement(repo: Path) -> AgentConfinementService:
    return service_for(repo)


def place(confinement: AgentConfinementService, team_id: str) -> SessionTeam:
    """The place of a team in a session opened at the top, once."""
    store = confinement.org()
    if not store.places():
        open_session_in(store, "root", "s1", checkout=str(confinement.repo_dir()))
    found = store.instance(team_id, "s1")
    assert found is not None
    return found


def test_a_token_names_the_conversation_it_was_made_for(
    confinement: AgentConfinementService,
):
    conversation_id = uuid4()
    token = agent_token(confinement.tool_key(), str(conversation_id))

    assert confinement.agent_of(token) == conversation_id


def test_a_token_this_director_did_not_give_out_names_no_agent(
    confinement: AgentConfinementService,
):
    with pytest.raises(NotFound):
        confinement.agent_of(agent_token(b"another key", str(uuid4())))


def test_the_key_of_the_tokens_is_the_same_after_a_restart(repo: Path):
    first = service_for(repo).tool_key()

    assert service_for(repo).tool_key() == first


def test_only_the_member_that_works_in_the_folder_writes_there(
    confinement: AgentConfinementService,
):
    a = place(confinement, "a")
    team = confinement.team_types.team(a)

    writer = confinement.scope_of(a, team.writer)
    other = confinement.scope_of(a, team.manager)

    assert (writer.writes, writer.edits) == (True, True)
    assert (other.writes, other.edits) == (False, False)
    assert writer.directory == a.boundary.path


def test_the_folders_of_the_teams_below_are_left_to_them(
    confinement: AgentConfinementService,
):
    a = place(confinement, "a")

    scope = confinement.scope_of(a, confinement.team_types.team(a).writer)

    assert set(scope.hidden) == {"svc/a/one", "svc/a/two"}


def test_a_hand_over_conversation_changes_no_file(
    confinement: AgentConfinementService,
):
    a = place(confinement, "a")
    writer = confinement.team_types.team(a).writer

    scope = confinement.scope_of(a, writer, hand_over=str(uuid4()))

    assert scope.aside and not scope.writes
    assert scope.merge_root == ""


def test_only_the_reviewer_of_the_team_above_finishes_a_teams_merge(repo: Path):
    run = RunState(None)
    confinement = service_for(repo, run)
    root, a = place(confinement, "root"), place(confinement, "a")
    merge = MergeCheckout(
        where=str(confinement.merges_for(root.id) / "a"),
        ours="1" * 40,
        tip="2" * 40,
        files=["x.py"],
    )
    run.mark(MERGING, a.id, merge.model_dump_json())
    team = confinement.team_types.team(root)

    reviewing = confinement.scope_of(root, team.reviewer)
    elsewhere = confinement.scope_of(a, confinement.team_types.team(a).reviewer)

    assert [area.own for area in reviewing.merges] == [True]
    assert [area.own for area in elsewhere.merges] == [False]
    assert reviewing.merge_root == str(confinement.merges_for(root.id))


def test_without_a_sandbox_file_tools_are_checked_and_own_tools_are_not_held(
    confinement: AgentConfinementService,
):
    confinement.look_for_sandbox()

    assert not confinement.is_sandboxed
    assert confinement.enforcement("openhands") == "checks"
    assert confinement.enforcement("acp") == "merge"


def test_with_a_sandbox_every_agent_is_held_by_it(repo: Path):
    confinement = service_for(repo, sandbox="/usr/bin/bwrap")

    confinement.look_for_sandbox()

    assert confinement.is_sandboxed
    assert confinement.enforcement("acp") == "sandbox"


def test_no_file_tool_opens_what_the_director_keeps_for_itself(
    confinement: AgentConfinementService,
):
    confinement.tool_key()
    kept = confinement.remembered.path
    assert kept is not None
    view = FileEditorAction(command="view", path=str(kept)).model_dump()

    denied = confinement.check(None, "file_editor", view)

    assert denied is not None and denied.kind == "private"


def test_a_team_that_is_gone_gets_no_shell(confinement: AgentConfinementService):
    assert confinement.confine("gone") is None
