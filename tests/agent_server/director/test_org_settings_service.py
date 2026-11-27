"""Settings by themselves: a settings file, the file the server keeps
things in, a store of teams and a repository, and no director."""

from pathlib import Path

import pytest

from openhands.agent_server.director.environment.git import GitCodingEnvironment
from openhands.agent_server.director.models import (
    DirectorSettingsUpdate,
    OrgSettings,
)
from openhands.agent_server.director.org_config import InvalidOrgConfig
from openhands.agent_server.director.org_store import OrgStore
from openhands.agent_server.director.remembered import RememberedFile
from openhands.agent_server.director.service.org_settings_service import (
    OrgSettingsService,
)
from openhands.agent_server.director.service.team_type_service import (
    TeamTypeService,
)
from openhands.sdk.errors import Invalid, NotFound
from tests.agent_server.director.conftest import (
    open_session_in,
    org_config,
    small_tree,
)
from tests.agent_server.director.test_team_documents import (
    RootAboveApi,
    TextsInADict,
)


def service_for(
    repo: Path,
    config_file: Path | None = None,
    *,
    teams: bool = True,
    max_concurrent_runs: int | None = None,
) -> OrgSettingsService:
    store = OrgStore(lambda node: team_types.type_of(node))
    tree = RootAboveApi()
    team_types = TeamTypeService(
        config_file=config_file,
        documents=TextsInADict(),
        tree=tree,
        org=lambda: store,
    )
    if teams:
        for node in small_tree():
            store.upsert(node)
    return OrgSettingsService(
        environment=GitCodingEnvironment(),
        config_file=config_file,
        remembered=RememberedFile(repo.parent / "director.json"),
        team_types=team_types,
        tree=tree,
        documents=team_types.documents,
        org=lambda: store,
        repo_dir=lambda: repo,
        max_concurrent_runs=max_concurrent_runs,
    )


def test_without_a_settings_file_the_settings_are_as_last_saved(repo: Path):
    service_for(repo).update_settings(DirectorSettingsUpdate(max_depth=3))

    assert service_for(repo).settings.max_depth == 3


def test_only_what_was_changed_holds_over_the_settings_file(repo: Path, tmp_path: Path):
    config = org_config(tmp_path, "max_teams: 5\nmax_depth: 4\n")
    service_for(repo, config).update_settings(
        DirectorSettingsUpdate(max_teams=5, max_depth=2)
    )
    config.write_text("max_teams: 7\nmax_depth: 4\n")

    again = service_for(repo, config).settings

    # max_teams was sent as it was, so the file still says what it is.
    assert (again.max_teams, again.max_depth) == (7, 2)


def test_a_settings_file_that_holds_no_settings_is_refused(repo: Path, tmp_path: Path):
    config = org_config(tmp_path, "max_teams: many\n")

    with pytest.raises(InvalidOrgConfig):
        service_for(repo, config)


def test_a_kind_of_org_nobody_knows_is_refused(repo: Path):
    with pytest.raises(Invalid, match="no kind of org"):
        service_for(repo).update_settings(DirectorSettingsUpdate(org_kind="guild"))


def test_the_state_says_how_large_the_org_is(repo: Path):
    state = service_for(repo, max_concurrent_runs=4).settings_state()

    assert (state.teams, state.depth, state.max_concurrent_runs) == (5, 3, 4)


def test_an_open_session_follows_a_new_cost_limit_unless_it_was_let_past_it(
    repo: Path,
):
    settings = service_for(repo)
    store = settings.org()
    before = settings.settings.session_cost_limit
    open_session_in(store, "a", "follows")
    open_session_in(store, "b", "let-past")
    for session in store.sessions():
        allowance = before if session.id == "follows" else (before or 0) + 50
        store.set_session_use(session.id, session.usage, allowance)

    settings.update_settings(DirectorSettingsUpdate(session_cost_limit=7))

    allowances = {session.id: session.cost_allowance for session in store.sessions()}
    assert allowances == {"follows": 7, "let-past": (before or 0) + 50}


def test_what_an_org_sets_for_itself_is_kept_in_its_repository(
    repo: Path, tmp_path: Path
):
    config = org_config(tmp_path)
    service_for(repo, config).update_org_settings(
        OrgSettings(default_agent_type="analyst", allowed_agent_types=["analyst"])
    )

    again = service_for(repo, config)

    assert again.default_type() == "analyst"
    assert [kind.name for kind in again.types_allowed()] == ["analyst"]
    assert again.settings.default_agent_type != "analyst"


def test_an_org_cannot_allow_a_type_the_director_does_not_find(repo: Path):
    with pytest.raises(Invalid, match="astrologer"):
        service_for(repo).update_org_settings(
            OrgSettings(allowed_agent_types=["astrologer"])
        )


def test_the_default_type_must_be_among_the_allowed_ones(repo: Path, tmp_path: Path):
    settings = service_for(repo, org_config(tmp_path))

    with pytest.raises(Invalid, match="not among the allowed"):
        settings.update_org_settings(
            OrgSettings(
                default_agent_type="analyst", allowed_agent_types=["software-engineer"]
            )
        )


def test_there_is_nothing_to_set_for_an_org_that_does_not_exist(repo: Path):
    with pytest.raises(NotFound):
        service_for(repo, teams=False).update_org_settings(OrgSettings())


def test_a_kind_of_org_that_is_gone_is_read_as_the_default_kind(repo: Path):
    settings = service_for(repo)
    default = type(settings.org_kind)
    settings.remembered.keep(settings={"org_kind": "guild"})

    assert type(service_for(repo).org_kind) is default
