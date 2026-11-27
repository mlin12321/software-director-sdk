"""Agent types: what a team's agents are told, from a YAML file, and the
org's settings from ``director.yaml``."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from openhands.agent_server.director.agent_types import (
    InvalidAgentType,
    load_types,
    read_type,
)
from openhands.agent_server.director.documents.declared import load_documents
from openhands.agent_server.director.models import (
    CreateOrgRequest,
    DirectorSettingsUpdate,
    OrgSettings,
    SplitChild,
)
from openhands.agent_server.director.org_config import InvalidOrgConfig
from openhands.agent_server.director.orgs import SHIPPED_DOCUMENTS, SHIPPED_TYPES
from openhands.agent_server.director.service import DirectorService
from openhands.agent_server.director.service.team_type_service import (
    DIRECTOR_TOOL_NAMES,
)
from openhands.agent_server.director.tools import (
    ApproveTeamsTool,
    ProposeTeamsTool,
    SubmitReviewTool,
)
from openhands.agent_server.director.tools.definition import SubmitReviewAction
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Invalid, Unprocessable
from tests.agent_server.director.conftest import (
    APPROVE,
    CODE_REVIEW,
    FakeLauncher,
    an_org,
    commit,
    ic,
    manager,
    members_yaml,
    node,
    org_config,
    proposing,
    root_conversation,
    split_into_typed_teams,
    tasked,
    use,
)


DOCUMENTS = load_documents([SHIPPED_DOCUMENTS])


@pytest.fixture
async def service(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
) -> AsyncIterator[DirectorService]:
    service = await an_org(repo, launcher, org_config(tmp_path))
    yield service
    await service.aclose()


def briefing(launcher: FakeLauncher, title: str) -> str:
    [start] = [start for start in launcher.derived if title in start.title]
    return start.task


def test_the_shipped_default_type_says_something_to_both_roles():
    types = load_types(SHIPPED_TYPES, DIRECTOR_TOOL_NAMES, DOCUMENTS)

    kind = types["software-engineer"]
    assert kind.text_for("ic").startswith('<TEAM_TYPE name="software-engineer">')
    assert "Split only by concern" in kind.text_for("ic")
    assert "one clear concern" in kind.text_for("manager")


@pytest.mark.parametrize(
    "body",
    [
        members_yaml() + "description: x\nactions:\n  no_such_tool: do it\n",
        members_yaml() + "description: x\nomit_tools: [no_such_tool]\n",
        members_yaml() + "description: x\nprimitives:\n  blackboard: shared\n",
        members_yaml() + "description: x\ngates:\n  merge: never\n",
        members_yaml() + "name: other\ndescription: x\n",
        "- not a mapping\n",
        members_yaml(ic="A type that does not say what it is for."),
        "description: A type that names no members.\n",
    ],
)
def test_a_type_that_says_what_no_type_may_is_refused(tmp_path: Path, body: str):
    path = tmp_path / "analyst.yaml"
    path.write_text(body)

    with pytest.raises(InvalidAgentType):
        read_type(path, DIRECTOR_TOOL_NAMES, DOCUMENTS)
    assert load_types([tmp_path], DIRECTOR_TOOL_NAMES, DOCUMENTS) == {}


async def test_a_team_is_told_what_its_own_type_says_and_not_its_parents(
    service: DirectorService, launcher: FakeLauncher
):
    context = service.context_of(node(service, "root"), "ic")
    assert context["types_a_sub_team_may_be"]["analyst"] == (
        "Reads results and says what they mean."
    )

    await split_into_typed_teams(service)

    tables, harness = briefing(launcher, "Tables"), briefing(launcher, "Harness")
    assert node(service, "tables").agent_type == "analyst"
    assert '<TEAM_TYPE name="analyst">' in tables
    assert "Your team analyses results." in tables
    assert "`submit_review`: Submit one table" in tables
    assert "software-engineer" not in tables.split("Task of the team:")[0]
    # A sub-team whose proposal names no type has the org's default type.
    assert node(service, "harness").agent_type == "software-engineer"
    assert '<TEAM_TYPE name="software-engineer">' in harness
    assert "analyst" not in harness.split("Task of the team:")[0]


async def test_a_type_takes_tools_away_but_not_the_ones_an_agent_needs(
    service: DirectorService, launcher: FakeLauncher
):
    await split_into_typed_teams(service)

    tables = node(service, "tables")
    offered = service.team_types.tools_for(tables, "ic")
    assert "propose_teams" not in offered
    assert "read_team_context" in offered and "submit_review" in offered
    harness = node(service, "harness")
    assert "propose_teams" in service.team_types.tools_for(harness, "ic")


async def test_a_team_cannot_be_of_a_type_the_director_does_not_find(
    service: DirectorService,
):
    answer = await use(
        service,
        ic(service),
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="A", path="a", task="Do a", agent_type="astrologer")]
        ),
    )

    assert answer.is_error
    assert "astrologer" in answer.text and "analyst" in answer.text
    assert node(service, "root").pending_split is None


async def test_work_is_written_up_and_reviewed_as_the_reviewed_teams_type_says(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await split_into_typed_teams(service)
    launcher.agent_answers = {"account of your tables": "One table: 0.97."}

    commit(repo, "tables/t1.md", "| warmup | 0.97 |\n")
    await use(
        service,
        ic(service, "tables"),
        SubmitReviewTool,
        SubmitReviewAction(summary="One table."),
    )
    commit(repo, "harness/run.sh", "echo run\n")
    await use(
        service,
        ic(service, "harness"),
        SubmitReviewTool,
        SubmitReviewAction(summary="Harness runs."),
    )
    launcher.report(str(ic(service)), ConversationExecutionStatus.RUNNING)
    await service.settled()

    asked = dict((str(who), text) for who, text in launcher.agents_asked)
    assert "account of your tables" in asked[str(ic(service, "tables"))]
    assert "pull request description" in asked[str(ic(service, "harness"))]
    both = "\n".join(
        text for text in launcher.messages_to(ic(service)) if "Review request" in text
    )
    tables = both[both.index("Team 'Tables'") :].split("Review request")[0]
    harness = both[both.index("Team 'Harness'") :].split("Review request")[0]
    assert "Every number in a table is in a committed result file." in tables
    assert CODE_REVIEW not in tables
    assert CODE_REVIEW in harness
    assert "Every number in a table" not in harness


async def test_the_top_team_is_of_the_type_the_request_or_the_settings_name(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    config = org_config(tmp_path, "default_agent_type: analyst\n")
    service = await an_org(repo, launcher, config)
    try:
        assert node(service, "root").agent_type == "analyst"
        kept = service.org.team("root")
        assert kept is not None and kept.agent_type == "analyst"
    finally:
        await service.aclose()


async def test_an_org_is_not_made_with_a_top_team_of_an_unknown_type(
    repo: Path, launcher: FakeLauncher
):
    service = DirectorService(repo, save_delay=0, run_limit_retry=0)
    await service.start(launcher)
    try:
        with pytest.raises(Unprocessable):
            await service.create_org(
                CreateOrgRequest(
                    task="Read the stars",
                    agent_type="astrologer",
                    conversation=root_conversation(repo),
                )
            )
        assert service.org.root_id is None
    finally:
        await service.aclose()


async def test_settings_come_from_director_yaml_and_a_change_holds_over_it(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    config = org_config(tmp_path, "soft_check_in_minutes: 30\nmax_depth: 2\n")
    service = await an_org(repo, launcher, config)
    try:
        assert service.settings.soft_check_in_minutes == 30
        assert service.settings.max_depth == 2
        # The org page sends every setting with the one that changed: only
        # that one holds over the file, also after a restart.
        service.update_settings(
            DirectorSettingsUpdate(soft_check_in_minutes=45, max_depth=2, max_teams=32)
        )
    finally:
        await service.aclose()
    config.write_text("soft_check_in_minutes: 30\nmax_depth: 3\nmax_teams: 5\n")
    again = DirectorService(
        repo, config_file=config, remembered=repo.parent / "director.json"
    )
    assert again.settings.soft_check_in_minutes == 45
    assert again.settings.max_depth == 3
    assert again.settings.max_teams == 5


@pytest.mark.parametrize(
    "body", ["stall_minutes: 30\n", "max_depth: 0\n", "- 30\n", "max_depth: [\n"]
)
def test_a_director_yaml_that_holds_no_settings_stops_the_server(
    repo: Path, tmp_path: Path, body: str
):
    config = tmp_path / "director.yaml"
    config.write_text(body)

    with pytest.raises(InvalidOrgConfig):
        DirectorService(repo, config_file=config)


async def test_an_org_sets_the_type_of_its_new_teams_and_which_they_may_be(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    state = service.org_settings_state()
    assert state.default_agent_type == "software-engineer"
    assert {kind.name for kind in state.agent_types if kind.allowed} >= {
        "analyst",
        "software-engineer",
    }

    changed = service.update_org_settings(
        OrgSettings(
            default_agent_type="analyst",
            allowed_agent_types=["analyst", "software-engineer"],
        )
    )
    assert changed.own.default_agent_type == "analyst"
    await use(
        service,
        ic(service),
        ProposeTeamsTool,
        proposing(
            [
                SplitChild(title="Tables", path="tables", task="Tabulate"),
                SplitChild(
                    title="Harness",
                    path="harness",
                    task="Build",
                    agent_type="software-engineer",
                ),
            ]
        ),
    )
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    # A team whose making names no type has the org's default one.
    assert node(service, "tables").agent_type == "analyst"
    assert node(service, "harness").agent_type == "software-engineer"

    # What is allowed holds for teams made from now on, not for the ones
    # there are: the team keeps its type and is still told what it says.
    service.update_org_settings(OrgSettings(allowed_agent_types=["analyst"]))
    assert node(service, "harness").agent_type == "software-engineer"
    kind = service.team_types.type_of(node(service, "harness"))
    assert kind.name == "software-engineer"
    refused = await use(
        service,
        ic(service, "tables"),
        ProposeTeamsTool,
        proposing(
            [
                SplitChild(
                    title="A",
                    path="tables/a",
                    task="Do a",
                    agent_type="software-engineer",
                )
            ]
        ),
    )
    assert refused.is_error and "software-engineer" in refused.text

    # The org keeps it in its own repository, not the server.
    await service.aclose()
    again = DirectorService(repo, save_delay=0, run_limit_retry=0)
    await again.start(launcher)
    try:
        assert again.own_settings.allowed_agent_types == ["analyst"]
        assert again.default_type() == "analyst"
    finally:
        await again.aclose()


async def test_an_org_cannot_set_a_type_there_is_not_or_a_default_not_allowed(
    service: DirectorService,
):
    with pytest.raises(Invalid, match="astrologer"):
        service.update_org_settings(OrgSettings(allowed_agent_types=["astrologer"]))
    with pytest.raises(Invalid, match="not among the allowed"):
        service.update_org_settings(
            OrgSettings(
                default_agent_type="analyst", allowed_agent_types=["software-engineer"]
            )
        )
    assert service.own_settings == OrgSettings()


async def test_a_type_added_while_the_server_runs_can_be_given_to_the_next_team(
    service: DirectorService, tmp_path: Path
):
    assert "auditor" not in service.team_types.agent_types

    (tmp_path / "config" / "types" / "auditor.yaml").write_text(
        members_yaml() + "description: Checks what other teams claim.\n"
    )

    assert "auditor" in {kind.name for kind in service.types_allowed()}
    service.update_org_settings(OrgSettings(default_agent_type="auditor"))
    assert service.default_type() == "auditor"


async def test_every_org_starts_with_what_director_yaml_says_of_types(
    repo: Path, launcher: FakeLauncher, tmp_path: Path
):
    config = org_config(
        tmp_path,
        "default_agent_type: analyst\nallowed_agent_types: [analyst]\n",
    )
    service = DirectorService(repo, save_delay=0, run_limit_retry=0, config_file=config)
    await service.start(launcher)
    try:
        with pytest.raises(Unprocessable, match="software-engineer"):
            await service.create_org(
                CreateOrgRequest(
                    task="Build it",
                    agent_type="software-engineer",
                    conversation=root_conversation(repo),
                )
            )
        await service.create_org(
            CreateOrgRequest(task="Tabulate", conversation=root_conversation(repo))
        )
        await service.settled()
        assert node(service, "root").agent_type == "analyst"
        state = service.org_settings_state()
        assert state.allowed_agent_types == ["analyst"]
        assert state.own == OrgSettings()
    finally:
        await service.aclose()


def a_type_on(folder: Path, body: str) -> None:
    (folder / "config" / "types" / "auditor.yaml").write_text(
        members_yaml() + "description: Checks what other teams claim.\n" + body
    )


async def test_a_type_can_name_the_profile_its_teams_run_on(
    service: DirectorService, launcher: FakeLauncher, tmp_path: Path
):
    launcher.profiles = {"checker": "openhands", "other": "openhands"}
    a_type_on(tmp_path, "sas_profile: checker\n")
    service.update_settings(DirectorSettingsUpdate(allowed_profiles=["other"]))
    context = service.context_of(node(service, "root"), "ic")
    assert "'checker'" in context["types_a_sub_team_may_be"]["auditor"]
    [info] = [
        kind
        for kind in service.org_settings_state().agent_types
        if kind.name == "auditor"
    ]
    assert info.sas_profile == "checker"

    await use(
        service,
        ic(service),
        ProposeTeamsTool,
        proposing(
            [
                # The type's profile, though no IC may choose it itself.
                SplitChild(title="A", path="a", task="Audit", agent_type="auditor"),
                # The proposal's own choice holds over the type's.
                SplitChild(
                    title="B",
                    path="b",
                    task="Audit",
                    agent_type="auditor",
                    agent_profile="other",
                ),
                # A type that names none says nothing of what runs it.
                SplitChild(title="C", path="c", task="Tabulate", agent_type="analyst"),
            ]
        ),
    )
    await use(service, manager(service), ApproveTeamsTool, APPROVE)
    await tasked(service)

    assert node(service, "a").agent_profile == "checker"
    assert node(service, "b").agent_profile == "other"
    assert node(service, "c").agent_profile is None


async def test_a_type_whose_profile_the_server_does_not_have_is_refused(
    service: DirectorService, launcher: FakeLauncher, tmp_path: Path
):
    a_type_on(tmp_path, "sas_profile: checker\n")

    answer = await use(
        service,
        ic(service),
        ProposeTeamsTool,
        proposing(
            [SplitChild(title="A", path="a", task="Audit", agent_type="auditor")]
        ),
    )

    assert answer.is_error
    assert "'checker'" in answer.text and "has none of that name" in answer.text
    assert node(service, "root").pending_split is None
