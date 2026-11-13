"""DirectorService running an org through a launcher."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest

from openhands.agent_server.director.models import CreateOrgRequest
from openhands.agent_server.director.org_persistence import ORG_REF
from openhands.agent_server.director.prompts import MANAGER_ROLE, POD_PROTOCOL
from openhands.agent_server.director.service import (
    AGENT_LOST,
    DIRECTOR_TOOL_NAMES,
    DirectorService,
    agent_conversation,
)
from openhands.sdk.conversation.request import AgentLaunchAdditions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict, NotFound, Unavailable, Unprocessable
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_node,
    root_conversation,
    run_git,
)


@pytest.fixture
def launcher() -> FakeLauncher:
    return FakeLauncher()


@pytest.fixture
async def service(repo: Path, launcher: FakeLauncher) -> AsyncIterator[DirectorService]:
    service = DirectorService(repo, save_delay=0, run_limit_retry=0)
    await service.start(launcher)
    yield service
    await service.aclose()


def org_request(repo: Path, **conversation) -> CreateOrgRequest:
    return CreateOrgRequest(
        title="Checkout",
        task="Build the checkout flow\nwith tests",
        conversation=root_conversation(repo).model_copy(update=conversation),
    )


async def test_starting_an_org_runs_a_root_team_briefed_with_the_task(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    root = await service.create_org(org_request(repo))
    await service.settled()

    assert (root.id, root.parent_id, root.title) == ("root", None, "Checkout")
    assert root.boundary.path == ""
    assert root.spec_summary == "Build the checkout flow"
    ic, manager = agent_conversation(root, "ic"), agent_conversation(root, "manager")
    assert ic is not None and manager is not None and ic != manager

    assert launcher.root is not None
    assert str(launcher.root.conversation_id) == ic
    assert launcher.root_tools == list(DIRECTOR_TOOL_NAMES)
    [briefing] = launcher.messages_to(ic)
    assert "IC" in briefing and "Build the checkout flow\nwith tests" in briefing

    [manager_start] = launcher.derived
    assert (str(manager_start.parent_id), str(manager_start.conversation_id)) == (
        ic,
        manager,
    )
    assert manager_start.isolation == "shared"
    assert manager_start.suffixes == [MANAGER_ROLE]
    assert "Manager" in manager_start.task


async def test_the_org_protocol_is_added_to_what_the_caller_appends(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await service.create_org(
        org_request(
            repo,
            agent_launch_additions=AgentLaunchAdditions(
                system_message_suffix_append="Prefer small commits."
            ),
        )
    )

    assert launcher.root is not None and launcher.root.agent_launch_additions
    suffix = launcher.root.agent_launch_additions.system_message_suffix_append
    assert suffix == f"Prefer small commits.\n\n{POD_PROTOCOL}"


async def test_an_agent_is_briefed_with_the_rules_taught_for_its_boundary(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    service.blackboard.teach("", "Never log card numbers")

    root = await service.create_org(org_request(repo))
    await service.settled()

    [briefing] = launcher.messages_to(agent_conversation(root, "ic"))
    assert "Never log card numbers" in briefing


async def test_a_director_has_one_org(service: DirectorService, repo: Path):
    await service.create_org(org_request(repo))

    with pytest.raises(Conflict):
        await service.create_org(org_request(repo))


async def test_an_org_must_work_inside_the_directors_repository(
    service: DirectorService, tmp_path: Path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(Unprocessable):
        await service.create_org(org_request(elsewhere))

    assert service.store.root_id is None


async def test_an_org_cannot_run_in_docker(service: DirectorService, repo: Path):
    with pytest.raises(Unprocessable):
        await service.create_org(org_request(repo, conversation_runtime="docker"))


async def test_an_org_needs_a_git_repository(tmp_path: Path, launcher: FakeLauncher):
    service = DirectorService(tmp_path)
    await service.start(launcher)
    try:
        with pytest.raises(Unavailable):
            await service.create_org(org_request(tmp_path))
    finally:
        await service.aclose()


async def test_the_fixture_tree_cannot_be_replaced_by_an_org(
    repo: Path, launcher: FakeLauncher
):
    service = DirectorService(repo, fixture=True)
    await service.start(launcher)
    try:
        with pytest.raises(Conflict):
            await service.create_org(org_request(repo))
    finally:
        await service.aclose()


async def test_an_org_cannot_be_started_before_the_director_runs(repo: Path):
    with pytest.raises(Unavailable):
        await DirectorService(repo).create_org(org_request(repo))


async def test_no_org_is_left_when_its_root_agent_cannot_start(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.root_error = Unprocessable("An org cannot be led by an ACP agent")

    with pytest.raises(Unprocessable):
        await service.create_org(org_request(repo))

    assert service.store.root_id is None
    assert service.summary().node_count == 0


async def test_a_node_works_while_an_agent_runs_and_is_blocked_by_a_failed_one(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    root = await service.create_org(org_request(repo))
    await service.settled()
    ic = agent_conversation(root, "ic")
    assert ic is not None

    def state() -> tuple[str, list[str]]:
        node = service.store.get("root")
        assert node is not None
        return node.status, [agent.status for agent in node.agents]

    launcher.report(ic, ConversationExecutionStatus.RUNNING)
    assert state() == ("working", ["working", "idle"])
    launcher.report(ic, ConversationExecutionStatus.FINISHED)
    assert state() == ("idle", ["idle", "idle"])
    launcher.report(ic, ConversationExecutionStatus.ERROR)
    assert state() == ("blocked", ["blocked", "idle"])


async def test_the_manager_starts_once_the_server_has_room(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.full_for = 2

    root = await service.create_org(org_request(repo))
    await service.settled()

    assert [str(start.conversation_id) for start in launcher.derived] == [
        agent_conversation(root, "manager")
    ]


async def test_the_org_is_saved_and_comes_back_with_its_conversations(
    repo: Path, launcher: FakeLauncher
):
    first = DirectorService(repo, save_delay=0)
    await first.start(launcher)
    root = await first.create_org(org_request(repo))
    await first.settled()
    await first.aclose()
    ic, manager = agent_conversation(root, "ic"), agent_conversation(root, "manager")
    assert ic is not None and manager is not None
    assert run_git(repo, "status", "--porcelain") == ""
    assert run_git(repo, "rev-parse", ORG_REF).strip()

    # The Manager's conversation was deleted while the server was down.
    del launcher.known[UUID(manager)]
    launcher.known[UUID(ic)] = ConversationExecutionStatus.RUNNING
    second = DirectorService(repo, save_delay=0)
    await second.start(launcher)
    try:
        loaded = second.store.get("root")
        await second.settled()
        resynced = second.store.get("root")
    finally:
        await second.aclose()

    assert loaded is not None and resynced is not None
    assert agent_conversation(loaded, "ic") == ic
    assert [agent.status for agent in resynced.agents] == ["working", AGENT_LOST]
    assert resynced.status == "working"


async def test_the_humans_answer_reaches_the_team_that_raised_the_escalation(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    root = await service.create_org(org_request(repo))
    service.store.raise_escalation(
        "root", "jwt", kind="design_decision", title="JWT or opaque tokens?"
    )

    node = service.answer_escalation("root", "jwt", "Opaque tokens.")
    await service.settled()

    assert node.escalations == []
    assert service.store.escalations("root") == []
    answer = launcher.messages_to(agent_conversation(root, "ic"))[-1]
    assert "JWT or opaque tokens?" in answer and "Opaque tokens." in answer


async def test_an_escalation_a_team_still_holds_cannot_be_answered_by_the_human(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    service.store.upsert(make_node("api", "root", path="api"))
    service.store.raise_escalation(
        "api", "schema", kind="design_decision", title="Which schema?"
    )

    with pytest.raises(Conflict):
        service.answer_escalation("api", "schema", "v2")
    with pytest.raises(NotFound):
        service.answer_escalation("api", "nothing", "v2")

    api = service.store.get("api")
    assert api is not None and len(api.escalations) == 1
