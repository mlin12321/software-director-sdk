"""DirectorService running an org through a launcher."""

import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.director import pod_actions
from openhands.agent_server.director.headline import TeamSummary
from openhands.agent_server.director.models import (
    AgentRole,
    CreateOrgRequest,
    OrgNode,
    RouteMessageRequest,
    SplitChild,
)
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
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool.tool import Action
from openhands.tools.apply_patch.definition import ApplyPatchAction
from openhands.tools.file_editor.definition import FileEditorAction
from openhands.tools.terminal.definition import TerminalAction
from tests.agent_server.director.conftest import (
    FakeLauncher,
    make_node,
    root_conversation,
    run_git,
    teams_block,
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
    assert (launcher.root_title, launcher.root.autotitle) == ("Checkout (IC)", False)
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


async def test_a_team_started_again_is_briefed_with_what_it_reported(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    root = await service.create_org(org_request(repo))
    await service.settled()
    service.report(root, "manager", "The API team needs a schema first.")

    briefing, _ = service._briefing(root, "ic", "Build")

    assert "Manager, " in briefing
    assert "The API team needs a schema first." in briefing
    [first] = launcher.messages_to(agent_conversation(root, "ic"))
    assert "reported so far" not in first


async def test_a_director_has_one_org(service: DirectorService, repo: Path):
    await service.create_org(org_request(repo))

    with pytest.raises(Conflict):
        await service.create_org(org_request(repo))


def other_repository(tmp_path: Path) -> Path:
    """A git repository that is not the one the server was started for."""
    other = tmp_path / "other"
    (other / "src").mkdir(parents=True)
    run_git(other, "init", "--quiet", "--initial-branch=main")
    run_git(other, "config", "user.name", "Test")
    run_git(other, "config", "user.email", "test@example.invalid")
    run_git(other, "commit", "--quiet", "--allow-empty", "-m", "init")
    return other


async def test_an_org_lives_in_the_repository_its_first_conversation_is_opened_in(
    repo: Path, tmp_path: Path, launcher: FakeLauncher
):
    other = other_repository(tmp_path)
    remembered = tmp_path / "state" / "director.json"
    first = DirectorService(repo, save_delay=0, remembered=remembered)
    await first.start(launcher)
    before = first.summary().repo_path
    routed = await first.route_message(
        RouteMessageRequest(
            text="Build it", conversation=root_conversation(other / "src")
        )
    )
    await first.settled()
    await first.aclose()

    assert (before, first.summary().repo_path) == (str(repo), str(other))
    assert routed.started_org
    assert run_git(other, "rev-parse", ORG_REF).strip()
    with pytest.raises(subprocess.CalledProcessError):
        run_git(repo, "rev-parse", "--verify", "--quiet", ORG_REF)

    # The server starts again where it was started before.
    second = DirectorService(repo, save_delay=0, remembered=remembered)
    await second.start(launcher)
    try:
        again = await second.route_message(
            RouteMessageRequest(text="And tests", path=str(other / "src"))
        )
        await second.settled()
    finally:
        await second.aclose()

    assert second.summary().repo_path == str(other)
    assert again.conversation_id == routed.conversation_id
    assert launcher.messages_to(again.conversation_id)[-1] == (
        "And tests\n\n(The human sent this from src.)"
    )


async def test_a_first_message_takes_up_the_org_saved_where_it_is_sent_from(
    repo: Path, tmp_path: Path, launcher: FakeLauncher
):
    other = other_repository(tmp_path)
    there = DirectorService(other, save_delay=0)
    await there.start(launcher)
    root = await there.create_org(org_request(other))
    await there.settled()
    await there.aclose()

    service = DirectorService(repo, save_delay=0)
    await service.start(launcher)
    try:
        routed = await service.route_message(
            RouteMessageRequest(
                text="Add refunds", conversation=root_conversation(other)
            )
        )
        await service.settled()
    finally:
        await service.aclose()

    assert not routed.started_org
    assert routed.conversation_id == agent_conversation(root, "ic")
    assert launcher.messages_to(routed.conversation_id)[-1] == "Add refunds"
    assert launcher.root is not None
    assert launcher.root.conversation_id == UUID(routed.conversation_id)


async def test_an_org_is_not_started_outside_a_git_repository(
    service: DirectorService, repo: Path, tmp_path: Path
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(Unavailable, match="elsewhere"):
        await service.create_org(org_request(elsewhere))

    assert service.store.root_id is None
    assert service.summary().repo_path == str(repo)


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


async def split_root(service: DirectorService, *children: SplitChild) -> None:
    """Have the root IC propose ``children`` and its Manager approve them."""
    root = service.store.get("root")
    assert root is not None

    def caller(role: AgentRole) -> pod_actions.Caller:
        conversation_id = agent_conversation(root, role)
        assert conversation_id is not None
        return pod_actions.caller_of(service, UUID(conversation_id))

    pod_actions.msg_manager(service, caller("ic"), teams_block(list(children)))
    pod_actions.approve_teams(service, caller("manager"))
    await service.settled()


API = SplitChild(title="API", path="services/api", task="Serve the checkout API")
WEB = SplitChild(title="Web", path="apps/web", task="Build the checkout pages")


def team(service: DirectorService, node_id: str) -> OrgNode:
    node = service.store.get(node_id)
    assert node is not None
    return node


async def test_an_org_without_a_title_is_named_by_a_model_from_its_task(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.summary = TeamSummary("Checkout Flow", "Builds the shop's checkout.")
    request = org_request(repo).model_copy(update={"title": None})

    root = await service.create_org(request)

    assert (root.title, root.headline) == (
        "Build the checkout flow",
        "Build the checkout flow",
    )
    assert launcher.root_title == "Build the checkout flow (IC)"

    await service.settled()

    root = team(service, "root")
    assert (root.title, root.headline) == (
        "Checkout Flow",
        "Builds the shop's checkout.",
    )
    ic, manager = agent_conversation(root, "ic"), agent_conversation(root, "manager")
    assert ic is not None and manager is not None
    assert launcher.summarized == [(UUID(ic), "Build the checkout flow\nwith tests")]
    assert launcher.titles[UUID(ic)] == "Checkout Flow (IC)"
    assert launcher.titles[UUID(manager)] == "Checkout Flow (Manager)"
    [briefing] = launcher.messages_to(ic)
    assert "the team 'Checkout Flow'" in briefing


async def test_a_model_that_takes_too_long_does_not_hold_the_org_up(
    repo: Path, launcher: FakeLauncher
):
    service = DirectorService(repo, save_delay=0, run_limit_retry=0, summary_timeout=0)
    launcher.summary_delay = 60
    await service.start(launcher)

    await service.create_org(org_request(repo).model_copy(update={"title": None}))
    await service.settled()
    root = service.store.get("root")
    await service.aclose()

    assert root is not None and root.title == "Build the checkout flow"
    assert len(launcher.messages_to(agent_conversation(root, "ic"))) == 1


async def test_a_given_title_is_kept_and_only_the_headline_is_written(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.summary = TeamSummary("Checkout Flow", "Builds the shop's checkout.")

    await service.create_org(org_request(repo))
    await service.settled()

    root = team(service, "root")
    assert (root.title, root.headline) == ("Checkout", "Builds the shop's checkout.")
    assert launcher.titles[UUID(agent_conversation(root, "ic") or "")] == (
        "Checkout (IC)"
    )


async def test_a_team_keeps_its_first_name_when_the_model_gives_nothing(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo).model_copy(update={"title": None}))
    await service.settled()

    root = team(service, "root")
    assert (root.title, root.headline) == (
        "Build the checkout flow",
        "Build the checkout flow",
    )


async def test_every_sub_team_gets_a_headline_and_keeps_the_title_its_ic_gave(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    launcher.summary = TeamSummary("Something Else", "Serves the checkout over HTTP.")

    await split_root(service, API, WEB)

    api = team(service, "services.api")
    assert (api.title, api.headline) == ("API", "Serves the checkout over HTTP.")
    assert {task for _, task in launcher.summarized[-2:]} == {API.task, WEB.task}


async def test_a_report_made_before_the_first_summary_is_not_overwritten_by_it(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    launcher.summary = TeamSummary("API", "Blocked on the session store.")

    def report_at_once(conversation_id: UUID) -> None:
        found = service.store.find_by_conversation(str(conversation_id))
        if found is not None and found[1] == "ic":
            service.refresh_headline(found[0].id, "it is blocked.")

    launcher.on_derived = report_at_once
    await split_root(service, API)

    assert team(service, "services.api").headline == "Blocked on the session store."
    assert [task for _, task in launcher.summarized[1:]] == [
        "Serve the checkout API\n\nLatest report of the team: it is blocked."
    ]


async def test_a_report_rewrites_the_headline(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    launcher.summary = TeamSummary("Ignored", "Checkout is done; tax is left out.")

    service.refresh_headline("root", "it is done. Tax is left out.")
    await service.settled()

    assert team(service, "root").headline == "Checkout is done; tax is left out."
    assert launcher.summarized[-1][1] == (
        "Build the checkout flow\n\nLatest report of the team: it is done. "
        "Tax is left out."
    )


def attempt(
    service: DirectorService, conversation_id: str | None, tool: str, action: Action
) -> str | None:
    """What the director says to an agent that is about to use ``tool``."""
    event = cast(ActionEvent, SimpleNamespace(tool_name=tool, action=action))
    return service.guard_action(UUID(conversation_id or ""), event)


def create(path: Path | str) -> FileEditorAction:
    return FileEditorAction(command="create", path=str(path), file_text="x")


async def test_a_file_tool_is_refused_a_write_outside_the_teams_boundary(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API, WEB)
    api_ic = agent_conversation(team(service, "services.api"), "ic")

    def refusal(path: Path | str) -> str | None:
        return attempt(service, api_ic, "file_editor", create(path))

    assert refusal(repo / "services" / "api" / "pay.py") is None
    assert "outside your team's boundary" in (refusal(repo / "README.md") or "")
    assert "outside your team's boundary" in (
        refusal(repo / "apps" / "web" / "page.tsx") or ""
    )
    assert "outside your team's boundary" in (
        refusal(repo / "services" / "api" / ".." / ".." / "README.md") or ""
    )


async def test_a_link_out_of_the_boundary_does_not_carry_a_write_out(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "up").symlink_to(repo)
    api_ic = agent_conversation(team(service, "services.api"), "ic")

    refused = attempt(
        service,
        api_ic,
        "file_editor",
        create(repo / "services" / "api" / "up" / "README.md"),
    )

    assert "outside your team's boundary" in (refused or "")


async def test_reading_and_tools_that_name_no_file_are_never_refused(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    api_ic = agent_conversation(team(service, "services.api"), "ic")

    view = FileEditorAction(command="view", path=str(repo / "README.md"))
    assert attempt(service, api_ic, "file_editor", view) is None
    command = TerminalAction(command="cat ../../README.md")
    assert attempt(service, api_ic, "terminal", command) is None


async def test_an_ic_is_refused_what_it_handed_to_a_sub_team_and_a_manager_all(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    root = team(service, "root")
    root_ic = agent_conversation(root, "ic")
    inside_api = create(repo / "services" / "api" / "pay.py")
    assert attempt(service, root_ic, "file_editor", inside_api) is None

    await split_root(service, API)

    assert "sub-team 'API'" in (
        attempt(service, root_ic, "file_editor", inside_api) or ""
    )
    assert attempt(service, root_ic, "file_editor", create(repo / "README.md")) is None
    manager = agent_conversation(root, "manager")
    assert "Manager" in (
        attempt(service, manager, "file_editor", create(repo / "README.md")) or ""
    )


async def test_every_file_of_a_patch_must_be_in_the_boundary(
    service: DirectorService, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    api_ic = agent_conversation(team(service, "services.api"), "ic")

    def patch(*files: str) -> ApplyPatchAction:
        body = "".join(f"*** Add File: {name}\n+x\n" for name in files)
        return ApplyPatchAction(patch=f"*** Begin Patch\n{body}*** End Patch\n")

    assert attempt(service, api_ic, "apply_patch", patch("services/api/a.py")) is None
    refused = attempt(
        service, api_ic, "apply_patch", patch("services/api/a.py", "docs/b.md")
    )
    assert "'docs/b.md' is outside your team's boundary" in (refused or "")


async def test_files_outside_the_org_and_other_conversations_are_left_alone(
    service: DirectorService, repo: Path, tmp_path: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    api_ic = agent_conversation(team(service, "services.api"), "ic")

    assert attempt(service, api_ic, "file_editor", create(tmp_path / "x.txt")) is None
    stranger = str(uuid4())
    assert attempt(service, stranger, "file_editor", create(repo / "README.md")) is None


async def test_a_checkout_of_another_team_is_closed(
    repo: Path, tmp_path: Path, launcher: FakeLauncher
):
    worktrees = tmp_path / "worktrees"
    (worktrees / "other" / "repo").mkdir(parents=True)
    service = DirectorService(
        repo, save_delay=0, run_limit_retry=0, worktree_root=worktrees
    )
    await service.start(launcher)
    await service.create_org(org_request(repo))
    await service.settled()
    root_ic = agent_conversation(team(service, "root"), "ic")

    refused = attempt(
        service,
        root_ic,
        "file_editor",
        create(worktrees / "other" / "repo" / "README.md"),
    )
    await service.aclose()

    assert "not your team's" in (refused or "")


async def test_the_first_message_starts_the_org_when_there_is_none(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    launcher.summary = TeamSummary("Checkout Flow", "Builds the shop's checkout.")

    routed = await service.route_message(
        RouteMessageRequest(
            text="Build the checkout flow", conversation=root_conversation(repo)
        )
    )
    await service.settled()

    root = team(service, "root")
    assert (routed.node_id, routed.started_org) == ("root", True)
    assert routed.conversation_id == agent_conversation(root, "ic")
    assert root.title == "Checkout Flow"
    [briefing] = launcher.messages_to(routed.conversation_id)
    assert "Build the checkout flow" in briefing


async def test_the_first_message_cannot_start_an_org_without_a_conversation(
    service: DirectorService,
):
    with pytest.raises(Unprocessable, match="no org yet"):
        await service.route_message(RouteMessageRequest(text="Build it"))

    assert service.store.root_id is None


async def test_a_message_goes_to_the_nearest_team_above_where_it_was_sent(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API, WEB)

    async def send(path: str | None) -> tuple[str, bool]:
        routed = await service.route_message(
            RouteMessageRequest(text="Add a refund endpoint", path=path)
        )
        await service.settled()
        assert launcher.messages_to(routed.conversation_id)[-1].startswith(
            "Add a refund endpoint"
        )
        assert routed.conversation_id == agent_conversation(
            team(service, routed.node_id), "ic"
        )
        return routed.node_id, routed.started_org

    assert await send("services/api/handlers") == ("services.api", False)
    assert await send(str(repo / "services" / "api")) == ("services.api", False)
    assert await send("services/billing") == ("root", False)
    assert await send(None) == ("root", False)

    api_ic = agent_conversation(team(service, "services.api"), "ic")
    assert launcher.messages_to(api_ic)[-2:] == [
        "Add a refund endpoint\n\n(The human sent this from services/api/handlers.)",
        "Add a refund endpoint",
    ]


async def test_a_message_passes_over_a_team_whose_ic_is_gone(
    service: DirectorService, launcher: FakeLauncher, repo: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    service.store.set_agent("services.api", "ic", status=AGENT_LOST)

    routed = await service.route_message(
        RouteMessageRequest(text="Add a refund endpoint", path="services/api")
    )

    assert routed.node_id == "root"


@pytest.mark.parametrize("path", ["../elsewhere", "/etc"])
async def test_a_message_from_outside_the_repository_is_refused(
    service: DirectorService, repo: Path, path: str
):
    await service.create_org(org_request(repo))
    await service.settled()

    with pytest.raises(Unprocessable):
        await service.route_message(RouteMessageRequest(text="Hello", path=path))


async def test_a_server_in_a_submodule_joins_the_org_of_the_repository_above(
    repo: Path, submodule: Path, launcher: FakeLauncher
):
    above = DirectorService(repo, save_delay=0)
    await above.start(launcher)
    await above.create_org(org_request(repo))
    await above.settled()
    await split_root(above, SplitChild(title="Libraries", path="libs", task="Libs"))
    await above.aclose()

    service = DirectorService(submodule, save_delay=0)
    await service.start(launcher)
    try:
        summary = service.summary()
        from_here = await service.route_message(RouteMessageRequest(text="Refunds"))
        by_path = await service.route_message(
            RouteMessageRequest(text="Refunds", path=str(submodule / "src"))
        )
        further_up = await service.route_message(
            RouteMessageRequest(text="Refunds", path="docs")
        )
        await service.settled()
    finally:
        await service.aclose()

    assert (summary.repo_path, summary.directory) == (str(repo), "libs/payments")
    assert (from_here.node_id, by_path.node_id, further_up.node_id) == (
        "libs",
        "libs",
        "root",
    )
    assert launcher.messages_to(from_here.conversation_id) == [
        "Refunds\n\n(The human sent this from libs/payments.)",
        "Refunds\n\n(The human sent this from libs/payments/src.)",
    ]
    with pytest.raises(subprocess.CalledProcessError):
        run_git(submodule, "rev-parse", "--verify", "--quiet", ORG_REF)


async def test_a_submodule_with_an_org_of_its_own_keeps_it(
    repo: Path, submodule: Path, launcher: FakeLauncher
):
    for directory in (submodule, repo):
        service = DirectorService(directory, save_delay=0)
        await service.start(launcher)
        await service.create_org(org_request(directory))
        await service.settled()
        await service.aclose()

    service = DirectorService(submodule, save_delay=0)

    assert (service.summary().repo_path, service.summary().directory) == (
        str(submodule),
        "",
    )
    assert team(service, "root").worktree_path == str(submodule)


async def test_the_first_message_in_a_submodule_starts_its_org_there_when_none_is_above(
    repo: Path, submodule: Path, launcher: FakeLauncher
):
    service = DirectorService(submodule, save_delay=0)
    await service.start(launcher)
    try:
        routed = await service.route_message(
            RouteMessageRequest(
                text="Add refunds", conversation=root_conversation(submodule)
            )
        )
        await service.settled()
    finally:
        await service.aclose()

    assert routed.started_org
    assert service.summary().repo_path == str(submodule)
    assert run_git(submodule, "rev-parse", ORG_REF).strip()
    with pytest.raises(subprocess.CalledProcessError):
        run_git(repo, "rev-parse", "--verify", "--quiet", ORG_REF)


@pytest.fixture
async def confining(
    repo: Path, launcher: FakeLauncher
) -> AsyncIterator[DirectorService]:
    """A director on a server that can confine shells."""
    service = DirectorService(
        repo, save_delay=0, run_limit_retry=0, find_sandbox=lambda: "/usr/bin/bwrap"
    )
    await service.start(launcher)
    yield service
    await service.aclose()


def shell_of(params: object) -> str:
    """The script a started agent's terminal runs."""
    assert isinstance(params, dict)
    terminal = params["terminal"]
    assert terminal["terminal_type"] == "subprocess"
    return Path(terminal["shell_path"]).read_text()


async def test_every_agent_starts_its_terminal_in_a_shell_held_to_its_boundary(
    confining: DirectorService, launcher: FakeLauncher, repo: Path
):
    assert confining.summary().boundary_enforcement == "sandbox"
    await confining.create_org(org_request(repo))
    await confining.settled()
    await split_root(confining, API)

    root_shell = shell_of(launcher.root_tool_params)
    manager, api_ic, api_manager = launcher.derived
    assert 'own="$top"\n' in shell_of(launcher.root_tool_params)
    assert "own=" not in shell_of(manager.tool_params)
    assert 'own="$top"/services/api\n' in shell_of(api_ic.tool_params)
    assert "own=" not in shell_of(api_manager.tool_params)
    # The root's next shell no longer writes what it gave away.
    assert '--ro-bind-try "$top"/services/api' in root_shell
    script = Path(cast(dict, api_ic.tool_params)["terminal"]["shell_path"])
    assert script.parent == repo / ".git" / "director-shells"
    assert script.name == f"{api_ic.conversation_id}.sh"


def run_in_shell(params: object, command: str, home: Path) -> str:
    """Run ``command`` in the shell a started agent's terminal runs."""
    assert isinstance(params, dict)
    home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(home),
        "GIT_AUTHOR_NAME": "Agent",
        "GIT_AUTHOR_EMAIL": "agent@example.invalid",
        "GIT_COMMITTER_NAME": "Agent",
        "GIT_COMMITTER_EMAIL": "agent@example.invalid",
    }
    done = subprocess.run(
        [params["terminal"]["shell_path"], "-ic", command],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.skipif(os.name == "nt", reason="the agents' git is a sh script")
async def test_an_ic_commits_from_its_own_index_where_shells_are_not_confined(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    assert service.summary().boundary_enforcement == "checks"
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    manager, api_ic, _ = launcher.derived
    (repo / "README.md").write_text("root v2\n")
    (repo / "services" / "api").mkdir(parents=True)
    (repo / "services" / "api" / "app.py").write_text("api\n")

    run_in_shell(
        launcher.root_tool_params,
        f"cd {repo} && git add -A && git commit -qm root",
        tmp_path / "home",
    )
    run_in_shell(
        api_ic.tool_params,
        f"cd {repo} && git add -A && git commit -qm api",
        tmp_path / "home",
    )

    def files(commit: str) -> list[str]:
        return run_git(repo, "show", "--name-only", "--format=", commit).split()

    assert files("HEAD~1") == ["README.md"]
    assert files("HEAD") == ["services/api/app.py"]
    assert run_git(repo, "status", "--porcelain") == ""
    # A Manager changes nothing, so it has no shell of its own here.
    assert manager.tool_params is None


@pytest.mark.skipif(os.name == "nt", reason="the agents' git is a sh script")
async def test_a_pruned_agents_shell_and_index_are_removed(
    service: DirectorService, launcher: FakeLauncher, repo: Path, tmp_path: Path
):
    await service.create_org(org_request(repo))
    await service.settled()
    await split_root(service, API)
    api_ic = launcher.derived[1]
    run_in_shell(api_ic.tool_params, f"cd {repo} && git status", tmp_path / "home")
    left = [
        *(repo / ".git" / "director-shells").glob(f"{api_ic.conversation_id}.*"),
        *(repo / ".git").glob(f"agent-index-{api_ic.conversation_id}*"),
    ]
    # The shell, its rc file, its git, its excludes, the launcher an ACP
    # agent would be started by, and the index with its marker.
    assert len(left) == 7

    service.prune("services.api", "x")
    await service.settled()

    assert not [path for path in left if path.exists()]
