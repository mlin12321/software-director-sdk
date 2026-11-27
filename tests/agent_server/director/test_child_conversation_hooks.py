"""What the director added to the start of a child conversation: a chosen
id, tags, a start point, a directory, tools of the server, and an agent
that holds nothing of its parent's process."""

from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.conversation_service import (
    ConversationService,
    _to_derive_from,
)
from openhands.agent_server.models import StartChildConversationRequest
from openhands.sdk import LLM, Agent
from openhands.sdk.agent import ACPAgent
from openhands.sdk.conversation.request import ConversationWorktreeOptions
from openhands.sdk.git.utils import run_git_command
from openhands.sdk.security.confirmation_policy import AlwaysConfirm, NeverConfirm
from openhands.sdk.tool.spec import Tool
from tests.agent_server.test_parent_child_conversations import (
    _ChildLaunch,
    _event_service_factory,
    _init_git_repo,
    _start_request,
)


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


async def _launch_child(
    tmp_path: Path,
    workspace_dir: Path,
    request: StartChildConversationRequest,
    **launch_options: Any,
) -> _ChildLaunch:
    """Start a parent in *workspace_dir*, then launch a child from it."""
    services: dict[UUID, Any] = {}
    worktree_root = tmp_path / "conversation-worktrees"
    parent_request = _start_request(workspace_dir).model_copy(
        update={"tags": {"source": "test"}}
    )
    with patch(
        "openhands.agent_server.conversation_service.EventService",
        side_effect=_event_service_factory(services),
    ):
        async with ConversationService(
            conversations_dir=tmp_path / "conversations",
            conversation_worktree_root=worktree_root,
        ) as service:
            parent_info, _ = await service.start_conversation(parent_request)
            result = await service.start_child_conversation(
                parent_info.id, request, **launch_options
            )
            parent = await service.get_conversation(parent_info.id)
            child = (
                await service.get_conversation(result.conversation_id)
                if result
                else None
            )
    return _ChildLaunch(
        parent_id=parent_info.id,
        result=result,
        parent=parent,
        child=child,
        services=services,
        worktree_root=worktree_root,
    )


@pytest.mark.asyncio
async def test_start_child_conversation_takes_a_chosen_id(tmp_path, workspace_dir):
    child_id = uuid4()
    request = StartChildConversationRequest(task="Review the plan")

    launch = await _launch_child(
        tmp_path,
        workspace_dir,
        request,
        child_id=child_id,
    )

    assert launch.result is not None
    assert launch.result.conversation_id == child_id
    assert child_id in launch.services


@pytest.mark.asyncio
async def test_start_child_conversation_adds_tags_to_the_ones_it_inherits(
    tmp_path, workspace_dir
):
    launch = await _launch_child(
        tmp_path,
        workspace_dir,
        StartChildConversationRequest(task="Review the plan"),
        # One that is named with no value is not inherited.
        extra_tags={"startedby": "director", "source": None},
        confirmation_policy=AlwaysConfirm(),
    )

    assert launch.result is not None
    child_stored = launch.services[launch.result.conversation_id].stored
    assert child_stored.tags == {"startedby": "director"}
    assert child_stored.confirmation_policy == AlwaysConfirm()
    assert launch.services[launch.parent_id].stored.tags == {"source": "test"}
    assert launch.services[launch.parent_id].stored.confirmation_policy == (
        NeverConfirm()
    )


@pytest.mark.asyncio
async def test_start_child_conversation_refuses_an_id_that_is_taken(
    tmp_path, workspace_dir
):
    services: dict[UUID, Any] = {}
    with patch(
        "openhands.agent_server.conversation_service.EventService",
        side_effect=_event_service_factory(services),
    ):
        async with ConversationService(
            conversations_dir=tmp_path / "conversations"
        ) as service:
            parent_info, _ = await service.start_conversation(
                _start_request(workspace_dir)
            )

            with pytest.raises(ValueError, match="already exists"):
                await service.start_child_conversation(
                    parent_info.id,
                    StartChildConversationRequest(task="Again"),
                    child_id=parent_info.id,
                )


@pytest.mark.asyncio
async def test_start_child_conversation_worktree_can_start_from_the_parents_head(
    tmp_path,
):
    repo_dir = tmp_path / "repo"
    _init_git_repo(repo_dir)
    run_git_command(["git", "checkout", "-b", "feature"], repo_dir)
    (repo_dir / "feature.txt").write_text("in progress\n")
    run_git_command(["git", "add", "feature.txt"], repo_dir)
    run_git_command(
        [
            "git",
            "-c",
            "user.name=OpenHands Test",
            "-c",
            "user.email=openhands@example.com",
            "commit",
            "-m",
            "feature",
        ],
        repo_dir,
    )
    request = StartChildConversationRequest(task="Continue", isolation="worktree")

    launch = await _launch_child(
        tmp_path,
        repo_dir,
        request,
        worktree_options=ConversationWorktreeOptions(base="head"),
    )

    assert launch.result is not None and launch.result.workspace is not None
    assert (Path(launch.result.workspace) / "feature.txt").exists()


@pytest.mark.asyncio
async def test_start_conversation_gives_the_agent_server_tools(tmp_path, workspace_dir):
    services: dict[UUID, Any] = {}
    with patch(
        "openhands.agent_server.conversation_service.EventService",
        side_effect=_event_service_factory(services),
    ):
        async with ConversationService(
            conversations_dir=tmp_path / "conversations"
        ) as service:
            info, _ = await service.start_conversation(
                _start_request(workspace_dir),
                server_tools=[Tool(name="start_child_conversation")],
            )

    agent = services[info.id].get_conversation().agent
    assert [tool.name for tool in agent.tools] == ["start_child_conversation"]


@pytest.mark.asyncio
async def test_a_taken_child_id_leaves_the_worktree_of_its_conversation(tmp_path):
    repo_dir = tmp_path / "repo"
    _init_git_repo(repo_dir)
    request = StartChildConversationRequest(task="Fix the bug", isolation="worktree")
    services: dict[UUID, Any] = {}
    with patch(
        "openhands.agent_server.conversation_service.EventService",
        side_effect=_event_service_factory(services),
    ):
        async with ConversationService(
            conversations_dir=tmp_path / "conversations",
            conversation_worktree_root=tmp_path / "worktrees",
        ) as service:
            parent, _ = await service.start_conversation(_start_request(repo_dir))
            first = await service.start_child_conversation(parent.id, request)
            assert first is not None and first.workspace is not None
            worktree = Path(first.workspace)
            (worktree / "draft.txt").write_text("uncommitted\n")

            with pytest.raises(ValueError, match="already exists"):
                await service.start_child_conversation(
                    parent.id, request, child_id=first.conversation_id
                )

            assert (worktree / "draft.txt").read_text() == "uncommitted\n"


def test_a_child_of_a_running_acp_agent_holds_none_of_its_process():
    parent = ACPAgent(acp_command=["npx", "-y", "some-acp-agent"])
    parent._executor = object()
    parent._process = object()
    parent._session_id = "the-parents-session"

    child = _to_derive_from(parent.model_copy(update={"acp_args": ["--flag"]}))

    assert isinstance(child, ACPAgent)
    assert child.acp_command == parent.acp_command
    assert child.acp_args == ["--flag"]
    assert child._executor is None
    assert child._process is None
    assert child._session_id is None


def test_a_child_of_any_other_agent_starts_from_that_agent():
    agent = Agent(llm=LLM(model="gpt-4o", usage_id="test"), tools=[])

    assert _to_derive_from(agent) is agent
