"""What the director added to OpenHands' own conversation and event
services: status listeners, the action guard, resume adapters, and a
worktree started from a commit that is already here."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import (
    ConversationService,
    _worktree_start_point,
)
from openhands.agent_server.event_service import EventService
from openhands.agent_server.models import StartConversationRequest, StoredConversation
from openhands.sdk import LLM, Agent
from openhands.sdk.conversation.request import ConversationWorktreeOptions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import ActionEvent
from openhands.sdk.git.utils import run_git_command
from openhands.sdk.security.confirmation_policy import NeverConfirm
from openhands.sdk.workspace import LocalWorkspace
from tests.agent_server.test_conversation_service import _init_git_repo


@pytest.mark.asyncio
async def test_a_resume_adapter_changes_the_saved_agent_before_it_starts(tmp_path):
    conversations_dir = tmp_path / "conversations"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    request = StartConversationRequest(
        agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace_dir)),
        confirmation_policy=NeverConfirm(),
    )
    async with ConversationService(conversations_dir=conversations_dir) as service:
        info, _ = await service.start_conversation(request)
        conv_id = info.id

    seen = []

    def adapt(conversation_id, agent):
        seen.append((conversation_id, agent.llm.model))
        return agent.model_copy(
            update={"llm": agent.llm.model_copy(update={"model": "gpt-4o-mini"})}
        )

    async with ConversationService(conversations_dir=conversations_dir) as service2:
        stop = service2.add_resume_adapter(adapt)
        assert await service2.resume_conversation(conv_id)
        event_service = await service2.get_event_service(conv_id)
        assert event_service is not None
        state = await event_service.get_state()

        assert seen == [(conv_id, "gpt-4o")]
        assert state.agent.llm.model == "gpt-4o-mini"
        assert event_service.get_conversation().agent.llm.model == "gpt-4o-mini"
        stop()
        assert service2._resume_adapters == []


def test_a_worktree_from_a_commit_here_fetches_nothing(tmp_path):
    """A commit id cannot move: the remote is not asked for it."""
    repo_dir = tmp_path / "repo"
    _init_git_repo(repo_dir)
    run_git_command(
        ["git", "remote", "add", "origin", str(tmp_path / "unreachable")],
        repo_dir,
    )
    head = run_git_command(["git", "rev-parse", "HEAD"], repo_dir).strip()
    fetched = []

    with patch(
        "openhands.agent_server.conversation_service._fetch_origin",
        side_effect=fetched.append,
    ):
        assert (
            _worktree_start_point(
                repo_dir, ConversationWorktreeOptions(base="ref", ref=head)
            )
            == head
        )
        _worktree_start_point(
            repo_dir, ConversationWorktreeOptions(base="ref", ref="main")
        )

    assert fetched == [repo_dir]


def _plain_request(workspace_dir: Path) -> StartConversationRequest:
    return StartConversationRequest(
        agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        workspace=LocalWorkspace(working_dir=str(workspace_dir)),
    )


@pytest.mark.asyncio
async def test_execution_status_listeners_hear_every_conversation_until_stopped(
    tmp_path,
):
    heard: list[tuple[UUID, ConversationExecutionStatus]] = []
    async with ConversationService(
        conversations_dir=tmp_path / "conversations"
    ) as service:
        stop = service.add_execution_status_listener(
            lambda conversation_id, status: heard.append((conversation_id, status))
        )
        first, _ = await service.start_conversation(_plain_request(tmp_path))
        second, _ = await service.start_conversation(_plain_request(tmp_path))
        await service.pause_conversation(second.id)
        stop()
        await service.pause_conversation(first.id)

    assert (first.id, ConversationExecutionStatus.IDLE) in heard
    assert heard[-1] == (second.id, ConversationExecutionStatus.PAUSED)
    assert (first.id, ConversationExecutionStatus.PAUSED) not in heard


@pytest.mark.asyncio
async def test_a_failing_execution_status_listener_does_not_break_the_conversation(
    tmp_path,
):
    def explode(conversation_id: UUID, status: ConversationExecutionStatus) -> None:
        raise RuntimeError("listener bug")

    async with ConversationService(
        conversations_dir=tmp_path / "conversations"
    ) as service:
        service.add_execution_status_listener(explode)

        info, _ = await service.start_conversation(_plain_request(tmp_path))

        assert await service.pause_conversation(info.id)


@pytest.mark.asyncio
async def test_get_execution_statuses_covers_unloaded_conversations(tmp_path):
    conversations_dir = tmp_path / "conversations"
    async with ConversationService(conversations_dir=conversations_dir) as service:
        info, _ = await service.start_conversation(_plain_request(tmp_path))
        await service.pause_conversation(info.id)
    unknown = uuid4()

    async with ConversationService(conversations_dir=conversations_dir) as reloaded:
        statuses = await reloaded.get_execution_statuses([info.id, unknown])

    assert statuses == {info.id: ConversationExecutionStatus.PAUSED}


@pytest.fixture
def event_service() -> EventService:
    return EventService(
        stored=StoredConversation(
            id=uuid4(),
            workspace=LocalWorkspace(working_dir="workspace/project"),
            confirmation_policy=NeverConfirm(),
        ),
        agent=Agent(llm=LLM(model="gpt-4o", usage_id="test-llm"), tools=[]),
        conversations_dir=Path("test_conversation_dir"),
    )


def test_an_action_its_guard_could_not_check_is_blocked(event_service: EventService):
    conversation = MagicMock()
    event_service._conversation = conversation
    event = MagicMock(spec=ActionEvent)
    event.id, event.action = "a1", MagicMock()

    def broken(conversation_id, event):
        raise RuntimeError("the guard's state is locked")

    event_service.action_guards = (broken,)

    event_service._guard_action(event)

    conversation.state.block_action.assert_called_once()
    assert conversation.state.block_action.call_args.args[0] == "a1"


def test_init_serves_conversations_when_the_director_cannot_start(tmp_path: Path):
    """As on a boot, an org the director cannot read leaves it off, not the
    whole server."""
    import json
    import subprocess

    from openhands.agent_server.director.environment.git import GitCodingEnvironment
    from openhands.agent_server.director.environment.records import StoreChange

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=a",
            "-c",
            "user.email=a@b",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    GitCodingEnvironment().org_tree(repo).commit(
        lambda read: [StoreChange("org.json", json.dumps({"version": 999}))], "bad"
    )
    user = tmp_path / "user"
    user.mkdir()
    (user / "director.json").write_text(json.dumps({"repo_path": str(repo)}))
    cfg = Config(
        deferred_init=True,
        conversations_path=tmp_path / "dormant" / "convs",
        bash_events_dir=tmp_path / "dormant" / "bash",
        workspace_path=repo,
    )
    with TestClient(create_app(cfg), raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/init",
            json={
                "session_api_keys": ["k"],
                "conversations_path": str(user / "convs"),
                "bash_events_dir": str(user / "bash"),
            },
        )
        assert resp.status_code == 200
        resp = client.get(
            "/api/conversations/search", headers={"X-Session-API-Key": "k"}
        )
        assert resp.status_code == 200
