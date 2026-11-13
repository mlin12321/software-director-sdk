"""Starts and reaches the conversations of an org's agents."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from openhands.agent_server.conversation_service import (
    ConversationService,
    ExecutionStatusListener,
)
from openhands.agent_server.director.models import IsolationMode
from openhands.agent_server.event_service import ConversationRunLimitExceeded
from openhands.agent_server.models import (
    StartChildConversationRequest,
    StartConversationRequest,
)
from openhands.sdk import Message, TextContent
from openhands.sdk.agent import ACPAgent
from openhands.sdk.conversation.request import ConversationWorktreeOptions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.errors import Conflict, NotFound, Unprocessable
from openhands.sdk.tool.spec import Tool


@dataclass(frozen=True)
class StartedAgent:
    workspace: str
    isolation: IsolationMode


class PodLauncher(Protocol):
    """What the director needs from the conversations of its agents."""

    async def start_root(
        self, request: StartConversationRequest, tools: Sequence[Tool]
    ) -> StartedAgent:
        """Start the conversation every other agent of the org derives from.

        Raises:
            Conflict: The requested conversation id is taken.
            Unprocessable: The request resolves to an agent that cannot be
                given the director's tools.
        """
        ...

    async def start_derived(
        self,
        parent_id: UUID,
        conversation_id: UUID,
        *,
        title: str,
        task: str,
        isolation: IsolationMode,
        suffixes: Sequence[str] = (),
    ) -> StartedAgent:
        """Start an agent with the configuration of ``parent_id``.

        Raises:
            NotFound: The parent conversation does not exist.
            ConversationRunLimitExceeded: The server cannot run another
                conversation now. Nothing was started.
        """
        ...

    async def send(self, conversation_id: UUID, text: str) -> None:
        """Deliver a message and make sure the agent runs to read it.

        Raises:
            NotFound: The conversation does not exist.
        """
        ...

    async def statuses(
        self, conversation_ids: Collection[UUID]
    ) -> dict[UUID, ConversationExecutionStatus]:
        """The status of each conversation that exists."""
        ...

    def on_status(self, listener: ExecutionStatusListener) -> Callable[[], None]:
        """Hear every status change of every conversation, until stopped."""
        ...


class ConversationPodLauncher:
    """Runs the org's agents as conversations of this server."""

    def __init__(self, conversations: ConversationService) -> None:
        self._conversations = conversations

    async def start_root(
        self, request: StartConversationRequest, tools: Sequence[Tool]
    ) -> StartedAgent:
        info, is_new = await self._conversations.start_conversation(
            request, server_tools=tools
        )
        if not is_new:
            raise Conflict(f"Conversation {info.id} already exists")
        if isinstance(info.agent, ACPAgent):
            await self._conversations.delete_conversation(info.id)
            raise Unprocessable(
                "An org cannot be led by an ACP agent: it runs its own tools "
                "and cannot be given the director's"
            )
        return StartedAgent(
            workspace=str(info.workspace.working_dir),
            isolation="worktree" if request.worktree else "shared",
        )

    async def start_derived(
        self,
        parent_id: UUID,
        conversation_id: UUID,
        *,
        title: str,
        task: str,
        isolation: IsolationMode,
        suffixes: Sequence[str] = (),
    ) -> StartedAgent:
        started = await self._conversations.start_child_conversation(
            parent_id,
            StartChildConversationRequest(task=task, title=title, isolation=isolation),
            child_id=conversation_id,
            extra_suffixes=suffixes,
            # A sub-team continues from its parent's work, not from the
            # repository's default branch.
            worktree_options=ConversationWorktreeOptions(base="head"),
        )
        if started is None or started.workspace is None:
            raise NotFound(f"Conversation {parent_id} does not exist")
        return StartedAgent(
            workspace=started.workspace, isolation=started.isolation or "shared"
        )

    async def send(self, conversation_id: UUID, text: str) -> None:
        event_service = await self._conversations.get_event_service(conversation_id)
        if event_service is None:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        message = Message(role="user", content=[TextContent(text=text)])
        try:
            await event_service.send_message(message, True)
        except ConversationRunLimitExceeded:
            # The message is saved; only the run still has to start.
            try:
                await event_service.run(wait_for_capacity=True)
            except ValueError:
                # Already running, so the agent reads the message anyway.
                pass

    async def statuses(
        self, conversation_ids: Collection[UUID]
    ) -> dict[UUID, ConversationExecutionStatus]:
        return await self._conversations.get_execution_statuses(conversation_ids)

    def on_status(self, listener: ExecutionStatusListener) -> Callable[[], None]:
        return self._conversations.add_execution_status_listener(listener)
