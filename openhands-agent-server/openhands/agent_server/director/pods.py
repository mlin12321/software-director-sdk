"""Starts and reaches the conversations of an org's agents."""

import asyncio
import logging
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from openhands.agent_server.conversation_service import (
    ConversationService,
    ExecutionStatusListener,
    ToolParams,
)
from openhands.agent_server.director.headline import TeamSummary, summarize_team
from openhands.agent_server.director.models import IsolationMode
from openhands.agent_server.event_service import (
    ActionGuard,
    ConversationRunLimitExceeded,
)
from openhands.agent_server.models import (
    StartChildConversationRequest,
    StartConversationRequest,
    UpdateConversationRequest,
)
from openhands.sdk import Message, TextContent
from openhands.sdk.agent import ACPAgent
from openhands.sdk.conversation.request import ConversationWorktreeOptions
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.conversation.title_utils import strip_reasoning_blocks
from openhands.sdk.errors import Conflict, NotFound, Unprocessable
from openhands.sdk.llm import LLM
from openhands.sdk.tool.spec import Tool


logger = logging.getLogger(__name__)

# On every conversation the director starts itself, so that a client can tell
# them from the ones a human started. The root IC is the human's own.
STARTED_BY_DIRECTOR = {"startedby": "director"}


@dataclass(frozen=True)
class StartedAgent:
    workspace: str
    isolation: IsolationMode


class PodLauncher(Protocol):
    """What the director needs from the conversations of its agents."""

    async def start_root(
        self,
        request: StartConversationRequest,
        tools: Sequence[Tool],
        *,
        title: str,
        tool_params: ToolParams | None = None,
    ) -> StartedAgent:
        """Start the conversation every other agent of the org derives from.

        Args:
            tool_params: Parameters to set on the agent's tools, by tool name.

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
        tool_params: ToolParams | None = None,
    ) -> StartedAgent:
        """Start an agent with the configuration of ``parent_id``.

        Args:
            tool_params: Parameters to set on the agent's tools, by tool
                name, over the ones it inherits.

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

    async def stop(self, conversation_id: UUID) -> None:
        """Stop an agent where it is. A conversation that is gone is fine."""
        ...

    async def delete(self, conversation_id: UUID) -> None:
        """Delete an agent's conversation, and its worktree when no other
        conversation works in it. A conversation that is gone is fine."""
        ...

    async def statuses(
        self, conversation_ids: Collection[UUID]
    ) -> dict[UUID, ConversationExecutionStatus]:
        """The status of each conversation that exists."""
        ...

    def on_status(self, listener: ExecutionStatusListener) -> Callable[[], None]:
        """Hear every status change of every conversation, until stopped."""
        ...

    def on_action(self, guard: ActionGuard) -> Callable[[], None]:
        """Let ``guard`` refuse actions of any conversation, until stopped.

        It is asked before an action runs, on the conversation's own thread.
        """
        ...

    async def summarize(self, conversation_id: UUID, task: str) -> TeamSummary | None:
        """Summarize ``task`` with the model of an agent, apart from its work.

        Returns:
            None when the model gave nothing usable.

        Raises:
            NotFound: The conversation does not exist.
        """
        ...

    async def ask(
        self, conversation_id: UUID, system: str, user: str, *, usage_id: str
    ) -> str | None:
        """Ask the model of an agent one question, apart from its work.

        Args:
            usage_id: What the call's cost is counted under.

        Returns:
            The answer; None when the model gave none.

        Raises:
            NotFound: The conversation does not exist.
        """
        ...

    async def ask_agent(self, conversation_id: UUID, question: str) -> str | None:
        """Ask an agent one question on top of its conversation so far.

        Nothing of it is kept in the conversation, and the agent's work is
        not disturbed.

        Returns:
            The answer; None when the agent gave none.

        Raises:
            NotFound: The conversation does not exist.
        """
        ...

    async def rename(self, conversation_id: UUID, title: str) -> None:
        """Give a conversation a new title.

        Raises:
            NotFound: The conversation does not exist.
        """
        ...


class ConversationPodLauncher:
    """Runs the org's agents as conversations of this server."""

    def __init__(self, conversations: ConversationService) -> None:
        self._conversations = conversations

    async def start_root(
        self,
        request: StartConversationRequest,
        tools: Sequence[Tool],
        *,
        title: str,
        tool_params: ToolParams | None = None,
    ) -> StartedAgent:
        info, is_new = await self._conversations.start_conversation(
            request, server_tools=tools, tool_params=tool_params
        )
        if not is_new:
            raise Conflict(f"Conversation {info.id} already exists")
        if isinstance(info.agent, ACPAgent):
            await self._conversations.delete_conversation(info.id)
            raise Unprocessable(
                "An org cannot be led by an ACP agent: it runs its own tools "
                "and cannot be given the director's"
            )
        await self.rename(info.id, title)
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
        tool_params: ToolParams | None = None,
    ) -> StartedAgent:
        started = await self._conversations.start_child_conversation(
            parent_id,
            StartChildConversationRequest(task=task, title=title, isolation=isolation),
            child_id=conversation_id,
            extra_suffixes=suffixes,
            # A sub-team continues from its parent's work, not from the
            # repository's default branch.
            worktree_options=ConversationWorktreeOptions(base="head"),
            tool_params=tool_params,
            extra_tags=STARTED_BY_DIRECTOR,
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

    async def stop(self, conversation_id: UUID) -> None:
        await self._conversations.interrupt_conversation(conversation_id)

    async def delete(self, conversation_id: UUID) -> None:
        await self._conversations.delete_conversation(conversation_id)

    async def statuses(
        self, conversation_ids: Collection[UUID]
    ) -> dict[UUID, ConversationExecutionStatus]:
        return await self._conversations.get_execution_statuses(conversation_ids)

    def on_status(self, listener: ExecutionStatusListener) -> Callable[[], None]:
        return self._conversations.add_execution_status_listener(listener)

    def on_action(self, guard: ActionGuard) -> Callable[[], None]:
        return self._conversations.add_action_guard(guard)

    async def summarize(self, conversation_id: UUID, task: str) -> TeamSummary | None:
        event_service = await self._conversations.get_event_service(conversation_id)
        if event_service is None:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        agent = event_service.get_conversation().agent
        if isinstance(agent, ACPAgent):
            return None
        return await asyncio.to_thread(summarize_team, agent.llm, task)

    async def ask(
        self, conversation_id: UUID, system: str, user: str, *, usage_id: str
    ) -> str | None:
        event_service = await self._conversations.get_event_service(conversation_id)
        if event_service is None:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        agent = event_service.get_conversation().agent
        if isinstance(agent, ACPAgent):
            return None
        llm = agent.llm.model_copy(update={"usage_id": usage_id})
        return await asyncio.to_thread(_ask, llm, system, user)

    async def ask_agent(self, conversation_id: UUID, question: str) -> str | None:
        event_service = await self._conversations.get_event_service(conversation_id)
        if event_service is None:
            raise NotFound(f"Conversation {conversation_id} does not exist")
        try:
            answer = await event_service.ask_agent(question)
        except Exception:
            logger.warning("An agent did not answer the director", exc_info=True)
            return None
        return strip_reasoning_blocks(answer).strip() or None

    async def rename(self, conversation_id: UUID, title: str) -> None:
        updated = await self._conversations.update_conversation(
            conversation_id, UpdateConversationRequest(title=title)
        )
        if not updated:
            raise NotFound(f"Conversation {conversation_id} does not exist")


def _ask(llm: LLM, system: str, user: str) -> str | None:
    messages = [
        Message(role="system", content=[TextContent(text=system)]),
        Message(role="user", content=[TextContent(text=user)]),
    ]
    try:
        response = llm.generate(messages, store=False)
    except Exception:
        logger.warning("A model call of the director failed", exc_info=True)
        return None
    text = "".join(
        part.text for part in response.message.content if isinstance(part, TextContent)
    )
    return strip_reasoning_blocks(text).strip() or None
